import os, tempfile, uuid, time
from pathlib import Path
from utils.config import load_config
from utils.logger import setup_logger
from mq import MQ, declare_topology, start_consumer_thread
from s3_client import download_s3, upload_s3, download_s3_folder
from core.progress import Progress
from core.trainer import train_yolo
from core.convert_onnx import to_onnx
from core.convert_trt import to_tensorrt
from core.prepare_yolo_dataset import prepare_yolo_dataset  # ✅ 추가: 데이터셋 준비(분할/검증)

from pika.exceptions import ChannelWrongStateError, ConnectionWrongStateError
from dotenv import load_dotenv
load_dotenv()

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "app.yaml")
S3_BUCKET = os.getenv("S3_BUCKET", "visioninapp-bucket")
logger = setup_logger()

def _paths():
    data_root  = os.getenv("LOCAL_DATA_ROOT", "/data")
    models_root= os.getenv("LOCAL_MODELS_ROOT", "/models")
    Path(data_root).mkdir(parents=True, exist_ok=True)
    Path(models_root).mkdir(parents=True, exist_ok=True)
    return data_root, models_root

def handle_train(mq: MQ, exchanges: dict, msg: dict):
    """
    message 예시 (s3 폴더 동기화 방식)
    {
        "job_id": "abc12345",
        "dataset": {
            "s3_prefix": "datasets/myset/",   // S3 내 prefix (끝 슬래시 권장)
            "name": "myset"                   // 로컬 동기화 폴더명(옵션)
        },
        "output": {
            "prefix": "models/abc12345",
            "model_name": "myset.pt",         // 업로드 파일명(기본: best.pt)
            "metrics_name": "results.csv"     // 결과 CSV 업로드 파일명(기본: results.csv)
        },
        "hyperparams": {
            "model": "yolo12n",
            "epochs": 20,
            "batch": 8,
            "imgsz": 640
        },
        "split": [0.8, 0.1, 0.1],             // (옵션) train/val/test 비율
        "split_seed": 42,                     // (옵션) 분할 시드
        "move_files": false                   // (옵션) 분할 시 복사 대신 이동
    }
    """
    logger.info("[trainer] 모델 학습 시작")
    job_id = msg["job_id"]
    conn, pub_ch = mq.channel()
    progress = Progress(pub_ch, exchanges["events"], job_id)
    data_root, models_root = _paths()

    try:
        progress.send("train.download_dataset", 5, "downloading dataset")

        # --- S3 prefix 동기화 ---
        prefix = msg["dataset"]["s3_prefix"]
        dataset_name = msg["dataset"].get("name")
        if not dataset_name:
            p = prefix.strip("/")
            dataset_name = p.split("/")[-1] if p else f"ds_{job_id}"

        # download_s3_folder는 LOCAL_DATA_ROOT/<dataset_name> 하위로 동기화된 경로를 반환해야 함
        local_dir = download_s3_folder(prefix, dataset_name)
        logger.info(f"[train] dataset synced: prefix='{prefix}' -> local_dir='{local_dir}'")

        # --- 데이터셋 준비 (data.yaml 존중, 없거나 미완성 시 자동 분할/생성) ---
        progress.send("train.prepare_split", 15, "prepare dataset (respect existing data.yaml)")
        splits = tuple(msg.get("split", (0.8, 0.1, 0.1)))
        split_seed = int(msg.get("split_seed", 42))
        move_files = bool(msg.get("move_files", False))

        data_yaml = prepare_yolo_dataset(
            root_dir=local_dir,
            splits=splits,
            seed=split_seed,
            move_files=move_files
        )
        logger.info(f"[train] data.yaml ready: {data_yaml}")

        # --- 학습 ---
        progress.send("train.start", 20, "start training")
        out_dir = os.path.join(models_root, dataset_name)
        metrics = train_yolo(local_dir, out_dir, msg.get("hyperparams", {}))

        # --- 결과 업로드 ---
        best_pt = os.path.join(out_dir, "train", "weights", "best.pt")
        bucket = S3_BUCKET
        model_name = msg["output"].get("model_name", "best.pt")
        key = f"{msg['output']['prefix'].rstrip('/')}/{model_name}"

        progress.send("upload", 95, "uploading model")
        upload_s3(best_pt, bucket, key)

        # results.csv 업로드 (있을 때만)
        results_csv = os.path.join(out_dir, "train", "results.csv")
        if os.path.exists(results_csv):
            metrics_name = msg["output"].get("metrics_name", "results.csv")
            metrics_key = f"{msg['output']['prefix'].rstrip('/')}/{metrics_name}"
            progress.send("upload.metrics", 96, f"uploading {metrics_name}")
            upload_s3(results_csv, bucket, metrics_key)

        progress.done({"s3_bucket": bucket, "s3_uri": f"s3://{bucket}/{key}"}, metrics)

    except Exception as e:
        time.sleep(5)
        logger.exception("train failed")
        progress.error("train", str(e))
    finally:
        logger.info("[trainer] 모델 학습 종료")
        for obj in (pub_ch, conn):
            try:
                obj.close()
            except (ChannelWrongStateError, ConnectionWrongStateError):
                pass
            except Exception:
                pass

def handle_onnx(mq: MQ, exchanges: dict, msg: dict):
    """
    {
        "job_id": job_id,
        "model": {
            "s3_uri": "s3://visioninapp-bucket/result/test/best.pt"
        },
        "output": {
            "s3_bucket": "visioninapp-bucket",
            "prefix": "result/test_onnx/<job_id>",
            "model_name": "my_model.onnx"
        }
    }
    """
    logger.info("[convert_onnx] onnx 변환 시작")
    job_id = msg["job_id"]
    conn, pub_ch = mq.channel()
    progress = Progress(pub_ch, exchanges["events"], job_id)
    try:
        progress.send("convert.onnx.download", 10, "download model")
        tmp_pt = os.path.join(tempfile.gettempdir(), f"{job_id}.pt")
        download_s3(msg["model"]["s3_uri"], tmp_pt)

        progress.send("convert.onnx", 60, "export to onnx")
        out_onnx = os.path.join(tempfile.gettempdir(), f"{job_id}.onnx")
        to_onnx(tmp_pt, out_onnx)

        progress.send("upload", 90, "upload onnx")
        bucket = S3_BUCKET
        model_name = msg["output"].get("model_name", "best.onnx")
        key = f"{msg['output']['prefix'].rstrip('/')}/{model_name}"
        upload_s3(out_onnx, bucket, key)
        progress.done({"s3_bucket": bucket, "s3_uri": f"s3://{bucket}/{key}"})
    except Exception as e:
        logger.exception("onnx failed")
        progress.error("convert.onnx", str(e))
    finally:
        logger.info("[convert_onnx] onnx 변환 종료")
        for obj in (pub_ch, conn):
            try:
                obj.close()
            except (ChannelWrongStateError, ConnectionWrongStateError):
                pass
            except Exception:
                pass

def handle_trt(mq: MQ, exchanges: dict, msg: dict):
    """
    {
        "job_id": job_id,
        "model": {
            "s3_uri": "s3://visioninapp-bucket/result/test/best.pt"   // PT 입력
        },
        "trt": {
            "precision": "fp16",   // fp16|fp32|int8 (옵션)
            "imgsz": 640,          // (옵션)
            "dynamic": true        // (옵션)
        },
        "output": {
            "s3_bucket": "visioninapp-bucket",
            "prefix": "result/test_engine/<job_id>",
            "model_name": "my_model.engine"
        }
    }
    """
    logger.info("[convert_trt] tensorRT 변환 시작")
    job_id = msg["job_id"]
    conn, pub_ch = mq.channel()
    progress = Progress(pub_ch, exchanges["events"], job_id)
    try:
        # PT 모델 다운로드
        progress.send("convert.trt.download", 10, "download model (.pt)")
        tmp_pt = os.path.join(tempfile.gettempdir(), f"{job_id}.pt")
        download_s3(msg["model"]["s3_uri"], tmp_pt)

        # 변환 옵션
        trt_cfg   = msg.get("trt", {}) or {}
        precision = (trt_cfg.get("precision") or "fp16").lower()   # fp16|fp32|int8
        imgsz     = int(trt_cfg.get("imgsz") or 640)
        dynamic   = bool(trt_cfg.get("dynamic")) if "dynamic" in trt_cfg else True

        progress.send("convert.trt", 60, f"build TensorRT ({precision})")
        out_engine = os.path.join(tempfile.gettempdir(), f"{job_id}.engine")

        # PT → TensorRT 엔진 변환
        to_tensorrt(tmp_pt, out_engine, precision=precision, imgsz=imgsz, dynamic=dynamic)

        # 업로드
        progress.send("upload", 90, "upload engine")
        bucket     = S3_BUCKET
        model_name = msg["output"].get("model_name", "best.engine")
        key        = f"{msg['output']['prefix'].rstrip('/')}/{model_name}"
        upload_s3(out_engine, bucket, key)

        progress.done({"s3_bucket": bucket, "s3_uri": f"s3://{bucket}/{key}"})
    except Exception as e:
        logger.exception("trt failed")
        progress.error("convert.trt", str(e))
    finally:
        logger.info("[convert_trt] tensorRT 변환 종료")
        for obj in (pub_ch, conn):
            try:
                obj.close()
            except (ChannelWrongStateError, ConnectionWrongStateError):
                pass
            except Exception:
                pass

def main():
    cfg = load_config(CFG_PATH)
    amqp = os.getenv("RABBITMQ_URL")
    if not amqp:
        raise RuntimeError("RABBITMQ_URL not set")

    mq = MQ(amqp)
    # 토폴로지 준비(한 번만)
    conn, ch = mq.channel()
    declare_topology(ch, cfg["mq"]["exchanges"], cfg["mq"]["queues"])
    ch.close(); conn.close()

    # 3개의 컨슈머 스레드 시작(하나의 프로세스)
    th_train = start_consumer_thread(
        mq, cfg["mq"]["queues"]["train"],
        handler=lambda m: handle_train(mq, cfg["mq"]["exchanges"], m)
    )
    th_onnx = start_consumer_thread(
        mq, cfg["mq"]["queues"]["onnx"],
        handler=lambda m: handle_onnx(mq, cfg["mq"]["exchanges"], m)
    )
    th_trt = start_consumer_thread(
        mq, cfg["mq"]["queues"]["trt"],
        handler=lambda m: handle_trt(mq, cfg["mq"]["exchanges"], m)
    )

    logger.info("[main] All routers started in one process. Waiting for jobs...")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Shutting down.")

if __name__ == "__main__":
    main()
