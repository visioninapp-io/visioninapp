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

from dotenv import load_dotenv
load_dotenv()

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "app.yaml")
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
            "s3_prefix": "datasets/myset/",  // S3 내 prefix (끝 슬래시 권장)
            "name": "myset"
        },
        "output": {
            "s3_bucket": "visioninapp-bucket",
            "prefix": "models/abc12345",
            "model_name": "myset.pt"
        },
        "hyperparams": {
            "model": "yolo12n"
            "epochs": 20, 
            "batch": 8, 
            "imgsz": 640 
        }
    }
    """
    logger.info("[trainer] 모델 학습 시작")
    job_id = msg["job_id"]
    _, pub_ch = mq.channel()
    progress = Progress(pub_ch, exchanges["events"], job_id)
    data_root, models_root = _paths()

    try:
        progress.send("train.download_dataset", 5, "downloading dataset")

        # --- 폴더(prefix) 동기화 ---
        prefix = msg["dataset"]["s3_prefix"]
        dataset_name = msg["dataset"].get("name")

        # name이 없으면 prefix의 마지막 디렉토리명을 기본값으로 사용
        if not dataset_name:
            p = prefix.strip("/")
            dataset_name = p.split("/")[-1] if p else f"ds_{job_id}"

        # s3_client.download_s3_folder(prefix, dataset_name) 가
        # LOCAL_DATA_ROOT/<dataset_name> 하위로 내려받고 존재 파일은 스킵하도록 구현되어 있어야 함
        local_dir = download_s3_folder(prefix, dataset_name)

        logger.info(f"[train] dataset synced: prefix='{prefix}' -> local_dir='{local_dir}'")

        # --- 학습 ---
        progress.send("train.start", 10, "start training")
        out_dir = os.path.join(models_root, job_id)
        metrics = train_yolo(local_dir, out_dir, msg.get("hyperparams", {}))

        # --- 결과 업로드 ---
        best_pt = os.path.join(out_dir, "train", "weights", "best.pt")
        bucket = msg["output"]["s3_bucket"]
        model_name = msg["output"].get("model_name", "best.pt")
        key = f"{msg['output']['prefix'].rstrip('/')}/{model_name}"

        progress.send("upload", 95, "uploading model")
        upload_s3(best_pt, bucket, key)

        # 2. results.csv 업로드 (있을 때만)
        results_csv = os.path.join(out_dir, "train", "results.csv")
        if os.path.exists(results_csv):
            metrics_name = msg["output"].get("metrics_name", "results.csv")
            metrics_key = f"{msg['output']['prefix'].rstrip('/')}/{metrics_name}"
            progress.send("upload.metrics", 96, f"uploading {metrics_name}")
            upload_s3(results_csv, bucket, metrics_key)

        progress.done({"s3_bucket": bucket, "s3_uri": f"s3://{bucket}/{key}"}, metrics)

    except Exception as e:
        logger.exception("train failed")
        progress.error("train", str(e))
    finally:
        logger.info("[trainer] 모델 학습 종료")
        pub_ch.close()


def handle_onnx(mq: MQ, exchanges: dict, msg: dict):
    """
    {
        "job_id": job_id,
        "model": {
            # 변환할 모델의 s3 경로
            "s3_uri": "s3://visioninapp-bucket/result/test/best.pt"
        },
        "output": {
            # 변환 결과를 저장할 S3 경로
            "s3_bucket": "visioninapp-bucket",
            "prefix": f"result/test_onnx/{job_id}",
            "model_name": "my_model.onnx"
        }
    }

    """
    logger.info("[convert_onnx] onnx 변환 시작")
    job_id = msg["job_id"]
    _, pub_ch = mq.channel()
    progress = Progress(pub_ch, exchanges["events"], job_id)
    try:
        progress.send("convert.onnx.download", 10, "download model")
        tmp_pt = os.path.join(tempfile.gettempdir(), f"{job_id}.pt")
        download_s3(msg["model"]["s3_uri"], tmp_pt)

        progress.send("convert.onnx", 60, "export to onnx")
        out_onnx = os.path.join(tempfile.gettempdir(), f"{job_id}.onnx")
        to_onnx(tmp_pt, out_onnx)

        progress.send("upload", 90, "upload onnx")
        bucket = msg["output"]["s3_bucket"]
        model_name = msg["output"].get("model_name", "best.onnx")
        key = f"{msg['output']['prefix'].rstrip('/')}/{model_name}"
        upload_s3(out_onnx, bucket, key)
        progress.done({"s3_bucket": bucket, "s3_uri": f"s3://{bucket}/{key}"})
    except Exception as e:
        logger.exception("onnx failed")
        progress.error("convert.onnx", str(e))
    finally:
        logger.info("[convert_onnx] onnx 변환 종료")
        pub_ch.close()

def handle_trt(mq: MQ, exchanges: dict, msg: dict):
    """
    {
        "job_id": job_id,
        "model": {
            # 변환할 모델의 s3 경로
            "s3_uri": "s3://visioninapp-bucket/result/test/best.pt"
        },
        "output": {
            # 변환 결과를 저장할 S3 경로
            "s3_bucket": "visioninapp-bucket",
            "prefix": f"result/test_engine/{job_id}",
            "model_name": "my_model.engine"
        }
    }

    """
    logger.info("[convert_trt] tensorRT 변환 시작")
    job_id = msg["job_id"]
    _, pub_ch = mq.channel()
    progress = Progress(pub_ch, exchanges["events"], job_id)
    try:
        # PT 모델 다운로드 (기존 ONNX -> PT 로 변경)
        progress.send("convert.trt.download", 10, "download model (.pt)")
        tmp_pt = os.path.join(tempfile.gettempdir(), f"{job_id}.pt")
        download_s3(msg["model"]["s3_uri"], tmp_pt)

        # 변환 옵션 (precision / imgsz 등)
        trt_cfg   = msg.get("trt", {}) or {}
        precision = (trt_cfg.get("precision") or "fp16").lower()   # fp16|fp32|int8
        imgsz     = int(trt_cfg.get("imgsz") or 640)
        dynamic   = bool(trt_cfg.get("dynamic")) if "dynamic" in trt_cfg else True

        progress.send("convert.trt", 60, f"build TensorRT ({precision})")
        out_engine = os.path.join(tempfile.gettempdir(), f"{job_id}.engine")

        # PT → TensorRT 엔진 변환
        to_tensorrt(tmp_pt, out_engine, precision=precision, imgsz=imgsz, dynamic=dynamic)

        # 업로드 (파일명 커스터마이즈 지원)
        progress.send("upload", 90, "upload engine")
        bucket     = msg["output"]["s3_bucket"]
        model_name = msg["output"].get("model_name", "best.engine")
        key        = f"{msg['output']['prefix'].rstrip('/')}/{model_name}"
        upload_s3(out_engine, bucket, key)

        progress.done({"s3_bucket": bucket, "s3_uri": f"s3://{bucket}/{key}"})
    except Exception as e:
        logger.exception("trt failed")
        progress.error("convert.trt", str(e))
    finally:
        logger.info("[convert_trt] tensorRT 변환 종료")
        pub_ch.close()


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
        # 메인 스레드는 단순 대기(신호 처리/헬스체크 확장 가능)
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Shutting down.")

if __name__ == "__main__":
    main()
