import os, tempfile, uuid, time, json
from pathlib import Path
from typing import Any, Dict, Optional
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

def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _as_model_name_from_onnx_filename(onnx_filename: str) -> str:
    # "my_model.onnx" -> "my_model"
    base = Path(onnx_filename).name
    if base.lower().endswith(".onnx"):
        base = base[:-5]
    return base or "model"

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
        metrics = train_yolo(local_dir, out_dir, msg.get("hyperparams", {}) or {}, progress=progress)

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
    메시지 예:
    {
        "job_id": "...",
        "model": {
            "s3_uri": "s3://.../best.pt"
        },
        "output": {
            "s3_bucket": "visioninapp-bucket",
            "prefix": "result/test_onnx/<job_id>",
            "model_name": "my_model.onnx"   # ← 이 파일명에서 model_name을 유추
        },
        "hyperparams": { "imgsz": 640, "dynamic": true, "half": false, "simplify": true, "precision": "fp16" },
        # (선택) 평가지표 전달 방식 1: 바로 dict로
        "metrics": {
            "map50": 0.83, "map50_95": 0.52, "precision": 0.87, "recall": 0.78, "best_epoch": 93
        },
        # (선택) 평가지표 전달 방식 2: S3에 저장된 파일 경로
        "metrics_s3_uri": "s3://visioninapp-bucket/runs/exp123/metrics.json"
        # 또는 csv일 수도 있음: "s3://.../results.csv"
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

        # --- 하이퍼파라미터 수집 ---
        hp = (msg.get("hyperparams") or {})
        def _as_bool(v):
            if isinstance(v, bool): return v
            if isinstance(v, str): return v.strip().lower() in ("1","true","yes","y","t")
            return bool(v)
        def _as_int(v, default=None):
            try: return int(v)
            except Exception: return default

        imgsz     = _as_int(hp.get("imgsz"), 640)
        dynamic   = _as_bool(hp.get("dynamic")) if "dynamic" in hp else True
        simplify  = _as_bool(hp.get("simplify")) if "simplify" in hp else False
        half_flag = _as_bool(hp.get("half")) if "half" in hp else False
        precision = (hp.get("precision") or "").strip().lower() or None
        opset     = _as_int(hp.get("opset"), 13)

        if precision == "fp16":
            half_flag = True
        elif precision == "fp32":
            half_flag = False

        # --- ONNX 내보내기 ---
        progress.send("convert.onnx", 60, f"export to onnx (imgsz={imgsz}, dynamic={dynamic}, half={half_flag}, simplify={simplify})")
        tmp_onnx = os.path.join(tempfile.gettempdir(), f"{job_id}.onnx")
        to_onnx(
            tmp_pt, tmp_onnx,
            opset=opset, imgsz=imgsz, dynamic=dynamic, half=half_flag, simplify=simplify, precision=precision
        )

        # --- 로컬 표준 경로로 정리: {LOCAL_MODELS_ROOT}/models/{model_name}/onnx/ ---
        onnx_filename = msg["output"].get("model_name", "best.onnx")
        model_name_for_dir = _as_model_name_from_onnx_filename(onnx_filename)

        models_root = os.getenv("LOCAL_MODELS_ROOT", "/models")
        local_dir = os.path.join(models_root, "models", model_name_for_dir, "onnx")
        _ensure_dir(local_dir)

        local_onnx_path = os.path.join(local_dir, onnx_filename)
        # temp -> local copy
        Path(local_onnx_path).write_bytes(Path(tmp_onnx).read_bytes())

        # --- 평가지표 저장 ---
        metrics_obj = msg.get("metrics")
        metrics_s3  = msg.get("metrics_s3_uri")

        if isinstance(metrics_obj, dict) and metrics_obj:
            _write_json(os.path.join(local_dir, "metrics.json"), metrics_obj)
        elif isinstance(metrics_s3, str) and metrics_s3.startswith("s3://"):
            # 확장자에 따라 저장 파일명 결정
            if metrics_s3.lower().endswith(".csv"):
                download_s3(metrics_s3, os.path.join(local_dir, "metrics.csv"))
            else:
                # json로 가정
                download_s3(metrics_s3, os.path.join(local_dir, "metrics.json"))
        else:
            logger.info("[convert_onnx] 전달된 평가지표가 없어 로컬 저장을 건너뜀")

        # --- S3 업로드(기존 로직 유지) ---
        progress.send("upload", 90, "upload onnx")
        bucket = S3_BUCKET  # 기존 상수 사용
        key = f"{msg['output']['prefix'].rstrip('/')}/{onnx_filename}"
        upload_s3(local_onnx_path, bucket, key)

        # (선택) 평가지표도 함께 업로드하고 싶다면:
        # if Path(os.path.join(local_dir, "metrics.json")).exists():
        #     upload_s3(os.path.join(local_dir, "metrics.json"), bucket, f"{msg['output']['prefix'].rstrip('/')}/metrics.json")
        # if Path(os.path.join(local_dir, "metrics.csv")).exists():
        #     upload_s3(os.path.join(local_dir, "metrics.csv"), bucket, f"{msg['output']['prefix'].rstrip('/')}/metrics.csv")

        progress.done({
            "s3_bucket": bucket,
            "s3_uri": f"s3://{bucket}/{key}",
            "local_dir": local_dir
        })
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
