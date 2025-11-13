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

def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _as_model_name_from_onnx_filename(onnx_filename: str) -> str:
    base = Path(onnx_filename).name
    return base[:-5] if base.lower().endswith(".onnx") else (base or "model")

def _as_model_name_from_engine_filename(engine_filename: str) -> str:
    base = Path(engine_filename).name
    return base[:-7] if base.lower().endswith(".engine") else (base or "model")

def _coerce_bool(v, default=None):
    if v is None: return default
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.strip().lower() in ("1", "true", "yes", "y", "t")
    return bool(v)


def _coerce_int(v, default=None):
    try: return int(v)
    except Exception: return default

def _maybe_download(uri_or_path: str, local_target: str) -> str:
    """s3://...면 받아오고, 로컬 경로면 그대로 반환"""
    if isinstance(uri_or_path, str) and uri_or_path.startswith("s3://"):
        download_s3(uri_or_path, local_target)
        return local_target
    return uri_or_path

# --- 평가 헬퍼: PT로 val() 수행 ---
def _eval_metrics_with_pt(pt_path: str, data_yaml_path: str, imgsz: int, batch: int, split: str = "val") -> dict:
    try:
        from ultralytics import YOLO
        model = YOLO(pt_path)
        # Ultralytics는 split을 data.yaml의 분기 기준으로 처리 (val/test)
        results = model.val(data=data_yaml_path, imgsz=imgsz, batch=batch, split=split, save_json=False, plots=False, verbose=False)
        # results.metrics에는 주요 지표가 dict 형태로 들어옴
        # (Ultralytics 버전에 따라 이름 차이 있을 수 있음)
        m = results.results_dict if hasattr(results, "results_dict") else getattr(results, "metrics", {}) or {}
        # 표준 키만 추려 안정화
        picked = {
            "map50":        float(m.get("metrics/mAP50(B)")) if "metrics/mAP50(B)" in m else float(m.get("metrics/mAP50", m.get("map50", 0.0))),
            "map50_95":     float(m.get("metrics/mAP50-95(B)")) if "metrics/mAP50-95(B)" in m else float(m.get("metrics/mAP50-95", m.get("map", 0.0))),
            "precision":    float(m.get("metrics/precision(B)")) if "metrics/precision(B)" in m else float(m.get("precision", 0.0)),
            "recall":       float(m.get("metrics/recall(B)")) if "metrics/recall(B)" in m else float(m.get("recall", 0.0)),
            "imgsz":        imgsz,
            "batch":        batch,
            "split":        split,
        }
        return picked
    except Exception as e:
        logger.warning(f"[convert_onnx] 평가 수행 실패: {e}")
        return {}

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
        "dataset": {"s3_prefix": "datasets/myset/", "name": "myset"},
        "output": {"prefix": "models/abc12345", "model_name": "myset.pt", "metrics_name": "results.csv"},
        "hyperparams": {"model": "yolo12n", "epochs": 20, "batch": 8, "imgsz": 640},
        "split": [0.8, 0.1, 0.1], "split_seed": 42, "move_files": false
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

        # 하이퍼파라미터에 job_id를 주입(고유 run 폴더 위해)
        hyper = (msg.get("hyperparams") or {}).copy()
        hyper.setdefault("job_id", job_id)

        train_out = train_yolo(local_dir, out_dir, hyper, progress=progress)
        # train_out: { "metrics": {...}, "run_dir": "...", "best_pt": "...", "results_csv": "..."|None }
        metrics = train_out.get("metrics", {}) or {}
        best_pt = train_out.get("best_pt")
        results_csv = train_out.get("results_csv")

        if not best_pt or not os.path.exists(best_pt):
            raise FileNotFoundError(f"best.pt not found at {best_pt}")

        # --- 결과 업로드 ---
        bucket = S3_BUCKET
        model_name = msg["output"].get("model_name", "best.pt")
        key = f"{msg['output']['prefix'].rstrip('/')}/{model_name}"

        progress.send("upload", 95, "uploading model")
        upload_s3(best_pt, bucket, key)

        # results.csv 업로드 (있을 때만)
        if results_csv and os.path.exists(results_csv):
            metrics_name = msg["output"].get("metrics_name", "results.csv")
            metrics_key = f"{msg['output']['prefix'].rstrip('/')}/{metrics_name}"
            progress.send("upload.metrics", 96, f"uploading {metrics_name}")
            upload_s3(results_csv, bucket, metrics_key)

        progress.send("done", 100, "finish work")
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
        imgsz     = _coerce_int(hp.get("imgsz"), 640)
        dynamic   = _coerce_bool(hp.get("dynamic"), True)
        simplify  = _coerce_bool(hp.get("simplify"), False)
        half_flag = _coerce_bool(hp.get("half"), False)
        precision = (hp.get("precision") or "").strip().lower() or None
        opset     = _coerce_int(hp.get("opset"), 13)

        if precision == "fp16": half_flag = True
        elif precision == "fp32": half_flag = False

        # --- ONNX export ---
        progress.send("convert.onnx", 60, f"export to onnx (imgsz={imgsz}, dynamic={dynamic}, half={half_flag}, simplify={simplify})")
        tmp_onnx = os.path.join(tempfile.gettempdir(), f"{job_id}.onnx")
        to_onnx(tmp_pt, tmp_onnx, opset=opset, imgsz=imgsz, dynamic=dynamic, half=half_flag, simplify=simplify, precision=precision)

        # --- 로컬 표준 경로 (중복 'models' 제거) ---
        onnx_filename = msg["output"].get("model_name", "best.onnx")
        model_name_for_dir = _as_model_name_from_onnx_filename(onnx_filename)

        models_root = os.getenv("LOCAL_MODELS_ROOT", "/models")
        local_dir = os.path.join(models_root, model_name_for_dir, "onnx")   # ✅ 여기! models/ 추가하지 않음
        _ensure_dir(local_dir)

        local_onnx_path = os.path.join(local_dir, onnx_filename)
        Path(local_onnx_path).write_bytes(Path(tmp_onnx).read_bytes())

        # --- 평가지표 저장: 우선 msg → s3 → (없으면) 자동 평가 ---
        metrics_obj = msg.get("metrics")
        metrics_s3  = msg.get("metrics_s3_uri")
        metrics_out_json = os.path.join(local_dir, "metrics.json")

        saved_metrics = False
        if isinstance(metrics_obj, dict) and metrics_obj:
            _write_json(metrics_out_json, metrics_obj)
            saved_metrics = True
        elif isinstance(metrics_s3, str) and metrics_s3.startswith("s3://"):
            if metrics_s3.lower().endswith(".csv"):
                download_s3(metrics_s3, os.path.join(local_dir, "metrics.csv"))
            else:
                download_s3(metrics_s3, metrics_out_json)
            saved_metrics = True
        else:
            # --- 자동 평가 시도 ---
            ev = (msg.get("eval") or {})
            data_yaml = ev.get("data_path") or ev.get("data_s3_uri")
            if data_yaml:
                # data.yaml 준비
                data_local = _maybe_download(data_yaml, os.path.join(tempfile.gettempdir(), f"{job_id}_data.yaml"))
                e_imgsz = _coerce_int(ev.get("imgsz"), imgsz)
                e_batch = _coerce_int(ev.get("batch"), 16)
                e_split = (ev.get("split") or "val").strip()
                progress.send("convert.onnx.eval", 75, f"evaluate model (split={e_split})")
                eval_metrics = _eval_metrics_with_pt(tmp_pt, data_local, imgsz=e_imgsz, batch=e_batch, split=e_split)
                if eval_metrics:
                    _write_json(metrics_out_json, eval_metrics)
                    saved_metrics = True
                else:
                    logger.info("[convert_onnx] 자동 평가를 시도했으나 유효한 결과가 없어 저장 생략")
            else:
                logger.info("[convert_onnx] 평가지표 없음 + eval 데이터 미지정 → 평가 생략")

        # --- S3 업로드(ONNX 및 지표) ---
        progress.send("upload", 90, "upload onnx (+metrics)")
        bucket = S3_BUCKET
        key = f"{msg['output']['prefix'].rstrip('/')}/{onnx_filename}"
        upload_s3(local_onnx_path, bucket, key)
        if saved_metrics and Path(metrics_out_json).exists():
            upload_s3(metrics_out_json, bucket, f"{msg['output']['prefix'].rstrip('/')}/metrics.json")

        progress.send("done", 100, "finish work")
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

        # ★ 로컬 표준 경로 계산: models/{model_name}/trt/best.engine
        models_root = os.getenv("LOCAL_MODELS_ROOT", "/models")
        # output.model_name이 "my_model.engine"이면 폴더명은 "my_model"
        engine_filename = msg["output"].get("model_name", "best.engine")
        model_name_for_dir = _as_model_name_from_engine_filename(engine_filename)

        local_dir = os.path.join(models_root, model_name_for_dir, "trt")
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        local_engine_path = os.path.join(local_dir, "best.engine")  # ← 고정 파일명

        progress.send("convert.trt", 60, f"build TensorRT ({precision})")
        # ★ 최종 위치로 바로 내보내기(정규화 복사 포함)
        to_tensorrt(
            tmp_pt,
            local_engine_path,
            precision=precision,
            imgsz=imgsz,
            dynamic=dynamic,
        )

        # 업로드는 로컬 표준 경로에서
        progress.send("upload", 90, "upload engine")
        bucket     = S3_BUCKET
        key        = f"{msg['output']['prefix'].rstrip('/')}/{engine_filename}"  # 업로드 파일명은 기존 스키마 유지
        upload_s3(local_engine_path, bucket, key)

        progress.send("done", 100, "finish work")
        progress.done({"s3_bucket": bucket, "s3_uri": f"s3://{bucket}/{key}", "local_path": local_engine_path})

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
