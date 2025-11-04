import os, tempfile, uuid, time
from pathlib import Path
from utils.config import load_config
from utils.logger import setup_logger
from mq import MQ, declare_topology, start_consumer_thread
from s3_client import download_s3, upload_s3, untar
from core.progress import Progress
from core.trainer import train_yolo
from core.convert_onnx import to_onnx
from core.convert_trt import to_tensorrt

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "app.yaml")

logger = setup_logger()

def _paths():
    data_root  = os.getenv("LOCAL_DATA_ROOT", "/data")
    models_root= os.getenv("LOCAL_MODELS_ROOT", "/models")
    Path(data_root).mkdir(parents=True, exist_ok=True)
    Path(models_root).mkdir(parents=True, exist_ok=True)
    return data_root, models_root

def handle_train(mq: MQ, exchanges: dict, msg: dict):
    job_id = msg["job_id"]
    _, pub_ch = mq.channel()
    progress = Progress(pub_ch, exchanges["events"], job_id)
    data_root, models_root = _paths()
    try:
        progress.send("train.download_dataset", 5, "downloading dataset")
        tmp_tar = os.path.join(tempfile.gettempdir(), f"{job_id}.tar.gz")
        download_s3(msg["dataset"]["s3_uri"], tmp_tar)

        local_dir = msg["dataset"].get("local_dir") or os.path.join(data_root, f"ds_{job_id}")
        untar(tmp_tar, local_dir)

        progress.send("train.start", 10, "start training")
        out_dir = os.path.join(models_root, job_id)
        metrics = train_yolo(local_dir, out_dir, msg.get("hyperparams", {}))

        best_pt = os.path.join(out_dir, "train", "weights", "best.pt")
        bucket = msg["output"]["s3_bucket"]
        key    = f"{msg['output']['prefix'].rstrip('/')}/best.pt"
        progress.send("upload", 95, "uploading model")
        upload_s3(best_pt, bucket, key)
        progress.done({"s3_bucket": bucket, "s3_uri": f"s3://{bucket}/{key}"}, metrics)
    except Exception as e:
        logger.exception("train failed")
        progress.error("train", str(e))
    finally:
        pub_ch.close()

def handle_onnx(mq: MQ, exchanges: dict, msg: dict):
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
        key    = f"{msg['output']['prefix'].rstrip('/')}/best.onnx"
        upload_s3(out_onnx, bucket, key)
        progress.done({"s3_bucket": bucket, "s3_uri": f"s3://{bucket}/{key}"})
    except Exception as e:
        logger.exception("onnx failed")
        progress.error("convert.onnx", str(e))
    finally:
        pub_ch.close()

def handle_trt(mq: MQ, exchanges: dict, msg: dict):
    job_id = msg["job_id"]
    _, pub_ch = mq.channel()
    progress = Progress(pub_ch, exchanges["events"], job_id)
    try:
        progress.send("convert.trt.download", 10, "download onnx")
        tmp_onnx = os.path.join(tempfile.gettempdir(), f"{job_id}.onnx")
        download_s3(msg["model"]["s3_uri"], tmp_onnx)

        precision = (msg.get("trt", {}).get("precision") or "fp16").lower()
        progress.send("convert.trt", 60, f"build TensorRT ({precision})")
        out_engine = os.path.join(tempfile.gettempdir(), f"{job_id}.engine")
        to_tensorrt(tmp_onnx, out_engine, precision=precision)

        progress.send("upload", 90, "upload engine")
        bucket = msg["output"]["s3_bucket"]
        key    = f"{msg['output']['prefix'].rstrip('/')}/best.engine"
        upload_s3(out_engine, bucket, key)
        progress.done({"s3_bucket": bucket, "s3_uri": f"s3://{bucket}/{key}"})
    except Exception as e:
        logger.exception("trt failed")
        progress.error("convert.trt", str(e))
    finally:
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

    logger.info("All routers started in one process. Waiting for jobs...")
    try:
        # 메인 스레드는 단순 대기(신호 처리/헬스체크 확장 가능)
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Shutting down.")

if __name__ == "__main__":
    main()
