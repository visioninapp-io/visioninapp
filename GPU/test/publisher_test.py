# publisher_train.py
import os
import json
import pika

AMQP_URL    = os.getenv("RABBITMQ_URL", "amqp://admin:ssafy1234@k13s303.p.ssafy.io:5672/%2F")
EXCHANGE    = os.getenv("RMQ_EXCHANGE", "jobs.cmd")   # topic exchange
ROUTING_KEY = "train.start"

def main():
    # 1) 발행할 메시지 구성 (요구 사양 그대로)
    payload = {
        "job_id": "abc12345",
        "dataset": {
            "s3_prefix": "datasets/dataset_4/",
            "name": "dataset_4"
        },
        "output": {
            "s3_bucket": "visioninapp-bucket",
            "prefix": "result/test/"
        },
        "hyperparams": {
            "model": "yolo12n",        # 필요시 "yolov12n.pt" 등으로 교체
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "device": None,            # 미지정 시 Ultralytics 기본
            "workers": 8,
            "optimizer": None,         # "null"/None이면 전달 안 함(서버에서 처리)
            "lr0": 0.01,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "patience": 30,
            "save": True,
            "augment": None,           # 미지정 시 라이브러리 기본
            "mosaic": None,
            "mixup": None
        }
    }

    # 2) 브로커 연결
    params = pika.URLParameters(AMQP_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()

    # 3) 토폴로지(멱등)
    ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)

    # 4) 메시지 속성 (correlation_id = job_id)
    props = pika.BasicProperties(
        content_type="application/json",
        delivery_mode=2,  # persistent
        correlation_id=payload["job_id"]
    )

    # 5) 발행
    ch.basic_publish(
        exchange=EXCHANGE,
        routing_key=ROUTING_KEY,
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        properties=props
    )

    print(f"[PUB] sent job_id={payload['job_id']} rk={ROUTING_KEY}")
    conn.close()

if __name__ == "__main__":
    main()
