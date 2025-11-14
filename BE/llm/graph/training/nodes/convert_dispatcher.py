# graph/training/nodes/convert_dispatcher.py
from __future__ import annotations

import json
import os
from typing import Any, Dict
from llm.graph.training.state import TrainState

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "")
EXCHANGE_CONVERT = os.getenv("RMQ_EXCHANGE_CONVERT", "jobs.event")      # ← 요구사항
RK_CONVERT = os.getenv("RMQ_ROUTING_CONVERT", "convert.exchanges")      # ← 요구사항

def convert_dispatcher(state: TrainState) -> TrainState:
    qa_ctx = (state.context or {}).get("query_analyzer", {})
    parsed = qa_ctx.get("parsed") or {}

    onnx = bool(parsed.get("onnx"))
    tensorrt = bool(parsed.get("tensorrt"))

    payload = {
        "job_id": getattr(state, "job_id", None),
        "onnx": str(onnx).lower(),       # "true"/"false"
        "tensorrt": str(tensorrt).lower(),
        "intent": getattr(state, "intent", None) or "export",
    }

    try:
        import pika
        params = pika.URLParameters(RABBITMQ_URL)
        conn = pika.BlockingConnection(params)
        ch = conn.channel()
        ch.exchange_declare(exchange=EXCHANGE_CONVERT, exchange_type="topic", durable=True)

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ch.basic_publish(
            exchange=EXCHANGE_CONVERT,
            routing_key=RK_CONVERT,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
        )
        conn.close()
        print(f"[convert_dispatcher] ✅ 변환요청 발행 완료: {payload}")
    except Exception as e:
        print(f"[convert_dispatcher] ❌ 변환요청 발행 실패: {e}")
        state.error = f"convert_dispatcher publish failed: {e}"

    return state
