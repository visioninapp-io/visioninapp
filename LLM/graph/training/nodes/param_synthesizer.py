# graph/training/nodes/param_synthesizer.py
from __future__ import annotations

import json
import os
from typing import Any, Dict

from graph.training.state import TrainState

# LLM 관련
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False

_LLM_TEMPLATE = """
너는 YOLO 학습 파라미터 설계 전문가다.
아래 입력(JSON)을 바탕으로 학습 파라미터를 JSON 한 줄로만 출력하라.
- 사용자가 지정한 값(user_overrides)은 절대 덮어쓰지 말고 그대로 둔다.
- intent, model_variant, device, precision 등을 참고해 나머지 값을 완성하라.
- 출력은 JSON 한 줄, 설명이나 추가 텍스트 없이.
- batch는 32이하로 설정하라

예시 출력:
{{
"epochs": 120,
"batch": 32,
"imgsz": 640,
"optimizer": "AdamW",
"lr0": 0.001,
"lrf": 0.01,
"momentum": 0.937,
"weight_decay": 0.0005,
"patience": 30,
"augmentation_level": "light"
}}

입력:
{payload}
"""

def param_synthesizer(state: TrainState) -> TrainState:
    """
    LLM을 사용해 학습 하이퍼파라미터를 생성하고,
    query_analyzer의 parsed 정보를 참고하여 반영.
    """

    # ----- 1️⃣ query_analyzer 결과 가져오기 -----
    ctx = dict(getattr(state, "context", {}) or {})
    qa_ctx = ctx.get("query_analyzer", {})
    parsed = qa_ctx.get("parsed") or {}

    # ----- 2️⃣ 입력 payload 구성 -----
    device = ctx.get("device") or "cpu"
    precision = state.precision or ("fp16" if ctx.get("amp") else "fp32")

    payload = {
        "intent": parsed.get("intent") or state.intent or "train",
        "model_variant": parsed.get("model_variant") or state.model_variant or "yolo11n",
        "precision": parsed.get("precision") or precision,
        "device": device,
        "user_overrides": {
            "epochs": parsed.get("epochs"),
            "batch": parsed.get("batch"),
            "imgsz": parsed.get("imgsz"),
            "optimizer": parsed.get("optimizer"),
        },
        "notes": parsed.get("notes"),
        "use_hpo": parsed.get("use_hpo"),
    }

    # None 제거
    payload = {k: v for k, v in payload.items() if v is not None}
    payload["user_overrides"] = {k: v for k, v in payload["user_overrides"].items() if v is not None}

    print(f"[param_synthesizer] 입력 payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    # ----- 3️⃣ LLM 호출 -----
    generated = {}
    if _LLM_AVAILABLE:
        try:
            llm = ChatOpenAI(
                model="gpt-5-mini", 
                api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                temperature=1
            )
            prompt = ChatPromptTemplate.from_template(_LLM_TEMPLATE)
            out = (prompt | llm).invoke({"payload": json.dumps(payload, ensure_ascii=False)})
            content = getattr(out, "content", "") or ""
            generated = json.loads(content)
        except Exception as e:
            print(f"[param_synthesizer] ⚠️ LLM 호출 실패 또는 JSON 파싱 오류: {e}")
            generated = {}

    # ----- 4️⃣ 병합 순서: train_defaults < generated < user_overrides -----
    train_defaults = (state.train_cfg or {}).get("train") or {}
    user_overrides = dict(state.train_overrides or {})

    final = {**train_defaults, **generated, **user_overrides}
    final["model_name"] = f"{state.model_variant}"

    # ----- 5️⃣ 결과 저장 -----
    ctx["param_synthesizer"] = {
        "input_payload": payload,
        "generated": generated,
        "final_params": final,
        "note": "merge order = train_defaults < LLM_generated < user_overrides",
    }
    state.context = ctx
    state.train_overrides = final

    print("[param_synthesizer] 최종 병합 결과:")
    print(json.dumps(final, ensure_ascii=False, indent=2))

    return state
