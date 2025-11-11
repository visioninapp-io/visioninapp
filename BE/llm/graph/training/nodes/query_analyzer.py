# graph/training/nodes/query_analyzer.py
from __future__ import annotations

import json
import re
import os
from typing import Any, Dict, Optional

from llm.graph.training.state import TrainState

# --- LLM (optional) ---
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    _LLM = True
except Exception:
    _LLM = False


def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def _merge_if_absent(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """dst에 없는 키만 src에서 채워 넣음 (얕은 병합)."""
    out = dict(dst or {})
    for k, v in (src or {}).items():
        if v is None:
            continue
        if k not in out:
            out[k] = v
    return out


def _parse_with_llm(user_query: str) -> Dict[str, Any]:
    """
    자연어 요구사항을 구조화. LLM 실패 시 빈 딕셔너리 반환.
    모델/정밀도 표기는 'v' 없이(yolo8n, yolo11s, yolo12x)라는 팀 규칙을 따름.
    """
    if not _LLM or not _clean(user_query):
        return {}
    try:
        llm = ChatOpenAI(
            model="gpt-5", 
            api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            temperature=1
        )
        prompt = ChatPromptTemplate.from_template(
            """
            너는 AutoML 파이프라인의 요청 분석기다.
            사용자의 자연어 요청에서 훈련 관련 신호를 추출해 JSON으로만(설명 없이, 한 줄) 출력하라.

            규칙:
            - 모델명은 yolo8n/yolo11n/yolo12x처럼 'v' 없이 모델 크기를 포함해서 표기.
            - 모델을 이야기하지 않았을 경우 yolo12n를 기본값으로 사용
            - 사용자가 '튜닝/최적화/HPO'를 원하면 use_hpo=true.
            - epochs, batch, imgsz에 대한 사용자 언급이 있으면 해당 값을 정수로 사용
            - 긴급/빨리/납기 촉박 등의 표현이 있으면 urgent=true, 없으면 urgent=false.
            - 수치가 있으면 정수로, 없으면 null.
            - onnx 혹은 tensorRT로 변환을 원하는 경우 해당 값을 true, 해당 내용이 없거나 원하지 않을 경우 false로 설정해라
            - 출력 키만 사용하라.

            출력 스키마:
            {{"intent":"train|retrain|optimize|export|add_dataset|null",
              "model_variant":"yolo8n|yolo11s|yolo12m|...|null",
              "precision":"fp16|fp32|int8|null",
              "epochs":null,
              "batch":null,
              "imgsz":null,
              "optimizer":"SGD|Adam|AdamW|null",
              "use_hpo":null,
              "max_trials":null,
              "urgent":null,
              "onnx":null,
              "tensorrt":null,
              "notes":"짧은 요약"
            }}

            사용자 요청: "{query}"
            """
        )
        res = (prompt | llm).invoke({"query": user_query})
        content = _clean(getattr(res, "content", ""))
        data = json.loads(content) if content else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# --- 규칙 기반(LLM 미사용/실패 폴백) ---
_HPO_TOKENS = ("hpo", "튜닝", "최적화", "탐색", "search", "hyperparam")
_URGENT_TOKENS = ("빨리", "급", "긴급", "asap", "urgent", "납기", "deadline")


def _parse_with_rules(user_query: str) -> Dict[str, Any]:
    uq = _clean(user_query).lower()
    if not uq:
        return {}

    def has_any(tokens) -> bool:
        return any(t in uq for t in tokens)

    # intent 추정
    intent = None
    if any(w in uq for w in ("재학습", "retrain", "fine-tune", "finetune", "파인튜닝")):
        intent = "retrain"
    elif any(w in uq for w in ("export", "배포", "내보내기", "onnx", "tensorrt")):
        intent = "export"
    elif any(w in uq for w in ("최적화", "튜닝", "optimize", "hpo")):
        intent = "optimize"
    elif any(w in uq for w in ("데이터", "dataset", "추가", "add dataset")):
        intent = "add_dataset"
    elif any(w in uq for w in ("학습", "train", "훈련", "start")):
        intent = "train"

    # 수치 추출(대략)
    def _find_int(patterns):
        for p in patterns:
            m = re.search(p, uq)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        return None

    epochs = _find_int([r"epochs?\s*[:= ]\s*(\d+)", r"(\d+)\s*epochs?"])
    batch = _find_int([r"batch(?:size)?\s*[:= ]\s*(\d+)", r"(\d+)\s*batch"])
    imgsz = _find_int([r"imgsz\s*[:= ]\s*(\d+)", r"image\s*size\s*[:= ]\s*(\d+)", r"(\d+)\s*px"])

    # optimizer/precision
    optimizer = None
    if "adamw" in uq:
        optimizer = "AdamW"
    elif "adam" in uq:
        optimizer = "Adam"
    elif "sgd" in uq:
        optimizer = "SGD"

    precision = None
    if "fp16" in uq or "half" in uq:
        precision = "fp16"
    elif "fp32" in uq or "full" in uq:
        precision = "fp32"
    elif "int8" in uq or "quant" in uq:
        precision = "int8"

    # model_variant (단순 토큰 캡처: yolo<digits><suffix>)
    mv = None
    m = re.search(r"(yolo\d{1,2}[nsmlx])", uq)
    if m:
        mv = m.group(1)

    return {
        "intent": intent,
        "model_variant": mv,
        "precision": precision,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "optimizer": optimizer,
        "use_hpo": True if has_any(_HPO_TOKENS) else None,
        "urgent": True if has_any(_URGENT_TOKENS) else None,
        "notes": user_query.strip()[:200],
    }


def query_analyzer(state: TrainState) -> TrainState:
    """
    - 사용자 자연어 요청을 분석하여 intent/overrides/HPO 힌트를 추출
    - LLM 우선, 실패 시 규칙 기반 파서 사용
    - set-if-absent 원칙으로 state에 합류
    """
    user_query = _clean(state.user_query)
    parsed: Dict[str, Any] = {}

    if user_query:
        parsed = _parse_with_llm(user_query)
        if not parsed:
            parsed = _parse_with_rules(user_query)
            print("[query_analyzer] llm 미사용")

    # ---- 결과를 state에 비파괴적으로 반영 ----
    # intent / model_variant / precision / notes
    for k in ("intent", "model_variant", "precision", "notes"):
        v = parsed.get(k)
        if v is not None and getattr(state, k, None) in (None, "", False):
            setattr(state, k, v)

    # train_overrides
    overrides_new = {}
    for k in ("epochs", "batch", "imgsz", "optimizer"):
        if parsed.get(k) is not None:
            overrides_new[k] = parsed[k]
    state.train_overrides = _merge_if_absent(state.train_overrides or {}, overrides_new)

    # HPO 힌트
    hpo = dict(state.hpo or {})
    use_hpo = parsed.get("use_hpo")
    if use_hpo is True:
        hpo["enabled"] = True
        # 사용자가 max_trials를 말했으면 반영, 없으면 기본(예: 20) 제안
        if "max_trials" in parsed and parsed["max_trials"] is not None:
            hpo["max_trials"] = int(parsed["max_trials"])
        elif "max_trials" not in hpo:
            hpo["max_trials"] = 20
        # 간단한 기본 탐색공간이 없으면 비워두지 말고 최소 골격 제공
        hpo.setdefault("search_space", {
            "lr0": [1e-3, 5e-3, 1e-2],
            "batch": [8, 16, 32],
            "imgsz": [512, 640],
            "optimizer": ["SGD", "AdamW"],
        })
    state.hpo = hpo or None


    # 컨텍스트 로깅
    c = state.context or {}
    c["query_analyzer"] = {
        "used_llm": bool(user_query and _LLM),
        "parsed": parsed,
        "train_overrides": state.train_overrides,
        "hpo_enabled": state.hpo.get("enabled") if state.hpo else False,
    }
    state.context = c
    print("[query_analyzer] 질의 분석 완료")
    return state
