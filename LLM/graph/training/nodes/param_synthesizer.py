# graph/training/nodes/param_synthesizer.py
from __future__ import annotations

import json
import os
from typing import Any, Dict

from graph.training.state import TrainState
from utils.model_normalizer import normalize_yolo_model_name

# LLM 관련 (옵션)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False


_LLM_TEMPLATE = """
너는 YOLO 계열 모델 학습 파라미터를 구조화하는 어시스턴트다.

아래는 현재까지 정리된 payload JSON이다:

{payload}

요구사항:

1. 반드시 "유효한 JSON만" 출력한다. 설명, 주석, 추가 텍스트 금지.
2. 최상위 depth만 사용한다. (중첩 객체/배열을 만들지 않는다. 배열이 꼭 필요할 때만 허용.)
3. 존재하지 않는 값은 넣지 말고, 추론 가능한 값만 채운다.
4. 다음 키들을 필요시 사용할 수 있다. (모두 선택적)
   - "model": 사용자가 언급한 모델 변형명 (예: "yolov5n", "yolo11n", "yolo12n")
   - "model_name": model과 동일 의미. 중복 지정 시 동일 값으로 둔다.
   - "model_variant": 최종 사용할 모델 가중치/이름(예: "yolov5n", "yolo11s")
   - "epochs", "imgsz", "batch", "device", "precision",
     "optimizer", "lr0", "lrf", "weight_decay", "momentum",
     "patience", "warmup_epochs", "warmup_bias_lr",
     "augment", "mosaic", "mixup", "amp",
     "use_hpo", "notes"
5. 사용자가 명시적으로 의도한 값(user_overrides, model_variant 등)은 최대한 존중한다.
6. 학습 YAML 기본값을 '무조건 덮어쓰지' 말고, 정말 변경 의도가 있을 때만 값을 제안한다.
7. 모델 이름과 관련해서:
   - 사용자가 "yolov5", "yolov5n" 처럼 말하면 그에 맞는 값을 설정한다.
   - 모델 크기에 대한 별다른 언급이 없으면 n모델을 기본으로 해서 작성한다.
   - 답변에 "yolo11", "yolo12" 등 다른 시리즈를 넣을 때는, payload 상에 그런 의도가 있을 때만 한다.
   - 애매하면 model은 비워둔다(생략한다). (이 경우 상위 로직이 기본값을 사용한다.)

출력 형식 예시는 다음과 같다(예시일 뿐, 상황에 맞게 키를 줄여도 된다):

{{
  "model": "yolov8n",
  "epochs": 100,
  "imgsz": 640,
  "batch": 16
}}
"""

def _parse_with_rules(uq: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}

    # ... (기존 epoch, imgsz, batch 등 파싱)

    # YOLO 모델 토큰 캡처
    # 예: yolo5, yolov5, yolo5n, yolov8s, yolo11, yolo11n ...
    m = re.search(r"(yolov?\d{1,2}[a-z0-9\-]*)", uq)
    if m:
        token = m.group(1)
        norm = normalize_yolo_model_name(token)
        if norm:
            parsed["model_variant"] = norm

    return parsed

def _clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """None / 빈 dict 제거."""
    return {
        k: v
        for k, v in (d or {}).items()
        if v is not None and v != {}
    }


def _merge_params(
    train_defaults: Dict[str, Any],
    generated: Dict[str, Any],
    user_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    병합 규칙:
    1) train_defaults    (training.yaml 기본값)
    2) generated         (LLM가 제안한 값)
    3) user_overrides    (사용자/상위 노드에서 직접 지정한 값, 최우선)
    """
    base = _clean_dict(train_defaults)
    gen = _clean_dict(generated)
    user = _clean_dict(user_overrides)

    merged: Dict[str, Any] = {**base, **gen, **user}
    return merged


def param_synthesizer(state: TrainState) -> TrainState:
    """
    query_analyzer 결과 + 기존 train_overrides를 바탕으로
    학습에 사용할 파라미터 후보를 생성해 state.train_overrides에 반영한다.

    ⚠️ 중요:
    - 여기서는 model/model_name/model_variant를 '강제로' 덮어쓰지 않는다.
    - 최종 모델 선택은 train_trial._merge_train_params에서 일관 규칙으로 처리한다.
    """
    # --- 컨텍스트 & query_analyzer 결과 ---
    ctx: Dict[str, Any] = dict(getattr(state, "context", {}) or {})
    qa_ctx: Dict[str, Any] = ctx.get("query_analyzer", {}) or {}
    parsed: Dict[str, Any] = qa_ctx.get("parsed") or {}

    # --- 기본 device / precision 추론 ---
    device = (
        parsed.get("device")
        or getattr(state, "device", "0")
        or "0"
    )

    if "precision" in parsed:
        precision = parsed["precision"]
    elif getattr(state, "precision", None):
        precision = state.precision
    else:
        precision = "fp16" if ctx.get("amp") else "fp32"

    # --- 상위에서 이미 들어온 overrides (최우선 레이어) ---
    user_overrides: Dict[str, Any] = dict(
        getattr(state, "train_overrides", {}) or {}
    )

    # --- LLM에 넘길 payload 구성 ---
    payload: Dict[str, Any] = {
        "intent": parsed.get("intent") or getattr(state, "intent", None) or "train",
        "model_variant": parsed.get("model_variant") or getattr(state, "model_variant", None),
        "device": device,
        "precision": precision,
        "epochs": parsed.get("epochs"),
        "imgsz": parsed.get("imgsz"),
        "batch": parsed.get("batch"),
        "optimizer": parsed.get("optimizer"),
        "lr0": parsed.get("lr0"),
        "lrf": parsed.get("lrf"),
        "weight_decay": parsed.get("weight_decay"),
        "momentum": parsed.get("momentum"),
        "patience": parsed.get("patience"),
        "warmup_epochs": parsed.get("warmup_epochs"),
        "warmup_bias_lr": parsed.get("warmup_bias_lr"),
        "augment": parsed.get("augment"),
        "mosaic": parsed.get("mosaic"),
        "mixup": parsed.get("mixup"),
        "amp": parsed.get("amp"),
        "use_hpo": parsed.get("use_hpo"),
        "notes": parsed.get("notes"),
        "user_overrides": user_overrides,
    }

    payload = _clean_dict(payload)

    print("[param_synthesizer] 입력 payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    # --- LLM 호출 (옵션) ---
    generated: Dict[str, Any] = {}

    if _LLM_AVAILABLE:
        try:
            llm = ChatOpenAI(
                model="gpt-5-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                temperature=1,
            )
            prompt = ChatPromptTemplate.from_template(_LLM_TEMPLATE)
            out = (prompt | llm).invoke(
                {"payload": json.dumps(payload, ensure_ascii=False)}
            )
            content = getattr(out, "content", "") or ""
            generated = json.loads(content)
        except Exception as e:
            print(f"[param_synthesizer] ⚠️ LLM 호출 실패 또는 JSON 파싱 오류: {e}")
            generated = {}
    else:
        # LLM 미사용 시, parsed/user_overrides 기반으로만 진행
        generated = {}

    raw = generated.get("model") or generated.get("model_name") or generated.get("model_variant")
    norm = normalize_yolo_model_name(raw)
    if norm:
        generated["model"] = norm
        generated["model_name"] = norm
        generated["model_variant"] = norm

    # --- training.yaml 기본값 로드 ---
    train_cfg = getattr(state, "train_cfg", {}) or {}
    if isinstance(train_cfg, dict):
        train_defaults: Dict[str, Any] = train_cfg.get("train") or {}
    else:
        train_defaults = {}

    # --- 최종 병합: 기본값 < LLM 결과 < user_overrides ---
    final = _merge_params(train_defaults, generated, user_overrides)

    # ❗ 여기서 model / model_name / model_variant를 강제 세팅하지 않는다.
    #    잘못된 덮어쓰기로 인해 항상 yolo12n으로 돌아가는 문제를 방지하기 위함.
    #    최종 모델 선택은 train_trial._merge_train_params에서 수행.

    # --- 컨텍스트에 저장 ---
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
