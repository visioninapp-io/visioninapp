from __future__ import annotations
from typing import Any, Dict, Optional
import os
from graph.training.state import TrainState

# LLM (optional)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    _LLM = True
except Exception:
    _LLM = False

def _param(name, default, state: TrainState, tr_cfg: Dict[str, Any]):
    # 1) 사용자 오버라이드
    ov = (state.train_overrides or {})
    if name in ov and ov[name] is not None:
        return ov[name]
    # 2) training.yaml(train 섹션)
    tr = tr_cfg.get("train") or {}
    if name in tr and tr[name] is not None:
        return tr[name]
    # 3) 기본값
    return default

def _heuristic_decide(state: TrainState) -> bool:
    """
    LLM이 없거나 실패할 때의 폴백 휴리스틱.
    대략적 기준:
      - 탐색공간이 비어있거나 max_trials < 2 → HPO 비활성
      - 데이터가 충분하고(이미지 5k+) 시간이 있고(epochs>=100) GPU면 → HPO 활성
    """
    hpo = state.hpo or {}
    enabled_cfg = bool(hpo.get("enabled", False))
    space = hpo.get("search_space") or {}
    max_trials = int(hpo.get("max_trials", 0) or 0)
    if not space or max_trials < 2:
        return False

    # 데이터 크기 추정
    stats = state.dataset_stats or {}
    n_tr = int(((stats.get("num_images") or {}).get("train") or 0))
    # 자원/시간 추정
    tr = (state.train_cfg or {}).get("train") or {}
    epochs = int(tr.get("epochs", 100))
    device = ((state.context or {}).get("device") or "cpu")
    gpu_like = device.startswith("cuda")

    # 간단한 규칙
    if gpu_like and (n_tr >= 5000 or epochs >= 100):
        return True
    # 사용자가 already enabled 해뒀으면 존중 (단, space/max_trials 조건 충족)
    if enabled_cfg:
        return True
    return False


def hpo_decider(state: TrainState) -> TrainState:
    """
    - 사용자/시스템 사양을 LLM에 요약 전달
    - LLM이 'use_hpo' | 'single_trial' 결정을 JSON으로 반환하도록 유도
    - 실패시 휴리스틱 폴백
    - 결과는 state.force_hpo(True/False)로 기록 (라우터가 최우선 반영)
    """
    # 이미 강제 플래그가 있으면 그대로 존중
    # if hasattr(state, "force_hpo") and state.force_hpo is not None:
    #     return state

    # 기본 입력 정리
    hpo = state.hpo or {}
    space      = hpo.get("search_space") or {}
    max_trials = int(hpo.get("max_trials", 0) or 0)
    metric     = hpo.get("metric", "mAP50-95")
    direction  = hpo.get("direction", "maximize")

    cfg  = state.train_cfg or {}
    # 여기서부터는 오버라이드 우선
    epochs    = int(_param("epochs",    100, state, cfg))
    _raw_batch = _param("batch", 16, state, cfg)
    try:
        batch = int(_raw_batch) if _raw_batch not in (None, "null", "") else 16
    except ValueError:
        batch = 16

    imgsz     = int(_param("imgsz",     640, state, cfg))
    optimizer =     _param("optimizer", "SGD", state, cfg)

    stats   = state.dataset_stats or {}
    n_train = int(((stats.get("num_images") or {}).get("train") or 0))
    n_val   = int(((stats.get("num_images") or {}).get("val") or 0))

    ctx     = state.context or {}
    device  = ctx.get("device", "cpu")
    amp     = bool(ctx.get("amp", False))
    model_variant = state.model_variant or (_param("model_name", "", state, cfg) or "")
    notes = state.notes or {}

    # LLM 시도
    use_hpo: Optional[bool] = None
    if _LLM:
        try:
            llm = ChatOpenAI(
                model="gpt-5-mini", 
                api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                temperature=1
            )
            prompt = ChatPromptTemplate.from_template(
                """
                당신은 AutoML 의사결정 보조입니다. 아래 사양을 보고
                하이퍼파라미터 탐색(HPO)을 할지 여부를 JSON으로만 반환하세요.

                조건:
                - search_space가 비어있거나 max_trials<2면 single_trial 권장
                - GPU 사용 가능, 데이터가 충분(예: train>=5000)하거나 epochs가 길면 HPO 권장
                - 임베디드/경량 타깃에서 급한 납기면 single_trial 권장
                - notes에 사용자가 hpo를 원하는 내용이 포함될 경우에만 HPO 사용
                - 출력은 {{"decision": "use_hpo" | "single_trial", "reason": "짧은 설명"}} 만.

                사양:
                - model_variant: {model_variant}
                - device: {device} (amp={amp})
                - dataset: train={n_train}, val={n_val}
                - train: epochs={epochs}, batch={batch}, imgsz={imgsz}, optimizer={optimizer}
                - hpo: dims={dims}, max_trials={max_trials}, metric={metric}, direction={direction}
                - notes: notes="{notes}"

                답변은 JSON 한 줄로만:
                """
            )
            out = (prompt | llm).invoke({
                "model_variant": model_variant,
                "device": device,
                "amp": str(amp),
                "n_train": n_train,
                "n_val": n_val,
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "optimizer": optimizer,
                "dims": len(space),
                "max_trials": max_trials,
                "metric": metric,
                "direction": direction,
                "notes": notes,
            })
            import json as _json
            txt = getattr(out, "content", "") or ""
            data = _json.loads(txt)
            decision = (data.get("decision") or "").strip().lower()
            if decision in ("use_hpo", "single_trial"):
                use_hpo = (decision == "use_hpo")
                # 간단 로그
                print(f"[hpo_decider] LLM decision={decision} reason={data.get('reason')}")
        except Exception as e:
            print(f"[hpo_decider] LLM failed: {e}")

    # 폴백 휴리스틱
    if use_hpo is None:
        use_hpo = _heuristic_decide(state)
        print(f"[hpo_decider] heuristic decision={'use_hpo' if use_hpo else 'single_trial'}")

    # 라우터 최우선 반영용 플래그
    setattr(state, "force_hpo", bool(use_hpo))

    # 컨텍스트 기록
    c = state.context or {}
    c["hpo_decider"] = {
        "llm_used": use_hpo is not None and _LLM,
        "decision": "use_hpo" if use_hpo else "single_trial",
        "dims": len(space),
        "max_trials": max_trials,
        "n_train": n_train,
        "epochs": epochs,
        "device": device,
    }
    state.context = c
    return state
