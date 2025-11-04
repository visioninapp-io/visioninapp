# graph/training/nodes/decide_mode.py
from __future__ import annotations

from typing import Any, Dict, Optional

from graph.training.state import TrainState


def _bool(x: Any) -> bool:
    return bool(x) and x is not False and x is not None


def _has_dataset(state: TrainState) -> bool:
    """
    dataset_version가 있거나, training.yaml의 data.yaml_path가 있으면
    '데이터셋이 준비되었다'고 판단.
    """
    if _bool(getattr(state, "dataset_version", None)):
        return True
    cfg = getattr(state, "train_cfg", None) or {}
    data_yaml = (cfg.get("data") or {}).get("yaml_path")
    return _bool(data_yaml)


def _has_base_model(state: TrainState) -> bool:
    """
    base_model이 있거나 training.yaml의 resume.finetune_base가 있으면 파인튜닝 모드.
    """
    if _bool(getattr(state, "base_model", None)):
        return True
    cfg = getattr(state, "train_cfg", None) or {}
    finetune_base = (cfg.get("resume") or {}).get("finetune_base")
    return _bool(finetune_base)


def _wants_resume(state: TrainState) -> bool:
    """
    명시적 resume 플래그 또는 training.yaml의 resume.enable/last_ckpt가 있으면 재개 학습 모드.
    """
    if _bool(getattr(state, "resume", None)):
        return True
    cfg = getattr(state, "train_cfg", None) or {}
    r = (cfg.get("resume") or {})
    return _bool(r.get("enable")) or _bool(r.get("last_ckpt"))


def decide_mode(state: TrainState) -> TrainState:
    """
    - 현재 상태와 설정을 확인해 모드 힌트를 context에 기록만 함.
    - 실제 분기는 route_mode에서 수행되며, 그래프에서는
      g.add_conditional_edges("decide_mode", route_mode, {...}) 로 연결.
    """
    has_ds = _has_dataset(state)
    has_base = _has_base_model(state)
    wants_resume = _wants_resume(state)

    if not has_ds:
        mode = "error_missing_dataset"
        reason = "dataset_version/data.yaml 없음"
    elif wants_resume:
        mode = "path_resume"
        reason = "resume 요청 감지(resume.enable/last_ckpt/flag)"
    elif has_base:
        mode = "path_finetune"
        reason = "base_model/finetune_base 지정"
    else:
        mode = "path_fresh"
        reason = "기본(fresh) 학습"

    # context에 기록 (로그/디버그용)
    ctx = getattr(state, "context", {}) or {}
    ctx["decide_mode"] = {
        "has_dataset": has_ds,
        "has_base_model": has_base,
        "wants_resume": wants_resume,
        "mode": mode,
        "reason": reason,
    }
    state.context = ctx

    print(f"[decide_mode] mode={mode} ({reason})")
    return state


def route_mode(state: TrainState) -> str:
    """
    그래프 분기용 라우터. decide_mode와 같은 로직을 재사용(독립적으로 계산).
    """
    if not _has_dataset(state):
        return "error_missing_dataset"
    if _wants_resume(state):
        return "path_resume"
    if _has_base_model(state):
        return "path_finetune"
    return "path_fresh"
