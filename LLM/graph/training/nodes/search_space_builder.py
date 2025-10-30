# graph/training/nodes/search_space_builder.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from graph.training.state import TrainState

Number = Union[int, float]


def _get(cfg: Optional[Dict[str, Any]], path: str, default: Any = None) -> Any:
    """중첩 dict 안전 접근: 'a.b.c'"""
    cur = cfg or {}
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_float_range(v: Any) -> Optional[Tuple[float, float]]:
    """
    [lo, hi] 형태를 float 구간으로 캐스팅. 잘못된 값은 None.
    """
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            lo = float(v[0])
            hi = float(v[1])
            if hi > lo:
                return (lo, hi)
        except Exception:
            return None
    return None


def _as_int_choices(v: Any) -> Optional[List[int]]:
    """
    [a, b, c] 형태를 int choices로 캐스팅. 잘못된 값은 None.
    """
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        try:
            return [int(x) for x in v]
        except Exception:
            return None
    return None


def _normalize_space(raw_space: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    training.yaml의 hpo.search_space 스키마를 내부 표준 스키마로 정규화.
    표준 스키마 예:
      {
        "lr0":   {"type":"float", "low":0.0005, "high":0.005, "log":True},
        "batch": {"type":"int",   "choices":[8,16,32]},
        "momentum": {"type":"float", "low":0.85, "high":0.95},
        ...
      }
    """
    space: Dict[str, Dict[str, Any]] = {}

    for k, v in (raw_space or {}).items():
        # float 구간?
        r = _as_float_range(v)
        if r:
            # 일부 하이퍼파라미터는 로그 스케일이 일반적(lr, wd)
            log_default = k.lower() in ("lr", "lr0", "lrf", "weight_decay")
            space[k] = {"type": "float", "low": r[0], "high": r[1], "log": log_default}
            continue

        # int choices?
        c = _as_int_choices(v)
        if c:
            space[k] = {"type": "int", "choices": c}
            continue

        # 단일 상수 → 고정값
        if isinstance(v, (int, float, str, bool)):
            # 고정값도 명시해두면 스케줄러가 기본값으로 사용 가능
            t = "int" if isinstance(v, int) else "float" if isinstance(v, float) else "str" if isinstance(v, str) else "bool"
            space[k] = {"type": t, "value": v}
            continue

        # 그 외 형식은 무시
    return space


def _defaults_from_train_cfg(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """train 섹션을 평탄화해 단일 시도 기본값으로 사용."""
    tr = train_cfg.get("train") or {}
    return {
        "optimizer": tr.get("optimizer", "SGD"),
        "epochs": int(tr.get("epochs", 100)),
        "batch": int(tr.get("batch", 16)),
        "imgsz": int(tr.get("imgsz", 640)),
        "workers": int(tr.get("workers", 8)),
        "lr0": float(tr.get("lr0", 0.001)),
        "lrf": float(tr.get("lrf", 0.01)),
        "weight_decay": float(tr.get("weight_decay", 5e-4)),
        "momentum": float(tr.get("momentum", 0.937)),
        "warmup_epochs": float(tr.get("warmup_epochs", 3)),
        "warmup_bias_lr": float(tr.get("warmup_bias_lr", 0.1)),
        "augment": bool(tr.get("augment", True)),
        "mosaic": bool(tr.get("mosaic", True)),
        "mixup": bool(tr.get("mixup", False)),
        "amp": bool(tr.get("amp", True)),
        "patience": int(tr.get("patience", 20)),
        # 모델 이름도 적당히 물고가자(ultralytics는 weights 인자로 사용)
        "model_name": tr.get("model_name", None),
    }


def search_space_builder(state: TrainState) -> TrainState:
    """
    - training.yaml에서 HPO 설정을 읽어 정규화된 search space와 기본 파라미터를 생성.
    - state.hpo({enabled, max_trials, metric, direction, sampler, pruner, search_space})을 채움/보정.
    - HPO 미사용이면 단일 시도용 trial 파라미터(state.best_trial)에 기본값 저장.
    """
    cfg = state.train_cfg or {}
    hpo_cfg = (cfg.get("hpo") or {}).copy()

    # 사용자가 state.hpo에 이미 넣어둔 값이 있으면 사용자 > yaml 우선으로 병합
    user_hpo = (state.hpo or {}).copy()
    for k, v in user_hpo.items():
        if v is not None:
            hpo_cfg[k] = v

    enabled = bool(hpo_cfg.get("enabled", False))
    max_trials = int(hpo_cfg.get("max_trials", 20))
    metric = hpo_cfg.get("metric", "mAP50-95")
    direction = hpo_cfg.get("direction", "maximize")
    sampler = hpo_cfg.get("sampler", "TPESampler")
    pruner = hpo_cfg.get("pruner", "MedianPruner")

    raw_space = hpo_cfg.get("search_space") or {}
    space = _normalize_space(raw_space)

    # 단일 시도 기본값(훈련 섹션)
    defaults = _defaults_from_train_cfg(cfg)

    # 정밀도/디바이스는 init_context에서 결정 → 참고용으로 기록
    ctx = state.context or {}
    device = ctx.get("device", "cpu")
    amp_dtype = ctx.get("amp_dtype", "fp32")

    # 최종 HPO 설정 저장
    state.hpo = {
        "enabled": enabled,
        "max_trials": max_trials,
        "metric": metric,
        "direction": direction,
        "sampler": sampler,
        "pruner": pruner,
        "search_space": space,
        "defaults": defaults,
        "device": device,
        "amp_dtype": amp_dtype,
    }

    # 단일 시도 모드면, 다음 노드가 바로 학습에 쓸 수 있게 best_trial에 기본 파라미터 박아둠
    if not enabled:
        state.best_trial = {
            "params": defaults,
            "note": "HPO disabled → defaults from training.yaml.train",
        }

    # 디버그 로그
    print("[search_space_builder] enabled=", enabled)
    print("[search_space_builder] space keys=", list(space.keys()))
    if not enabled:
        print("[search_space_builder] single-trial defaults=", {k: defaults[k] for k in ["optimizer", "epochs", "batch", "lr0", "imgsz"]})

    # context에도 요약 남기기
    c = state.context or {}
    c["search_space"] = {
        "enabled": enabled,
        "num_dims": len(space),
        "max_trials": max_trials,
        "metric": metric,
        "direction": direction,
    }
    state.context = c

    return state
