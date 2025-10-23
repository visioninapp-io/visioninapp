# graph/training/nodes/select_best.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from graph.training.state import TrainState

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        return float(str(x).strip().replace("%",""))
    except Exception:
        return None

_METRIC_ALIASES: Dict[str, List[str]] = {
    "mAP50-95": ["mAP50-95","map50-95","metrics/mAP50-95(B)","metrics/mAP50-95","box/mAP50-95"],
    "mAP50":    ["mAP50","map50","metrics/mAP50","box/mAP50"],
    "precision":["precision","metrics/precision","P","box/P"],
    "recall":   ["recall","metrics/recall","R","box/R"],
    "f1":       ["f1","metrics/f1","F1","box/F1"],
    "fitness":  ["fitness","metrics/fitness"],  # ✅ HPO 기본
}

def _get_metric_value(metrics: Dict[str, Any], metric_key: str) -> Optional[float]:
    if not metrics: return None
    if metric_key in metrics:
        v = _to_float(metrics[metric_key]);  return v
    for alias in _METRIC_ALIASES.get(metric_key, []):
        if alias in metrics:
            v = _to_float(metrics[alias]);  return v
    low = {str(k).lower(): v for k, v in metrics.items()}
    if metric_key.lower() in low:
        return _to_float(low[metric_key.lower()])
    return None

def _is_better(a: Optional[float], b: Optional[float], direction: str) -> bool:
    if a is None: return False
    if b is None: return True
    return a > b if direction == "maximize" else a < b

def _choose_best_trial(trials: List[Dict[str, Any]], direction: str, prefer_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[float], Optional[str]]:
    # ✅ prefer_key를 최우선으로, 없으면 아래 순서
    search_order = [prefer_key] + [k for k in ["mAP50-95","mAP50","f1","precision","recall","fitness"] if k != prefer_key]
    best_trial, best_idx, best_score, used_key = None, None, None, None

    for i, t in enumerate(trials):
        m = t.get("metrics") or {}
        for key in search_order:
            score = _get_metric_value(m, key)
            if score is None:
                continue
            if _is_better(score, best_score, direction):
                best_trial, best_idx, best_score, used_key = t, i, score, key
    return best_trial, best_idx, best_score, used_key

def select_best(state: TrainState) -> TrainState:
    hpo_cfg = (state.hpo or {})
    # ✅ HPO에서 지정한 metric(기본 fitness)을 우선
    metric_key = (hpo_cfg.get("metric") or "fitness")
    direction = (hpo_cfg.get("direction", "maximize") or "maximize").lower()
    if direction not in ("maximize","minimize"):
        direction = "maximize"

    trials = state.hpo_trials or []
    used_hpo = bool(trials)

    if used_hpo:
        best_trial, best_idx, best_score, used_key = _choose_best_trial(trials, direction, metric_key)

        if best_trial is None:
            print("[select_best] WARN: trials는 있으나 유효 metric을 찾지 못했습니다. single-run으로 폴백합니다.")
            used_hpo = False
        else:
            state.best_trial = {
                "index": best_idx,
                "score": best_score,
                "metric": used_key or metric_key,
                "direction": direction,
                "params": (best_trial.get("params") or {}),
                "metrics": (best_trial.get("metrics") or {}),
                "model_path": best_trial.get("model_path"),
                "name": best_trial.get("name"),
            }
            state.metrics = state.best_trial["metrics"]
            state.model_path = state.best_trial.get("model_path")

            c = state.context or {}
            c["select_best"] = {
                "used_hpo": True,
                "trials": len(trials),
                "metric": used_key or metric_key,
                "direction": direction,
                "best_index": best_idx,
                "best_score": best_score,
                "best_model_path": state.model_path,
            }
            state.context = c

            print(f"[select_best] HPO best: idx={best_idx} {(used_key or metric_key)}={best_score} path={state.model_path}")
            return state

    # single-run (학습 결과 그대로)
    metrics = state.metrics or {}
    model_path = state.model_path
    used_key = None
    score = None
    for key in ["mAP50-95","mAP50","f1","precision","recall","fitness"]:
        score = _get_metric_value(metrics, key)
        if score is not None:
            used_key = key; break

    state.best_trial = {
        "index": None,
        "score": score,
        "metric": used_key or metric_key,
        "direction": direction,
        "params": (state.train_overrides or {}) or ((state.best_trial or {}).get("params") or {}),
        "metrics": metrics,
        "model_path": model_path,
        "name": "single_run",
    }

    c = state.context or {}
    c["select_best"] = {"used_hpo": False, "metric": used_key or metric_key, "direction": direction, "score": score, "best_model_path": model_path}
    state.context = c

    print(f"[select_best] single-run: {(used_key or metric_key)}={score} path={model_path}")
    return state
