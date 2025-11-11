# graph/training/nodes/hpo_scheduler.py
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from llm.graph.training.state import TrainState


def _rstate(seed: int) -> random.Random:
    return random.Random(int(seed))

def _sample_one(rng: random.Random, spec: Dict[str, Any]) -> Any:
    t = (spec or {}).get("type")
    if t == "float":
        lo = float(spec["low"]); hi = float(spec["high"])
        if spec.get("log", False):
            return math.exp(rng.uniform(math.log(lo), math.log(hi)))
        return rng.uniform(lo, hi)
    if t == "int":
        if "choices" in spec:
            return rng.choice(list(spec["choices"]))
    if t == "bool":
        v = spec.get("value")
        return bool(v) if v is not None else rng.choice([True, False])
    if t == "str":
        v = spec.get("value")
        return str(v) if v is not None else ""
    if "value" in spec:
        return spec["value"]
    return None

def _sample_params(rng: random.Random, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {k: _sample_one(rng, spec) for k, spec in (space or {}).items()}

def _mock_objective(params: Dict[str, Any], defaults: Dict[str, Any], seed: int) -> float:
    rng = _rstate(seed + 1337)
    lr = float(params.get("lr0", defaults.get("lr0", 1e-3)) or 1e-3)
    batch = int(params.get("batch", defaults.get("batch", 16)) or 16)
    momentum = float(params.get("momentum", defaults.get("momentum", 0.9)) or 0.9)
    wd = float(params.get("weight_decay", defaults.get("weight_decay", 5e-4)) or 5e-4)
    epochs = int(params.get("epochs", defaults.get("epochs", 100)) or 100)

    lr_score = math.exp(-((math.log10(lr) - math.log10(1e-3)) ** 2) / 0.5)
    batch_score = min(1.0, (batch / 16.0)) * 0.2 + 0.8 * (1.0 if batch >= 8 else 0.7)
    mom_score = math.exp(-((momentum - 0.9) ** 2) / 0.01)
    wd_score = math.exp(-((math.log10(wd) - math.log10(5e-4)) ** 2) / 1.2)
    ep_score = min(1.0, epochs / 150.0)
    noise = rng.uniform(-0.03, 0.03)

    score = 0.35*lr_score + 0.15*batch_score + 0.2*mom_score + 0.15*wd_score + 0.15*ep_score + noise
    return max(0.0, min(1.0, score))

def hpo_scheduler(state: TrainState) -> TrainState:
    """
    의사 랜덤 HPO: 더미 목적함수 점수는 mAP가 아닌 'fitness'로 저장한다.
    """
    hpo: Dict[str, Any] = state.hpo or {}
    hpo["metric"] = "fitness"
    state.hpo = hpo
    
    space: Dict[str, Any] = hpo.get("search_space") or {}
    defaults: Dict[str, Any] = hpo.get("defaults") or {}
    max_trials: int = int(hpo.get("max_trials", 0) or 0)
    direction: str = (hpo.get("direction") or "maximize").lower()

    # HPO 스코어의 의미를 명확히: 기본은 fitness
    # (원하면 config에서 hpo.metric='fitness'로 명시 가능)
    metric_name: str = (hpo.get("metric") or "fitness")

    if not space or max_trials <= 0:
        state.hpo_trials = []
        state.best_trial = {
            "params": defaults,
            "metric": None,
            "note": "HPO disabled or empty space → fallback to defaults",
        }
        ctx = state.context or {}
        ctx["hpo_scheduler"] = {"ran": False, "reason": "empty_space_or_zero_trials", "num_trials": 0}
        state.context = ctx
        print("[hpo_scheduler] No search space or trials; fell back to defaults.")
        return state

    seed = int(getattr(state, "seed", 42))
    rng = _rstate(seed)

    trials: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for t in range(max_trials):
        params = _sample_params(rng, space)
        score = _mock_objective(params, defaults, seed + t)

        trial = {
            "trial_id": t,
            "params": params,
            # ✅ 점수의 정체를 명확히 fitness로 기록
            "metrics": {metric_name: float(score)},
            # 참고용으로 그대로도 둠
            "score": float(score),
        }
        trials.append(trial)

        if best is None:
            best = trial
        else:
            if direction == "maximize":
                if trial["score"] > best["score"]:
                    best = trial
            else:
                if trial["score"] < best["score"]:
                    best = trial

    state.hpo_trials = trials
    if best is None:
        state.best_trial = {"params": defaults, "metric": None, "note": "no_trials_evaluated"}
    else:
        state.best_trial = {
            "params": {**defaults, **best["params"]},
            "metric": best["metrics"].get(metric_name),
            "metric_name": metric_name,  # ✅ 무엇을 기준으로 뽑았는지 명시
            "direction": direction,
            "trial_id": best["trial_id"],
            "metrics": best["metrics"],  # 베스트의 metrics도 함께 보존
        }

    ctx = state.context or {}
    ctx["hpo_scheduler"] = {
        "ran": True,
        "num_trials": len(trials),
        "direction": direction,
        "best_score": state.best_trial.get("metric"),
        "best_metric_name": metric_name,  # ✅ 기록
        "best_trial_id": state.best_trial.get("trial_id"),
    }
    state.context = ctx

    print(f"[hpo_scheduler] ran {len(trials)} trials → best_trial_id={state.best_trial.get('trial_id')} "
          f"{metric_name}={state.best_trial.get('metric'):.4f}")
    return state
