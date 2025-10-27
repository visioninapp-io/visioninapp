from __future__ import annotations
from typing import Any, Dict


DEFAULTS = {
    "min_overall": 0.65,
    "min_key_hits": 0.50,
    "max_loops": 5,
    "key_weights": ("model_variant", "precision", "use_hpo"),
}

# graph/eval/selfrag_decider.py
def selfrag_decider(state) -> dict:
    """
    selfrag_scorer의 결과를 평가하여 다음 단계를 결정:
    - overall/key_hits 기준 미달 → feedback
    - 일정 횟수 초과 → fallback
    - 통과 → pass
    """
    ctx = dict(state.context or {})
    scorer = (ctx.get("selfrag_scorer") or {})
    overall = float(scorer.get("overall", 0.0))
    key_hits = float(scorer.get("key_hits", 0.0))
    thresholds = scorer.get("thresholds", {})

    # retry counter
    loop_info = ctx.get("selfrag_decider", {})
    retry_count = loop_info.get("retry_count", 0) + 1

    # 기본 임계값
    min_overall = thresholds.get("min_overall", 0.5)
    min_key_hits = thresholds.get("min_key_hits", 0.5)

    # 결정 로직
    if retry_count > 5:
        decision = "fallback"
    elif overall >= min_overall and key_hits >= min_key_hits:
        decision = "pass"
    else:
        decision = "feedback"
    print(f"[selfrag_decider] current count: {retry_count} decision: {decision}")

    ctx["selfrag_decider"] = {
        "decision": decision,
        "overall": overall,
        "key_hits": key_hits,
        "retry_count": retry_count,
    }
    return {"context": ctx}
