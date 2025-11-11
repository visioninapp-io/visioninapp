from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from llm.graph.training.state import TrainState


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x).strip().replace("%", ""))
    except Exception:
        return None


# evaluate_trial와 동일(또는 상위집합) 별칭 매핑
_METRIC_ALIASES: Dict[str, List[str]] = {
    "mAP50-95": ["mAP50-95", "map50-95", "metrics/mAP50-95(B)", "metrics/mAP50-95",
                 "box/mAP50-95", "seg/mAP50-95", "bbox_map50_95", "map_50_95"],
    "mAP50":    ["mAP50", "map50", "metrics/mAP50", "box/mAP50", "seg/mAP50", "bbox_map50"],
    "precision":["precision", "metrics/precision", "P", "box/P", "seg/P"],
    "recall":   ["recall", "metrics/recall", "R", "box/R", "seg/R"],
    "f1":       ["f1", "metrics/f1", "F1", "box/F1", "seg/F1"],
    "fitness":  ["fitness", "metrics/fitness"],
}


def _get_metric_value(metrics: Dict[str, Any], canonical_key: str) -> Optional[float]:
    if not metrics:
        return None
    # 정확 매칭 우선
    if canonical_key in metrics:
        v = _to_float(metrics[canonical_key])
        if v is not None:
            return v
    # 별칭 탐색
    for alias in _METRIC_ALIASES.get(canonical_key, []):
        if alias in metrics:
            v = _to_float(metrics[alias])
            if v is not None:
                return v
    # 소문자 키 매칭(최후)
    low = {str(k).lower(): v for k, v in metrics.items()}
    if canonical_key.lower() in low:
        return _to_float(low[canonical_key.lower()])
    return None


def _resolve_prev_score_strict(state: TrainState, primary_metric: str) -> Tuple[Optional[float], str, Optional[str]]:
    """
    현재 대표 메트릭(primary_metric)과 **동일 메트릭일 때만** prev_score를 반환.
    우선순위:
      1) state.regression.prev_score (단, regression.prev_metric == primary_metric 인 경우)
      2) state.best_trial.metrics[primary_metric] (별칭 포함)
      3) state.best_trial.metric == primary_metric 인 경우에 한해 state.best_trial.score
      (그 외: fitness 등 불일치 지표는 **사용하지 않음**)
    반환: (prev_score, prev_source, prev_metric_name)
    """
    # 1) 명시 prev_score + prev_metric가 들어있다면 동일할 때만 사용
    reg = state.regression or {}
    reg_prev = _to_float(reg.get("prev_score"))
    reg_metric = reg.get("prev_metric") or reg.get("metric")
    if reg_prev is not None and (reg_metric == primary_metric):
        return reg_prev, "regression.prev_score", reg_metric

    # 2) best_trial.metrics에서 동일 키로 탐색
    bt = state.best_trial or {}
    bt_metrics = (bt.get("metrics") or {})
    v = _get_metric_value(bt_metrics, primary_metric)
    if v is not None:
        return v, f"best_trial.metrics[{primary_metric}]", primary_metric

    # 3) best_trial.metric 이 명시되어 있고, 그게 primary_metric과 같으면 score 사용
    bt_metric_name = bt.get("metric")
    bt_score = _to_float(bt.get("score"))
    if bt_score is not None and (bt_metric_name == primary_metric):
        return bt_score, "best_trial.score(matched_metric)", bt_metric_name

    # ❗ 그 외(예: fitness 등 불일치) 는 사용하지 않음
    return None, f"none(incompatible or missing; bt.metric={bt_metric_name})", None


def regression_gate(state: TrainState) -> TrainState:
    """
    회귀/게이트 판정.
    - 현재 점수: evaluate_trial가 기록한 state.metrics['score'] (대표 메트릭은 state.metrics['primary_metric'])
    - 이전 점수: 동일 메트릭일 때만 사용 (fitness ↔ mAP 혼용 금지)
    """
    print("[regression_gate] 노드 시작")

    metrics = state.metrics or {}
    gate_cfg = state.gate or {}

    current_score = _to_float(metrics.get("score"))
    primary_metric = (metrics.get("primary_metric") or "mAP50-95")

    prev_score, prev_src, prev_metric = _resolve_prev_score_strict(state, primary_metric)

    gate_threshold = _to_float(
        gate_cfg.get("min_map")
        or gate_cfg.get("min_score")
        or gate_cfg.get("threshold")
        or 0.0
    )
    tolerance = float(gate_cfg.get("tolerance", 0.01))  # 1% 기본 허용 오차

    print(f"[regression_gate] current={current_score}({primary_metric}), prev={prev_score}({prev_metric}) src={prev_src}, threshold={gate_threshold}")

    # --- 판정 ---
    if current_score is None:
        result, reason = "FAIL", "현재 점수가 없음"
    elif current_score < gate_threshold:
        result, reason = "FAIL", f"기준치({gate_threshold}) 미달"
    elif prev_score is not None:
        # prev 존재(동일 메트릭)인 경우에만 회귀 비교
        if current_score + tolerance < prev_score:
            result, reason = "FAIL", f"이전({prev_score}) 대비 성능 하락"
        elif abs(current_score - prev_score) <= tolerance:
            result, reason = "WARN", f"이전 대비 변화 미미(±{tolerance})"
        else:
            result, reason = "PASS", "성능 기준 통과"
    else:
        # prev가 없거나 메트릭 불일치라 비교 불가 → 기준만으로 판단
        result, reason = ("PASS", "이전 동일 메트릭 없음; 기준만 통과")
        if current_score < gate_threshold:
            result, reason = "FAIL", f"기준치({gate_threshold}) 미달(비교 불가)"

    # state 반영
    # test용 "FAIL" 원래 값 result
    state.gate_result = "FAIL"
    state.regression = {
        "prev_score": prev_score,
        "prev_metric": prev_metric,
        "current_score": current_score,
        "primary_metric": primary_metric,
        "threshold": gate_threshold,
        "tolerance": tolerance,
        "reason": reason,
        "prev_source": prev_src,
    }

    # context 로그
    c = state.context or {}
    c["regression_gate"] = {
        "result": result,
        "reason": reason,
        "current": current_score,
        "primary_metric": primary_metric,
        "prev": prev_score,
        "prev_metric": prev_metric,
        "prev_source": prev_src,
        "threshold": gate_threshold,
        "tolerance": tolerance,
    }
    state.context = c

    print(f"[regression_gate] 결과 = {result} ({reason}) ✅")
    return state
