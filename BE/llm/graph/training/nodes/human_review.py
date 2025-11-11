from __future__ import annotations
from llm.graph.training.state import TrainState

def human_review(state: TrainState) -> TrainState:
    """
    FAIL/WARN일 경우 사용자의 입력을 통해 다음 action을 결정.
    PASS일 경우 자동 승인.
    """
    print("[human_review] 노드 실행")

    regression = state.regression or {}
    result = regression.get("result") or state.gate_result
    reason = regression.get("reason")
    prev = regression.get("prev_score")
    curr = regression.get("current_score")
    primary_metric = regression.get("primary_metric", "mAP50-95")

    print(f"[human_review] 결과={result}, 사유={reason}")
    print(f"[human_review] 현재({primary_metric})={curr}, 이전={prev}")

    # --- 사람의 의사 결정 입력 ---
    if result in ("FAIL", "WARN"):
        print("\n[human_review] 모델 성능이 이전보다 낮거나 경계선입니다.")
        print("다음 중 하나를 선택하세요:")
        print("  1. widen_search  → 탐색 범위 확장 재시도")
        print("  2. minor_tune    → 미세 조정 재학습")
        print("  3. approve       → 수동 승인 (다음 단계 진행)")
        print("  4. abort         → 중단")
        choice = input("입력 (1-4 또는 직접 action 입력): ").strip().lower()

        # 숫자 입력 지원
        mapping = {"1": "widen_search", "2": "minor_tune", "3": "approve", "4": "abort"}
        action = mapping.get(choice, choice)
        if action not in ("widen_search", "minor_tune", "approve", "abort"):
            print("[human_review] 잘못된 입력 → 기본값 'abort'로 설정")
            action = "abort"

    else:
        # PASS는 자동 승인
        action = "approve"
        print("[human_review] PASS → 자동 승인 (approve)")

    # --- TrainState에 저장 ---
    state.action = action
    ctx = state.context or {}
    ctx["human_review"] = {
        "result": result,
        "reason": reason,
        "prev": prev,
        "current": curr,
        "primary_metric": primary_metric,
        "action": action,
    }
    state.context = ctx

    print(f"[human_review] 결정된 action = {action}")
    return state
