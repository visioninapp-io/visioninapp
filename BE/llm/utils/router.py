from llm.graph.training.state import TrainState
from typing import Literal, Any, Dict, Optional, Tuple

def route_by_selfrag(state):
    val = (state.context or {}).get("selfrag_decider", {}).get("decision", "feedback")
    return str(val).strip().lower()


def route_mode(state: TrainState) -> Literal["path_fresh", "path_finetune", "path_resume", "error_missing_dataset"]:
    cfg = state.train_cfg or {}
    data_cfg = (cfg.get("data") or {})
    has_ds = bool(state.dataset_version) or bool(data_cfg.get("yaml_path")) or bool((state.data or {}).get("yaml_path"))
    has_base = bool(getattr(state, "base_model", None)) or bool((cfg.get("resume") or {}).get("finetune_base"))
    resume = bool(state.resume) or bool((cfg.get("resume") or {}).get("enable")) or bool((cfg.get("resume") or {}).get("last_ckpt"))

    if not has_ds:
        print('[route_mode] move -> error_missing_dataset')
        return "error_missing_dataset"
    if resume:
        print('[route_mode] move -> path_resume')
        return "path_resume"
    if has_base:
        print('[route_mode] move -> path_finetune')
        return "path_finetune"
    
    print('[route_mode] move -> path_fresh')
    return "path_fresh"


def route_hpo_or_single(state: TrainState) -> Literal["singel_trial", "use_hpo"]:
    """
    - hpo_decider 노드가 먼저 실행되어 state.force_hpo / context.hpo_decider를 채웠다고 가정.
    - 우선순위:
        1) 강제 플래그(state.force_hpo)
        2) 설정 기반(state.hpo.enabled, search_space, max_trials)
        3) 폴백(single_trial)
    """

    force = getattr(state, "force_hpo", None)
    if force is True:
        print("[route_hpo_or_single] forced → use_hpo")
        return "use_hpo"
    if force is False:
        print("[route_hpo_or_single] forced → single_trial")
        return "single_trial"
    
    def _truthy(x: Any) -> bool:
        return bool(x) and x is not False and x is not None

    # 1) 설정 기반 판단
    hpo: Dict[str, Any] = (state.hpo or {})
    enabled     = _truthy(hpo.get("enabled"))
    space       = hpo.get("search_space") or {}
    max_trials  = int(hpo.get("max_trials", 0) or 0)

    # HPO를 수행하려면: 켜져 있고, 탐색공간이 존재하며, 시도 횟수 2회 이상
    if enabled and isinstance(space, dict) and len(space) > 0 and max_trials >= 2:
        print("[route_hpo_or_single] config → use_hpo")
        return "use_hpo"

    # 2) 폴백: 단일 학습
    print("[route_hpo_or_single] config → single_trial")
    return "single_trial"

def route_gate(state: TrainState) -> Literal["publish", "human_review"]:
    result = (getattr(state, "get_result", None) or "").upper()
    if result=="PASS":
        return "publish"
    return "human_review"

def route_interrupt_action(state: TrainState) -> Literal["widen_search", "minor_tune", "approve", "abort"]:
    """
    human_review 이후 사람의 action 값에 따라 다음 노드를 결정하는 라우터.
    """
    action = (getattr(state, "action", None) or "").strip().lower()

    if action in ("widen_search", "minor_tune", "approve", "abort"):
        print(f"[route_interrupt_action] action={action}")
        return action

    # 안전 폴백
    print(f"[route_interrupt_action] 알 수 없는 action='{action}' → abort로 폴백")
    return "abort"

def route_onnx_or_tensor(state: TrainState) -> Literal["pure", "onnx", "tensor"]:
    ctx = state.context or {}
    qa = ctx.get("query_analyzer", {})
    parsed = qa.get("parsed", {})

    onnx = bool(parsed.get("onnx", False))
    tensor = bool(parsed.get("tensorrt", False))

    if tensor==True:
        return "tensor"
    if onnx==True:
        return "onnx"
    
    return "pure"

def route_after_train(state: TrainState) -> str:
    """
    train_trial 이후 분기:
    - success: 다음 스텝 진행
    - failed: 에러 처리 플로우
    - timeout: 타임아웃 처리 플로우
    """
    action = (state.action or "").upper()

    if action == "TRAIN_COMPLETED":
        return "success"
    if action == "TRAIN_TIMEOUT":
        return "timeout"
    # 그 외는 일단 실패로 몰기 (TRAIN_FAILED 포함)
    return "failed"