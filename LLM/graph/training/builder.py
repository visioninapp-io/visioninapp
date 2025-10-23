# graph/training/builder.py
"""
TrainingGraph builder

- init_context → load_dataset → decide_mode
- fresh        : (no base model) → (optional HPO) → train/eval → select_best → regression_gate
- finetune/resume: load_model → (optional HPO) → train/eval → select_best → regression_gate
- regression_gate:
    PASS  → registry_publish → export → smoke_infer → END
    FAIL  → __interrupt__ (action: widen_search | minor_tune | abort)
             widen_search → search_space_builder → (HPO or single trial) …
             minor_tune   → train_trial → evaluate_trial → select_best → regression_gate
             abort        → END
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END

# 상태와 노드 함수들은 각 모듈에서 가져옵니다.
from .state import TrainState
from .nodes.init_context import init_context
from .nodes.load_dataset import load_dataset
from .nodes.decide_mode import decide_mode   # 선택: mode 계산을 별도 노드로 분리했다면
from .nodes.load_model import load_model
from .nodes.search_space_builder import search_space_builder
from .nodes.hpo_scheduler import hpo_scheduler
from .nodes.train_trial import train_trial
from .nodes.evaluate_trial import evaluate_trial
from .nodes.select_best import select_best
from .nodes.regression_gate import regression_gate
from .nodes.registry_publish import registry_publish
from .nodes.export import export
from .nodes.smoke_infer import smoke_infer
from .nodes.human_review import human_review  # __interrupt__ 처리(상태에 action 설정)

# -------------------------
# 라우터(조건부 분기) 헬퍼들
# -------------------------

def _route_mode(state: TrainState) -> Literal["path_fresh", "path_finetune", "path_resume", "error_missing_dataset"]:
    """dataset/base_model/resume 플래그로 학습 모드를 결정"""
    has_ds = bool(getattr(state, "dataset_version", None))
    has_base = bool(getattr(state, "base_model", None))
    resume = bool(getattr(state, "resume", False) or getattr(state, "last_ckpt", None))

    if not has_ds:
        return "error_missing_dataset"
    if resume:
        return "path_resume"
    if has_base:
        return "path_finetune"
    return "path_fresh"


def _route_hpo_or_single(state: TrainState) -> Literal["use_hpo", "single_trial"]:
    """hpo.enabled 여부에 따라 분기"""
    hpo_cfg = getattr(state, "hpo", None) or {}
    enabled = bool(hpo_cfg.get("enabled", False))
    return "use_hpo" if enabled else "single_trial"


def _route_gate(state: TrainState) -> Literal["publish", "__interrupt__"]:
    """회귀 게이트 결과 분기"""
    gate_pass = bool(getattr(state, "gate_pass", False))
    return "publish" if gate_pass else "__interrupt__"


def _route_interrupt_action(state: TrainState) -> Literal["widen_search", "minor_tune", "abort"]:
    """
    human_review가 설정한 action에 따라 라우팅
    - "widen_search": 탐색 공간/설정 확장 후 HPO 또는 단일 트라이얼로 복귀
    - "minor_tune"  : 현재 best 파라미터 기준으로 소폭 재학습만 수행
    - "abort"       : 종료
    """
    action = getattr(state, "action", None) or "abort"
    if action not in ("widen_search", "minor_tune", "abort"):
        return "abort"
    return action


def _nop_error_missing_dataset(state: TrainState):
    raise ValueError("dataset_version is required but not provided.")


# -------------------------
# 그래프 빌더
# -------------------------

def build_training_graph() -> StateGraph:
    """
    TrainingGraph를 구성하여 반환합니다.
    노드 구현은 graph/training/nodes/ 모듈들을 참고하세요.
    """
    g = StateGraph(TrainState)

    # 노드 등록
    g.add_node("init_context", init_context)
    g.add_node("load_dataset", load_dataset)

    # decide_mode 노드를 직접 쓰지 않고 라우터만 쓸 수도 있지만,
    # 디버깅/로그를 위해 빈 노드(혹은 경량 노드)를 둬도 좋습니다.
    g.add_node("decide_mode", lambda s: s)

    g.add_node("load_model", load_model)
    g.add_node("search_space_builder", search_space_builder)
    g.add_node("hpo_scheduler", hpo_scheduler)

    # single-trial 경로에서도 동일하게 사용
    g.add_node("train_trial", train_trial)
    g.add_node("evaluate_trial", evaluate_trial)
    g.add_node("select_best", select_best)

    g.add_node("regression_gate", regression_gate)
    g.add_node("registry_publish", registry_publish)
    g.add_node("export", export)
    g.add_node("smoke_infer", smoke_infer)

    # 게이트 실패 시 사용자/규칙 개입
    g.add_node("human_review", human_review)

    # 오류 처리용
    g.add_node("error_missing_dataset", _nop_error_missing_dataset)

    # 엣지 연결
    g.add_edge(START, "init_context")
    g.add_edge("init_context", "load_dataset")
    g.add_edge("load_dataset", "decide_mode")

    # 모드 라우팅: fresh / finetune / resume / error
    g.add_conditional_edges(
        "decide_mode",
        _route_mode,
        {
            "path_fresh": "search_space_builder",
            "path_finetune": "load_model",
            "path_resume": "load_model",
            "error_missing_dataset": "error_missing_dataset",
        },
    )

    # finetune/resume 공통: 모델 로드 후 탐색공간 구성
    g.add_edge("load_model", "search_space_builder")

    # HPO 사용할지 단일 트라이얼로 갈지
    g.add_conditional_edges(
        "search_space_builder",
        _route_hpo_or_single,
        {
            "use_hpo": "hpo_scheduler",
            "single_trial": "train_trial",
        },
    )

    # HPO 경로: 내부에서 여러 trial 실행 후 best 갱신 → select_best 호출을 마지막에 한 번 더 안전하게
    g.add_edge("hpo_scheduler", "select_best")

    # 단일 트라이얼 경로
    g.add_edge("train_trial", "evaluate_trial")
    g.add_edge("evaluate_trial", "select_best")

    # 베스트 모델 선정 후 회귀 게이트
    g.add_edge("select_best", "regression_gate")

    # 게이트 결과에 따라 분기
    g.add_conditional_edges(
        "regression_gate",
        _route_gate,
        {
            "publish": "registry_publish",
            "human_review": "human_review",
        },
    )

    # 인터럽트 결과에 따른 재시도 루프
    g.add_conditional_edges(
        "human_review",
        _route_interrupt_action,
        {
            "widen_search": "search_space_builder",  # 탐색 확장 → HPO/단일 분기 재사용
            "minor_tune": "train_trial",            # 소폭 수정 → train→eval→select_best→gate
            "approve": "registry_publish",
            "abort": END,
        },
    )

    # 발행 이후 내보내기 및 스모크 테스트
    g.add_edge("registry_publish", "export")
    g.add_edge("export", "smoke_infer")
    g.add_edge("smoke_infer", END)

    return g


def compile_training_graph():
    """
    컴파일된 그래프(ExecutableGraph)를 반환합니다.
    사용 예:
        graph = compile_training_graph()
        result = graph.invoke(initial_state)
    """
    g = build_training_graph()
    return g.compile()
