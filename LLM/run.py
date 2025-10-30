# run_init_context.py
from pathlib import Path
import json
from typing import Literal, Any, Dict, Optional, Tuple
from langgraph.graph import StateGraph, START, END

# 1) 정확한 경로로 클래스 임포트
from graph.training.state import TrainState
from graph.training.nodes.init_context import init_context
from graph.training.nodes.query_analyzer import query_analyzer
from graph.training.nodes.param_synthesizer import param_synthesizer
from graph.training.nodes.selfrag_scorer import selfrag_scorer
from graph.training.nodes.selfrag_decider import selfrag_decider
from graph.training.nodes.load_dataset import load_dataset
from graph.training.nodes.decide_mode import decide_mode
from graph.training.nodes.load_model import load_model
from graph.training.nodes.search_space_builder import search_space_builder
from graph.training.nodes.hpo_decider import hpo_decider
from graph.training.nodes.hpo_scheduler import hpo_scheduler
from graph.training.nodes.train_trial import train_trial
from graph.training.nodes.select_best import select_best
from graph.training.nodes.evaluate_trial import evaluate_trial
from graph.training.nodes.regression_gate import regression_gate
from graph.training.nodes.human_review import human_review
from graph.training.nodes.registry_publish import registry_publish
from graph.training.nodes.onnx_converter import onnx_converter
from graph.training.nodes.tensor_converter import tensor_converter
from graph.training.nodes.evaluate_convert_model import evaluate_convert_model

from dotenv import load_dotenv
load_dotenv()

# 2) 내부 변수로 값 주입 (CLI 없이)
CONFIG_PATH  = "configs/training.yaml"  # 없으면 None로 두세요 (프로젝트 루트의 training.yaml 탐색)
RUN_NAME     = "devtest"

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

def main(user_query: str, dataset_path: str):
    state = TrainState()
    if CONFIG_PATH and Path(CONFIG_PATH).exists():
        state.config_path = CONFIG_PATH
    elif Path("training.yaml").exists():
        state.config_path = "training.yaml"

    state.run_name   = RUN_NAME

    graph = StateGraph(TrainState)
    graph.add_node("init_context", init_context)
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("param_synthesizer", param_synthesizer)
    graph.add_node("selfrag_scorer", selfrag_scorer)
    # graph.add_node("selfrag_decider", selfrag_decider)
    graph.add_node("load_dataset", load_dataset)
    graph.add_node("decide_mode", decide_mode)
    graph.add_node("load_model", load_model)
    graph.add_node("search_space_builder", search_space_builder)
    graph.add_node("hpo_decider", hpo_decider)
    graph.add_node("hpo_scheduler", hpo_scheduler)
    graph.add_node("select_best", select_best)
    graph.add_node("train_trial", train_trial)
    graph.add_node("evaluate_trial", evaluate_trial)
    graph.add_node("regression_gate", regression_gate)
    graph.add_node("human_review", human_review)
    graph.add_node("registry_publish", registry_publish)
    graph.add_node("onnx_converter", onnx_converter)
    graph.add_node("tensor_converter", tensor_converter)
    graph.add_node("evaluate_convert_model", evaluate_convert_model)

    graph.add_edge(START, "init_context")

    graph.add_edge("init_context", "query_analyzer")
    graph.add_edge("query_analyzer", "param_synthesizer")
    graph.add_edge("param_synthesizer", "selfrag_scorer")
    graph.add_edge("selfrag_scorer", "load_dataset")
    # graph.add_conditional_edges(
    #     "selfrag_decider",
    #     route_by_selfrag,
    #     {
    #         "pass": "load_dataset",
    #         "feedback": "query_analyzer",
    #         "fallback": END,
    #     }
    # )
    # graph.add_edge("selfrag_decider", "load_dataset")
    graph.add_edge("load_dataset", "decide_mode")
    graph.add_conditional_edges(
        "decide_mode",
        route_mode,
        {
            "path_fresh": "search_space_builder",
            "path_finetune": "load_model",
            "path_resume": "load_model",
            "error_missing_dataset": END,
        }
    )
    graph.add_edge("load_model", "search_space_builder")
    graph.add_edge("search_space_builder", "hpo_decider")
    graph.add_conditional_edges(
        "hpo_decider",
        route_hpo_or_single,
        {
            "single_trial": "select_best",
            "use_hpo": "hpo_scheduler",
        },
    )

    graph.add_edge("hpo_scheduler", "select_best")
    graph.add_edge("select_best", "train_trial")
    graph.add_edge("train_trial", "evaluate_trial")
    graph.add_edge("evaluate_trial", "regression_gate")
    graph.add_conditional_edges(
        "regression_gate",
        route_gate,
        {
            "publish": "registry_publish",
            "human_review": "human_review",
        }
    )
    # graph.add_edge("regression_gate", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_interrupt_action,
        {
            "approve": "registry_publish",
            "minor_tune": "train_trial",
            "widen_search": "search_space_builder",
            "abort": END,
        }
    )

    graph.add_conditional_edges(
        "registry_publish",
        route_onnx_or_tensor,
        {
            "onnx": "onnx_converter",
            "tensor": "tensor_converter",
            "pure": END
        }
    )

    graph.add_edge("onnx_converter", "evaluate_convert_model")
    graph.add_edge("tensor_converter", "evaluate_convert_model")
    graph.add_edge("evaluate_convert_model", END)

    train_graph = graph.compile()

    final_state = train_graph.invoke({
        "user_query": user_query,
        "dataset_version": dataset_path
    })
    # print(final_state.get("context"))


if __name__ == "__main__":
    user_query = input("어떤 방식으로 모델을 학습할까요?\n")
    dataset_path = "dataset@1.0.0"
    # dataset_path = "rock-paper-scissors"
    main(user_query, dataset_path)
