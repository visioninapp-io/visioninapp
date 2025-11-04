# run_init_context.py
from pathlib import Path
import json

from langgraph.graph import StateGraph, START, END

# 1) 정확한 경로로 클래스 임포트
from graph.training.state import TrainState
from graph.training.nodes.registry import NODE_REGISTRY
from utils.router import route_by_selfrag, route_mode, route_hpo_or_single, route_gate, route_interrupt_action, route_onnx_or_tensor

from dotenv import load_dotenv
load_dotenv()

# 2) 내부 변수로 값 주입 (CLI 없이)
CONFIG_PATH  = "configs/training.yaml"  # 없으면 None로 두세요 (프로젝트 루트의 training.yaml 탐색)
RUN_NAME     = "devtest"

def builder(user_query: str, dataset_path: str):
    state = TrainState()
    if CONFIG_PATH and Path(CONFIG_PATH).exists():
        state.config_path = CONFIG_PATH
    elif Path("training.yaml").exists():
        state.config_path = "training.yaml"

    state.run_name   = RUN_NAME

    graph = StateGraph(TrainState)
    graph.add_node("init_context", NODE_REGISTRY["init_context"])
    graph.add_node("query_analyzer", NODE_REGISTRY["query_analyzer"])
    graph.add_node("param_synthesizer", NODE_REGISTRY["param_synthesizer"])
    # graph.add_node("selfrag_scorer", NODE_REGISTRY["selfrag_scorer"])
    # graph.add_node("selfrag_decider", NODE_REGISTRY["selfrag_decider"])
    graph.add_node("load_dataset", NODE_REGISTRY["load_dataset"])
    graph.add_node("decide_mode", NODE_REGISTRY["decide_mode"])
    graph.add_node("load_model", NODE_REGISTRY["load_model"])
    graph.add_node("search_space_builder", NODE_REGISTRY["search_space_builder"])
    graph.add_node("hpo_decider", NODE_REGISTRY["hpo_decider"])
    graph.add_node("hpo_scheduler", NODE_REGISTRY["hpo_scheduler"])
    graph.add_node("select_best", NODE_REGISTRY["select_best"])
    graph.add_node("train_trial", NODE_REGISTRY["train_trial"])
    graph.add_node("evaluate_trial", NODE_REGISTRY["evaluate_trial"])
    graph.add_node("regression_gate", NODE_REGISTRY["regression_gate"])
    graph.add_node("human_review", NODE_REGISTRY["human_review"])
    graph.add_node("registry_publish", NODE_REGISTRY["registry_publish"])
    graph.add_node("onnx_converter", NODE_REGISTRY["onnx_converter"])
    graph.add_node("tensor_converter", NODE_REGISTRY["tensor_converter"])
    graph.add_node("evaluate_convert_model", NODE_REGISTRY["evaluate_convert_model"])

    graph.add_edge(START, "init_context")

    graph.add_edge("init_context", "query_analyzer")
    graph.add_edge("query_analyzer", "param_synthesizer")
    graph.add_edge("param_synthesizer", "load_dataset")
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