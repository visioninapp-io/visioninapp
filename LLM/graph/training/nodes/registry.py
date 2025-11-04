from .init_context import init_context
from .query_analyzer import query_analyzer
from .param_synthesizer import param_synthesizer
from .selfrag_scorer import selfrag_scorer
from .selfrag_decider import selfrag_decider
from .load_dataset import load_dataset
from .decide_mode import decide_mode
from .load_model import load_model
from .search_space_builder import search_space_builder
from .hpo_decider import hpo_decider
from .hpo_scheduler import hpo_scheduler
from .train_trial import train_trial
from .select_best import select_best
from .evaluate_trial import evaluate_trial
from .regression_gate import regression_gate
from .human_review import human_review
from .registry_publish import registry_publish
from .onnx_converter import onnx_converter
from .tensor_converter import tensor_converter
from .evaluate_convert_model import evaluate_convert_model

NODE_REGISTRY = {
    "init_context": init_context,
    "query_analyzer": query_analyzer,
    "param_synthesizer": param_synthesizer,
    "selfrag_scorer": selfrag_scorer,
    "selfrag_decider": selfrag_decider,
    "load_dataset": load_dataset,
    "decide_mode": decide_mode,
    "load_model": load_model,
    "search_space_builder": search_space_builder,
    "hpo_decider": hpo_decider,
    "hpo_scheduler": hpo_scheduler,
    "train_trial": train_trial,
    "select_best": select_best,
    "evaluate_trial": evaluate_trial,
    "regression_gate": regression_gate,
    "human_review": human_review,
    "registry_publish": registry_publish,
    "onnx_converter": onnx_converter,
    "tensor_converter": tensor_converter,
    "evaluate_convert_model": evaluate_convert_model,
}