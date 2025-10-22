from typing import Any, Dict, Optional, Tuple

from .factory import build_model
from .trainer import train_model


def train_from_payload(payload: Dict[str, Any]) -> Tuple[Any, Optional[float]]:
    """
    Minimal payload-based training entrypoint.

    Expected keys in payload:
    - model: string | class | instance | zero-arg callable | dict-spec with framework
    - hyperparameters / params: dict (applied at build time)
    - fit_params / train: dict (passed to fit/train at training time)
    - X_train, y_train, X_val, y_val: optional (framework-dependent)
    """
    model_spec = payload.get("model")
    hyperparameters: Dict[str, Any] = payload.get("hyperparameters") or payload.get("params") or {}
    fit_params: Dict[str, Any] = payload.get("fit_params") or payload.get("train") or {}

    X_train = payload.get("X_train")
    y_train = payload.get("y_train")
    X_val = payload.get("X_val")
    y_val = payload.get("y_val")

    model = build_model(model_spec, hyperparameters)
    trained, score = train_model(
        model,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        fit_params=fit_params,
    )
    return trained, score


