import importlib
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Type, Union

from .integrations.yolo import build_yolo_model


ModelInput = Union[str, Type[Any], Any, Callable[[], Any]]


def _resolve_class_from_string(qualified_name: str) -> Type[Any]:
    """
    Resolve a fully-qualified class name like
    "sklearn.linear_model.LogisticRegression" to the class object.
    """
    if "." not in qualified_name:
        raise ValueError(
            "String model identifiers must be fully-qualified, e.g. "
            '"sklearn.linear_model.LogisticRegression"'
        )

    module_path, class_name = qualified_name.rsplit(".", 1)
    module: ModuleType = importlib.import_module(module_path)
    cls: Type[Any] = getattr(module, class_name)
    return cls


def _apply_hyperparameters(instance: Any, hyperparameters: Dict[str, Any]) -> Any:
    """
    Apply hyperparameters to a model instance using the most appropriate method.
    Prefers scikit-learn's set_params if available; falls back to setattr.
    """
    if not hyperparameters:
        return instance

    if hasattr(instance, "set_params") and callable(getattr(instance, "set_params")):
        instance.set_params(**hyperparameters)
        return instance

    for name, value in hyperparameters.items():
        setattr(instance, name, value)
    return instance


def _is_yolo_identifier(text: str) -> bool:
    lower = text.lower()
    return lower.startswith("yolov") or lower.endswith(".pt")


def build_model(
    model: ModelInput,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Build a model given a variety of input forms and apply hyperparameters.

    Accepts:
    - A fully-qualified string path to a class (e.g.,
      "sklearn.linear_model.LogisticRegression").
    - A class/type (it will be instantiated).
    - A pre-instantiated object (it will be returned after applying params).
    - A zero-arg callable that returns a model instance.
    """
    # Zero-arg callable factory
    if callable(model) and not isinstance(model, type):
        try:
            instance = model()
        except TypeError:
            # Callable may not be a zero-arg factory; treat it as an instance
            instance = model
        return _apply_hyperparameters(instance, hyperparameters or {})

    # Fully-qualified class name
    if isinstance(model, str):
        # YOLO selector: 'yolov8n' or local/remote .pt checkpoint
        if _is_yolo_identifier(model):
            return build_yolo_model(model, hyperparameters)
        cls = _resolve_class_from_string(model)
        instance = cls(**(hyperparameters or {}))
        return instance

    # Class/type
    if isinstance(model, type):
        instance = model(**(hyperparameters or {}))
        return instance

    # Assume already-instantiated object
    instance = model
    return _apply_hyperparameters(instance, hyperparameters or {})


