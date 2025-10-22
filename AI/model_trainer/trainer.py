from typing import Any, Callable, Dict, Optional, Tuple


def _has_method(obj: Any, method_name: str) -> bool:
    method = getattr(obj, method_name, None)
    return callable(method)


def train_model(
    model: Any,
    X_train: Any,
    y_train: Any = None,
    *,
    X_val: Any = None,
    y_val: Any = None,
    fit_params: Optional[Dict[str, Any]] = None,
    metric: Optional[Callable[[Any, Any], float]] = None,
) -> Tuple[Any, Optional[float]]:
    """
    Train a model with provided data. Works with scikit-learn estimators and
    any object exposing a compatible "fit" method. Optionally evaluates on a
    validation set using either a provided metric(y_true, y_pred) or the
    model's inherent "score" method when available.

    Returns (model, score_or_none).
    """
    # Two training paths:
    # 1) sklearn-like: model.fit(X[, y], **fit_params)
    # 2) adapter-like (e.g., YOLO): model.train(**fit_params) when no 'fit'
    if not _has_method(model, "fit"):
        if _has_method(model, "train"):
            train_kwargs = fit_params or {}
            model.train(**train_kwargs)
            return model, None
        raise TypeError("Provided model does not have a callable 'fit' or 'train' method.")

    fit_kwargs = fit_params or {}
    # Handle unsupervised / no target scenarios
    if y_train is None:
        model.fit(X_train, **fit_kwargs)
    else:
        model.fit(X_train, y_train, **fit_kwargs)

    score: Optional[float] = None
    if X_val is not None and y_val is not None:
        # Prefer provided metric; fall back to model.score
        if metric is not None:
            if _has_method(model, "predict"):
                y_pred = model.predict(X_val)
                score = float(metric(y_val, y_pred))
            else:
                raise TypeError(
                    "A custom metric was provided but the model lacks 'predict'."
                )
        elif _has_method(model, "score"):
            score = float(model.score(X_val, y_val))

    return model, score


def evaluate_model(
    model: Any,
    X: Any,
    y: Any,
    metric: Optional[Callable[[Any, Any], float]] = None,
) -> float:
    """
    Evaluate a trained model on provided data.
    Prefers custom metric; falls back to model.score when available.
    """
    if metric is not None:
        if not _has_method(model, "predict"):
            raise TypeError("Model lacks 'predict' required for the provided metric.")
        y_pred = model.predict(X)
        return float(metric(y, y_pred))

    if _has_method(model, "score"):
        return float(model.score(X, y))

    raise TypeError("No metric provided and model does not implement 'score'.")


