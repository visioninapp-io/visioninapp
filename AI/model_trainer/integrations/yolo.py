from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import threading
import time


def _import_ultralytics_yolo():
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise ImportError(
            "ultralytics is required for YOLO integration. Install with 'pip install ultralytics'."
        ) from exc
    return YOLO


def _normalize_model_spec(spec: Union[str, Path]) -> str:
    s = str(spec)
    lower = s.lower()
    # Allow shorthand like 'yolov8n' by appending .pt
    if lower.startswith("yolov") and not lower.endswith(".pt"):
        return s + ".pt"
    return s


class YOLOAdapter:
    """
    Minimal adapter to expose a fit-like API over ultralytics.YOLO.

    Use fit(X_train=None, y_train=None, **fit_params) and pass ultralytics
    training arguments via fit_params (e.g., data, epochs, imgsz, batch).
    """

    def __init__(self, model_spec: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> None:
        YOLO = _import_ultralytics_yolo()
        self._model = YOLO(_normalize_model_spec(model_spec))
        self._overrides: Dict[str, Any] = dict(overrides or {})
        self.last_epoch: Optional[int] = None
        self.total_epochs: Optional[int] = None
        self.last_metrics: Optional[Dict[str, Any]] = None
        self.final_metrics: Optional[Dict[str, Any]] = None
        self.save_dir: Optional[str] = None
        self.last_batch: Optional[int] = None
        self.batches_per_epoch: Optional[int] = None
        # ticker state
        self._tick_thread: Optional[threading.Thread] = None
        self._tick_stop: Optional[threading.Event] = None
        self._tick_interval: float = 1.0

    def _metrics_to_dict(self, metrics_obj: Any) -> Dict[str, Any]:
        if metrics_obj is None:
            return {}
        # Common Ultralytics metrics exposure
        for attr in ("results_dict", "as_dict", "to_dict"):
            v = getattr(metrics_obj, attr, None)
            if callable(v):
                try:
                    d = v()
                    if isinstance(d, dict):
                        return d
                except Exception:
                    pass
        # Fallback: collect public numeric fields
        out: Dict[str, Any] = {}
        for name in dir(metrics_obj):
            if name.startswith("_"):
                continue
            try:
                value = getattr(metrics_obj, name)
            except Exception:
                continue
            if isinstance(value, (int, float)):
                out[name] = float(value)
        return out

    def set_params(self, **params: Any) -> "YOLOAdapter":
        self._overrides.update(params)
        return self

    def fit(self, X_train: Any = None, y_train: Any = None, **fit_params: Any) -> Any:
        args: Dict[str, Any] = {**self._overrides, **(fit_params or {})}
        if "data" not in args:
            raise ValueError("YOLO training requires 'data' to point to a dataset YAML file.")

        # Inject progress callbacks if provided
        progress_cb = args.pop("progress_callback", None)
        tick_interval = float(args.pop("tick_interval", 1.0))
        user_callbacks = args.get("callbacks")
        callbacks: Dict[str, Any] = {}

        if progress_cb is not None:
            # start per-second ticker
            try:
                self._tick_interval = max(0.1, float(tick_interval))
            except Exception:
                self._tick_interval = 1.0
            self._tick_stop = threading.Event()

            def _ticker():
                while self._tick_stop is not None and not self._tick_stop.is_set():
                    status = {
                        "event": "tick",
                        "epoch": self.last_epoch,
                        "total_epochs": self.total_epochs,
                        "batch": self.last_batch,
                        "batches_per_epoch": self.batches_per_epoch,
                        "metrics": self.last_metrics,
                        "save_dir": self.save_dir,
                        "ts": time.time(),
                    }
                    try:
                        progress_cb(status)
                    except Exception:
                        pass
                    time.sleep(self._tick_interval)

            self._tick_thread = threading.Thread(target=_ticker, daemon=True)
            self._tick_thread.start()

            def on_fit_epoch_end(trainer):  # type: ignore
                try:
                    epoch = getattr(trainer, "epoch", None)
                    epochs = getattr(trainer, "epochs", None)
                    metrics_obj = getattr(trainer, "metrics", None)
                    metrics = self._metrics_to_dict(metrics_obj)
                    self.last_epoch = int(epoch) if epoch is not None else None
                    self.total_epochs = int(epochs) if epochs is not None else None
                    self.last_metrics = metrics
                    try:
                        progress_cb({
                            "event": "epoch_end",
                            "epoch": self.last_epoch,
                            "total_epochs": self.total_epochs,
                            "metrics": metrics,
                        })
                    except Exception:
                        pass
                except Exception:
                    pass

            def on_train_batch_end(trainer):  # type: ignore
                try:
                    batch_i = getattr(trainer, "batch_i", None)
                    self.last_batch = int(batch_i) if batch_i is not None else self.last_batch
                    dl = getattr(trainer, "train_loader", None)
                    try:
                        self.batches_per_epoch = len(dl) if dl is not None else self.batches_per_epoch
                    except Exception:
                        pass
                except Exception:
                    pass

            def on_train_end(trainer):  # type: ignore
                try:
                    metrics_obj = getattr(trainer, "metrics", None)
                    self.final_metrics = self._metrics_to_dict(metrics_obj)
                    save_dir = getattr(trainer, "save_dir", None)
                    if save_dir is not None:
                        self.save_dir = str(save_dir)
                    try:
                        progress_cb({
                            "event": "train_end",
                            "metrics": self.final_metrics,
                            "save_dir": self.save_dir,
                        })
                    except Exception:
                        pass
                except Exception:
                    pass
                # stop ticker
                try:
                    if self._tick_stop is not None:
                        self._tick_stop.set()
                    if self._tick_thread is not None and self._tick_thread.is_alive():
                        self._tick_thread.join(timeout=1.0)
                except Exception:
                    pass

            callbacks.update({
                "on_fit_epoch_end": on_fit_epoch_end,
                "on_train_batch_end": on_train_batch_end,
                "on_train_end": on_train_end,
            })

        if isinstance(user_callbacks, dict):
            callbacks.update(user_callbacks)
        if callbacks:
            args["callbacks"] = callbacks

        # Delegate to ultralytics
        result = self._model.train(**args)
        # Capture final metrics and save_dir if available post-train
        try:
            trainer = getattr(self._model, "trainer", None)
            if trainer is not None:
                metrics_obj = getattr(trainer, "metrics", None)
                self.final_metrics = self._metrics_to_dict(metrics_obj)
                save_dir = getattr(trainer, "save_dir", None)
                if save_dir is not None:
                    self.save_dir = str(save_dir)
        except Exception:
            pass
        return result

    # Convenience passthroughs
    def predict(self, source: Any, **kwargs: Any) -> Any:  # pragma: no cover - passthrough
        return self._model.predict(source=source, **kwargs)

    def score(self, *args: Any, **kwargs: Any) -> float:  # pragma: no cover - optional
        # YOLO doesn't expose a single scalar score in sklearn style; users should call .val
        raise NotImplementedError("YOLOAdapter.score is not implemented; use model._model.val instead.")


def build_yolo_model(model_spec: Union[str, Path], hyperparameters: Optional[Dict[str, Any]] = None) -> YOLOAdapter:
    return YOLOAdapter(model_spec, overrides=hyperparameters)


