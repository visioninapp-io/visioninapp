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
        
        # If it's already a dict, return it directly
        if isinstance(metrics_obj, dict):
            return metrics_obj
        
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
                    print(f"[CALLBACK] on_fit_epoch_end triggered!")
                    
                    # Debug: List all trainer attributes
                    trainer_attrs = [attr for attr in dir(trainer) if not attr.startswith('_')]
                    print(f"[CALLBACK] Trainer attributes: {trainer_attrs[:20]}...")  # First 20 to avoid spam
                    
                    epoch = getattr(trainer, "epoch", None)
                    epochs = getattr(trainer, "epochs", None)
                    
                    # Extract metrics directly from trainer
                    metrics = {}
                    
                    # Debug: Check what loss-related attributes exist
                    print(f"[CALLBACK] trainer.tloss = {getattr(trainer, 'tloss', 'NOT FOUND')}")
                    print(f"[CALLBACK] trainer.loss = {getattr(trainer, 'loss', 'NOT FOUND')}")
                    print(f"[CALLBACK] trainer.loss_items = {getattr(trainer, 'loss_items', 'NOT FOUND')}")
                    
                    # Loss from trainer.tloss (total loss tensor)
                    tloss = getattr(trainer, "tloss", None)
                    if tloss is not None:
                        try:
                            # tloss is a tensor, convert to float
                            if hasattr(tloss, 'item'):
                                metrics['train_loss'] = float(tloss.item())
                                print(f"[CALLBACK] Extracted train_loss from tloss.item(): {metrics['train_loss']}")
                            elif hasattr(tloss, 'sum'):
                                metrics['train_loss'] = float(tloss.sum().item())
                                print(f"[CALLBACK] Extracted train_loss from tloss.sum(): {metrics['train_loss']}")
                            else:
                                metrics['train_loss'] = float(tloss)
                                print(f"[CALLBACK] Extracted train_loss from tloss: {metrics['train_loss']}")
                        except Exception as e:
                            print(f"[CALLBACK] Failed to extract tloss: {e}")
                    
                    # Try trainer.loss as alternative
                    loss = getattr(trainer, "loss", None)
                    if loss is not None and 'train_loss' not in metrics:
                        try:
                            if hasattr(loss, 'item'):
                                metrics['train_loss'] = float(loss.item())
                                print(f"[CALLBACK] Extracted train_loss from loss.item(): {metrics['train_loss']}")
                        except Exception as e:
                            print(f"[CALLBACK] Failed to extract loss: {e}")
                    
                    # Get loss components from trainer.loss_items
                    loss_items = getattr(trainer, "loss_items", None)
                    if loss_items is not None:
                        try:
                            print(f"[CALLBACK] loss_items type: {type(loss_items)}, len: {len(loss_items) if hasattr(loss_items, '__len__') else 'N/A'}")
                            # loss_items is typically a tensor with [box_loss, cls_loss, dfl_loss]
                            if hasattr(loss_items, '__iter__') and len(loss_items) >= 3:
                                metrics['box_loss'] = float(loss_items[0])
                                metrics['cls_loss'] = float(loss_items[1])
                                metrics['dfl_loss'] = float(loss_items[2])
                                print(f"[CALLBACK] Extracted loss components: box={metrics['box_loss']}, cls={metrics['cls_loss']}, dfl={metrics['dfl_loss']}")
                        except Exception as e:
                            print(f"[CALLBACK] Failed to extract loss_items: {e}")
                    
                    # Get validation metrics from trainer.metrics
                    metrics_obj = getattr(trainer, "metrics", None)
                    print(f"[CALLBACK] trainer.metrics = {metrics_obj}")
                    if metrics_obj is not None:
                        val_metrics = self._metrics_to_dict(metrics_obj)
                        print(f"[CALLBACK] Extracted val_metrics: {val_metrics}")
                        metrics.update(val_metrics)
                    
                    # Get mAP from validator results if available
                    validator = getattr(trainer, "validator", None)
                    print(f"[CALLBACK] trainer.validator = {validator}")
                    if validator is not None:
                        results = getattr(validator, "results", None)
                        print(f"[CALLBACK] validator.results = {results}")
                        if results is not None:
                            # results is typically a dict or object with mAP values
                            if hasattr(results, 'results_dict'):
                                metrics.update(results.results_dict)
                                print(f"[CALLBACK] Extracted from results.results_dict")
                            elif isinstance(results, dict):
                                metrics.update(results)
                                print(f"[CALLBACK] Extracted from results dict")
                    
                    self.last_epoch = int(epoch) + 1 if epoch is not None else None  # +1 for 1-based indexing
                    self.total_epochs = int(epochs) if epochs is not None else None
                    self.last_metrics = metrics
                    
                    print(f"[CALLBACK] ===== EPOCH {self.last_epoch}/{self.total_epochs} =====")
                    print(f"[CALLBACK] Available metrics: {list(metrics.keys())}")
                    print(f"[CALLBACK] Metrics values: {metrics}")
                    print(f"[CALLBACK] Loss: {metrics.get('train_loss', 'N/A')}, mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
                    
                    try:
                        progress_cb({
                            "event": "epoch_end",
                            "epoch": self.last_epoch,
                            "total_epochs": self.total_epochs,
                            "metrics": metrics,
                        })
                        print(f"[CALLBACK] Progress callback executed successfully")
                    except Exception as e:
                        print(f"[CALLBACK] Progress callback failed: {e}")
                        import traceback
                        traceback.print_exc()
                except Exception as e:
                    import traceback
                    print(f"[CALLBACK] on_fit_epoch_end error: {e}")
                    print(traceback.format_exc())

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
        
        # Register callbacks using the new Ultralytics API (add_callback method)
        # NOTE: In newer Ultralytics, callbacks are NOT passed as train() parameter
        if callbacks:
            print(f"[YOLO] Registering {len(callbacks)} callbacks: {list(callbacks.keys())}")
            for event_name, callback_fn in callbacks.items():
                try:
                    self._model.add_callback(event_name, callback_fn)
                    print(f"[YOLO] ✅ Registered callback: {event_name}")
                except Exception as e:
                    # Log warning but continue - callback registration is optional
                    print(f"[YOLO] ⚠️ Could not register callback '{event_name}': {e}")
        else:
            print(f"[YOLO] ⚠️ No callbacks to register!")

        # Delegate to ultralytics (without callbacks in args)
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


