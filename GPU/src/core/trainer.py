from pathlib import Path
from ultralytics import YOLO

# 유틸: 문자열로 온 값들 안전 변환
def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)

def _as_float(v, default=None):
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

def _as_int(v, default=None):
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _norm_optimizer(v):
    # "null", "", None 등은 미지정 처리
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
        return None
    return str(v)

def train_yolo(data_dir: str, out_dir: str, hyper: dict, progress=None) -> dict:
    """
    hyper 예시(모두 선택적):
    {
      "model": "yolo12n.pt",
      "epochs": 1, "imgsz": 640, "batch": 32, "device": "cuda:0",
      "workers": 8, "optimizer": "null", "lr0": 0.01, "lrf": 0.01,
      "weight_decay": 0.0005, "momentum": 0.937, "patience": 30,
      "save": true, "augment": true, "mosaic": true, "mixup": false
    }
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model_name = hyper.get("model", "yolo12n.pt")
    model = YOLO(model_name)

    # 공통 기본값
    defaults = {
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "device": None,        # 미지정 시 Ultralytics 기본
        "workers": 8,
        "optimizer": None,     # "null"/None이면 전달 안 함
        "lr0": 0.01,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "momentum": 0.937,
        "patience": 30,
        "save": True,
        "augment": None,       # 미지정 시 라이브러리 기본
        "mosaic": None,
        "mixup": None,
    }

    # 입력값 정규화
    epochs       = _as_int(hyper.get("epochs"), defaults["epochs"])
    imgsz        = _as_int(hyper.get("imgsz"), defaults["imgsz"])
    batch        = _as_int(hyper.get("batch"), defaults["batch"])
    device       = hyper.get("device", defaults["device"])
    workers      = _as_int(hyper.get("workers"), defaults["workers"])
    optimizer    = _norm_optimizer(hyper.get("optimizer", defaults["optimizer"]))
    lr0          = _as_float(hyper.get("lr0"), defaults["lr0"])
    lrf          = _as_float(hyper.get("lrf"), defaults["lrf"])
    weight_decay = _as_float(hyper.get("weight_decay"), defaults["weight_decay"])
    momentum     = _as_float(hyper.get("momentum"), defaults["momentum"])
    patience     = _as_int(hyper.get("patience"), defaults["patience"])

    # bool 계열(문자열도 허용)
    save   = defaults["save"] if hyper.get("save")   is None else _as_bool(hyper.get("save"))
    augment= None if hyper.get("augment") is None else _as_bool(hyper.get("augment"))
    mosaic = None if hyper.get("mosaic")  is None else _as_bool(hyper.get("mosaic"))
    mixup  = None if hyper.get("mixup")   is None else _as_bool(hyper.get("mixup"))

    # .train에 넘길 인자 구성 (None은 제외해서 라이브러리 기본을 쓰게 함)
    train_kwargs = {
        "data": str(Path(data_dir, "data.yaml")),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": out_dir,
        "name": "train",
        "exist_ok": True,
        "workers": workers,
        "lr0": lr0,
        "lrf": lrf,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "patience": patience,
        "save": save,
    }
    if device is not None:
        train_kwargs["device"] = device
    if optimizer is not None:
        train_kwargs["optimizer"] = optimizer
    if augment is not None:
        train_kwargs["augment"] = augment
    if mosaic is not None:
        train_kwargs["mosaic"] = mosaic
    if mixup is not None:
        train_kwargs["mixup"] = mixup

    if progress is not None:
        def _on_fit_epoch_end(trainer):
            """
            Ultralytics Trainer 객체에서 epoch/metrics를 뽑아서 그대로 전송.
            """
            try:
                epoch = int(getattr(trainer, "epoch", 0))
            except Exception:
                epoch = 0

            # "모든 내용" 그대로 보내기
            metrics = {}

            # trainer.metrics가 dict이면 그대로 사용
            m = getattr(trainer, "metrics", None)
            if isinstance(m, dict):
                metrics = m
            elif m is not None:
                # dict가 아니면 문자열로라도 던져줌
                metrics = {"metrics": str(m)}

            # 추가로 자주 쓰는 값들 있으면 합쳐줌 (중복 키는 metrics 쪽이 우선)
            extra = {}
            for attr in ("tloss", "nloss", "loss", "lr", "ema_loss"):
                if hasattr(trainer, attr):
                    extra[attr] = getattr(trainer, attr)
            if extra:
                extra.update(metrics)
                metrics = extra

            try:
                progress.train_log(epoch=epoch, metrics=metrics)
            except Exception as e:
                # 로깅만 하고 학습은 계속
                print(f"[progress] train.log publish failed: {e}")

        # 콜백 등록 (Ultralytics v8/v12 공통 on_fit_epoch_end 훅)
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    
    r = model.train(**train_kwargs)

    # 간단 메트릭 반환
    try:
        return {"map50": float(getattr(r, "metrics", {}).get("map50", 0.0))}
    except Exception:
        return {}