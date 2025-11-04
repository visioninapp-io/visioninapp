from pathlib import Path
from ultralytics import YOLO

def train_yolo(data_dir: str, out_dir: str, hyper: dict) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_name = hyper.get("model", "yolov8n.pt")
    model = YOLO(model_name)
    r = model.train(
        data=str(Path(data_dir, "data.yaml")),
        epochs=hyper.get("epochs", 100),
        imgsz=hyper.get("imgsz", 640),
        batch=hyper.get("batch", 16),
        project=out_dir, name="train", exist_ok=True
    )
    # 간단 메트릭
    try:
        return {"map50": float(getattr(r, "metrics", {}).get("map50", 0.0))}
    except Exception:
        return {}
