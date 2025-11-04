from ultralytics import YOLO
from pathlib import Path
import shutil, os

def to_tensorrt(onnx_path: str, out_path: str, precision: str = "fp16"):
    # ultralytics export는 보통 디렉토리 기준으로 engine을 생성함
    work_dir = Path(out_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(onnx_path)
    res = model.export(format="engine", half=(precision.lower()=="fp16"), imgsz=640, dynamic=True)
    produced = str(res) if isinstance(res, (str, Path)) else None

    # 엔진 파일 찾기
    if produced and produced.endswith(".engine"):
        if produced != out_path:
            shutil.copyfile(produced, out_path)
    else:
        # fallback: common default path
        cand = work_dir / "best.engine"
        if cand.exists():
            shutil.copyfile(cand, out_path)
        else:
            raise FileNotFoundError("TensorRT engine not found after export")
