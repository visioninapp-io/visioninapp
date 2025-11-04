from ultralytics import YOLO
from pathlib import Path
import shutil

def to_onnx(pt_path: str, out_path: str, opset: int = 13):
    model = YOLO(pt_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # 보통 export가 weights/best.onnx로 만들어줌
    results = model.export(format="onnx", opset=opset, imgsz=640, dynamic=True, half=False)
    # 결과 경로가 다를 수 있으므로 out_path로 복사 정규화
    produced = str(results) if isinstance(results, (str, Path)) else None
    if produced and produced != out_path:
        shutil.copyfile(produced, out_path)
