from ultralytics import YOLO
from pathlib import Path
import shutil

from ultralytics import YOLO
from pathlib import Path
import shutil

def to_onnx(
    pt_path: str,
    out_path: str,
    opset: int = 13,
    imgsz: int = 640,
    dynamic: bool = True,
    half: bool = False,
    simplify: bool = False,
    precision: str | None = None,  # "fp16"|"fp32"|None (half와 중복 지정 가능, precision이 있으면 half를 재설정)
):
    """
    PT → ONNX 변환
    - precision: "fp16"이면 half=True, "fp32"이면 half=False (명시적 half보다 우선)
    - simplify: onnx-simplifier 활성화
    """
    # precision 우선 규칙
    if isinstance(precision, str):
        p = precision.strip().lower()
        if p == "fp16":
            half = True
        elif p == "fp32":
            half = False

    model = YOLO(pt_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Ultralytics export 호출
    results = model.export(
        format="onnx",
        opset=opset,
        imgsz=imgsz,
        dynamic=bool(dynamic),
        half=bool(half),
        simplify=bool(simplify),
    )

    # 결과 경로 정규화
    produced = str(results) if isinstance(results, (str, Path)) else None
    if produced and produced != out_path:
        shutil.copyfile(produced, out_path)
    else:
        # 혹시 결과 경로를 못받았는데 기본 경로로 생겼다면 보정
        cand = Path(out_path).parent / "best.onnx"
        if cand.exists() and str(cand) != out_path:
            shutil.copyfile(str(cand), out_path)
