from ultralytics import YOLO
from pathlib import Path
import shutil

def to_tensorrt(
    pt_path: str,
    out_path: str,
    precision: str = "fp16",
    imgsz: int = 640,
    dynamic: bool = True,
):
    """
    PT 가중치 → TensorRT 엔진 변환
    precision: fp16 | fp32 | int8
    """
    work_dir = Path(out_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(pt_path)

    # export 인자 구성
    prec = (precision or "fp16").lower()
    export_kwargs = dict(format="engine", imgsz=imgsz, dynamic=dynamic)

    if prec == "fp16":
        export_kwargs["half"] = True          # FP16
    elif prec == "fp32":
        export_kwargs["half"] = False         # FP32
    elif prec == "int8":
        export_kwargs["int8"] = True          # INT8 (칼리브레이션 필요할 수 있음)
    else:
        export_kwargs["half"] = True          # 기본 FP16

    res = model.export(**export_kwargs)
    produced = str(res) if isinstance(res, (str, Path)) else None

    # 엔진 파일 복사/정규화
    if produced and produced.endswith(".engine"):
        if produced != out_path:
            shutil.copyfile(produced, out_path)
    else:
        cand = work_dir / "best.engine"
        if cand.exists():
            shutil.copyfile(cand, out_path)
        else:
            raise FileNotFoundError("TensorRT engine not found after export")
