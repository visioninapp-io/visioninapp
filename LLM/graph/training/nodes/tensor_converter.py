# graph/export/tensor_converter.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from ultralytics import YOLO, settings
from graph.training.state import TrainState


# --------- small helpers ----------
def _get_imgsz_from_state(state: TrainState) -> int:
    ctx = getattr(state, "context", {}) or {}
    tt = (ctx.get("train_trial") or {})
    args = tt.get("args") or {}
    if "imgsz" in args and args["imgsz"]:
        try:
            return int(args["imgsz"])
        except Exception:
            pass
    if state.train_overrides and state.train_overrides.get("imgsz"):
        try:
            return int(state.train_overrides["imgsz"])
        except Exception:
            pass
    return 640


def _resolve_best_pt_path(state: TrainState) -> Path:
    reg = getattr(state, "registry_info", {}) or {}
    model_dir = reg.get("path")
    if not model_dir:
        raise RuntimeError("registry_info.path가 비었습니다. registry_publish 이후에 실행하세요.")
    pt = Path(model_dir).resolve() / "best.pt"
    if not pt.exists():
        raise FileNotFoundError(f"best.pt가 없습니다: {pt}")
    return pt


# --------- main node ----------
def tensor_converter(state: TrainState) -> TrainState:
    """
    best.pt → best.engine 변환만 수행. (평가는 하지 않음)
    결과 경로는 state.context['tensor_converter']와 state.registry_info에 기록.
    """
    print("[tensor_converter] TensorRT 변환 시작")
    settings.update({"datasets_dir": str(Path("data/datasets").resolve())})

    pt_path = _resolve_best_pt_path(state)
    imgsz = _get_imgsz_from_state(state)
    model_dir = pt_path.parent
    eng_path = model_dir / "best.engine"

    if eng_path.exists():
        print(f"[tensor_converter] 기존 엔진 발견 → {eng_path}")
    else:
        model = YOLO(str(pt_path))
        model.export(
            format="engine",
            device="cuda",
            half=True,
            simplify=True,
            imgsz=imgsz,
            dynamic=False,
        )
        print(f"[tensor_converter] export 완료 → {eng_path}")

    # 기록
    ctx = state.context or {}
    ctx["tensor_converter"] = {
        "engine_path": str(eng_path),
        "imgsz": imgsz,
    }
    state.context = ctx

    reg = dict(state.registry_info or {})
    reg["engine_path"] = str(eng_path)
    state.registry_info = reg

    print("[tensor_converter] 완료 ✅")
    return state
