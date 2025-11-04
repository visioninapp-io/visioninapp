# graph/export/evaluate_convert_model.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from ultralytics import YOLO, settings
from graph.training.state import TrainState


# --------- small helpers ----------
def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x).strip().replace("%", ""))
    except Exception:
        return None


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


def _resolve_data_yaml(state: TrainState) -> str:
    train_cfg: Dict[str, Any] = state.train_cfg or {}
    data_cfg: Dict[str, Any] = train_cfg.get("data", {}) if train_cfg else {}
    yaml_path = data_cfg.get("yaml_path")

    if yaml_path:
        p = Path(yaml_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"지정된 data.yaml이 없습니다: {p}")
        return str(p.resolve())

    ds_ver = getattr(state, "dataset_version", None) or "dataset@1.0.0"
    p = Path(f"data/datasets/{ds_ver}/data.yaml").resolve()
    if not p.exists():
        ds_root = settings.get("datasets_dir", None)
        raise FileNotFoundError(
            f"Dataset '{p}' 가 존재하지 않습니다.\n"
            f"Ultralytics datasets_dir = {ds_root}\n"
            f"→ dataset_version 또는 data.yaml 경로를 확인하세요."
        )
    return str(p)


def _summarize_val_results(results) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        rd = dict(getattr(results, "results_dict", {}) or {})
    except Exception:
        rd = {}

    if "metrics/mAP50-95" in rd:
        out["mAP50-95"] = _to_float(rd["metrics/mAP50-95"])
    elif "mAP50-95" in rd:
        out["mAP50-95"] = _to_float(rd["mAP50-95"])
    elif "map50-95" in rd:
        out["mAP50-95"] = _to_float(rd["map50-95"])

    if "metrics/mAP50" in rd:
        out["mAP50"] = _to_float(rd["metrics/mAP50"])
    elif "mAP50" in rd:
        out["mAP50"] = _to_float(rd["mAP50"])
    elif "map50" in rd:
        out["mAP50"] = _to_float(rd["map50"])

    for k in ("precision", "recall", "f1", "fitness"):
        if k in rd:
            out[k] = _to_float(rd[k])
        elif f"metrics/{k}" in rd:
            out[k] = _to_float(rd[f"metrics/{k}"])

    out["raw"] = rd
    return out


def _eval_one(yolo_weight_path: Path, data_yaml: str, imgsz: int) -> Dict[str, Any]:
    model = YOLO(str(yolo_weight_path))
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=1,
        workers=0,
    )
    return _summarize_val_results(results)


# --------- main node ----------
def evaluate_convert_model(state: TrainState) -> TrainState:
    """
    query_analyzer → parsed → onnx/tensorrt 플래그를 바탕으로
    변환된 모델(onnx/engine)을 평가하고 summary 파일을 저장.
    """
    print("[evaluate_convert_model] 변환 모델 평가 시작")
    settings.update({"datasets_dir": str(Path("data/datasets").resolve())})

    # 어떤 걸 평가할지 플래그 확인
    ctx = state.context or {}
    qa = ctx.get("query_analyzer", {})
    parsed = qa.get("parsed", {}) or {}
    use_onnx = bool(parsed.get("onnx", False))
    use_tensor = bool(parsed.get("tensorrt", False))

    # 경로/공통 파라미터
    reg = getattr(state, "registry_info", {}) or {}
    model_dir = reg.get("path")
    if not model_dir:
        raise RuntimeError("registry_info.path가 비었습니다. registry_publish 이후에 실행하세요.")
    model_dir = Path(model_dir).resolve()

    data_yaml = _resolve_data_yaml(state)
    imgsz = _get_imgsz_from_state(state)

    # 결과 기록용
    eval_ctx: Dict[str, Any] = {"imgsz": imgsz, "data_yaml": str(Path(data_yaml).resolve())}

    # TensorRT 평가
    if use_tensor:
        eng_path = model_dir / "best.engine"
        if not eng_path.exists():
            print(f"[evaluate_convert_model] 경고: TensorRT 엔진 미존재 → {eng_path}")
            eval_ctx["tensor_error"] = "engine file not found"
        else:
            try:
                metrics = _eval_one(eng_path, data_yaml, imgsz)
                sum_path = model_dir / "summary_tensor.json"
                with open(sum_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "registry_id": reg.get("registry_id"),
                        "engine_path": str(eng_path),
                        "data_yaml": eval_ctx["data_yaml"],
                        "imgsz": imgsz,
                        "evaluated_at": datetime.now().isoformat(),
                        "metrics": metrics,
                    }, f, indent=2, ensure_ascii=False)
                print(f"[evaluate_convert_model] TensorRT 요약 저장 → {sum_path}")
                eval_ctx["tensor_summary"] = str(sum_path)
            except Exception as e:
                print(f"[evaluate_convert_model] 경고: TensorRT 평가 실패 → {e}")
                eval_ctx["tensor_error"] = str(e)

    # ONNX 평가
    if use_onnx:
        onnx_path = model_dir / "best.onnx"
        if not onnx_path.exists():
            print(f"[evaluate_convert_model] 경고: ONNX 파일 미존재 → {onnx_path}")
            eval_ctx["onnx_error"] = "onnx file not found"
        else:
            try:
                metrics = _eval_one(onnx_path, data_yaml, imgsz)
                sum_path = model_dir / "summary_onnx.json"
                with open(sum_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "registry_id": reg.get("registry_id"),
                        "onnx_path": str(onnx_path),
                        "data_yaml": eval_ctx["data_yaml"],
                        "imgsz": imgsz,
                        "evaluated_at": datetime.now().isoformat(),
                        "metrics": metrics,
                    }, f, indent=2, ensure_ascii=False)
                print(f"[evaluate_convert_model] ONNX 요약 저장 → {sum_path}")
                eval_ctx["onnx_summary"] = str(sum_path)
            except Exception as e:
                print(f"[evaluate_convert_model] 경고: ONNX 평가 실패 → {e}")
                eval_ctx["onnx_error"] = str(e)

    # 아무 것도 true가 아니면 스킵
    if not use_onnx and not use_tensor:
        print("[evaluate_convert_model] onnx/tensorrt 플래그가 없어 평가 스킵")

    # state 업데이트
    ctx["evaluate_convert_model"] = eval_ctx
    state.context = ctx

    reg2 = dict(state.registry_info or {})
    if "tensor_summary" in eval_ctx:
        reg2["summary_tensor"] = eval_ctx["tensor_summary"]
    if "onnx_summary" in eval_ctx:
        reg2["summary_onnx"] = eval_ctx["onnx_summary"]
    state.registry_info = reg2

    print("[evaluate_convert_model] 완료 ✅")
    return state
