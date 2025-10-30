# graph/training/nodes/train_trial.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import os
import shutil

from graph.training.state import TrainState
from utils.default_weight import ensure_weight_local

# Ultralytics (optional)
try:
    from ultralytics import YOLO  # type: ignore
    _ULTRA = True
except Exception:
    _ULTRA = False


def _merge_train_params(state: TrainState) -> Dict[str, Any]:
    """
    학습 파라미터 병합 규칙:
    YAML.train 기본값  <- HPO best_trial.params  <- 사용자 train_overrides (최우선)
    """
    cfg = state.train_cfg or {}
    base = (cfg.get("train") or {}).copy()
    best = ((state.best_trial or {}).get("params") or {}).copy()
    over = (state.train_overrides or {}).copy()

    merged = {**base, **best, **over}

    # 최소 세트 보정(기본값)
    merged.setdefault("epochs", 100)
    merged.setdefault("batch", 16)
    merged.setdefault("imgsz", 640)
    merged.setdefault("workers", 8)
    merged.setdefault("optimizer", merged.get("optimizer", "SGD"))
    merged.setdefault("lr0", 0.001)
    merged.setdefault("lrf", 0.01)
    merged.setdefault("weight_decay", 5e-4)
    merged.setdefault("momentum", 0.937)
    merged.setdefault("patience", 20)

    return merged

def _normalize_hub_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.replace("yolov11", "yolo11")  # 흔한 오타 보정, 필요시 확장
    if s and not s.endswith((".pt", ".yaml")):
        s += ".pt"
    return s

def _resolve_weights(state: TrainState) -> Optional[str]:
    over = (getattr(state, "train_overrides", None) or {})
    user_over = over.get("model_name") or over.get("weights")
    if user_over:
        return _normalize_hub_name(user_over)

    ctx = state.context or {}
    dm = (ctx.get("decide_mode") or {}).get("mode", "")
    last_ckpt = getattr(state, "last_ckpt", None)
    base_model = getattr(state, "base_model", None)

    if dm == "path_resume" and last_ckpt:
        return last_ckpt
    if dm == "path_finetune" and base_model:
        return base_model

    mv = getattr(state, "model_variant", None)
    if mv:
        return _normalize_hub_name(mv)

    tr = (state.train_cfg or {}).get("train") or {}
    if tr.get("model_name"):
        return _normalize_hub_name(tr["model_name"])

    return "yolo11n.pt"  # 안전 폴백


def _extract_metrics_ultralytics(model: Any) -> Dict[str, Any]:
    """
    다양한 버전 호환을 위해 가능한 경로를 순차적으로 시도.
    """
    # 1) 최신 trainer 인터페이스
    try:
        tr = getattr(model, "trainer", None)
        if tr is not None:
            # 일부 버전: tr.metrics or tr.metrics.results_dict
            md = getattr(tr, "metrics", None)
            if isinstance(md, dict):
                return md
            if hasattr(md, "results_dict"):
                return dict(md.results_dict)
    except Exception:
        pass

    # 2) results 반환 / save_dir의 metrics.json
    try:
        save_dir = Path(getattr(model.trainer, "save_dir"))  # type: ignore
        mfile = save_dir / "results.json"
        if mfile.exists():
            return json.loads(mfile.read_text(encoding="utf-8"))
        mfile = save_dir / "metrics.json"
        if mfile.exists():
            return json.loads(mfile.read_text(encoding="utf-8"))
    except Exception:
        pass

    return {}


def train_trial(state: TrainState) -> TrainState:
    """
    - 데이터/디바이스/AMP/가중치/하이퍼파라미터를 정리해서 실제 학습 실행
    - 성공 시: state.model_path, state.metrics, context.train_trial 요약 채움
    - 실패/미설치 시: 안전 폴백(모의 결과)
    """
    # --- 입력 준비 ---
    data_info = state.data or {}
    data_yaml = data_info.get("yaml_path")
    if not data_yaml:
        raise ValueError("[train_trial] data.yaml 경로가 없습니다. load_dataset 노드를 확인하세요.")

    params = _merge_train_params(state)

    ctx = state.context or {}
    device = ctx.get("device", "cpu")
    amp = bool(ctx.get("amp", False))
    amp_dtype = ctx.get("amp_dtype", "fp16" if amp else "fp32")

    weights = _resolve_weights(state)
    freeze_backbone = bool(((state.train_cfg or {}).get("resume") or {}).get("freeze_backbone", False))
    run_dir = Path(ctx.get("log_dir", "./runs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics 실행 인자 구성
    train_args = {
        "data": data_yaml,
        "epochs": int(params["epochs"]),
        "imgsz": int(params["imgsz"]),
        "batch": int(params["batch"]),
        "device": device,
        "workers": int(params["workers"]),
        "optimizer": params["optimizer"],
        "lr0": float(params["lr0"]),
        "lrf": float(params["lrf"]),
        "weight_decay": float(params["weight_decay"]),
        "momentum": float(params["momentum"]),
        "patience": int(params["patience"]),
        # 로깅/출력
        "project": str(run_dir),
        "name": "train_trial",
        "save": True,
        # AMP 관리 (버전에 따라 'amp'/'half'/'amp' 키가 다를 수 있어 try로)
    }

    # AMP/precision 힌트
    if amp:
        # 일부 버전은 'amp' bool, 일부는 자동 처리. 안전하게 그냥 둠.
        train_args["amp"] = True
    # freeze 옵션은 버전에 따라 'freeze'로 레이어 범위 전달이 일반적
    if freeze_backbone:
        train_args["freeze"] = 10  # 보수적 예시(백본 앞단 일부)

    if weights:
        weights = str(ensure_weight_local(state, _normalize_hub_name(weights)))

    # --- Ultralytics 유무에 따른 실행 ---
    if not _ULTRA:
        # 안전 폴백(모의)
        state.model_path = None
        state.metrics = {"mAP50-95": 0.0, "note": "ultralytics not installed (mock run)"}
        c = state.context or {}
        c["train_trial"] = {
            "ran": False,
            "reason": "ultralytics_not_available",
            "device": device,
            "amp_dtype": amp_dtype,
            "weights": weights,
            "args": train_args,
        }
        state.context = c
        print("[train_trial] Ultralytics 미설치: 모의 결과로 대체합니다.")
        return state

    # 실제 학습 실행
    try:
        print(f"[train_trial] starting: device={device} amp={amp_dtype} weights={weights}")
        model = YOLO(weights) if weights else YOLO()  # weights 없으면 family default 사용
        results = model.train(**train_args)

        # best 가중치 경로
        best_path = None
        try:
            best_path = str(Path(getattr(model.trainer, "best")))  # type: ignore
        except Exception:
            # save_dir/weights/best.pt 추정
            try:
                save_dir = Path(getattr(model.trainer, "save_dir"))  # type: ignore
                cand = save_dir / "weights" / "best.pt"
                if cand.exists():
                    best_path = str(cand)
            except Exception:
                pass

        # 메트릭 추출
        metrics = _extract_metrics_ultralytics(model)
        # 중요한 메트릭 별칭 보정
        if "metrics/mAP50-95(B)" in metrics and "mAP50-95" not in metrics:
            metrics["mAP50-95"] = metrics["metrics/mAP50-95(B)"]

        state.model_path = best_path
        state.metrics = metrics or {}

        # 컨텍스트 로그
        c = state.context or {}
        c["train_trial"] = {
            "ran": True,
            "save_dir": str(getattr(model.trainer, "save_dir", "")),
            "best_path": best_path,
            "device": device,
            "amp_dtype": amp_dtype,
            "weights": weights,
            "args": train_args,
            "headline_metric": state.metrics.get("mAP50-95") or state.metrics.get("map50-95") or None,
        }
        state.context = c

        print(f"[train_trial] done: best={best_path} mAP50-95={c['train_trial']['headline_metric']}")
        return state

    except Exception as e:
        # 실패 시 안전 폴백
        state.model_path = None
        state.metrics = {"mAP50-95": 0.0, "error": str(e)}
        c = state.context or {}
        c["train_trial"] = {
            "ran": False,
            "reason": "exception",
            "error": str(e),
            "device": device,
            "amp_dtype": amp_dtype,
            "weights": weights,
            "args": train_args,
        }
        state.context = c
        print(f"[train_trial] 학습 실패: {e}")
        return state
