from __future__ import annotations
import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from llm.graph.training.state import TrainState


# ---- 유틸: 안전 float 변환 ----
def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x).strip().replace("%", ""))
    except Exception:
        return None


# ---- 유틸: 최종 하이퍼파라미터 수집 ----
def _collect_final_hparams(state: TrainState) -> Dict[str, Any]:
    """
    1순위: train_trial에서 실제로 YOLO에 전달한 args (가장 신뢰도 높음)
    2순위: train_cfg.train + best_trial.params + train_overrides 병합
    """
    # 1) context.train_trial.args 사용
    ctx = state.context or {}
    tt = (ctx.get("train_trial") or {})
    args = (tt.get("args") or {}).copy()
    if args:
        # 가중치/디바이스/AMP 등 힌트도 같이 태워주자
        if tt.get("weights"):
            args["weights"] = tt["weights"]
        if tt.get("device"):
            args["device"] = tt["device"]
        if tt.get("amp_dtype"):
            args["amp_dtype"] = tt["amp_dtype"]
        return args

    # 2) 병합 폴백: train_cfg.train <- best_trial.params <- train_overrides
    cfg = (getattr(state, "train_cfg", None) or {})
    base = (cfg.get("train") or {}).copy()
    best = ((getattr(state, "best_trial", None) or {}).get("params") or {}).copy()
    over = (getattr(state, "train_overrides", None) or {}).copy()
    merged = {**base, **best, **over}

    # 기본값 보정(일부 필수 키)
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

    # 힌트 키
    if getattr(state, "model_path", None):
        merged.setdefault("best_path_hint", state.model_path)
    if (ctx.get("device")):
        merged.setdefault("device", ctx["device"])
    if (ctx.get("amp")) is not None:
        merged.setdefault("amp", bool(ctx["amp"]))

    return merged


# ---- 유틸: 메트릭 요약 ----
def _summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    primary = metrics.get("primary_metric")
    score = _to_float(metrics.get("score"))
    # 자주 쓰는 지표들만 추려서 요약
    keys_prefer = [
        "mAP50-95", "map50-95", "metrics/mAP50-95", "metrics/mAP50-95(B)",
        "mAP50", "map50", "metrics/mAP50",
        "precision", "metrics/precision",
        "recall", "metrics/recall",
        "f1", "metrics/f1",
        "fitness", "metrics/fitness",
    ]
    picked: Dict[str, Any] = {}
    for k in keys_prefer:
        if k in metrics:
            picked[k] = metrics[k]
    # 보기 좋은 alias도 함께
    alias_view = {}
    alias_view["primary_metric"] = primary
    alias_view["score"] = score
    # 대표 지표 별칭 정리
    if "mAP50-95" in metrics:
        alias_view["mAP50-95"] = _to_float(metrics["mAP50-95"])
    elif "map50-95" in metrics:
        alias_view["mAP50-95"] = _to_float(metrics["map50-95"])
    elif "metrics/mAP50-95" in metrics:
        alias_view["mAP50-95"] = _to_float(metrics["metrics/mAP50-95"])
    elif "metrics/mAP50-95(B)" in metrics:
        alias_view["mAP50-95"] = _to_float(metrics["metrics/mAP50-95(B)"])

    if "mAP50" in metrics:
        alias_view["mAP50"] = _to_float(metrics["mAP50"])
    elif "map50" in metrics:
        alias_view["mAP50"] = _to_float(metrics["map50"])
    elif "metrics/mAP50" in metrics:
        alias_view["mAP50"] = _to_float(metrics["metrics/mAP50"])

    if "precision" in metrics:
        alias_view["precision"] = _to_float(metrics["precision"])
    elif "metrics/precision" in metrics:
        alias_view["precision"] = _to_float(metrics["metrics/precision"])

    if "recall" in metrics:
        alias_view["recall"] = _to_float(metrics["recall"])
    elif "metrics/recall" in metrics:
        alias_view["recall"] = _to_float(metrics["metrics/recall"])

    if "f1" in metrics:
        alias_view["f1"] = _to_float(metrics["f1"])
    elif "metrics/f1" in metrics:
        alias_view["f1"] = _to_float(metrics["metrics/f1"])

    if "fitness" in metrics:
        alias_view["fitness"] = _to_float(metrics["fitness"])
    elif "metrics/fitness" in metrics:
        alias_view["fitness"] = _to_float(metrics["metrics/fitness"])

    return {
        "primary_metric": primary,
        "score": score,
        "aliases": alias_view,
        "raw": picked,   # 원본 키 그대로 모아둔 뷰
    }


def registry_publish(state: TrainState) -> TrainState:
    """
    학습된 모델을 data/models 폴더 아래에 등록하는 노드.
    - 등록 폴더: data/models/<model_name>_<timestamp>/
    - best.pt 및 meta.json, summary.json 저장
    - 콘솔에 최종 하이퍼/성능 요약 출력
    """
    print("[registry_publish] 노드 실행")

    model_path = getattr(state, "model_path", None)
    metrics = getattr(state, "metrics", {}) or {}
    context = state.context or {}

    if not model_path:
        print(f"[registry_publish] 경고: model_path가 없음 → 등록 건너뜀")
        state.registry_info = {
            "status": "skipped",
            "reason": "missing_model",
            "registered_at": datetime.now().isoformat(),
        }
        return state
    
    # S3 URI인 경우 다운로드
    local_model_path = model_path
    if isinstance(model_path, str) and model_path.startswith("s3://"):
        print(f"[registry_publish] S3 URI 감지: {model_path}")
        try:
            from llm.tools.s3_client import download_s3
            import tempfile
            
            # 임시 파일로 다운로드
            local_model_path = os.path.join(tempfile.gettempdir(), f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
            print(f"[registry_publish] S3에서 다운로드 중: {model_path} -> {local_model_path}")
            download_s3(model_path, local_model_path)
            print(f"[registry_publish] 다운로드 완료")
        except Exception as e:
            print(f"[registry_publish] S3 다운로드 실패: {e} → 등록 건너뜀")
            state.registry_info = {
                "status": "skipped",
                "reason": f"s3_download_failed: {e}",
                "registered_at": datetime.now().isoformat(),
            }
            return state
    
    if not os.path.exists(local_model_path):
        print(f"[registry_publish] 경고: model_path가 존재하지 않음 → 등록 건너뜀 ({local_model_path})")
        state.registry_info = {
            "status": "skipped",
            "reason": "missing_model",
            "registered_at": datetime.now().isoformat(),
        }
        return state

    # --- 레지스트리 경로: ./data/models ---
    base_dir = Path("data/models").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # --- 버전/폴더명 구성 ---
    model_name = Path(local_model_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    registry_id = f"{model_name}_{timestamp}"

    target_dir = base_dir / registry_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # --- 모델 복사 ---
    try:
        shutil.copy2(local_model_path, target_dir / Path(local_model_path).name)
        print(f"[registry_publish] 모델 복사 완료 → {target_dir}")
    except Exception as e:
        print(f"[registry_publish] 모델 복사 실패: {e}")

    # ---- 최종 하이퍼/성능 요약 수집 ----
    final_hparams = _collect_final_hparams(state)
    metric_summary = _summarize_metrics(metrics)

    # ---- 콘솔 출력 (사람이 보기 좋게) ----
    print("\n[registry_publish] 📦 최종 등록 요약")
    print("  • 모델 경로 :", model_path)  # 원본 경로 (S3 URI 포함)
    if local_model_path != model_path:
        print("  • 로컬 경로 :", local_model_path)
    print("  • 레지스트리:", str(target_dir))
    # 성능
    pm = metric_summary.get("primary_metric")
    sc = metric_summary.get("score")
    print(f"  • 성능: primary_metric={pm}  score={sc}")

    aliases = (state.context.get("evaluate_trial") or {}).get("metrics_aliases", {})
    for k, v in aliases.items():
        print(f"      - {k:14s}: {v}")

    # 하이퍼
    print("  • 하이퍼파라미터(핵심):")
    for key in ["epochs", "imgsz", "batch", "optimizer", "lr0", "lrf", "weight_decay", "momentum", "patience", "device", "amp_dtype", "weights"]:
        if key in final_hparams:
            print(f"      - {key:12s}: {final_hparams[key]}")

    # ---- 메타데이터 기록 ----
    meta = {
        "model_name": model_name,
        "registry_id": registry_id,
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics,
        "metric_summary": metric_summary,
        "final_hparams": final_hparams,
        "source_model_path": str(model_path),  # 원본 S3 URI 저장
        "local_model_path": str(local_model_path) if local_model_path != model_path else None,
        "context_summary": {
            "dataset_version": getattr(state, "dataset_version", None),
            "base_model": getattr(state, "base_model", None),
            "resume": getattr(state, "resume", False),
        },
    }

    meta_path = target_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---- 요약 전용 파일 추가 (사내 도구/대시보드용) ----
    summary_path = target_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "registry_id": registry_id,
                "primary_metric": pm,
                "score": sc,
                "metrics_aliases": aliases,
                "final_hparams": final_hparams,
                "registered_at": meta["registered_at"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --- TrainState 업데이트 ---
    state.registry_info = {
        "status": "registered",
        "registry_id": registry_id,
        "path": str(target_dir),
        "meta_path": str(meta_path),
        "summary_path": str(summary_path),
    }

    context["registry_publish"] = {
        "registry_id": registry_id,
        "path": str(target_dir),
        "registered_at": meta["registered_at"],
    }
    state.context = context

    print(f"\n[registry_publish] 등록 완료 ✅ ({registry_id})")
    return state
