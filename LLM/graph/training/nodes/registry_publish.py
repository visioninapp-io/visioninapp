from __future__ import annotations
import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from graph.training.state import TrainState


def registry_publish(state: TrainState) -> TrainState:
    """
    학습된 모델을 data/models 폴더 아래에 등록하는 노드.
    - 등록 폴더: data/models/<model_name>_<timestamp>/
    - best.pt 및 meta.json 저장
    """
    print("[registry_publish] 노드 실행")

    model_path = getattr(state, "model_path", None)
    metrics = getattr(state, "metrics", {}) or {}
    context = state.context or {}

    if not model_path or not os.path.exists(model_path):
        print(f"[registry_publish] 경고: model_path가 존재하지 않음 → 등록 건너뜀 ({model_path})")
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
    model_name = Path(model_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    registry_id = f"{model_name}_{timestamp}"

    target_dir = base_dir / registry_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # --- 모델 복사 ---
    try:
        shutil.copy2(model_path, target_dir / Path(model_path).name)
        print(f"[registry_publish] 모델 복사 완료 → {target_dir}")
    except Exception as e:
        print(f"[registry_publish] 모델 복사 실패: {e}")

    # --- 메타데이터 저장 ---
    meta = {
        "model_name": model_name,
        "registry_id": registry_id,
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics,
        "source_model_path": str(model_path),
        "context_summary": {
            "dataset_version": getattr(state, "dataset_version", None),
            "base_model": getattr(state, "base_model", None),
            "resume": getattr(state, "resume", False),
        },
    }

    meta_path = target_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # --- TrainState 업데이트 ---
    state.registry_info = {
        "status": "registered",
        "registry_id": registry_id,
        "path": str(target_dir),
        "meta_path": str(meta_path),
    }

    context["registry_publish"] = {
        "registry_id": registry_id,
        "path": str(target_dir),
        "registered_at": meta["registered_at"],
    }
    state.context = context

    print(f"[registry_publish] 등록 완료 ✅ ({registry_id})")
    return state
