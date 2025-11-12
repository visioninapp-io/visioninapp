# graph/training/nodes/init_context.py
from __future__ import annotations

import os
import json
import time
import uuid
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from llm.graph.training.state import TrainState

try:
    import torch
except Exception:
    torch = None

try:
    import psutil
except Exception:
    psutil = None

try:
    import yaml
except Exception:
    yaml = None
import logging

logger = logging.getLogger("uvicorn.error")

# -----------------------------
# 헬퍼 함수
# -----------------------------
def _load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML 미설치: pip install pyyaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _detect_system_env() -> Dict[str, Any]:
    info = {
        "cuda_available": False,
        "gpu_name": None,
        "gpu_mem_gb": 0.0,
        "cpu_threads": os.cpu_count() or 1,
        "ram_gb": None,
        "platform": os.name,
    }
    if psutil:
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)

    if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
        info["cuda_available"] = True
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_mem_gb"] = round(props.total_memory / (1024**3), 1)
        except Exception:
            pass
    return info


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def _resolve_device(user_device: Optional[str], sysinfo: Dict[str, Any]) -> str:
    if user_device:
        return user_device
    return "cuda:0" if sysinfo.get("cuda_available") else "cpu"


def _ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _norm_str(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    low = s.lower()
    if low in {"null", "none", "na"}:
        return None
    return s


# -----------------------------
# init_context 노드 (LLM 제거)
# -----------------------------
def init_context(state: TrainState) -> TrainState:
    """
    LLM 관련 로직 제거 버전
    - training.yaml 로드
    - 시스템 자원 감지/시드/런ID/로그/디바이스/AMP 정책 세팅
    - HPO/게이트 초기화
    - context 구성
    """
    # 1️⃣ training.yaml 로드
    config_path = state.config_path or "configs/training.yaml"
    train_cfg = _load_yaml(config_path) if Path(config_path).exists() else {}
    state.train_cfg = train_cfg

    # 2️⃣ 시스템 자원 감지
    sysinfo = _detect_system_env()

    # 3️⃣ 디바이스/정밀도 정책
    hw = train_cfg.get("hardware") or {}
    device = _resolve_device(hw.get("device"), sysinfo)
    mp_cfg = bool(hw.get("mixed_precision", True))
    amp_dtype = "fp16" if mp_cfg else "fp32"

    if torch:
        try:
            cudnn_benchmark = bool(hw.get("cudnn_benchmark", True))
            torch.backends.cudnn.benchmark = cudnn_benchmark
        except Exception:
            pass

    # 4️⃣ 시드/런 ID/로그 경로
    seed = int(state.seed or (train_cfg.get("train", {}).get("seed") if train_cfg.get("train") else 42) or 42)
    state.seed = seed
    _seed_everything(seed)

    prefix = f"{state.run_name}_" if state.run_name else ""
    run_id = f"{prefix}{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    project_name = (train_cfg.get("meta") or {}).get("project_name", "yolo-training")
    base_log_dir = Path((train_cfg.get("meta") or {}).get("log_dir", f"./runs/{project_name}"))
    run_dir = Path(base_log_dir) / run_id
    _ensure_dir(run_dir)

    # 5️⃣ HPO / 게이트 초기화
    hpo_cfg = train_cfg.get("hpo") or {}
    hpo_cfg.setdefault("enabled", False)
    state.hpo = {**hpo_cfg, **(state.hpo or {})}

    gate_cfg = train_cfg.get("gate") or {}
    gate_cfg.setdefault("enabled", True)
    state.gate = {**gate_cfg, **(state.gate or {})}

    # 6️⃣ context 구성
    state.context = {
        "project_name": project_name,
        "run_id": run_id,
        "seed": seed,
        "device": device,
        "amp": mp_cfg,
        "amp_dtype": amp_dtype,
        "log_dir": str(run_dir),
        "system_info": sysinfo,
        "env": {
            "working_dir": str(Path.cwd()),
            "python": os.sys.version.split()[0],
        },
        # SSAFY/RAG용 필드 기본값
        "intent": state.intent,
        "model_variant": state.model_variant,
        "precision": state.precision or amp_dtype,
        # "dataset_version": state.dataset_version,
        "base_model": state.base_model,
        "resume": state.resume,
    }

    logger.info(json.dumps({
        "init_context": {
            "project": project_name,
            "run_id": run_id,
            "device": device,
            "amp": mp_cfg,
            "amp_dtype": amp_dtype,
            "log_dir": str(run_dir),
            "gpu": sysinfo.get("gpu_name"),
            "gpu_mem_gb": sysinfo.get("gpu_mem_gb"),
            "ram_gb": sysinfo.get("ram_gb"),
            "hpo_enabled": state.hpo.get("enabled"),
            "precision": state.precision or amp_dtype,
            # "dataset_version": state.dataset_version,
            "model_variant": state.model_variant,
            "intent": state.intent,
        }
    }, ensure_ascii=False, indent=2))

    # 7️⃣ model_variant 보정
    mv = _norm_str(state.model_variant)
    if mv:
        if not mv.endswith(".pt"):
            mv = mv + ".pt"
        cfg = state.train_cfg or {}
        tr = (cfg.get("train") or {}).copy()
        tr["model_name"] = mv
        cfg["train"] = tr
        state.train_cfg = cfg

    return state
