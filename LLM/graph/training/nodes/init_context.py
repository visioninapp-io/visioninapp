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

from graph.training.state import TrainState

# optional deps
try:
    import torch
except Exception:
    torch = None  # CPU-only 허용

try:
    import psutil
except Exception:
    psutil = None

try:
    import yaml
except Exception:
    yaml = None

# --- LLM (optional) ---
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False


# -----------------------------
# LLM 파서 (안전 폴백 포함)
# -----------------------------
def _safe_parse_user_query(user_input: str) -> dict:
    """자연어 입력을 구조화 JSON으로 파싱. 실패 시 {} 반환."""
    if not _LLM_AVAILABLE or not user_input:
        return {}
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            """
            당신은 AI 학습 파이프라인 매니저입니다.
            사용자의 요청을 아래 JSON 형식으로만(설명 없이) 반환하세요.

            규칙(요약):
            - 모델 패밀리만 적으면 접미사(n/s/m/l/x)를 추론
            - 임베디드/저전력 기기면 n 또는 s 선호
            - 모든 모델명은 'yolo8n', 'yolo11n', 'yolo12x'처럼 'v' 없이 표기
            - 증분학습/파인튜닝 요청 시 base_model 기입
            - 이전 학습 이어하기 요청 시 resume=true

            {{
              "intent": "train_model | add_dataset | retrain | optimize | export",
              "target_device": "예: Jetson Orin Nano / RTX 3060 / CPU",
              "model_variant": "예: yolo8n | yolo11n | yolo12x",
              "precision": "fp16 | int8 | fp32",
              "dataset_version": "예: dataset@1.0.0",
              "base_model": "예: model@1.0.0.pt 또는 null",
              "resume": true/false,
              "notes": "추가 설명"
            }}

            사용자 요청: {query}
            """
        )
        chain = prompt | llm
        res = chain.invoke({"query": user_input})
        content = getattr(res, "content", "") or ""
        data = json.loads(content)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# -----------------------------
# 헬퍼: YAML/경로/시드/자원 감지
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
            info["gpu_name"] = torch.cuda.get_device_name(0)  # type: ignore[attr-defined]
            props = torch.cuda.get_device_properties(0)       # type: ignore[attr-defined]
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
            if hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
                torch.cuda.manual_seed_all(seed)                      # type: ignore[attr-defined]
            if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True            # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False               # type: ignore[attr-defined]
        except Exception:
            pass

def _norm_str(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    low = s.lower()
    if low in {"null", "none", "na"}:
        return None
    if s.startswith("예:"):
        return None
    return s

def _resolve_device(user_device: Optional[str], sysinfo: Dict[str, Any]) -> str:
    """사용자/설정에서 지정된 device가 있으면 그것을, 없으면 CUDA 유무로 결정."""
    if user_device:
        return user_device
    return "cuda:0" if sysinfo.get("cuda_available") else "cpu"

def _ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def _as_none(x):
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"null", "none", ""}:
        return None
    return x

def _strip_example(val: str | None) -> str | None:
    if not isinstance(val, str):
        return val
    t = val.strip()
    if t.startswith("예:") or t.lower().startswith("e.g.") or t.lower().startswith("ex:"):
        return None
    return t or None


# -----------------------------
# init_context 노드 (target_profile 완전 제거판)
# -----------------------------
def init_context(state: TrainState) -> TrainState:
    """
    - LLM으로 사용자 의도를 파싱하여 state 표준 필드에 매핑
    - training.yaml 로드
    - 시스템 자원 감지/시드/런ID/로그/디바이스/AMP 정책 세팅
    - HPO/게이트 초기화
    - context 패키징
    (※ target_profile 의존성 전면 제거)
    """
    # 0) LLM 파싱 → 내부 필드 매핑
    if state.user_query:
        parsed = _safe_parse_user_query(state.user_query)

        # 예시/공란 정리
        for key in ("base_model", "dataset", "dataset_version"):
            parsed[key] = _as_none(_strip_example(parsed.get(key)))

        # precision만 상태에 반영 (device는 YAML/hardware에서만 결정)
        if parsed.get("precision") and not getattr(state, "precision", None):
            state.precision = parsed["precision"]

        # 기타 필드 매핑
        for k in ("intent", "model_variant", "base_model", "resume", "notes"):
            v = parsed.get(k)
            if v is not None and getattr(state, k, None) in (None, False):
                setattr(state, k, v)

        # dataset_version 우선 결정
        if not state.dataset_version:
            state.dataset_version = parsed.get("dataset_version") or parsed.get("dataset")

        print("[LLM 분류 결과]", json.dumps(parsed, ensure_ascii=False, indent=2))

    # 1) training.yaml 로드
    config_path = state.config_path or "configs/training.yaml"
    train_cfg = _load_yaml(config_path) if Path(config_path).exists() else {}
    state.train_cfg = train_cfg

    # 2) 시스템 자원 감지
    sysinfo = _detect_system_env()

    # 3) 디바이스/정밀도 정책 (target_profile 제거)
    hw = (train_cfg.get("hardware") or {})
    # 디바이스는 YAML의 hardware.device가 최우선, 없으면 자동결정
    device = _resolve_device(hw.get("device"), sysinfo)

    # mixed precision 기본값 (없으면 True)
    mp_cfg = bool(hw.get("mixed_precision", True))

    # state.precision이 명시되면 이를 최우선으로 반영
    prec = (state.precision or "").lower() if state.precision else ""
    if prec == "fp32":
        mp_cfg = False
    elif prec == "fp16":
        mp_cfg = True
    # int8은 export 단계에서 처리

    amp_dtype = "fp16" if mp_cfg else "fp32"
    if torch:
        try:
            cudnn_benchmark = bool(hw.get("cudnn_benchmark", True))
            if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = cudnn_benchmark  # type: ignore[attr-defined]
        except Exception:
            pass

    # 4) 시드/런 ID/로그 경로
    seed = int(state.seed or (train_cfg.get("train", {}).get("seed") if train_cfg.get("train") else 42) or 42)
    state.seed = seed
    _seed_everything(seed)

    prefix = f"{state.run_name}_" if state.run_name else ""
    run_id = getattr(state, "context", {}).get("run_id") if state.context else None
    if not run_id:
        run_id = f"{prefix}{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    project_name = (train_cfg.get("meta") or {}).get("project_name", "yolo-training")
    base_log_dir = Path((train_cfg.get("meta") or {}).get("log_dir", f"./runs/{project_name}"))
    run_dir = Path(base_log_dir) / run_id
    _ensure_dir(run_dir)

    # 5) HPO / 게이트 초기화
    hpo_cfg = train_cfg.get("hpo") or {}
    if "enabled" not in hpo_cfg:
        hpo_cfg["enabled"] = False
    state.hpo = {**hpo_cfg, **(state.hpo or {})}

    gate_cfg = train_cfg.get("gate") or {}
    gate_cfg.setdefault("enabled", True)
    state.gate = {**gate_cfg, **(state.gate or {})}

    # 6) context 패키징 (target_profile, target_device 제거)
    state.context = {
        "project_name": project_name,
        "run_id": run_id,
        "seed": seed,
        "device": device,              # "cuda:0" | "cpu"
        "amp": mp_cfg,                 # True → fp16 학습 사용
        "amp_dtype": amp_dtype,        # "fp16"/"fp32"
        "log_dir": str(run_dir),
        "system_info": sysinfo,        # GPU/RAM/CPU 정보
        "env": {
            "working_dir": str(Path.cwd()),
            "python": os.sys.version.split()[0],
        },
        # 참고용 정보
        "intent": state.intent,
        "model_variant": state.model_variant,
        "precision": state.precision or ("fp16" if mp_cfg else "fp32"),
        "dataset_version": state.dataset_version,
        "base_model": state.base_model,
        "resume": state.resume,
    }

    # 7) 로그
    print(json.dumps({
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
            "hpo_enabled": state.hpo.get("enabled") if state.hpo else False,
            "precision": state.precision or ("fp16" if mp_cfg else "fp32"),
            "dataset_version": state.dataset_version,
            "model_variant": state.model_variant,
            "intent": state.intent,
        }
    }, ensure_ascii=False, indent=2))

    # 8) model_variant가 있으면 .pt 보정만 유지
    mv = _norm_str(state.model_variant)
    if mv:
        if not mv.endswith(".pt"):
            mv = mv + ".pt"
        cfg = state.train_cfg or {}
        tr = (cfg.get("train") or {}).copy()
        tr["model_name"] = mv  # ← YAML 기본도 함께 맞춰 둔다
        cfg["train"] = tr
        state.train_cfg = cfg

    return state
