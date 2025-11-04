# graph/training/nodes/load_model.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from graph.training.state import TrainState

# optional: 울트라리틱스가 없더라도 동작(메타만 세팅)하도록
try:
    from ultralytics import YOLO  # type: ignore
    _ULTRA = True
except Exception:
    YOLO = None  # type: ignore
    _ULTRA = False


def _get_bool(d: Dict[str, Any], key: str, default: bool = False) -> bool:
    v = (d or {}).get(key, default)
    return bool(v) if v is not None else default

def _normalize_id(x):
    return None if (isinstance(x, str) and x.strip().lower() in {"null", "none"}) else x

def _resolve_mode_and_weights(state: TrainState) -> Dict[str, Any]:
    """
    decide_mode와 동일 기준으로 'fresh/finetune/resume' 판단하고,
    가중치 파일(또는 모델명)을 고른다.
    - fresh: training.yaml.train.model_name
    - finetune: state.base_model or training.yaml.resume.finetune_base
    - resume: training.yaml.resume.last_ckpt (반드시 존재해야 함)
    """
    cfg: Dict[str, Any] = state.train_cfg or {}
    train_cfg = cfg.get("train") or {}
    resume_cfg = cfg.get("resume") or {}

    base_model = _normalize_id(getattr(state, "base_model", None))
    finetune_base = _normalize_id(resume_cfg.get("finetune_base"))
    last_ckpt = _normalize_id(resume_cfg.get("last_ckpt"))

    # 판정
    has_dataset = bool(state.dataset_version) or bool((cfg.get("data") or {}).get("yaml_path"))
    wants_resume = bool(state.resume) or bool(resume_cfg.get("enable")) or bool(last_ckpt)
    has_base = bool(base_model) or bool(finetune_base)

    if not has_dataset:
        return {"mode": "error_missing_dataset", "weights": None}

    if wants_resume:
        mode = "resume"
        weights = resume_cfg.get("last_ckpt")
        # resume은 실제 체크포인트 존재 여부를 엄격히 확인
        if not weights or not Path(str(weights)).exists():
            raise FileNotFoundError(f"[load_model] resume 모드인데 체크포인트가 없습니다: {weights}")
    elif has_base:
        mode = "finetune"
        weights = base_model or finetune_base
        # 파인튜닝은 로컬 파일이 없어도 허용(예: 허브/레지스트리에서 당겨오는 경우)
    else:
        mode = "fresh"
        weights = train_cfg.get("model_name")  # 예: yolov12n.pt

    return {"mode": mode, "weights": weights}


def _pick_device_and_dtype(state: TrainState) -> Dict[str, Any]:
    """
    init_context에서 정리해둔 실행 디바이스/정밀도 사용.
    - device: state.context["device"]  (예: "cuda:0" 또는 "cpu")
    - dtype:  state.context["amp_dtype"] ("fp16"/"fp32")
    """
    ctx = state.context or {}
    device = ctx.get("device") or "cpu"
    amp_dtype = ctx.get("amp_dtype") or "fp32"
    return {"device": device, "amp_dtype": amp_dtype}


def load_model(state: TrainState) -> TrainState:
    """
    - 학습 모드(fresh/finetune/resume) 결정
    - 가중치 경로/이름 확정
    - (가능하면) YOLO 모델 로드 시도
    - 결과 요약을 state.context["load_model"]에 기록
    - state.model_path에 '사용할 가중치 식별자'를 저장
    """
    # 1) 모드/가중치 확정
    sel = _resolve_mode_and_weights(state)
    mode = sel["mode"]
    weights = sel["weights"]

    if mode == "error_missing_dataset":
        # 데이터셋 없으면 여기서도 조용히 종료(그래프 설계에 따라 END로 분기)
        state.context = (state.context or {})
        state.context["load_model"] = {
            "mode": mode,
            "reason": "dataset missing",
            "loaded": False,
        }
        return state

    # 2) 디바이스/정밀도
    hw = _pick_device_and_dtype(state)
    device = hw["device"]
    amp_dtype = hw["amp_dtype"]  # 문자열 메모용

    # 3) freeze 옵션(파인튜닝 시 선택적으로 사용)
    cfg: Dict[str, Any] = state.train_cfg or {}
    resume_cfg = cfg.get("resume") or {}
    freeze_backbone = bool(resume_cfg.get("freeze_backbone", False))

    # 4) 실제 모델 로딩 시도 (없어도 실패하지 않음)
    loaded = False
    model_summary: Dict[str, Any] = {}
    model_id = str(weights) if weights else None

    if _ULTRA and model_id:
        try:
            model = YOLO(model_id)  # 파일이 없으면 자동 다운로드 가능한 이름(yolov8n.pt 등) 처리
            # 장치 이동 (ultralytics는 .to 사용 가능)
            try:
                model.to(device)
            except Exception:
                pass

            # fp16 학습 의사결정은 학습 루프에서 amp='auto'로 다루는 편이 안전하므로
            # 여기서는 기록만 남긴다.
            if freeze_backbone:
                # 최소 구현: 전체 동결 대신 헤드 제외 전부 동결 등은 프레임워크별 상이 → 기록만
                model_summary["frozen"] = True
            loaded = True
        except Exception as e:
            model_summary["load_error"] = repr(e)

    # 5) 상태 기록
    ctx = state.context or {}
    ctx["load_model"] = {
        "mode": mode,                 # fresh / finetune / resume
        "weights": model_id,          # 가중치 식별자(파일/허브이름)
        "device": device,             # cuda:0 / cpu
        "amp_dtype": amp_dtype,       # fp16 / fp32 (메모용)
        "freeze_backbone": freeze_backbone,
        "ultralytics_available": _ULTRA,
        "loaded": loaded,
        **model_summary,
    }
    state.context = ctx

    # 모델 파일/이름을 후속 노드가 참조할 수 있게 저장
    state.model_path = model_id

    print(f"[load_model] mode={mode} weights={model_id} device={device} amp={amp_dtype} loaded={loaded}")
    if not loaded and model_summary.get("load_error"):
        print(f"[load_model] (non-fatal) load error: {model_summary['load_error']}")

    return state
