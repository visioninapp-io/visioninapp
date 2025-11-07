# graph/training/state/train_state.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TrainState(BaseModel):
    """
    전체 학습 그래프에서 공통으로 사용하는 상태 구조체(State)
    각 노드는 이 state를 입력/출력으로 받아 필요한 필드를 채워 넣는다.
    """
    model_config = {"protected_namespaces": ()}
    # -------------------- 사용자 입력 / 초기 파라미터 --------------------
    user_query: Optional[str] = None               # LLM이 파싱할 자연어 요청
    config_path: Optional[str] = None              # training.yaml 경로
    dataset_version: Optional[str] = None          # ex) "dataset@1.0.0"
    base_model: Optional[str] = None               # ex) "model@1.0.0.pt"
    resume: bool = False                           # 이전 학습 이어하기 여부
    seed: int = 42                                 # 랜덤 시드
    run_name: Optional[str] = None                 # 실험 이름 (폴더 prefix)
    model_variant: Optional[str] = None            # ex) "yolov12m"
    precision: Optional[str] = None                # fp16 / fp32 / int8
    intent: Optional[str] = None                   # train_model / retrain 등

    # -------------------- 환경 / 설정 --------------------
    context: Optional[Dict[str, Any]] = None       # init_context에서 생성한 런 정보
    train_cfg: Optional[Dict[str, Any]] = None     # training.yaml 파싱 결과
    target_profile: Optional[Dict[str, Any]] = None  # 배포 대상 디바이스 설정
    hpo: Optional[Dict[str, Any]] = None           # 하이퍼파라미터 탐색 설정
    gate: Optional[Dict[str, Any]] = None          # 게이트 설정(성능 기준 등)
    train_overrides: Optional[Dict[str, Any]] = None # 하이퍼파라미터 값
    force_hpo: Optional[bool] = False

    # -------------------- 데이터셋 --------------------
    data: Optional[Dict[str, Any]] = None          # data.yaml 정보 및 names, nc
    dataset_stats: Optional[Dict[str, Any]] = None # 클래스 분포, 경고 등
    paths: Optional[Dict[str, str]] = None         # 경로 모음 (dataset_root 등)
    raw_images_dir: Optional[str] = None           # 원본 이미지 경로
    raw_labels_dir: Optional[str] = None           # 원본 라벨 경로
    stratified_split: bool = False                 # 계층 분할 여부

    # -------------------- 학습 / 검증 / 결과 --------------------
    model_path: Optional[str] = None               # 학습 완료 모델 경로(.pt)
    export_path: Optional[str] = None              # export된 모델 경로(.onnx/.engine)
    metrics: Optional[Dict[str, Any]] = None       # 검증 결과 (mAP 등)
    regression: Optional[Dict[str, Any]] = None    # 이전 대비 회귀 비교 결과
    gate_result: Optional[str] = None              # PASS / FAIL / WARN 등
    last_ckpt: Optional[str] = None                # 마지막 체크포인트

    # -------------------- HPO / 튜닝 --------------------
    hpo_trials: Optional[List[Dict[str, Any]]] = None  # 시도된 하이퍼파라미터 목록
    best_trial: Optional[Dict[str, Any]] = None        # 최고 성능 trial 요약

    # -------------------- 상태 / 메타 --------------------
    action: Optional[str] = None
    registry_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None                    # 오류 메시지
    notes: Optional[str] = None                    # LLM이 남긴 설명/의도
    timestamp: Optional[str] = None                # 마지막 갱신 시간
    job_id: Optional[str] = None