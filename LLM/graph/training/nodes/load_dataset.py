# graph/training/nodes/load_dataset.py
from __future__ import annotations

import os
from typing import Any, Dict

from graph.training.state import TrainState

# boto3는 EC2에서 S3 접근용 (Credentials는 .env / 환경변수로 세팅)
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except Exception:
    boto3 = None
    ClientError = Exception  # type: ignore
    NoCredentialsError = Exception  # type: ignore


def _use_s3() -> bool:
    return os.getenv("USE_S3_STORAGE", "false").lower() == "true"


def _get_bucket() -> str:
    # .env: S3_BUCKET=visioninapp-bucket
    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("S3_BUCKET 환경변수가 설정되어 있지 않습니다.")
    return bucket


def _resolve_dataset_path(state: TrainState) -> str:
    """
    main.py에서 전달한 dataset_path를 사용.
    - 우선순위: state.dataset_path -> state.dataset_version
    - 결과 예: 'dog_sample'
    """
    # TrainState에 dataset_path 필드를 추가했거나, dataset_version을 재활용한다고 가정
    dataset_path = getattr(state, "dataset_path", None) or state.dataset_version
    if not dataset_path:
        raise ValueError("dataset_path (또는 dataset_version)가 설정되어 있지 않습니다.")
    return str(dataset_path).strip().strip("/")


def load_dataset(state: TrainState) -> TrainState:
    """
    EC2 역할:
    - S3에 models/{dataset_path}/data.yaml 이 존재하는지만 확인
    - 존재 여부를 state.data['yaml_path']에 True/False로 기록
    - 여기서는 실제 학습/다운로드/분할 절대 수행하지 않음
    """
    dataset_path = _resolve_dataset_path(state)
    use_s3 = _use_s3()

    # 기본 결과 구조 초기화
    data_info: Dict[str, Any] = {
        "dataset_path": dataset_path,
        "yaml_path": False,      # default: 없음
        "s3_checked": False,
        "s3_key": None,
        "bucket": None,
    }

    if use_s3:
        if boto3 is None:
            raise RuntimeError("boto3가 설치되어 있지 않습니다. pip install boto3 후 다시 시도하세요.")

        bucket = _get_bucket()
        s3_key = f"datasets/{dataset_path}/data.yaml"
        data_info["bucket"] = bucket
        data_info["s3_key"] = s3_key

        print(f"[load_dataset] S3에서 data.yaml 존재 여부 확인: s3://{bucket}/{s3_key}")

        s3 = boto3.client("s3")

        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
            # 객체가 존재하면 예외 없이 통과
            data_info["yaml_path"] = True
            data_info["s3_checked"] = True
            print(f"[load_dataset] ✅ data.yaml 존재 확인됨: s3://{bucket}/{s3_key}")
        except NoCredentialsError:
            # Credentials 설정 문제
            print("[load_dataset] ❌ AWS Credentials를 찾을 수 없습니다. (.env 설정 확인)")
            data_info["s3_checked"] = False
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code == "404":
                print(f"[load_dataset] ⚠️ data.yaml 없음: s3://{bucket}/{s3_key}")
                data_info["s3_checked"] = True
            else:
                print(f"[load_dataset] ⚠️ S3 head_object 오류: {e}")
                data_info["s3_checked"] = False
    else:
        # USE_S3_STORAGE=false 인 경우: S3 체크 스킵, 단순 경로 정보만 기록
        print("[load_dataset] USE_S3_STORAGE != true, S3 확인 없이 통과합니다.")
        # 여기서는 로컬에서 models/{dataset_path}/data.yaml을 확인하는 로직을
        # 추가할 수도 있지만, 현재 요구사항 기준으로는 생략.

    # state에 결과 반영
    state.data = data_info
    state.dataset_stats = {
        "checked_via": "s3" if use_s3 else "none",
        "exists": bool(data_info["yaml_path"]),
    }

    # paths에는 참고용으로 dataset_root만 남김
    paths = state.paths or {}
    paths["dataset_root"] = f"models/{dataset_path}"
    state.paths = paths

    return state
