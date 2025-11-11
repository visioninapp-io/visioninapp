import os
import pathlib
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from boto3.s3.transfer import TransferConfig
from dotenv import load_dotenv

# .env 로드 (현재 실행 디렉토리 기준 또는 상위에서 자동 탐색)
load_dotenv()

# boto3 S3 클라이언트 - 명시적으로 자격 증명 주입
_s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"),
)

def _parse_s3(s3_uri: str):
    assert s3_uri.startswith("s3://")
    p = s3_uri[5:]
    bucket, _, key = p.partition("/")
    return bucket, key

def download_s3(s3_uri: str, local_path: str):
    b, k = _parse_s3(s3_uri)
    pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    _s3.download_file(b, k, local_path)

def upload_s3(local_path: str, bucket: str, key: str):
    _s3.upload_file(local_path, bucket, key)

def untar(src_tar: str, dst_dir: str):
    pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(src_tar, "r:*") as tf:
        tf.extractall(dst_dir)

def download_s3_folder(prefix: str, dataset_name: str,
                       workers: int = 8,
                       multipart_chunk_mb: int = 8) -> str:
    """
    S3 'prefix' 이하의 모든 객체를 로컬 동일 구조로 다운로드.
    이미 존재하는 파일은 S3 ContentLength와 같으면 스킵.
    - prefix 예: "datasets/myset/" 또는 "datasets/myset"
    - 반환: 로컬 저장 루트 디렉토리 경로
    """
    bucket = os.getenv("S3_BUCKET", "visioninapp-bucket")
    local_root = os.getenv("LOCAL_DATA_ROOT", "data/datasets")
    local_dir = os.path.join(local_root, dataset_name)

    prefix = prefix.strip("/")
    norm_prefix = prefix + "/"

    cfg = TransferConfig(
        multipart_threshold=multipart_chunk_mb * 1024 * 1024,
        multipart_chunksize=multipart_chunk_mb * 1024 * 1024,
        max_concurrency=workers,
        use_threads=True,
    )

    paginator = _s3.get_paginator("list_objects_v2")
    objects: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=norm_prefix):
        contents = page.get("Contents")
        if not contents:
            continue
        for obj in contents:
            key = obj["Key"]
            if key.endswith("/") or key == norm_prefix:
                continue
            objects.append((key, obj["Size"]))

    os.makedirs(local_dir, exist_ok=True)

    def _should_skip(local_path: str, size: int) -> bool:
        try:
            return os.path.exists(local_path) and os.path.getsize(local_path) == size
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for key, size in objects:
            rel_posix = key[len(norm_prefix):] if key.startswith(norm_prefix) else key
            rel_native = rel_posix.replace("/", os.sep)
            local_path = os.path.join(local_dir, rel_native)

            pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            if _should_skip(local_path, size):
                continue

            futs.append(ex.submit(
                _s3.download_file,
                bucket, key, local_path,
                ExtraArgs={}, Config=cfg
            ))

        for f in as_completed(futs):
            f.result()

    return local_dir
