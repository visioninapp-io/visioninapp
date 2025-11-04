import os, pathlib, tarfile
import boto3

_s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"))

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
