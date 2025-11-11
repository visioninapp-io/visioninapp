# utils/model_normalizer.py
from __future__ import annotations

import os
import re
from typing import Optional


# YOLO 버전별 네이밍 규칙
# - v3 ~ v10 : "yolov{ver}{suffix}"
# - 11 이상  : "yolo{ver}{suffix}"  (v 없음)
# suffix 예: n, s, m, l, x, -seg, -cls, -obb 등
# 참고: Ultralytics 공식 문서 기준

_EXT_PATTERN = re.compile(
    r"\.(pt|onnx|engine|trt|torchscript|bin|xml)$"
)
_SUFFIX_PATTERN = re.compile(
    r"(-best|-last|-final)$"
)


def _strip_path_ext(name: str) -> str:
    # 경로 제거
    base = os.path.basename(name.strip())
    base = base.lower()

    # 공통 접두어 제거 (선택)
    if base.startswith("ultralytics-"):
        base = base[len("ultralytics-") :]

    # 확장자 제거
    base = _EXT_PATTERN.sub("", base)
    # best/last 같은 접미사 제거
    base = _SUFFIX_PATTERN.sub("", base)

    # 공백, 언더바 제거
    base = re.sub(r"[\s_]+", "", base)

    return base


def normalize_yolo_model_name(raw: Optional[str]) -> Optional[str]:
    """
    자유형 입력(raw)을 받아서 일관된 YOLO 모델 이름으로 정규화한다.

    규칙 요약:
    - "yolo5" / "yolov5"       -> "yolov5n"   (없으면 n 기본)
    - "yolov5s" / "yolo5s"     -> "yolov5s"
    - "yolo11" / "yolov11"     -> "yolo11n"
    - "yolov11n"               -> "yolo11n"
    - "yolo12x" / "yolov12x"   -> "yolo12x"
    - 알 수 없는 형태면: 확장자/공백만 정리해서 반환하거나, 매치 못하면 원본(cleaned) 반환.
    """
    if not raw:
        return None

    s = _strip_path_ext(raw)

    # yolo(v)?{num}{tail} 패턴 탐색
    m = re.search(r"yolo(v)?(\d{1,2})([a-z0-9\-]*)", s)
    if not m:
        # "yolo" 토큰이 없으면 건드리지 않고 clean 값 반환
        return s or None

    has_v, num_s, tail = m.groups()
    ver = int(num_s)

    # version sanity check
    if ver < 3 or ver > 30:
        # YOLO 버전 범위를 벗어나면 대충 yolo{num}{tail}로 정리만
        return f"yolo{ver}{tail}"

    # v 사용 여부 결정
    if ver <= 10:
        # v3 ~ v10 계열: yolov{ver}
        prefix = f"yolov{ver}"
    else:
        # 11 이상: yolo{ver}
        prefix = f"yolo{ver}"

    # tail 처리
    # 예: "", "n", "s", "x", "-seg", "-cls", "-obb" 등
    tail = tail or ""

    # scale (n/s/m/l/x) 없이 숫자만 온 경우엔 기본 n 부여
    # ex) "yolo5" -> "yolov5n", "yolo11" -> "yolo11n"
    # 이미 -seg/-cls 등 태스크 접미사만 있는 경우는 그대로 둔다.
    if tail == "":
        tail = "n"
    elif tail.startswith("-"):
        # "yolo11-seg" 같이 scale 없는 태스크 지정이면 n 추가
        tail = "n" + tail

    normalized = prefix + tail
    return normalized
