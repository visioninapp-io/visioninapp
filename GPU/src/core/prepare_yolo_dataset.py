# core/prepare_yolo_dataset.py
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# PyYAML은 requirements에 있으니 보통 import 가능
try:
    import yaml
except Exception:
    yaml = None  # 안전 폴백


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# YAML I/O 유틸
# -----------------------------
def _read_yaml(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists() or yaml is None:
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_yaml(p: Path, data: Dict[str, Any]) -> None:
    assert yaml is not None, "PyYAML이 필요합니다."
    p.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _normalize_names_from_yaml(d: Dict[str, Any]) -> Optional[List[str]]:
    """
    data.yaml에서 names는 list 또는 dict일 수 있음.
    dict라면 {idx:name} 형태를 인덱스 기준 정렬하여 list로 변환.
    """
    names = d.get("names")
    if names is None:
        return None
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        try:
            items = sorted(
                ((int(k), str(v)) for k, v in names.items()),
                key=lambda x: x[0],
            )
            return [v for _, v in items]
        except Exception:
            # 인덱스가 정수가 아닐 경우 키순으로 값만 사용
            return [str(v) for _, v in sorted(names.items(), key=lambda kv: kv[0])]
    return None


# -----------------------------
# 데이터 스캔/분할 유틸
# -----------------------------
def _pair_items(root: Path) -> List[Tuple[Path, Path]]:
    """
    root/images, root/labels 기준으로 이미지-라벨 쌍을 수집.
    이미 train/val/test 하위에 있는 파일은 '이미 분할된 데이터'로 보고 무시.
    라벨이 없으면 빈 파일을 생성(학습 파이프라인 호환 목적).
    """
    img_dir = root / "images"
    lbl_dir = root / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        return []

    pairs: List[Tuple[Path, Path]] = []
    for img in img_dir.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue

        rel = img.relative_to(img_dir)
        # 이미 train/val/test 아래에 있는 건 재분할 대상에서 제외
        if rel.parts and rel.parts[0] in ("train", "val", "test"):
            continue

        # 라벨 경로: 이미지와 동일한 상대경로 구조 유지
        if len(rel.parts) > 1:
            lbl_rel = rel.with_suffix(".txt")
            lbl = lbl_dir / lbl_rel
        else:
            lbl = lbl_dir / f"{img.stem}.txt"

        if not lbl.exists():
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("", encoding="utf-8")

        pairs.append((img, lbl))
    return pairs


def _extract_classes_from_labels(label_dir: Path) -> List[int]:
    """labels/* 내 모든 txt에서 class_id를 스캔해 정렬된 목록 반환."""
    class_ids = set()
    for txt in label_dir.rglob("*.txt"):
        try:
            for line in txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cls_id = int(line.split()[0])
                class_ids.add(cls_id)
        except Exception:
            continue
    return sorted(class_ids) if class_ids else [0]


def _auto_names_from_labels(root: Path) -> List[str]:
    """라벨에서 class_id를 스캔해 class0, class1 ... 이름을 자동 생성."""
    cids = _extract_classes_from_labels(root / "labels")
    return [f"class{i}" for i in cids]


def _copy_subset(root: Path, name: str, items: List[Tuple[Path, Path]], move_files: bool) -> None:
    """
    지정된 split(name)으로 이미지/라벨을 복사 또는 이동.
    이미 목적지와 동일한 경로인 경우(SameFile)에는 스킵.
    """
    img_dst = root / "images" / name
    lbl_dst = root / "labels" / name
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    copier = shutil.move if move_files else shutil.copy2

    for img, lbl in items:
        # 평탄화: 파일명 기준으로 저장 (필요시 rel 경로 보존 로직으로 확장 가능)
        dst_img = img_dst / img.name
        dst_lbl = lbl_dst / lbl.name

        # 동일 파일이면 스킵
        try:
            if img.resolve() != dst_img.resolve():
                copier(img, dst_img)
        except FileNotFoundError:
            # 원본이 이미 이동되었거나 없으면 무시
            continue

        try:
            if lbl.resolve() != dst_lbl.resolve():
                copier(lbl, dst_lbl)
        except FileNotFoundError:
            continue


# -----------------------------
# data.yaml 유효성 검사 (data.yaml 위치 기준 상대경로 존중)
# -----------------------------
def _has_ready_split_by_yaml_dir(yaml_path: Path, d: Optional[Dict[str, Any]]) -> bool:
    """
    data.yaml 파일이 있는 디렉터리를 기준으로 train/val/test 경로의 존재를 확인.
    값이 '../train/images' 같은 상대경로여도 정상 처리.
    labels 경로는 YOLO가 자동으로 images 경로 기반으로 해석하므로
    여기서는 이미지 디렉터리 존재만 확인.
    """
    if not d:
        return False

    yaml_dir = yaml_path.parent
    base_path = Path(d.get("path")) if d.get("path") else yaml_dir

    for k in ("train", "val", "test"):
        sub = d.get(k)
        if not sub:
            return False

        p = Path(sub)
        if not p.is_absolute():
            p = base_path / sub

        if not p.exists():
            return False

    return True


# -----------------------------
# 메인 함수
# -----------------------------
def prepare_yolo_dataset(
    root_dir: str,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    move_files: bool = False,
) -> str:
    """
    케이스 자동 판별:
      1) data.yaml이 있고, train/val/test 경로가 (data.yaml 위치 기준으로) 유효 → 그대로 사용
      2) data.yaml이 있으나 분할 정보가 없거나 경로가 존재하지 않음 → images/labels에서 분할 후 data.yaml 업데이트(기존 names 존중)
      3) data.yaml이 없음 → 분할 + names 자동 생성 + data.yaml 신규 생성

    추가 조건:
      - n >= 3: train/val/test 각각 최소 1장씩 배치.
        (예: 3장이면 train=1, val=1, test=1)
      - 각 폴더에 1장씩 넣은 뒤 남은 이미지를 splits 비율에 최대한 맞춰 분배.
      - n == 2: train=1, val=1, test=0
      - n == 1: train=1, val/test는 train 경로 재사용(fallback)

    반환: 최종 data.yaml 절대 경로
    """
    root = Path(root_dir)
    data_yaml_path = root / "data.yaml"
    exist_yaml = _read_yaml(data_yaml_path)

    # 1) 이미 유효한 data.yaml이면 그대로 사용
    if exist_yaml and _has_ready_split_by_yaml_dir(data_yaml_path, exist_yaml):
        return str(data_yaml_path.resolve())

    # 2/3) 분할 필요
    pairs = _pair_items(root)
    if not pairs:
        raise RuntimeError("유효한 images/labels 구조를 찾을 수 없습니다. (images/, labels/ 필요)")

    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    t, v, te = splits
    assert abs((t + v + te) - 1.0) < 1e-6, "splits 합이 1.0이어야 합니다."

    # ---------- 분할 규칙 ----------
    if n == 1:
        n_tr, n_va, n_te = 1, 0, 0
    elif n == 2:
        n_tr, n_va, n_te = 1, 1, 0
    else:
        # n >= 3: 각 split에 1장씩 먼저 할당
        base_tr = base_va = base_te = 1
        remaining = n - 3

        # 남은 이미지를 비율에 따라 분배
        # (t, v, te는 전체 비율이지만, 여기서는 상대 가중치로 사용)
        # 분배 후 합이 맞지 않으면 마지막은 test 쪽으로 보정
        total_ratio = t + v + te
        rt = t / total_ratio if total_ratio > 0 else 1 / 3
        rv = v / total_ratio if total_ratio > 0 else 1 / 3
        rte = te / total_ratio if total_ratio > 0 else 1 / 3

        add_tr = int(remaining * rt)
        add_va = int(remaining * rv)
        add_te = remaining - add_tr - add_va

        n_tr = base_tr + add_tr
        n_va = base_va + add_va
        n_te = base_te + add_te

        # 합 오차 보정 (이론상 필요 없지만 안전장치)
        diff = n - (n_tr + n_va + n_te)
        if diff != 0:
            # 남는 건 train에 몰아주고, 모자라면 train에서 조정(최소 1 유지)
            n_tr = max(1, n_tr + diff)

    # 실제 쌍 자르기
    train_pairs = pairs[:n_tr]
    val_pairs = pairs[n_tr:n_tr + n_va]
    test_pairs = pairs[n_tr + n_va:n_tr + n_va + n_te]

    # 복사/이동 (SameFile은 내부에서 회피)
    _copy_subset(root, "train", train_pairs, move_files)
    _copy_subset(root, "val", val_pairs, move_files)
    _copy_subset(root, "test", test_pairs, move_files)

    # names 결정: 기존 yaml에 있으면 우선 사용, 없으면 라벨에서 유추
    if exist_yaml:
        names = _normalize_names_from_yaml(exist_yaml) or _auto_names_from_labels(root)
    else:
        names = _auto_names_from_labels(root)
    nc = len(names)

    # ---------- data.yaml 생성/업데이트 ----------
    has_val = len(val_pairs) > 0
    has_test = len(test_pairs) > 0

    train_path = "images/train"
    val_path = "images/val" if has_val else train_path
    test_path = "images/test" if has_test else (val_path if has_val else train_path)

    out_yaml = {
        "path": str(root.resolve()),
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": nc,
        "names": names,
    }

    _write_yaml(data_yaml_path, out_yaml)
    return str(data_yaml_path.resolve())
