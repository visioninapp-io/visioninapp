# graph/training/nodes/load_dataset.py
from __future__ import annotations

import os
import shutil
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict

from graph.training.state import TrainState

try:
    import yaml
except Exception:
    yaml = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LBL_EXT = ".txt"


# ===================== 공용 유틸 =====================

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML 미설치: pip install pyyaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML 파일이 없습니다: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def _dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML 미설치: pip install pyyaml")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

def _list_images(dir_path: str | Path) -> List[Path]:
    d = Path(dir_path)
    files = []
    for ext in IMG_EXTS:
        files.extend(d.rglob(f"*{ext}"))
    return sorted(files)

def _label_path_from_image(img_path: Path, labels_root: Path) -> Path:
    # 관례: images/.../a/b.jpg → labels/.../a/b.txt
    parts = img_path.parts
    try:
        idx = parts.index("images")
        sub = Path(*parts[idx+1:]).with_suffix(LBL_EXT)
        return labels_root / sub
    except ValueError:
        # images/ 구조가 아닐 때: 동일 파일명 기준
        return labels_root / img_path.with_suffix(LBL_EXT).name

def _read_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    items: List[Tuple[int, float, float, float, float]] = []
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            items.append((cls, x, y, w, h))
        except Exception:
            continue
    return items

def _stratified_split(
    images: List[Path],
    labels_root: Path,
    num_classes: int,
    ratio: Tuple[float, float],
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """근사적 계층 분할(이미지당 대표 클래스 기준)."""
    random.seed(seed)
    buckets: Dict[int, List[Path]] = defaultdict(list)
    no_label_bucket: List[Path] = []

    for img in images:
        lbl = _label_path_from_image(img, labels_root)
        items = _read_yolo_label(lbl)
        if not items:
            no_label_bucket.append(img)
            continue
        major_cls = Counter([it[0] for it in items]).most_common(1)[0][0]
        if 0 <= major_cls < num_classes:
            buckets[major_cls].append(img)
        else:
            no_label_bucket.append(img)

    def split_list(lst: List[Path], r: Tuple[float, float]) -> Tuple[List[Path], List[Path]]:
        random.shuffle(lst)
        n_train = int(len(lst) * r[0])
        return lst[:n_train], lst[n_train:]

    train, val = [], []
    for _, imgs in buckets.items():
        t, v = split_list(imgs, ratio)
        train.extend(t); val.extend(v)
    t2, v2 = split_list(no_label_bucket, ratio)
    train.extend(t2); val.extend(v2)
    return train, val

def _plain_split(images: List[Path], ratio: Tuple[float, float], seed: int = 42) -> Tuple[List[Path], List[Path]]:
    random.seed(seed)
    images = images[:]
    random.shuffle(images)
    n_train = int(len(images) * ratio[0])
    return images[:n_train], images[n_train:]

def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def _mirror_split(
    split_imgs: List[Path],
    images_root_out: Path,
    labels_root_in: Path,
    labels_root_out: Path,
) -> None:
    for img in split_imgs:
        # 이미지
        try:
            rel_from_images = img.relative_to(img.parents[1]) if "images" in img.parts else img.name
        except Exception:
            rel_from_images = img.name
        dst_img = images_root_out / (rel_from_images if isinstance(rel_from_images, Path) else Path(rel_from_images))
        _symlink_or_copy(img, dst_img)

        # 라벨
        lbl_in = _label_path_from_image(img, labels_root_in)
        rel_lbl = dst_img.with_suffix(LBL_EXT).relative_to(images_root_out)
        dst_lbl = labels_root_out / rel_lbl
        if lbl_in.exists():
            _symlink_or_copy(lbl_in, dst_lbl)
        else:
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("", encoding="utf-8")

def _scan_max_class_id(labels_dir: Path) -> int:
    max_id = -1
    if not labels_dir.exists():
        return max_id
    for p in labels_dir.rglob(f"*{LBL_EXT}"):
        for item in _read_yolo_label(p):
            max_id = max(max_id, item[0])
    return max_id

def _count_images(dir_path: str | Path) -> int:
    return len(_list_images(dir_path))

def _basic_warnings(n_train: int, n_val: int, dist_tr: Counter, dist_va: Counter, names: List[str]) -> List[str]:
    warns: List[str] = []
    if n_train == 0 or n_val == 0:
        warns.append("train/val 중 한쪽 이미지가 0개입니다.")
    all_classes = set(range(len(names)))
    seen = set(dist_tr.keys()) | set(dist_va.keys())
    missing = sorted(list(all_classes - seen))
    if missing:
        warns.append(f"아래 클래스에 라벨이 없습니다: {', '.join(names[i] for i in missing)}")
    total_tr = sum(dist_tr.values())
    if total_tr > 0:
        top_cls, top_cnt = max(dist_tr.items(), key=lambda x: x[1])
        if top_cnt / total_tr > 0.9:
            warns.append(f"train 라벨의 90% 이상이 '{names[top_cls]}' 한 클래스에 편중되었습니다.")
    return warns

def _collect_stats(train_dir: str, val_dir: str, names: List[str]) -> Dict[str, Any]:
    n_train = _count_images(train_dir)
    n_val   = _count_images(val_dir)

    label_root_train = Path(train_dir).parent.parent / "labels" / "train"
    label_root_val   = Path(val_dir).parent.parent / "labels" / "val"

    def cls_dist(lbl_root: Path) -> Counter:
        c = Counter()
        if not lbl_root.exists():
            return c
        for p in lbl_root.rglob(f"*{LBL_EXT}"):
            for t in _read_yolo_label(p):
                c[t[0]] += 1
        return c

    dist_tr = cls_dist(label_root_train)
    dist_va = cls_dist(label_root_val)

    def to_named(cnt: Counter) -> Dict[str, int]:
        return {names[k] if k < len(names) else f"cls_{k}": v for k, v in sorted(cnt.items())}

    return {
        "num_images": {"train": n_train, "val": n_val},
        "labels_per_class": {
            "train": to_named(dist_tr),
            "val":   to_named(dist_va),
        },
        "warnings": _basic_warnings(n_train, n_val, dist_tr, dist_va, names),
    }


# ===================== 핵심 노드 =====================

def load_dataset(state: TrainState) -> TrainState:
    """
    - data.yaml이 있으면 그대로 사용
    - 없으면 raw_images_dir/raw_labels_dir에서 자동 분할 후 data.yaml 생성
    - 결과를 state.data / state.dataset_stats / state.paths에 기록
    """
    train_cfg: Dict[str, Any] = state.train_cfg or {}
    data_cfg: Dict[str, Any] = train_cfg.get("data", {}) if train_cfg else {}

    # 1) data.yaml 경로가 명시되어 있으면 그대로 사용
    yaml_path = data_cfg.get("yaml_path")
    if state.dataset_version is not None:
        yaml_path = f"data/datasets/{state.dataset_version}/data.yaml"
    if yaml_path and Path(yaml_path).exists():
        data_yaml = _load_yaml(yaml_path)
        for k in ("train", "val", "names"):
            if k not in data_yaml:
                raise ValueError(f"data.yaml에 '{k}' 항목이 없습니다: {yaml_path}")

        nc = int(data_yaml.get("nc", len(data_yaml["names"])))
        names = list(data_yaml["names"])
        train_dir = str((Path(yaml_path).parent / Path(data_yaml["train"])).resolve())
        val_dir   = str((Path(yaml_path).parent / Path(data_yaml["val"])).resolve())

        stats = _collect_stats(train_dir, val_dir, names)

        state.data = {
            "yaml_path": str(Path(yaml_path).resolve()),
            "train_dir": train_dir,
            "val_dir": val_dir,
            "names": names,
            "nc": nc,
        }
        state.paths = (state.paths or {}) | {"dataset_root": str(Path(yaml_path).parent.resolve())}
        state.dataset_stats = stats
        print(f"[load_dataset] 기존 data.yaml 사용: {yaml_path}")
        return state

    # 2) data.yaml이 없으면 자동 분할/생성
    raw_images_dir = state.raw_images_dir
    raw_labels_dir = state.raw_labels_dir
    if not raw_images_dir or not raw_labels_dir:
        raise ValueError("data.yaml이 없고 원천 폴더도 없습니다. raw_images_dir, raw_labels_dir를 지정하거나 data.yaml을 제공하세요.")

    # 클래스 이름
    names: List[str] = data_cfg.get("names") or getattr(state, "names", None) or []
    if not names:
        max_cls = _scan_max_class_id(Path(raw_labels_dir))
        names = [f"class_{i}" for i in range(max(0, max_cls + 1))]
    nc = len(names)
    if nc == 0:
        raise ValueError("클래스 이름이 비어 있습니다. data.names 또는 state.names를 지정하세요.")

    # 버전/출력 루트
    dataset_version = state.dataset_version or "dataset@auto"
    root = Path(f"data/datasets/{dataset_version}")
    images_out = root / "images"
    labels_out = root / "labels"
    for sub in ("train", "val"):
        (images_out / sub).mkdir(parents=True, exist_ok=True)
        (labels_out / sub).mkdir(parents=True, exist_ok=True)

    # 모든 원천 이미지
    src_images = _list_images(raw_images_dir)
    if not src_images:
        raise ValueError(f"원천 이미지가 없습니다: {raw_images_dir}")

    # 분할 비율
    ratio_list = data_cfg.get("split_ratio") or getattr(state, "split_ratio", None) or [0.8, 0.2]
    if isinstance(ratio_list, (list, tuple)) and len(ratio_list) == 2:
        ratio = (float(ratio_list[0]), float(ratio_list[1]))
    else:
        ratio = (0.8, 0.2)
    if abs(sum(ratio) - 1.0) > 1e-6:
        s = sum(ratio_list)
        ratio = (float(ratio_list[0]) / s, float(ratio_list[1]) / s)

    # (선택) 계층 분할
    stratified = bool(getattr(state, "stratified_split", False))
    labels_root_in = Path(raw_labels_dir)
    seed = int(getattr(state, "seed", 42))
    if stratified:
        train_imgs, val_imgs = _stratified_split(src_images, labels_root_in, num_classes=nc, ratio=ratio, seed=seed)
    else:
        train_imgs, val_imgs = _plain_split(src_images, ratio, seed=seed)

    # 물리적 링크/복사 구성
    _mirror_split(train_imgs, images_out / "train", labels_root_in, labels_out / "train")
    _mirror_split(val_imgs,   images_out / "val",   labels_root_in, labels_out / "val")

    # data.yaml 생성
    data_yaml = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc": int(nc),
        "names": list(names),
    }
    yaml_path = root / "data.yaml"
    _dump_yaml(data_yaml, yaml_path)

    # 통계
    stats = _collect_stats(str(images_out / "train"), str(images_out / "val"), names)

    state.data = {
        "yaml_path": str(yaml_path.resolve()),
        "train_dir": str((images_out / "train").resolve()),
        "val_dir":   str((images_out / "val").resolve()),
        "names": names,
        "nc": int(nc),
    }
    state.paths = (state.paths or {}) | {"dataset_root": str(root.resolve())}
    state.dataset_stats = stats

    print(f"[load_dataset] 자동 분할 완료 → {yaml_path}")
    return state
