from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional, List

# Ultralytics (optional)
try:
    from ultralytics import YOLO  # type: ignore
    _ULTRA = True
except Exception:
    _ULTRA = False


# --- 이름 정규화/별칭 ---------------------------------------------------------

def _normalize_to_no_v(name: str) -> str:
    """
    'yolov11n.pt' 같은 이름을 프로젝트 표준인 'yolo11n.pt'로 변환.
    이미 'yolo11n.pt' 형식이면 그대로 둔다.
    """
    # yolo + v + 숫자 → yolo + 숫자
    return re.sub(r"^yolov(?=\d)", "yolo", name.strip())


def _alias_with_v(name_no_v: str) -> str:
    """
    'yolo11n.pt' → 'yolov11n.pt' (허브/환경에 따라 v가 필요한 경우 대비).
    이미 yolov* 이면 그대로 반환.
    """
    if re.match(r"^yolov\d", name_no_v):
        return name_no_v
    return re.sub(r"^yolo(?=\d)", "yolov", name_no_v)


# --- 경로 유틸 ---------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _models_root(state=None) -> Path:
    # 버전 디렉토리 제거: 모든 기본 가중치는 여기로 모음
    return _ensure_dir(Path("models/yolo"))

def _target_path(state, model_name_no_v: str) -> Path:
    # 최종 저장 경로: models/yolo/<파일명>
    return _models_root(state) / model_name_no_v

def _exists(fp: Optional[Path]) -> bool:
    return bool(fp and fp.exists() and fp.is_file())

def _copy_if_exists(src: Path, dst: Path) -> Optional[Path]:
    try:
        if _exists(src):
            _ensure_dir(dst.parent)
            if dst.exists():
                dst.unlink()
            dst.write_bytes(src.read_bytes())
            return dst
    except Exception:
        pass
    return None


# --- 캐시 후보 경로 -----------------------------------------------------------

def _default_cache_candidates(model_name: str) -> List[Path]:
    """
    OS별로 흔한 캐시 위치를 폭넓게 커버.
    """
    home = Path.home()
    cands: List[Path] = [
        # Linux/macOS 공통
        home / ".cache" / "ultralytics" / "models" / model_name,
        home / ".cache" / "ultralytics" / model_name,
        home / ".cache" / "torch" / "hub" / "checkpoints" / model_name,
        # macOS 전용 캐시 위치
        home / "Library" / "Caches" / "ultralytics" / model_name,
        home / "Library" / "Caches" / "torch" / "hub" / "checkpoints" / model_name,
    ]
    # Windows
    appdata = os.getenv("APPDATA")           # C:\Users\<User>\AppData\Roaming
    localapp = os.getenv("LOCALAPPDATA")     # C:\Users\<User>\AppData\Local
    if appdata:
        cands += [
            Path(appdata) / "Ultralytics" / "models" / model_name,
            Path(appdata) / "Ultralytics" / model_name,
            Path(appdata) / "torch" / "hub" / "checkpoints" / model_name,
        ]
    if localapp:
        cands += [
            Path(localapp) / "Ultralytics" / "models" / model_name,
            Path(localapp) / "Ultralytics" / model_name,
            Path(localapp) / "torch" / "hub" / "checkpoints" / model_name,
        ]
    return cands


# --- YOLO 객체에서 실제 체크포인트 경로 추출 ---------------------------------

def _extract_ckpt_path_from_model(obj) -> Optional[Path]:
    """
    YOLO 객체 내부에서 실제 가중치 파일 경로를 최대한 추출.
    버전별 내부 속성이 달라질 수 있으므로 여러 후보를 시도.
    """
    cand_attrs = [
        "ckpt_path",   # 자주 쓰이는 속성
        "pt_path",
        "weights",     # 경로 혹은 리스트/튜플일 수 있음
        "model",       # model 객체 내부에 경로 속성이 있을 수 있음
    ]
    for attr in cand_attrs:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            # 문자열/Path
            if isinstance(val, (str, Path)):
                p = Path(val)
                if _exists(p):
                    return p
            # 리스트/튜플
            if isinstance(val, (list, tuple)) and val:
                p = Path(val[0])
                if _exists(p):
                    return p
            # model 내부 재탐색
            if attr == "model" and val is not None:
                for sub in ("ckpt_path", "pt_path", "yaml_file", "weights"):
                    if hasattr(val, sub):
                        subval = getattr(val, sub)
                        if isinstance(subval, (str, Path)):
                            p = Path(subval)
                            if _exists(p):
                                return p
                        if isinstance(subval, (list, tuple)) and subval:
                            p = Path(subval[0])
                            if _exists(p):
                                return p
    return None


# --- 메인 함수 ----------------------------------------------------------------

def ensure_weight_local(state, model_name: str) -> Path:
    """
    원하는 허브 모델명(ex. 'yolo11n.pt')을
    1) models/yolo/ 아래에서 먼저 찾고
    2) 없으면 OS 캐시에서 탐색하여 복사
    3) 캐시에 없으면 YOLO(model_name)로 허브 다운로드(우선 no-v 이름, 실패 시 v-삽입 별칭)
    4) YOLO 객체에서 실제 ckpt 경로를 뽑아 models/yolo/로 복사
    5) 그래도 실패하면 홈 전체 rglob로 (no-v / v-삽입 양쪽) 검색

    최종적으로 models/yolo/<model_name_no_v> 경로(Path)를 반환.
    """
    raw = model_name.strip()
    name_no_v = _normalize_to_no_v(raw)      # 프로젝트 표준 파일명
    name_with_v = _alias_with_v(name_no_v)   # 허브/환경별 호환용

    target = _target_path(state, name_no_v)

    # 1) 우리 저장소에 이미 있으면 그대로 반환
    if _exists(target):
        return target

    # 2) OS 캐시에서 먼저 복사 (no-v → with-v 순서)
    for nm in (name_no_v, name_with_v):
        for cand in _default_cache_candidates(nm):
            hit = _copy_if_exists(cand, target)
            if hit:
                return hit

    # 3) 허브 다운로드 (환경 설명 상 no-v 이름을 우선 사용)
    if not _ULTRA:
        raise FileNotFoundError(
            f"Ultralytics not installed and local weight missing: {target}\n"
            f"→ pip install ultralytics 후 다시 시도하세요."
        )

    model_obj = None
    # 3-1) no-v 이름으로 먼저 시도
    try:
        model_obj = YOLO(name_no_v)
    except Exception:
        # 3-2) 실패 시 v-삽입 별칭으로 한번 더
        try:
            model_obj = YOLO(name_with_v)
        except Exception as e:
            raise FileNotFoundError(
                f"허브 다운로드 실패: '{name_no_v}' 및 '{name_with_v}' 시도\n"
                f"에러: {e}"
            )

    # 4) YOLO 객체에서 실제 ckpt 경로를 직접 추출해 복사(가장 신뢰도 높음)
    ckpt = _extract_ckpt_path_from_model(model_obj)
    if ckpt and _copy_if_exists(ckpt, target):
        return target

    # 4-1) 혹시라도 캐시에만 놓였으면 다시 캐시 후보를 체크
    for nm in (name_no_v, name_with_v):
        for cand in _default_cache_candidates(nm):
            hit = _copy_if_exists(cand, target)
            if hit:
                return hit

    # 5) 홈 전체에서 교차 이름으로 광역 검색
    try:
        names_to_search = {name_no_v, name_with_v}
        for nm in names_to_search:
            for p in Path.home().rglob(nm):
                hit = _copy_if_exists(p, target)
                if hit:
                    return hit
    except Exception:
        pass

    # 그래도 실패
    raise FileNotFoundError(
        "허브 다운로드가 되었을 수 있으나 가중치 파일을 찾지 못했습니다.\n"
        f"다음 경로로 수동 배치하세요: '{target}'\n"
        f"또는 캐시(~/.cache/ultralytics, %APPDATA%/Ultralytics, %LOCALAPPDATA%/Ultralytics, "
        "…/torch/hub/checkpoints) 경로를 확인하세요."
    )
