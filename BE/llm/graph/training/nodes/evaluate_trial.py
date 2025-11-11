from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llm.graph.training.state import TrainState


# -------------------------------
# Helpers
# -------------------------------

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("%", "")
        return float(s)
    except Exception:
        return None


_METRIC_ALIASES: Dict[str, List[str]] = {
    "mAP50-95": [
        "mAP50-95","map50-95","metrics/mAP50-95(B)","metrics/mAP50-95",
        "box/mAP50-95","bbox_map50_95","map_50_95","seg/mAP50-95",
        # 흔한 축약
        "map", "maps"
    ],
    "mAP50": [
        "mAP50","map50","metrics/mAP50","box/mAP50","bbox_map50","seg/mAP50",
        "map50(B)"  # 일부 CSV 표기
    ],
    "precision": [
        "precision","metrics/precision","P","box/P","seg/P",
        "metrics/precision(B)","precision(B)","mp"  # ← 추가
    ],
    "recall": [
        "recall","metrics/recall","R","box/R","seg/R",
        "metrics/recall(B)","recall(B)","mr"  # ← 추가
    ],
    "f1": [
        "f1", "F1",
        "metrics/f1", "metrics/F1",
        "box/F1", "seg/F1",
        "metrics/f1(B)", "f1(B)",  # ← CSV에서 괄호 버전
        "metrics/f1_score", "f1_score"  # ← 다른 버전 호환
    ],
    # 손실 쪽도 버전마다 달라서 넓게
    "box_loss": ["box_loss","train/box_loss","loss/box","metrics/box_loss","val/box_loss"],
    "cls_loss": ["cls_loss","train/cls_loss","loss/cls","metrics/cls_loss","val/cls_loss"],
    "obj_loss": ["obj_loss","train/obj_loss","loss/obj","metrics/obj_loss","val/obj_loss"],
    "val_loss": ["val_loss","metrics/loss","loss/val","loss","val/loss"],
    "fitness":  ["fitness","metrics/fitness"],
}


_PRIMARY_ORDER: List[str] = [
    "mAP50-95",
    "mAP50",
    "f1",
    "precision",
    "recall",
]


def _get_metric_value(metrics: Dict[str, Any], canonical_key: str) -> Optional[float]:
    if not metrics:
        return None
    if canonical_key in metrics:
        return _to_float(metrics[canonical_key])
    for alias in _METRIC_ALIASES.get(canonical_key, []):
        if alias in metrics:
            v = _to_float(metrics[alias])
            if v is not None:
                return v
    # 소문자 맵핑(마지막 시도)
    low = {str(k).lower(): k for k in metrics.keys()}
    lk = canonical_key.lower()
    if lk in low:
        return _to_float(metrics[low[lk]])
    return None


def _collect_metric_files(run_dir: Optional[Union[str, Path]]) -> List[Path]:
    paths: List[Path] = []
    if not run_dir:
        return paths
    d = Path(run_dir)
    if not d.exists() or not d.is_dir():
        return paths

    # ✅ CSV까지 포함
    globs = ["results*.json", "metrics*.json", "val*.json", "results*.csv", "val*.csv"]

    for pattern in globs:
        paths.extend(sorted(d.glob(pattern)))
    return paths


def _load_metrics_from_jsons(files: List[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for f in files:
        if f.suffix.lower() != ".json":
            continue
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    merged.update(data)
        except Exception:
            continue
    return merged


def _load_last_row_from_csv(f: Path) -> Dict[str, Any]:
    """
    Ultralytics results.csv의 마지막 행(최신 에폭)의 컬럼들을 dict로 반환.
    버전에 따라 컬럼명이 조금씩 달라서, 가능한 한 원문 키를 그대로 유지.
    """
    out: Dict[str, Any] = {}
    try:
        with f.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            last: Optional[Dict[str, str]] = None
            for row in reader:
                last = row
            if not last:
                return out
            # 숫자 변환 시도
            for k, v in last.items():
                fv = _to_float(v)
                out[k] = fv if fv is not None else v
    except Exception:
        pass
    return out


def _load_metrics_from_files(files: List[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    # JSON 먼저 흡수
    merged.update(_load_metrics_from_jsons(files))
    # CSV(여러 개일 수 있지만 보통 1개)에서 마지막 행 흡수
    for f in files:
        if f.suffix.lower() == ".csv":
            merged.update(_load_last_row_from_csv(f))
    return merged


def _pick_primary_score(metrics: Dict[str, Any]) -> Tuple[str, float]:
    # 1) 우선순위 기반 탐색
    for key in _PRIMARY_ORDER:
        v = _get_metric_value(metrics, key)
        if v is not None:
            return key, float(v)

    # 2) precision/recall 평균
    p = _get_metric_value(metrics, "precision")
    r = _get_metric_value(metrics, "recall")
    if p is not None and r is not None:
        return "pr-avg", float((p + r) / 2.0)

    # 3) 모든 스칼라 평균
    vals: List[float] = []
    for _, v in metrics.items():
        fv = _to_float(v)
        if fv is not None:
            vals.append(fv)
    if vals:
        return "scalar-mean", float(sum(vals) / max(1, len(vals)))

    # 4) 실패 시
    return "none", float("nan")


# -------------------------------
# Node: evaluate_trial
# -------------------------------

def evaluate_trial(state: TrainState) -> TrainState:
    """
    학습 trial의 성능을 평가하고 대표 점수(score)를 결정해 state.metrics에 저장한다.

    개선사항:
      - train_trial.save_dir, context.run_dir/workdir 모두 자동 탐색
      - JSON + CSV(results.csv)까지 파싱하여 가능한 한 많은 지표를 수집
      - score 및 primary_metric 자동 보정
      - 주요 지표 alias를 context에 요약 저장
    """
    metrics: Dict[str, Any] = {}
    if isinstance(state.metrics, dict):
        metrics.update(state.metrics)

    print("[evaluate_trial] 노드 실행")

    # -------------------
    # 1️⃣ run_dir 탐색 우선순위
    # -------------------
    run_dir = None
    if isinstance(state.context, dict):
        run_dir = (
            state.context.get("run_dir")
            or state.context.get("workdir")
            or (state.context.get("train_trial") or {}).get("save_dir")
        )

    # -------------------
    # 2️⃣ 파일에서 메트릭 읽기 (JSON + CSV)
    # -------------------
    files = _collect_metric_files(run_dir)
    if files:
        file_metrics = _load_metrics_from_files(files)
        if file_metrics:
            metrics.update(file_metrics)
            state.notes = f"[evaluate_trial] 결과 파일 {len(files)}개(JSON/CSV)에서 메트릭을 읽었습니다."
    else:
        if not metrics:
            state.notes = "[evaluate_trial] 결과 파일과 기존 metrics가 모두 없어 평가를 건너뜁니다."

    # -------------------
    # 3️⃣ 대표 점수 계산
    # -------------------
    primary_key, score = _pick_primary_score(metrics)
    metrics["primary_metric"] = primary_key
    metrics["score"] = _to_float(metrics.get("score")) or score

    # -------------------
    # 4️⃣ 결과 반영
    # -------------------
    state.metrics = metrics
    if primary_key == "none" or (isinstance(score, float) and score != score):  # NaN
        state.error = (state.error or "") + "[evaluate_trial] 유효한 메트릭을 찾지 못했습니다. "
        print("[evaluate_trial] 유효한 메트릭을 찾지 못했습니다.")
    else:
        note = f"[evaluate_trial] 대표 메트릭={primary_key}, 점수={metrics['score']}"
        print(note)
        state.notes = f"{(state.notes or '')}\n{note}".strip()

    # -------------------
    # 5️⃣ context에도 요약 기록 (aliases)
    # -------------------
    aliases: Dict[str, Any] = {}
    for key in [
        "mAP50-95", "mAP50", "precision", "recall", "f1", "fitness",
        "box_loss", "cls_loss", "obj_loss", "val_loss"
    ]:
        v = _get_metric_value(metrics, key)
        if v is not None:
            aliases[key] = v

    c = state.context or {}
    c["evaluate_trial"] = {
        "run_dir": run_dir,
        "primary_metric": primary_key,
        "score": metrics["score"],
        "files_found": len(files),
        "metrics_aliases": aliases,
        "metrics_all_count": len(metrics),
    }
    state.context = c

    return state
