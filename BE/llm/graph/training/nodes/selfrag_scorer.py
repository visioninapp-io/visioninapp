# graph/eval/selfrag_scorer.py
from __future__ import annotations

"""
Requirements Alignment Scorer
- 목적: 사용자 프롬프트의 요구사항과
       (1) query_analyzer의 parsed 결과,
       (2) param_synthesizer의 synthesized/resolved 결과
  사이의 일치도를 정량화.

- 외부 사용:
    from graph.eval.selfrag_scorer import score_requirements_alignment, align_from_state

    # 직접 점수화
    score = score_requirements_alignment(
        user_prompt="yolov12n, fp16로 학습하고 hpo 20 트라이얼",
        parsed={"model_variant":"yolo12n","precision":"fp16","use_hpo":True,"max_trials":20},
        synthesized={"epochs":100,"batch":16,"imgsz":640,"optimizer":"SGD","model_variant":"yolo12n","precision":"fp16","use_hpo":True,"max_trials":20},
        resolved=None,        # (선택) 실제 적용된 최종 파라미터가 있으면 전달
        use_llm=True          # LLM 있으면 True, 없으면 False로 휴리스틱만
    )
    print(score.dict())

    # TrainState에서 자동 수집
    align = align_from_state(state, use_llm=True, prefer_resolved=True)
    print(align.dict())
"""

import json
import re
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# --- Optional LLM deps ---
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    _LLM_OK = True
except Exception:
    _LLM_OK = False


# =========================
# Utilities
# =========================

def _norm_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: Any) -> List[str]:
    s = _norm_text(s)
    return [t for t in re.split(r"[^\w]+", s) if t]

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# =========================
# Data structures
# =========================

@dataclass
class FieldMatch:
    key: str
    expected: Any      # Parsed (사용자 요구) 기준
    actual: Any        # Synthesized/Resolved (시스템 결정)
    score: float
    note: Optional[str] = None

@dataclass
class AlignmentScore:
    overall: float
    per_field: List[FieldMatch]
    used_llm: bool
    model_info: Optional[str] = None
    def dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "per_field": [asdict(x) for x in self.per_field],
            "used_llm": self.used_llm,
            "model_info": self.model_info,
        }


# =========================
# Heuristic scoring
# =========================

_DEFAULT_WEIGHTS = {
    "intent": 1.0,
    "model_variant": 1.2,
    "precision": 1.0,
    "epochs": 0.8,
    "batch": 0.8,
    "imgsz": 0.8,
    "optimizer": 0.6,
    "use_hpo": 1.0,
    "max_trials": 0.6,
}

def _num_tolerance_match(exp: Any, act: Any, rel_tol: float = 0.1) -> float:
    a, b = _to_float(exp), _to_float(act)
    if a is None or b is None:
        return 0.0
    if a == 0:
        return 1.0 if b == 0 else 0.0
    return _clip01(1.0 - min(abs(b - a) / max(abs(a), 1e-9), 1.0))

def _cat_exact_or_soft(exp: Any, act: Any) -> float:
    if exp is None or act is None:
        return 0.0
    if _norm_text(exp) == _norm_text(act):
        return 1.0
    et, at = set(_tokenize(exp)), set(_tokenize(act))
    return _clip01(len(et & at) / len(et | at))

def _bool_match(exp: Any, act: Any) -> float:
    try:
        return 1.0 if bool(exp) == bool(act) else 0.0
    except Exception:
        return 0.0

def _guess_and_score(key: str, exp: Any, act: Any) -> Tuple[float, str]:
    k = key.lower()
    if k in ("epochs", "batch", "imgsz", "max_trials"):
        return _num_tolerance_match(exp, act), "numeric±10%"
    if k in ("intent", "model_variant", "precision", "optimizer"):
        s = _cat_exact_or_soft(exp, act)
        return s, ("exact" if s == 1.0 else "soft")
    if k == "use_hpo":
        return _bool_match(exp, act), "bool"
    # 기본: 토큰 교집합 기반 소프트 매치
    s = _cat_exact_or_soft(exp, act)
    return s, ("exact" if s == 1.0 else "soft")

def _weighted_overall(per: List[FieldMatch], weights: Dict[str, float]) -> float:
    num = den = 0.0
    for fm in per:
        w = float(weights.get(fm.key, 1.0))
        num += w * fm.score
        den += w
    return _clip01(num / den if den > 0 else 0.0)


# =========================
# LLM-based scoring (optional)
# =========================

_ALIGNMENT_LLM_PROMPT = """
너는 AutoML 품질감사자다. 아래 정보를 바탕으로
"사용자 요구(원문 Prompt 및 Parsed)"와 "시스템 파라미터(Synthesized/Resolved)"의 일치도를 0~1로 채점한다.
키별 점수(per_field)를 주고, 가중 평균 overall을 계산한다.

출력은 JSON 한 줄:
{{"overall": float, "per_field": [{{"key": str, "expected": any, "actual": any, "score": float, "note": str}}]}}

[User Prompt]
{user_prompt}

[Parsed from query_analyzer]
{parsed_json}

[Synthesized/Resolved from param_synthesizer]
{synth_json}
""".strip()


# =========================
# Public APIs
# =========================

def score_requirements_alignment(
    user_prompt: Optional[str],
    parsed: Dict[str, Any],
    synthesized: Dict[str, Any],
    resolved: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
) -> AlignmentScore:
    """
    - user_prompt: 원문 사용자 요구(자연어)
    - parsed: query_analyzer 결과(JSON)
    - synthesized: param_synthesizer 결과(JSON)
    - resolved: 실제 최종 적용(있으면 synthesized 대신 비교 우선)
    """
    expected = dict(parsed or {})
    actual = dict(resolved or synthesized or {})

    # 비교 대상 키(필요시 추가/수정)
    keys = ["intent","model_variant","precision","epochs","batch","imgsz","optimizer","use_hpo","max_trials"]
    per: List[FieldMatch] = []

    # 1) LLM 시도
    if use_llm and _LLM_OK:
        try:
            llm = ChatOpenAI(
                model="gpt-5", 
                api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                temperature=1
            )
            tpl = ChatPromptTemplate.from_template(_ALIGNMENT_LLM_PROMPT)
            out = (tpl | llm).invoke({
                "user_prompt": (user_prompt or "").strip(),
                "parsed_json": json.dumps(expected, ensure_ascii=False),
                "synth_json": json.dumps(actual, ensure_ascii=False),
            })
            data = json.loads(getattr(out, "content", "") or "{}")
            if isinstance(data.get("per_field"), list):
                seen = set()
                for item in data["per_field"]:
                    k = str(item.get("key"))
                    seen.add(k)
                    per.append(FieldMatch(
                        key=k,
                        expected=expected.get(k),
                        actual=actual.get(k),
                        score=_clip01(float(item.get("score", 0.0))),
                        note=str(item.get("note") or "").strip() or "llm",
                    ))
                for k in keys:
                    if k in seen:  # 누락 보강
                        continue
                    s, note = _guess_and_score(k, expected.get(k), actual.get(k))
                    per.append(FieldMatch(k, expected.get(k), actual.get(k), _clip01(s), f"heuristic:{note}"))
                overall = _weighted_overall(per, _DEFAULT_WEIGHTS)
                return AlignmentScore(overall, per, used_llm=True, model_info=llm_model)
        except Exception:
            pass  # LLM 실패 시 휴리스틱 폴백

    # 2) 휴리스틱 폴백
    for k in keys:
        s, note = _guess_and_score(k, expected.get(k), actual.get(k))
        per.append(FieldMatch(k, expected.get(k), actual.get(k), _clip01(s), f"heuristic:{note}"))
    overall = _weighted_overall(per, _DEFAULT_WEIGHTS)
    return AlignmentScore(overall, per, used_llm=False, model_info="heuristic")


def align_from_state(
    state,                    # TrainState (직접 임포트 순환 방지)
    use_llm: bool = True,
    prefer_resolved: bool = True,
) -> AlignmentScore:
    """
    TrainState에서 필요한 정보를 모아 요구 정합성 점수 계산:
      - user_prompt: state.user_query
      - parsed: state.context["query_analyzer"]["parsed"]
      - synthesized: state.context["param_synthesizer"]["final_params"] 또는 state.train_overrides
      - resolved: prefer_resolved=True면 train_trial에서 실제로 사용된 args
    """
    ctx = getattr(state, "context", {}) or {}

    user_prompt = getattr(state, "user_query", None)

    qa = ctx.get("query_analyzer") or {}
    parsed = qa.get("parsed") or {
        "intent": getattr(state, "intent", None),
        "model_variant": getattr(state, "model_variant", None),
        "precision": getattr(state, "precision", None),
        "use_hpo": ((state.hpo or {}).get("enabled") if getattr(state, "hpo", None) else None),
        "max_trials": ((state.hpo or {}).get("max_trials") if getattr(state, "hpo", None) else None),
        "epochs": ((state.train_overrides or {}).get("epochs") if getattr(state, "train_overrides", None) else None),
        "batch": ((state.train_overrides or {}).get("batch") if getattr(state, "train_overrides", None) else None),
        "imgsz": ((state.train_overrides or {}).get("imgsz") if getattr(state, "train_overrides", None) else None),
        "optimizer": ((state.train_overrides or {}).get("optimizer") if getattr(state, "train_overrides", None) else None),
    }

    ps = ctx.get("param_synthesizer") or {}
    synthesized = (
        ps.get("final_params")
        or ps.get("proposed_params")
        or (getattr(state, "train_overrides", None) or {})
    )

    resolved = None
    if prefer_resolved:
        tt = ctx.get("train_trial") or {}
        args = tt.get("args")
        if args:
            # 필요한 필드만 축약
            resolved = {
                "epochs": args.get("epochs"),
                "batch": args.get("batch"),
                "imgsz": args.get("imgsz"),
                "optimizer": args.get("optimizer"),
                "precision": ("fp16" if ctx.get("amp") else "fp32"),
                "use_hpo": ((state.hpo or {}).get("enabled") if getattr(state, "hpo", None) else None),
                "max_trials": ((state.hpo or {}).get("max_trials") if getattr(state, "hpo", None) else None),
                "model_variant": getattr(state, "model_variant", None),
                "intent": getattr(state, "intent", None),
            }

    return score_requirements_alignment(
        user_prompt=user_prompt,
        parsed=parsed,
        synthesized=synthesized,
        resolved=resolved,
        use_llm=use_llm,
    )

PASS_DEFAULTS = {
    "min_overall": 0.65,     # 전체 정합성 하한
    "min_key_hits": 0.5,     # 핵심 키들 평균 스코어 하한
    "key_weights": ("model_variant", "precision", "use_hpo"),  # 핵심 키
}

def _key_hits(per_field, keys) -> float:
    vals = []
    for k in keys:
        for f in per_field:
            if f.key == k and f.score is not None:
                vals.append(float(f.score))
                break
    return sum(vals)/len(vals) if vals else 0.0

def selfrag_scorer(
    state,                    # TrainState
    use_llm: bool = True,
    thresholds: dict | None = None,
) -> Dict[str, Any]:
    """
    LangGraph write-node: 반드시 dict만 반환
    - 정합성 점수 계산 → context.selfrag_scorer 에 기록
    - 분기 판단은 별도의 selfrag_decider에서 수행 (여기서는 라벨을 반환하지 않음)
    """
    th = {**PASS_DEFAULTS, **(thresholds or {})}

    align = align_from_state(
        state=state,
        use_llm=use_llm,
        prefer_resolved=True,
    )
    overall = float(align.overall)
    key_hits = _key_hits(align.per_field, th["key_weights"])
    print(f"[selfrag_scorer] scores - overall: {overall} | key_hits: {key_hits}")

    ctx = dict(state.context or {})
    ctx["selfrag_scorer"] = {
        "overall": overall,
        "key_hits": key_hits,
        "used_llm": align.used_llm,
        "per_field": [f.__dict__ for f in align.per_field],
        "thresholds": {
            "min_overall": th["min_overall"],
            "min_key_hits": th["min_key_hits"],
            "key_weights": list(th["key_weights"]),
        },
    }
    # ✅ 반드시 부분 업데이트 dict만 반환
    return {"context": ctx}