# core/gpt_eval.py
import json
import re
import warnings
from transformers import pipeline
from transformers.utils import logging as hf_logging

# ------------ Noise control ------------
hf_logging.set_verbosity_warning()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

DEBUG = False  # set True to print RAW model output

# ------------ Model loading (CPU) ------------
DEVICE = -1  # CPU only

def _load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn", device=DEVICE)
    except Exception:
        return None

def _load_scorer(model_name: str):
    try:
        return pipeline("text2text-generation", model=model_name, device=DEVICE)
    except Exception:
        return None

summarizer = _load_summarizer()
# Try large, fall back to base
scorer = _load_scorer("google/flan-t5-large") or _load_scorer("google/flan-t5-base")
print(
    f"✅ Loaded scorer model: {getattr(getattr(scorer,'model',None),'name_or_path','<none>')}"
    if scorer else "❌ Scorer model failed to load."
)

# ------------ Helpers ------------
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _clamp(x, lo=0.0, hi=10.0):
    return max(lo, min(hi, x))

def _coerce_json_from_text(raw: str):
    """
    If the model returns non-JSON prose, try to extract usable fields.
    Returns a dict or None.
    """
    if not raw:
        return None

    # Case 1: proper JSON wrapped in braces
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Case 2: loose prose like "overall: 8.5, min_required: 7.5, excellence_1: true..."
    def _find_num(key):
        mm = re.search(rf"{key}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.I)
        return float(mm.group(1)) if mm else None

    def _find_bool(key):
        mm = re.search(rf"{key}\s*[:=]\s*(true|false)", raw, flags=re.I)
        return (mm.group(1).lower() == "true") if mm else None

    overall = _find_num("overall")
    minreq  = _find_num("min_required") or _find_num("threshold") or 7.5
    ex1     = _find_bool("excellence_1")
    ex3     = _find_bool("excellence_3")

    reason = None
    mm = re.search(r"(reason|because|why)\s*[:=-]\s*(.+)", raw, flags=re.I)
    if mm:
        reason = mm.group(2).strip()

    if overall is None and ("{" not in raw):
        return None

    return {
        "overall": overall if overall is not None else 0.0,
        "min_required": minreq,
        "excellence_1": bool(ex1) if ex1 is not None else False,
        "excellence_3": bool(ex3) if ex3 is not None else False,
        "reason": reason or "Model text parsed (coerced).",
    }

# ------------ Root cause extraction ------------
def gpt_extract_root_cause(reason_text: str) -> str:
    """
    Summarize the 5-Why/Reasons text. Uses only max_new_tokens to avoid HF warnings.
    Falls back to a simple heuristic if the model is unavailable.
    """
    t = (reason_text or "").strip()
    if not t:
        return ""

    if summarizer is not None:
        try:
            words = len(t.split())
            # ~40–50% of input words, clamped to a sensible range
            max_new = max(20, min(80, int(words * 0.5)))
            out = summarizer(
                t,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=4,
                truncation=True,
            )
            return (out[0]["summary_text"] or "").strip()
        except Exception:
            pass

    # Fallback heuristic
    m = re.search(r"(?:because|due to|root cause(?: is)?)\s+(.+)", t, flags=re.I)
    if m:
        return m.group(1).strip()[:200]
    first = re.split(r"[.!?]\s+", t, maxsplit=1)[0]
    return (first or t)[:200]

# ------------ Heuristic fallback scoring ------------
def _fallback_rule_based_score(reason_text: str, finding_text: str) -> dict:
    text = f"{finding_text}\n\n{reason_text}".lower()
    if not text.strip():
        return {
            "overall": 0.0,
            "min_required": 7.5,
            "excellence_1": False,
            "excellence_3": False,
            "reason": "Empty input.",
        }
    pts = 0.0
    checks = {
        "has_because": r"\b(because|due to|root cause|原因|سبب)\b",
        "has_where": r"\b(at|on|in|station|line|area|kanban|process)\b",
        "has_mechanism": r"\b(mechanism|bent|twisted|leak|overload|mismatch|not respected)\b",
        "has_evidence": r"\b(evidence|photo|id|wi|ri|cp|rev|data|measurement|mpe|audit)\b",
        "has_action_link": r"\b(corrective|action|training|update|procure|calibrate|control plan)\b",
    }
    for pat in checks.values():
        if re.search(pat, text):
            pts += 2.0
    overall = _clamp(pts, 0, 10)
    ex1 = re.search(checks["has_evidence"], text) is not None
    ex3 = re.search(checks["has_action_link"], text) is not None
    return {
        "overall": overall,
        "min_required": 7.5,
        "excellence_1": bool(ex1),
        "excellence_3": bool(ex3),
        "reason": "Fallback rule-based scoring (model unavailable).",
    }

# ------------ Model-driven scoring ------------
JSON_INSTRUCTIONS = (
    "You MUST return ONLY valid JSON, no extra text, no code fences. "
    "Start with '{' and end with '}'. Keys: "
    "{\"overall\": number, \"min_required\": number, "
    "\"excellence_1\": boolean, \"excellence_3\": boolean, "
    "\"reason\": string}. Rules: overall in [0,10]. "
    "If uncertain, estimate a number (no strings)."
)

def _generate_with_retries(prompt: str) -> str:
    """
    Call the T5 scorer with deterministic params. Retry with shorter length.
    Fall back to a smaller model if needed.
    """
    global scorer
    if scorer is None:
        return ""

    attempts = [
        dict(max_new_tokens=120, do_sample=False, num_beams=4, temperature=0.0),
        dict(max_new_tokens=80,  do_sample=False, num_beams=4, temperature=0.0),
    ]
    for params in attempts:
        try:
            out = scorer(prompt, **params)
            txt = (out[0].get("generated_text") or "").strip()
            if txt:
                return txt
        except Exception:
            continue

    # last chance: smaller model
    small = _load_scorer("google/flan-t5-base") or _load_scorer("google/flan-t5-small")
    if small is not None:
        try:
            out = small(prompt, max_new_tokens=80, do_sample=False, num_beams=4, temperature=0.0)
            txt = (out[0].get("generated_text") or "").strip()
            if txt:
                return txt
        except Exception:
            pass
    return ""

def gpt_score_corrective_action(reason_text: str, finding_text: str) -> dict:
    """
    Ask the model for a STRICT JSON verdict. If parsing fails, try to coerce from text;
    if that also fails, use heuristic fallback.
    """
    prompt = (
        f"{JSON_INSTRUCTIONS}\n\n"
        f"Finding:\n{finding_text}\n\n"
        f"Root Cause / Reasoning:\n{reason_text}\n\n"
        f"Return JSON now:\n"
    )

    raw = _generate_with_retries(prompt)
    if DEBUG:
        print("🧠 RAW MODEL OUTPUT:", repr(raw[:400]))

    data = None
    if raw:
        # 1) strict JSON
        try:
            m = re.search(r"\{[\s\S]*\}", raw)
            data = json.loads(m.group(0) if m else raw)
        except Exception:
            # 2) coerce from prose
            data = _coerce_json_from_text(raw)

    if data:
        overall = _clamp(_safe_float(data.get("overall", 0.0)))
        min_required = _clamp(_safe_float(data.get("min_required", 7.5)))
        ex1 = bool(data.get("excellence_1", False))
        ex3 = bool(data.get("excellence_3", False))
        reason = (str(data.get("reason", "")) or "Model JSON parsed.").strip()
        if overall == 0.0 and (reason_text.strip() or finding_text.strip()):
            overall = 1.0  # minimal partial credit when content exists
        return {
            "overall": overall,
            "min_required": min_required,
            "excellence_1": ex1,
            "excellence_3": ex3,
            "reason": reason,
        }

    # 3) Model gave nothing usable → heuristic
    return _fallback_rule_based_score(reason_text, finding_text)

# ------------ PDCA wrapper ------------
def gpt_score_pdca(case_ctx: dict, file_type: str = "") -> dict:
    finding = case_ctx.get("Finding", "")
    root_cause = case_ctx.get("Root Cause", "")
    corrective_action = case_ctx.get("Corrective Action", "")

    joined_text = f"{root_cause}\n\n{corrective_action}"
    verdict = gpt_score_corrective_action(joined_text, finding)

    return {
        "overall_score": verdict["overall"],
        "min_required": verdict["min_required"],
        "status": "Accepted" if (verdict["overall"] >= verdict["min_required"]
                                 and verdict["excellence_1"] and verdict["excellence_3"]) else "Not Accepted",
        "overall_comment": verdict["reason"],
        "excellence_1": verdict["excellence_1"],
        "excellence_3": verdict["excellence_3"],
    }
