# core/gpt_eval.py
import json, re
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
scorer = pipeline("text2text-generation", model="google/flan-t5-large")

def gpt_extract_root_cause(reason_text: str) -> str:
    try:
        result = summarizer(reason_text, max_length=50, min_length=10, do_sample=False)
        return result[0]['summary_text'].strip()
    except Exception as e:
        print(f"❌ Local Model Error (extraction): {e}")
        return "Root cause not extracted"

def gpt_score_pdca(case_context: dict, file_type: str):
    """
    Evaluate a case using PDCA barème V4.
    - case_context: {"Finding": str, "Root Cause": str, "Corrective Action": str, ...}
    - file_type: "pdca_system" | "pdca_audit_process"
    Acceptance rule:
      • If overall > 7.5 → Accepted
      • If overall ≤ 7.5 → Accepted only if excellence.req1 and excellence.req3 are both True
    """
    min_score = 7.5
    excellence_keys = ["req1", "req3"]

    prompt = f"""
You are a professional PDCA auditor AI.
For EACH PDCA stage (Plan, Do, Check, Act), rate every criterion with a numeric score from 0 to 10.
Also output two boolean excellence flags: req1 and req3.
Return STRICT JSON ONLY with this shape:
{{
  "stages": {{
    "Plan": {{"scores": {{"<criterion>": <0-10>, ...}}, "comment": "<one-line>"}},
    "Do":   {{"scores": {{...}}, "comment": "..."}},
    "Check":{{"scores": {{...}}, "comment": "..."}},
    "Act":  {{"scores": {{...}}, "comment": "..."}}
  }},
  "excellence": {{"req1": true, "req3": true}},
  "overall_comment": "<1-2 lines>"
}}

Case context:
{json.dumps(case_context, indent=2)}
"""
    try:
        result = scorer(prompt, max_new_tokens=512)
        raw = (result[0].get("generated_text") or result[0].get("text", "")).strip()
        print("\n🧪 Raw PDCA JSON:\n", raw, "\n")
        data = json.loads(raw)
    except Exception:
        data = {"stages": {}, "excellence": {"req1": False, "req3": False}, "overall_comment": "Model failed."}

    # --- Collect and normalize scores ---
    all_scores, by_stage, flat_scores = [], [], []
    for stage, block in (data.get("stages") or {}).items():
        scores = block.get("scores") or {}
        nums = []
        for crit, val in scores.items():
            try:
                num = float(val)
            except Exception:
                m = re.search(r"(\d+(\.\d+)?)", str(val))
                num = float(m.group(1)) if m else 0.0
            num = max(0.0, min(10.0, num))
            nums.append(num)
            all_scores.append(num)
            flat_scores.append((stage, str(crit), num))
        by_stage.append({
            "stage": stage,
            "scores": scores,
            "avg": round(sum(nums)/len(nums), 2) if nums else 0.0,
            "comment": block.get("comment", "")
        })

    overall = round(sum(all_scores)/len(all_scores), 2) if all_scores else 0.0
    excellence = data.get("excellence") or {}
    req1_ok = bool(excellence.get("req1", False))
    req3_ok = bool(excellence.get("req3", False))

    # --- New acceptance rule ---
    if overall > min_score:
        status = "Accepted"
    elif req1_ok and req3_ok:
        status = "Accepted"
    else:
        status = "Not Accepted"

    # --- Compose explicit failure reason when Not Accepted ---
    failure_reason = ""
    if status == "Not Accepted":
        reasons = []
        if overall <= min_score:
            reasons.append(f"Overall score {overall:.2f} ≤ minimum {min_score:.2f}")
        missing = [k for k in excellence_keys if not excellence.get(k, False)]
        if missing:
            friendly = {"req1": "Excellence #1", "req3": "Excellence #3"}
            miss_labels = ", ".join(friendly.get(m, m) for m in missing)
            reasons.append(f"{miss_labels} not satisfied")
        if flat_scores:
            worst = sorted(flat_scores, key=lambda t: t[2])[:2]
            worst_txt = "; ".join(f"{st}/{cr}: {sc:.2f}" for st, cr, sc in worst)
            reasons.append(f"Lowest criteria: {worst_txt}")
        failure_reason = " | ".join(reasons) if reasons else "Failed acceptance rules."

    # Ensure overall_comment contains the reason when failing
    overall_comment = (data.get("overall_comment") or "").strip()
    if status == "Not Accepted":
        if overall_comment:
            if failure_reason and failure_reason not in overall_comment:
                overall_comment = f"{failure_reason}. {overall_comment}"
        else:
            overall_comment = failure_reason

    return {
        "file_type": file_type,
        "by_stage": by_stage,
        "overall_score": overall,
        "excellence": {"req1": req1_ok, "req3": req3_ok},
        "min_required": min_score,
        "status": status,
        "overall_comment": overall_comment,
        "failure_reason": failure_reason,  # empty if Accepted
        "raw": data
    }
