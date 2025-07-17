import re
from transformers import pipeline

# Load local models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
scorer = pipeline("text2text-generation", model="google/flan-t5-large")  # large version

def gpt_extract_root_cause(reason_text):
    """
    Extract the root cause from a 5-Why analysis using a local summarization model.
    """
    try:
        result = summarizer(reason_text, max_length=50, min_length=10, do_sample=False)
        return result[0]['summary_text'].strip()
    except Exception as e:
        print(f"❌ Local Model Error (extraction): {e}")
        return "Root cause not extracted"

def gpt_score_root_cause(reason_text, finding_text):
    """
    Evaluate the root cause using a word-based quality rating and map it to a score + status.
    """
    prompt = f"""
You are a professional quality auditor AI.

Evaluate the quality of the root cause below based on these 5 criteria:
1. Linkage to the Finding
2. Systematic Approach (e.g. 5 Whys)
3. Completeness
4. Depth of Analysis
5. Consideration of Secondary Causes

You must respond in this exact format:
Rating: <one of: Excellent, Good, Average, Poor, Unacceptable>
Comment: <your explanation in 1–2 sentences>

Here are examples:

Example 1:
Rating: Good
Comment: The root cause is well-linked to the finding and shows structure, but lacks depth and secondary causes.

Example 2:
Rating: Unacceptable
Comment: The root cause is vague, incomplete, and does not address the original finding.

Now evaluate this case:
Finding: {finding_text}
Root Cause: {reason_text}
"""

    try:
        result = scorer(prompt, max_new_tokens=256)
        result_text = result[0].get('generated_text') or result[0].get('text', '')
        result_text = result_text.strip()

        print("\n🧪 Raw model output (Root Cause):\n" + result_text + "\n")

        rating_match = re.search(r"Rating\s*[:=\-]?\s*(Excellent|Good|Average|Poor|Unacceptable)", result_text, re.IGNORECASE)
        if not rating_match:
            rating_match = re.search(r"^(Excellent|Good|Average|Poor|Unacceptable)$", result_text.strip(), re.IGNORECASE)

        comment_match = re.search(r"Comment\s*[:=\-]?\s*(.*)", result_text, re.DOTALL)

        rating = rating_match.group(1).strip().lower() if rating_match else "unknown"
        comment = comment_match.group(1).strip() if comment_match else ""

        score_map = {
            "excellent": 5,
            "good": 4,
            "average": 3,
            "poor": 2,
            "unacceptable": 1
        }

        score = score_map.get(rating, None)
        status = "Accepted" if score and score >= 4 else "Not Accepted" if score else "Unknown"

        return {
            "score": score,
            "status": status,
            "comment": comment,
            "raw": result_text
        }

    except Exception as e:
        print(f"❌ Local Model Error (score): {e}")
        return {
            "score": None,
            "status": "Unknown",
            "comment": "Evaluation failed.",
            "raw": "Evaluation failed."
        }

def gpt_score_corrective_action(measure_text, root_cause_text):
    """
    Evaluate the corrective action using a word-based quality rating.
    """
    prompt = f"""
You are a professional quality auditor AI.

Evaluate the following corrective action based on these 4 criteria:
1. Specificity and clarity
2. Linkage to the root cause
3. Preventive nature (avoids recurrence)
4. Systematic and structured approach

Respond using this format:
Rating: <Excellent, Good, Average, Poor, Unacceptable>
Comment: <your explanation in 1–2 sentences>

Examples:

Example 1:
Rating: Excellent
Comment: The action is clear, targeted directly at the root cause, and includes preventive measures with structured follow-up.

Example 2:
Rating: Poor
Comment: The action is vague and does not fully address the root cause or prevent recurrence.

Now evaluate:
Root Cause: {root_cause_text}
Corrective Action: {measure_text}
"""

    try:
        result = scorer(prompt, max_new_tokens=256)
        result_text = result[0].get('generated_text') or result[0].get('text', '')
        result_text = result_text.strip()

        print("\n🛠️ Raw CA output:\n" + result_text + "\n")

        rating_match = re.search(r"Rating\s*[:=\-]?\s*(Excellent|Good|Average|Poor|Unacceptable)", result_text, re.IGNORECASE)
        if not rating_match:
            rating_match = re.search(r"^(Excellent|Good|Average|Poor|Unacceptable)$", result_text.strip(), re.IGNORECASE)

        comment_match = re.search(r"Comment\s*[:=\-]?\s*(.*)", result_text, re.DOTALL)

        rating = rating_match.group(1).strip().lower() if rating_match else "unknown"
        comment = comment_match.group(1).strip() if comment_match else ""

        score_map = {
            "excellent": 5,
            "good": 4,
            "average": 3,
            "poor": 2,
            "unacceptable": 1
        }

        score = score_map.get(rating, None)
        status = "Accepted" if score and score >= 4 else "Not Accepted" if score else "Unknown"

        return {
            "score": score,
            "status": status,
            "comment": comment,
            "raw": result_text
        }

    except Exception as e:
        print(f"❌ Local Model Error (CA score): {e}")
        return {
            "score": None,
            "status": "Unknown",
            "comment": "Evaluation failed.",
            "raw": "Evaluation failed."
        }
