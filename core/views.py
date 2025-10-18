import os
import io
import re
import pandas as pd
from io import BytesIO

from django.http import HttpResponse
from django.template.loader import get_template
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Count, Avg

from xhtml2pdf import pisa

from .models import AnalysisHistory, AnalysisRun
from .excel_reader import read_excel_file
from .text_processor import clean_text
from .gpt_eval import (
    gpt_extract_root_cause,
    gpt_score_pdca,
)

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _clamp10(x):
    return max(0.0, min(10.0, x))

def _normalize_pdca(pdca: dict, has_content: bool = False):
    """
    Accept different shapes from gpt_score_pdca and normalize.
    Guarantees: overall_score ∈ [0,10], status, overall_comment strings.
    """
    pdca = pdca or {}
    raw_score = (
        pdca.get("overall_score")
        or pdca.get("overall")
        or pdca.get("score_10")
        or pdca.get("score")
        or pdca.get("score_over_10")
    )
    overall_score = _clamp10(_safe_float(raw_score, 0.0))

    min_required = _clamp10(_safe_float(pdca.get("min_required", 7.5)))
    ex1 = bool(pdca.get("excellence_1", False))
    ex3 = bool(pdca.get("excellence_3", False))

    status = pdca.get("status")
    if not status:
        status = "Accepted" if (overall_score >= min_required and ex1 and ex3) else "Not Accepted"

    overall_comment = (pdca.get("overall_comment") or pdca.get("reason") or "").strip()

    # only bump if there is content but score is exactly zero
    if overall_score == 0.0 and has_content:
        overall_score = 1.0

    return {
        "overall_score": overall_score,
        "status": status,
        "overall_comment": overall_comment,
        "min_required": min_required,
        "excellence_1": ex1,
        "excellence_3": ex3,
    }


# --- small helper to coerce any Excel cell to a clean string ---
def cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or (hasattr(pd, "isna") and pd.isna(v)):
        return ""
    return str(v).strip()


# -----------------------------
# Auth + History
# -----------------------------
@login_required
def history_view(request):
    runs_qs = (
        AnalysisRun.objects
        .filter(user=request.user)
        .annotate(
            items_count=Count('items'),
            avg_score=Avg('items__score'),        # PDCA overall score (/10)
            avg_ca_score=Avg('items__ca_score'),  # kept for backward compat
        )
        .order_by('-created_at')
    )
    paginator = Paginator(runs_qs, 10)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(request, "history_grouped.html", {"page_obj": page_obj})


def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("upload")
    else:
        form = UserCreationForm()
    return render(request, "core/register.html", {"form": form})


# -----------------------------
# Upload + Analysis (PDCA)
# -----------------------------
@login_required
def upload_file_view(request):
    if request.method == "POST" and request.FILES.get("excel_file"):
        file_type = (request.POST.get("file_type") or "").strip().lower()  # "system" | "audit"

        excel_file = request.FILES["excel_file"]
        file_path = f"temp_{excel_file.name}"

        # Save uploaded file to temp path
        with open(file_path, "wb+") as dest:
            for chunk in excel_file.chunks():
                dest.write(chunk)

        df = read_excel_file(file_path)

        if df is None:
            try:
                os.remove(file_path)
            except Exception:
                pass
            return render(request, "core/upload.html", {"error": "Invalid Excel file."})

        # ---- OPTIONAL: column validation for nicer errors ----
        if file_type == "system":
            expected = {"Nr.", "Process: Non-conformity / Potential", "Root cause analysis (5 WHY)", "Actions"}
        else:
            expected = {"Reference", "Findings", "Reasons", "Measures"}

        missing = [c for c in expected if c not in df.columns]
        if missing:
            try:
                os.remove(file_path)
            except Exception:
                pass
            return render(
                request,
                "core/upload.html",
                {"error": f"Missing columns for '{file_type or 'audit'}': {', '.join(missing)}"},
            )
        # ------------------------------------------------------

        results = []

        # Create the run entry
        run = AnalysisRun.objects.create(
            user=request.user,
            filename=excel_file.name
        )

        bulk_rows = []

        for _, row in df.iterrows():
            # -------------------------------
            # Column mapping depending on file type (system vs audit)
            # -------------------------------
            if file_type == "system":
                ref = cell(row.get("Nr.", ""))
                finding = clean_text(cell(row.get("Process: Non-conformity / Potential", "")))
                reasons = clean_text(cell(row.get("Root cause analysis (5 WHY)", "")))
                measures = clean_text(cell(row.get("Actions", "")))
            else:  # default = audit
                ref = cell(row.get("Reference", ""))
                finding = clean_text(cell(row.get("Findings", "")))
                reasons = clean_text(cell(row.get("Reasons", "")))
                measures = clean_text(cell(row.get("Measures", "")))

            # -------------------------------
            # AI Processing (PDCA only)
            # -------------------------------
            root_cause = gpt_extract_root_cause(reasons) if reasons else ""

            case_ctx = {
                "Reference": ref,
                "Finding": finding,
                "Root Cause": reasons or root_cause,  # if Reasons empty, pass summary
                "Corrective Action": measures,
            }

            pdca_file_type = "pdca_system" if file_type == "system" else "pdca_audit_process"
            raw_pdca = gpt_score_pdca(case_ctx, file_type=pdca_file_type)


            # Normalize fields for templates/session
            has_content = bool(finding or reasons or measures)
            pdca_eval = _normalize_pdca(raw_pdca , has_content=has_content)
            overall_score = pdca_eval["overall_score"]
            status = pdca_eval["status"]
            overall_comment = pdca_eval["overall_comment"]

            result_item = {
                "reference": ref,
                "finding": finding,
                "reasons": reasons,
                "measures": measures,
                "root_cause": root_cause,
                "pdca": pdca_eval,                 # full PDCA dictionary
                "score": overall_score,            # legacy keys used in history
                "status": status,
                "comment": overall_comment,
                # CA legacy fields (not used in PDCA mode) — avoid NOT NULL issues
                "ca_score": 0.0,
                "ca_status": "",
                "ca_comment": "",
            }
            results.append(result_item)

            # Save per-item row (use safe defaults for CA fields)
            bulk_rows.append(
                AnalysisHistory(
                    user=request.user,
                    run=run,
                    reference=ref,
                    finding=finding,
                    reasons=reasons,
                    root_cause=root_cause,
                    score=overall_score,
                    status=status,
                    comment=overall_comment,
                    measures=measures,
                    ca_score=0.0,      # if your model allows NULL, you can switch to None
                    ca_status="",
                    ca_comment="",
                )
            )

        # Bulk insert for speed
        AnalysisHistory.objects.bulk_create(bulk_rows, batch_size=500)

        # Cleanup temp file
        try:
            os.remove(file_path)
        except Exception:
            pass

        # Cache for PDF export
        request.session["last_results"] = results

        return render(
            request,
            "core/result.html",
            {
                "results": results,
                "root_model": "facebook/bart-large-cnn",
                "score_model": "google/flan-t5-large",
            },
        )

    # GET or missing file
    return render(request, "core/upload.html")


# -----------------------------
# PDF Export
# -----------------------------

# optional: strip CSS that xhtml2pdf doesn't support
UNSAFE_CSS_PATTERNS = [
    r'@keyframes[\s\S]*?\{[\s\S]*?\}',           # animations
    r'\banimation\s*:[^;"}]+;?',                 # animation: ...
    r'\btransform\s*:[^;"}]+;?',                 # transform: rotate/scale/…
    r'\bdisplay\s*:\s*(flex|grid)[^;"}]*;?',     # flex/grid
    r'@media[^{]*\{[\s\S]*?\}',                  # complex media queries
]

def _strip_unsafe_css(html: str) -> str:
    for pat in UNSAFE_CSS_PATTERNS:
        html = re.sub(pat, "", html, flags=re.IGNORECASE)
    return html


@login_required
def download_pdf_view(request):
    results = request.session.get("last_results")
    if not results:
        return HttpResponse("No data to export.", status=400)

    # IMPORTANT: result_pdf.html must be a standalone template (no extends)
    template = get_template("core/result_pdf.html")
    html = template.render({"results": results})
    html = _strip_unsafe_css(html)  # safety net

    out = BytesIO()
    pdf = pisa.CreatePDF(src=html, dest=out, encoding="utf-8")

    if pdf.err:
        return HttpResponse("Error generating PDF", status=500)

    response = HttpResponse(out.getvalue(), content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="evalyze_results.pdf"'
    return response
