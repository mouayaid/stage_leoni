import os
import io
import pandas as pd
from django.http import HttpResponse
from django.template.loader import get_template
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from xhtml2pdf import pisa
from django.core.paginator import Paginator
from django.db.models import Count, Avg

from .models import AnalysisHistory, AnalysisRun
from .excel_reader import read_excel_file
from .text_processor import clean_text
from .gpt_eval import (  # gpt_eval now has PDCA only
    gpt_extract_root_cause,
    gpt_score_pdca,
)

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
            avg_score=Avg('items__score'),       # we store PDCA overall_score here
            avg_ca_score=Avg('items__ca_score')  # kept for backward compat; will be None
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
            return render(request, "core/upload.html", {
                "error": f"Missing columns for '{file_type or 'audit'}': {', '.join(missing)}"
            })
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
            # NEW RULE: gpt_score_pdca no longer takes 'critical'
            pdca_eval = gpt_score_pdca(case_ctx, file_type=pdca_file_type)

            result_item = {
                "reference": ref,
                "finding": finding,
                "reasons": reasons,
                "measures": measures,
                "root_cause": root_cause,
                # PDCA summary
                "pdca": pdca_eval,
                # keep legacy keys populated for history/templates
                "score": pdca_eval.get("overall_score"),
                "status": pdca_eval.get("status"),
                "comment": pdca_eval.get("overall_comment"),
                "ca_score": None,
                "ca_status": None,
                "ca_comment": None,
            }
            results.append(result_item)

            # Save per-item row
            bulk_rows.append(AnalysisHistory(
                user=request.user,
                run=run,
                reference=ref,
                finding=finding,
                reasons=reasons,
                root_cause=root_cause,
                score=pdca_eval.get("overall_score"),     # overall score (/10)
                status=pdca_eval.get("status"),
                comment=pdca_eval.get("overall_comment"),
                measures=measures,
                ca_score=None,
                ca_status=None,
                ca_comment=None,
            ))

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
@login_required
def download_pdf_view(request):
    results = request.session.get("last_results")
    if not results:
        return HttpResponse("No data to export.", status=400)

    template = get_template("core/result_pdf.html")
    html = template.render({"results": results})

    pdf_out = io.BytesIO()
    pdf = pisa.pisaDocument(io.BytesIO(html.encode("utf-8")), dest=pdf_out)

    if not pdf.err:
        response = HttpResponse(pdf_out.getvalue(), content_type="application/pdf")
        response["Content-Disposition"] = 'attachment; filename="evalyze_results.pdf"'
        return response

    return HttpResponse("Error generating PDF", status=500)
