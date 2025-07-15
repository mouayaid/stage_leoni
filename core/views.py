import os
from django.shortcuts import render
from .excel_reader import read_excel_file
from .text_processor import clean_text
from .gpt_eval import gpt_extract_root_cause, gpt_score_root_cause

def upload_file_view(request):
    if request.method == 'POST' and request.FILES.get('excel_file'):
        excel_file = request.FILES['excel_file']
        file_path = f"temp_{excel_file.name}"

        # نحفظ الملف مؤقتًا
        with open(file_path, 'wb+') as dest:
            for chunk in excel_file.chunks():
                dest.write(chunk)

        df = read_excel_file(file_path)

        if df is not None:
            results = []

            for _, row in df.iterrows():
                ref = row['Reference']
                finding = clean_text(row['Findings'])
                reasons = clean_text(row['Reasons'])

                # 📌 استخراج Root Cause و Score من GPT
                root_cause = gpt_extract_root_cause(reasons)
                gpt_feedback = gpt_score_root_cause(reasons, finding)

                results.append({
                    'reference': ref,
                    'finding': finding,
                    'reasons': reasons,
                    'root_cause': root_cause,
                    'gpt_feedback': gpt_feedback
                })

            os.remove(file_path)  # نحذف الملف المؤقت
            return render(request, 'core/result.html', {'results': results})

        else:
            return render(request, 'core/upload.html', {'error': 'Fichier Excel non valide.'})

    return render(request, 'core/upload.html')
