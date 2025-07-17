import os
from django.shortcuts import render
from .excel_reader import read_excel_file
from .text_processor import clean_text
from .gpt_eval import gpt_extract_root_cause, gpt_score_root_cause, gpt_score_corrective_action

def upload_file_view(request):
    if request.method == 'POST' and request.FILES.get('excel_file'):
        excel_file = request.FILES['excel_file']
        file_path = f"temp_{excel_file.name}"

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
                measures = clean_text(row.get('Measures', ""))  # Handle missing column gracefully

                root_cause = gpt_extract_root_cause(reasons)
                gpt_feedback = gpt_score_root_cause(reasons, finding)
                ca_feedback = gpt_score_corrective_action(measures, root_cause)

                results.append({
                    'reference': ref,
                    'finding': finding,
                    'reasons': reasons,
                    'measures': measures,
                    'root_cause': root_cause,
                    'score': gpt_feedback.get('score'),
                    'status': gpt_feedback.get('status'),
                    'comment': gpt_feedback.get('comment'),
                    'ca_score': ca_feedback.get('score'),
                    'ca_status': ca_feedback.get('status'),
                    'ca_comment': ca_feedback.get('comment'),
                })

            os.remove(file_path)
            return render(request, 'core/result.html', {'results': results})

        else:
            return render(request, 'core/upload.html', {'error': 'Fichier Excel non valide.'})

    return render(request, 'core/upload.html')
