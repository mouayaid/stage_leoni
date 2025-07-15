import openai
import os
from dotenv import load_dotenv

# تحميل مفتاح API من .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization="org-ruk7j93n7f6fhEenUXySqV32"
openai.project="proj_SBTJ0zB0j5X90EXMROI7wUX5"

def gpt_extract_root_cause(reason_text):
    """
    استخراج السبب الجذري فقط من تحليل Why باستخدام GPT
    """
    prompt = f"""
Voici une analyse de causes utilisant la méthode 5 Why :
{reason_text}

Quelle est la cause racine identifiée à la fin de cette analyse ?
Donne uniquement la cause racine sous forme de phrase simple.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ GPT Error (extraction): {e}")
        return "Root cause non extraite"

def gpt_score_root_cause(reason_text, finding_text):
    """
    تقييم الأسباب (Reasons) باستعمال GPT حسب barème مرن
    """
    prompt = f"""
Tu es un auditeur qualité expérimenté.

Voici un Finding : "{finding_text}"
Et voici une analyse de causes :
{reason_text}

Évalue cette analyse selon :
- Clarté
- Lien logique avec le finding
- Structure logique de Why (peu importe le nombre)
- Profondeur d’analyse
- Présence de causes secondaires

Donne une évaluation sous ce format :
Score: X/5
Statut: Accepted ou Not Accepted
Commentaire: ...
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ GPT Error (score): {e}")
        return "Échec d'évaluation"
