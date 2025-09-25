import spacy
nlp = spacy.load("en_core_web_sm")
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip().replace("\n", " ")
    return ' '.join(text.split())

def summarize_text(text, max_sentences=3):
    doc = nlp(text)
    sentences = list(doc.sents)
    summary = " ".join([sent.text for sent in sentences[:max_sentences]])
    return summary