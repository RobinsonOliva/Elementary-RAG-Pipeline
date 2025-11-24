# loaders/docx_loader.py
from docx import Document

def load_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])
