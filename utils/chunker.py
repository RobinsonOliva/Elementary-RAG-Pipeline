# utils/chunker.py
def chunk_text(text, max_chars=2000, overlap=200):
    # Divide the text into fragments of maximum size max_chars,
    # with optional overlap between fragments.
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += max_chars - overlap
    return chunks
