"""
TextCleaner features for RAG.
--------------------------------------------
Característics:
Optimized for Spanish.
Automatic code detection (UTF-8, ISO-8859-1, etc.).
Eliminacnoise removal (URLs, emails, metadata, OCR).
Parallel and logging process.
It retains accents, ñ, and diacritical marks.
"""

import re
import unicodedata
import logging
from multiprocessing import Pool, cpu_count
from typing import List, Union

try:
    import chardet  # to detect code page
except ImportError:
    chardet = None

try:
    from langdetect import detect
except ImportError:
    detect = None

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# MAIN CLASS
class TextCleaner:
    def __init__(self, target_lang: str = "es", parallel: bool = True):
        self.target_lang = target_lang
        self.parallel = parallel

    def _detect_encoding(self, raw_data: bytes) -> str:
        # Detects code page and return 'utf-8' if it cannot be detected
        if not chardet:
            return "utf-8"
        result = chardet.detect(raw_data)
        encoding = result.get("encoding", "utf-8") or "utf-8"
        confidence = result.get("confidence", 0)
        logging.debug(f"Encoding detection: {encoding} (confidence {confidence:.2f})")
        return encoding

    def _to_unicode(self, text: Union[str, bytes]) -> str:
        # Convert text bytes to str UTF-8, detecting code page.
        if isinstance(text, str):
            return text
        if isinstance(text, bytes):
            encoding = self._detect_encoding(text)
            try:
                return text.decode(encoding, errors="ignore")
            except Exception:
                return text.decode("utf-8", errors="ignore")
        return str(text)

    # --- Atomic cleaning methods ---

    def _normalize_unicode(self, text: str) -> str:
        # Use NFC format (keeps accents and ñ intact).
        return unicodedata.normalize("NFC", text)

    def _remove_control_chars(self, text: str) -> str:
        return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    def _fix_ocr_and_typography(self, text: str) -> str:
        replacements = {
            'ﬁ': 'fi', 'ﬂ': 'fl', '“': '"', '”': '"', '‘': "'", '’': "'",
            '–': '-', '—': '-', '•': '-', '…': '...', '©': '(c)', '®': '(r)',
            'º': 'o', 'ª': 'a'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def _remove_urls_emails(self, text: str) -> str:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        return text

    def _remove_headers_footers(self, text: str) -> str:
        text = re.sub(r'\b[Pp]age\s*\d+\b', '', text)
        text = re.sub(r'\b[Pp]ágina\s*\d+\b', '', text)
        lines = text.split('\n')
        seen, filtered = {}, []
        for line in lines:
            key = line.strip()
            if key and seen.get(key, 0) < 2:
                filtered.append(key)
                seen[key] = seen.get(key, 0) + 1
        return "\n".join(filtered)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _remove_redundant_sections(self, text: str) -> str:
        patterns = [
            r'(?i)este documento es confidencial.*',
            r'(?i)no distribuir sin autorización.*',
            r'(?i)todos los derechos reservados.*'
        ]
        for p in patterns:
            text = re.sub(p, '', text)
        return text

    def _ensure_language_consistency(self, text: str) -> str:
        # Marks texts detected in another language (does not translate).
        if detect:
            try:
                lang = detect(text)
                if lang and lang != self.target_lang:
                    text = f"[{lang.upper()}] {text}"
            except Exception:
                pass
        return text

    # --- Pipeline to complete cleaning ---

    def clean_text(self, text: Union[str, bytes]) -> str:
        # Complete cleaning with coding detection.
        if not text:
            return ""

        text = self._to_unicode(text)
        text = self._normalize_unicode(text)
        text = self._remove_control_chars(text)
        text = self._fix_ocr_and_typography(text)
        text = self._remove_headers_footers(text)
        text = self._remove_urls_emails(text)
        text = self._remove_redundant_sections(text)
        text = self._normalize_whitespace(text)
        text = self._ensure_language_consistency(text)
        return text

    # --- Batch process ---

    def clean_batch(self, texts: List[Union[str, bytes]], show_progress=True) -> List[str]:
        # Parallel cleaning of multiple texts.
        if not texts:
            return []

        if self.parallel and len(texts) > 10:
            workers = min(cpu_count(), 8)
            if show_progress:
                logging.info(f"🧹 Processing {len(texts)} texts using {workers} cores...")

            with Pool(workers) as p:
                cleaned = list(p.map(self.clean_text, texts))
        else:
            cleaned = [self.clean_text(t) for t in texts]

        if show_progress:
            logging.info("✅ Completed cleaning.")
        return cleaned
