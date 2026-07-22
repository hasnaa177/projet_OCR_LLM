import tempfile, os
from markitdown import MarkItDown
from pdf2image import convert_from_bytes
import pytesseract

_md = None

def _get_markitdown():
    global _md
    if _md is None:
        _md = MarkItDown()
    return _md

def run_ocr_markitdown(pdf_bytes: bytes) -> list:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        result = _get_markitdown().convert(tmp_path)
        markdown_text = (result.text_content or "").strip()
    finally:
        os.unlink(tmp_path)

    # Fallback : si le PDF est scanné (pas de couche texte), MarkItDown
    # renvoie ~rien. On complète page par page avec Tesseract.
    if len(markdown_text) < 20:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
        markdown_text = "\n\n".join(
            pytesseract.image_to_string(p, lang="deu+eng") for p in pages
        )

    return [{
        "page_number": 1,
        "text": markdown_text,
        "bounding_boxes": []  # MarkItDown ne fournit pas de bbox, contrairement aux autres
    }]