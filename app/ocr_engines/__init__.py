import os

OCR_ENGINE = os.getenv("OCR_ENGINE", "docling")


def run_ocr(pdf_bytes: bytes) -> list:
    if OCR_ENGINE == "docling":
        from .docling_engine import run_ocr_docling
        return run_ocr_docling(pdf_bytes)
    elif OCR_ENGINE == "marker":
        from .marker_engine import run_ocr_marker
        return run_ocr_marker(pdf_bytes)
    else:
        raise ValueError(f"Moteur OCR inconnu : {OCR_ENGINE}")


def needs_easyocr_preprocessing() -> bool:
    # Plus aucun moteur "image + prétraitement texte" dans ce benchmark
    return False