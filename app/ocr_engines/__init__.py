import os
import json
from pathlib import Path

OCR_ENGINE = os.getenv("OCR_ENGINE")

def run_ocr(pdf_bytes: bytes) -> list:
    if OCR_ENGINE == "easyocr":
        from .easyocr_engine import run_ocr_easyocr
        return run_ocr_easyocr(pdf_bytes)
    elif OCR_ENGINE == "paddleocr":
        from .paddleocr_engine import run_ocr_paddleocr
        return run_ocr_paddleocr(pdf_bytes)
    elif OCR_ENGINE == "docling":
        from .docling_engine import run_ocr_docling
        return run_ocr_docling(pdf_bytes)
    elif OCR_ENGINE == "tesseract":
        from .tesseract_engine import run_ocr_tesseract
        return run_ocr_tesseract(pdf_bytes)
    else:
        raise ValueError(f"Moteur OCR inconnu : {OCR_ENGINE}")


def needs_easyocr_preprocessing() -> bool:
    return OCR_ENGINE == "easyocr"