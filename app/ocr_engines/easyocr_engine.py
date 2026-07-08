import easyocr
import numpy as np
from pdf2image import convert_from_bytes

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        # Les documents traites contiennent des noms propres et adresses allemandes
        # (Boblingen, WeiSs, Jantsch...). EasyOCR doit charger le jeu de caracteres
        # allemand pour reconnaitre correctement les tremas (a, o, u) et le ss.
        _reader = easyocr.Reader(['de', 'en'], gpu=False)
    return _reader


def run_ocr_easyocr(pdf_bytes: bytes) -> list:
    pages = convert_from_bytes(pdf_bytes, dpi=200)
    results = []
    for page_number, page_image in enumerate(pages, start=1):
        page_array = np.array(page_image)
        ocr_result = _get_reader().readtext(page_array)
        boxes = []
        full_text = []
        for (bbox, text, confidence) in ocr_result:
            cleaned_bbox = [[int(coord[0]), int(coord[1])] for coord in bbox]
            boxes.append({
                "text": text,
                "confidence": round(float(confidence), 3),
                "bbox": cleaned_bbox
            })
            full_text.append(text)
        results.append({
            "page_number": page_number,
            "text": " ".join(full_text),
            "bounding_boxes": boxes
        })
    return results