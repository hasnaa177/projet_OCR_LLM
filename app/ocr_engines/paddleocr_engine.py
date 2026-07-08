import numpy as np
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            lang="german",
            use_textline_orientation=True,
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _ocr


def run_ocr_paddleocr(pdf_bytes: bytes) -> list:
    pages = convert_from_bytes(pdf_bytes, dpi=200)
    results = []

    for page_number, page_image in enumerate(pages, start=1):
        page_array = np.array(page_image)
        result = _get_ocr().predict(page_array)
        res = result[0] if result else None

        texts = res["rec_texts"] if res else []
        scores = res["rec_scores"] if res else []
        polys = res["rec_polys"] if res else []

        boxes = []
        full_text = []
        for bbox, text, conf in zip(polys, texts, scores):
            cleaned_bbox = [[int(x), int(y)] for x, y in bbox]
            boxes.append({
                "text": text,
                "confidence": round(float(conf), 3),
                "bbox": cleaned_bbox
            })
            full_text.append(text)

        results.append({
            "page_number": page_number,
            "text": " ".join(full_text),
            "bounding_boxes": boxes
        })

    return results