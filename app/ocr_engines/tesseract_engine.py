import pytesseract
from pytesseract import Output
from pdf2image import convert_from_bytes

LANG = "deu+eng"  # cohérent avec paddleocr(lang="german") et easyocr(['de','en'])

def run_ocr_tesseract(pdf_bytes: bytes) -> list:
    pages = convert_from_bytes(pdf_bytes, dpi=200)
    results = []

    for page_number, page_image in enumerate(pages, start=1):
        data = pytesseract.image_to_data(page_image, lang=LANG, output_type=Output.DICT)

        boxes = []
        full_text = []
        n = len(data["text"])
        for i in range(n):
            word = data["text"][i].strip()
            conf = data["conf"][i]
            if not word or conf == "-1":
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            boxes.append({
                "text": word,
                "confidence": round(float(conf) / 100, 3),
                "bbox": bbox
            })
            full_text.append(word)

        results.append({
            "page_number": page_number,
            "text": " ".join(full_text),
            "bounding_boxes": boxes
        })

    return results