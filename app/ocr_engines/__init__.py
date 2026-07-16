import os
import json
from pathlib import Path


def _load_engine_from_config() -> str:
    """
    Lit le champ "engine" de config.json (a la racine du projet, monte dans le
    conteneur via docker-compose en /app/config.json). C'est la source de
    verite pour le moteur OCR utilise par l'API -- la meme que celle lue par
    main.py et mlops_docling.py, pour eviter toute divergence entre eux.
    """
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("engine", "easyocr")
    except FileNotFoundError:
        print(f"[CONFIG] {config_path} introuvable, moteur par defaut 'easyocr' utilise.")
        return "easyocr"
    except json.JSONDecodeError as e:
        print(f"[CONFIG] {config_path} invalide ({e}), moteur par defaut 'easyocr' utilise.")
        return "easyocr"


# OCR_ENGINE (variable d'environnement) reste un override manuel prioritaire,
# utile pour un test local rapide sans toucher a config.json. En son absence,
# c'est config.json qui decide reellement du moteur utilise.
OCR_ENGINE = os.getenv("OCR_ENGINE") or _load_engine_from_config()


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
    else:
        raise ValueError(f"Moteur OCR inconnu : {OCR_ENGINE}")


def needs_easyocr_preprocessing() -> bool:
    return OCR_ENGINE == "easyocr"