"""
Évaluation du pipeline OCR + LLM sur le jeu de données synthétique.
Affiche uniquement les champs erronés et la précision globale.

Usage:
    python evaluate_pipeline.py
"""

import json
import requests
from pathlib import Path

API_URL = "http://localhost:8000"
DATA_DIR = Path("./data")


def normalize(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def evaluate_document(doc_id: str, pdf_path: Path, gt: dict) -> tuple:
    # 1. Upload
    with open(pdf_path, "rb") as f:
        resp = requests.post(f"{API_URL}/upload", files={"file": (pdf_path.name, f, "application/pdf")})
    resp.raise_for_status()
    document_id = resp.json()["document_id"]

    # 2. OCR
    resp = requests.post(f"{API_URL}/ocr/{document_id}")
    resp.raise_for_status()

    # 3. Extraction LLM
    resp = requests.post(f"{API_URL}/extract/{document_id}")
    resp.raise_for_status()
    extracted = resp.json().get("extracted_data", {})

    # 4. Comparaison champ par champ
    correct, total = 0, 0
    for field, expected_value in gt.items():
        total += 1
        obtained_value = extracted.get(field)
        if normalize(expected_value) == normalize(obtained_value):
            correct += 1
        else:
            print(f"    ❌ '{field}': Attendu='{expected_value}', Obtenu='{obtained_value}'")

    return correct, total


def main():
    # Vérification API
    print("🔍 Vérification de l'API...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            print("✅ API accessible")
        else:
            print("❌ API inaccessible")
            return
    except Exception:
        print("❌ L'API FastAPI n'est pas accessible.")
        return

    # Chargement de l'index
    index_file = DATA_DIR / "index.json"
    if not index_file.exists():
        print(f"❌ index.json introuvable dans {DATA_DIR}")
        return

    with open(index_file, encoding="utf-8") as f:
        index = json.load(f)

    print(f"🚀 Début de l'évaluation sur {len(index)} documents...")

    total_correct, total_fields = 0, 0

    for i, item in enumerate(index, 1):
        doc_id = item["document_id"]
        pdf_path = DATA_DIR / "pdfs" / f"{doc_id}.pdf"
        gt_path = DATA_DIR / "ground_truth" / f"{doc_id}.json"

        if not pdf_path.exists() or not gt_path.exists():
            print(f"⚠️  Fichiers manquants pour {doc_id}, ignoré.")
            continue

        with open(gt_path, encoding="iso-8859-1") as f:
            gt = json.load(f)

        print(f"🤖 Traitement de {doc_id} ({pdf_path.name})...")
        correct, total = evaluate_document(doc_id, pdf_path, gt)
        total_correct += correct
        total_fields += total

    # Résumé
    accuracy = round(total_correct / total_fields * 100, 2) if total_fields else 0
    print("=" * 50)
    print(f"Précision Globale : {accuracy}% ({total_correct}/{total_fields})")


if __name__ == "__main__":
    main()