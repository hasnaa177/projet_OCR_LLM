import os
import json
import requests
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.metrics import accuracy_score
import Levenshtein  # pip install python-Levenshtein

# ==========================================
# CONFIGURATION
# ==========================================
# Chaque variante OCR tourne sur un port différent dans docker-compose.
# Adapter les ports selon votre configuration.
OCR_APIS = {
    "Tesseract": "http://localhost:8000",
    "EasyOCR":   "http://localhost:8001",
    "Docling":   "http://localhost:8002",
}

FIELDS_TO_EVALUATE = [
    "company_name", "legal_form", "date_of_incorporation", "business_address",
    "commercial_register", "vat_id", "property_type", "property_name",
    "property_address", "purchase_price", "financing_amount", "purpose_of_use",
    "equity_contribution", "year_of_construction", "total_area_m2",
    "desired_loan_amount", "term_years", "monthly_installment", "interest_rate",
    "early_repayment", "public_subsidies", "signature_city", "signature_date"
]

# Champs texte libres (métriques de similarité string)
TEXT_FIELDS = [
    "company_name", "legal_form", "business_address", "commercial_register",
    "property_type", "property_name", "property_address", "purpose_of_use",
    "interest_rate", "signature_city",
]

# Champs numériques (tolérance 1%)
NUMERIC_FIELDS = [
    "purchase_price", "financing_amount", "equity_contribution",
    "total_area_m2", "desired_loan_amount", "monthly_installment",
    "year_of_construction", "term_years",
]

# Champs date (format DD/MM/YYYY)
DATE_FIELDS = ["date_of_incorporation", "signature_date"]

# Champs booléens
BOOL_FIELDS = ["early_repayment", "public_subsidies"]

# Champs identifiants (exactitude stricte)
STRICT_FIELDS = ["vat_id"]

DEBUG_DIR = Path("./debug_ocr_benchmark")


# ==========================================
# MÉTRIQUES
# ==========================================

def clean_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip().lower()


def exact_match(gt, pred) -> bool:
    return clean_str(gt) == clean_str(pred)


def numeric_match(gt, pred, tol=0.01) -> bool:
    try:
        g, p = float(gt), float(pred)
        if g == 0:
            return p == 0
        return abs(g - p) / abs(g) < tol
    except (TypeError, ValueError):
        return False


def character_error_rate(gt, pred) -> float:
    """
    CER = distance d'édition (Levenshtein) / longueur de la référence.
    0.0 = parfait, 1.0 = entièrement faux. Utilisé pour les champs texte libre.
    """
    gt_s = clean_str(gt)
    pred_s = clean_str(pred)
    if not gt_s:
        return 0.0 if not pred_s else 1.0
    return Levenshtein.distance(gt_s, pred_s) / len(gt_s)


def word_error_rate(gt, pred) -> float:
    """
    WER = distance d'édition au niveau mot / nombre de mots de la référence.
    Pertinent pour les adresses et noms composés.
    """
    gt_words   = clean_str(gt).split()
    pred_words = clean_str(pred).split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return Levenshtein.distance(" ".join(gt_words), " ".join(pred_words)) / len(gt_words)


def f1_token(gt, pred) -> float:
    """
    F1 au niveau token (mot) entre gt et pred.
    Utile pour les adresses longues où l'ordre peut varier légèrement.
    """
    gt_tokens   = set(clean_str(gt).split())
    pred_tokens = set(clean_str(pred).split())
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0
    tp = len(gt_tokens & pred_tokens)
    precision = tp / len(pred_tokens)
    recall    = tp / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def date_field_accuracy(gt, pred) -> bool:
    """Comparaison stricte sur les dates DD/MM/YYYY (normalisation des séparateurs)."""
    def normalize_date(d):
        if d is None:
            return ""
        return re.sub(r'[-/.]', '/', str(d).strip())
    return normalize_date(gt) == normalize_date(pred)


def evaluate_field(field: str, gt, pred) -> Dict[str, Any]:
    """
    Calcule toutes les métriques applicables à un champ donné selon sa nature.
    Retourne un dict avec : correct (bool) + métriques spécifiques au type.
    """
    result = {"field": field, "expected": gt, "actual": pred}

    if field in BOOL_FIELDS:
        if gt is None or pred is None:
            result["correct"] = False
            result["type"] = "bool"
        else:
            result["correct"] = bool(gt) == bool(pred)
            result["type"] = "bool"

    elif field in NUMERIC_FIELDS:
        result["correct"] = numeric_match(gt, pred)
        result["type"] = "numeric"
        try:
            g, p = float(gt), float(pred)
            result["abs_error"] = abs(g - p)
            result["rel_error"] = abs(g - p) / abs(g) if g != 0 else 0.0
        except (TypeError, ValueError):
            result["abs_error"] = None
            result["rel_error"] = None

    elif field in DATE_FIELDS:
        result["correct"] = date_field_accuracy(gt, pred)
        result["type"] = "date"
        result["cer"] = character_error_rate(gt, pred)

    elif field in STRICT_FIELDS:
        result["correct"] = exact_match(gt, pred)
        result["type"] = "strict"
        result["cer"] = character_error_rate(gt, pred)

    elif field in TEXT_FIELDS:
        result["correct"] = exact_match(gt, pred)
        result["type"] = "text"
        result["cer"] = character_error_rate(gt, pred)
        result["wer"] = word_error_rate(gt, pred)
        result["f1_token"] = f1_token(gt, pred)

    else:
        result["correct"] = exact_match(gt, pred)
        result["type"] = "other"

    return result


# ==========================================
# APPEL API
# ==========================================

def call_pipeline(api_url: str, pdf_path: Path) -> Dict[str, Any]:
    """Upload → OCR → Extract sur une API donnée. Retourne extracted_data + ocr_text + latences."""
    out = {"extracted_data": {}, "ocr_text": "", "latency_ocr_s": None, "latency_extract_s": None}
    try:
        with open(pdf_path, "rb") as f:
            up = requests.post(f"{api_url}/upload", files={"file": f}, timeout=60).json()
        doc_id = up.get("document_id")
        if not doc_id:
            return out

        t0 = time.time()
        ocr_r = requests.post(f"{api_url}/ocr/{doc_id}", timeout=120)
        out["latency_ocr_s"] = round(time.time() - t0, 2)
        if ocr_r.status_code != 200:
            return out

        time.sleep(0.5)

        ocr_get = requests.get(f"{api_url}/ocr/{doc_id}", timeout=30)
        if ocr_get.status_code == 200:
            pages = sorted(ocr_get.json(), key=lambda p: p.get("page_number", 0))
            out["ocr_text"] = " ".join(p.get("text", "") for p in pages)

        t1 = time.time()
        ext_r = requests.post(f"{api_url}/extract/{doc_id}", timeout=120)
        out["latency_extract_s"] = round(time.time() - t1, 2)
        if ext_r.status_code == 200:
            out["extracted_data"] = ext_r.json().get("extracted_data", {})

    except Exception as e:
        print(f"    ⚠️  Erreur API {api_url} : {e}")
    return out


def check_health(api_url: str, name: str) -> bool:
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        if r.status_code == 200:
            print(f"  ✅ {name} ({api_url}) — accessible")
            return True
    except Exception:
        pass
    print(f"  ❌ {name} ({api_url}) — inaccessible, ignoré")
    return False


# ==========================================
# BENCHMARK PRINCIPAL
# ==========================================

def run_benchmark(data_dir: str = "./data", limit: Optional[int] = None):
    data_path = Path(data_dir)
    index_file = data_path / "index.json"
    if not index_file.exists():
        print(f"❌ index.json introuvable dans {data_path.resolve()}")
        return

    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    if limit:
        index_data = index_data[:limit]

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Vérifier quelles APIs sont disponibles
    print("\n🔍 Vérification des APIs OCR...")
    active_apis = {name: url for name, url in OCR_APIS.items() if check_health(url, name)}
    if not active_apis:
        print("❌ Aucune API accessible.")
        return

    # Structure de résultats : {model: {doc_id: {field: metrics}}}
    all_results: Dict[str, Dict] = {name: {} for name in active_apis}
    latencies:   Dict[str, List] = {name: [] for name in active_apis}

    print(f"\n🚀 Benchmark OCR sur {len(index_data)} documents × {len(active_apis)} modèles\n")

    for idx, item in enumerate(index_data, 1):
        doc_id   = item["document_id"]
        pdf_path = data_path / "pdfs" / f"{doc_id}.pdf"
        gt_path  = data_path / "ground_truth" / f"{doc_id}.json"

        if not pdf_path.exists() or not gt_path.exists():
            print(f"  ⚠️  [{idx}] {doc_id} — fichiers manquants, ignoré")
            continue

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt = json.load(f)
        except Exception:
            with open(gt_path, "r", encoding="latin-1") as f:
                gt = json.load(f)

        print(f"📄 [{idx}/{len(index_data)}] {doc_id}")

        for model_name, api_url in active_apis.items():
            pipeline = call_pipeline(api_url, pdf_path)
            pred     = pipeline["extracted_data"]
            lat_ocr  = pipeline["latency_ocr_s"]
            lat_ext  = pipeline["latency_extract_s"]

            if lat_ocr is not None:
                latencies[model_name].append(lat_ocr)

            doc_field_results = {}
            for field in FIELDS_TO_EVALUATE:
                metrics = evaluate_field(field, gt.get(field), pred.get(field))
                doc_field_results[field] = metrics

            all_results[model_name][doc_id] = doc_field_results

            correct = sum(1 for m in doc_field_results.values() if m["correct"])
            print(f"    [{model_name}] {correct}/{len(FIELDS_TO_EVALUATE)} champs corrects  "
                  f"| OCR: {lat_ocr}s  Extract: {lat_ext}s")

            # Debug par document et par modèle
            debug_path = DEBUG_DIR / f"{doc_id}_{model_name}_debug.json"
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump({
                    "doc_id": doc_id, "model": model_name,
                    "ocr_text": pipeline["ocr_text"],
                    "extracted_data": pred,
                    "ground_truth": gt,
                    "field_results": doc_field_results,
                }, f, indent=2, ensure_ascii=False)

        print()

    # ==========================================
    # AFFICHAGE DES RÉSULTATS COMPARATIFS
    # ==========================================
    print("\n" + "=" * 70)
    print("  📊 RÉSULTATS BENCHMARK OCR — COMPARAISON DES 3 MODÈLES")
    print("=" * 70)

    # --- Tableau 1 : Métriques globales par modèle ---
    print("\n📌 1. MÉTRIQUES GLOBALES\n")
    print(f"{'Métrique':<30}", end="")
    for name in active_apis:
        print(f"{name:>15}", end="")
    print()
    print("-" * (30 + 15 * len(active_apis)))

    global_metrics = {}
    for model_name in active_apis:
        docs = all_results[model_name]
        all_fields = [m for doc in docs.values() for m in doc.values()]

        accuracy   = sum(m["correct"] for m in all_fields) / len(all_fields) if all_fields else 0
        cer_vals   = [m["cer"] for m in all_fields if "cer" in m]
        wer_vals   = [m["wer"] for m in all_fields if "wer" in m]
        f1_vals    = [m["f1_token"] for m in all_fields if "f1_token" in m]
        rel_errs   = [m["rel_error"] for m in all_fields if m.get("rel_error") is not None]
        lats       = latencies[model_name]

        global_metrics[model_name] = {
            "accuracy":       accuracy,
            "avg_cer":        sum(cer_vals) / len(cer_vals) if cer_vals else None,
            "avg_wer":        sum(wer_vals) / len(wer_vals) if wer_vals else None,
            "avg_f1_token":   sum(f1_vals)  / len(f1_vals)  if f1_vals  else None,
            "avg_rel_error":  sum(rel_errs) / len(rel_errs) if rel_errs else None,
            "avg_ocr_lat_s":  sum(lats)     / len(lats)     if lats     else None,
        }

    metric_labels = {
        "accuracy":      "Exactitude globale (%)",
        "avg_cer":       "CER moyen (texte) ↓",
        "avg_wer":       "WER moyen (texte) ↓",
        "avg_f1_token":  "F1-token moyen (texte) ↑",
        "avg_rel_error": "Erreur relative moy. num. ↓",
        "avg_ocr_lat_s": "Latence OCR moy. (s) ↓",
    }

    for key, label in metric_labels.items():
        print(f"{label:<30}", end="")
        for model_name in active_apis:
            val = global_metrics[model_name].get(key)
            if val is None:
                print(f"{'N/A':>15}", end="")
            elif key == "accuracy":
                print(f"{val*100:>14.1f}%", end="")
            elif key == "avg_ocr_lat_s":
                print(f"{val:>13.2f}s", end="")
            else:
                print(f"{val:>15.4f}", end="")
        print()

    # --- Tableau 2 : Exactitude par champ ---
    print("\n\n📌 2. EXACTITUDE PAR CHAMP\n")
    header = f"{'Champ':<26}"
    for name in active_apis:
        header += f"{name:>15}"
    header += f"{'Meilleur':>12}"
    print(header)
    print("-" * (26 + 15 * len(active_apis) + 12))

    n_docs = len(index_data) if not limit else min(limit, len(index_data))

    for field in FIELDS_TO_EVALUATE:
        row = f"{field:<26}"
        field_accs = {}
        for model_name in active_apis:
            docs = all_results[model_name]
            correct = sum(1 for doc in docs.values() if doc.get(field, {}).get("correct", False))
            total   = len(docs)
            acc     = correct / total if total else 0
            field_accs[model_name] = acc
            row += f"{acc*100:>14.1f}%"
        best = max(field_accs, key=field_accs.get) if field_accs else "-"
        row += f"{best:>12}"
        print(row)

    # --- Tableau 3 : CER moyen par champ texte ---
    print("\n\n📌 3. CER MOYEN PAR CHAMP TEXTE (↓ = meilleur)\n")
    header2 = f"{'Champ':<26}"
    for name in active_apis:
        header2 += f"{name:>15}"
    print(header2)
    print("-" * (26 + 15 * len(active_apis)))

    for field in TEXT_FIELDS + DATE_FIELDS + STRICT_FIELDS:
        row = f"{field:<26}"
        for model_name in active_apis:
            docs = all_results[model_name]
            cers = [doc[field]["cer"] for doc in docs.values()
                    if field in doc and "cer" in doc[field]]
            avg = sum(cers) / len(cers) if cers else None
            row += f"{avg:>15.4f}" if avg is not None else f"{'N/A':>15}"
        print(row)

    # --- Tableau 4 : Erreur relative par champ numérique ---
    print("\n\n📌 4. ERREUR RELATIVE MOY. PAR CHAMP NUMÉRIQUE (↓ = meilleur)\n")
    header3 = f"{'Champ':<26}"
    for name in active_apis:
        header3 += f"{name:>15}"
    print(header3)
    print("-" * (26 + 15 * len(active_apis)))

    for field in NUMERIC_FIELDS:
        row = f"{field:<26}"
        for model_name in active_apis:
            docs = all_results[model_name]
            errs = [doc[field]["rel_error"] for doc in docs.values()
                    if field in doc and doc[field].get("rel_error") is not None]
            avg = sum(errs) / len(errs) if errs else None
            row += f"{avg:>15.4f}" if avg is not None else f"{'N/A':>15}"
        print(row)

    # --- Tableau 5 : Résumé par document ---
    print("\n\n📌 5. RÉSUMÉ PAR DOCUMENT\n")
    header4 = f"{'Document':<20}"
    for name in active_apis:
        header4 += f"{name:>18}"
    print(header4)
    print("-" * (20 + 18 * len(active_apis)))

    all_doc_ids = list(list(all_results.values())[0].keys()) if all_results else []
    for doc_id in all_doc_ids:
        row = f"{doc_id:<20}"
        for model_name in active_apis:
            doc = all_results[model_name].get(doc_id, {})
            correct = sum(1 for m in doc.values() if m.get("correct", False))
            total   = len(FIELDS_TO_EVALUATE)
            row += f"{correct}/{total} ({correct/total*100:.0f}%):>15"
        print(row)

    # --- Classement final ---
    print("\n\n📌 6. CLASSEMENT FINAL\n")
    ranked = sorted(active_apis.keys(),
                    key=lambda n: global_metrics[n]["accuracy"], reverse=True)
    for rank, name in enumerate(ranked, 1):
        acc = global_metrics[name]["accuracy"]
        cer = global_metrics[name]["avg_cer"]
        lat = global_metrics[name]["avg_ocr_lat_s"]
        cer_str = f"{cer:.4f}" if cer is not None else "N/A"
        lat_str = f"{lat:.2f}s" if lat is not None else "N/A"
        print(f"  #{rank}  {name:<12}  Accuracy={acc*100:.1f}%  CER={cer_str}  Latence={lat_str}")

    print(f"\n📁 Fichiers debug : {DEBUG_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Benchmark comparatif des moteurs OCR")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--limit", type=int, help="Nombre max de documents")
    args = ap.parse_args()
    run_benchmark(data_dir=args.data_dir, limit=args.limit)