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
# Les deux variantes LLM utilisent Tesseract comme OCR commun
# et tournent sur des ports différents dans docker-compose.
LLM_APIS = {
    "Llama3.1 (Ollama)": "http://localhost:8000",
    "Groq (LLaMA-70B)":  "http://localhost:8003",
}

FIELDS_TO_EVALUATE = [
    "company_name", "legal_form", "date_of_incorporation", "business_address",
    "commercial_register", "vat_id", "property_type", "property_name",
    "property_address", "purchase_price", "financing_amount", "purpose_of_use",
    "equity_contribution", "year_of_construction", "total_area_m2",
    "desired_loan_amount", "term_years", "monthly_installment", "interest_rate",
    "early_repayment", "public_subsidies", "signature_city", "signature_date"
]

TEXT_FIELDS = [
    "company_name", "legal_form", "business_address", "commercial_register",
    "property_type", "property_name", "property_address", "purpose_of_use",
    "interest_rate", "signature_city",
]
NUMERIC_FIELDS = [
    "purchase_price", "financing_amount", "equity_contribution",
    "total_area_m2", "desired_loan_amount", "monthly_installment",
    "year_of_construction", "term_years",
]
DATE_FIELDS  = ["date_of_incorporation", "signature_date"]
BOOL_FIELDS  = ["early_repayment", "public_subsidies"]
STRICT_FIELDS = ["vat_id"]

DEBUG_DIR = Path("./debug_llm_benchmark")


# ==========================================
# MÉTRIQUES (identiques benchmark_ocr)
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
    gt_s, pred_s = clean_str(gt), clean_str(pred)
    if not gt_s:
        return 0.0 if not pred_s else 1.0
    return Levenshtein.distance(gt_s, pred_s) / len(gt_s)


def word_error_rate(gt, pred) -> float:
    gt_w   = clean_str(gt).split()
    pred_w = clean_str(pred).split()
    if not gt_w:
        return 0.0 if not pred_w else 1.0
    return Levenshtein.distance(" ".join(gt_w), " ".join(pred_w)) / len(gt_w)


def f1_token(gt, pred) -> float:
    gt_t   = set(clean_str(gt).split())
    pred_t = set(clean_str(pred).split())
    if not gt_t and not pred_t:
        return 1.0
    if not gt_t or not pred_t:
        return 0.0
    tp = len(gt_t & pred_t)
    p  = tp / len(pred_t)
    r  = tp / len(gt_t)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def date_field_accuracy(gt, pred) -> bool:
    def norm(d):
        return re.sub(r'[-/.]', '/', str(d).strip()) if d else ""
    return norm(gt) == norm(pred)


def hallucination_rate(gt, pred) -> bool:
    """
    Détecte une hallucination : le LLM retourne une valeur non nulle alors
    que la ground truth est None (champ absent du document).
    """
    return gt is None and pred is not None


def null_rate(gt, pred) -> bool:
    """
    Détecte un champ manquant : le LLM retourne None alors que la ground
    truth est présente.
    """
    return gt is not None and pred is None


def evaluate_field(field: str, gt, pred) -> Dict[str, Any]:
    result = {
        "field": field, "expected": gt, "actual": pred,
        "hallucination": hallucination_rate(gt, pred),
        "missing":       null_rate(gt, pred),
    }

    if field in BOOL_FIELDS:
        result["correct"] = (bool(gt) == bool(pred)) if (gt is not None and pred is not None) else False
        result["type"] = "bool"

    elif field in NUMERIC_FIELDS:
        result["correct"] = numeric_match(gt, pred)
        result["type"]    = "numeric"
        try:
            g, p = float(gt), float(pred)
            result["abs_error"] = abs(g - p)
            result["rel_error"] = abs(g - p) / abs(g) if g != 0 else 0.0
        except (TypeError, ValueError):
            result["abs_error"] = None
            result["rel_error"] = None

    elif field in DATE_FIELDS:
        result["correct"] = date_field_accuracy(gt, pred)
        result["type"]    = "date"
        result["cer"]     = character_error_rate(gt, pred)

    elif field in STRICT_FIELDS:
        result["correct"] = exact_match(gt, pred)
        result["type"]    = "strict"
        result["cer"]     = character_error_rate(gt, pred)

    elif field in TEXT_FIELDS:
        result["correct"]   = exact_match(gt, pred)
        result["type"]      = "text"
        result["cer"]       = character_error_rate(gt, pred)
        result["wer"]       = word_error_rate(gt, pred)
        result["f1_token"]  = f1_token(gt, pred)

    else:
        result["correct"] = exact_match(gt, pred)
        result["type"]    = "other"

    return result


# ==========================================
# APPEL API
# ==========================================

def call_pipeline(api_url: str, pdf_path: Path) -> Dict[str, Any]:
    """
    Upload → OCR → Extract.
    Mesure séparément la latence OCR et la latence LLM (extract).
    """
    out = {
        "extracted_data": {}, "ocr_text": "",
        "latency_ocr_s": None, "latency_llm_s": None,
        "tokens_approx": None,
    }
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
            ocr_text = " ".join(p.get("text", "") for p in pages)
            out["ocr_text"] = ocr_text
            # Approximation du nombre de tokens envoyés au LLM (4 chars ≈ 1 token)
            out["tokens_approx"] = len(ocr_text) // 4

        t1 = time.time()
        ext_r = requests.post(f"{api_url}/extract/{doc_id}", timeout=180)
        out["latency_llm_s"] = round(time.time() - t1, 2)
        if ext_r.status_code == 200:
            out["extracted_data"] = ext_r.json().get("extracted_data", {})

    except Exception as e:
        print(f"    ⚠️  Erreur API {api_url} : {e}")
    return out


def check_health(api_url: str, name: str) -> bool:
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            llm  = info.get("llm", "?")
            print(f"  ✅ {name} ({api_url}) — LLM: {llm}")
            return True
    except Exception:
        pass
    print(f"  ❌ {name} ({api_url}) — inaccessible, ignoré")
    return False


# ==========================================
# BENCHMARK PRINCIPAL
# ==========================================

def run_benchmark(data_dir: str = "./data", limit: Optional[int] = None):
    data_path  = Path(data_dir)
    index_file = data_path / "index.json"
    if not index_file.exists():
        print(f"❌ index.json introuvable dans {data_path.resolve()}")
        return

    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    if limit:
        index_data = index_data[:limit]

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🔍 Vérification des APIs LLM...")
    active_apis = {name: url for name, url in LLM_APIS.items() if check_health(url, name)}
    if not active_apis:
        print("❌ Aucune API accessible.")
        return

    all_results: Dict[str, Dict] = {name: {} for name in active_apis}
    latencies_llm: Dict[str, List[float]] = {name: [] for name in active_apis}
    latencies_ocr: Dict[str, List[float]] = {name: [] for name in active_apis}
    tokens_counts: Dict[str, List[int]]   = {name: [] for name in active_apis}

    print(f"\n🚀 Benchmark LLM sur {len(index_data)} documents × {len(active_apis)} modèles\n")

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

            if pipeline["latency_ocr_s"] is not None:
                latencies_ocr[model_name].append(pipeline["latency_ocr_s"])
            if pipeline["latency_llm_s"] is not None:
                latencies_llm[model_name].append(pipeline["latency_llm_s"])
            if pipeline["tokens_approx"] is not None:
                tokens_counts[model_name].append(pipeline["tokens_approx"])

            doc_field_results = {}
            for field in FIELDS_TO_EVALUATE:
                metrics = evaluate_field(field, gt.get(field), pred.get(field))
                doc_field_results[field] = metrics

            all_results[model_name][doc_id] = doc_field_results

            correct       = sum(1 for m in doc_field_results.values() if m["correct"])
            hallucs       = sum(1 for m in doc_field_results.values() if m["hallucination"])
            missing_count = sum(1 for m in doc_field_results.values() if m["missing"])

            print(f"    [{model_name}]  {correct}/{len(FIELDS_TO_EVALUATE)} corrects  "
                  f"| Halluc: {hallucs}  Manquants: {missing_count}  "
                  f"| LLM: {pipeline['latency_llm_s']}s  "
                  f"| ~{pipeline['tokens_approx']} tokens")

            debug_path = DEBUG_DIR / f"{doc_id}_{model_name.replace(' ', '_')}_debug.json"
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
    print("  📊 RÉSULTATS BENCHMARK LLM — LLAMA 3.1 vs GROQ")
    print("=" * 70)

    # Calcul des métriques globales
    global_metrics = {}
    for model_name in active_apis:
        docs      = all_results[model_name]
        all_fields = [m for doc in docs.values() for m in doc.values()]

        accuracy    = sum(m["correct"] for m in all_fields) / len(all_fields) if all_fields else 0
        cer_vals    = [m["cer"]      for m in all_fields if "cer"      in m]
        wer_vals    = [m["wer"]      for m in all_fields if "wer"      in m]
        f1_vals     = [m["f1_token"] for m in all_fields if "f1_token" in m]
        rel_errs    = [m["rel_error"] for m in all_fields if m.get("rel_error") is not None]
        hallucs     = [m["hallucination"] for m in all_fields]
        missings    = [m["missing"]       for m in all_fields]
        bool_fields = [m for m in all_fields if m["type"] == "bool"]
        lats_llm    = latencies_llm[model_name]
        lats_ocr    = latencies_ocr[model_name]
        toks        = tokens_counts[model_name]

        global_metrics[model_name] = {
            "accuracy":           accuracy,
            "avg_cer":            sum(cer_vals)  / len(cer_vals)  if cer_vals  else None,
            "avg_wer":            sum(wer_vals)  / len(wer_vals)  if wer_vals  else None,
            "avg_f1_token":       sum(f1_vals)   / len(f1_vals)   if f1_vals   else None,
            "avg_rel_error_num":  sum(rel_errs)  / len(rel_errs)  if rel_errs  else None,
            "hallucination_rate": sum(hallucs)   / len(hallucs)   if hallucs   else None,
            "missing_rate":       sum(missings)  / len(missings)  if missings  else None,
            "bool_accuracy":      sum(m["correct"] for m in bool_fields) / len(bool_fields)
                                  if bool_fields else None,
            "avg_llm_lat_s":      sum(lats_llm)  / len(lats_llm)  if lats_llm  else None,
            "avg_ocr_lat_s":      sum(lats_ocr)  / len(lats_ocr)  if lats_ocr  else None,
            "avg_tokens":         sum(toks)       / len(toks)       if toks      else None,
        }

    # --- Tableau 1 : Métriques globales ---
    print("\n📌 1. MÉTRIQUES GLOBALES\n")
    print(f"{'Métrique':<35}", end="")
    for name in active_apis:
        print(f"{name:>22}", end="")
    print()
    print("-" * (35 + 22 * len(active_apis)))

    metric_labels = {
        "accuracy":           "Exactitude globale (%) ↑",
        "avg_cer":            "CER moyen texte ↓",
        "avg_wer":            "WER moyen texte ↓",
        "avg_f1_token":       "F1-token moyen texte ↑",
        "avg_rel_error_num":  "Erreur relative moy. num. ↓",
        "hallucination_rate": "Taux hallucination ↓",
        "missing_rate":       "Taux champs manquants ↓",
        "bool_accuracy":      "Précision checkboxes (%) ↑",
        "avg_llm_lat_s":      "Latence LLM moy. (s) ↓",
        "avg_ocr_lat_s":      "Latence OCR moy. (s)",
        "avg_tokens":         "Tokens envoyés moy.",
    }

    for key, label in metric_labels.items():
        print(f"{label:<35}", end="")
        for model_name in active_apis:
            val = global_metrics[model_name].get(key)
            if val is None:
                print(f"{'N/A':>22}", end="")
            elif key in ("accuracy", "bool_accuracy"):
                print(f"{val*100:>21.1f}%", end="")
            elif key in ("hallucination_rate", "missing_rate"):
                print(f"{val*100:>21.2f}%", end="")
            elif key in ("avg_llm_lat_s", "avg_ocr_lat_s"):
                print(f"{val:>20.2f}s", end="")
            elif key == "avg_tokens":
                print(f"{val:>21.0f}", end="")
            else:
                print(f"{val:>22.4f}", end="")
        print()

    # --- Tableau 2 : Exactitude par champ ---
    print("\n\n📌 2. EXACTITUDE PAR CHAMP\n")
    header = f"{'Champ':<26}"
    for name in active_apis:
        header += f"{name:>22}"
    header += f"{'Meilleur':>14}"
    print(header)
    print("-" * (26 + 22 * len(active_apis) + 14))

    for field in FIELDS_TO_EVALUATE:
        row = f"{field:<26}"
        field_accs = {}
        for model_name in active_apis:
            docs    = all_results[model_name]
            correct = sum(1 for doc in docs.values() if doc.get(field, {}).get("correct", False))
            total   = len(docs)
            acc     = correct / total if total else 0
            field_accs[model_name] = acc
            row += f"{acc*100:>21.1f}%"
        best = max(field_accs, key=field_accs.get) if field_accs else "-"
        row += f"{best:>14}"
        print(row)

    # --- Tableau 3 : Hallucinations et champs manquants par champ ---
    print("\n\n📌 3. HALLUCINATIONS & CHAMPS MANQUANTS PAR CHAMP\n")
    header3 = f"{'Champ':<26}"
    for name in active_apis:
        header3 += f"{'Halluc ' + name[:8]:>16}{'Manq ' + name[:8]:>14}"
    print(header3)
    print("-" * (26 + 30 * len(active_apis)))

    for field in FIELDS_TO_EVALUATE:
        row = f"{field:<26}"
        for model_name in active_apis:
            docs = all_results[model_name]
            hallucs  = sum(1 for doc in docs.values()
                           if doc.get(field, {}).get("hallucination", False))
            missings = sum(1 for doc in docs.values()
                           if doc.get(field, {}).get("missing", False))
            n = len(docs)
            row += f"{hallucs/n*100:>15.1f}%{missings/n*100:>13.1f}%"
        print(row)

    # --- Tableau 4 : CER par champ texte ---
    print("\n\n📌 4. CER MOYEN PAR CHAMP TEXTE (↓ = meilleur)\n")
    header4 = f"{'Champ':<26}"
    for name in active_apis:
        header4 += f"{name:>22}"
    print(header4)
    print("-" * (26 + 22 * len(active_apis)))

    for field in TEXT_FIELDS + DATE_FIELDS + STRICT_FIELDS:
        row = f"{field:<26}"
        for model_name in active_apis:
            docs = all_results[model_name]
            cers = [doc[field]["cer"] for doc in docs.values()
                    if field in doc and "cer" in doc[field]]
            avg  = sum(cers) / len(cers) if cers else None
            row += f"{avg:>22.4f}" if avg is not None else f"{'N/A':>22}"
        print(row)

    # --- Tableau 5 : Erreur relative champs numériques ---
    print("\n\n📌 5. ERREUR RELATIVE MOY. PAR CHAMP NUMÉRIQUE (↓ = meilleur)\n")
    header5 = f"{'Champ':<26}"
    for name in active_apis:
        header5 += f"{name:>22}"
    print(header5)
    print("-" * (26 + 22 * len(active_apis)))

    for field in NUMERIC_FIELDS:
        row = f"{field:<26}"
        for model_name in active_apis:
            docs = all_results[model_name]
            errs = [doc[field]["rel_error"] for doc in docs.values()
                    if field in doc and doc[field].get("rel_error") is not None]
            avg  = sum(errs) / len(errs) if errs else None
            row += f"{avg:>22.4f}" if avg is not None else f"{'N/A':>22}"
        print(row)

    # --- Tableau 6 : Résumé par document ---
    print("\n\n📌 6. RÉSUMÉ PAR DOCUMENT\n")
    header6 = f"{'Document':<20}"
    for name in active_apis:
        header6 += f"{name:>28}"
    print(header6)
    print("-" * (20 + 28 * len(active_apis)))

    all_doc_ids = list(list(all_results.values())[0].keys()) if all_results else []
    for doc_id in all_doc_ids:
        row = f"{doc_id:<20}"
        for model_name in active_apis:
            doc     = all_results[model_name].get(doc_id, {})
            correct = sum(1 for m in doc.values() if m.get("correct", False))
            total   = len(FIELDS_TO_EVALUATE)
            hallucs = sum(1 for m in doc.values() if m.get("hallucination", False))
            row += f"  {correct}/{total} ({correct/total*100:.0f}%)  H:{hallucs:>2}"
        print(row)

    # --- Classement final ---
    print("\n\n📌 7. CLASSEMENT FINAL\n")
    ranked = sorted(active_apis.keys(),
                    key=lambda n: global_metrics[n]["accuracy"], reverse=True)
    print(f"{'Rang':<6}{'Modèle':<25}{'Accuracy':>12}{'CER':>10}{'Halluc.':>10}{'Manquants':>12}{'Lat. LLM':>12}")
    print("-" * 87)
    for rank, name in enumerate(ranked, 1):
        m   = global_metrics[name]
        acc = m["accuracy"]
        cer = m["avg_cer"]
        hal = m["hallucination_rate"]
        mis = m["missing_rate"]
        lat = m["avg_llm_lat_s"]
        print(f"#{rank:<5}{name:<25}{acc*100:>11.1f}%"
              f"{cer:>10.4f}" if cer else f"{'N/A':>10}",
              end="")
        print(f"{hal*100:>9.2f}%" if hal is not None else f"{'N/A':>10}", end="")
        print(f"{mis*100:>11.2f}%" if mis is not None else f"{'N/A':>12}", end="")
        print(f"{lat:>10.2f}s" if lat is not None else f"{'N/A':>12}")

    print(f"\n📁 Fichiers debug : {DEBUG_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Benchmark comparatif des LLMs (Llama vs Groq)")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--limit", type=int, help="Nombre max de documents")
    args = ap.parse_args()
    run_benchmark(data_dir=args.data_dir, limit=args.limit)