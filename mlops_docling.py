"""
mlops_docling.py — Benchmark comparatif entre moteurs OCR/structuration, avec
tracking MLflow. Variante de benchmark_ocr_engines.py dediee au suivi MLOps.

Enrichissements par rapport a la version de base :
  - Taux normalises (%) plutot que des comptes bruts (comparables entre runs
    de tailles differentes) : hallucination_rate_pct, missing_rate_pct,
    incorrect_rate_pct.
  - perfect_document_rate_pct : % de documents extraits sans AUCUNE erreur.
  - extraction_completeness_pct : % de champs non-null renvoyes par le LLM,
    independamment de leur exactitude (isole un bug "le LLM ne renvoie rien"
    d'un simple probleme de precision).
  - Precision par categorie de champ (numerique / booleen / date / texte).
  - Precision par champ individuel, loggee comme metrique MLflow nommee
    (field_acc_<field>) pour comparaison directe entre runs dans l'UI MLflow.
  - Latences en percentiles (p50/p95) plutot que la seule moyenne.
  - Debit (documents_per_minute) et temps total du run.
  - avg_ocr_char_count : proxy de completude de l'OCR en amont du LLM.
  - Export CSV par champ, loggee comme artefact MLflow.

Usage:
    python mlops_docling.py --engine docling
    python mlops_docling.py --engine marker
    python mlops_docling.py --compare docling marker
    mlflow ui --port 5000
"""
import argparse
import csv
import json
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
import mlflow

import sys

API_URL = "http://localhost:8000"
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./benchmark_results")
CONTAINER_NAME = "pfa-ocr-api-1"

# Sous-dossiers possibles sous DATA_DIR pour les PDF a traiter. "scanned"
# correspond aux fichiers produits par generate_fake_loan_data.py --degrade
# (meme nommage {doc_id}.pdf, meme ground_truth/ associe -- seul le rendu
# visuel change : niveaux de gris, rotation, bruit, sans couche texte, ce
# qui stresse davantage l'OCR).
SOURCE_FOLDERS = {"pdfs": "pdfs", "scanned": "scanned"}

# A mettre a jour manuellement chaque fois que le prompt dans main.py change,
# pour garder une correspondance claire dans l'historique MLflow.
PROMPT_VERSION = "prompt1"
LLM_MODEL = "llama3.1"

# ==========================================
# 0. TYPOLOGIE DES CHAMPS (pour les metriques par categorie)
# ==========================================
ALL_FIELDS = [
    "company_name", "legal_form", "date_of_incorporation", "business_address",
    "commercial_register", "vat_id", "property_type", "property_name",
    "property_address", "purchase_price", "financing_amount", "purpose_of_use",
    "equity_contribution", "year_of_construction", "total_area_m2",
    "desired_loan_amount", "term_years", "monthly_installment", "interest_rate",
    "early_repayment", "public_subsidies", "signature_city", "signature_date",
]
NUMERIC_FIELDS = {
    "purchase_price", "financing_amount", "equity_contribution",
    "year_of_construction", "total_area_m2", "desired_loan_amount",
    "term_years", "monthly_installment",
}
BOOLEAN_FIELDS = {"early_repayment", "public_subsidies"}
DATE_FIELDS = {"date_of_incorporation", "signature_date"}
TEXT_FIELDS = set(ALL_FIELDS) - NUMERIC_FIELDS - BOOLEAN_FIELDS - DATE_FIELDS

FIELD_CATEGORIES = {
    "numeric": NUMERIC_FIELDS,
    "boolean": BOOLEAN_FIELDS,
    "date": DATE_FIELDS,
    "text": TEXT_FIELDS,
}


# ==========================================
# 1. MESURE MEMOIRE (echantillonnage en arriere-plan pendant les appels API)
# ==========================================
class MemorySampler:
    """
    Echantillonne `docker stats` en arriere-plan pendant qu'une requete bloquante
    (upload/ocr/extract) est en cours, pour capturer le pic memoire du conteneur.
    """

    def __init__(self, container_name: str, interval: float = 0.3):
        self.container_name = container_name
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self.samples_mib = []

    def _parse_mem_usage(self, raw: str) -> float:
        """Convertit '123.4MiB / 7.6GiB' -> 123.4 (en MiB)."""
        try:
            used_part = raw.split("/")[0].strip()
            if used_part.endswith("GiB"):
                return float(used_part.replace("GiB", "")) * 1024
            if used_part.endswith("MiB"):
                return float(used_part.replace("MiB", ""))
            if used_part.endswith("KiB"):
                return float(used_part.replace("KiB", "")) / 1024
        except (ValueError, IndexError):
            pass
        return 0.0

    def _run(self):
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", self.container_name],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    self.samples_mib.append(self._parse_mem_usage(result.stdout.strip()))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.samples_mib = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples_mib:
            return {"peak_mib": None, "avg_mib": None, "samples": 0}
        return {
            "peak_mib": round(max(self.samples_mib), 1),
            "avg_mib": round(sum(self.samples_mib) / len(self.samples_mib), 1),
            "samples": len(self.samples_mib),
        }


# ==========================================
# 2. COMPARAISON CHAMP PAR CHAMP (precision + hallucinations)
# ==========================================
def normalize(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        # Representation numerique canonique : evite les faux mismatchs du
        # type 4316000 (int) vs 4316000.0 (float) qui sont la meme valeur.
        return format(float(value), ".6f").rstrip("0").rstrip(".") or "0"
    s = str(value).strip().lower()
    try:
        f = float(s.replace(",", ""))
        return format(f, ".6f").rstrip("0").rstrip(".") or "0"
    except ValueError:
        return s


def compare_fields(expected: dict, obtained: dict) -> dict:
    """
    Classe chaque champ en 4 categories :
    - correct       : valeur obtenue == valeur attendue
    - incorrect      : les deux ont une valeur, mais differentes (erreur d'extraction)
    - hallucinated   : le ground truth est null/absent MAIS le LLM a produit une valeur
                        (signal fort d'hallucination : il invente une info absente du document)
    - missing        : le ground truth a une valeur MAIS le LLM a renvoye null
                        (sous-extraction, differente d'une hallucination)
    """
    correct, incorrect, hallucinated, missing = [], [], [], []

    for field, expected_value in expected.items():
        obtained_value = obtained.get(field) if obtained else None
        exp_norm = normalize(expected_value)
        obt_norm = normalize(obtained_value)

        if exp_norm == obt_norm:
            correct.append(field)
        elif exp_norm == "" and obt_norm != "":
            hallucinated.append({"field": field, "obtained": obtained_value})
        elif exp_norm != "" and obt_norm == "":
            missing.append({"field": field, "expected": expected_value})
        else:
            incorrect.append({"field": field, "expected": expected_value, "obtained": obtained_value})

    total = len(expected)
    return {
        "total_fields": total,
        "correct": len(correct),
        "correct_fields": correct,  # liste des noms, necessaire pour l'agregation par champ
        "incorrect": incorrect,
        "hallucinated": hallucinated,
        "missing": missing,
        "accuracy_pct": round(len(correct) / total * 100, 2) if total else 0,
    }


def print_field_errors(doc_label: str, comparison: dict) -> None:
    """
    Affiche, pour un document, le detail lisible de chaque champ errone :
    ce qui etait attendu (ground truth) vs ce que le pipeline a produit.
    N'affiche rien si le document est parfait (aucune ligne inutile en cas de succes).
    """
    incorrect = comparison["incorrect"]
    missing = comparison["missing"]
    hallucinated = comparison["hallucinated"]

    if not (incorrect or missing or hallucinated):
        return

    print(f"    Détail des écarts pour {doc_label} :")

    for item in incorrect:
        print(
            f"      [INCORRECT]   {item['field']:<25} "
            f"attendu={item['expected']!r:<30} obtenu={item['obtained']!r}"
        )
    for item in missing:
        print(
            f"      [MANQUANT]    {item['field']:<25} "
            f"attendu={item['expected']!r:<30} obtenu=None"
        )
    for item in hallucinated:
        print(
            f"      [HALLUCINE]   {item['field']:<25} "
            f"attendu=None{'':<26} obtenu={item['obtained']!r}"
        )


def print_global_error_report(per_document_results: list) -> None:
    """
    Regroupe TOUTES les erreurs (incorrect/manquant/hallucine) de tous les documents
    par nom de champ, pour reperer d'un coup d'oeil les champs systematiquement
    problematiques (ex: 'financing_amount' faux sur 6 documents/10) plutot que
    d'avoir a recouper document par document.
    """
    per_field = defaultdict(lambda: {"incorrect": 0, "missing": 0, "hallucinated": 0, "examples": []})

    for result in per_document_results:
        doc_label = result["doc_id_ref"]
        comparison = result["comparison"]

        for item in comparison["incorrect"]:
            stats = per_field[item["field"]]
            stats["incorrect"] += 1
            if len(stats["examples"]) < 3:
                stats["examples"].append(
                    f"{doc_label}: attendu={item['expected']!r} / obtenu={item['obtained']!r}"
                )
        for item in comparison["missing"]:
            stats = per_field[item["field"]]
            stats["missing"] += 1
            if len(stats["examples"]) < 3:
                stats["examples"].append(f"{doc_label}: attendu={item['expected']!r} / obtenu=None")
        for item in comparison["hallucinated"]:
            stats = per_field[item["field"]]
            stats["hallucinated"] += 1
            if len(stats["examples"]) < 3:
                stats["examples"].append(f"{doc_label}: attendu=None / obtenu={item['obtained']!r}")

    if not per_field:
        print("\nAucune erreur detectee sur l'ensemble du dataset.")
        return

    ranked = sorted(
        per_field.items(),
        key=lambda kv: kv[1]["incorrect"] + kv[1]["missing"] + kv[1]["hallucinated"],
        reverse=True,
    )

    print("\n" + "=" * 78)
    print("RAPPORT DETAILLE DES CHAMPS ERRONES (agrege sur tout le dataset)")
    print("=" * 78)
    for field, stats in ranked:
        total_errors = stats["incorrect"] + stats["missing"] + stats["hallucinated"]
        print(
            f"\n{field}  -- {total_errors} erreur(s) au total "
            f"(incorrect={stats['incorrect']}, manquant={stats['missing']}, "
            f"hallucine={stats['hallucinated']})"
        )
        for example in stats["examples"]:
            print(f"    ex: {example}")
    print("=" * 78)


# ==========================================
# 2bis. METRIQUES DE HAUT NIVEAU (normalisees, MLOps-friendly)
# ==========================================
def _percentile(data: list, p: float):
    """Percentile par interpolation lineaire, sans dependance a numpy."""
    if not data:
        return None
    data_sorted = sorted(data)
    if len(data_sorted) == 1:
        return round(data_sorted[0], 2)
    k = (len(data_sorted) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return round(data_sorted[f], 2)
    return round(data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f), 2)


def aggregate_field_level_stats(per_document_results: list) -> dict:
    """
    Agrege correct/incorrect/missing/hallucinated par nom de champ sur tout le
    dataset, et calcule une precision par champ. Base pour les metriques
    field_acc_<field> et pour le CSV exporte.
    """
    per_field = defaultdict(lambda: {"correct": 0, "incorrect": 0, "missing": 0, "hallucinated": 0})

    for result in per_document_results:
        c = result["comparison"]
        for f in c.get("correct_fields", []):
            per_field[f]["correct"] += 1
        for item in c["incorrect"]:
            per_field[item["field"]]["incorrect"] += 1
        for item in c["missing"]:
            per_field[item["field"]]["missing"] += 1
        for item in c["hallucinated"]:
            per_field[item["field"]]["hallucinated"] += 1

    field_level = {}
    for field, stats in per_field.items():
        total = stats["correct"] + stats["incorrect"] + stats["missing"] + stats["hallucinated"]
        field_level[field] = {
            **stats,
            "total": total,
            "accuracy_pct": round(stats["correct"] / total * 100, 2) if total else 0.0,
        }
    return field_level


def aggregate_category_stats(field_level: dict) -> dict:
    """Precision agregee par categorie de champ (numerique/booleen/date/texte)."""
    category_stats = {}
    for cat_name, fields in FIELD_CATEGORIES.items():
        correct = sum(field_level.get(f, {}).get("correct", 0) for f in fields)
        total = sum(field_level.get(f, {}).get("total", 0) for f in fields)
        category_stats[cat_name] = {
            "correct": correct,
            "total": total,
            "accuracy_pct": round(correct / total * 100, 2) if total else 0.0,
        }
    return category_stats


def compute_advanced_metrics(per_document_results: list) -> dict:
    """
    Metriques de haut niveau, normalisees pour rester comparables d'un run a
    l'autre meme si le nombre de documents traites avec succes differe :
      - taux (%) plutot que comptes bruts
      - taux de documents parfaits (0 erreur)
      - completude d'extraction (isole un LLM qui ne renvoie rien)
      - latences en percentiles plutot que moyenne seule
      - proxy de completude OCR (nb de caracteres extraits)
    """
    if not per_document_results:
        return {}

    total_fields = sum(r["comparison"]["total_fields"] for r in per_document_results)
    total_correct = sum(r["comparison"]["correct"] for r in per_document_results)
    total_incorrect = sum(len(r["comparison"]["incorrect"]) for r in per_document_results)
    total_hallucinated = sum(len(r["comparison"]["hallucinated"]) for r in per_document_results)
    total_missing = sum(len(r["comparison"]["missing"]) for r in per_document_results)

    perfect_docs = sum(1 for r in per_document_results if r["comparison"]["accuracy_pct"] == 100.0)

    completeness_values = []
    for r in per_document_results:
        extracted = r.get("extracted_data") or {}
        total_schema_fields = r["comparison"]["total_fields"]
        if total_schema_fields:
            non_null = sum(1 for v in extracted.values() if v is not None)
            completeness_values.append(non_null / total_schema_fields * 100)

    ocr_times = [r["timings"]["ocr_seconds"] for r in per_document_results]
    extract_times = [r["timings"]["extract_seconds"] for r in per_document_results]
    total_times = [r["timings"]["total_seconds"] for r in per_document_results]
    ocr_chars = [r.get("ocr_char_count", 0) for r in per_document_results]

    return {
        "perfect_document_rate_pct": round(perfect_docs / len(per_document_results) * 100, 2),
        "incorrect_rate_pct": round(total_incorrect / total_fields * 100, 2) if total_fields else 0.0,
        "hallucination_rate_pct": round(total_hallucinated / total_fields * 100, 2) if total_fields else 0.0,
        "missing_rate_pct": round(total_missing / total_fields * 100, 2) if total_fields else 0.0,
        "extraction_completeness_pct": (
            round(sum(completeness_values) / len(completeness_values), 2) if completeness_values else 0.0
        ),
        "ocr_p50_seconds": _percentile(ocr_times, 50),
        "ocr_p95_seconds": _percentile(ocr_times, 95),
        "extract_p50_seconds": _percentile(extract_times, 50),
        "extract_p95_seconds": _percentile(extract_times, 95),
        "total_p50_seconds": _percentile(total_times, 50),
        "total_p95_seconds": _percentile(total_times, 95),
        "avg_ocr_char_count": round(sum(ocr_chars) / len(ocr_chars), 1) if ocr_chars else 0.0,
    }


def write_field_level_csv(engine_label: str, field_level: dict) -> Path:
    """Exporte le detail par champ en CSV, trie par precision croissante (les pires en premier)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"field_level_{engine_label}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "correct", "incorrect", "missing", "hallucinated", "total", "accuracy_pct"])
        for field, stats in sorted(field_level.items(), key=lambda kv: kv[1]["accuracy_pct"]):
            writer.writerow([
                field, stats["correct"], stats["incorrect"], stats["missing"],
                stats["hallucinated"], stats["total"], stats["accuracy_pct"],
            ])
    return path


def print_high_level_summary(advanced: dict, category_stats: dict, field_level: dict) -> None:
    print("\n" + "-" * 60)
    print("METRIQUES DE HAUT NIVEAU")
    print("-" * 60)
    print(f"  Documents parfaits (0 erreur) : {advanced['perfect_document_rate_pct']}%")
    print(f"  Complétude d'extraction        : {advanced['extraction_completeness_pct']}% (champs non-null)")
    print(f"  Taux d'erreur                  : {advanced['incorrect_rate_pct']}%")
    print(f"  Taux d'hallucination           : {advanced['hallucination_rate_pct']}%")
    print(f"  Taux de champs manquants       : {advanced['missing_rate_pct']}%")
    print(f"  Latence OCR      p50/p95 (s)   : {advanced['ocr_p50_seconds']} / {advanced['ocr_p95_seconds']}")
    print(f"  Latence LLM      p50/p95 (s)   : {advanced['extract_p50_seconds']} / {advanced['extract_p95_seconds']}")
    print(f"  Latence totale   p50/p95 (s)   : {advanced['total_p50_seconds']} / {advanced['total_p95_seconds']}")
    print(f"  Caracteres OCR moyens/doc      : {advanced['avg_ocr_char_count']}")

    print("\n  Précision par catégorie de champ :")
    for cat_name, stats in category_stats.items():
        print(f"    {cat_name:<10} : {stats['accuracy_pct']}% ({stats['correct']}/{stats['total']})")

    ranked_fields = sorted(field_level.items(), key=lambda kv: kv[1]["accuracy_pct"])
    worst = [f for f in ranked_fields if f[1]["total"] > 0][:3]
    if worst:
        print("\n  Champs les plus problématiques :")
        for field, stats in worst:
            print(f"    {field:<25} {stats['accuracy_pct']}% ({stats['correct']}/{stats['total']})")
    print("-" * 60)


# ==========================================
# 3. TRAITEMENT D'UN DOCUMENT (avec chronometrage + memoire)
# ==========================================
def process_document(doc_id_ref: str, pdf_path: Path, gt: dict, sampler: MemorySampler) -> dict:
    timings = {}

    with open(pdf_path, "rb") as f:
        resp = requests.post(f"{API_URL}/upload", files={"file": (pdf_path.name, f, "application/pdf")})
    resp.raise_for_status()
    document_id = resp.json()["document_id"]

    sampler.start()
    t0 = time.perf_counter()
    resp = requests.post(f"{API_URL}/ocr/{document_id}")
    resp.raise_for_status()
    timings["ocr_seconds"] = round(time.perf_counter() - t0, 2)
    mem_ocr = sampler.stop()

    # Proxy de completude de l'OCR : nb de caracteres extraits, recupere depuis
    # la reponse deja retournee par /ocr/{id} (pas d'appel API supplementaire).
    ocr_char_count = sum(len(page.get("text", "")) for page in resp.json().get("results", []))

    sampler.start()
    t0 = time.perf_counter()
    resp = requests.post(f"{API_URL}/extract/{document_id}")
    resp.raise_for_status()
    timings["extract_seconds"] = round(time.perf_counter() - t0, 2)
    mem_extract = sampler.stop()

    extracted = resp.json().get("extracted_data", {})
    timings["total_seconds"] = round(timings["ocr_seconds"] + timings["extract_seconds"], 2)

    comparison = compare_fields(gt, extracted)

    return {
        "doc_id_ref": doc_id_ref,
        "document_id": document_id,
        "timings": timings,
        "memory": {"ocr": mem_ocr, "extract": mem_extract},
        "comparison": comparison,
        "extracted_data": extracted,
        "ocr_char_count": ocr_char_count,
    }


# ==========================================
# 4. RUN COMPLET SUR LE DATASET (+ tracking MLflow)
# ==========================================
def run_benchmark(engine_label: str, show_errors: bool = True, source: str = "pdfs", fail_under=None, data_dir: Path = None, scanned_dpi: int = None):
    print(f"Verification de l'API...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        print("API accessible")
    except Exception:
        print("L'API FastAPI n'est pas accessible.")
        return

    if data_dir is None:
        data_dir = DATA_DIR

    index_file = data_dir / "index.json"
    if not index_file.exists():
        print(f"index.json introuvable dans {data_dir}")
        return

    with open(index_file, encoding="utf-8") as f:
        index = json.load(f)

    # Dossier source des PDF : "pdfs" (propres) ou "scanned" (degrades, generes
    # par generate_fake_loan_data.py --degrade). Meme doc_id, meme ground_truth/
    # pour les deux -- seul le rendu visuel du PDF change.
    pdf_source_dir = data_dir / SOURCE_FOLDERS[source]
    if not pdf_source_dir.exists():
        print(f"Dossier introuvable : {pdf_source_dir}")
        if source == "scanned":
            print("Generez d'abord les versions scannees avec :")
            print(f"  python generate_fake_loan_data.py --count {len(index)} --outdir {data_dir} --degrade")
        return
    print(f"Source des PDF : {pdf_source_dir}")

    # Suffixe le nom du run/des fichiers de resultats par la source pour ne
    # jamais ecraser un run "pdfs" avec un run "scanned" du meme moteur.
    run_label = f"{engine_label}_{source}" if source != "pdfs" else engine_label

    mlflow.set_experiment("pfa-ocr-benchmark")

    with mlflow.start_run(run_name=f"{run_label}_{PROMPT_VERSION}"):
        mlflow.log_param("engine", engine_label)
        mlflow.log_param("source", source)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("llm_model", LLM_MODEL)
        mlflow.log_param("dataset_size", len(index))
        mlflow.set_tag("run_date_utc", datetime.now(timezone.utc).isoformat())
        try:
            git_sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=3
            ).stdout.strip()
            if git_sha:
                mlflow.set_tag("git_commit", git_sha)
        except Exception:
            pass

        sampler = MemorySampler(CONTAINER_NAME)
        per_document_results = []
        run_start = time.perf_counter()

        print(f"Benchmark du moteur '{engine_label}' sur {len(index)} documents...")

        for i, item in enumerate(index, 1):
            doc_id = item["document_id"]
            pdf_path = pdf_source_dir / f"{doc_id}.pdf"
            gt_path = data_dir / "ground_truth" / f"{doc_id}.json"
            if not pdf_path.exists() or not gt_path.exists():
                print(f"[{i}/{len(index)}] Fichiers manquants pour {doc_id} "
                      f"(pdf={pdf_path.exists()}, ground_truth={gt_path.exists()}), ignore.")
                continue

            with open(gt_path, encoding="iso-8859-1") as f:
                gt = json.load(f)

            print(f"[{i}/{len(index)}] Traitement de {doc_id} ({pdf_path.name})...")
            try:
                result = process_document(doc_id, pdf_path, gt, sampler)
            except Exception as e:
                print(f"    Erreur sur {doc_id} : {e}")
                continue

            per_document_results.append(result)

            c = result["comparison"]
            print(
                f"    precision {c['accuracy_pct']}% | "
                f"hallucines {len(c['hallucinated'])} | manquants {len(c['missing'])} | "
                f"temps total {result['timings']['total_seconds']}s | "
                f"pic memoire (ocr) {result['memory']['ocr']['peak_mib']} MiB"
            )

            if show_errors:
                print_field_errors(doc_id, c)

            mlflow.log_metric("accuracy_pct", c["accuracy_pct"], step=i)
            mlflow.log_metric("doc_time_seconds", result["timings"]["total_seconds"], step=i)

        run_duration_seconds = round(time.perf_counter() - run_start, 2)

        # --- Agregation globale (comptes bruts, conserves pour compatibilite) ---
        total_fields = sum(r["comparison"]["total_fields"] for r in per_document_results)
        total_correct = sum(r["comparison"]["correct"] for r in per_document_results)
        total_hallucinated = sum(len(r["comparison"]["hallucinated"]) for r in per_document_results)
        total_missing = sum(len(r["comparison"]["missing"]) for r in per_document_results)
        total_incorrect = sum(len(r["comparison"]["incorrect"]) for r in per_document_results)

        avg_total_time = round(
            sum(r["timings"]["total_seconds"] for r in per_document_results) / len(per_document_results), 2
        ) if per_document_results else 0

        peak_memories = [
            r["memory"]["ocr"]["peak_mib"] for r in per_document_results
            if r["memory"]["ocr"]["peak_mib"] is not None
        ]
        global_peak_mem = round(max(peak_memories), 1) if peak_memories else None

        # --- Metriques de haut niveau (normalisees) ---
        advanced = compute_advanced_metrics(per_document_results)
        field_level = aggregate_field_level_stats(per_document_results)
        category_stats = aggregate_category_stats(field_level)
        documents_per_minute = (
            round(len(per_document_results) / (run_duration_seconds / 60), 2)
            if run_duration_seconds > 0 and per_document_results else 0.0
        )

        summary = {
            "engine": engine_label,
            "documents_processed": len(per_document_results),
            "total_fields": total_fields,
            "total_correct": total_correct,
            "total_incorrect": total_incorrect,
            "total_hallucinated": total_hallucinated,
            "total_missing": total_missing,
            "global_accuracy_pct": round(total_correct / total_fields * 100, 2) if total_fields else 0,
            "avg_time_per_doc_seconds": avg_total_time,
            "peak_memory_mib": global_peak_mem,
            "run_duration_seconds": run_duration_seconds,
            "documents_per_minute": documents_per_minute,
            **advanced,
        }

        summary["source"] = source

        RESULTS_DIR.mkdir(exist_ok=True)
        output_path = RESULTS_DIR / f"results_{run_label}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "field_level": field_level,
                    "category_stats": category_stats,
                    "documents": per_document_results,
                },
                f, ensure_ascii=False, indent=2,
            )
        csv_path = write_field_level_csv(run_label, field_level)

        # --- Logging MLflow ---
        mlflow.log_metric("global_accuracy_pct", summary["global_accuracy_pct"])
        mlflow.log_metric("documents_processed", summary["documents_processed"])
        mlflow.log_metric("total_incorrect", total_incorrect)
        mlflow.log_metric("total_hallucinated", total_hallucinated)
        mlflow.log_metric("total_missing", total_missing)
        mlflow.log_metric("avg_time_per_doc_seconds", avg_total_time)
        mlflow.log_metric("run_duration_seconds", run_duration_seconds)
        mlflow.log_metric("documents_per_minute", documents_per_minute)
        if global_peak_mem is not None:
            mlflow.log_metric("peak_memory_mib", global_peak_mem)

        for key, value in advanced.items():
            if value is not None:
                mlflow.log_metric(key, value)

        for cat_name, stats in category_stats.items():
            mlflow.log_metric(f"accuracy_{cat_name}_fields_pct", stats["accuracy_pct"])

        for field, stats in field_level.items():
            mlflow.log_metric(f"field_acc_{field}", stats["accuracy_pct"])

        mlflow.log_artifact(str(output_path))
        mlflow.log_artifact(str(csv_path))



        # --- Enregistrement dans le Model Registry (nouvelle version) ---
        from mlflow_registry import register_pipeline_model
        register_pipeline_model(
            engine=engine_label,
            llm_model=LLM_MODEL,
            prompt_version=PROMPT_VERSION,
            accuracy_pct=summary["global_accuracy_pct"],
            registered_model_name="pfa-ocr-pipeline",
            stage="Staging" if summary["global_accuracy_pct"] >= 80 else None,
        )




        print("=" * 60)
        print(f"Resume pour '{engine_label}' (source: {source}) :")
        print(f"  Precision globale     : {summary['global_accuracy_pct']}% ({total_correct}/{total_fields})")
        print(f"  Champs incorrects     : {total_incorrect}")
        print(f"  Champs hallucines     : {total_hallucinated}")
        print(f"  Champs manquants      : {total_missing}")
        print(f"  Temps moyen/document  : {avg_total_time}s")
        print(f"  Pic memoire observe   : {global_peak_mem} MiB")
        print(f"  Debit                 : {documents_per_minute} documents/minute")
        print(f"Resultats sauvegardes dans : {output_path}")
        print(f"Détail par champ sauvegardé dans : {csv_path}")

        if per_document_results:
            print_high_level_summary(advanced, category_stats, field_level)

        if fail_under is not None and summary["global_accuracy_pct"] < fail_under:
            print(f"\nECHEC : precision {summary['global_accuracy_pct']}% sous le seuil requis {fail_under}%")
            sys.exit(1)    

        if show_errors:
            print_global_error_report(per_document_results)


# ==========================================
# 5. COMPARAISON ENTRE DEUX RUNS DEJA SAUVEGARDES
# ==========================================
def compare_engines(engine_a: str, engine_b: str, show_errors: bool = True, source: str = "pdfs"):
    # Les labels de fichiers suivent la meme convention que dans run_benchmark :
    # pas de suffixe pour la source "pdfs" (comportement par defaut d'origine),
    # suffixe "_scanned" sinon -- pour comparer deux moteurs sur le meme jeu
    # (propre ou scanne), pas un moteur "scanned" contre un moteur "pdfs".
    label_a = f"{engine_a}_{source}" if source != "pdfs" else engine_a
    label_b = f"{engine_b}_{source}" if source != "pdfs" else engine_b
    path_a = RESULTS_DIR / f"results_{label_a}.json"
    path_b = RESULTS_DIR / f"results_{label_b}.json"

    if not path_a.exists() or not path_b.exists():
        print(f"Fichiers manquants. Lance d'abord :")
        if not path_a.exists():
            print(f"  python mlops_docling.py --engine {engine_a} --source {source}")
        if not path_b.exists():
            print(f"  python mlops_docling.py --engine {engine_b} --source {source}")
        return

    with open(path_a, encoding="utf-8") as f:
        full_a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        full_b = json.load(f)

    data_a = full_a["summary"]
    data_b = full_b["summary"]

    rows = [
        ("Documents traites", data_a["documents_processed"], data_b["documents_processed"]),
        ("Precision globale (%)", data_a["global_accuracy_pct"], data_b["global_accuracy_pct"]),
        ("Documents parfaits (%)", data_a.get("perfect_document_rate_pct"), data_b.get("perfect_document_rate_pct")),
        ("Completude extraction (%)", data_a.get("extraction_completeness_pct"), data_b.get("extraction_completeness_pct")),
        ("Champs corrects", data_a["total_correct"], data_b["total_correct"]),
        ("Champs incorrects", data_a["total_incorrect"], data_b["total_incorrect"]),
        ("Taux hallucination (%)", data_a.get("hallucination_rate_pct"), data_b.get("hallucination_rate_pct")),
        ("Taux manquant (%)", data_a.get("missing_rate_pct"), data_b.get("missing_rate_pct")),
        ("Temps moyen/doc (s)", data_a["avg_time_per_doc_seconds"], data_b["avg_time_per_doc_seconds"]),
        ("Latence totale p95 (s)", data_a.get("total_p95_seconds"), data_b.get("total_p95_seconds")),
        ("Debit (docs/min)", data_a.get("documents_per_minute"), data_b.get("documents_per_minute")),
        ("Pic memoire (MiB)", data_a["peak_memory_mib"], data_b["peak_memory_mib"]),
    ]

    col1_width = max(len(r[0]) for r in rows) + 2
    col2_width = max(len(engine_a), 12)
    col3_width = max(len(engine_b), 12)

    print("=" * (col1_width + col2_width + col3_width + 6))
    print(f"{'Metrique':<{col1_width}} | {engine_a:^{col2_width}} | {engine_b:^{col3_width}}")
    print("-" * (col1_width + col2_width + col3_width + 6))
    for label, val_a, val_b in rows:
        print(f"{label:<{col1_width}} | {str(val_a):^{col2_width}} | {str(val_b):^{col3_width}}")
    print("=" * (col1_width + col2_width + col3_width + 6))

    if data_a["global_accuracy_pct"] > data_b["global_accuracy_pct"]:
        print(f"\n-> {engine_a} a une meilleure precision globale.")
    elif data_b["global_accuracy_pct"] > data_a["global_accuracy_pct"]:
        print(f"\n-> {engine_b} a une meilleure precision globale.")
    else:
        print(f"\n-> Precision identique entre les deux moteurs.")

    if data_a["total_hallucinated"] != data_b["total_hallucinated"]:
        moins_hallu = engine_a if data_a["total_hallucinated"] < data_b["total_hallucinated"] else engine_b
        print(f"-> {moins_hallu} hallucine moins de champs.")

    if data_a["avg_time_per_doc_seconds"] != data_b["avg_time_per_doc_seconds"]:
        plus_rapide = engine_a if data_a["avg_time_per_doc_seconds"] < data_b["avg_time_per_doc_seconds"] else engine_b
        print(f"-> {plus_rapide} est plus rapide en moyenne.")

    # --- Comparaison par categorie de champ (numerique/booleen/date/texte) ---
    cat_a = full_a.get("category_stats")
    cat_b = full_b.get("category_stats")
    if cat_a and cat_b:
        print("\nPrécision par catégorie de champ :")
        for cat_name in FIELD_CATEGORIES:
            acc_a = cat_a.get(cat_name, {}).get("accuracy_pct")
            acc_b = cat_b.get(cat_name, {}).get("accuracy_pct")
            print(f"  {cat_name:<10} : {engine_a}={acc_a}%  |  {engine_b}={acc_b}%")

    if show_errors:
        print(f"\n--- Detail des champs errones pour '{engine_a}' ---")
        print_global_error_report(full_a["documents"])
        print(f"\n--- Detail des champs errones pour '{engine_b}' ---")
        print_global_error_report(full_b["documents"])


# ==========================================
# 6. CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark comparatif de moteurs OCR/structuration, avec tracking MLflow.")
    parser.add_argument("--engine", type=str, help="Nom du moteur actuellement actif dans le conteneur (ex: docling, marker). Lance un run complet et sauvegarde les resultats.")
    parser.add_argument("--compare", nargs=2, metavar=("ENGINE_A", "ENGINE_B"), help="Compare deux runs deja sauvegardes (ex: --compare docling marker).")
    parser.add_argument("--fail-under", type=float, default=None, help="Fait echouer le script (exit code 1) si global_accuracy_pct est sous ce seuil. Utile pour la CI.")
    parser.add_argument("--quiet-errors", action="store_true", help="N'affiche pas le detail des champs errones (attendu vs obtenu), seulement les resumes chiffres.")
    parser.add_argument(
        "--source", choices=sorted(SOURCE_FOLDERS.keys()), default="pdfs",
        help="Quel jeu de PDF traiter : 'pdfs' (propres, defaut) ou 'scanned' "
             "(versions degradees generees par generate_fake_loan_data.py --degrade). "
             "S'applique a --engine comme a --compare.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Dossier racine du dataset a utiliser (defaut: ./data). "
             "Ex: ./data_scanee_100 pour tester sur les PDF degrades a 100 DPI.",
    )
    parser.add_argument(
    "--scanned-dpi", type=int, default=None,
    help="DPI des PDF scannes traites (ex: 150). Si fourni, ecrase toute deduction "
         "automatique depuis index.json -- utile car un dossier donne (ex: "
         "data_scanee_150) contient toujours un DPI unique et connu a l'avance.",
)
    args = parser.parse_args()

    show_errors = not args.quiet_errors

    if args.engine:
        run_benchmark(
        args.engine, show_errors=show_errors, source=args.source,
        fail_under=args.fail_under, data_dir=Path(args.data_dir),
        scanned_dpi=args.scanned_dpi,
    )
    elif args.compare:
        compare_engines(args.compare[0], args.compare[1], show_errors=show_errors, source=args.source)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()