"""
Benchmark comparatif entre moteurs OCR/structuration (ex: Docling vs Marker).
Mesure : précision champ par champ, hallucinations, temps de réponse, mémoire conteneur.

Comme un seul conteneur `api` tourne à la fois (un seul OCR_ENGINE actif), ce script
s'exécute une fois PAR MOTEUR : bascule OCR_ENGINE dans .env, recrée le conteneur,
relance ce script avec --engine <nom>. Chaque run sauvegarde un fichier JSON détaillé
sous benchmark_results/. Une fois les deux runs faits, utilise --compare pour
générer le tableau comparatif final.

Usage:
    # Run 1 (apres avoir bascule OCR_ENGINE=docling et --force-recreate)
    python benchmark_ocr_engines.py --engine docling

    # Run 2 (apres avoir bascule OCR_ENGINE=marker et --force-recreate)
    python benchmark_ocr_engines.py --engine marker

    # Comparaison finale (n'a pas besoin de l'API, lit juste les JSON sauvegardes)
    python benchmark_ocr_engines.py --compare docling marker

Options d'affichage detaille :
    --show-errors           Affiche, pour chaque document, le detail des champs
                             incorrects/manquants/hallucines (attendu vs obtenu).
    --quiet-errors           Desactive cet affichage (comportement d'origine, resume seul).
"""

import argparse
import json
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import requests

API_URL = "http://localhost:8000"
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./benchmark_results")
CONTAINER_NAME = "pfa-ocr-api-1" 

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
        # Representation numerique canonique : evite les faux mismatchs
        # du type 4316000 (int) vs 4316000.0 (float) qui sont la meme valeur.
        return format(float(value), ".6f").rstrip("0").rstrip(".") or "0"
    s = str(value).strip().lower()
    # Si c'est une chaine qui represente en fait un nombre (ex: "4316000"
    # cote ground truth vs 4316000.0 cote API), on la normalise pareil.
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

    # Tri par nombre total d'erreurs decroissant : les champs les plus problematiques en premier
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
# 3. TRAITEMENT D'UN DOCUMENT (avec chronometrage + memoire)
# ==========================================
def process_document(doc_id_ref: str, pdf_path: Path, gt: dict, sampler: MemorySampler) -> dict:
    timings = {}

    # --- Upload (non chronometre pour le benchmark moteur, c'est juste I/O disque) ---
    with open(pdf_path, "rb") as f:
        resp = requests.post(f"{API_URL}/upload", files={"file": (pdf_path.name, f, "application/pdf")})
    resp.raise_for_status()
    document_id = resp.json()["document_id"]

    # --- OCR (chronometre + memoire) ---
    sampler.start()
    t0 = time.perf_counter()
    resp = requests.post(f"{API_URL}/ocr/{document_id}")
    resp.raise_for_status()
    timings["ocr_seconds"] = round(time.perf_counter() - t0, 2)
    mem_ocr = sampler.stop()

    # --- Extraction LLM (chronometre + memoire) ---
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
    }


# ==========================================
# 4. RUN COMPLET SUR LE DATASET
# ==========================================
def run_benchmark(engine_label: str, show_errors: bool = True):
    print(f"Verification de l'API...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        print("API accessible")
    except Exception:
        print("L'API FastAPI n'est pas accessible.")
        return

    index_file = DATA_DIR / "index.json"
    if not index_file.exists():
        print(f"index.json introuvable dans {DATA_DIR}")
        return

    with open(index_file, encoding="utf-8") as f:
        index = json.load(f)

    sampler = MemorySampler(CONTAINER_NAME)
    per_document_results = []

    print(f"Benchmark du moteur '{engine_label}' sur {len(index)} documents...")

    for i, item in enumerate(index, 1):
        doc_id = item["document_id"]
        pdf_path = DATA_DIR / "pdfs" / f"{doc_id}.pdf"
        gt_path = DATA_DIR / "ground_truth" / f"{doc_id}.json"

        if not pdf_path.exists() or not gt_path.exists():
            print(f"[{i}/{len(index)}] Fichiers manquants pour {doc_id}, ignore.")
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

    # --- Agregation globale ---
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
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"results_{engine_label}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "documents": per_document_results}, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Resume pour '{engine_label}' :")
    print(f"  Precision globale     : {summary['global_accuracy_pct']}% ({total_correct}/{total_fields})")
    print(f"  Champs incorrects     : {total_incorrect}")
    print(f"  Champs hallucines     : {total_hallucinated}")
    print(f"  Champs manquants      : {total_missing}")
    print(f"  Temps moyen/document  : {avg_total_time}s")
    print(f"  Pic memoire observe   : {global_peak_mem} MiB")
    print(f"Resultats sauvegardes dans : {output_path}")

    if show_errors:
        print_global_error_report(per_document_results)


# ==========================================
# 5. COMPARAISON ENTRE DEUX RUNS DEJA SAUVEGARDES
# ==========================================
def compare_engines(engine_a: str, engine_b: str, show_errors: bool = True):
    path_a = RESULTS_DIR / f"results_{engine_a}.json"
    path_b = RESULTS_DIR / f"results_{engine_b}.json"

    if not path_a.exists() or not path_b.exists():
        print(f"Fichiers manquants. Lance d'abord :")
        if not path_a.exists():
            print(f"  python benchmark_ocr_engines.py --engine {engine_a}")
        if not path_b.exists():
            print(f"  python benchmark_ocr_engines.py --engine {engine_b}")
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
        ("Champs corrects", data_a["total_correct"], data_b["total_correct"]),
        ("Champs incorrects", data_a["total_incorrect"], data_b["total_incorrect"]),
        ("Champs hallucines", data_a["total_hallucinated"], data_b["total_hallucinated"]),
        ("Champs manquants", data_a["total_missing"], data_b["total_missing"]),
        ("Temps moyen/doc (s)", data_a["avg_time_per_doc_seconds"], data_b["avg_time_per_doc_seconds"]),
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

    # Verdict rapide sur la precision
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

    if show_errors:
        print(f"\n--- Detail des champs errones pour '{engine_a}' ---")
        print_global_error_report(full_a["documents"])
        print(f"\n--- Detail des champs errones pour '{engine_b}' ---")
        print_global_error_report(full_b["documents"])


# ==========================================
# 6. CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark comparatif de moteurs OCR/structuration.")
    parser.add_argument("--engine", type=str, help="Nom du moteur actuellement actif dans le conteneur (ex: docling, marker). Lance un run complet et sauvegarde les resultats.")
    parser.add_argument("--compare", nargs=2, metavar=("ENGINE_A", "ENGINE_B"), help="Compare deux runs deja sauvegardes (ex: --compare docling marker).")
    parser.add_argument("--quiet-errors", action="store_true", help="N'affiche pas le detail des champs errones (attendu vs obtenu), seulement les resumes chiffres.")
    args = parser.parse_args()

    show_errors = not args.quiet_errors

    if args.engine:
        run_benchmark(args.engine, show_errors=show_errors)
    elif args.compare:
        compare_engines(args.compare[0], args.compare[1], show_errors=show_errors)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()