"""
promote_best_model.py — Promeut en Production la version du modele
`pfa-ocr-pipeline` qui obtient le meilleur score.

Le score n'est PAS juste global_accuracy_pct : il est pondere par la
couverture du dataset (documents_processed / dataset_size_max), ou
dataset_size_max = le plus grand `dataset_size` observe parmi toutes les
versions existantes (proxy du dataset "complet"). Ca evite de promouvoir
une version testee sur 2 documents avec 100% de precision plutot qu'une
version testee sur tout le dataset avec 95%.

    score = accuracy_pct * (documents_processed / dataset_size_max)

Usage:
    python promote_best_model.py              # calcule + promeut
    python promote_best_model.py --dry-run    # affiche juste le classement
"""
import argparse
import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "pfa-ocr-pipeline"
PRODUCTION_ALIAS = "production"

# Params qui definissent la "configuration" du pipeline. Si deux versions ont
# les memes valeurs ici, ce n'est pas un nouveau modele -- juste un nouveau
# run sur d'autres donnees. Dans ce cas, la promotion n'apporte rien.
# scanned_dpi est inclus car un meme pipeline teste a un DPI different n'est
# pas comparable (scan plus/moins degrade = difficulte differente).
#CONFIG_PARAMS = ["engine", "llm_model", "prompt_version", "scanned_dpi"]
CONFIG_PARAMS = ["engine", "llm_model", "prompt_version"] #si on utilise pas les fichiers scanees 


def get_version_config(client: MlflowClient, mv) -> dict:
    run = client.get_run(mv.run_id)
    return {p: run.data.params.get(p) for p in CONFIG_PARAMS}


def get_version_score(client: MlflowClient, mv, dataset_size_max: int):
    run = client.get_run(mv.run_id)
    metrics = run.data.metrics
    params = run.data.params

    accuracy_pct = metrics.get("global_accuracy_pct")
    documents_processed = metrics.get("documents_processed")
    dataset_size = int(params.get("dataset_size", 0))

    if accuracy_pct is None or documents_processed is None or dataset_size_max == 0:
        return None

    coverage = documents_processed / dataset_size_max
    score = round(accuracy_pct * coverage, 4)
    return {
        "version": mv.version,
        "accuracy_pct": accuracy_pct,
        "documents_processed": int(documents_processed),
        "dataset_size": dataset_size,
        "coverage": round(coverage, 4),
        "score": score,
    }


def promote_best(dry_run: bool = False):
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        print(f"Aucune version trouvee pour '{REGISTERED_MODEL_NAME}'.")
        return

    # Reference "dataset complet" = le plus grand dataset_size vu parmi les runs
    dataset_size_max = 0
    for mv in versions:
        run = client.get_run(mv.run_id)
        ds = int(run.data.params.get("dataset_size", 0))
        dataset_size_max = max(dataset_size_max, ds)

    scored = [s for mv in versions if (s := get_version_score(client, mv, dataset_size_max))]

    if not scored:
        print("Aucune version avec des metriques exploitables "
              "(relance un benchmark avec mlops_docling.py d'abord).")
        return

    scored.sort(key=lambda x: x["score"], reverse=True)

    print(f"{'Version':<10}{'Accuracy%':<12}{'Docs':<8}{'Dataset':<10}{'Coverage':<10}{'Score':<10}")
    for s in scored:
        print(f"{s['version']:<10}{s['accuracy_pct']:<12}{s['documents_processed']:<8}"
              f"{s['dataset_size']:<10}{s['coverage']:<10}{s['score']:<10}")

    best = scored[0]
    print(f"\nMeilleure version : v{best['version']} (score={best['score']})")

    # --- Verification : config identique a la production actuelle ? ---
    best_mv = next(mv for mv in versions if mv.version == best["version"])
    best_config = get_version_config(client, best_mv)

    try:
        current_prod_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    except Exception:
        current_prod_mv = None

    if current_prod_mv is not None:
        current_config = get_version_config(client, current_prod_mv)
        if current_prod_mv.version == best["version"]:
            print(f"v{best['version']} est deja l'alias '{PRODUCTION_ALIAS}'. Rien a faire.")
            return
        if current_config == best_config:
            print(
                f"Config identique a la production actuelle (v{current_prod_mv.version}) : "
                f"{best_config}. Seules les donnees/le score changent, promotion inutile."
            )
            return

    if dry_run:
        print("(--dry-run, aucune promotion effectuee)")
        return

    # Alias moderne (remplace les "stages" depreciees comme Staging/Production)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, best["version"])
    print(f"Alias '{PRODUCTION_ALIAS}' -> v{best['version']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promeut la meilleure version du modele en Production.")
    parser.add_argument("--dry-run", action="store_true", help="Affiche le classement sans promouvoir.")
    args = parser.parse_args()
    promote_best(dry_run=args.dry_run)