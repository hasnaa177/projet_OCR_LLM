import os
import re
import json
import requests
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import time
from typing import Dict, Any, Optional

# URL de l'API FastAPI qui s'exécute dans Docker
API_URL = "http://localhost:8000"

# Liste complète des 23 champs à évaluer
FIELDS_TO_EVALUATE = [
    "company_name", "legal_form", "date_of_incorporation", "business_address",
    "commercial_register", "vat_id", "property_type", "property_name",
    "property_address", "purchase_price", "financing_amount", "purpose_of_use",
    "equity_contribution", "year_of_construction", "total_area_m2",
    "desired_loan_amount", "term_years", "monthly_installment", "interest_rate",
    "early_repayment", "public_subsidies", "signature_city", "signature_date"
]

# Champs booléens (checkboxes) : on ne peut pas chercher "True"/"False" dans le
# texte OCR, donc leur diagnostic OCR vs LLM doit passer par une autre voie
# (cf. diagnose_field_error). On les liste ici pour ce traitement spécial.
BOOLEAN_FIELDS = {"early_repayment", "public_subsidies"}

# Dossier où sont écrits les fichiers de debug détaillés (un par document),
# contenant le texte OCR brut complet + la sortie LLM complète + la liste des
# erreurs. Permet d'inspecter un document précis sans avoir à tout relire dans
# le terminal.
DEBUG_DIR = Path("./debug_output")


def clean_value(val):
    """Normalise la valeur pour une comparaison équitable."""
    if val is None:
        return "MISSING"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    return str(val).strip().lower()


def compare_values(gt_val, pred_val):
    """Compare la vérité terrain et la prédiction (gère une tolérance pour les normes)."""
    gt = clean_value(gt_val)
    pred = clean_value(pred_val)

    if isinstance(gt, (int, float)) and isinstance(pred, (int, float)):
        # Écart inférieur à 1% considéré comme correct (erreurs d'arrondi OCR)
        if gt == 0: return pred == 0
        return abs(gt - pred) / gt < 0.01

    return gt == pred


def normalize_for_ocr_search(value: str) -> str:
    """
    Nettoie une valeur pour la recherche dans le texte OCR brut, en retirant
    tout ce que le pipeline lui-même retire/normalise avant de comparer
    (symboles monétaires, virgules de milliers, espaces, m²).

    Pourquoi c'est nécessaire : sans ce nettoyage, une valeur comme 517000.0
    ne sera JAMAIS trouvée dans un texte OCR qui contient littéralement
    "€517,000" -- ce qui ferait diagnostiquer à tort une "erreur OCR" alors
    que Tesseract a parfaitement bien lu le montant. C'est le bug principal
    de la version précédente de ce script : il comparait la valeur brute
    sans tenir compte du formatage (devises, séparateurs de milliers, unités).
    """
    s = str(value).strip().lower()
    s = s.replace("€", "").replace("$", "").replace("£", "")
    s = s.replace(",", "").replace(" ", "")
    s = s.replace("m²", "").replace("m2", "")
    # Les nombres flottants type "517000.0" doivent matcher "517000" dans le texte
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_ocr_text_for_search(ocr_text: str) -> str:
    """Applique le même nettoyage que normalize_for_ocr_search, mais sur tout
    le texte OCR, pour que la recherche de sous-chaîne soit cohérente des deux côtés."""
    s = ocr_text.lower()
    s = s.replace("€", "").replace("$", "").replace("£", "")
    s = s.replace(",", "").replace(" ", "")
    s = s.replace("m²", "").replace("m2", "")
    return s


def find_relevant_ocr_snippet(ocr_text: str, field: str, gt_value, window: int = 80) -> str:
    """
    Cherche dans le texte OCR brut la portion de texte la plus probablement liée
    à un champ donné, pour montrer ce que l'OCR a réellement vu à cet endroit --
    avant toute intervention du LLM. Stratégie en cascade :
      1. cherche un libellé de champ connu (ex: "VAT ID" pour vat_id) et montre
         ce qui suit ce libellé sur `window` caractères.
      2. si la valeur attendue (gt_value) ou une sous-chaîne significative
         apparaît ailleurs dans le texte OCR, montre son contexte -- utile
         quand le libellé lui-même a été mal lu mais que la valeur, elle, est
         repérable.
      3. si rien n'est trouvé, le signale explicitement : ça veut dire que
         l'OCR a perdu le passage entièrement (pas seulement mal lu un
         caractère), ce qui oriente le diagnostic différemment.
    """
    if not ocr_text:
        return "(texte OCR vide ou introuvable)"

    # Libellés tels qu'ils apparaissent dans le PDF généré, pour chaque champ.
    # Reflète exactement le texte utilisé dans build_pdf() du générateur.
    field_labels = {
        "company_name": "Company Name",
        "legal_form": "Legal Form",
        "date_of_incorporation": "Date of Incorporation",
        "business_address": "Business Address",
        "commercial_register": "Commercial Register Number",
        "vat_id": "VAT ID",
        "property_type": "Type of Property",
        "property_name": "Property Name",
        "property_address": "Adress",
        "purchase_price": "Purchase Price",
        "financing_amount": "Desired Financing Amount",
        "purpose_of_use": "Purpose of Use",
        "equity_contribution": "Equity Contribution",
        "year_of_construction": "Year of Construction",
        "total_area_m2": "Total Area",
        "desired_loan_amount": "Desired Loan Amount",
        "term_years": "Term",
        "monthly_installment": "Preferred Installment Amount",
        "interest_rate": "Interest Rate",
        "early_repayment": "Early Repayment Desired",
        "public_subsidies": "Public Subsidies Applied",
        "signature_city": None,   # pas de libellé dédié dans le PDF
        "signature_date": None,
    }

    label = field_labels.get(field)
    lower_text = ocr_text.lower()

    if label:
        label_pos = lower_text.find(label.lower())
        if label_pos != -1:
            start = label_pos
            end = min(len(ocr_text), label_pos + len(label) + window)
            return ocr_text[start:end].strip()

    # Fallback : chercher la valeur attendue elle-même dans le texte OCR
    if gt_value is not None:
        gt_str = str(gt_value).strip()
        if gt_str:
            pos = ocr_text.find(gt_str)
            if pos == -1 and len(gt_str) > 4:
                # essai partiel (ex: début du nom de société) si la valeur
                # complète n'apparaît pas telle quelle (cas d'une erreur OCR
                # sur une partie seulement de la valeur)
                pos = ocr_text.find(gt_str[:max(4, len(gt_str)//2)])
            if pos != -1:
                start = max(0, pos - 20)
                end = min(len(ocr_text), pos + len(gt_str) + window)
                return ocr_text[start:end].strip()

    return "(libellé ET valeur introuvables dans le texte OCR -- le passage semble avoir été perdu, pas seulement mal lu)"


def diagnose_field_error(field: str, gt_raw, pred_raw, ocr_text: str) -> str:
    """
    Détermine, pour un champ en erreur, si la cause la plus probable est :
      - "OCR"    : Tesseract a mal lu/perdu la donnée -- elle n'apparaît pas
                   (même normalisée) dans le texte OCR brut.
      - "LLM"    : la donnée correcte EST présente dans le texte OCR brut,
                   donc l'erreur a été introduite après coup, par le LLM ou
                   par le post-traitement (extract.py / postprocess.py).
      - "INDETERMINE" : cas où la recherche textuelle directe ne permet pas
                   de trancher de façon fiable (champs booléens comme les
                   checkboxes, ou champs sans libellé/valeur assez distinctive
                   comme signature_city/signature_date qui sont de simples
                   mots courants).

    Le diagnostic se fait sur les valeurs NORMALISÉES (cf. normalize_for_ocr_search)
    pour éviter les faux positifs "erreur OCR" causés simplement par le
    formatage (€, virgules, m², espaces) que le pipeline lui-même retire.
    """
    if gt_raw is None:
        return "INDETERMINE"

    # Champs booléens : pas de recherche textuelle directe possible pour "True"/
    # "False". On regarde plutôt si la ligne du document contenant le label est
    # présente dans l'OCR ; si elle l'est, on ne peut pas savoir sans relire la
    # ligne nous-mêmes si c'est l'OCR du marqueur "x" ou le LLM qui a fauté --
    # donc INDETERMINE, mais on donne le contexte pour que l'humain tranche.
    if field in BOOLEAN_FIELDS:
        return "INDETERMINE"

    gt_str = str(gt_raw).strip()
    if not gt_str:
        return "INDETERMINE"

    # Champs trop génériques pour une recherche fiable (mots courts/courants
    # qui peuvent apparaître ailleurs dans le document sans rapport avec le champ)
    if field in ("signature_city",) and len(gt_str) < 4:
        return "INDETERMINE"

    gt_normalized = normalize_for_ocr_search(gt_str)
    ocr_normalized = normalize_ocr_text_for_search(ocr_text) if ocr_text else ""

    if not gt_normalized:
        return "INDETERMINE"

    if gt_normalized in ocr_normalized:
        # La valeur correcte est bien présente dans l'OCR (une fois le
        # formatage neutralisé) -> l'OCR a fait son travail, l'erreur vient
        # d'après (LLM ou post-traitement).
        return "LLM"
    else:
        # La valeur correcte n'apparaît nulle part, même normalisée ->
        # Tesseract a mal lu ou perdu le passage.
        return "OCR"


def test_pipeline_on_pdf(pdf_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Envoie le PDF aux 3 endpoints successifs de l'API FastAPI et renvoie un
    dict structuré contenant à la fois :
      - extracted_data : la sortie finale du LLM (ce que le pipeline produit)
      - ocr_text       : le texte OCR brut concaténé de toutes les pages,
                          récupéré via GET /ocr/{doc_id}, pour pouvoir
                          comparer ce que Tesseract a réellement lu avant
                          toute intervention du LLM.
      - doc_id          : l'identifiant du document sur l'API, utile pour
                          investiguer manuellement si besoin.
    """
    result = {"extracted_data": {}, "ocr_text": "", "doc_id": None}

    try:
        # Étape 1 : /upload
        if verbose:
            print(f"    📤 Upload du fichier {pdf_path.name}...")

        with open(pdf_path, "rb") as f:
            up_resp = requests.post(f"{API_URL}/upload", files={"file": f}).json()

        doc_id = up_resp.get("document_id")
        if not doc_id:
            print(f" ❌ Échec du dépôt sur /upload pour {pdf_path.name}")
            print(f"    Réponse: {up_resp}")
            return result

        result["doc_id"] = doc_id

        if verbose:
            print(f"    ✅ Document uploadé avec ID: {doc_id}")

        # Étape 2 : /ocr/{doc_id}
        if verbose:
            print(f"    🔍 Lancement de l'OCR...")

        ocr_resp = requests.post(f"{API_URL}/ocr/{doc_id}")

        if ocr_resp.status_code != 200:
            print(f" ❌ Échec de l'OCR: {ocr_resp.status_code}")
            print(f"    Réponse: {ocr_resp.text}")
            return result

        if verbose:
            print(f"    ✅ OCR terminé")

        # Attendre un peu que l'OCR soit bien enregistré
        time.sleep(1)

        # Récupération du texte OCR brut (GET /ocr/{doc_id}), pour pouvoir
        # comparer ce que l'OCR a réellement vu avant toute action du LLM.
        # Cette étape ne modifie rien côté API -- c'est une simple lecture
        # supplémentaire à but de diagnostic.
        try:
            ocr_get_resp = requests.get(f"{API_URL}/ocr/{doc_id}")
            if ocr_get_resp.status_code == 200:
                ocr_pages = ocr_get_resp.json()
                # Concatène le texte de toutes les pages, dans l'ordre, comme
                # le fait le pipeline lui-même avant le pré-traitement.
                ocr_pages_sorted = sorted(ocr_pages, key=lambda p: p.get("page_number", 0))
                result["ocr_text"] = " ".join(p.get("text", "") for p in ocr_pages_sorted)
            else:
                print(f"    ⚠️ Impossible de récupérer le texte OCR brut (status {ocr_get_resp.status_code})")
        except Exception as e:
            print(f"    ⚠️ Erreur lors de la récupération du texte OCR brut : {e}")

        # Étape 3 : /extract/{doc_id}
        if verbose:
            print(f"    🧠 Extraction des données avec LLM...")

        ext_resp = requests.post(f"{API_URL}/extract/{doc_id}")

        if ext_resp.status_code != 200:
            print(f" ❌ Échec de l'extraction: {ext_resp.status_code}")
            print(f"    Réponse: {ext_resp.text}")
            return result

        ext_json = ext_resp.json()
        result["extracted_data"] = ext_json.get("extracted_data", {})

        if verbose:
            extracted = result["extracted_data"]
            non_null = {k: v for k, v in extracted.items() if v is not None}
            print(f"    ✅ Extraction terminée: {len(non_null)}/{len(FIELDS_TO_EVALUATE)} champs extraits")
            if non_null:
                print(f"    📝 Champs extraits: {', '.join(non_null.keys())}")

        return result

    except requests.exceptions.ConnectionError:
        print(f" ❌ Erreur de connexion à l'API. Vérifiez que FastAPI est en cours d'exécution sur {API_URL}")
        return result
    except Exception as e:
        print(f" ❌ Erreur d'appel API pour {pdf_path.name} : {e}")
        return result


def check_api_health() -> bool:
    """Vérifie que l'API est accessible."""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False


def analyze_missing_fields(results: Dict[str, Dict]) -> Dict[str, int]:
    """Analyse quels champs sont le plus souvent manquants."""
    field_stats = {field: {'present': 0, 'missing': 0} for field in FIELDS_TO_EVALUATE}

    for doc_id, comparisons in results.items():
        for field in FIELDS_TO_EVALUATE:
            if field in comparisons:
                if comparisons[field]['actual'] is None:
                    field_stats[field]['missing'] += 1
                else:
                    field_stats[field]['present'] += 1

    return field_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data", help="Dossier contenant index.json")
    ap.add_argument("--use_scanned", action="store_true", help="Tester sur les versions dégradées/scannées")
    ap.add_argument("--verbose", action="store_true", help="Afficher plus de détails")
    ap.add_argument("--limit", type=int, help="Limiter le nombre de documents à traiter")
    ap.add_argument("--show_ocr_on_error", action="store_true", default=True,
                     help="Affiche l'extrait du texte OCR brut pour chaque champ en erreur (activé par défaut)")
    ap.add_argument("--save_debug_files", action="store_true", default=True,
                     help="Sauvegarde un fichier JSON détaillé par document dans ./debug_output (activé par défaut)")
    args = ap.parse_args()

    # Vérifier que l'API est accessible
    print("🔍 Vérification de l'API...")
    if not check_api_health():
        print("❌ L'API FastAPI n'est pas accessible. Assurez-vous qu'elle est en cours d'exécution.")
        print(f"   Vérifiez que vous avez bien lancé: uvicorn main:app --host 0.0.0.0 --port 8000")
        return
    print("✅ API accessible")

    data_path = Path(args.data_dir)
    index_file = data_path / "index.json"

    if not index_file.exists():
        print(f"❌ Erreur : Impossible de trouver l'index sur {index_file.resolve()}")
        print(f"   Vérifie que tu te trouves bien dans le dossier C:\\pfa-ocr")
        return

    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    # Limiter le nombre de documents si demandé
    if args.limit:
        index_data = index_data[:args.limit]

    if args.save_debug_files:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []
    all_results = {}

    # Compteur global du diagnostic OCR vs LLM vs INDETERMINE, toutes erreurs
    # confondues sur l'ensemble des documents -- c'est ce compteur qui répond
    # directement à la question "le problème vient-il de l'OCR ou du LLM ?"
    diagnosis_counter = {"OCR": 0, "LLM": 0, "INDETERMINE": 0}

    print(f"\n🚀 Début de l'évaluation sur {len(index_data)} documents...\n")

    for idx, item in enumerate(index_data, 1):
        doc_id = item["document_id"]
        print(f"📄 [{idx}/{len(index_data)}] Traitement de {doc_id}...")

        # Résolution robuste des chemins
        gt_path = data_path / "ground_truth" / f"{doc_id}.json"

        if args.use_scanned:
            pdf_path = data_path / "scanned" / f"{doc_id}.pdf"
            if not pdf_path.exists():
                pdf_path = data_path / "pdfs" / f"{doc_id}.pdf"
        else:
            pdf_path = data_path / "pdfs" / f"{doc_id}.pdf"

        # Sécurité : Si un fichier est introuvable localement
        if not pdf_path.exists():
            print(f" ⚠️ Fichier PDF introuvable: {pdf_path}")
            continue

        if not gt_path.exists():
            print(f" ⚠️ Fichier ground truth introuvable: {gt_path}")
            continue

        # Charger la ground truth
        try:
            with open(gt_path, "r", encoding="latin-1") as f:
                gt_json = json.load(f)
        except UnicodeDecodeError:
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_json = json.load(f)
            except:
                print(f" ❌ Impossible de lire le fichier ground truth: {gt_path}")
                continue

        # Appeler le pipeline FastAPI (retourne extracted_data + ocr_text + doc_id)
        pipeline_result = test_pipeline_on_pdf(pdf_path, verbose=args.verbose)
        pred_json = pipeline_result.get("extracted_data", {})
        ocr_text = pipeline_result.get("ocr_text", "")
        api_doc_id = pipeline_result.get("doc_id")

        # Collecter les résultats pour chaque champ
        doc_results = {}
        doc_errors_detail = []  # pour le fichier de debug : liste détaillée des erreurs avec extrait OCR + diagnostic

        for field in FIELDS_TO_EVALUATE:
            gt_raw = gt_json.get(field)
            pred_raw = pred_json.get(field) if pred_json else None

            is_correct = compare_values(gt_raw, pred_raw)

            doc_results[field] = {
                'expected': gt_raw,
                'actual': pred_raw,
                'match': is_correct
            }

            if not is_correct:
                ocr_snippet = find_relevant_ocr_snippet(ocr_text, field, gt_raw) if args.show_ocr_on_error else None
                diagnosis = diagnose_field_error(field, gt_raw, pred_raw, ocr_text)
                diagnosis_counter[diagnosis] += 1

                if pred_raw is None:
                    print(f"    ❌ '{field}': Attendu='{gt_raw}', Obtenu='None'")
                else:
                    print(f"    ❌ '{field}': Attendu='{gt_raw}', Obtenu='{pred_raw}'")

                if ocr_snippet is not None:
                    print(f"        🔍 OCR brut à cet endroit : {ocr_snippet!r}")

                # Message de diagnostic explicite et sans ambiguïté sur la
                # localisation réelle du problème : OCR (Tesseract) ou LLM
                # (Llama 3.1 / post-traitement).
                if diagnosis == "LLM":
                    print(f"        💡 DIAGNOSTIC : erreur LLM — la valeur correcte EST présente dans le texte OCR brut (une fois le formatage neutralisé). Le problème vient du LLM ou du post-traitement, PAS de Tesseract.")
                elif diagnosis == "OCR":
                    print(f"        💡 DIAGNOSTIC : erreur OCR — la valeur correcte n'apparaît nulle part dans le texte OCR brut, même en ignorant le formatage (€, virgules, espaces). Tesseract a mal lu ou perdu ce passage ; corriger le prompt LLM n'aidera pas ici.")
                else:
                    print(f"        💡 DIAGNOSTIC : indéterminé — champ booléen ou valeur trop générique pour trancher par simple recherche textuelle. Inspecter manuellement l'extrait OCR ci-dessus.")

                doc_errors_detail.append({
                    "field": field,
                    "expected": gt_raw,
                    "actual": pred_raw,
                    "ocr_snippet": ocr_snippet,
                    "diagnosis": diagnosis,
                })
            elif args.verbose:
                print(f"    ✅ '{field}': '{gt_raw}'")

            y_true.append("CORRECT")
            y_pred.append("CORRECT" if is_correct else "INCORRECT")

        all_results[doc_id] = doc_results

        # Statistiques pour ce document
        total = len(doc_results)
        correct = sum(1 for v in doc_results.values() if v['match'])
        print(f"    📊 Précision: {correct}/{total} ({correct/total*100:.1f}%)\n")

        # Sauvegarde du fichier de debug détaillé pour ce document : contient
        # le texte OCR brut COMPLET, la sortie LLM complète, la ground truth,
        # et la liste des erreurs avec leur extrait OCR + diagnostic OCR/LLM.
        if args.save_debug_files:
            debug_payload = {
                "doc_id": doc_id,
                "api_doc_id": api_doc_id,
                "ocr_text_full": ocr_text,
                "extracted_data": pred_json,
                "ground_truth": gt_json,
                "errors": doc_errors_detail,
                "accuracy": f"{correct}/{total}",
            }
            debug_path = DEBUG_DIR / f"{doc_id}_debug.json"
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug_payload, f, indent=2, ensure_ascii=False)

    # ==========================================
    # MÉTRIQUES ET AFFICHAGE DES RÉSULTATS
    # ==========================================
    if not y_true:
        print("❌ Aucun fichier n'a pu être traité.")
        return

    print("\n" + "="*50)
    print(" 📊 RÉSULTATS DE L'ÉVALUATION DU PIPELINE")
    print("="*50)

    # 1. Précision Globale
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Précision Globale du modèle (Accuracy) : {accuracy * 100:.2f}%\n")

    # 2. Rapport Détaillé
    print("📋 Rapport de Classification :")
    print(classification_report(y_true, y_pred, labels=["CORRECT", "INCORRECT"], zero_division=0))

    # 3. Matrice de Confusion
    print("📊 Matrice de Confusion (Champs Bien Extraits vs Faux) :")
    cm = confusion_matrix(y_true, y_pred, labels=["CORRECT", "INCORRECT"])
    cm_df = pd.DataFrame(cm, index=["Réel_CORRECT", "Réel_INCORRECT"], columns=["Prédit_CORRECT", "Prédit_INCORRECT"])
    print(cm_df)

    # 4. Analyse des champs manquants
    field_stats = analyze_missing_fields(all_results)

    print("\n📊 Analyse par champ:")
    print("-" * 60)
    print(f"{'Champ':<25} {'Extrait':<10} {'Manquant':<10} {'Taux extraction':<15}")
    print("-" * 60)

    for field in FIELDS_TO_EVALUATE:
        stats = field_stats[field]
        total = stats['present'] + stats['missing']
        if total > 0:
            rate = (stats['present'] / total) * 100
            print(f"{field:<25} {stats['present']:<10} {stats['missing']:<10} {rate:.1f}%")

    # 5. Résumé par document
    print("\n📊 Résumé par document:")
    print("-" * 60)
    for doc_id, doc_results in all_results.items():
        total = len(doc_results)
        correct = sum(1 for v in doc_results.values() if v['match'])
        rate = (correct / total) * 100 if total > 0 else 0
        print(f"{doc_id:<20} {correct}/{total} ({rate:.1f}%)")

    print("\n" + "="*50)

    # 6. Diagnostic global OCR vs LLM -- répond directement à la question
    # "où se trouve réellement le problème ?" sur l'ensemble des erreurs.
    total_errors = sum(diagnosis_counter.values())
    print("\n🔬 DIAGNOSTIC GLOBAL : où se trouvent réellement les erreurs ?")
    print("-" * 60)
    if total_errors == 0:
        print("   Aucune erreur détectée sur ce lot de documents.")
    else:
        for label, key in [("Erreurs OCR (Tesseract)", "OCR"),
                            ("Erreurs LLM / post-traitement", "LLM"),
                            ("Indéterminé (booléens, cas ambigus)", "INDETERMINE")]:
            count = diagnosis_counter[key]
            pct = (count / total_errors * 100) if total_errors else 0
            print(f"   {label:<38} {count:>3} ({pct:.1f}%)")
        print()
        if diagnosis_counter["OCR"] > diagnosis_counter["LLM"]:
            print("   ➜ La majorité des erreurs vient de l'OCR (Tesseract) : améliorer le")
            print("     prompt LLM n'aura qu'un effet limité. Pistes : DPI, qualité scan,")
            print("     config Tesseract (--psm), ou un autre moteur OCR.")
        elif diagnosis_counter["LLM"] > diagnosis_counter["OCR"]:
            print("   ➜ La majorité des erreurs vient du LLM / post-traitement : l'OCR lit")
            print("     correctement les données. Pistes : affiner le prompt, ajouter des")
            print("     règles déterministes de post-traitement, ou fine-tuning du LLM.")
        else:
            print("   ➜ Erreurs réparties à égalité entre OCR et LLM : traiter les deux pistes.")

    # 7. Suggestions d'amélioration
    print("\n💡 Suggestions d'amélioration:")
    low_extraction = [f for f, stats in field_stats.items() if stats['present'] / (stats['present'] + stats['missing']) < 0.5]
    if low_extraction:
        print(f"   - Améliorer l'extraction pour les champs: {', '.join(low_extraction[:5])}")
        if len(low_extraction) > 5:
            print(f"     et {len(low_extraction) - 5} autres...")

    # Identifier les champs avec le plus d'erreurs
    error_fields = {}
    for doc_id, doc_results in all_results.items():
        for field, result in doc_results.items():
            if not result['match'] and result['actual'] is not None:
                error_fields[field] = error_fields.get(field, 0) + 1

    if error_fields:
        most_errors = sorted(error_fields.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   - Champs avec le plus d'erreurs (fausses valeurs): {', '.join([f'{f} ({count})' for f, count in most_errors])}")

    if args.save_debug_files:
        print(f"\n📁 Fichiers de debug détaillés sauvegardés dans : {DEBUG_DIR.resolve()}")
        print(f"   (un fichier {{doc_id}}_debug.json par document, avec texte OCR complet + sortie LLM + erreurs + diagnostic)")


if __name__ == "__main__":
    main()