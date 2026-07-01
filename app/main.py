import os
import uuid
import json
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from io import BytesIO
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import create_engine, text
from minio import Minio
from minio.error import S3Error
from pdf2image import convert_from_bytes
from PIL import Image
from paddleocr import PaddleOCR
from pydantic import BaseModel
import ollama

# ==========================================
# SCHÉMA PYDANTIC – 23 champs
# ==========================================
class CreditExtractionSchema(BaseModel):
    company_name: Optional[str] = None
    legal_form: Optional[str] = None
    date_of_incorporation: Optional[str] = None
    business_address: Optional[str] = None
    commercial_register: Optional[str] = None
    vat_id: Optional[str] = None
    property_type: Optional[str] = None
    property_name: Optional[str] = None
    property_address: Optional[str] = None
    purchase_price: Optional[float] = None
    financing_amount: Optional[float] = None
    purpose_of_use: Optional[str] = None
    equity_contribution: Optional[float] = None
    year_of_construction: Optional[int] = None
    total_area_m2: Optional[float] = None
    desired_loan_amount: Optional[float] = None
    term_years: Optional[int] = None
    monthly_installment: Optional[float] = None
    interest_rate: Optional[str] = None
    early_repayment: Optional[bool] = None
    public_subsidies: Optional[bool] = None
    signature_city: Optional[str] = None
    signature_date: Optional[str] = None


# ==========================================
# APP FASTAPI
# ==========================================
app = FastAPI(title="PFA OCR – Dossiers de Crédit Immobilier Commercial")

# ==========================================
# BASE DE DONNÉES – PostgreSQL
# ==========================================
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://pfa:pfa123@postgres:5432/pfa_ocr"
        )
        _engine = create_engine(database_url)
    return _engine


def init_db():
    with get_engine().connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                filename TEXT,
                blob_path TEXT,
                status TEXT DEFAULT 'recu',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ocr_results (
                id UUID PRIMARY KEY,
                document_id UUID,
                page_number INTEGER,
                text TEXT,
                bounding_boxes JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS extractions (
                id UUID PRIMARY KEY,
                document_id UUID UNIQUE,
                extracted_data JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()


# ==========================================
# MINIO – Stockage objet
# ==========================================
BUCKET_NAME = "credit-documents"

def get_minio_client() -> Minio:
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False
    )


def init_storage():
    client = get_minio_client()
    try:
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            print(f"✅ Bucket '{BUCKET_NAME}' créé")
        else:
            print(f"✅ Bucket '{BUCKET_NAME}' déjà existant")
    except S3Error as e:
        print(f"⚠️  Erreur MinIO init : {e}")


def save_file(file_bytes: bytes, filename: str) -> str:
    """Sauvegarde dans MinIO, retourne le blob_path."""
    client = get_minio_client()
    blob_path = f"uploads/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
    client.put_object(
        BUCKET_NAME,
        blob_path,
        BytesIO(file_bytes),
        length=len(file_bytes),
        content_type="application/pdf"
    )
    return blob_path


def get_file(blob_path: str) -> bytes:
    """Récupère les bytes depuis MinIO."""
    client = get_minio_client()
    response = client.get_object(BUCKET_NAME, blob_path)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


# ==========================================
# PADDLEOCR – Moteur OCR
# ==========================================
# Singleton chargé une seule fois au démarrage
_ocr_engine: Optional[PaddleOCR] = None

def get_ocr_engine() -> PaddleOCR:
    """
    Lazy singleton : évite de recharger le modèle à chaque requête.
    lang='en' : les documents sont en anglais avec noms/adresses allemands.
    Passer lang='german' si les documents sont majoritairement en allemand.
    """
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,   # correction d'orientation
            lang="en",            # ← changer en 'german' si texte 100% allemand
            show_log=False,
            use_gpu=False         # ← True si GPU disponible dans le container
        )
        print("✅ PaddleOCR initialisé")
    return _ocr_engine


def run_ocr(pdf_bytes: bytes) -> List[Dict]:
    """
    Convertit le PDF en images (300 DPI) et applique PaddleOCR page par page.

    Retourne une liste de dicts :
      {
        "page_number": int,
        "text": str,            # texte complet de la page (ordre lecture)
        "bounding_boxes": [     # chaque token détecté
          {"text": str, "confidence": float, "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
        ]
      }
    """
    engine = get_ocr_engine()
    # Conversion PDF → PIL Images à 300 DPI pour qualité optimale
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    results = []

    for page_num, pil_image in enumerate(pages, start=1):
        img_array = np.array(pil_image)          # HxWx3 uint8, RGB
        ocr_result = engine.ocr(img_array, cls=True)

        full_text_tokens: List[str] = []
        bounding_boxes: List[Dict] = []

        # PaddleOCR retourne une liste de pages ; pour une image unique : ocr_result[0]
        page_lines = ocr_result[0] if (ocr_result and ocr_result[0]) else []

        for line in page_lines:
            bbox = line[0]                    # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            token_text = line[1][0]           # texte reconnu
            confidence = float(line[1][1])    # score [0..1]

            full_text_tokens.append(token_text)
            bounding_boxes.append({
                "text": token_text,
                "confidence": round(confidence, 3),
                "bbox": bbox
            })

        results.append({
            "page_number": page_num,
            "text": " ".join(full_text_tokens),
            "bounding_boxes": bounding_boxes
        })

    return results


# ==========================================
# PRÉ-TRAITEMENT DU TEXTE OCR
# ==========================================
def pre_process_ocr_text(pages: List[Dict]) -> str:
    """
    Concatène les pages et nettoie le texte avant envoi au LLM.

    Corrections appliquées :
    - Normalisation des espaces
    - Remplacement des tokens de checkbox OCR par des formes canoniques
    - Suppression des lignes de séparation (tirets, underscores)
    """
    raw = "\n\n".join(p["text"] for p in pages)

    # --- Normalisation des checkboxes ---
    # PaddleOCR peut lire [x], [X], [✓], (x), ☑ comme la case cochée
    # et [ ], [], ( ), ☐ comme la case vide
    raw = re.sub(r'\[\s*[xX✓✗]\s*\]|\(x\)|☑|✅', '[x]', raw)
    raw = re.sub(r'\[\s*\]|\(\s*\)|☐', '[ ]', raw)

    # --- Suppression des lignes de trait (signature, séparateur) ---
    raw = re.sub(r'_{3,}', '', raw)
    raw = re.sub(r'-{3,}', '', raw)

    # --- Normalisation des espaces multiples ---
    raw = re.sub(r'[ \t]{2,}', ' ', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)

    return raw.strip()


# ==========================================
# DÉTECTION DES CHECKBOXES
# ==========================================
def extract_checkbox_value(ocr_text: str, field_label: str) -> Optional[bool]:
    """
    Cherche dans le texte OCR un pattern du type :
      'Early Repayment Desired? [ ] yes [x] no'
    et retourne True/False selon la case cochée.

    Fonctionne aussi avec l'ordre inversé (yes/no ou no/yes).
    """
    # On cherche la ligne contenant le label
    pattern = re.compile(
        r'(?i)' + re.escape(field_label) + r'.{0,20}'
        r'(\[x\]|\[ \])\s*(yes|no).{0,10}'
        r'(\[x\]|\[ \])\s*(yes|no)',
        re.DOTALL
    )
    match = pattern.search(ocr_text)
    if not match:
        return None

    checkbox1, label1, checkbox2, label2 = (
        match.group(1), match.group(2),
        match.group(3), match.group(4)
    )

    checked_label = label1 if checkbox1 == '[x]' else label2
    return checked_label.lower() == 'yes'


# ==========================================
# EXTRACTION LLM – Llama 3.1 via Ollama
# ==========================================
EXTRACTION_PROMPT = """You are a precise data extraction assistant for commercial real estate loan applications.
Extract exactly the following 23 fields from the document text below.

Return ONLY a valid JSON object. No explanations, no markdown, no extra text.

Fields to extract:
- company_name (string)
- legal_form (string)
- date_of_incorporation (string, format DD/MM/YYYY)
- business_address (string, full address)
- commercial_register (string, number and court)
- vat_id (string)
- property_type (string)
- property_name (string)
- property_address (string, full address)
- purchase_price (number, euros, no currency symbol)
- financing_amount (number, euros)
- purpose_of_use (string)
- equity_contribution (number, euros)
- year_of_construction (integer)
- total_area_m2 (number, numeric value only, no unit)
- desired_loan_amount (number, euros)
- term_years (integer)
- monthly_installment (number, euros)
- interest_rate (string, e.g. "Variable" or "4.5%")
- early_repayment (boolean: true if [x] next to "yes", false if [x] next to "no")
- public_subsidies (boolean: true if [x] next to "yes", false if [x] next to "no")
- signature_city (string)
- signature_date (string, format DD/MM/YYYY)

Use null for any field not found in the document.

Document text:
{ocr_text}
"""

def extract_credit_info_with_llm(ocr_text: str) -> Dict[str, Any]:
    """
    Envoie le texte OCR pré-traité à Llama 3.1 via Ollama.
    Retourne un dict avec les 23 champs extraits.
    """
    prompt = EXTRACTION_PROMPT.format(ocr_text=ocr_text)

    try:
        response = ollama.chat(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}   # déterministe
        )
        raw_json = response["message"]["content"].strip()

        # Supprime les éventuels blocs markdown ```json ... ```
        raw_json = re.sub(r'^```json\s*', '', raw_json, flags=re.MULTILINE)
        raw_json = re.sub(r'^```\s*', '', raw_json, flags=re.MULTILINE)

        return json.loads(raw_json)

    except json.JSONDecodeError as e:
        print(f"⚠️  JSON invalide retourné par le LLM : {e}")
        return {}
    except Exception as e:
        print(f"⚠️  Erreur Ollama : {e}")
        return {}


# ==========================================
# POST-TRAITEMENT – Nettoyage et typage
# ==========================================
def post_process_extraction(
    llm_data: Dict[str, Any],
    ocr_text: str
) -> CreditExtractionSchema:
    """
    1. Surcharge les checkboxes avec la détection déterministe (plus fiable que le LLM)
    2. Corrige les types numériques
    3. Valide via Pydantic
    """
    # -- Surcharge checkboxes (détection pattern > LLM) --
    early_repayment = extract_checkbox_value(ocr_text, "Early Repayment Desired")
    if early_repayment is not None:
        llm_data["early_repayment"] = early_repayment

    public_subsidies = extract_checkbox_value(ocr_text, "Public Subsidies Applied For")
    if public_subsidies is not None:
        llm_data["public_subsidies"] = public_subsidies

    # -- Nettoyage des valeurs numériques (supprime €, espaces, virgules) --
    numeric_fields = [
        "purchase_price", "financing_amount", "equity_contribution",
        "total_area_m2", "desired_loan_amount", "monthly_installment"
    ]
    for field in numeric_fields:
        val = llm_data.get(field)
        if isinstance(val, str):
            cleaned = re.sub(r'[€,\s]', '', val).replace(',', '.')
            try:
                llm_data[field] = float(cleaned)
            except ValueError:
                llm_data[field] = None

    # -- Nettoyage des champs entiers --
    for field in ["year_of_construction", "term_years"]:
        val = llm_data.get(field)
        if isinstance(val, str):
            match = re.search(r'\d{1,4}', val)
            llm_data[field] = int(match.group()) if match else None

    # -- Validation Pydantic (champs inconnus ignorés) --
    return CreditExtractionSchema(**{
        k: v for k, v in llm_data.items()
        if k in CreditExtractionSchema.model_fields
    })


# ==========================================
# ENDPOINTS FASTAPI
# ==========================================

@app.on_event("startup")
def startup():
    init_db()
    init_storage()
    get_ocr_engine()   # préchargement du modèle au démarrage


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Reçoit un PDF, le stocke dans MinIO et crée l'entrée en base.
    Retourne le document_id pour les appels suivants.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Fichier vide")

    doc_id = str(uuid.uuid4())
    blob_path = save_file(file_bytes, file.filename)

    with get_engine().connect() as conn:
        conn.execute(text("""
            INSERT INTO documents (id, filename, blob_path, status, created_at)
            VALUES (:id, :filename, :blob_path, 'recu', NOW())
        """), {"id": doc_id, "filename": file.filename, "blob_path": blob_path})
        conn.commit()

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "status": "recu",
        "message": "Document uploadé. Lancez /ocr/{document_id} pour démarrer l'OCR."
    }


@app.post("/ocr/{doc_id}")
def trigger_ocr(doc_id: str):
    """
    Lance PaddleOCR sur le PDF stocké, persiste les résultats en base.
    """
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT blob_path FROM documents WHERE id = :id"),
            {"id": doc_id}
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Document introuvable")

    # Téléchargement depuis MinIO
    try:
        pdf_bytes = get_file(row.blob_path)
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Erreur MinIO : {e}")

    # OCR
    try:
        pages = run_ocr(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur OCR : {e}")

    # Persistance des résultats page par page
    with get_engine().connect() as conn:
        for page in pages:
            conn.execute(text("""
                INSERT INTO ocr_results (id, document_id, page_number, text, bounding_boxes, created_at)
                VALUES (:id, :document_id, :page_number, :text, :bounding_boxes::jsonb, NOW())
                ON CONFLICT DO NOTHING
            """), {
                "id": str(uuid.uuid4()),
                "document_id": doc_id,
                "page_number": page["page_number"],
                "text": page["text"],
                "bounding_boxes": json.dumps(page["bounding_boxes"])
            })
        conn.execute(
            text("UPDATE documents SET status = 'ocr_fait' WHERE id = :id"),
            {"id": doc_id}
        )
        conn.commit()

    total_tokens = sum(len(p["bounding_boxes"]) for p in pages)
    return {
        "document_id": doc_id,
        "pages_processed": len(pages),
        "total_tokens_detected": total_tokens,
        "status": "ocr_fait"
    }


@app.post("/extract/{doc_id}")
def extract_fields(doc_id: str):
    """
    Récupère le texte OCR depuis la base, envoie au LLM, 
    persiste et retourne les 23 champs extraits.
    """
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("""
                SELECT page_number, text FROM ocr_results
                WHERE document_id = :id ORDER BY page_number
            """),
            {"id": doc_id}
        ).fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucun résultat OCR. Lancez /ocr/{doc_id} d'abord."
        )

    pages = [{"page_number": r.page_number, "text": r.text} for r in rows]
    ocr_text = pre_process_ocr_text(pages)

    # Extraction LLM
    llm_raw = extract_credit_info_with_llm(ocr_text)
    extraction = post_process_extraction(llm_raw, ocr_text)

    # Persistance
    with get_engine().connect() as conn:
        conn.execute(text("""
            INSERT INTO extractions (id, document_id, extracted_data, created_at)
            VALUES (:id, :document_id, :data::jsonb, NOW())
            ON CONFLICT (document_id) DO UPDATE
            SET extracted_data = EXCLUDED.extracted_data, created_at = NOW()
        """), {
            "id": str(uuid.uuid4()),
            "document_id": doc_id,
            "data": json.dumps(extraction.model_dump())
        })
        conn.execute(
            text("UPDATE documents SET status = 'extrait' WHERE id = :id"),
            {"id": doc_id}
        )
        conn.commit()

    return {
        "document_id": doc_id,
        "status": "extrait",
        "extracted_fields": extraction.model_dump()
    }


@app.get("/status/{doc_id}")
def get_status(doc_id: str):
    """Retourne le statut du pipeline pour un document."""
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT filename, status, created_at FROM documents WHERE id = :id"),
            {"id": doc_id}
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Document introuvable")

    return {
        "document_id": doc_id,
        "filename": row.filename,
        "status": row.status,
        "created_at": row.created_at.isoformat()
    }


@app.get("/results/{doc_id}")
def get_results(doc_id: str):
    """Retourne les 23 champs extraits pour un document traité."""
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT extracted_data FROM extractions WHERE document_id = :id"),
            {"id": doc_id}
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Aucune extraction disponible. Lancez /extract/{doc_id} d'abord."
        )

    return {
        "document_id": doc_id,
        "extracted_fields": row.extracted_data
    }


@app.get("/documents")
def list_documents():
    """Liste tous les documents et leur statut."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT id, filename, status, created_at FROM documents ORDER BY created_at DESC")
        ).fetchall()

    return [
        {
            "document_id": str(r.id),
            "filename": r.filename,
            "status": r.status,
            "created_at": r.created_at.isoformat()
        }
        for r in rows
    ]