import os
import uuid
from datetime import datetime
from pathlib import Path
from io import BytesIO
import json
import re

from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import create_engine, text
from minio import Minio
import easyocr                          # pip install easyocr
import numpy as np
from pdf2image import convert_from_bytes
from pydantic import BaseModel, Field
from typing import Optional
import ollama

# ==========================================
# 1. SCHÉMA DE SORTIE STRICT (23 CHAMPS)
# ==========================================
class CreditExtractionSchema(BaseModel):
    company_name: Optional[str] = Field(None)
    legal_form: Optional[str] = Field(None)
    date_of_incorporation: Optional[str] = Field(None)
    business_address: Optional[str] = Field(None)
    commercial_register: Optional[str] = Field(None)
    vat_id: Optional[str] = Field(None)
    property_type: Optional[str] = Field(None)
    property_name: Optional[str] = Field(None)
    property_address: Optional[str] = Field(None)
    purchase_price: Optional[float] = Field(None)
    financing_amount: Optional[float] = Field(None)
    purpose_of_use: Optional[str] = Field(None)
    equity_contribution: Optional[float] = Field(None)
    year_of_construction: Optional[int] = Field(None)
    total_area_m2: Optional[float] = Field(None)
    desired_loan_amount: Optional[float] = Field(None)
    term_years: Optional[int] = Field(None)
    monthly_installment: Optional[float] = Field(None)
    interest_rate: Optional[str] = Field(None)
    early_repayment: Optional[bool] = Field(None)
    public_subsidies: Optional[bool] = Field(None)
    signature_city: Optional[str] = Field(None)
    signature_date: Optional[str] = Field(None)


# ==========================================
# 2. PRÉ / POST-TRAITEMENT (identique main.py)
# ==========================================
def pre_process_ocr_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    cleaned = re.sub(r'(Property Name|Property Address|Steuergasse)', r'\n\1', raw_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'(Purchase Price|Financing Amount|Purpose of Use|Equity Contribution|Year of Construction|Total Area)', r'\n\1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(Desired Loan Amount|Term|Monthly Installment|Interest Rate|Early Repayment|Public Subsidies|Signature City|Signature Date)', r'\n\1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?<![.])\b([A-ZÄÖÜ])\s(?=[a-zäöüß]{2,}\b)', r'\1', cleaned)
    cleaned = re.sub(r'[€£$]\s*(?=\d)', '', cleaned)
    return cleaned


def post_process_extracted_data(data: dict) -> dict:
    if not data:
        return data

    for field in ("date_of_incorporation", "signature_date"):
        value = data.get(field)
        if value and isinstance(value, str):
            m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', value.strip())
            if m:
                year, month, day = m.groups()
                data[field] = f"{day}/{month}/{year}"

    text_fields = [
        "company_name", "business_address", "property_name",
        "property_address", "signature_city", "commercial_register"
    ]
    for field in text_fields:
        value = data.get(field)
        if value and isinstance(value, str):
            value = re.sub(r'\s+', ' ', value).strip()
            value = value.replace(' ,', ',').replace(' ;', ',').replace(';', ',')
            data[field] = value

    # EasyOCR gère nativement les caractères allemands via le modèle 'de',
    # donc la confusion ß/B est beaucoup plus rare qu'avec Tesseract.
    # On conserve le filet de sécurité par cohérence mais il devrait peu servir.
    ESZETT_FIXES = [
        (r'(?<=[a-zäöü])B(?=[a-zäöüß])', 'ß'),
        (r'(?<=[a-zäöü])B\b', 'ß'),
    ]
    for field in text_fields:
        value = data.get(field)
        if value and isinstance(value, str):
            for pattern, replacement in ESZETT_FIXES:
                value = re.sub(pattern, replacement, value)
            data[field] = value

    vat = data.get("vat_id")
    if vat and isinstance(vat, str):
        data["vat_id"] = re.sub(r'\s+', '', vat)

    price = data.get("purchase_price")
    financing = data.get("financing_amount")
    equity = data.get("equity_contribution")
    if all(isinstance(v, (int, float)) for v in (price, financing, equity)):
        expected = financing + equity
        if abs(price - expected) >= 1:
            digits = str(int(price))
            if len(digits) > 1:
                candidate = int(digits[1:])
                if abs(candidate - expected) < 1:
                    data["purchase_price"] = candidate

    return data


def extract_checkbox_value(text: str, label: str) -> Optional[bool]:
    lines = text.split('\n')
    for line in lines:
        lower_line = line.lower()
        if label.lower() not in lower_line:
            continue
        label_pos = lower_line.find(label.lower())
        after_label = lower_line[label_pos + len(label):]
        yes_idx = after_label.find('yes')
        no_idx = after_label.find('no')
        if yes_idx == -1 and no_idx == -1:
            return None
        if yes_idx != -1:
            prefix = after_label[:yes_idx]
            if 'x' in prefix[-4:]:
                return True
        if no_idx != -1:
            prefix = after_label[:no_idx]
            if 'x' in prefix[-4:]:
                return False
        return None
    return None


# ==========================================
# 3. OCR AVEC EASYOCR
# ==========================================
# EasyOCR utilise des modèles de deep learning (CRAFT + CRNN) entraînés
# par langue. On charge 'de' (allemand) + 'en' (anglais) pour couvrir
# les formulaires bilingues. Le modèle 'de' reconnaît nativement ä, ö, ü, ß
# sans pack système supplémentaire, contrairement à Tesseract qui nécessite
# 'tesseract-ocr-deu'. EasyOCR est plus lent que Tesseract (~3-5x) mais
# plus robuste sur les polices inhabituelles et les petites tailles.
#
# gpu=False : mode CPU uniquement, compatible avec tous les environnements.
# Passer gpu=True si un GPU CUDA est disponible pour accélérer x10.

_easyocr_reader = None

def get_easyocr_reader():
    """
    Singleton : instancie le Reader une seule fois au démarrage.
    Le chargement des modèles prend ~5-10s la première fois (téléchargement
    automatique dans ~/.EasyOCR/model/ si absent).
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        print("[EASYOCR] Chargement des modèles de langue ['de', 'en'] ...")
        _easyocr_reader = easyocr.Reader(
            ['de', 'en'],
            gpu=os.getenv("EASYOCR_GPU", "false").lower() == "true",
            verbose=False,
        )
        print("[EASYOCR] ✅ Modèles chargés.")
    return _easyocr_reader


def run_ocr(pdf_bytes: bytes) -> list:
    """
    Convertit le PDF en images (dpi=300) puis applique EasyOCR page par page.
    EasyOCR retourne directement une liste de tuples (bbox, text, confidence)
    — pas besoin de reconstruire les bboxes comme avec Tesseract.
    """
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    reader = get_easyocr_reader()
    results = []

    for page_number, page_image in enumerate(pages, start=1):
        page_array = np.array(page_image)

        # readtext retourne : [(bbox_4pts, text, confidence), ...]
        # bbox_4pts = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (coin haut-gauche → sens horaire)
        ocr_raw = reader.readtext(page_array, detail=1)

        boxes = []
        full_text = []
        for (bbox, word, conf) in ocr_raw:
            word = word.strip()
            if not word:
                continue
            boxes.append({
                "text":       word,
                "confidence": round(float(conf), 3),
                "bbox":       [[int(pt[0]), int(pt[1])] for pt in bbox],
            })
            full_text.append(word)

        results.append({
            "page_number":    page_number,
            "text":           " ".join(full_text),
            "bounding_boxes": boxes,
        })

    return results


# ==========================================
# 4. EXTRACTION LLM AVEC LLAMA 3.1 (Ollama)
# ==========================================
def extract_credit_info_with_llm(raw_text: str) -> dict:
    if not raw_text.strip():
        return {}

    rep_val = extract_checkbox_value(raw_text, "Early Repayment")
    sub_val = extract_checkbox_value(raw_text, "Public Subsidies")

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    client = ollama.Client(host=ollama_url)

    prompt = f"""
    You are an expert financial analyst. Analyze the following raw OCR text and extract EVERY single field listed in the targeted output format.

    IMPORTANT: The following two fields have already been pre-detected by a separate
    deterministic check of the checkbox marks. Use these exact values, do not override them:
    - early_repayment: {rep_val}
    - public_subsidies: {sub_val}

    Raw document text:
    ---
    {raw_text}
    ---

    TARGETED FIELDS TO EXTRACT (Do not skip any):
    1. company_name
    2. legal_form
    3. date_of_incorporation
    4. business_address
    5. commercial_register (Format strictly as 'NUMBER / COURT' e.g. 'HRB 937055 / Sangerhausen Local Court')
    6. vat_id
    7. property_type
    8. property_name
    9. property_address
    10. purchase_price (Number)
    11. financing_amount (Number)
    12. purpose_of_use
    13. equity_contribution (Number)
    14. year_of_construction (Integer)
    15. total_area_m2 (Number)
    16. desired_loan_amount (Number)
    17. term_years (Integer)
    18. monthly_installment (Number)
    19. interest_rate
    20. early_repayment (Boolean)
    21. public_subsidies (Boolean)
    22. signature_city
    23. signature_date

    CRITICAL INSTRUCTIONS:
    - COPY TEXT FIELDS EXACTLY AS WRITTEN, INCLUDING REPEATED WORDS.
    - You must find and include ALL 23 fields.
    - Convert all financial/area/year values into clean numbers (remove '€', 'm²', spaces, commas).
    - For vat_id: NO space anywhere in the value.
    - For purchase_price, financing_amount, equity_contribution: copy digits EXACTLY, do not add extra leading digits.
    - DATE FIELDS: copy DD/MM/YYYY exactly as written, do not convert to ISO.
    - Preserve German special characters (ä, ö, ü, ß) exactly.
    """

    try:
        response = client.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}],
            format=CreditExtractionSchema.model_json_schema(),
            options={"temperature": 0.0},
        )
        raw_content = response['message']['content']
    except Exception as e:
        import traceback
        print(f"[EXTRACTION] Erreur Ollama : {e}")
        traceback.print_exc()
        return {}

    try:
        raw_extracted = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"[EXTRACTION] Réponse non-JSON : {e}")
        return {}

    if rep_val is not None:
        raw_extracted["early_repayment"] = rep_val
    if sub_val is not None:
        raw_extracted["public_subsidies"] = sub_val

    try:
        return post_process_extracted_data(raw_extracted)
    except Exception as e:
        import traceback
        print(f"[EXTRACTION] Erreur post-traitement : {e}")
        traceback.print_exc()
        return raw_extracted


# ==========================================
# 5. INITIALISATION FASTAPI + STOCKAGE
# ==========================================
app = FastAPI(title="PFA OCR – EasyOCR + Llama 3.1")
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = create_engine(os.getenv("DATABASE_URL", "postgresql://pfa:pfa123@postgres:5432/pfa_ocr"))
    return engine

def init_db():
    with get_engine().connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY, filename TEXT NOT NULL, blob_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'recu', created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ocr_results (
                id UUID PRIMARY KEY, document_id UUID REFERENCES documents(id),
                page_number INTEGER NOT NULL, text TEXT NOT NULL,
                bounding_boxes JSONB NOT NULL, created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()

def get_minio_client():
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )

def init_storage():
    backend = os.getenv("STORAGE_BACKEND", "local")
    if backend == "minio":
        client = get_minio_client()
        bucket = os.getenv("MINIO_BUCKET", "credit-docs")
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    elif backend == "local":
        Path(os.getenv("UPLOAD_DIR", "/app/uploads")).mkdir(parents=True, exist_ok=True)

def save_file(file_id: str, filename: str, content: bytes) -> str:
    backend = os.getenv("STORAGE_BACKEND", "local")
    if backend == "minio":
        client = get_minio_client()
        bucket = os.getenv("MINIO_BUCKET", "credit-docs")
        object_name = f"{file_id}/{filename}"
        client.put_object(bucket, object_name, BytesIO(content), length=len(content), content_type="application/pdf")
        return object_name
    upload_dir = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
    file_dir = upload_dir / file_id
    file_dir.mkdir(parents=True, exist_ok=True)
    file_path = file_dir / filename
    with open(file_path, "wb") as f:
        f.write(content)
    return str(file_path)

def get_file(file_id: str, filename: str) -> bytes:
    backend = os.getenv("STORAGE_BACKEND", "local")
    if backend == "minio":
        client = get_minio_client()
        bucket = os.getenv("MINIO_BUCKET", "credit-docs")
        response = client.get_object(bucket, f"{file_id}/{filename}")
        return response.read()
    file_path = Path(os.getenv("UPLOAD_DIR", "/app/uploads")) / file_id / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
    with open(file_path, "rb") as f:
        return f.read()


# ==========================================
# 6. ENDPOINTS
# ==========================================
@app.on_event("startup")
def startup():
    init_db()
    init_storage()
    # Pré-charger les modèles EasyOCR au démarrage pour éviter la latence
    # sur la première requête OCR (~10s de chargement des modèles CRAFT+CRNN).
    get_easyocr_reader()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ocr_engine": "easyocr",
        "llm": "llama3.1",
        "easyocr_languages": ["de", "en"],
        "gpu": os.getenv("EASYOCR_GPU", "false"),
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")
    doc_id = str(uuid.uuid4())
    content = await file.read()
    blob_path = save_file(doc_id, file.filename, content)
    with get_engine().connect() as conn:
        conn.execute(text("""
            INSERT INTO documents (id, filename, blob_path, status, created_at)
            VALUES (:id, :filename, :blob_path, 'recu', :created_at)
        """), {"id": doc_id, "filename": file.filename, "blob_path": blob_path, "created_at": datetime.utcnow()})
        conn.commit()
    return {"document_id": doc_id, "filename": file.filename, "status": "recu", "blob_path": blob_path}

@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    with get_engine().connect() as conn:
        result = conn.execute(text("SELECT * FROM documents WHERE id = :id"), {"id": doc_id}).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Document non trouvé.")
    return dict(result._mapping)

@app.post("/ocr/{doc_id}")
def trigger_ocr(doc_id: str):
    with get_engine().connect() as conn:
        doc = conn.execute(text("SELECT * FROM documents WHERE id = :id"), {"id": doc_id}).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouvé.")
    pdf_bytes = get_file(doc_id, doc.filename)
    ocr_results = run_ocr(pdf_bytes)
    with get_engine().connect() as conn:
        for page_result in ocr_results:
            conn.execute(text("""
                INSERT INTO ocr_results (id, document_id, page_number, text, bounding_boxes, created_at)
                VALUES (:id, :document_id, :page_number, :text, :bounding_boxes, :created_at)
            """), {
                "id": str(uuid.uuid4()), "document_id": doc_id,
                "page_number": page_result["page_number"], "text": page_result["text"],
                "bounding_boxes": json.dumps(page_result["bounding_boxes"]),
                "created_at": datetime.utcnow(),
            })
        conn.execute(text("UPDATE documents SET status = 'ocr_effectue' WHERE id = :id"), {"id": doc_id})
        conn.commit()
    return {"document_id": doc_id, "status": "ocr_effectue", "pages_processed": len(ocr_results), "results": ocr_results}

@app.get("/ocr/{doc_id}")
def get_ocr_results(doc_id: str):
    with get_engine().connect() as conn:
        results = conn.execute(text("SELECT * FROM ocr_results WHERE document_id = :id ORDER BY page_number"), {"id": doc_id}).fetchall()
    if not results:
        raise HTTPException(status_code=404, detail="Aucun résultat OCR trouvé.")
    return [dict(r._mapping) for r in results]

@app.post("/extract/{doc_id}")
def trigger_extraction(doc_id: str):
    with get_engine().connect() as conn:
        results = conn.execute(text("SELECT text FROM ocr_results WHERE document_id = :id ORDER BY page_number"), {"id": doc_id}).fetchall()
    if not results:
        raise HTTPException(status_code=400, detail="Veuillez d'abord exécuter l'OCR sur ce document.")
    full_text = " ".join([row.text for row in results])
    cleaned_text = pre_process_ocr_text(full_text)
    extracted_data = extract_credit_info_with_llm(cleaned_text)
    with get_engine().connect() as conn:
        conn.execute(text("UPDATE documents SET status = 'extraction_complete' WHERE id = :id"), {"id": doc_id})
        conn.commit()
    return {"document_id": doc_id, "status": "extraction_complete", "extracted_data": extracted_data}