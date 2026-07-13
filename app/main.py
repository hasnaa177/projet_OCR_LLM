import os
import uuid
from datetime import datetime
from pathlib import Path
from io import BytesIO
import json
import re

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy import create_engine, text
from minio import Minio

from pydantic import BaseModel, Field
from typing import Optional
import ollama
from ocr_engines import run_ocr, needs_easyocr_preprocessing

import hashlib

# ==========================================
# 1. SCHÉMA DE SORTIE STRICT (23 CHAMPS)
# ==========================================
class CreditExtractionSchema(BaseModel):
    # Section 1 - Applicant
    company_name: Optional[str] = Field(None, description="The legal name of the applicant company")
    legal_form: Optional[str] = Field(None, description="The legal form of the company (e.g., GmbH, AG, UG)")
    date_of_incorporation: Optional[str] = Field(None, description="The date of incorporation (DD/MM/YYYY)")
    business_address: Optional[str] = Field(None, description="Full business address of the applicant")
    commercial_register: Optional[str] = Field(None, description="The commercial register number and court")
    vat_id: Optional[str] = Field(None, description="The VAT ID or Tax Number starting with DE")

    # Section 2 - Property Description
    property_type: Optional[str] = Field(None, description="The classification of the commercial property")
    property_name: Optional[str] = Field(None, description="The commercial name of the property")
    property_address: Optional[str] = Field(None, description="Full address of the property")
    purchase_price: Optional[float] = Field(None, description="Purchase price or construction costs in Euros")
    financing_amount: Optional[float] = Field(None, description="Desired financing amount in Euros")
    purpose_of_use: Optional[str] = Field(None, description="Purpose of use (e.g., Refinancing, Purchase)")
    equity_contribution: Optional[float] = Field(None, description="Equity contribution in Euros")
    year_of_construction: Optional[int] = Field(None, description="The year the property was built")
    total_area_m2: Optional[float] = Field(None, description="Total area in square meters")

    # Section 3 - Financing Proposal
    desired_loan_amount: Optional[float] = Field(None, description="The total loan amount requested in Euros")
    term_years: Optional[int] = Field(None, description="The duration of the loan in years")
    monthly_installment: Optional[float] = Field(None, description="Preferred installment amount per month")
    interest_rate: Optional[str] = Field(None, description="Interest rate type (e.g., Fixed, Variable)")
    early_repayment: Optional[bool] = Field(None, description="True if early repayment is desired, otherwise False")
    public_subsidies: Optional[bool] = Field(None, description="True if public subsidies are applied for, otherwise False")

    # Section 4 - Declaration
    signature_city: Optional[str] = Field(None, description="The city where the application was signed")
    signature_date: Optional[str] = Field(None, description="The date when the application was signed (DD/MM/YYYY)")


# ==========================================
# 2. FONCTIONS DE TRAITEMENT ET IA
# ==========================================
def pre_process_ocr_text(raw_text: str) -> str:
    
    if not raw_text:
        return ""
    cleaned = re.sub(r'(Property Name|Property Address|Steuergasse)', r'\n\1', raw_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'(Purchase Price|Financing Amount|Purpose of Use|Equity Contribution|Year of Construction|Total Area)', r'\n\1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(Desired Loan Amount|Term|Monthly Installment|Interest Rate|Early Repayment|Public Subsidies|Signature City|Signature Date)', r'\n\1', cleaned, flags=re.IGNORECASE)

    # --- FIX : recoller les mots qu'EasyOCR a coupés en deux tokens ---
    cleaned = re.sub(r'(?<![.])\b([A-ZÄÖÜ])\s(?=[a-zäöüß]{2,}\b)', r'\1', cleaned)

    # --- FIX : retirer les symboles monétaires avant le LLM ---
    cleaned = re.sub(r'[€£$]\s*(?=\d)', '', cleaned)

    return cleaned


def _clean_numeric_field(value):
  
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        cleaned = re.sub(r'[€$£]', '', cleaned)
        cleaned = cleaned.replace('m²', '').replace('m2', '')
        cleaned = cleaned.replace(',', '')
        cleaned = cleaned.replace(' ', '')
        if cleaned == '':
            return None
        try:
            if '.' in cleaned:
                return float(cleaned)
            return int(cleaned)
        except ValueError:
            return None
    return value


def _clean_boolean_field(value):
    
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1", "x"):
            return True
        if v in ("false", "no", "0"):
            return False
    return None


def post_process_extracted_data(data: dict) -> dict:

    if not data:
        return data

    numeric_fields = [
        "purchase_price", "financing_amount", "equity_contribution",
        "total_area_m2", "desired_loan_amount", "monthly_installment",
        "year_of_construction", "term_years"
    ]
    for field in numeric_fields:
        if field in data:
            data[field] = _clean_numeric_field(data[field])

    boolean_fields = ["early_repayment", "public_subsidies"]
    for field in boolean_fields:
        if field in data:
            data[field] = _clean_boolean_field(data[field])

    date_fields = ["date_of_incorporation", "signature_date"]
    for field in date_fields:
        value = data.get(field)
        if value and isinstance(value, str):
            iso_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', value.strip())
            if iso_match:
                year, month, day = iso_match.groups()
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


def _finalize_output(data: dict) -> dict:
    
    ordered = {}
    for field_name in CreditExtractionSchema.model_fields.keys():
        value = data.get(field_name)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        ordered[field_name] = value
    return ordered

def extract_credit_info_with_llm(structured_text: str) -> dict:
    
    if not structured_text.strip():
        return {}

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    client = ollama.Client(host=ollama_url)

    prompt = f"""
    You are an expert financial analyst. Analyze the following document text and extract EVERY field listed below.

    Document text:
    ---
    {structured_text}
    ---

    FIELDS TO EXTRACT (use these EXACT snake_case keys, all 23 must appear in the output):
    1. company_name
    2. legal_form
    3. date_of_incorporation
    4. business_address
    5. commercial_register
    6. vat_id
    7. property_type
    8. property_name
    9. property_address
    10. purchase_price
    11. financing_amount
    12. purpose_of_use
    13. equity_contribution
    14. year_of_construction
    15. total_area_m2
    16. desired_loan_amount
    17. term_years
    18. monthly_installment
    19. interest_rate
    20. early_repayment
    21. public_subsidies
    22. signature_city
    23. signature_date

    CRITICAL INSTRUCTIONS:
    - Only the KEYS of the JSON must be snake_case (e.g. company_name, purchase_price).
      The VALUES themselves must be copied EXACTLY as written in the source text, preserving
      original casing, spacing, punctuation and wording. Never convert a value into snake_case,
      lowercase, or a slug -- e.g. if the source says "Entrepreneurial Company (UG)", the value
      must stay "Entrepreneurial Company (UG)", NOT "entrepreneurial_company_ug".
    - You must find and include ALL 23 fields. Look closely through the entire text, including
      every row of every table -- do not stop partway through a section.
    - Convert only financial/area/year VALUES into clean numbers (remove '€', 'm²', spaces, or commas).
      Do not alter any other text value's wording.
    - If a field is genuinely absent from the document, use null -- but only after checking carefully.
    """

    try:
        response = client.chat(
            model="llama3.1",
            messages=[
                {"role": "system", "content": "You are a precise data extraction engine. Output ONLY raw JSON matching the given schema exactly. Preserve original wording in values."},
                {"role": "user", "content": prompt}
            ],
            format=CreditExtractionSchema.model_json_schema(),
            options={"temperature": 0.0}
        )
        raw_content = response['message']['content']
        raw_extracted = json.loads(raw_content)

        # Garantit que les 23 cles du schema sont toujours presentes (None si
        # le LLM les a omises), sans jamais ecraser une valeur deja trouvee.
        validated = CreditExtractionSchema(**raw_extracted)
        complete_data = validated.model_dump()

        return post_process_extracted_data(complete_data)

    except Exception as e:
        print(f"[EXTRACTION ERROR] {e}")
        return {}


# ==========================================
# 3. INITIALISATION DE L'API FASTAPI
# ==========================================
app = FastAPI(title="PFA OCR – Dossiers de Crédit")
engine = None

def get_engine():
    global engine
    if engine is None:
        database_url = os.getenv("DATABASE_URL", "postgresql://pfa:pfa123@postgres:5432/pfa_ocr")
        engine = create_engine(database_url)
    return engine

def init_db():
    with get_engine().connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                filename TEXT NOT NULL,
                blob_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'recu',
                created_at TIMESTAMP DEFAULT NOW()
            )                  
        """))
        conn.execute(text("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash TEXT
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ocr_results (
                id UUID PRIMARY KEY,
                document_id UUID REFERENCES documents(id),
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                bounding_boxes JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS extractions (
                id UUID PRIMARY KEY,
                document_id UUID REFERENCES documents(id),
                extracted_data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))

        conn.commit()

def get_minio_client():
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False
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
    elif backend == "local":
        upload_dir = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
        file_dir = upload_dir / file_id
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / filename
        with open(file_path, "wb") as f:
            f.write(content)
        return str(file_path)
    raise ValueError(f"Backend inconnu : {backend}")

def get_file(file_id: str, filename: str) -> bytes:
    backend = os.getenv("STORAGE_BACKEND", "local")
    if backend == "minio":
        client = get_minio_client()
        bucket = os.getenv("MINIO_BUCKET", "credit-docs")
        object_name = f"{file_id}/{filename}"
        response = client.get_object(bucket, object_name)
        return response.read()
    elif backend == "local":
        upload_dir = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
        file_path = upload_dir / file_id / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
        with open(file_path, "rb") as f:
            return f.read()
    raise ValueError(f"Backend inconnu : {backend}")


# ==========================================
# 4. ENDPOINTS / ROUTES FASTAPI
# ==========================================
@app.on_event("startup")
def startup():
    init_db()
    init_storage()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")
    content = await file.read()

    file_hash = hashlib.sha256(content).hexdigest()

    # Verifie si ce meme contenu de fichier a deja ete uploade
    with get_engine().connect() as conn:
        existing = conn.execute(
            text("SELECT id, status FROM documents WHERE file_hash = :file_hash"),
            {"file_hash": file_hash}
        ).fetchone()

    if existing:
        return {
            "document_id": str(existing.id),
            "filename": file.filename,
            "status": existing.status,
            "already_processed": True,
        }

    doc_id = str(uuid.uuid4())
    blob_path = save_file(doc_id, file.filename, content)
    with get_engine().connect() as conn:
        conn.execute(text("""
            INSERT INTO documents (id, filename, blob_path, status, created_at, file_hash)
            VALUES (:id, :filename, :blob_path, 'recu', :created_at, :file_hash)
        """), {
            "id": doc_id, "filename": file.filename, "blob_path": blob_path,
            "created_at": datetime.utcnow(), "file_hash": file_hash
        })
        conn.commit()
    return {"document_id": doc_id, "filename": file.filename, "status": "recu", "already_processed": False}


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

    # Si l'OCR a deja ete fait pour ce document, renvoie le resultat existant
    with get_engine().connect() as conn:
        existing = conn.execute(
            text("SELECT page_number, text, bounding_boxes FROM ocr_results WHERE document_id = :id ORDER BY page_number"),
            {"id": doc_id}
        ).fetchall()
    if existing:
        results = [{"page_number": r.page_number, "text": r.text, "bounding_boxes": r.bounding_boxes} for r in existing]
        return {"document_id": doc_id, "status": "ocr_effectue", "pages_processed": len(results), "results": results, "already_processed": True}

    pdf_bytes = get_file(doc_id, doc.filename)
    ocr_results = run_ocr(pdf_bytes)
    with get_engine().connect() as conn:
        for page_result in ocr_results:
            conn.execute(text("""
                INSERT INTO ocr_results (id, document_id, page_number, text, bounding_boxes, created_at)
                VALUES (:id, :document_id, :page_number, :text, :bounding_boxes, :created_at)
            """), {
                "id": str(uuid.uuid4()), "document_id": doc_id, "page_number": page_result["page_number"],
                "text": page_result["text"], "bounding_boxes": json.dumps(page_result["bounding_boxes"]), "created_at": datetime.utcnow()
            })
        conn.execute(text("UPDATE documents SET status = 'ocr_effectue' WHERE id = :id"), {"id": doc_id})
        conn.commit()
    return {"document_id": doc_id, "status": "ocr_effectue", "pages_processed": len(ocr_results), "results": ocr_results, "already_processed": False}


@app.get("/ocr/{doc_id}")
def get_ocr_results(doc_id: str):
    with get_engine().connect() as conn:
        results = conn.execute(text("SELECT * FROM ocr_results WHERE document_id = :id ORDER BY page_number"), {"id": doc_id}).fetchall()
    if not results:
        raise HTTPException(status_code=404, detail="Aucun résultat OCR trouvé pour ce document.")
    return [dict(r._mapping) for r in results]

@app.get("/ocr/{doc_id}/markdown", response_class=PlainTextResponse)
def get_ocr_markdown(doc_id: str):
    with get_engine().connect() as conn:
        results = conn.execute(
            text("SELECT text FROM ocr_results WHERE document_id = :id ORDER BY page_number"),
            {"id": doc_id}
        ).fetchall()
    if not results:
        raise HTTPException(status_code=404, detail="Aucun résultat OCR trouvé pour ce document.")
    full_text = "\n\n".join(row.text for row in results)
    return full_text

@app.post("/extract/{doc_id}")
def trigger_extraction(doc_id: str):
    # Si l'extraction a deja ete faite pour ce document, renvoie le resultat existant
    with get_engine().connect() as conn:
        existing = conn.execute(
            text("SELECT extracted_data FROM extractions WHERE document_id = :id ORDER BY created_at DESC LIMIT 1"),
            {"id": doc_id}
        ).fetchone()
    if existing:
        return {"document_id": doc_id, "status": "extraction_complete", "extracted_data": existing.extracted_data, "already_processed": True}

    with get_engine().connect() as conn:
        results = conn.execute(text("SELECT text FROM ocr_results WHERE document_id = :id ORDER BY page_number"), {"id": doc_id}).fetchall()
    if not results:
        raise HTTPException(status_code=400, detail="Veuillez d'abord exécuter l'OCR sur ce document.")

    full_document_text = " ".join([row.text for row in results])

    cleaned_document_text = (
        pre_process_ocr_text(full_document_text)
        if needs_easyocr_preprocessing()
        else full_document_text
    )

    extracted_data = extract_credit_info_with_llm(cleaned_document_text)
    extracted_data = _finalize_output(extracted_data)

    with get_engine().connect() as conn:
        conn.execute(text("""
            INSERT INTO extractions (id, document_id, extracted_data, created_at)
            VALUES (:id, :document_id, :extracted_data, :created_at)
        """), {
            "id": str(uuid.uuid4()),
            "document_id": doc_id,
            "extracted_data": json.dumps(extracted_data),
            "created_at": datetime.utcnow()
        })
        conn.execute(text("UPDATE documents SET status = 'extraction_complete' WHERE id = :id"), {"id": doc_id})
        conn.commit()
    return {"document_id": doc_id, "status": "extraction_complete", "extracted_data": extracted_data, "already_processed": False}

from fastapi.responses import Response

@app.get("/ocr/{doc_id}/visualize")
def get_ocr_visualization(doc_id: str, page: int = 1):
    """
    Renvoie une image PNG de la page avec les zones detectees (tableaux, titres,
    texte...) encadrees en rouge. Fonction additive, n'affecte pas /ocr ou /extract.
    """
    with get_engine().connect() as conn:
        doc = conn.execute(text("SELECT * FROM documents WHERE id = :id"), {"id": doc_id}).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouvé.")

    pdf_bytes = get_file(doc_id, doc.filename)

    from ocr_engines.docling_engine import render_annotated_image
    try:
        image_bytes = render_annotated_image(pdf_bytes, page_number=page)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de visualisation : {e}")

    return Response(content=image_bytes, media_type="image/png")
def get_extracted_data_from_db(document_id: str) -> dict:
    with get_engine().connect() as conn:
        result = conn.execute(
            text("SELECT extracted_data FROM extractions WHERE document_id = :id ORDER BY created_at DESC LIMIT 1"),
            {"id": document_id}
        ).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Aucune extraction trouvée pour ce document.")
    return result.extracted_data


def find_differences(extracted: dict, ground_truth: dict) -> list:
    """
    Compare chaque champ et renvoie une liste détaillée des erreurs.
    """
    differences = []
    
    # On itère sur tous les champs attendus dans la vérité terrain
    for key, expected_value in ground_truth.items():
        # Conversion en string pour éviter les erreurs de type (int vs string)
        val_ext = str(extracted.get(key, "N/A")).lower().strip()
        val_gt = str(expected_value).lower().strip()
        
        if val_ext != val_gt:
            differences.append({
                "field": key,
                "expected": expected_value,
                "extracted": extracted.get(key, "MISSING")
            })
            
    return differences

@app.get("/compare/{document_id}")
def compare_results(document_id: str):
    extracted_data = get_extracted_data_from_db(document_id)

    with get_engine().connect() as conn:
        doc = conn.execute(text("SELECT filename FROM documents WHERE id = :id"), {"id": document_id}).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouvé.")

    gt_stem = Path(doc.filename).stem  # "loan_0001.pdf" -> "loan_0001"
    gt_path = Path(f"./data_scanee_150/ground_truth/{gt_stem}.json")
    if not gt_path.exists():
        raise HTTPException(status_code=404, detail=f"Vérité terrain introuvable pour {gt_stem}.")
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    diffs = find_differences(extracted_data, ground_truth_data)
    total_fields = len(ground_truth_data)
    correct_fields = total_fields - len(diffs)
    accuracy = (correct_fields / total_fields) * 100 if total_fields else 0

    return {
        "accuracy": round(accuracy, 2),
        "total_fields": total_fields,
        "errors": diffs
    }