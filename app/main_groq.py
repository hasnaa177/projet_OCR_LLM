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
import pytesseract
from pytesseract import Output
import numpy as np
from pdf2image import convert_from_bytes
from pydantic import BaseModel, Field
from typing import Optional
from groq import Groq          # pip install groq

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
# 3. OCR AVEC TESSERACT (identique main.py)
# ==========================================
TESSERACT_LANG   = os.getenv("TESSERACT_LANG",   "deu+eng")
TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG",  "--oem 3 --psm 3")

def check_tesseract_languages():
    try:
        available = pytesseract.get_languages()
    except Exception as e:
        print(f"[STARTUP] ⚠️  Impossible de vérifier les langues Tesseract : {e}")
        return
    required_langs = [lang for lang in TESSERACT_LANG.split('+') if lang != 'eng']
    missing = [lang for lang in required_langs if lang not in available]
    if missing:
        print(f"[STARTUP] ⚠️  Pack(s) manquant(s) : {missing}. Caractères allemands mal reconnus.")
    else:
        print(f"[STARTUP] ✅ Langues Tesseract OK : {TESSERACT_LANG}")

def run_ocr(pdf_bytes: bytes) -> list:
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    results = []
    for page_number, page_image in enumerate(pages, start=1):
        page_array = np.array(page_image)
        ocr_data = pytesseract.image_to_data(
            page_array,
            lang=TESSERACT_LANG,
            config=TESSERACT_CONFIG,
            output_type=Output.DICT,
        )
        boxes, full_text = [], []
        for i in range(len(ocr_data["text"])):
            word = ocr_data["text"][i].strip()
            if not word:
                continue
            raw_conf  = float(ocr_data["conf"][i])
            confidence = max(raw_conf, 0.0) / 100.0
            left, top  = int(ocr_data["left"][i]), int(ocr_data["top"][i])
            w, h       = int(ocr_data["width"][i]), int(ocr_data["height"][i])
            boxes.append({
                "text": word, "confidence": round(confidence, 3),
                "bbox": [[left, top], [left+w, top], [left+w, top+h], [left, top+h]],
            })
            full_text.append(word)
        results.append({
            "page_number": page_number,
            "text": " ".join(full_text),
            "bounding_boxes": boxes,
        })
    return results


# ==========================================
# 4. EXTRACTION LLM AVEC GROQ
# ==========================================
# Groq est une API cloud qui expose des LLMs open-source (LLaMA 3, Mixtral,
# Gemma) sur du matériel LPU (Language Processing Unit), ce qui donne des
# vitesses d'inférence ~10-20x supérieures à Ollama local sur CPU.
#
# Modèle par défaut : "llama-3.1-70b-versatile" — même poids que Llama 3.1
# mais inférence cloud instantanée. Peut être changé via GROQ_MODEL.
# Autres options disponibles sur Groq : "mixtral-8x7b-32768", "gemma2-9b-it".
#
# Différence clé avec Ollama : Groq n'implémente pas le paramètre "format"
# (JSON schema structuré). On force la sortie JSON via le system prompt et
# on parse le résultat manuellement, avec un fallback robuste.
#
# Prérequis : définir la variable d'environnement GROQ_API_KEY.
# Obtenir une clé gratuite sur https://console.groq.com

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# Schéma JSON attendu en sortie — injecté dans le system prompt pour guider
# Groq à produire un JSON valide sans l'API "format" d'Ollama.
_JSON_SCHEMA_STR = json.dumps(CreditExtractionSchema.model_json_schema(), indent=2)

_GROQ_SYSTEM_PROMPT = f"""You are an expert financial document analyst.
Your ONLY output must be a single valid JSON object — no prose, no markdown fences,
no explanation — matching EXACTLY this schema:

{_JSON_SCHEMA_STR}

Rules:
- Output raw JSON only. Start with {{ and end with }}.
- All 23 fields must be present (use null if genuinely absent).
- Monetary values: plain numbers, no € or commas.
- Dates: DD/MM/YYYY, never ISO.
- vat_id: no spaces.
- Preserve ä, ö, ü, ß exactly.
"""


def _parse_groq_json(raw: str) -> dict:
    """
    Parse robuste de la réponse Groq :
    1. Tentative directe sur le texte brut.
    2. Extraction du premier bloc {...} si du texte parasite entoure le JSON.
    3. Nettoyage des fences Markdown (```json ... ```) si présentes.
    Retourne {} si aucun parse ne réussit.
    """
    raw = raw.strip()

    # Tentative 1 : parse direct
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Tentative 2 : extraire le premier {...} bien formé
    start = raw.find('{')
    end   = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end+1])
        except json.JSONDecodeError:
            pass

    # Tentative 3 : retirer les fences Markdown
    cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    print(f"[EXTRACTION] Impossible de parser la réponse Groq (premiers 500 chars) : {raw[:500]!r}")
    return {}


def extract_credit_info_with_llm(raw_text: str) -> dict:
    if not raw_text.strip():
        return {}

    rep_val = extract_checkbox_value(raw_text, "Early Repayment")
    sub_val = extract_checkbox_value(raw_text, "Public Subsidies")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY manquante. "
            "Définir la variable d'environnement avant de lancer l'API. "
            "Clé gratuite sur https://console.groq.com"
        )

    client = Groq(api_key=api_key)

    user_prompt = f"""
    Extract all 23 fields from the following OCR text of a German commercial real estate
    loan application. Return ONLY a JSON object, no other text.

    IMPORTANT: Use these pre-detected checkbox values exactly, do not override them:
    - early_repayment: {rep_val}
    - public_subsidies: {sub_val}

    OCR text:
    ---
    {raw_text}
    ---

    TARGETED FIELDS:
    1. company_name            — exact company name as written (keep repeated words)
    2. legal_form              — descriptive label only, e.g. "Stock Corporation (AG)"
    3. date_of_incorporation   — DD/MM/YYYY exactly as written
    4. business_address        — full address
    5. commercial_register     — format: "NUMBER / COURT Local Court"
    6. vat_id                  — no spaces, e.g. "DE755741810"
    7. property_type
    8. property_name
    9. property_address
    10. purchase_price          — number only
    11. financing_amount        — number only
    12. purpose_of_use
    13. equity_contribution     — number only
    14. year_of_construction    — integer
    15. total_area_m2           — number only
    16. desired_loan_amount     — number only
    17. term_years              — integer
    18. monthly_installment     — number only
    19. interest_rate           — "Fixed" or "Variable" only
    20. early_repayment         — boolean (use pre-detected value above)
    21. public_subsidies        — boolean (use pre-detected value above)
    22. signature_city
    23. signature_date          — DD/MM/YYYY exactly as written

    Remember: raw JSON only, starting with {{.
    """

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _GROQ_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
            # Groq supporte response_format JSON sur certains modèles :
            # on l'active ici comme hint supplémentaire ; le system prompt
            # reste la contrainte principale car tous les modèles ne le
            # supportent pas.
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
    except Exception as e:
        import traceback
        print(f"[EXTRACTION] Erreur Groq API : {e}")
        traceback.print_exc()
        return {}

    raw_extracted = _parse_groq_json(raw_content)

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
app = FastAPI(title="PFA OCR – Tesseract + Groq")
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
    check_tesseract_languages()

@app.get("/health")
def health():
    try:
        available_langs = pytesseract.get_languages()
    except Exception:
        available_langs = []
    required = [l for l in TESSERACT_LANG.split('+') if l != 'eng']
    missing  = [l for l in required if l not in available_langs]
    return {
        "status":                    "ok",
        "ocr_engine":                "tesseract",
        "llm":                       "groq",
        "groq_model":                GROQ_MODEL,
        "tesseract_lang_configured": TESSERACT_LANG,
        "tesseract_lang_available":  available_langs,
        "tesseract_lang_missing":    missing,
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