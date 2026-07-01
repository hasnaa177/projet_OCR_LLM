import os
import uuid
from datetime import datetime
from pathlib import Path
from io import BytesIO
import json
import re
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import create_engine, text
from minio import Minio
from pydantic import BaseModel, Field
from typing import Optional
import ollama

# Docling imports
# pip install docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

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
# 2. PRÉ / POST-TRAITEMENT
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

    # Docling extrait le texte directement du PDF (couche texte) quand possible,
    # donc la confusion ß/B n'arrive que sur les PDF scannés sans couche texte.
    # Le filet ci-dessous est conservé pour couvrir ce cas.
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
# 3. OCR / PARSING AVEC DOCLING
# ==========================================
# Docling (IBM Research) est une bibliothèque de compréhension de documents
# qui combine extraction de texte natif PDF, OCR (via EasyOCR ou Tesseract
# en backend configurable) ET analyse de structure (tableaux, titres, listes).
#
# Avantage clé pour ce cas d'usage : Docling reconnaît automatiquement les
# tableaux du formulaire et les exporte en Markdown structuré, ce qui permet
# au LLM de recevoir "| Legal Form | Entrepreneurial Company (UG) |" au lieu
# de "Legal Form Entrepreneurial Company (UG)" sur une seule ligne aplatie.
# Cela élimine la source principale d'erreur du pipeline Tesseract (fusion
# de colonnes de tableau en texte linéaire ambigu).
#
# do_ocr=False : utilise la couche texte native du PDF si disponible (plus
# rapide et plus précis). Passer à True pour les PDF scannés sans texte.
# do_table_structure=True : active la détection et reconstruction des tableaux.

_docling_converter = None

def get_docling_converter() -> DocumentConverter:
    """
    Singleton : instancie le DocumentConverter une seule fois.
    Le premier appel charge les modèles de layout (~quelques secondes).
    """
    global _docling_converter
    if _docling_converter is None:
        print("[DOCLING] Initialisation du DocumentConverter ...")
        pipeline_options = PdfPipelineOptions(
            do_ocr=os.getenv("DOCLING_DO_OCR", "false").lower() == "true",
            do_table_structure=True,          # reconstruction des tableaux
            table_structure_options={
                "do_cell_matching": True,     # aligner cellules sur le texte natif
            },
        )
        _docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        print("[DOCLING] ✅ Converter prêt.")
    return _docling_converter


def run_ocr(pdf_bytes: bytes) -> list:
    """
    Traite le PDF avec Docling et retourne le texte structuré page par page.

    Docling exporte le document en Markdown, ce qui préserve la structure
    des tableaux (lignes et colonnes) sous forme de pipes Markdown.
    On stocke ce Markdown dans la colonne 'text' pour que le LLM en aval
    puisse voir "| Champ | Valeur |" plutôt qu'une chaîne aplatie.

    Les bounding_boxes sont vides ([]) car Docling n'expose pas nativement
    les coordonnées mot par mot dans l'API publique v2. Le champ est conservé
    pour compatibilité avec le schéma de la table ocr_results.
    """
    converter = get_docling_converter()

    # Docling travaille sur des fichiers — on écrit le PDF dans un fichier
    # temporaire, on convertit, puis on supprime le fichier.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        result = converter.convert(tmp_path)
    finally:
        os.unlink(tmp_path)

    # export_to_markdown() produit un Markdown avec tableaux pour tout le doc.
    # On traite l'intégralité comme "page 1" car Docling ne segmente pas
    # nécessairement par page dans l'export Markdown de base.
    # Si la segmentation par page est critique, utiliser result.pages et
    # exporter chaque DoclingPage individuellement.
    full_markdown = result.document.export_to_markdown()

    return [{
        "page_number":    1,
        "text":           full_markdown,
        "bounding_boxes": [],   # non disponible dans l'export Markdown Docling
    }]


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
    You are an expert financial analyst. Analyze the following document text (in Markdown format,
    with tables represented as Markdown pipe tables) and extract EVERY single field listed below.

    IMPORTANT: The following two fields have already been pre-detected deterministically.
    Use these exact values, do not override them:
    - early_repayment: {rep_val}
    - public_subsidies: {sub_val}

    Document text (Markdown):
    ---
    {raw_text}
    ---

    TARGETED FIELDS TO EXTRACT (Do not skip any):
    1. company_name
    2. legal_form
    3. date_of_incorporation
    4. business_address
    5. commercial_register (Format: 'NUMBER / COURT' e.g. 'HRB 937055 / Sangerhausen Local Court')
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
    - The input is a Markdown document. Pipe tables (| col1 | col2 |) represent form fields:
      the LEFT column is the field label, the RIGHT column is the value. Read table rows carefully.
    - COPY TEXT FIELDS EXACTLY AS WRITTEN, INCLUDING REPEATED WORDS.
    - Convert financial/area/year values into clean numbers (remove '€', 'm²', spaces, commas).
    - For vat_id: NO space anywhere in the value.
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
app = FastAPI(title="PFA OCR – Docling + Llama 3.1")
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
    get_docling_converter()   # pré-charger les modèles Docling au démarrage

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ocr_engine": "docling",
        "llm": "llama3.1",
        "docling_do_ocr": os.getenv("DOCLING_DO_OCR", "false"),
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
    # Pour Docling, le texte est déjà en Markdown structuré :
    # on applique seulement le nettoyage des symboles monétaires, pas le split
    # par label (inutile car le tableau Markdown sépare déjà les champs).
    full_text = " ".join([row.text for row in results])
    cleaned_text = pre_process_ocr_text()
    extracted_data = extract_credit_info_with_llm(cleaned_text)
    with get_engine().connect() as conn:
        conn.execute(text("UPDATE documents SET status = 'extraction_complete' WHERE id = :id"), {"id": doc_id})
        conn.commit()
    return {"document_id": doc_id, "status": "extraction_complete", "extracted_data": extracted_data}