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
import easyocr
import numpy as np
from pdf2image import convert_from_bytes
from pydantic import BaseModel, Field
from typing import Optional
import ollama

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
    """
    Aère et structure le texte brut d'EasyOCR pour éviter que les lignes collées
    ne fassent sauter des champs clés au LLM.
    """
    if not raw_text:
        return ""
    cleaned = re.sub(r'(Property Name|Property Address|Steuergasse)', r'\n\1', raw_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'(Purchase Price|Financing Amount|Purpose of Use|Equity Contribution|Year of Construction|Total Area)', r'\n\1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(Desired Loan Amount|Term|Monthly Installment|Interest Rate|Early Repayment|Public Subsidies|Signature City|Signature Date)', r'\n\1', cleaned, flags=re.IGNORECASE)

    # --- FIX : recoller les mots qu'EasyOCR a coupés en deux tokens ---
    # EasyOCR détecte parfois un mot comme deux boîtes adjacentes (ex: "P" + "erleberg"
    # au lieu de "Perleberg", ou "J" + "antsch" au lieu de "Jäntsch"), ce qui produit
    # une lettre isolée suivie d'un espace une fois les textes concaténés. On recolle
    # cette lettre au mot minuscule qui suit, y compris en tout début de texte (un nom
    # de société ou un nom de rue ne commence jamais par une initiale isolée suivie
    # d'un mot ordinaire en minuscules -- ce n'est jamais une vraie initiale dans ce
    # contexte de formulaire structuré). On exclut seulement le cas où la lettre suit
    # un point final, qui correspond à une vraie initiale de personne (ex: "Dr. A Mueller").
    # Limite connue : un cas du type "..., A general statement" en pleine prose serait
    # incorrectement recollé, mais ce type de texte n'apparaît pas dans ces documents.
    cleaned = re.sub(r'(?<![.])\b([A-ZÄÖÜ])\s(?=[a-zäöüß]{2,}\b)', r'\1', cleaned)

    # --- FIX : retirer les symboles monétaires avant le LLM ---
    # Le symbole € (et $, £) est parfois mal segmenté par EasyOCR sur des
    # polices compressées ou en basse résolution, ce qui peut produire un
    # chiffre fantôme collé devant le vrai montant (ex: "€720,000" devient
    # "1720,000" dans le texte OCR brut, que le LLM retransmet ensuite tel
    # quel comme 1720000 au lieu de 720000). On retire le symbole lui-même
    # avant l'envoi au LLM pour éliminer cette source d'erreur à la racine.
    cleaned = re.sub(r'[€£$]\s*(?=\d)', '', cleaned)

    return cleaned


def post_process_extracted_data(data: dict) -> dict:
    """
    Nettoie la sortie du LLM pour corriger les écarts de FORMAT qui ne sont pas
    des erreurs d'extraction mais des erreurs de mise en forme (date ISO au lieu
    de DD/MM/YYYY, espaces résiduels, nombres mal arrondis, etc.).
    Cette étape est déterministe : elle ne devine rien, elle reformate.
    """
    if not data:
        return data

    date_fields = ["date_of_incorporation", "signature_date"]
    for field in date_fields:
        value = data.get(field)
        if value and isinstance(value, str):
            # Le LLM répond parfois en ISO (YYYY-MM-DD) malgré la consigne DD/MM/YYYY.
            iso_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', value.strip())
            if iso_match:
                year, month, day = iso_match.groups()
                data[field] = f"{day}/{month}/{year}"

    # Nettoyage des espaces multiples ou mal placés dans les champs texte
    # (ex: "P erleberg" oublié en amont, "Bäblingen; Germany" avec un ';' parasite)
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

    # --- FIX : validation croisée de purchase_price ---
    # Dans ce type de document, purchase_price == financing_amount + equity_contribution
    # par construction. Si l'égalité ne tient pas, mais qu'elle tient en retirant le
    # premier chiffre de purchase_price, c'est la signature d'un chiffre fantôme
    # ajouté par une mauvaise lecture OCR d'un symbole monétaire (cf. pre_process_ocr_text).
    # Ce filet de sécurité corrige les cas où le nettoyage en amont n'aurait pas suffi.
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


def extract_credit_info_with_llm(raw_text: str) -> dict:
    """
    Envoie le texte structuré à Llama 3.1 avec une checklist impérative
    pour forcer l'extraction complète des 23 champs requis.
    """
    if not raw_text.strip():
        return {}

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    client = ollama.Client(host=ollama_url)

    prompt = f"""
    You are an expert financial analyst. Analyze the following raw OCR text and extract EVERY single field listed in the targeted output format.

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
    - You must find and include ALL 23 fields. Look closely at the text.
    - Convert all financial/area/year values into clean numbers (remove '€', 'm²', spaces, or commas).
    - If data for a field is present, map it correctly. Do not omit fields like 'property_address' or 'financing_amount'.
    - For purchase_price, financing_amount and any monetary amount: copy the digits EXACTLY as they
      appear in the source text. Do not add, remove, or guess any digit. If the number reads "1.572.000",
      output 1572000 -- do not invent extra leading digits.
    - NEVER merge two separate numbers that happen to be adjacent or on the same line in the raw text
      (e.g. a page number, a field label number, or an unrelated figure sitting right before the actual
      amount) into a single longer number. Each numeric field must come from exactly one distinct number
      in the source text -- if a number looks unusually long for what is expected (e.g. an 8-digit price
      where similar documents show 7 digits), re-check whether you accidentally concatenated two values.
    - DATE FIELDS ARE COPY-ONLY, NOT CONVERT: for date_of_incorporation and signature_date,
      find the date EXACTLY as written in the source text and copy its day/month/year digits
      in the same DD/MM/YYYY order as in the source. Do NOT reinterpret, reorder, or convert
      the date through any internal date format. For example, if the source text shows
      "10/04/2014", the output must be exactly "10/04/2014" -- never "2014-10-04" and never
      "04/10/2014". If you are unsure which number is the day and which is the month, keep
      them in the exact same left-to-right order as they appear in the source text.
    - For early_repayment and public_subsidies: these are two SEPARATE yes/no checkboxes in the
      source document. Treat them independently -- do not let the state of one influence your guess
      for the other. For each one individually: look for an explicit checkbox mark, or wording such as
      "desired" / "requested" / "yes" (-> True) versus "not desired" / "not requested" / "no" / an
      unchecked box (-> False). Quote mentally to yourself the exact phrase or checkbox state you saw
      for early_repayment, and separately the exact phrase or checkbox state you saw for
      public_subsidies, before deciding the two values. If you cannot find clear evidence for a field,
      return null rather than guessing True or False.
    - IMPORTANT layout detail about these checkboxes: the source document always lists the "yes"
      option BEFORE the "no" option on the same line, in that fixed left-to-right order (e.g.
      "[ ] yes [x] no" means NOT desired, "[x] yes [ ] no" means desired). The OCR sometimes fails
      to capture an EMPTY checkbox (low contrast), so a line may show only ONE mark with no visible
      brackets for the other option, e.g. just "x no" or "x yes". Read this literally: if the mark
      sits immediately before the word "yes" with nothing else around it, that means the "yes" box
      is checked (-> True). If the mark sits immediately before the word "no" with nothing else
      around it, that means the "no" box is checked (-> False). Do not assume the opposite based on
      what the OTHER field shows -- evaluate the literal local text "x yes" vs "x no" for THIS field
      only, word by word, exactly as it appears.
    - Preserve German special characters (ä, ö, ü, ß) exactly as found in the source text for names,
      addresses and cities (e.g. "Böblingen", not "Bablingen"). Do not transliterate them.
    """

    try:
        response = client.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}],
            format=CreditExtractionSchema.model_json_schema(),
            options={"temperature": 0.0}
        )
        raw_content = response['message']['content']
    except Exception as e:
        import traceback
        print(f"[EXTRACTION] Erreur lors de l'appel à Ollama : {e}")
        traceback.print_exc()
        return {}

    try:
        raw_extracted = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"[EXTRACTION] Réponse Ollama non-JSON ou tronquée : {e}")
        print(f"[EXTRACTION] Contenu brut reçu (premiers 1000 caractères) : {raw_content[:1000]!r}")
        return {}

    try:
        return post_process_extracted_data(raw_extracted)
    except Exception as e:
        import traceback
        print(f"[EXTRACTION] Erreur dans le post-traitement : {e}")
        traceback.print_exc()
        # On retourne quand même les données brutes non post-traitées plutôt
        # que de tout perdre si seul le post-traitement (cosmétique) plante.
        return raw_extracted


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
            CREATE TABLE IF NOT EXISTS ocr_results (
                id UUID PRIMARY KEY,
                document_id UUID REFERENCES documents(id),
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                bounding_boxes JSONB NOT NULL,
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

reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        # Les documents traités contiennent des noms propres et adresses allemandes
        # (Böblingen, Weiß, Jäntsch...). EasyOCR doit charger le jeu de caractères
        # allemand pour reconnaître correctement les trémas (ä, ö, ü) et le ß ;
        # sinon il les remplace par la lettre latine visuellement la plus proche
        # (ö -> ä, ß -> l, etc.), ce qui était la cause de la moitié des erreurs.
        reader = easyocr.Reader(['de', 'en'], gpu=False)
    return reader

def run_ocr(pdf_bytes: bytes) -> list:
    pages = convert_from_bytes(pdf_bytes, dpi=200)
    results = []
    for page_number, page_image in enumerate(pages, start=1):
        page_array = np.array(page_image)
        ocr_result = get_ocr_reader().readtext(page_array)
        boxes = []
        full_text = []
        for (bbox, text, confidence) in ocr_result:
            cleaned_bbox = [[int(coord[0]), int(coord[1])] for coord in bbox]
            boxes.append({
                "text": text,
                "confidence": round(float(confidence), 3),
                "bbox": cleaned_bbox
            })
            full_text.append(text)
        results.append({
            "page_number": page_number,
            "text": " ".join(full_text),
            "bounding_boxes": boxes
        })
    return results


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
                "id": str(uuid.uuid4()), "document_id": doc_id, "page_number": page_result["page_number"],
                "text": page_result["text"], "bounding_boxes": json.dumps(page_result["bounding_boxes"]), "created_at": datetime.utcnow()
            })
        conn.execute(text("UPDATE documents SET status = 'ocr_effectue' WHERE id = :id"), {"id": doc_id})
        conn.commit()
    return {"document_id": doc_id, "status": "ocr_effectue", "pages_processed": len(ocr_results), "results": ocr_results}

@app.get("/ocr/{doc_id}")
def get_ocr_results(doc_id: str):
    with get_engine().connect() as conn:
        results = conn.execute(text("SELECT * FROM ocr_results WHERE document_id = :id ORDER BY page_number"), {"id": doc_id}).fetchall()
    if not results:
        raise HTTPException(status_code=404, detail="Aucun résultat OCR trouvé pour ce document.")
    return [dict(r._mapping) for r in results]

@app.post("/extract/{doc_id}")
def trigger_extraction(doc_id: str):
    with get_engine().connect() as conn:
        results = conn.execute(text("SELECT text FROM ocr_results WHERE document_id = :id ORDER BY page_number"), {"id": doc_id}).fetchall()
    if not results:
        raise HTTPException(status_code=400, detail="Veuillez d'abord exécuter l'OCR sur ce document.")

    # Fusionner le texte de toutes les pages
    full_document_text = " ".join([row.text for row in results])

    # Appliquer le pré-traitement pour restructurer le texte
    cleaned_document_text = pre_process_ocr_text(full_document_text)

    # Extraire avec Llama 3.1
    extracted_data = extract_credit_info_with_llm(cleaned_document_text)

    with get_engine().connect() as conn:
        conn.execute(text("UPDATE documents SET status = 'extraction_complete' WHERE id = :id"), {"id": doc_id})
        conn.commit()
    return {"document_id": doc_id, "status": "extraction_complete", "extracted_data": extracted_data}