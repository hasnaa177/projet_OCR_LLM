import os
import uuid
from datetime import datetime
from pathlib import Path
from io import BytesIO
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import create_engine, text
from minio import Minio
import easyocr
import numpy as np
from pdf2image import convert_from_bytes


app = FastAPI(title="PFA OCR – Dossiers de Crédit")

# --- Base de données ---
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = create_engine(os.getenv("DATABASE_URL"))
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

# --- Couche d'abstraction stockage ---
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
        Path(os.getenv("UPLOAD_DIR", "/app/uploads")).mkdir(
            parents=True, exist_ok=True
        )

def save_file(file_id: str, filename: str, content: bytes) -> str:
    backend = os.getenv("STORAGE_BACKEND", "local")
    if backend == "minio":
        client = get_minio_client()
        bucket = os.getenv("MINIO_BUCKET", "credit-docs")
        object_name = f"{file_id}/{filename}"
        client.put_object(
            bucket, object_name, BytesIO(content),
            length=len(content), content_type="application/pdf"
        )
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

# --- OCR ---
reader = None

def get_ocr_reader():
    """Initialise EasyOCR une seule fois."""
    global reader
    if reader is None:
        reader = easyocr.Reader(['fr', 'en'], gpu=False)
    return reader

def run_ocr(pdf_bytes: bytes) -> list:
    """
    Convertit le PDF en images puis applique EasyOCR sur chaque page.
    Retourne une liste de résultats par page avec texte + bounding boxes converties.
    """
    pages = convert_from_bytes(pdf_bytes, dpi=200)
    results = []

    for page_number, page_image in enumerate(pages, start=1):
        page_array = np.array(page_image)
        ocr_result = get_ocr_reader().readtext(page_array)

        boxes = []
        full_text = []
        for (bbox, text, confidence) in ocr_result:
            # Conversion des coordonnées NumPy en entiers Python natifs pour le JSON
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

# --- Démarrage ---
@app.on_event("startup")
def startup():
    init_db()
    init_storage()

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers PDF sont acceptés."
        )
    doc_id = str(uuid.uuid4())
    content = await file.read()
    blob_path = save_file(doc_id, file.filename, content)

    with get_engine().connect() as conn:
        conn.execute(text("""
            INSERT INTO documents (id, filename, blob_path, status, created_at)
            VALUES (:id, :filename, :blob_path, 'recu', :created_at)
        """), {
            "id": doc_id,
            "filename": file.filename,
            "blob_path": blob_path,
            "created_at": datetime.utcnow()
        })
        conn.commit()

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "status": "recu",
        "blob_path": blob_path
    }

@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    with get_engine().connect() as conn:
        result = conn.execute(
            text("SELECT * FROM documents WHERE id = :id"),
            {"id": doc_id}
        ).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Document non trouvé.")
    return dict(result._mapping)

@app.post("/ocr/{doc_id}")
def trigger_ocr(doc_id: str):
    """Déclenche l'OCR sur un document déjà uploadé."""

    # 1. Vérifier que le document existe
    with get_engine().connect() as conn:
        doc = conn.execute(
            text("SELECT * FROM documents WHERE id = :id"),
            {"id": doc_id}
        ).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Document non trouvé.")

    # 2. Récupérer le PDF depuis le stockage
    pdf_bytes = get_file(doc_id, doc.filename)

    # 3. Lancer l'OCR
    ocr_results = run_ocr(pdf_bytes)

    # 4. Sauvegarder les résultats en base
    with get_engine().connect() as conn:
        for page_result in ocr_results:
            conn.execute(text("""
                INSERT INTO ocr_results
                    (id, document_id, page_number, text, bounding_boxes, created_at)
                VALUES
                    (:id, :document_id, :page_number, :text, :bounding_boxes, :created_at)
            """), {
                "id": str(uuid.uuid4()),
                "document_id": doc_id,
                "page_number": page_result["page_number"],
                "text": page_result["text"],
                "bounding_boxes": json.dumps(page_result["bounding_boxes"]),
                "created_at": datetime.utcnow()
            })
        conn.execute(text("""
            UPDATE documents SET status = 'ocr_effectue' WHERE id = :id
        """), {"id": doc_id})
        conn.commit()

    return {
        "document_id": doc_id,
        "status": "ocr_effectue",
        "pages_processed": len(ocr_results),
        "results": ocr_results
    }

@app.get("/ocr/{doc_id}")
def get_ocr_results(doc_id: str):
    """Retourne les résultats OCR d'un document."""
    with get_engine().connect() as conn:
        results = conn.execute(
            text("""
                SELECT * FROM ocr_results
                WHERE document_id = :id
                ORDER BY page_number
            """),
            {"id": doc_id}
        ).fetchall()
    if not results:
        raise HTTPException(
            status_code=404,
            detail="Aucun résultat OCR trouvé pour ce document."
        )
    return [dict(r._mapping) for r in results]