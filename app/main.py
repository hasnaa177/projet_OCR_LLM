import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

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
        conn.commit()

# --- Couche d'abstraction stockage ---
def save_file(file_id: str, filename: str, content: bytes) -> str:
    """
    Sauvegarde un fichier localement.
    En production, on remplacera cette fonction par une version MinIO/S3
    sans toucher au reste du code.
    """
    backend = os.getenv("STORAGE_BACKEND", "local")

    if backend == "local":
        upload_dir = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
        file_dir = upload_dir / file_id
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / filename
        with open(file_path, "wb") as f:
            f.write(content)
        return str(file_path)

    raise ValueError(f"Backend de stockage inconnu : {backend}")

def get_file(file_id: str, filename: str) -> bytes:
    """
    Récupère un fichier depuis le stockage.
    En production, on remplacera cette fonction par une version MinIO/S3.
    """
    backend = os.getenv("STORAGE_BACKEND", "local")

    if backend == "local":
        upload_dir = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
        file_path = upload_dir / file_id / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
        with open(file_path, "rb") as f:
            return f.read()

    raise ValueError(f"Backend de stockage inconnu : {backend}")

# --- Démarrage ---
@app.on_event("startup")
def startup():
    init_db()
    # Créer le dossier uploads au démarrage
    Path(os.getenv("UPLOAD_DIR", "/app/uploads")).mkdir(
        parents=True, exist_ok=True
    )

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

    # 1. Sauvegarde du fichier via la couche d'abstraction
    blob_path = save_file(doc_id, file.filename, content)

    # 2. Enregistrement des métadonnées en base
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