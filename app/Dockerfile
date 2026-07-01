services:
  api:
    image: pfa-ocr-api-with-models   # ← remplace "build: ."
    # build: .                        # ← commenter cette ligne
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://pfa:pfa123@postgres:5432/pfa_ocr
      - MINIO_ENDPOINT=minio:9000
      - OLLAMA_MODEL=llama3.1
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy