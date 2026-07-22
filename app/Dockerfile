FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    for i in 1 2 3 4 5; do \
        apt-get install -y --no-install-recommends \
            libpq-dev gcc \
            libgl1 \
            libglib2.0-0 \
            poppler-utils \
            tesseract-ocr tesseract-ocr-deu \
        && break || sleep 10; \
    done && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements.txt

COPY . .