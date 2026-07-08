FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installe PyTorch en version CPU uniquement (evite les dependances CUDA
# inutiles, puisque ce conteneur n'a pas de GPU) -- necessaire pour docling
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements.txt

COPY . .