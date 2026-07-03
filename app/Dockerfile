FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système nécessaires (PostgreSQL, OCR, OpenGl...)
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1. On force l'installation de PyTorch et Torchvision en version CPU (Léger et rapide)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. On installe le reste des packages du requirements.txt depuis PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .