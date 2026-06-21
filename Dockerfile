# ============================================================
# Dockerfile — Application de gestion des nappes phréatiques
# ============================================================
# Build  : docker build -t gwl-app .
# Run    : docker run -p 8501:8501 gwl-app
# ============================================================

FROM python:3.13-slim

# Métadonnées
LABEL maintainer="Amine"
LABEL description="Application Streamlit - Modélisation et optimisation des nappes phréatiques par IA"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Répertoire de travail
WORKDIR /app

# Installer les dépendances système (nécessaires pour certaines libs Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app/ ./app/

# Créer les répertoires de données et modèles
RUN mkdir -p data/processed models/prediction

# Exposer le port Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande de démarrage
# On se place dans app/ pour que les chemins relatifs (../data, ../models) fonctionnent
CMD ["sh", "-c", "cd app && streamlit run main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]
