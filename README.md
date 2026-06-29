# 🌊 Modélisation intelligente et optimisation des réseaux d'eaux souterraines par IA

Application Streamlit pour la modélisation et l'optimisation des nappes phréatiques, couplant **LSTM**, **théorie des graphes** et **algorithme génétique**.

> Projet de Fin d'Études (PFE) - Master Sciences et Ingénierie de Données, 2026

## Description

Cette application permet aux gestionnaires de nappes phréatiques de :

1. **Observer** l'état de leur nappe via des cartes interactives et des graphiques temporels
2. **Prédire** le niveau futur des puits grâce à des réseaux LSTM enrichis par la théorie des graphes
3. **Décider** du plan de pompage optimal via un algorithme génétique récursif

L'application est **générique** : elle accepte n'importe quel dataset de n'importe quel pays, à condition de respecter le nommage des colonnes.

## Architecture

```
GWL_PREDICTION_OPTIMIZATION/
├── app/
│   ├── main.py                    # Page d'accueil + nettoyage + authentification
│   ├── config.yaml                # Configuration utilisateurs (bcrypt)
│   ├── pages/
│   │   ├── 1_visualisation.py     # Module 1 — Observer
│   │   ├── 2_prediction.py        # Module 2 — Prédire
│   │   └── 3_optimisation.py      # Module 3 — Décider
│   ├── utils/
│   │   └── auth.py                # Authentification + RBAC
│   └── assets/
├── data/                          # Datasets (CSV)
├── models/
│   └── prediction/                # Modèles LSTM sauvegardés (.keras)
├── notebooks/                     # Jupyter notebooks d'exploration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/workflows/ci.yml       # Intégration continue
└── README.md
```

## Installation et lancement

### Option 1 - Avec Docker (recommandé)

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/GWL_PREDICTION_OPTIMIZATION.git
cd GWL_PREDICTION_OPTIMIZATION

# Lancer l'application
docker-compose up -d

# Ouvrir dans le navigateur
# http://localhost:8501
```

Pour reconstruire après modification :

```bash
docker-compose up -d --build
```

### Option 2 - Installation locale

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/GWL_PREDICTION_OPTIMIZATION.git
cd GWL_PREDICTION_OPTIMIZATION

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app/main.py
```

## Authentification et rôles

L'application utilise un système d'authentification avec 3 niveaux d'accès :

| Rôle             | Module 1 (Observer) | Module 2 (Prédire) | Module 3 (Décider) | Administration |
| ---------------- | :-----------------: | :----------------: | :----------------: | :------------: |
| **lecteur**      |         ✅          |         ❌         |         ❌         |       ❌       |
| **gestionnaire** |         ✅          |         ✅         |         ✅         |       ❌       |
| **admin**        |         ✅          |         ✅         |         ✅         |       ✅       |

### Comptes par défaut

| Identifiant    | Mot de passe | Rôle         |
| -------------- | ------------ | ------------ |
| `admin`        | `admin123`   | admin        |
| `gestionnaire` | `gest123`    | gestionnaire |
| `lecteur`      | `lect123`    | lecteur      |

> **Changez les mots de passe par défaut avant tout déploiement.**

Les mots de passe sont hashés avec **bcrypt** dans le fichier `app/config.yaml`.

## Données d'entrée

Le fichier CSV doit respecter le nommage suivant :

| Type                 | Préfixe obligatoire     | Exemple                       |
| -------------------- | ----------------------- | ----------------------------- |
| Date                 | `Date`                  | `Date`                        |
| Niveau piézométrique | `Depth_to_Groundwater_` | `Depth_to_Groundwater_Puits1` |
| Volume de pompage    | `Volume_`               | `Volume_Puits1`               |
| Pluviométrie         | `Rainfall_`             | `Rainfall_Station1`           |
| Température          | `Temperature_`          | `Temperature_Station1`        |

**Durée minimale recommandée** : 3 ans de données journalières.

## Stack technique

| Couche       | Technologies                           |
| ------------ | -------------------------------------- |
| Interface    | Streamlit, Folium, Plotly              |
| IA           | TensorFlow/Keras (LSTM), Scikit-learn  |
| Optimisation | DEAP (algorithme génétique), NetworkX  |
| Sécurité     | streamlit-authenticator, bcrypt, RBAC  |
| DevOps       | Docker, docker-compose, GitHub Actions |
| Données      | Pandas, NumPy                          |

## Résultats

### Comparaison des modèles (Petrignano - 1 puits)

| Modèle        | R²         | RMSE (m)   | MAE (m)    |
| ------------- | ---------- | ---------- | ---------- |
| **LSTM**      | **0.9565** | **0.1812** | **0.1239** |
| Random Forest | 0.9469     | 0.1979     | 0.1606     |
| ANN           | 0.9093     | 0.2586     | 0.2047     |

### Optimisation multi-puits (Doganella — 9 puits)

- **6/9 puits fiables** (R² > 0.82)
- **0 violation de seuil** sur 30 jours avec l'AG récursif
- Stratégie identifiée : **transfert de charge** des puits centraux vers les puits isolés

## 👤 Auteur

**Amine** - Master Sciences et Ingénierie de Données, 2026
