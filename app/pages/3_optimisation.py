# ============================================================
# Module 3 — Optimisation (Décider)
# ============================================================
# Ce module utilise l'algorithme génétique couplé aux LSTM
# pour trouver le plan de pompage optimal.
#
# Il réutilise les LSTM entraînés dans le Module 2
# et le graphe déjà construit.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import json
import random
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from deap import base, creator, tools
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.auth import guard_page

# === Vérifier que les données sont chargées ===
if 'data' not in st.session_state:
    st.warning("⚠️ Aucune donnée chargée. Retournez à la page d'accueil.")
    st.page_link("main.py", label="← Retour à l'accueil", icon="🏠")
    st.stop()

# === Vérifier le rôle (gestionnaire ou admin requis) ===
guard_page("optimisation")

# Récupérer les données depuis la session
df = st.session_state['data']
niveau_cols = st.session_state['niveau_cols']
volume_cols = st.session_state['volume_cols']
pluie_cols = st.session_state['pluie_cols']
temp_cols = st.session_state.get('temp_cols', [])
autres_cols = st.session_state.get('autres_cols', [])

tous_les_puits = [col.replace('Depth_to_Groundwater_', '') for col in niveau_cols]

puits_disponibles = []
puits_exclus_data = []
for col in niveau_cols:
    nom = col.replace('Depth_to_Groundwater_', '')
    vol_correspondant = [v for v in volume_cols if nom in v]
    if vol_correspondant:
        puits_disponibles.append(nom)
    else:
        puits_exclus_data.append(nom)

# ============================================================
# Fonctions utilitaires
# ============================================================
def create_sequences(data_arr, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data_arr)):
        X.append(data_arr[i - window_size:i, :])
        y.append(data_arr[i, -1])
    return np.array(X), np.array(y)

def construire_features(nom_puits):
    niv_col = f'Depth_to_Groundwater_{nom_puits}'
    v_col = [v for v in volume_cols if nom_puits in v][0]
    voisins_p = st.session_state.get('voisins', {}).get(nom_puits, [])
    
    feat = []
    for col in pluie_cols:
        if col in df.columns:
            feat.append(col)
    for col in temp_cols:
        if col in df.columns:
            feat.append(col)
    for col in autres_cols:
        if col in df.columns and col not in ['Mois_sin', 'Mois_cos']:
            feat.append(col)
    
    feat.append(v_col)
    
    voisins_vol = []
    for v in voisins_p:
        vv = [vc for vc in volume_cols if v in vc]
        if vv and vv[0] in df.columns:
            feat.append(vv[0])
            voisins_vol.append(vv[0])
    
    feat.extend(['Mois_sin', 'Mois_cos'])
    feat.append(niv_col)
    
    return feat, voisins_vol

# === Titre ===
st.title("⚡ Module 3 - Optimisation")
st.markdown("Trouvez le plan de pompage optimal pour tous vos puits.")

# ============================================================
# Charger le résultat précédent depuis le disque (si existe)
# ============================================================
JSON_PATH = '../models/resultat_optimisation.json'

if 'resultat_optimisation' not in st.session_state:
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, 'r') as f:
                st.session_state['resultat_optimisation'] = json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Impossible de charger le résultat précédent : {e}")

st.divider()

# ============================================================
# SECTION 1 : Vérification des modèles
# ============================================================
st.subheader("🔍 Étape 1 - Vérification des modèles")

st.markdown("""
L'application vérifie que les modèles LSTM sont disponibles et évalue leur fiabilité. Les puits fiables seront optimisés par l'algorithme génétique. Les puits non fiables seront exclus et leur pompage sera maintenu à la dernière valeur connue.
""")

# === Avertissement puits exclus par le pipeline de données ===
if puits_exclus_data:
    st.warning(
        f"⚠️ **Puits exclus du pipeline de données** : {', '.join(puits_exclus_data)}. "
        f"Ces puits n'ont pas de volume de pompage disponible — ils ne sont pas pris en "
        f"compte dans l'optimisation."
    )

if 'graphe' not in st.session_state or 'voisins' not in st.session_state:
    st.error(
        "❌ Le graphe n'a pas été construit. "
        "Allez dans le **Module 2** et cliquez sur **'Analyser les relations'** d'abord."
    )
    st.page_link("pages/2_prediction.py", label="→ Aller au Module 2", icon="🔮")
    st.stop()

WINDOW = 30

# Charger les métadonnées pour vérifier la compatibilité
METADATA_PATH = '../models/prediction/models_metadata.json'
metadata = {}
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

# Vérifier la compatibilité AVANT de charger les modèles
if metadata:
    incompatibles = []
    for nom_puits in puits_disponibles:
        if nom_puits in metadata:
            feat_actuelles, _ = construire_features(nom_puits)
            if metadata[nom_puits].get('n_features') != len(feat_actuelles):
                incompatibles.append(nom_puits)
    
    if incompatibles:
        st.error(
            f"🔴 **Modèles incompatibles détectés** : {', '.join(incompatibles)}. "
            f"Les données ont changé (puits exclus ou ajoutés) — les modèles existants ne "
            f"correspondent plus aux features actuelles. "
            f"**Retournez au Module 2 pour réentraîner les modèles.**"
        )
        st.page_link("pages/2_prediction.py", label="→ Aller au Module 2 pour réentraîner", icon="🔮")
        st.stop()

modeles = {}
scalers = {}
features_map = {}
resultats_r2 = {}

puits_fiables = []
puits_exclus = []

r2_deja_calcules = {}
if 'resultats_entrainement' in st.session_state:
    for r in st.session_state['resultats_entrainement']:
        r2_deja_calcules[r['Puits']] = float(r['R²'])

progress = st.progress(0, text="Chargement des modèles...")

for idx, nom_puits in enumerate(puits_disponibles):
    
    progress.progress(
        (idx + 1) / len(puits_disponibles),
        text=f"Chargement de {nom_puits}... ({idx + 1}/{len(puits_disponibles)})"
    )
    
    model_path = f'../models/prediction/lstm_{nom_puits.lower()}.keras'
    
    if not os.path.exists(model_path):
        st.error(
            f"❌ Modèle manquant pour {nom_puits}. "
            f"Allez dans le **Module 2** et entraînez tous les puits d'abord."
        )
        st.page_link("pages/2_prediction.py", label="→ Aller au Module 2", icon="🔮")
        st.stop()
    
    model = load_model(model_path)
    
    feat, voisins_vol = construire_features(nom_puits)
    niv_col = f'Depth_to_Groundwater_{nom_puits}'
    
    data_p = df[feat].copy()
    data_values = data_p.values
    
    split_idx = int(len(data_values) * 0.8)
    train_d = data_values[:split_idx]
    
    scaler_p = MinMaxScaler(feature_range=(0, 1))
    scaler_p.fit(train_d)
    
    modeles[nom_puits] = model
    scalers[nom_puits] = scaler_p
    features_map[nom_puits] = feat
    
    if nom_puits in r2_deja_calcules:
        r2 = r2_deja_calcules[nom_puits]
    else:
        test_d = data_values[split_idx:]
        train_sc = scaler_p.transform(train_d)
        test_sc = scaler_p.transform(test_d)
        
        X_te, y_te = create_sequences(test_sc, WINDOW)
        n_f = X_te.shape[2]
        
        pred_sc = model.predict(X_te, verbose=0)
        dummy = np.zeros((len(pred_sc), n_f))
        dummy[:, -1] = pred_sc.flatten()
        pred = scaler_p.inverse_transform(dummy)[:, -1]
        
        dummy_t = np.zeros((len(y_te), n_f))
        dummy_t[:, -1] = y_te
        real = scaler_p.inverse_transform(dummy_t)[:, -1]
        
        r2 = r2_score(real, pred)
    
    resultats_r2[nom_puits] = r2
    
    if r2 >= 0.80:
        puits_fiables.append(nom_puits)
    else:
        puits_exclus.append(nom_puits)

progress.empty()

st.success(f"✅ {len(modeles)} modèles chargés et évalués")

col1, col2 = st.columns(2)

with col1:
    st.metric("✅ Puits fiables (optimisés)", len(puits_fiables))
    if puits_fiables:
        for p in puits_fiables:
            st.markdown(f"  - **{p}** - R² = {resultats_r2[p]:.4f}")

with col2:
    st.metric("⚠️ Puits exclus (pompage fixe)", len(puits_exclus))
    if puits_exclus:
        for p in puits_exclus:
            st.markdown(f"  - **{p}** - R² = {resultats_r2[p]:.4f}")

if puits_exclus:
    with st.expander("ℹ️ Pourquoi certains puits sont exclus ?"):
        st.markdown("""
        Un puits est exclu quand son modèle de prédiction n'est **pas fiable** (R² < 0.80). Cela signifie que le modèle ne parvient pas à prédire correctement le comportement de ce puits.
        
        **Causes possibles :**
        - Données historiques insuffisantes ou de mauvaise qualité.
        - Événements extrêmes non représentés dans les données d'entraînement.
        - Amplitude trop faible des variations du niveau.
        
        **Conséquence pour l'optimisation :**
        - Ces puits ne seront **pas optimisés** par l'algorithme génétique.
        - Leur pompage sera **maintenu à la dernière valeur connue**.
        - **Il est recommandé de conserver le régime actuel pour ces puits et de collecter plus de données.**
        - Leur influence sur les puits voisins est **quand même prise en compte** dans le calcul de l'optimisation.
        """)

if len(puits_fiables) < 2:
    st.error(
        "❌ L'optimisation nécessite au moins 2 puits fiables. "
        "Actuellement seul(s) " + ", ".join(puits_fiables) + " est/sont fiable(s). "
        "Améliorez la qualité des données ou collectez plus de mesures."
    )
    st.stop()

st.divider()

# ============================================================
# SECTION 2 : Paramètres de l'optimisation
# ============================================================
st.subheader("⚙️ Étape 2 - Paramètres de l'optimisation")

# Calcul de la demande actuelle
demande_actuelle = 0
for p in puits_disponibles:
    vol_col = [v for v in volume_cols if p in v][0]
    demande_actuelle += df[vol_col].mean()

# Pompage fixe des puits exclus (dernière valeur = régime actuel)
pompage_exclus = {}
total_exclus = 0
for p in puits_exclus:
    vol_col = [v for v in volume_cols if p in v][0]
    dernier_pompage = df[vol_col].iloc[-1]
    pompage_exclus[p] = dernier_pompage
    total_exclus += dernier_pompage

# Formulaire
st.markdown("#### 💧 Demande en eau")

choix_demande = st.radio(
    "Type de demande :",
    ["Demande actuelle", "Demande personnalisée"],
    help="La demande actuelle est calculée à partir des volumes de pompage historiques."
)

# Flag pour savoir si on doit comparer avec le régime actuel
demande_est_personnalisee = (choix_demande == "Demande personnalisée")

if choix_demande == "Demande actuelle":
    DEMANDE_TOTALE = demande_actuelle
    st.info(f"📊 Demande actuelle calculée : **{DEMANDE_TOTALE:.0f} m³/jour**")
else:
    DEMANDE_TOTALE = st.number_input(
        "Entrez la demande totale en eau (m³/jour) :",
        min_value=1000.0,
        max_value=100000.0,
        value=float(round(demande_actuelle)),
        step=1000.0
    )
    ecart = ((DEMANDE_TOTALE - demande_actuelle) / demande_actuelle) * 100
    if ecart > 0:
        st.warning(f"⚠️ Demande supérieure à l'actuelle de **{ecart:+.1f}%** — "
                   f"l'AG cherchera comment satisfaire cette nouvelle demande.")
    elif ecart < -5:
        st.info(f"ℹ️ Demande inférieure à l'actuelle de **{ecart:.1f}%** — "
                f"certains puits pourront être soulagés.")
    
    st.info(
        "ℹ️ Avec une demande personnalisée, l'application affichera **uniquement le plan optimisé** "
        "sans comparaison avec le régime actuel (les deux scénarios répondent à des demandes différentes)."
    )

DEMANDE_AG = DEMANDE_TOTALE - total_exclus

st.markdown("#### 🎛️ Paramètres avancés")

col1, col2 = st.columns(2)

with col1:
    tolerance = st.slider(
        "Tolérance sur la demande (%) :",
        min_value=1, max_value=10, value=5, step=1,
        help="L'AG acceptera un plan de pompage dont la somme "
             "est dans cette marge autour de la demande."
    )

with col2:
    horizon_optim = st.slider(
        "Horizon de simulation (jours) :",
        min_value=7, max_value=30, value=30, step=7,
        help="Nombre de jours simulés pour évaluer l'impact du pompage. "
             "Plus l'horizon est long, plus l'évaluation est réaliste mais lente."
    )

# Contraintes calculées
st.markdown("#### 📋 Contraintes calculées")

contraintes = {}
vol_map = {}

for p in puits_fiables:
    vol_col = [v for v in volume_cols if p in v][0]
    niv_col = f'Depth_to_Groundwater_{p}'
    vol_map[p] = vol_col
    
    contraintes[p] = {
        'vol_mean': df[vol_col].mean(),
        'vol_max': df[vol_col].max(),
        'niv_min': df[niv_col].min(),
        'seuil': df[niv_col].min() - 2
    }

contraintes_exclus = {}
for p in puits_exclus:
    vol_col = [v for v in volume_cols if p in v][0]
    vol_map[p] = vol_col
    contraintes_exclus[p] = {
        'vol_max': df[vol_col].max()
    }

with st.expander("📋 Détail des contraintes - Cliquez pour voir"):
    
    st.markdown("**Puits optimisés :**")
    
    contraintes_data = []
    for p in puits_fiables:
        contraintes_data.append({
            'Puits': p,
            'Débit moyen (m³/j)': f"{contraintes[p]['vol_mean']:.0f}",
            'Débit max (m³/j)': f"{contraintes[p]['vol_max']:.0f}",
            'Seuil critique (m)': f"{contraintes[p]['seuil']:.2f}",
        })
    
    st.dataframe(pd.DataFrame(contraintes_data), 
                 use_container_width=True, hide_index=True)
    
    if puits_exclus:
        st.markdown("**Puits exclus (pompage fixe) :**")
        
        exclus_data = []
        for p in puits_exclus:
            vol_col = [v for v in volume_cols if p in v][0]
            moy_hist = df[vol_col].mean()
            exclus_data.append({
                'Puits': p,
                'Pompage actuel (m³/j)': f"{pompage_exclus[p]:.0f}",
                'Moyenne historique (m³/j)': f"{moy_hist:.0f}",
                'R²': f"{resultats_r2[p]:.4f}",
                'Recommandation': 'Maintenir le régime actuel'
            })
        
        st.dataframe(pd.DataFrame(exclus_data),
                     use_container_width=True, hide_index=True)
    
    st.markdown(f"""
    **Résumé de la demande :**
    - Demande totale : **{DEMANDE_TOTALE:.0f}** m³/jour
    - Pompage fixe des puits exclus : **{total_exclus:.0f}** m³/jour
    - Demande à répartir par l'AG ({len(puits_fiables)} puits) : **{DEMANDE_AG:.0f}** m³/jour
    - Tolérance : ±{tolerance}%
    """)

st.divider()

# ============================================================
# Vérification du résultat précédent (si existe)
# ============================================================
if 'resultat_optimisation' in st.session_state:
    res_stocke = st.session_state['resultat_optimisation']
    
    date_calcul = res_stocke.get('date_calcul', 'Date inconnue')
    type_demande_stocke = res_stocke.get('type_demande', 'Demande actuelle')
    
    # Vérifier si la config actuelle correspond au résultat stocké
    config_identique = (
        res_stocke.get('horizon') == horizon_optim and
        abs(res_stocke.get('DEMANDE_TOTALE', 0) - DEMANDE_TOTALE) < 1 and
        res_stocke.get('tolerance') == tolerance and
        res_stocke.get('type_demande') == choix_demande
    )
    
    if config_identique:
        st.success(
            f"✅ Un résultat d'optimisation correspondant à votre configuration "
            f"actuelle est déjà disponible (voir ci-dessous)."
        )
    else:
        with st.warning(""):
            pass
        st.warning(f"""
        ⚠️ **Un résultat d'optimisation existe** avec une configuration suivante :
        
        - Date du calcul : **{date_calcul}**
        - Demande : **{res_stocke.get('DEMANDE_TOTALE', 0):.0f} m³/jour** ({type_demande_stocke})
        - Horizon : **{res_stocke.get('horizon', 0)} jours**
        - Tolérance : **±{res_stocke.get('tolerance', 0)}%**
        
        Relancez l'optimisation pour obtenir un résultat adapté à votre jeux de données et configuration actuelle.
        Le résultat précédent sera **écrasé**.
        """)

btn_optimiser = st.button("⚡ Lancer l'optimisation", use_container_width=True)


if btn_optimiser:
    
    # ============================================================
    # ÉTAPE 3a : Fonction de simulation récursive
    # ============================================================
    
    df_work = df.copy()
    df_work['mois'] = df_work['Date'].dt.month
    dernier_mois = df['Date'].iloc[-1].month
    
    def get_moyennes_mensuelles(feature_names_puits):
        moyennes = {}
        cols_climat = [col for col in feature_names_puits 
                       if 'Volume' not in col
                       and 'Depth' not in col
                       and col in df_work.columns]
        for col in cols_climat:
            moyennes[col] = df_work.groupby('mois')[col].mean()
        return moyennes
    
    def simuler_puits_recursif(puits, debit_propose, debits_voisins, n_jours):
        """
        Simule l'évolution du niveau d'un puits sur n_jours.
        
        - Les 30 jours de départ sont les données RÉELLES (non modifiées)
        - Les jours futurs utilisent :
            - Pompage = valeur proposée par l'AG
            - Climat = moyennes mensuelles historiques
            - Niveau = prédit par le LSTM
        """
        
        model = modeles[puits]
        scaler = scalers[puits]
        feat = features_map[puits]
        niv_col = f'Depth_to_Groundwater_{puits}'
        vol_col = vol_map[puits]
        voisins_p = st.session_state['voisins'].get(puits, [])
        
        moyennes = get_moyennes_mensuelles(feat)
        
        data_puits = df[feat].copy()
        sequence = data_puits.values[-30:].copy()
        
        idx_vol = feat.index(vol_col)
        idx_niv = feat.index(niv_col)
        
        n_feat = sequence.shape[1]
        niveaux = []
        
        for jour in range(n_jours):
            seq_scaled = scaler.transform(sequence)
            X = seq_scaled.reshape(1, 30, n_feat)
            
            pred_sc = model.predict(X, verbose=0)
            
            dummy = np.zeros((1, n_feat))
            dummy[0, -1] = pred_sc[0, 0]
            niveau_predit = scaler.inverse_transform(dummy)[0, -1]
            
            niveaux.append(niveau_predit)
            
            # Date du jour futur
            date_future = pd.Timestamp(df['Date'].iloc[-1]) + pd.Timedelta(days=jour + 1)
            mois_futur = date_future.month
            
            nouveau_jour = np.zeros(n_feat)
            
            for idx_feat, col in enumerate(feat):
                if col == niv_col:
                    nouveau_jour[idx_feat] = niveau_predit
                elif col == 'Mois_sin':
                    nouveau_jour[idx_feat] = np.sin(2 * np.pi * mois_futur / 12)
                elif col == 'Mois_cos':
                    nouveau_jour[idx_feat] = np.cos(2 * np.pi * mois_futur / 12)
                elif col == vol_col:
                    nouveau_jour[idx_feat] = debit_propose
                elif col in moyennes:
                    nouveau_jour[idx_feat] = moyennes[col].get(
                        mois_futur, sequence[-1, idx_feat])
                else:
                    trouve = False
                    for v in voisins_p:
                        v_vol_col = vol_map.get(v)
                        if v_vol_col == col and v in debits_voisins:
                            nouveau_jour[idx_feat] = debits_voisins[v]
                            trouve = True
                            break
                    if not trouve:
                        nouveau_jour[idx_feat] = sequence[-1, idx_feat]
            
            sequence = np.vstack([sequence[1:], nouveau_jour.reshape(1, -1)])
        
        return niveaux
    
    # ============================================================
    # ÉTAPE 3b : Fonction de fitness
    # ============================================================
    
    TOLERANCE = tolerance / 100.0
    G = st.session_state['graphe']
    
    def fitness(individu):
        debits = dict(zip(puits_fiables, individu))
        penalites = 0
        
        # 1. Vérifier la demande (pénalité)
        somme = sum(individu)
        ecart = abs(somme - DEMANDE_AG) / DEMANDE_AG
        if ecart > TOLERANCE:
            penalites += ecart * 100
        
        # 2. Simuler chaque puits
        niveaux_min = {}
        
        for puits in puits_fiables:
            voisins_p = st.session_state['voisins'].get(puits, [])
            debits_voisins = {}
            
            for v in voisins_p:
                if v in debits:
                    debits_voisins[v] = debits[v]
                else:
                    debits_voisins[v] = pompage_exclus.get(
                        v, df[vol_map.get(v, volume_cols[0])].mean())
            
            niveaux = simuler_puits_recursif(
                puits, debits[puits], debits_voisins, horizon_optim)
            
            niveaux_min[puits] = min(niveaux)
        
        # 3. Vérifier les seuils (pénalité)
        for puits in puits_fiables:
            niveau_min = niveaux_min[puits]
            seuil = contraintes[puits]['seuil']
            if niveau_min < seuil:
                penalites += abs(niveau_min - seuil) * 10
        
        # 4. Contraintes du graphe (pénalité)
        tous_les_debits = dict(debits)
        tous_les_debits.update(pompage_exclus)
        
        if G is not None:
            for u, v, data_edge in G.edges(data=True):
                if u in tous_les_debits and v in tous_les_debits:
                    poids = data_edge['weight']
                    
                    if u in contraintes:
                        max_u = contraintes[u]['vol_max']
                    elif u in contraintes_exclus:
                        max_u = contraintes_exclus[u]['vol_max']
                    else:
                        continue
                    
                    if v in contraintes:
                        max_v = contraintes[v]['vol_max']
                    elif v in contraintes_exclus:
                        max_v = contraintes_exclus[v]['vol_max']
                    else:
                        continue
                    
                    ratio_u = tous_les_debits[u] / max_u
                    ratio_v = tous_les_debits[v] / max_v
                    penalite_graphe = poids * ratio_u * ratio_v * 20
                    penalites += penalite_graphe
        
        # 5. Score
        niveau_min_moyen = np.mean(list(niveaux_min.values()))
        score = niveau_min_moyen - penalites
        
        return (score,)
    
    # ============================================================
    # Score de référence (uniquement si demande actuelle)
    # ============================================================
    st.divider()
    st.subheader("🔄 Optimisation en cours")
    
    if not demande_est_personnalisee:
        with st.spinner("📊 Évaluation du régime actuel..."):
            individu_ref = []
            for p in puits_fiables:
                individu_ref.append(df[vol_map[p]].iloc[-1])
            score_ref = fitness(individu_ref)[0]
        
        st.info(
            f"📊 Régime actuel évalué. L'algorithme génétique va maintenant "
            f"chercher une meilleure répartition du pompage pour les "
            f"**{len(puits_fiables)} puits fiables**."
        )
    else:
        individu_ref = []
        for p in puits_fiables:
            individu_ref.append(df[vol_map[p]].iloc[-1])
        score_ref = None
        
        st.info(
            f"📊 Recherche d'un plan de pompage optimal pour une demande de "
            f"**{DEMANDE_TOTALE:.0f} m³/jour** ({len(puits_fiables)} puits fiables)."
        )
    
    # ============================================================
    # ÉTAPE 4 : Algorithme génétique
    # ============================================================
    
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    BORNES_MIN = [0] * len(puits_fiables)
    BORNES_MAX = [contraintes[p]['vol_max'] for p in puits_fiables]
    
    def creer_individu():
        return [random.uniform(bmin, bmax) 
                for bmin, bmax in zip(BORNES_MIN, BORNES_MAX)]
    
    def borner_individu(individu):
        for i in range(len(individu)):
            individu[i] = max(BORNES_MIN[i], min(BORNES_MAX[i], individu[i]))
        return individu
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, creer_individu)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=500, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    POP_SIZE = 20
    N_GEN = 30
    CX_PROB = 0.7
    MUT_PROB = 0.2
    
    random.seed(42)
    np.random.seed(42)
    
    population = toolbox.population(n=POP_SIZE)
    progress_ag = st.progress(0, text="Création de la population initiale...")
    
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    
    progress_ag.progress(1 / (N_GEN + 1), text=f"Génération 0/{N_GEN} terminée")
    
    for gen in range(1, N_GEN + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                borner_individu(mutant)
                del mutant.fitness.values
        
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalids:
            ind.fitness.values = toolbox.evaluate(ind)
        
        population[:] = offspring
        
        progress_ag.progress(
            (gen + 1) / (N_GEN + 1),
            text=f"Génération {gen}/{N_GEN} terminée"
        )
    
    progress_ag.progress(1.0, text="Optimisation terminée !")
    
    meilleur = tools.selBest(population, 1)[0]
    score_final = meilleur.fitness.values[0]
    
    # Simulation finale
    debits_opt = dict(zip(puits_fiables, meilleur))
    debits_ref = dict(zip(puits_fiables, individu_ref))
    
    simulation_resultats = {}
    
    for puits in puits_fiables:
        voisins_p = st.session_state['voisins'].get(puits, [])
        
        # Voisins scénario actuel (uniquement si demande actuelle)
        if not demande_est_personnalisee:
            dv_ref = {}
            for v in voisins_p:
                if v in debits_ref:
                    dv_ref[v] = debits_ref[v]
                else:
                    dv_ref[v] = pompage_exclus.get(v, df[vol_map.get(v, volume_cols[0])].iloc[-1])
            niv_ref = simuler_puits_recursif(puits, debits_ref[puits], dv_ref, horizon_optim)
        else:
            niv_ref = None
        
        # Voisins scénario optimisé
        dv_opt = {}
        for v in voisins_p:
            if v in debits_opt:
                dv_opt[v] = debits_opt[v]
            else:
                dv_opt[v] = pompage_exclus.get(v, df[vol_map.get(v, volume_cols[0])].mean())
        
        niv_opt = simuler_puits_recursif(puits, debits_opt[puits], dv_opt, horizon_optim)
        
        simulation_resultats[puits] = {
            'niveaux_ref': [float(x) for x in niv_ref] if niv_ref is not None else None,
            'niveaux_opt': [float(x) for x in niv_opt]
        }
    
    # Sauvegarder dans la session
    resultat_complet = {
        'date_calcul': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'type_demande': choix_demande,
        'demande_est_personnalisee': demande_est_personnalisee,
        'meilleur': [float(x) for x in meilleur],
        'score_ref': float(score_ref) if score_ref is not None else None,
        'score_final': float(score_final),
        'individu_ref': [float(x) for x in individu_ref],
        'puits_fiables': puits_fiables,
        'puits_exclus': puits_exclus,
        'pompage_exclus': {k: float(v) for k, v in pompage_exclus.items()},
        'contraintes': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in contraintes.items()},
        'horizon': horizon_optim,
        'tolerance': tolerance,
        'DEMANDE_AG': float(DEMANDE_AG),
        'DEMANDE_TOTALE': float(DEMANDE_TOTALE),
        'total_exclus': float(total_exclus),
        'simulation': simulation_resultats
    }
    
    st.session_state['resultat_optimisation'] = resultat_complet
    
    # Sauvegarder sur disque
    try:
        os.makedirs('../models', exist_ok=True)
        with open(JSON_PATH, 'w') as f:
            json.dump(resultat_complet, f, indent=2)
    except Exception as e:
        st.warning(f"⚠️ Impossible de sauvegarder sur disque : {e}")
    
    st.rerun()

# ============================================================
# SECTION 5 : Affichage des résultats
# ============================================================

if 'resultat_optimisation' in st.session_state:
    
    res = st.session_state['resultat_optimisation']
    
    meilleur = res['meilleur']
    score_ref = res.get('score_ref')
    score_final = res['score_final']
    individu_ref = res['individu_ref']
    puits_f = res['puits_fiables']
    puits_e = res['puits_exclus']
    pomp_exclus = res['pompage_exclus']
    contr = res['contraintes']
    horizon_res = res['horizon']
    demande_ag = res['DEMANDE_AG']
    demande_totale = res['DEMANDE_TOTALE']
    total_excl = res['total_exclus']
    simulation = res['simulation']
    est_personnalisee = res.get('demande_est_personnalisee', False)
    date_calc = res.get('date_calcul', 'N/A')
    type_dem = res.get('type_demande', 'Demande actuelle')
    
    st.divider()
    st.subheader("📊 Résultats de l'optimisation")
    
    st.caption(
        f"📅 Calculé le **{date_calc}** | "
        f"Demande : **{demande_totale:.0f} m³/j** ({type_dem}) | "
        f"Horizon : **{horizon_res} jours** | "
        f"{len(puits_f)} puits optimisés | "
        f"{len(puits_e)} puits exclus"
    )
    
    # ── 5a : Plan de pompage ──
    st.markdown("#### 💧 Plan de pompage recommandé")
    
    plan_data = []
    total_actuel = 0
    total_optimise = 0
    
    for i, puits in enumerate(puits_f):
        debit_actuel = individu_ref[i]
        debit_opt = meilleur[i]
        total_actuel += debit_actuel
        total_optimise += debit_opt
        
        if est_personnalisee:
            # Pas de comparaison, juste le plan
            plan_data.append({
                'Puits': puits,
                'Pompage recommandé (m³/j)': f"{debit_opt:.0f}",
                'Capacité max (m³/j)': f"{contr[puits]['vol_max']:.0f}",
                'Utilisation': f"{(debit_opt / contr[puits]['vol_max'] * 100):.0f}%"
            })
        else:
            variation = ((debit_opt - debit_actuel) / debit_actuel * 100) if debit_actuel > 0 else 0
            
            if variation > 5:
                action = "↑ Augmenter"
            elif variation < -5:
                action = "↓ Réduire"
            else:
                action = "≈ Maintenir"
            
            plan_data.append({
                'Puits': puits,
                'Pompage actuel (m³/j)': f"{debit_actuel:.0f}",
                'Pompage optimisé (m³/j)': f"{debit_opt:.0f}",
                'Variation': f"{variation:+.1f}%",
                'Action': action
            })
    
    plan_df = pd.DataFrame(plan_data)
    st.dataframe(plan_df, use_container_width=True, hide_index=True)
    
    # Totaux
    col1, col2, col3 = st.columns(3)
    if est_personnalisee:
        with col1:
            st.metric("Demande à satisfaire", f"{demande_totale:.0f} m³/j")
        with col2:
            st.metric("Plan optimisé (fiables)", f"{total_optimise:.0f} m³/j")
        with col3:
            total_avec_exclus = total_optimise + total_excl
            taux = total_avec_exclus / demande_totale * 100
            st.metric("Total satisfait", 
                      f"{total_avec_exclus:.0f} m³/j",
                      delta=f"{taux:.1f}% de la demande")
    else:
        with col1:
            st.metric("Pompage actuel (puits optimisés)", f"{total_actuel:.0f} m³/j")
        with col2:
            st.metric("Pompage optimisé", f"{total_optimise:.0f} m³/j")
        with col3:
            st.metric("Total avec exclus", 
                      f"{total_optimise + total_excl:.0f} m³/j",
                      delta=f"sur {demande_totale:.0f} demandés")
    
    # ── 5b : Puits exclus ──
    if puits_e:
        st.markdown("#### ⚠️ Puits non optimisés — Recommandation")
        for p in puits_e:
            st.warning(
                f"**{p}** → maintenir le régime actuel "
                f"(**{pomp_exclus[p]:.0f} m³/jour**, dernière valeur mesurée). "
                f"Ce puits n'a pas pu être modélisé avec une précision suffisante."
            )
    
    st.divider()
    
    # ── 5c : Analyse des niveaux prédits ──
    if est_personnalisee:
        # Mode demande personnalisée : pas de comparaison, juste le diagnostic
        st.markdown("#### 📈 Diagnostic des niveaux avec le plan optimisé")
        st.markdown(
            f"Simulation sur **{horizon_res} jours** avec le plan de pompage "
            f"proposé par l'algorithme génétique."
        )
        
        diag_data = []
        violations = 0
        marges = []
        
        for puits in puits_f:
            niv_opt = simulation[puits]['niveaux_opt']
            seuil = contr[puits]['seuil']
            
            min_opt = min(niv_opt)
            final_opt = niv_opt[-1]
            marge = min_opt - seuil
            marges.append(marge)
            
            if marge < 0:
                violations += 1
                etat = "🔴 Danger"
            elif marge < 3:
                etat = "🟡 Attention"
            else:
                etat = "🟢 Sûr"
            
            diag_data.append({
                'Puits': puits,
                'Niveau min prédit': f"{min_opt:.2f}m",
                'Niveau final': f"{final_opt:.2f}m",
                'Seuil critique': f"{seuil:.2f}m",
                'Marge au seuil': f"{marge:.2f}m",
                'État': etat
            })
        
        st.dataframe(pd.DataFrame(diag_data), use_container_width=True, hide_index=True)
    
    else:
        # Mode demande actuelle : comparaison complète
        st.markdown("#### 📈 Comparaison des niveaux : Actuel vs Optimisé")
        st.markdown(
            f"Simulation sur **{horizon_res} jours**. "
            f"Le scénario actuel (rouge) maintient le pompage actuel. "
            f"Le scénario optimisé (vert) applique le plan recommandé."
        )
        
        comparaison_data = []
        violations = 0
        marges = []
        
        for puits in puits_f:
            niv_ref = simulation[puits]['niveaux_ref']
            niv_opt = simulation[puits]['niveaux_opt']
            seuil = contr[puits]['seuil']
            
            min_ref = min(niv_ref)
            min_opt = min(niv_opt)
            gain = min_opt - min_ref
            marge = min_opt - seuil
            marges.append(marge)
            
            if marge < 0:
                violations += 1
                etat = "🔴 Danger"
            elif marge < 3:
                etat = "🟡 Attention"
            else:
                etat = "🟢 Sûr"
            
            comparaison_data.append({
                'Puits': puits,
                'Niveau min (actuel)': f"{min_ref:.2f}m",
                'Niveau min (optimisé)': f"{min_opt:.2f}m",
                'Gain': f"{gain:+.2f}m",
                'Marge au seuil': f"{marge:.2f}m",
                'État': etat
            })
        
        st.dataframe(pd.DataFrame(comparaison_data), use_container_width=True, hide_index=True)
    
    # ── 5d : Graphiques par puits ──
    st.markdown("#### 📉 Courbes d'évolution par puits")
    
    jours = list(range(1, horizon_res + 1))
    
    for i in range(0, len(puits_f), 2):
        cols = st.columns(2)
        
        for j in range(2):
            if i + j < len(puits_f):
                puits = puits_f[i + j]
                niv_opt = simulation[puits]['niveaux_opt']
                seuil = contr[puits]['seuil']
                debit_opt_val = meilleur[i + j]
                
                fig = go.Figure()
                
                if not est_personnalisee:
                    niv_ref = simulation[puits]['niveaux_ref']
                    debit_act = individu_ref[i + j]
                    gain = min(niv_opt) - min(niv_ref)
                    
                    fig.add_trace(go.Scatter(
                        x=jours, y=niv_ref,
                        mode='lines', name=f'Actuel ({debit_act:.0f} m³/j)',
                        line=dict(color='#DC2626', width=1.5)
                    ))
                    
                    titre = f"{puits} - Gain : {gain:+.2f}m"
                else:
                    titre = f"{puits} - {debit_opt_val:.0f} m³/j"
                
                fig.add_trace(go.Scatter(
                    x=jours, y=niv_opt,
                    mode='lines', name=f'Optimisé ({debit_opt_val:.0f} m³/j)',
                    line=dict(color='#059669', width=1.5)
                ))
                
                fig.add_hline(
                    y=seuil, line_dash="dash", line_color="gray",
                    annotation_text=f"Seuil ({seuil:.0f}m)"
                )
                
                fig.add_hrect(
                    y0=seuil - 10, y1=seuil,
                    fillcolor="rgba(220, 38, 38, 0.05)",
                    line_width=0
                )
                
                fig.update_layout(
                    title=titre,
                    xaxis_title="Jour", yaxis_title="Niveau (m)",
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
                )
                
                with cols[j]:
                    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ── 5e : Analyse et interprétation ──
    if not est_personnalisee:
        st.markdown("#### 🔍 Analyse et interprétation")
        
        augmentes = []
        reduits = []
        maintenus = []
        
        for i, puits in enumerate(puits_f):
            variation = ((meilleur[i] - individu_ref[i]) / individu_ref[i] * 100) if individu_ref[i] > 0 else 0
            if variation > 5:
                augmentes.append((puits, variation))
            elif variation < -5:
                reduits.append((puits, variation))
            else:
                maintenus.append((puits, variation))
        
        if augmentes:
            st.markdown("**↑ Puits à augmenter :**")
            for puits, var in augmentes:
                voisins_p = st.session_state['voisins'].get(puits, [])
                if not voisins_p:
                    raison = "puits isolé - aucun impact sur les voisins"
                else:
                    raison = f"connecté à {', '.join(voisins_p)} - marge suffisante"
                st.success(f"**{puits}** ({var:+.1f}%) - {raison}")
        
        if reduits:
            st.markdown("**↓ Puits à réduire :**")
            for puits, var in reduits:
                voisins_p = st.session_state['voisins'].get(puits, [])
                if voisins_p:
                    raison = f"zone sous pression (voisins : {', '.join(voisins_p)})"
                else:
                    raison = "proche du seuil critique"
                st.warning(f"**{puits}** ({var:+.1f}%) - {raison}")
        
        if maintenus:
            st.markdown("**≈ Puits à maintenir :**")
            for puits, var in maintenus:
                st.info(f"**{puits}** ({var:+.1f}%) - régime actuel adéquat")
        
        st.divider()
    
    # ── 5f : Conclusion adaptative ──
    st.markdown("#### ✅ Conclusion")
    
    marge_min = min(marges)
    amelioration = (score_final - score_ref) if score_ref is not None else None
    
    # Identifier le cas
    if est_personnalisee:
        # Cas demande personnalisée
        if violations == 0:
            st.success(
                f"🟢 **Le plan proposé est viable.** L'algorithme a trouvé une "
                f"répartition qui satisfait la demande de **{demande_totale:.0f} m³/jour** "
                f"sur les {horizon_res} prochains jours sans dépasser aucun seuil critique."
            )
            if marge_min < 3:
                st.warning(
                    f"⚠️ Cependant, certains puits ont une marge faible "
                    f"(minimum : **{marge_min:.1f}m**). Surveillez ces puits "
                    f"de près et collectez des données supplémentaires."
                )
        elif violations <= 2:
            st.warning(
                f"🟡 **Le plan est risqué** : {violations} puits dépassent le seuil critique. "
                f"La demande de **{demande_totale:.0f} m³/jour** est probablement trop élevée. "
                f"Réduisez la demande ou acceptez ce risque avec une surveillance accrue."
            )
        else:
            st.error(
                f"🔴 **Demande non satisfiable en toute sécurité.** "
                f"{violations} puits violeraient le seuil critique. "
                f"Réduisez significativement la demande ou explorez d'autres sources d'eau."
            )
    else:
        # Cas demande actuelle
        if amelioration is None or amelioration <= 1:
            st.info(
                f"ℹ️ **Le régime actuel est déjà proche de l'optimal.** "
                f"L'algorithme n'a pas trouvé d'amélioration significative. "
                f"Continuez avec la répartition actuelle."
            )
        elif violations == 0:
            st.success(
                f"🟢 **Optimisation réussie.** Le plan optimisé est sûr "
                f"(aucune violation de seuil) et meilleur que le régime actuel. "
                f"Adopter cette nouvelle répartition améliorera la durabilité de votre nappe."
            )
        elif violations <= 2:
            st.warning(
                f"🟡 **Optimisation partielle.** Le plan améliore la répartition globale "
                f"mais {violations} puits restent en attention. Recommandation : adopter "
                f"le plan tout en surveillant les puits concernés."
            )
        else:
            st.error(
                f"🔴 **L'algorithme n'a pas trouvé de solution viable.** "
                f"Le plan présente {violations} violations de seuil. "
                f"Réduisez la demande ou augmentez la tolérance et relancez."
            )
    
    # Résumé de la stratégie (uniquement si demande actuelle)
    if not est_personnalisee:
        st.markdown(f"""
        **Stratégie identifiée par l'algorithme :**
        
        L'optimisation propose un **transfert de charge** : réduire le pompage des puits 
        situés dans les zones centrales du réseau (fortement connectés) et augmenter celui 
        des puits isolés ou périphériques qui peuvent absorber plus de charge sans affecter 
        leurs voisins.
        
        - **{len(augmentes)}** puits à augmenter (puits isolés ou avec marge suffisante)
        - **{len(reduits)}** puits à réduire (zones sous pression)
        - **{len(maintenus)}** puits à maintenir (régime actuel adéquat)
        - **{len(puits_e)}** puits exclus (pompage fixe, données insuffisantes)
        """)
    else:
        st.markdown(f"""
        **Répartition proposée :**
        
        - **{len(puits_f)}** puits actifs dans l'optimisation
        - **{len(puits_e)}** puits exclus (pompage fixe)
        - Demande totale satisfaite : **{total_optimise + total_excl:.0f} m³/jour** 
          sur **{demande_totale:.0f}** demandés
        """)
    
    with st.expander("🔧 Détails techniques"):
        if score_ref is not None:
            st.markdown(f"""
            - Score régime actuel : **{score_ref:.2f}**
            - Score optimisé : **{score_final:.2f}**
            - Amélioration : **{amelioration:+.2f}** points
            """)
        else:
            st.markdown(f"""
            - Score du plan optimisé : **{score_final:.2f}**
            - Pas de comparaison avec le régime actuel (demande personnalisée)
            """)
        
        st.markdown(f"""
        - Population : 20 individus
        - Générations : 30
        - Horizon de simulation : {horizon_res} jours
        """)
    
    with st.expander("ℹ️ Comment lire ces résultats ?"):
        st.markdown(f"""
        **Le tableau du plan de pompage** montre le pompage recommandé pour chaque puits.
        
        **Les courbes d'évolution** montrent comment le niveau de chaque puits évoluera 
        sur les {horizon_res} prochains jours avec le plan proposé.
        
        **Les états :**
        - 🟢 **Sûr** : marge > 3m au-dessus du seuil
        - 🟡 **Attention** : marge entre 0 et 3m
        - 🔴 **Danger** : le niveau descend sous le seuil critique
        
        **Si les résultats ne sont pas satisfaisants :**
        - Réduisez la demande en eau
        - Augmentez la tolérance
        - Essayez un horizon différent
        """)
