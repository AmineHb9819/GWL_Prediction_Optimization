# ============================================================
# Module 2 — Prédiction (Prédire)
# ============================================================
# Deux actions possibles :
# 1. "Entraîner tous les puits" → entraîne les LSTM, affiche les R²
# 2. "Prédire ce puits" → charge le modèle existant, affiche la prédiction
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import networkx as nx
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.auth import guard_page

# === Vérifier que les données sont chargées ===
if 'data' not in st.session_state:
    st.warning("⚠️ Aucune donnée chargée. Retournez à la page d'accueil.")
    st.page_link("main.py", label="← Retour à l'accueil", icon="🏠")
    st.stop()

# === Vérifier le rôle (gestionnaire ou admin requis) ===
guard_page("prediction")

# Récupérer les données depuis la session
df = st.session_state['data']
niveau_cols = st.session_state['niveau_cols']
volume_cols = st.session_state['volume_cols']
pluie_cols = st.session_state['pluie_cols']
temp_cols = st.session_state.get('temp_cols', [])
autres_cols = st.session_state.get('autres_cols', [])

# Noms de tous les puits
tous_les_puits = [col.replace('Depth_to_Groundwater_', '') for col in niveau_cols]

# Puits avec volume de pompage (utilisables pour la prédiction)
puits_disponibles = []
puits_exclus_data = []
for col in niveau_cols:
    nom = col.replace('Depth_to_Groundwater_', '')
    vol_correspondant = [v for v in volume_cols if nom in v]
    if vol_correspondant:
        puits_disponibles.append(nom)
    else:
        puits_exclus_data.append(nom)

# Chemin du fichier de métadonnées des modèles
METADATA_PATH = '../models/prediction/models_metadata.json'

# Fonction réutilisable pour créer les séquences
def create_sequences(data_arr, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data_arr)):
        X.append(data_arr[i - window_size:i, :])
        y.append(data_arr[i, -1])
    return np.array(X), np.array(y)

# Fonction réutilisable pour construire les features d'un puits
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
st.title("🔮 Module 2 - Prédiction")
st.markdown("Prédisez le niveau futur d'un puits sélectionné grâce au modèle LSTM.")

# === Avertissement puits exclus par le pipeline de données ===
if puits_exclus_data:
    st.warning(
        f"⚠️ **Puits exclus du pipeline de données** : {', '.join(puits_exclus_data)}. "
        f"Ces puits n'ont pas de volume de pompage disponible, ils ne seront pas pris en compte dans l'analyse, la prédiction et l'optimisation."
    )

st.divider()

# ============================================================
# SECTION 1 : Relations entre les puits (Graphe)
# ============================================================
st.subheader("🔗 Étape 1 - Relations entre les puits")

st.markdown("""
Avant de prédire, l'application analyse les **relations d'influence** entre les puits. Si deux puits sont liés (corrélation > 0.6), le pompage de l'un affecte le niveau de l'autre.  
Ces relations seront utilisées pour améliorer la prédiction.
""")

btn_graphe = st.button("🔗 Analyser les relations", use_container_width=True)

if btn_graphe or 'graphe' in st.session_state:
    
    if 'graphe' not in st.session_state:
        if len(puits_disponibles) >= 2:
            # Colonnes de niveau uniquement pour les puits avec pompage
            niveau_cols_dispo = [f'Depth_to_Groundwater_{p}' for p in puits_disponibles]
            corr = df[niveau_cols_dispo].corr()
            
            G = nx.Graph()
            for p in puits_disponibles:
                G.add_node(p)
            
            for i in range(len(puits_disponibles)):
                for j in range(i + 1, len(puits_disponibles)):
                    p1 = puits_disponibles[i]
                    p2 = puits_disponibles[j]
                    c = abs(corr.loc[niveau_cols_dispo[i], niveau_cols_dispo[j]])
                    if c > 0.6:
                        G.add_edge(p1, p2, weight=c)
            
            st.session_state['graphe'] = G
            st.session_state['voisins'] = {
                p: list(G.neighbors(p)) for p in puits_disponibles
            }
        else:
            st.session_state['graphe'] = None
            st.session_state['voisins'] = {puits_disponibles[0]: []} if puits_disponibles else {}
    
    if st.session_state['graphe'] is not None:
        G = st.session_state['graphe']
        voisins_dict = st.session_state['voisins']
        
        st.success(f"✅ Graphe construit : {G.number_of_nodes()} puits, "
                   f"{G.number_of_edges()} connexions")
        
        relations_data = []
        for p in puits_disponibles:
            v = voisins_dict.get(p, [])
            if v:
                relations_data.append({
                    'Puits': p,
                    'Nombre de voisins': len(v),
                    'Voisins': ', '.join(v),
                    'Statut': '🔗 Connecté'
                })
            else:
                relations_data.append({
                    'Puits': p,
                    'Nombre de voisins': 0,
                    'Voisins': '-',
                    'Statut': '🔘 Isolé'
                })
        
        relations_df = pd.DataFrame(relations_data)
        st.dataframe(relations_df, use_container_width=True, hide_index=True)
        with st.expander("ℹ️ Comment lire le tableau ?"):
            st.markdown("""
            - **Connecté** = ce puits est influencé par ses voisins. Le LSTM utilisera le pompage des voisins pour améliorer la prédiction.
            - **Isolé** = ce puits est indépendant. Le LSTM utilisera uniquement ses propres données.
            """)
    # Visualisation du graphe avec Plotly
        st.markdown("#### 🕸️ Réseau des puits")
        
        import math
        
        # Calculer les positions des nœuds (layout circulaire)
        n_nodes = len(puits_disponibles)
        pos = {}
        for i, p in enumerate(puits_disponibles):
            angle = 2 * math.pi * i / n_nodes
            pos[p] = (math.cos(angle), math.sin(angle))
        
        # Calculer la centralité de degré (nombre de connexions / max possible)
        if G.number_of_edges() > 0:
            centralite = nx.degree_centrality(G)
        else:
            centralite = {p: 0 for p in puits_disponibles}
        
        fig_graphe = go.Figure()
        
        # Dessiner les arêtes
        niveau_cols_dispo = [f'Depth_to_Groundwater_{p}' for p in puits_disponibles]
        corr_matrix = df[niveau_cols_dispo].corr()
        
        for u, v, data_edge in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Trouver la corrélation réelle (avec signe)
            col_u = f'Depth_to_Groundwater_{u}'
            col_v = f'Depth_to_Groundwater_{v}'
            corr_val = corr_matrix.loc[col_u, col_v]
            
            if corr_val < 0:
                couleur_arete = '#DC2626'  # rouge = corrélation négative
                style = 'dash'
                label = f"{u} ↔ {v} : {corr_val:.2f} (opposition)"
            else:
                couleur_arete = '#2563EB'  # bleu = corrélation positive
                style = 'solid'
                label = f"{u} ↔ {v} : {corr_val:.2f} (liaison)"
            
            epaisseur = abs(corr_val) * 4
            
            fig_graphe.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(color=couleur_arete, width=epaisseur, dash=style),
                hoverinfo='text',
                text=label,
                showlegend=False
            ))
        
        # Dessiner les nœuds
        for p in puits_disponibles:
            x, y = pos[p]
            
            # Taille selon la centralité
            taille = 30 + centralite.get(p, 0) * 60
            
            # Couleur selon le nombre de connexions
            n_voisins = len(voisins_dict.get(p, []))
            if n_voisins == 0:
                couleur = '#9CA3AF'  # gris = isolé
                statut = 'Isolé'
            elif n_voisins >= 4:
                couleur = '#DC2626'  # rouge = très connecté (zone sensible)
                statut = f'{n_voisins} connexions (central)'
            elif n_voisins >= 2:
                couleur = '#F59E0B'  # orange = moyennement connecté
                statut = f'{n_voisins} connexions'
            else:
                couleur = '#059669'  # vert = peu connecté
                statut = f'{n_voisins} connexion'
            
            fig_graphe.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=taille, color=couleur, 
                           line=dict(width=2, color='white')),
                text=p,
                textposition='top center',
                textfont=dict(size=12, color='black'),
                hoverinfo='text',
                hovertext=f"<b>{p}</b><br>"
                          f"Centralité : {centralite.get(p, 0):.3f}<br>"
                          f"Voisins : {n_voisins}<br>"
                          f"Statut : {statut}",
                showlegend=False
            ))
        
        # Légende manuelle
        fig_graphe.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='#2563EB'),
            name='Corrélation positive'
        ))
        fig_graphe.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='#DC2626'),
            name='Corrélation négative'
        ))
        fig_graphe.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='#9CA3AF'),
            name='Puits isolé'
        ))
        
        fig_graphe.update_layout(
            title="Réseau d'influence entre les puits",
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_graphe, use_container_width=True)
        
        with st.expander("ℹ️ Comment lire ce graphe ?"):
            st.markdown("""
            **Les cercles représentent les puits :**
            - **Taille** : plus le cercle est grand, plus le puits est central dans le réseau
            - **Rouge** : puits très connecté (zone sensible : pomper ici affecte beaucoup de voisins)
            - **Orange** : puits moyennement connecté
            - **Vert** : puits peu connecté
            - **Gris** : puits isolé (pomper ici n'affecte aucun voisin)
            
            **Les lignes représentent les connexions :**
            - **Ligne bleue continue** : corrélation positive (les deux puits montent et descendent ensemble)
            - **Ligne rouge pointillée** : corrélation négative (quand l'un monte, l'autre descend = vases communicants)
            - **Épaisseur** : plus la ligne est épaisse, plus la corrélation est forte
            """)
    else:
        st.info("ℹ️ Un seul puits détecté, pas de graphe à construire")

st.divider()

# ============================================================
# SECTION 2 : Entraînement des modèles
# ============================================================
st.subheader("🧠 Étape 2 - Entraînement des modèles")

st.markdown("""
Entraînez les modèles LSTM pour tous les puits.  
Cette étape est nécessaire **une seule fois**. Les modèles seront sauvegardés et réutilisés pour la prédiction et pour l'optimisation (Module 3).
""")

# Vérifier quels modèles existent déjà
modeles_existants = []
modeles_manquants = []
modeles_incompatibles = []

# Charger les métadonnées si elles existent
metadata = {}
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

for nom in puits_disponibles:
    model_path = f'../models/prediction/lstm_{nom.lower()}.keras'
    if os.path.exists(model_path):
        # Vérifier la compatibilité des features
        if 'voisins' in st.session_state:
            feat_actuelles, _ = construire_features(nom)
            n_features_actuelles = len(feat_actuelles)
            
            # Comparer avec les métadonnées sauvegardées
            if nom in metadata and metadata[nom].get('n_features') != n_features_actuelles:
                modeles_incompatibles.append(nom)
            else:
                modeles_existants.append(nom)
        else:
            modeles_existants.append(nom)
    else:
        modeles_manquants.append(nom)

# Afficher le statut des modèles
if modeles_incompatibles:
    st.error(
        f"🔴 **Modèles incompatibles détectés** : {', '.join(modeles_incompatibles)}. "
        f"Les données ont changé (puits exclus ou ajoutés), les modèles existants ne correspondent plus aux features actuelles. **Un réentraînement est obligatoire.**"
    )
elif len(modeles_existants) == len(puits_disponibles) and not modeles_manquants:
    st.success(f"✅ Tous les modèles sont prêts ({len(modeles_existants)} puits). "
               f"Vous pouvez passer directement à la prédiction (Étape 3).")
elif modeles_existants:
    st.info(f"📂 Modèles existants : {', '.join(modeles_existants)}")
    st.warning(f"🔄 Modèles à entraîner : {', '.join(modeles_manquants)}")
else:
    st.warning(f"🔄 Aucun modèle entraîné. Cliquez sur le bouton ci-dessous pour commencer.")

# Afficher les détails des modèles existants (R² depuis les métadonnées)
if metadata and (modeles_existants or modeles_incompatibles):
    with st.expander("📋 Détails des modèles existants"):
        meta_data = []
        for nom in puits_disponibles:
            if nom in metadata:
                r2_val = metadata[nom].get('r2', 'N/A')
                n_feat = metadata[nom].get('n_features', 'N/A')
                statut = "✅ Fiable" if isinstance(r2_val, (int, float)) and r2_val >= 0.80 else "❌ Non fiable"
                if nom in modeles_incompatibles:
                    statut = "🔴 Incompatible"
                meta_data.append({
                    'Puits': nom,
                    'R²': f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else r2_val,
                    'Features': n_feat,
                    'Statut': statut
                })
        if meta_data:
            st.dataframe(pd.DataFrame(meta_data), use_container_width=True, hide_index=True)

btn_tous = st.button("🧠 Entraînement", use_container_width=True)

if btn_tous:
    if 'voisins' not in st.session_state:
        st.warning("⚠️ Cliquez d'abord sur 'Analyser les relations'.")
    else:
        if modeles_incompatibles or modeles_manquants:
            # Réentraînement direct si incompatibilité ou modèles manquants
            st.session_state['lancer_entrainement'] = True
        elif len(modeles_existants) == len(puits_disponibles):
            st.session_state['demande_reentrainement'] = True
        else:
            st.session_state['lancer_entrainement'] = True

# Demander confirmation si tous les modèles existent
if st.session_state.get('demande_reentrainement', False):
    st.warning('''**Les modèles seront réentraînés et remplacés**.  
                **Si une modification de données a été effectuée, merci de confirmer le réentraînement.** 
    ''')
    
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        if st.button("🔄 Confirmer le réentraînement", use_container_width=True):
            st.session_state['lancer_entrainement'] = True
            st.session_state['demande_reentrainement'] = False
            st.rerun()
    with col_conf2:
        if st.button("❌ Annuler", use_container_width=True):
            st.session_state['demande_reentrainement'] = False
            st.rerun()

# Lancer l'entraînement
if st.session_state.get('lancer_entrainement', False):
    
    st.session_state['lancer_entrainement'] = False
    
    WINDOW = 30
    
    st.divider()
    st.subheader("🔄 Entraînement en cours")
    
    resultats_tous = []
    progress_global = st.progress(0, text="Préparation...")
    
    for idx_puits, nom_puits in enumerate(puits_disponibles):
        
        progress_global.progress(
            idx_puits / len(puits_disponibles),
            text=f"Entraînement de {nom_puits}... ({idx_puits + 1}/{len(puits_disponibles)})"
        )
        
        feat, voisins_vol = construire_features(nom_puits)
        niv_col = f'Depth_to_Groundwater_{nom_puits}'
        
        data_p = df[feat].copy()
        data_values = data_p.values
        
        split_idx = int(len(data_values) * 0.8)
        train_d = data_values[:split_idx]
        test_d = data_values[split_idx:]
        
        scaler_p = MinMaxScaler(feature_range=(0, 1))
        train_sc = scaler_p.fit_transform(train_d)
        test_sc = scaler_p.transform(test_d)
        
        X_tr, y_tr = create_sequences(train_sc, WINDOW)
        X_te, y_te = create_sequences(test_sc, WINDOW)
        n_f = X_tr.shape[2]
        
        best_r2_p = -999
        best_model_p = None
        best_seed_p = 7
        
        for seed in [7, 42, 123]:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            m = Sequential([
                LSTM(128, input_shape=(WINDOW, n_f)),
                Dropout(0.2),
                Dense(1)
            ])
            m.compile(optimizer='adam', loss='mse')
            
            m.fit(
                X_tr, y_tr,
                epochs=200, batch_size=16,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10,
                                  restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=5, min_lr=0.00001)
                ],
                verbose=0
            )
            
            pred_sc = m.predict(X_te, verbose=0)
            dummy = np.zeros((len(pred_sc), n_f))
            dummy[:, -1] = pred_sc.flatten()
            pred = scaler_p.inverse_transform(dummy)[:, -1]
            
            dummy_t = np.zeros((len(y_te), n_f))
            dummy_t[:, -1] = y_te
            real = scaler_p.inverse_transform(dummy_t)[:, -1]
            
            r2_p = r2_score(real, pred)
            
            if r2_p > best_r2_p:
                best_r2_p = r2_p
                best_model_p = m
                best_seed_p = seed
        
        os.makedirs('../models/prediction', exist_ok=True)
        best_model_p.save(f'../models/prediction/lstm_{nom_puits.lower()}.keras')
        
        if 'scalers' not in st.session_state:
            st.session_state['scalers'] = {}
        st.session_state['scalers'][nom_puits] = scaler_p
        
        if 'feature_names_map' not in st.session_state:
            st.session_state['feature_names_map'] = {}
        st.session_state['feature_names_map'][nom_puits] = feat
        
        # Sauvegarder les métadonnées (R², nombre de features, seed)
        metadata[nom_puits] = {
            'r2': best_r2_p,
            'n_features': n_f,
            'seed': best_seed_p,
            'feature_names': feat
        }
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if best_r2_p >= 0.80:
            statut = "✅ Fiable"
        else:
            statut = "❌ Non fiable"
        
        resultats_tous.append({
            'Puits': nom_puits,
            'R²': f"{best_r2_p:.4f}",
            'Features': n_f,
            'Voisins': len(st.session_state['voisins'].get(nom_puits, [])),
            'Statut': statut
        })
    
    progress_global.progress(1.0, text="Terminé !")
    st.session_state['resultats_entrainement'] = resultats_tous

# Afficher le résumé d'entraînement
if 'resultats_entrainement' in st.session_state:
    
    resultats_tous = st.session_state['resultats_entrainement']
    
    n_fiables = sum(1 for r in resultats_tous if '✅' in r['Statut'])
    n_non_fiables = sum(1 for r in resultats_tous if '❌' in r['Statut'])
    
    with st.expander("📊 Résumé de l'entraînement — Cliquez pour voir les détails"):
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Fiables (R² ≥ 0.80)", n_fiables)
        with col2:
            st.metric("❌ Non fiables (R² < 0.80)", n_non_fiables)
        
        st.markdown(f"""
        **Que signifient ces résultats ?**
        
        L'application a créé un modèle de prédiction pour chaque puits.  
        Chaque modèle a été testé sur des données historiques pour mesurer  
        sa **précision** (R²) :
        
        - **Fiable (R² ≥ 0.80)** → le modèle prédit correctement plus de 80%  
          du comportement du puits. Ses prédictions sont dignes de confiance.
        
        - **Non fiable (R² < 0.80)** → le modèle ne parvient pas à prédire  
          ce puits. Causes possibles : données insuffisantes, événements  
          extrêmes, ou signal trop faible. Ce puits sera exclu de  
          l'optimisation (Module 3) et son pompage sera maintenu à la dernière  
          valeur connue.
        
        **Résumé** : sur {len(resultats_tous)} puits, **{n_fiables} sont fiables**  
        et pourront être utilisés pour la prédiction et l'optimisation.
        """)
        
        with st.expander("📋 Tableau détaillé par puits"):
            res_df = pd.DataFrame(resultats_tous)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
    
    if n_non_fiables > 0:
        st.warning(
            f"⚠️ {n_non_fiables} puits non fiable(s) — ils seront exclus "
            f"de l'optimisation et leur pompage sera maintenu à la dernière valeur connue."
        )

st.divider()

# ============================================================
# SECTION 3 : Prédiction d'un puits
# ============================================================
st.subheader("🎯 Étape 3 - Prédiction d'un puits")

if not puits_disponibles:
    st.error("❌ Aucun puits avec volume de pompage disponible.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    puits_choisi = st.selectbox("Sélectionner un puits :", puits_disponibles)

with col2:
    horizon = st.slider("Horizon de prédiction (jours) :", 
                        min_value=7, max_value=30, value=14, step=7)

# Afficher les voisins du puits choisi
if 'voisins' in st.session_state:
    voisins_choisi = st.session_state['voisins'].get(puits_choisi, [])
    if voisins_choisi:
        st.info(f"🔗 {puits_choisi} sera prédit en tenant compte de ses "
                f"voisins : **{', '.join(voisins_choisi)}**")
    else:
        st.info(f"🔘 {puits_choisi} est isolé — prédit avec ses propres données uniquement")

btn_prediction = st.button("🚀 Prédire ce puits", use_container_width=True)

if btn_prediction:
    
    # Vérifier que le modèle existe
    model_path = f'../models/prediction/lstm_{puits_choisi.lower()}.keras'
    
    if not os.path.exists(model_path):
        st.error(
            f"❌ Aucun modèle trouvé pour {puits_choisi}. "
            f"Cliquez d'abord sur 'Entraîner tous les puits' (Étape 2)."
        )
        st.stop()
    
    if 'voisins' not in st.session_state:
        st.warning("⚠️ Cliquez d'abord sur 'Analyser les relations' (Étape 1).")
        st.stop()
    
    # === Charger le modèle ===
    with st.spinner(f"📂 Chargement du modèle pour {puits_choisi}..."):
        model = load_model(model_path)
    
    # === Vérifier la compatibilité des features ===
    feature_names, voisins_vol_cols = construire_features(puits_choisi)
    n_features_actuelles = len(feature_names)
    
    # Comparer avec les métadonnées sauvegardées
    if puits_choisi in metadata and metadata[puits_choisi].get('n_features') != n_features_actuelles:
        st.error(
            f"🔴 **Modèle incompatible pour {puits_choisi}** — le modèle a été entraîné avec "
            f"{metadata[puits_choisi]['n_features']} features, mais les données actuelles en ont "
            f"{n_features_actuelles}. **Réentraînez les modèles (Étape 2).**"
        )
        st.stop()
    
    st.success(f"✅ Modèle chargé pour {puits_choisi}")
    
    # === Préparer les données ===
    niveau_col = f'Depth_to_Groundwater_{puits_choisi}'
    
    data = df[feature_names].copy()
    WINDOW = 30
    
    data_values = data.values
    split_idx = int(len(data_values) * 0.8)
    train_data = data_values[:split_idx]
    test_data = data_values[split_idx:]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    X_test, y_test = create_sequences(test_scaled, WINDOW)
    n_feat = X_test.shape[2]
    
    # Évaluer le modèle
    pred_sc = model.predict(X_test, verbose=0)
    dummy = np.zeros((len(pred_sc), n_feat))
    dummy[:, -1] = pred_sc.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, -1]
    
    dummy_t = np.zeros((len(y_test), n_feat))
    dummy_t[:, -1] = y_test
    y_real = scaler.inverse_transform(dummy_t)[:, -1]
    
    r2 = r2_score(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    
    # === Résultats du modèle (masqués par défaut) ===
    st.divider()
    
    # Message simple selon le R²
    if r2 >= 0.80:
        st.success(f"✅ Modèle fiable pour {puits_choisi}, les prédictions sont dignes de confiance.")
    else:
        st.error(f"❌ Modèle non fiable pour {puits_choisi} (R² < 0.80), les prédictions sont à prendre avec précaution.")
    
    with st.expander("📊 Détails techniques du modèle"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R²", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f} m")
        with col3:
            st.metric("MAE", f"{mae:.4f} m")
        
        st.markdown(f"""
        - **R² = {r2:.4f}** → le modèle explique **{r2*100:.1f}%** des variations du niveau de ce puits. Plus c'est proche de 1.00, mieux c'est.
        
        - **RMSE = {rmse:.4f} m** → l'erreur moyenne de prédiction est de **{rmse:.2f} mètres**. C'est la marge d'erreur typique.
        
        - **MAE = {mae:.4f} m** → en moyenne, la prédiction se trompe de **{mae:.2f} mètres** (sans tenir compte du sens de l'erreur).
        """)
    
    # === Graphique prédiction vs réalité ===
    st.subheader("📈 Prédiction vs Réalité (données de test)")
    
    dates_test = df['Date'].iloc[
        split_idx + WINDOW:split_idx + WINDOW + len(y_real)
    ].values
    
    erreur = np.abs(y_real - y_pred)
    
    fig = go.Figure()
    
    # Zone d'erreur
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_test, dates_test[::-1]]),
        y=np.concatenate([y_pred + erreur, (y_pred - erreur)[::-1]]),
        fill='toself', fillcolor='rgba(5, 150, 105, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=True, name="Marge d'erreur"
    ))
    
    # Courbe réelle
    fig.add_trace(go.Scatter(
        x=dates_test, y=y_real,
        mode='lines', name='Valeurs réelles',
        line=dict(color='#2563EB', width=2)
    ))
    
    # Courbe prédite
    fig.add_trace(go.Scatter(
        x=dates_test, y=y_pred,
        mode='lines', name='Prédictions LSTM',
        line=dict(color='#059669', width=2, dash='dash')
    ))
    
    # Ligne de la moyenne
    moyenne_reelle = np.mean(y_real)
    fig.add_hline(
        y=moyenne_reelle, line_dash="dot", line_color="gray",
        annotation_text=f"Moyenne : {moyenne_reelle:.1f}m"
    )
    
    fig.update_layout(
        title=f"Prédiction - {puits_choisi} (R² = {r2:.4f})",
        xaxis_title="Date", yaxis_title="Niveau (m)",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # # Statistiques de l'erreur
    # with st.expander("📊 Analyse détaillée de la précision"):
    #     col1, col2, col3, col4 = st.columns(4)
    #     with col1:
    #         st.metric("Erreur moyenne", f"{np.mean(erreur):.3f} m")
    #     with col2:
    #         st.metric("Erreur max", f"{np.max(erreur):.3f} m")
    #     with col3:
    #         st.metric("Erreur min", f"{np.min(erreur):.3f} m")
    #     with col4:
    #         pct_bon = np.sum(erreur < 1.0) / len(erreur) * 100
    #         st.metric("Prédictions < 1m d'erreur", f"{pct_bon:.0f}%")
        
    #     st.markdown(f"""
    #     **Interprétation :**
    #     - En moyenne, le modèle se trompe de **{np.mean(erreur):.2f} mètres**
    #     - L'erreur la plus grande est de **{np.max(erreur):.2f} mètres**
    #     - **{pct_bon:.0f}%** des prédictions ont moins de 1 mètre d'erreur
    #     """)
    
    # === Prédiction future (simulation récursive) ===
    st.divider()
    st.subheader(f"🔮 Prédiction future - {horizon} jours")
    
    st.markdown(
        f"Simulation récursive sur **{horizon} jours** : le LSTM prédit jour "
        f"par jour. Le pompage reste constant (régime actuel). "
        f"Les conditions climatiques suivent les moyennes saisonnières."
    )
    
    with st.spinner(f"Simulation sur {horizon} jours..."):
        
        # Calculer les moyennes mensuelles pour les features CLIMATIQUES
        df_work = df.copy()
        df_work['mois'] = df_work['Date'].dt.month
        dernier_mois = df['Date'].iloc[-1].month
        
        moyennes_mensuelles = {}
        cols_climat = [col for col in feature_names 
                       if col != niveau_col 
                       and 'Volume' not in col]
        
        for col in cols_climat:
            moyennes_mensuelles[col] = df_work.groupby('mois')[col].mean()
        
        derniers_jours = data.values[-30:].copy()
        n_feat_sim = derniers_jours.shape[1]
        
        niveaux_futurs = []
        sequence = derniers_jours.copy()
        
        for jour in range(horizon):
            seq_scaled = scaler.transform(sequence)
            X = seq_scaled.reshape(1, 30, n_feat_sim)
            
            pred_sc = model.predict(X, verbose=0)
            
            dummy = np.zeros((1, n_feat_sim))
            dummy[0, -1] = pred_sc[0, 0]
            niveau_predit = scaler.inverse_transform(dummy)[0, -1]
            
            niveaux_futurs.append(niveau_predit)
            
            # Construire le nouveau jour
            date_future = pd.Timestamp(df['Date'].iloc[-1]) + pd.Timedelta(days=jour + 1)
            mois_futur = date_future.month
            
            nouveau_jour = np.zeros(n_feat_sim)

            for idx_feat, col in enumerate(feature_names):
                if col == niveau_col:
                    # Niveau = prédiction du LSTM
                    nouveau_jour[idx_feat] = niveau_predit
                elif col == 'Mois_sin':
                    nouveau_jour[idx_feat] = np.sin(2 * np.pi * mois_futur / 12)
                elif col == 'Mois_cos':
                    nouveau_jour[idx_feat] = np.cos(2 * np.pi * mois_futur / 12)
                elif col in moyennes_mensuelles:
                    # Climat = moyenne mensuelle historique
                    nouveau_jour[idx_feat] = moyennes_mensuelles[col].get(
                        mois_futur, sequence[-1, idx_feat])
                else:
                    # Pompage = dernière valeur connue (régime actuel)
                    nouveau_jour[idx_feat] = sequence[-1, idx_feat]
            # for idx_feat, col in enumerate(feature_names):
            #     if col == niveau_col:
            #         # Niveau = prédiction du LSTM
            #         nouveau_jour[idx_feat] = niveau_predit
            #     elif col in moyennes_mensuelles:
            #         # Climat = moyenne mensuelle historique
            #         nouveau_jour[idx_feat] = moyennes_mensuelles[col].get(
            #             mois_futur, sequence[-1, idx_feat])
            #     else:
            #         # Pompage = dernière valeur connue (régime actuel)
            #         nouveau_jour[idx_feat] = sequence[-1, idx_feat]
            
            sequence = np.vstack([sequence[1:], nouveau_jour.reshape(1, -1)])
    
    # Graphique futur
    derniere_date_str = df['Date'].iloc[-1].strftime('%Y-%m-%d')
    dates_futures = pd.date_range(
        start=pd.Timestamp(derniere_date_str) + pd.Timedelta(days=1),
        periods=horizon, freq='D'
    )
    
    n_historique = 60
    dates_hist = df['Date'].iloc[-n_historique:].values
    niveaux_hist = data[niveau_col].iloc[-n_historique:].values
    
    fig_futur = go.Figure()
    
    # Zone historique
    fig_futur.add_trace(go.Scatter(
        x=dates_hist, y=niveaux_hist,
        mode='lines', name='Historique (60 derniers jours)',
        line=dict(color='#2563EB', width=2),
        fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.05)'
    ))
    
    # Point de transition
    fig_futur.add_trace(go.Scatter(
        x=[dates_hist[-1]], y=[niveaux_hist[-1]],
        mode='markers', name='Dernier niveau mesuré',
        marker=dict(color='#2563EB', size=10, symbol='circle')
    ))
    
    # Prédiction future
    fig_futur.add_trace(go.Scatter(
        x=dates_futures, y=niveaux_futurs,
        mode='lines+markers', name=f'Prédiction ({horizon} jours)',
        line=dict(color='#059669', width=2, dash='dash'),
        marker=dict(size=4)
    ))
    
    # Ligne de séparation "Aujourd'hui"
    fig_futur.add_shape(
        type="line",
        x0=derniere_date_str, x1=derniere_date_str,
        y0=0, y1=1, yref="paper",
        line=dict(color="gray", width=1, dash="dot")
    )
    fig_futur.add_annotation(
        x=derniere_date_str, y=1, yref="paper",
        text="Aujourd'hui", showarrow=False,
        font=dict(color="gray")
    )
    
    # Seuil critique
    niveau_min = data[niveau_col].min()
    seuil = niveau_min - 2
    fig_futur.add_hline(
        y=seuil, line_dash="dash", line_color="#DC2626",
        annotation_text=f"Seuil critique ({seuil:.0f}m)"
    )
    # Seuil critique (pour test)
    # niveau_min = data[niveau_col].min()
    # seuil = niveau_min
    # fig_futur.add_hline(
    #     y=seuil, line_dash="dash", line_color="#DC2626",
    #     annotation_text=f"Seuil critique ({seuil:.0f}m)"
    # )
    
    # Zone de danger
    fig_futur.add_hrect(
        y0=seuil - 10, y1=seuil,
        fillcolor="rgba(220, 38, 38, 0.05)",
        line_width=0
    )
    
    fig_futur.update_layout(
        title=f"Prédiction future - {puits_choisi} ({horizon} jours)",
        xaxis_title="Date", yaxis_title="Niveau (m)",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        )
    )
    
    st.plotly_chart(fig_futur, use_container_width=True)
    
    # Résumé
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Niveau actuel", f"{niveaux_hist[-1]:.2f}m")
    
    with col2:
        delta = niveaux_futurs[-1] - niveaux_hist[-1]
        st.metric(
            f"Niveau prédit (J+{horizon})", 
            f"{niveaux_futurs[-1]:.2f}m",
            delta=f"{delta:+.2f}m"
        )
    
    with col3:
        niveau_min_futur = min(niveaux_futurs)
        if niveau_min_futur > seuil:
            st.metric(
                "Statut", "✅ Sûr", 
                delta=f"{niveau_min_futur - seuil:.1f}m au-dessus du seuil"
            )
        else:
            st.metric(
                "Statut", "⚠️ Alerte",
                delta=f"{niveau_min_futur - seuil:.1f}m sous le seuil"
            )
    
    with st.expander("ℹ️ Comment lire ce graphique ?"):
        st.markdown(f"""
        - **Zone bleue** : les niveaux réels mesurés (60 derniers jours)
        - **Point bleu** : le dernier niveau mesuré (point de départ de la prédiction)
        - **Ligne verte pointillée** : la prédiction pour les {horizon} prochains jours
        - **Zone rouge en bas** : zone de danger (sous le seuil critique)
        
        **Hypothèses de la prédiction :**
        - Le pompage reste constant au régime actuel (dernière valeur mesurée pour chaque puits)
        - La pluviométrie et la température suivent les moyennes saisonnières historiques (conditions normales pour la saison)
        - Les événements exceptionnels (sécheresses, fortes pluies) ne sont pas pris en compte
        
        Pour voir l'impact d'un changement de pompage, utilisez le Module 3 (Optimisation).
        """)
