# ============================================================
# Page d'accueil — Application de gestion des nappes phréatiques
# ============================================================
# C'est le point d'entrée de l'application Streamlit.
# L'utilisateur arrive ici, uploade son CSV, et navigue
# vers les 3 modules (Visualisation, Prédiction, Optimisation).
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(__file__))

from utils.auth import (
    initialiser_authentification,
    get_role_utilisateur,
    afficher_panneau_admin,
)

# === Configuration de la page ===
st.set_page_config(
    page_title="Gestion des nappes phréatiques",
    page_icon="🌊",
    layout="wide"
)

# === Authentification ===
authenticator, config = initialiser_authentification()

# Afficher le formulaire de login dans un conteneur effaçable
login_placeholder = st.empty()

with login_placeholder.container():
    try:
        authenticator.login()
    except Exception:
        # Fallback pour différentes versions de streamlit-authenticator
        pass

# Vérifier le statut d'authentification
auth_status = st.session_state.get("authentication_status", None)

if auth_status is False:
    st.error("❌ Identifiant ou mot de passe incorrect.")
    st.stop()

if auth_status is None:
    st.info("👋 Veuillez vous connecter pour accéder à l'application.")
    st.stop()

# Utilisateur connecté — masquer le formulaire de login
login_placeholder.empty()

# === Utilisateur connecté — récupérer le rôle ===
role = get_role_utilisateur(config)
st.session_state["role"] = role

# === Barre latérale : info utilisateur + déconnexion ===
with st.sidebar:
    st.markdown(f"👤 **{st.session_state.get('name', '')}**")
    st.markdown(f"🔑 Rôle : `{role}`")
    authenticator.logout("Se déconnecter", "sidebar")
    st.divider()
    # Panneau admin (visible uniquement pour les admins)
    afficher_panneau_admin(config)

# === Titre et description ===
st.title("🌊 Gestion intelligente des nappes phréatiques")
st.markdown("""
Cette application utilise l'intelligence artificielle pour :
- Observer l'état de votre nappe phréatique (Module 1)
- Prédire le niveau futur de l'eau (Module 2)
- Optimiser la répartition du pompage entre les puits (Module 3)
""")

st.divider()

# === Guide de préparation des données ===
with st.expander("📋 Guide de préparation des données — Cliquez pour ouvrir"):
    st.markdown("""
    ### 📁 Format du fichier
    
    - Format : CSV (séparé par des virgules ou des points-virgules)
    - Encodage : UTF-8 recommandé
    - Fréquence : mesures journalières (une ligne = un jour)
    - Durée minimale : 3 ans de données (1095 lignes minimum)
    - Durée recommandée : 5 ans ou plus pour des prédictions fiables
    
    > Le LSTM a besoin d'observer plusieurs cycles saisonniers complets pour comprendre le comportement de la nappe.  
    > Avec moins de 3 ans, le modèle n'a vu que 2-3 hivers et 2-3 étés, c'est insuffisant pour distinguer les tendances des anomalies.
    
    ---
    
    ### 📋 Colonnes obligatoires
    
    | Type | Préfixe obligatoire | Exemple | Description |
    |------|---------------------|---------|-------------|
    | 📅 Date | `Date` | `Date` | Format JJ/MM/AAAA ou AAAA-MM-JJ |
    | 💧 Niveau | `Depth_to_Groundwater_` | `Depth_to_Groundwater_Puits1` | Niveau piézométrique (en mètres) |
    | 🔧 Pompage | `Volume_` | `Volume_Puits1` | Volume pompé par jour (en m³/jour) |
    | 🌧 Pluie | `Rainfall_` | `Rainfall_Station1` | Pluviométrie (en mm/jour) |
    | 🌡 Température | `Temperature_` | `Temperature_Station1` | Température (en °C) |
    
    ---
    
    ### 📊 Colonnes supplémentaires (améliorent la précision)
    
    | Type | Préfixe | Exemple |
    |------|---------|---------|
    | 💨 Hydrométrie | `Hydrometry_` | `Hydrometry_Riviere1` |
    | 💦 Évapotranspiration | `Evapotranspiration_` | `Evapotranspiration_Station1` |
    | Autre | Nom libre | `Humidity_Station1` |
    
    ---
    
    ### ✅ Bonnes pratiques pour des données propres
    
    Nommage des colonnes :
    - Chaque puits doit avoir deux colonnes correspondantes :  
      `Depth_to_Groundwater_MonPuits` et `Volume_MonPuits`
    - Le nom après le préfixe doit être identique dans les deux colonnes
    - ✅ Correct : `Depth_to_Groundwater_Nord` + `Volume_Nord`
    - ❌ Incorrect : `Depth_to_Groundwater_Nord` + `Volume_Puits_Nord`
    
    Valeurs manquantes :
    - Les trous isolés (quelques jours) seront comblés automatiquement par interpolation linéaire
    - Les périodes longues sans données seront coupées automatiquement
    - Les colonnes avec plus de 50% de valeurs manquantes seront exclues
    - Recommandation : fournissez les données les plus complètes possibles
    
    Valeurs aberrantes :
    - Ne supprimez pas les valeurs extrêmes (sécheresses, fortes pluies)  
      - ce sont des événements réels que le modèle doit apprendre
    - Supprimez uniquement les erreurs de capteur évidentes (ex : un niveau de -999m ou une pluie de 5000mm/jour)
    
    Unités :
    - Gardez les mêmes unités sur toute la période
    - Ne mélangez pas mètres et centimètres, ou litres et m³
    - Les niveaux piézométriques sont généralement négatifs (profondeur sous la surface)
    
    Dates :
    - Une seule ligne par jour, pas de doublons
    - Pas de trous dans la série (tous les jours doivent être présents)
    - Si des jours manquent, l'application les détectera et les interpolera
    
    ---
    
    ### 📐 Exemple de fichier CSV correct
                Date,Depth_to_Groundwater_Nord,Depth_to_Groundwater_Sud,Volume_Nord,Volume_Sud,Rainfall_Meteo1,Temperature_Meteo1
2020-01-01,-25.3,-30.1,1500,2000,3.2,8.5
2020-01-02,-25.4,-30.0,1600,2100,0.0,9.1
2020-01-03,-25.3,-29.9,1550,1900,1.5,7.8
    ---
    
    ### ⚠️ Ce que l'application ne fait PAS
    
    - Elle ne corrige pas les erreurs de nommage des colonnes
    - Elle ne convertit pas les unités
    - Elle ne détecte pas les erreurs de capteur (valeurs physiquement impossibles)
    - Elle ne gère pas les fichiers Excel (.xlsx)  
      - convertissez en CSV d'abord
    """)
def nettoyer_donnees(df_brut):
    """
    Nettoyage automatique — même logique que le notebook 05.
    
    Étapes :
    1. Trouver la date où les colonnes de volume commencent
    2. Couper à cette date
    3. Exclure les colonnes avec < 50% de données
    4. Interpoler les NaN isolés
    5. Supprimer les lignes restantes avec NaN
    6. Ajouter Mois_sin et Mois_cos
    """
    
    rapport = {
        'lignes_avant': len(df_brut),
        'colonnes_avant': len(df_brut.columns),
        'colonnes_exclues': [],
        'nan_avant': 0,
        'nan_interpoles': 0,
        'lignes_supprimees': 0,
        'periode_coupee': None,
        'messages': []
    }
    
    df = df_brut.copy()
    
    # Colonnes numériques
    cols_numeriques = [col for col in df.columns 
                       if col != 'Date' and df[col].dtype in ['float64', 'int64']]
    
    rapport['nan_avant'] = df[cols_numeriques].isnull().sum().sum()
    
    # === ÉTAPE 1 : Trouver où les données commencent vraiment ===
    # Pour chaque colonne, trouver la première valeur non-NaN
    # La coupure = la date la plus tardive parmi les colonnes importantes
    
    # Identifier les colonnes importantes (niveaux + volumes + pluie)
    cols_niveau = [c for c in cols_numeriques if 'Depth' in c or 'Level' in c]
    cols_volume = [c for c in cols_numeriques if 'Volume' in c or 'Pumping' in c]
    cols_pluie = [c for c in cols_numeriques if 'Rain' in c or 'Precip' in c]
    
    # Colonnes qui déterminent la coupure (niveaux + volumes + pluie)
    # On ne prend PAS la température car elle peut être partielle
    cols_coupure = cols_niveau + cols_volume + cols_pluie
    
    if not cols_coupure:
        cols_coupure = cols_numeriques
    
    # Pour chaque colonne importante, trouver le premier index non-NaN
    idx_debut = 0
    
    for col in cols_coupure:
        premier_valide = df[col].first_valid_index()
        if premier_valide is not None and premier_valide > idx_debut:
            idx_debut = premier_valide
    
    # Couper
    if idx_debut > 0:
        date_coupure = df['Date'].iloc[idx_debut]
        lignes_coupees = idx_debut
        df = df.iloc[idx_debut:].reset_index(drop=True)
        rapport['periode_coupee'] = {
            'date': str(date_coupure.date()),
            'lignes': lignes_coupees
        }
        rapport['messages'].append(
            f"📅 Période avant {date_coupure.date()} coupée "
            f"({lignes_coupees} lignes) — données incomplètes"
        )
    
    # === ÉTAPE 2 : Exclure les colonnes avec < 50% de données ===
    colonnes_a_garder = ['Date']
    
    # Recalculer après la coupure
    cols_numeriques = [col for col in df.columns 
                       if col != 'Date' and df[col].dtype in ['float64', 'int64']]
    
    for col in cols_numeriques:
        pct_valid = df[col].notna().sum() / len(df) * 100
        if pct_valid >= 50:
            colonnes_a_garder.append(col)
        else:
            rapport['colonnes_exclues'].append({
                'nom': col,
                'pct_valid': round(pct_valid, 1)
            })
            rapport['messages'].append(
                f"🗑️ Colonne `{col}` exclue ({pct_valid:.1f}% de données valides)"
            )
    
    df = df[colonnes_a_garder]
    
    # === ÉTAPE 3 : Interpolation linéaire ===
    cols_a_interpoler = [col for col in df.columns if col != 'Date']
    nan_avant_interp = df[cols_a_interpoler].isnull().sum().sum()
    
    df[cols_a_interpoler] = df[cols_a_interpoler].interpolate(method='linear')
    
    nan_apres_interp = df[cols_a_interpoler].isnull().sum().sum()
    rapport['nan_interpoles'] = nan_avant_interp - nan_apres_interp
    
    # === ÉTAPE 4 : Supprimer les lignes restantes avec NaN ===
    lignes_avant_drop = len(df)
    df = df.dropna()
    rapport['lignes_supprimees'] = lignes_avant_drop - len(df)
    
    if rapport['lignes_supprimees'] > 0:
        rapport['messages'].append(
            f"🗑️ {rapport['lignes_supprimees']} lignes supprimées "
            f"(NaN non récupérables en début/fin de série)"
        )
    
    # === ÉTAPE 5 : Ajouter Mois_sin et Mois_cos ===
    df['Mois_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['Mois_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
    
    # === Résumé ===
    rapport['lignes_apres'] = len(df)
    rapport['colonnes_apres'] = len(df.columns)
    rapport['nan_apres'] = df.drop(columns=['Date']).isnull().sum().sum()
    
    return df, rapport

# === Upload du fichier CSV ===
st.subheader("📁 Charger vos données")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV",
        type=['csv'],
        help="Fichier CSV avec les mesures journalières de vos puits"
    )

with col2:
    st.markdown("**Ou utiliser les données de démonstration :**")
    demo_button = st.button("🔬 Charger les données Doganella (démo)", 
                            use_container_width=True)

# === Traitement du fichier ===
df_brut = None

if demo_button:
    try:
        df_brut = pd.read_csv('../data/processed/Aquifer_Doganella_clean.csv')
        df_brut['Date'] = pd.to_datetime(df_brut['Date'])
        
        # Séparer Volume_Pozzo_5+6
        if 'Volume_Pozzo_5+6' in df_brut.columns:
            df_brut['Volume_Pozzo_5'] = df_brut['Volume_Pozzo_5+6']
            df_brut['Volume_Pozzo_6'] = df_brut['Volume_Pozzo_5+6']
            df_brut = df_brut.drop(columns=['Volume_Pozzo_5+6'])
        
        st.session_state['source'] = 'demo'
    except FileNotFoundError:
        st.error("❌ Fichier de démonstration non trouvé.")

if uploaded_file is not None:
    try:
        df_brut = pd.read_csv(uploaded_file)
        for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']:
            try:
                df_brut['Date'] = pd.to_datetime(df_brut['Date'], format=fmt)
                break
            except:
                continue
        st.session_state['source'] = 'upload'
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement : {e}")

# === Nettoyage automatique ===
if df_brut is not None:
    
    with st.spinner("🧹 Nettoyage automatique des données en cours..."):
        df_propre, rapport = nettoyer_donnees(df_brut)
    
    st.session_state['data'] = df_propre
    st.session_state['rapport_nettoyage'] = rapport
    
    # Nettoyer les données de l'ancien dataset (graphe, modèles, résultats)
    for key in ['graphe', 'voisins', 'resultats_entrainement', 'scalers', 
                'feature_names_map', 'resultat_optimisation', 'r2_scores']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("✅ Données chargées et nettoyées")

# === Affichage du résumé si des données sont chargées ===
# === Affichage du résumé si des données sont chargées ===
if 'data' in st.session_state:
    df = st.session_state['data']
    
    st.divider()
    st.subheader("📊 Résumé des données détectées")
    
    # Détecter les types de colonnes
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    niveau_cols = [col for col in df.columns if 'Depth' in col or 'Level' in col]
    volume_cols = [col for col in df.columns if 'Volume' in col or 'Pumping' in col]
    pluie_cols = [col for col in df.columns if 'Rain' in col or 'Precip' in col]
    temp_cols = [col for col in df.columns if 'Temp' in col]
    
    # Colonnes restantes = features supplémentaires
    colonnes_connues = date_cols + niveau_cols + volume_cols + pluie_cols + temp_cols
    autres_cols = [col for col in df.columns if col not in colonnes_connues 
                   and df[col].dtype in ['float64', 'int64']]
    
    # Stocker la détection
    st.session_state['niveau_cols'] = niveau_cols
    st.session_state['volume_cols'] = volume_cols
    st.session_state['pluie_cols'] = pluie_cols
    st.session_state['temp_cols'] = temp_cols
    st.session_state['autres_cols'] = autres_cols
    
    # Affichage en colonnes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📅 Période", 
                  f"{df['Date'].min().date()} → {df['Date'].max().date()}")
        st.metric("📏 Lignes", f"{len(df):,}")
    
    with col2:
        st.metric("💧 Puits détectés", len(niveau_cols))
        st.metric("🔧 Volumes de pompage", len(volume_cols))
    
    with col3:
        st.metric("🌧 Stations de pluie", len(pluie_cols))
        nan_total = df.isnull().sum().sum()
        st.metric("⚠️ Valeurs manquantes", f"{nan_total:,}")
    
    # === Rapport de nettoyage ===
    if 'rapport_nettoyage' in st.session_state:
        rapport = st.session_state['rapport_nettoyage']
        
        with st.expander("🧹 Rapport de nettoyage — Cliquez pour voir les détails"):
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Lignes avant", f"{rapport['lignes_avant']:,}")
            with col2:
                st.metric("Lignes après", f"{rapport['lignes_apres']:,}")
            with col3:
                st.metric("NaN traitées", f"{rapport['nan_interpoles']}")
            with col4:
                st.metric("NaN restantes", f"{rapport['nan_apres']}")
            
            if rapport['periode_coupee']:
                st.warning(
                    f"📅 Période coupée : les données avant "
                    f"**{rapport['periode_coupee']['date']}** ont été exclues "
                    f"({rapport['periode_coupee']['lignes']} lignes) car elles "
                    f"contenaient trop de valeurs manquantes."
                )
            
            if rapport['colonnes_exclues']:
                st.warning("🗑️ Colonnes exclues (< 50% de données valides) :")
                for col_info in rapport['colonnes_exclues']:
                    st.markdown(f"  - `{col_info['nom']}` — {col_info['pct_valid']}% de données valides")
            
            if rapport['nan_interpoles'] > 0:
                st.info(
                    f"🔧 **{rapport['nan_interpoles']} valeurs manquantes** "
                    f"ont été comblées par interpolation linéaire "
                    f"(estimation entre les deux valeurs connues les plus proches)."
                )
            
            if rapport['lignes_supprimees'] > 0:
                st.info(
                    f"🗑️ **{rapport['lignes_supprimees']} lignes** supprimées "
                    f"(NaN en début ou fin de série, non récupérables par interpolation)."
                )
            
            if rapport['nan_apres'] == 0:
                st.success("✅ Données finales : **0 valeur manquante** — prêtes pour l'analyse")

   # Détail des colonnes détectées
    with st.expander("🔍 Détail des colonnes détectées"):
        
        st.markdown("#### 💧 Niveaux piézométriques (targets)")
        for col in niveau_cols:
            nom = col.replace('Depth_to_Groundwater_', '')
            dernier = df[col].dropna().iloc[-1] if df[col].notna().any() else "N/A"
            moy = df[col].mean()
            vol_correspondant = [v for v in volume_cols if nom in v]
            
            if vol_correspondant:
                statut_vol = f"✅ Volume associé : `{vol_correspondant[0]}`"
            else:
                statut_vol = "⚠️ Aucun volume de pompage associé"
            
            st.markdown(
                f"- **{nom}** — Dernier niveau : {dernier:.2f}m | "
                f"Moyenne : {moy:.2f}m | {statut_vol}"
            )
        
        st.markdown("#### 🔧 Volumes de pompage")
        for col in volume_cols:
            moy = df[col].mean()
            st.markdown(f"- `{col}` — Moyenne : {moy:.0f} m³/jour")
        
        st.markdown("#### 🌧 Pluviométrie")
        for col in pluie_cols:
            moy = df[col].mean()
            st.markdown(f"- `{col}` — Moyenne : {moy:.2f} mm/jour")
        
        if temp_cols:
            st.markdown("#### 🌡 Température")
            for col in temp_cols:
                moy = df[col].mean()
                st.markdown(f"- `{col}` — Moyenne : {moy:.1f}°C")
        
        if autres_cols:
            st.markdown("#### 📊 Features supplémentaires")
            for col in autres_cols:
                moy = df[col].mean()
                st.markdown(f"- `{col}` — Moyenne : {moy:.2f}")
        
        # Résumé
        st.markdown("---")
        st.markdown(
            f"**Total** : {len(niveau_cols)} puits, "
            f"{len(volume_cols)} volumes, "
            f"{len(pluie_cols)} stations pluie, "
            f"{len(temp_cols)} stations température, "
            f"{len(autres_cols)} features supplémentaires"
        )
                
    # === Télécharger les données nettoyées ===
    if 'rapport_nettoyage' in st.session_state:
        csv_propre = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données nettoyées (CSV)",
            data=csv_propre,
            file_name="donnees_nettoyees.csv",
            mime="text/csv"
        )
    
    # === REMARQUES ET RECOMMANDATIONS ===
    # === REMARQUES ET RECOMMANDATIONS ===
    st.divider()
    st.subheader("🔎 Analyse et recommandations")
    
    remarques = []
    avertissements = []
    erreurs = []
    
    # --- Vérification de la durée des données ---
    n_jours = (df['Date'].max() - df['Date'].min()).days
    n_annees = n_jours / 365.25
    if n_annees >= 5:
        remarques.append(
            f"✅ **Durée excellente** : {n_annees:.1f} ans de données ({n_jours} jours). "
            f"Le LSTM aura suffisamment de cycles saisonniers pour apprendre."
        )
    elif n_annees >= 3:
        remarques.append(
            f"✅ **Durée suffisante** : {n_annees:.1f} ans de données ({n_jours} jours). "
            f"Minimum atteint. Pour de meilleurs résultats, fournissez 5 ans ou plus."
        )
    elif n_annees >= 2:
        avertissements.append(
            f"⚠️ **Durée limite** : {n_annees:.1f} ans de données ({n_jours} jours). "
            f"Le minimum recommandé est 3 ans. Les prédictions risquent "
            f"d'être moins fiables car le LSTM n'a observé que "
            f"{int(n_annees)} cycles saisonniers complets."
        )
    else:
        erreurs.append(
            f"❌ **Durée insuffisante** : seulement {n_annees:.1f} ans de données "
            f"({n_jours} jours). Le minimum requis est 3 ans. "
            f"Avec moins de 2 ans, le LSTM ne peut pas apprendre les patterns "
            f"saisonniers de la nappe. Fournissez un historique plus long."
        )
    
    # --- Vérification du nombre de lignes après nettoyage ---
    if len(df) < 1000:
        avertissements.append(
            f"⚠️ **Nombre de lignes limité** : {len(df)} lignes après nettoyage. "
            f"Avec la fenêtre de 30 jours et le split 80/20, le LSTM aura "
            f"environ {int(len(df) * 0.8) - 30} séquences d'entraînement. "
            f"Pour de meilleurs résultats, fournissez plus de données."
        )
    
    # --- Vérification des colonnes obligatoires ---
    if len(niveau_cols) == 0:
        erreurs.append(
            "❌ **Aucun puits détecté** : aucune colonne `Depth_to_Groundwater_*` trouvée. "
            "Vérifiez le nommage de vos colonnes (respectez le préfixe exact)."
        )
    else:
        remarques.append(f"✅ **{len(niveau_cols)} puits détectés** : prêts pour l'analyse.")
    
    if len(volume_cols) == 0:
        erreurs.append(
            "❌ **Aucun volume de pompage détecté** : aucune colonne `Volume_*` trouvée. "
            "Les Modules 2 (prédiction) et 3 (optimisation) ne pourront pas fonctionner."
        )
    else:
        remarques.append(f"✅ **{len(volume_cols)} volumes de pompage** détectés.")
    
    if len(pluie_cols) == 0:
        erreurs.append(
            "❌ **Aucune donnée de pluie détectée** : aucune colonne `Rainfall_*` trouvée. "
            "La pluviométrie est essentielle pour la prédiction du niveau de la nappe."
        )
    else:
        remarques.append(f"✅ **{len(pluie_cols)} station(s) de pluie** détectée(s).")
    
    # --- Vérification de la température ---
    if len(temp_cols) == 0:
        avertissements.append(
            "⚠️ **Aucune donnée de température détectée**. "
            "L'application fonctionne sans, mais la température influence "
            "l'évaporation et donc le niveau de la nappe. "
            "Ajoutez une colonne `Temperature_*` pour améliorer la précision."
        )
    else:
        remarques.append(f"✅ **{len(temp_cols)} station(s) de température** détectée(s).")
    
    # --- Vérification de la correspondance puits / volumes ---
    puits_sans_volume = []
    for col in niveau_cols:
        nom_puits = col.replace('Depth_to_Groundwater_', '')
        volume_correspondant = [v for v in volume_cols if nom_puits in v]
        if len(volume_correspondant) == 0:
            puits_sans_volume.append(nom_puits)
    
    if puits_sans_volume:
        for nom in puits_sans_volume:
            avertissements.append(
                f"⚠️ **{nom}** : aucun volume de pompage correspondant trouvé. "
                f"Vérifiez que la colonne de volume contient le nom du puits "
                f"(exemple : `Volume_{nom}`). "
                f"Sans volume, ce puits sera utilisable uniquement pour la "
                f"visualisation (Module 1)."
            )
    
    # --- Vérification des features supplémentaires ---
    if autres_cols:
        remarques.append(
            f"✅ **{len(autres_cols)} feature(s) supplémentaire(s)** détectée(s) : "
            f"{', '.join(autres_cols)}. Elles seront incluses automatiquement "
            f"dans les modèles de prédiction."
        )
    
    # --- Affichage des remarques ---
    if erreurs:
        st.markdown("### ❌ Erreurs à corriger")
        for e in erreurs:
            st.error(e)
    
    if avertissements:
        st.markdown("### ⚠️ Avertissements")
        for a in avertissements:
            st.warning(a)
    
    if remarques:
        st.markdown("### ✅ Points validés")
        for r in remarques:
            st.success(r)
    
    # --- Recommandation d'étape suivante ---
    st.divider()
    st.subheader("📌 Prochaine étape recommandée")
    
    if erreurs:
        st.error("""
        **Corrigez les erreurs ci-dessus avant de continuer.**  
        Vérifiez le nommage de vos colonnes et rechargez le fichier.
        """)
    elif len(niveau_cols) == 1:
        st.info("""
        **Un seul puits détecté** → vous pouvez utiliser :
        - **Module 1** pour visualiser les données
        - **Module 2** pour prédire le niveau futur de ce puits
        - Le **Module 3** (optimisation) nécessite au moins 2 puits
        """)
    else:
        if avertissements:
            st.warning(
                "**Des avertissements ont été détectés** (voir ci-dessus). "
                "L'application fonctionnera mais les résultats seront "
                "potentiellement moins précis pour les colonnes concernées."
            )
        st.success(f"""
        **Vos données sont prêtes !** {len(niveau_cols)} puits détectés.
        
        📊 **Module 1 — Observer** : commencez par visualiser vos données  
        pour comprendre le comportement de votre nappe.
        
        🔮 **Module 2 — Prédire** : sélectionnez un puits pour prédire  
        son niveau futur.
        
        ⚡ **Module 3 — Optimiser** : trouvez le plan de pompage optimal  
        pour tous vos puits.
        """)    
    # Navigation
    st.divider()
    st.subheader("🚀 Choisir un module")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Module 1 — Observer
        Carte interactive, graphiques temporels, statistiques par puits.""")
        st.page_link("pages/1_visualisation.py", label="Ouvrir le Module 1",
                     icon="📊", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### 🔮 Module 2 — Prédire
        Prédiction du niveau futur d'un puits sélectionné.
        """)
        if role in ["gestionnaire", "admin"]:
            st.page_link("pages/2_prediction.py", label="Ouvrir le Module 2",
                         icon="🔮", use_container_width=True)
        else:
            st.info("🔒 Accès réservé aux gestionnaires et admins.")
    
    with col3:
        st.markdown("""
        ### ⚡ Module 3 — Décider
        Plan de pompage optimisé  
        pour tous les puits
        """)
        if role in ["gestionnaire", "admin"]:
            st.page_link("pages/3_optimisation.py", label="Ouvrir le Module 3",
                         icon="⚡", use_container_width=True)
        else:
            st.info("🔒 Accès réservé aux gestionnaires et admins.")

else:
    st.info("👆 Chargez un fichier CSV ou utilisez les données de démonstration pour commencer.")