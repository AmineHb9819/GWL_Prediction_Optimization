# ============================================================
# Module 1 — Visualisation (Observer)
# ============================================================
# Ce module permet à l'utilisateur de visualiser l'état de sa nappe phréatique à travers :
# - Une carte interactive avec les puits (Folium)
# - Des graphiques temporels par puits
# - Des statistiques résumées
# - La matrice de corrélation entre les puits
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import DivIcon
from streamlit_folium import st_folium
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.auth import guard_page

# === Vérifier que les données sont chargées ===
if 'data' not in st.session_state:
    st.warning("⚠️ Aucune donnée chargée. Retournez à la page d'accueil.")
    st.page_link("main.py", label="Retour à l'accueil", icon="🏠")
    st.stop()

# === Vérifier le rôle (lecteur, gestionnaire, admin) ===
guard_page("visualisation")

df = st.session_state['data']
niveau_cols = st.session_state['niveau_cols']

# === Titre ===
st.title("📊 Module 1 - Visualisation")
st.markdown("**Explorez l'état de votre nappe phréatique à travers les données historiques.**")

st.divider()

# ============================================================
# SECTION 1 : Carte interactive (Folium)
# ============================================================
st.subheader("🗺️ Carte des puits")

# Vérifier si des coordonnées GPS sont déjà configurées
if 'gps_coords' not in st.session_state:
    st.session_state['gps_coords'] = {}

# Demander les coordonnées GPS
with st.expander("⚙️ Configurer les coordonnées GPS des puits"):
    st.markdown("""
    Entrez la **latitude** et la **longitude** de chaque puits pour les afficher sur la carte. 
    Si vous ne connaissez pas les coordonnées exactes, vous pouvez utiliser des valeurs approximatives.
    """)
    
    coords = {}
    default_lat = 41.7
    default_lon = 12.7
    
    for i, col in enumerate(niveau_cols):
        nom = col.replace('Depth_to_Groundwater_', '')
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input(f"Latitude - {nom}", 
                                  value=default_lat + i * 0.005,
                                  format="%.4f", key=f"lat_{nom}")
        with col2:
            lon = st.number_input(f"Longitude - {nom}", 
                                  value=default_lon + i * 0.005,
                                  format="%.4f", key=f"lon_{nom}")
        coords[nom] = {'lat': lat, 'lon': lon}
    
    st.session_state['gps_coords'] = coords

# Afficher la carte si les coordonnées existent
if st.session_state['gps_coords']:
    coordonnees = st.session_state['gps_coords']
    
    # Centre de la carte
    lat_centre = sum(coordonnees[p]['lat'] for p in coordonnees) / len(coordonnees)
    lon_centre = sum(coordonnees[p]['lon'] for p in coordonnees) / len(coordonnees)
    
    carte = folium.Map(location=[lat_centre, lon_centre], zoom_start=13)
    
    # Ajouter les marqueurs pour chaque puits
    for col in niveau_cols:
        nom = col.replace('Depth_to_Groundwater_', '')
        
        if nom not in coordonnees:
            continue
        
        lat = coordonnees[nom]['lat']
        lon = coordonnees[nom]['lon']
        
        # Dernier niveau et statistiques
        dernier_niveau = df[col].dropna().iloc[-1]
        niveau_min = df[col].min()
        niveau_max = df[col].max()
        niveau_moy = df[col].mean()
        
        # Couleur selon le niveau
        amplitude = niveau_max - niveau_min
        if amplitude > 0:
            ratio = (dernier_niveau - niveau_min) / amplitude
        else:
            ratio = 0.5
        
        if ratio > 0.6:
            couleur = '#059669'
            statut = 'Bon'
        elif ratio > 0.3:
            couleur = '#F59E0B'
            statut = 'Moyen'
        else:
            couleur = '#DC2626'
            statut = 'Bas'
        
        # Popup détaillé
        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="margin: 0; color: {couleur};">{nom}</h4>
            <hr style="margin: 5px 0;">
            <b>Dernier niveau :</b> {dernier_niveau:.2f}m<br>
            <b>Minimum :</b> {niveau_min:.2f}m<br>
            <b>Maximum :</b> {niveau_max:.2f}m<br>
            <b>Moyenne :</b> {niveau_moy:.2f}m<br>
            <b>Statut :</b> <span style="color: {couleur}; font-weight: bold;">{statut}</span>
        </div>
        """
        
        # Cercle coloré
        folium.CircleMarker(
            location=[lat, lon],
            radius=15,
            color=couleur,
            fill=True,
            fillColor=couleur,
            fillOpacity=0.7,
            weight=3,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{nom} : {dernier_niveau:.2f}m ({statut})"
        ).add_to(carte)
        
        # Nom du puits comme label permanent
        folium.Marker(
            location=[lat, lon],
            icon=DivIcon(
                html=f"""
                <div style="
                    font-size: 11px; 
                    font-weight: bold; 
                    color: #1F2937;
                    background-color: white;
                    border: 1px solid {couleur};
                    border-radius: 3px;
                    padding: 2px 5px;
                    white-space: nowrap;
                    transform: translate(-50%, -35px);
                ">{nom}</div>
                """,
                icon_size=(0, 0),
                icon_anchor=(0, 0)
            )
        ).add_to(carte)
    
    # Afficher la carte
    st_folium(carte, width=None, height=500)
    
    with st.expander("ℹ️ Comment lire la carte ?"):
        st.markdown("""
        **Les cercles** représentent les puits :
        - 🟢 **Vert** : le niveau actuel est dans la partie haute de sa plage historique (bon état)
        - 🟡 **Orange** : le niveau est dans la partie médiane (à surveiller)
        - 🔴 **Rouge** : le niveau est dans la partie basse de sa plage historique (état critique)
        
        **Cliquez** sur un cercle pour voir les détails du puits (dernier niveau, min, max, moyenne).
        
        **Survolez** un cercle pour voir un résumé rapide.
        """)

st.divider()

# ============================================================
# SECTION 2 : Graphique temporel par puits
# ============================================================
st.subheader("📈 Évolution temporelle")

noms_puits = [col.replace('Depth_to_Groundwater_', '') for col in niveau_cols]
puits_choisi = st.selectbox("Sélectionner un puits :", noms_puits)
col_choisi = f'Depth_to_Groundwater_{puits_choisi}'

fig_temp = go.Figure()

fig_temp.add_trace(go.Scatter(
    x=df['Date'], y=df[col_choisi],
    mode='lines', name=puits_choisi,
    line=dict(color='#2563EB', width=1)
))

moyenne = df[col_choisi].mean()
fig_temp.add_hline(y=moyenne, line_dash="dash", line_color="gray",
                   annotation_text=f"Moyenne : {moyenne:.2f}m")

fig_temp.update_layout(
    title=f"Niveau piézométrique - {puits_choisi}",
    xaxis_title="Date",
    yaxis_title="Niveau (m)",
    height=450
)

st.plotly_chart(fig_temp, use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Minimum", f"{df[col_choisi].min():.2f}m")
with col2:
    st.metric("Maximum", f"{df[col_choisi].max():.2f}m")
with col3:
    st.metric("Moyenne", f"{df[col_choisi].mean():.2f}m")
with col4:
    st.metric("Dernier niveau", f"{df[col_choisi].dropna().iloc[-1]:.2f}m")

st.divider()

# ============================================================
# SECTION 3 : Comparaison de tous les puits
# ============================================================
st.subheader("📊 Comparaison de tous les puits")

fig_all = go.Figure()

for col in niveau_cols:
    nom = col.replace('Depth_to_Groundwater_', '')
    fig_all.add_trace(go.Scatter(
        x=df['Date'], y=df[col],
        mode='lines', name=nom,
        line=dict(width=1)
    ))

fig_all.update_layout(
    title="Évolution de tous les puits",
    xaxis_title="Date",
    yaxis_title="Niveau (m)",
    height=500
)

st.plotly_chart(fig_all, use_container_width=True)

st.divider()

# ============================================================
# SECTION 4 : Matrice de corrélation
# ============================================================
st.subheader("🔗 Matrice de corrélation entre les puits")

st.markdown("""
#### Qu'est-ce que la corrélation ?

La corrélation mesure si deux puits **réagissent de la même manière** au fil du temps.  
C'est un nombre entre **-1** et **+1** :

| Valeur | Signification | Exemple concret |
|--------|--------------|-----------------|
| **+1** (rouge foncé) | Les deux puits montent et descendent **ensemble** | Ils partagent probablement la même zone de la nappe |
| **0** (blanc) | Les deux puits n'ont **aucun lien** | Ils sont indépendants, dans des zones différentes |
| **-1** (bleu foncé) | Quand l'un monte, l'autre **descend** | Effet de vases communicants : pomper l'un remplit l'autre |

#### Comment lire la matrice ?

Chaque case montre la corrélation entre deux puits.  
La diagonale est toujours **1.00** (un puits est parfaitement corrélé avec lui-même).

**Ce qu'il faut surveiller :**
- Les cases **rouge foncé** (> 0.7) → ces puits sont fortement liés. Si on pompe trop de l'un, l'autre sera aussi affecté.
- Les cases **bleu foncé** (< -0.7) → ces puits sont en opposition. Pomper l'un peut faire remonter l'autre.
- Les cases **blanches** (~0) → ces puits sont indépendants. On peut les gérer séparément.
""")

corr = df[niveau_cols].corr().round(3)
corr.columns = [col.replace('Depth_to_Groundwater_', '') for col in corr.columns]
corr.index = [col.replace('Depth_to_Groundwater_', '') for col in corr.index]

fig_corr = px.imshow(
    corr, text_auto='.2f',
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    height=500
)
fig_corr.update_layout(title="Corrélation entre les niveaux piézométriques")

st.plotly_chart(fig_corr, use_container_width=True)

# Résumé automatique des corrélations fortes
st.markdown("#### 🔍 Relations détectées automatiquement")

relations_fortes = []
relations_negatives = []
n = len(corr)

for i in range(n):
    for j in range(i + 1, n):
        val = corr.iloc[i, j]
        p1 = corr.index[i]
        p2 = corr.columns[j]
        if val > 0.7:
            relations_fortes.append(f"**{p1}** et **{p2}** : corrélation de {val:.2f} → ces puits sont fortement liés")
        elif val < -0.7:
            relations_negatives.append(f"**{p1}** et **{p2}** : corrélation de {val:.2f} → effet de vases communicants")

if relations_fortes:
    st.markdown("**🔴 Puits fortement liés (corrélation > 0.7) :**")
    for r in relations_fortes:
        st.success(r)

if relations_negatives:
    st.markdown("**🔵 Puits en opposition (corrélation < -0.7) :**")
    for r in relations_negatives:
        st.info(r)

if not relations_fortes and not relations_negatives:
    st.info("Aucune corrélation forte détectée entre les puits. Ils semblent relativement indépendants.")
# ============================================================
# SECTION 5 : Statistiques résumées
# ============================================================
st.subheader("📋 Statistiques résumées")

stats_data = []
for col in niveau_cols:
    nom = col.replace('Depth_to_Groundwater_', '')
    
    # Volatilité (écart-type des variations journalières)
    variations = df[col].diff().dropna()
    volatilite = variations.std()
    
    stats_data.append({
        'Puits': nom,
        'Min (m)': f"{df[col].min():.2f}",
        'Max (m)': f"{df[col].max():.2f}",
        'Amplitude (m)': f"{df[col].max() - df[col].min():.2f}",
        'Moyenne (m)': f"{df[col].mean():.2f}",
        'Écart-type (m)': f"{df[col].std():.2f}",
        'Volatilité (m/j)': f"{volatilite:.4f}",
        'Dernier (m)': f"{df[col].dropna().iloc[-1]:.2f}"
    })

stats_df = pd.DataFrame(stats_data)
st.dataframe(stats_df, use_container_width=True, hide_index=True)

with st.expander("ℹ️ Que signifient ces statistiques ?"):
    st.markdown("""
    - **Amplitude** : la différence entre le niveau le plus haut et le plus bas jamais mesurés. Une grande
     amplitude signifie que le puits est sensible aux variations saisonnières ou au pompage.
    
    - **Écart-type** : mesure la dispersion des niveaux autour de la moyenne.  
      Un écart-type élevé indique un puits avec des variations importantes.
    
    - **Volatilité** : l'écart-type des variations journalières. Mesure la "nervosité" du puits — un puits volatile change beaucoup d'un jour à l'autre, un puits stable varie peu.
      
    **Interprétation pour la gestion :**
    - Un puits avec une **grande amplitude** et une **forte volatilité**, il nécessite une surveillance plus fréquente.
    - Un puits avec une **faible amplitude** est plus stable mais peut être plus difficile à modéliser (signal faible).
    """)
