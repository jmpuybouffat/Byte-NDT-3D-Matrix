import streamlit as st
import numpy as np
import plotly.graph_objects as go
from stl import mesh
import math

st.set_page_config(layout="wide", page_title="Byte NDT - Intrados")
st.title("Byte NDT - Digital Twin LSB 941 (Version Intégrale) - INTRADOS")

with st.sidebar:
    mode = st.radio("Mode :", ["1. Preuve de Concept", "2. Phase 2 : PAUT"])
    index_scan = st.slider("Position du Sabot", 0, 99, 50)
    
    st.header("📐 Ajustement CAO (STL)")
    scale_cao = st.slider("Échelle / Scale", 0.1, 5.0, 1.0, 0.01) 
    
    with st.expander("Rotations CAO / CAD Rotations"):
        rx_cao = st.number_input("Rotation X (CAO)", value=0)
        ry_cao = st.number_input("Rotation Y (CAO)", value=0)
        rz_cao = st.number_input("Rotation Z (CAO)", value=0)
    
    with st.expander("Offsets CAO / CAD Offsets"):
        dx_cao = st.number_input("Offset X (CAO)", value=0.0)
        dy_cao = st.number_input("Offset Y (CAO)", value=0.0)
        dz_cao = st.number_input("Offset Z (CAO)", value=0.0)

    st.header("📉 Calage Courbes (Maple)")
    with st.expander("Offsets Courbes / Curves Offsets"):
        off_x = st.number_input("Ajustement Latéral X", value=0.0)
        off_y = st.number_input("Ajustement Profondeur Y", value=0.0)
        off_z = st.number_input("Ajustement Hauteur Z", value=0.0)

@st.cache_data
def load_and_process_stl():
    try:
        aube_mesh = mesh.Mesh.from_file('DuvhaLPStg5BldRoot_EDM_NOTCHES.stl')
        return aube_mesh.vectors
    except Exception as e:
        return None

# ==========================================
# 🎯 VRAIES DONNÉES MAPLE (Matrice C2 et Sabot)
# ==========================================
t = np.linspace(-57, 154, 100) 

# 1. Courbe du Sabot
x_sabot_brut = t
y_sabot_brut = 0.001808931726 * t**2 - 0.1314772754 * t + 17.33684201
z_sabot_brut = -0.003862202594 * t**2 + 0.1391553436 * t + 276.4372004

# 2. Les 14 Indications EDM (Matrice C2 exacte)
x_edm_brut = np.array([-57.33, -40.74, -24.02, -7.27, 9.52, 26.29, 42.98, 59.55, 75.96, 92.17, 108.12, 123.77, 139.08, 154.02])
y_edm_brut = np.full_like(x_edm_brut, -27.16) # Profondeur constante
z_edm_brut = np.array([315.33, 318.1, 319.64, 320.62, 320.4, 319.62, 317.81, 315.12, 311.57, 307.17, 301.93, 295.85, 288.97, 281.3]) # Courbe Z

# Application des offsets sur les calculs si besoin
x_sabot = x_sabot_brut + off_x
y_sabot = y_sabot_brut + off_y
z_sabot = z_sabot_brut + off_z

x_edm = x_edm_brut + off_x
y_edm = y_edm_brut + off_y
z_edm = z_edm_brut + off_z
# ==========================================

cible = int((index_scan / 99) * 13) # 14 points = index 0 à 13

col1, col2 = st.columns([2.5, 1.5])
with col1:
    fig = go.Figure()
    vectors = load_and_process_stl()
    
    if vectors is not None:
        p, q, r = vectors.shape
        vertices, faces = np.unique(vectors.reshape([p*q, r]), axis=0, return_inverse=True)
        faces = faces.reshape([p, q])
        
        # Logique pure Extrados : Echelle directe, pas de recentrage forcé
        vertices_scaled = vertices * scale_cao

        # Rotations
        rx, ry, rz = math.radians(rx_cao), math.radians(ry_cao), math.radians(rz_cao)
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        vertices_rotated = vertices_scaled @ (Rz @ Ry @ Rx).T
        
        # Offsets CAO manuels
        x_stl = vertices_rotated[:, 0] + dx_cao
        y_stl = vertices_rotated[:, 1] + dy_cao
        z_stl = vertices_rotated[:, 2] + dz_cao
        
        fig.add_trace(go.Mesh3d(x=x_stl, y=y_stl, z=z_stl, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color='lightgrey', opacity=0.4))

    # Affichage des trajectoires Maple
    fig.add_trace(go.Scatter3d(x=x_sabot, y=y_sabot, z=z_sabot, mode='lines', line=dict(color='lightblue', width=6)))
    fig.add_trace(go.Scatter3d(x=x_edm, y=y_edm, z=z_edm, mode='markers', marker=dict(color='red', size=5)))
    fig.add_trace(go.Scatter3d(x=[x_sabot[index_scan]], y=[y_sabot[index_scan]], z=[z_sabot[index_scan]], mode='markers', marker=dict(color='orange', symbol='square', size=12)))
    fig.add_trace(go.Scatter3d(x=[x_sabot[index_scan], x_edm[cible]], y=[y_sabot[index_scan], y_edm[cible]], z=[z_sabot[index_scan], z_edm[cible]], mode='lines', line=dict(color='yellow', width=6)))

    fig.update_layout(scene=dict(aspectmode='data'), width=800, height=700, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    dist = math.sqrt((x_sabot[index_scan]-x_edm[cible])**2 + (y_sabot[index_scan]-y_edm[cible])**2 + (z_sabot[index_scan]-z_edm[cible])**2)
    st.markdown("### 📡 Suivi de l'Inspection")
    if dist < 160: 
        st.success(f"🎯 **EDM DÉTECTÉ : N° {cible + 1} / 14**")
    else:
        st.info(f"🔍 Recherche de l'EDM n° {cible + 1}...")