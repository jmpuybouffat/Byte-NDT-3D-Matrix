import streamlit as st
import trimesh
import plotly.graph_objects as go
import os
import numpy as np
import math

st.set_page_config(layout="wide", page_title="Byte NDT Digital Twin LSB")

# --- VOS RÉGLAGES GÉOMÉTRIQUES PARFAITS (INTOUCHABLES) ---
SCALE_PARFAIT = 0.700
RX_PARFAIT, RY_PARFAIT, RZ_PARFAIT = 180.0, 0.0, 180.0
DX_PARFAIT, DY_PARFAIT, DZ_PARFAIT = 40.0, 291.0, 15.0

# --- PARAMÈTRES PHYSIQUES DU SABOT ---
V_REXOLITE = 2330.0
ANGLE_SABOT = 36.0

st.title("Byte NDT - Digital Twin LSB 941 (Version Manuelle Fiable)")

dossier_actuel = os.path.dirname(__file__)
nom_du_fichier_stl = "DuvhaLPStg5BldRoot_EDM_NOTCHES.stl" 
chemin_fichier = os.path.join(dossier_actuel, nom_du_fichier_stl)

@st.cache_resource
def load_and_center_mesh(chemin):
    mesh = trimesh.load_mesh(chemin)
    centroid = mesh.centroid
    return mesh.vertices - centroid, mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]

try:
    v_centered, i_mesh, j_mesh, k_mesh = load_and_center_mesh(chemin_fichier)

    st.sidebar.header("🛠️ Mode de Travail")
    mode_travail = st.sidebar.radio(
        "Choisissez l'interface :",
        ["1. Preuve de Concept (Faisceau Simple)", "2. Phase 2 : Sabot & S-Scan (Imasonic)"]
    )
    st.sidebar.markdown("---")

    # Application de la transformation avec VOS valeurs
    v_scaled = v_centered * SCALE_PARFAIT
    rx, ry, rz = math.radians(RX_PARFAIT), math.radians(RY_PARFAIT), math.radians(RZ_PARFAIT)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    v_rot = v_scaled @ (Rz @ Ry @ Rx).T
    x_fin, y_fin, z_fin = v_rot[:, 0] + DX_PARFAIT, v_rot[:, 1] + DY_PARFAIT, v_rot[:, 2] + DZ_PARFAIT

    # --- MATHÉMATIQUES DES TRAJECTOIRES ---
    NB_POINTS = 100
    t = np.linspace(-30, 110, NB_POINTS)
    x_capteurs = t
    y_capteurs = -0.005488828669 * t**2 + 0.4063509956 * t + 322.5637799
    z_capteurs = -0.003609404789 * t**2 + 0.3901494991 * t + 19.70324623

    matrice_c1 = np.array([
        [-34.97, 273.07, -27.16], [-20.63, 274.52, -27.16], [-6.24, 275.23, -27.16],
        [8.18, 275.18, -27.16], [22.57, 274.38, -27.16], [36.89, 272.81, -27.16],
        [51.12, 270.51, -27.16], [65.21, 267.46, -27.16], [79.12, 263.68, -27.16],
        [92.82, 259.2, -27.16], [106.25, 253.37, -27.16], [113.39, 248.06, -27.16]
    ])
    x_edm, y_edm, z_edm = matrice_c1[:, 0], matrice_c1[:, 1], matrice_c1[:, 2]

    # --- LE PILOTAGE 100% MANUEL (Plus de plantage) ---
    st.sidebar.header("🕹️ Pilotage Manuel")
    st.sidebar.info("Utilisez le curseur ci-dessous pour déplacer le sabot pas à pas.")
    index_scan = st.sidebar.slider("Position du Sabot (Scan Index)", 0, NB_POINTS - 1, 0)

    pos_x, pos_y, pos_z = x_capteurs[index_scan], y_capteurs[index_scan], z_capteurs[index_scan]
    cible_x, cible_y, cible_z = pos_x, np.interp(pos_x, x_edm, y_edm), np.interp(pos_x, x_edm, z_edm)
    distance_faisceau = math.sqrt((cible_x - pos_x)**2 + (cible_y - pos_y)**2 + (cible_z - pos_z)**2)

    # --- VRAIE PHYSIQUE : Calcul de la distance entre le faisceau et les vrais défauts ---
    distances_aux_defauts = np.sqrt((x_edm - cible_x)**2 + (y_edm - cible_y)**2 + (z_edm - cible_z)**2)
    erreur_pointage = np.min(distances_aux_defauts)
    # L'amplitude s'effondre si on rate le défaut de plus de 2-3 mm
    amplitude_reelle = 0.85 * math.exp(-0.5 * (erreur_pointage / 3.0)**2)

    col1, col2 = st.columns([2.5, 1])

    with col1:
        fig3d = go.Figure()
        fig3d.add_trace(go.Mesh3d(x=x_fin, y=y_fin, z=z_fin, i=i_mesh, j=j_mesh, k=k_mesh, color='lightgray', opacity=0.3, name="Aube", hoverinfo='skip'))
        fig3d.add_trace(go.Scatter3d(x=x_capteurs, y=y_capteurs, z=z_capteurs, mode='lines', line=dict(color='lightblue', width=4), name="Trajectoire"))
        fig3d.add_trace(go.Scatter3d(x=x_edm, y=y_edm, z=z_edm, mode='markers', marker=dict(color='red', size=4), name="EDM"))
        
        if "1. Preuve de Concept" in mode_travail:
            taille_sabot = 15
            hx, hy, hz = taille_sabot/2, taille_sabot/2, 8
            sabot_x = [pos_x-hx, pos_x+hx, pos_x+hx, pos_x-hx, pos_x-hx, pos_x+hx, pos_x+hx, pos_x-hx]
            sabot_y = [pos_y-hy, pos_y-hy, pos_y+hy, pos_y+hy, pos_y-hy, pos_y-hy, pos_y+hy, pos_y+hy]
            sabot_z = [pos_z, pos_z, pos_z, pos_z, pos_z+hz, pos_z+hz, pos_z+hz, pos_z+hz]
            sabot_i = [0, 1, 5, 4, 0, 4, 7, 3, 3, 2, 6, 7, 1, 2, 6, 5, 0, 1, 2, 3, 4, 5, 6, 7]
            sabot_j = [1, 5, 6, 0, 4, 7, 6, 2, 2, 6, 5, 4, 2, 6, 5, 4, 1, 2, 3, 0, 5, 6, 7, 4]
            sabot_k = [5, 6, 2, 4, 7, 6, 5, 1, 6, 5, 1, 0, 6, 5, 1, 0, 2, 3, 0, 1, 6, 7, 4, 5]
            fig3d.add_trace(go.Mesh3d(x=sabot_x, y=sabot_y, z=sabot_z, i=sabot_i, j=sabot_j, k=sabot_k, color='orange', opacity=0.8, name="PA Wedge"))
            fig3d.add_trace(go.Scatter3d(x=[pos_x, cible_x], y=[pos_y, cible_y], z=[pos_z, cible_z], mode='lines', line=dict(color='yellow', width=6), name="Faisceau Central", hoverinfo='skip'))

        else:
            L, W = 25, 20
            H_avant = 5
            H_arriere = 5 + L * math.tan(math.radians(ANGLE_SABOT))
            wx = [pos_x-W/2, pos_x+W/2, pos_x+W/2, pos_x-W/2, pos_x-W/2, pos_x+W/2, pos_x+W/2, pos_x-W/2]
            wy = [pos_y-L/2, pos_y-L/2, pos_y+L/2, pos_y+L/2, pos_y-L/2, pos_y-L/2, pos_y+L/2, pos_y+L/2]
            wz = [pos_z, pos_z, pos_z, pos_z, pos_z+H_avant, pos_z+H_avant, pos_z+H_arriere, pos_z+H_arriere]
            wi = [0, 1, 5, 4, 0, 4, 7, 3, 3, 2, 6, 7, 1, 2, 6, 5, 0, 1, 2, 3, 4, 5, 6, 7]
            wj = [1, 5, 6, 0, 4, 7, 6, 2, 2, 6, 5, 4, 2, 6, 5, 4, 1, 2, 3, 0, 5, 6, 7, 4]
            wk = [5, 6, 2, 4, 7, 6, 5, 1, 6, 5, 1, 0, 6, 5, 1, 0, 2, 3, 0, 1, 6, 7, 4, 5]
            fig3d.add_trace(go.Mesh3d(x=wx, y=wy, z=wz, i=wi, j=wj, k=wk, color='darkorange', opacity=0.9, name=f"Wedge {ANGLE_SABOT}° (Rexolite)"))
            
            longueur_ray = 60
            for angle in [35, 70]: 
                for skew in [-10, 0, 10]: 
                    r_angle = math.radians(angle)
                    r_skew = math.radians(skew)
                    dy = -longueur_ray * math.cos(r_angle)
                    dz = -longueur_ray * math.sin(r_angle)
                    dx = longueur_ray * math.sin(r_skew)
                    fig3d.add_trace(go.Scatter3d(
                        x=[pos_x, pos_x+dx], y=[pos_y, pos_y+dy], z=[pos_z, pos_z+dz], 
                        mode='lines', line=dict(color='yellow', width=2, dash='solid' if skew==0 else 'dot'), showlegend=False, hoverinfo='skip'
                    ))

        fig3d.update_layout(
            scene=dict(aspectmode='data', dragmode='turntable', camera=dict(eye=dict(x=-1.8, y=0.5, z=1.2), center=dict(x=0, y=0, z=-0.1))), 
            margin=dict(l=0, r=0, b=0, t=0), legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)")
        )
        st.plotly_chart(fig3d, use_container_width=True, key="fig3d")

    with col2:
        axe_profondeur = np.linspace(0, 150.0, 500)
        signal = np.random.normal(0, 0.02, 500)
        
        # Le signal ne monte que si le faisceau est sur le défaut
        echo_defaut = amplitude_reelle * np.exp(-0.5 * ((axe_profondeur - distance_faisceau) / 1.5)**2)
        signal += echo_defaut

        fig_ascan = go.Figure()
        fig_ascan.add_trace(go.Scatter(x=axe_profondeur, y=signal, mode='lines', line=dict(color='lime', width=2), name="Signal RF"))
        fig_ascan.update_layout(
            plot_bgcolor='black', paper_bgcolor='white', margin=dict(l=0, r=0, b=0, t=30), showlegend=False,
            xaxis=dict(title="Profondeur (mm)", color='black', gridcolor='#333333', range=[0, 150]), 
            yaxis=dict(title="Amplitude (%)", range=[-0.1, 1.1], color='black', gridcolor='#333333')
        )
        
        if amplitude_reelle > 0.4:
            st.success(f"🔴 **DÉFAUT DÉTECTÉ**\n\nDistance = {distance_faisceau:.2f} mm")
        else:
            st.info(f"📍 **Scan en cours...**\n\nCherche EDM...")
            
        st.plotly_chart(fig_ascan, use_container_width=True, key="fig_ascan")

except Exception as e:
    st.error(f"Erreur technique : {e}")
