import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Importation de VOTRE propre moteur de calcul
from src.hardware import Probe2D, Wedge, Specimen
from src.physics import FocalLawCalculator

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Byte NDT - Fermat 3D Engine", layout="wide")
st.title("🚀 Byte NDT : Moteur Fermat 3D (Sabot + Pièce)")

# --- INTERFACE UTILISATEUR (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Configuration Matérielle")
    pitch = st.slider("Pitch Sonde (mm)", 0.3, 1.5, 0.6)
    v_wedge = st.number_input("Vitesse Sabot (m/s) [ex: Rexolite]", value=2330.0)
    v_specimen = st.number_input("Vitesse Pièce (m/s) [ex: Acier L]", value=5900.0)
    
    st.header("🎯 Cible Focale 3D")
    fx = st.slider("Position X (mm)", -30.0, 30.0, 10.0)
    fy = st.slider("Position Y (mm)", -30.0, 30.0, 5.0)
    fz = st.slider("Profondeur Z (mm)", 10.0, 80.0, 40.0)
    
    f_mhz = 5.0 # Fréquence fixe pour la démo

# --- 1. INSTANCIATION DU MATÉRIEL ---
imasonic_probe = Probe2D(nx=8, ny=8, pitch_x=pitch, pitch_y=pitch)
wedge = Wedge(velocity=v_wedge)
specimen = Specimen(velocity=v_specimen)

# --- 2. APPEL AU MOTEUR DE LOIS FOCALES ---
with st.spinner("Calcul des trajectoires de Fermat (Ray Tracing 3D)..."):
    calculator = FocalLawCalculator(imasonic_probe, wedge, specimen)
    # On demande au moteur les retards pour notre cible
    delays = calculator.compute_fermat_3d(fx, fy, fz)

# --- 3. CALCUL DU CHAMP DE PRESSION (Huygens) ---
v_mm_s = v_specimen * 1000.0
omega = 2 * np.pi * (f_mhz * 1e6)
k = omega / v_mm_s

# Grille d'affichage 3D optimisée
x_grid = np.linspace(-30, 30, 30)
y_grid = np.linspace(-30, 30, 30)
z_grid = np.linspace(2, 60, 30)
X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

pressure = np.zeros_like(X, dtype=complex)

for idx, el in enumerate(imasonic_probe.elements):
    # La source virtuelle (simplifiée pour l'affichage ici, au niveau z=0)
    ex, ey = el[0], el[1]
    r = np.sqrt((X - ex)**2 + (Y - ey)**2 + Z**2)
    
    # On applique les retards calculés par VOTRE moteur physique
    phase = k * r - omega * delays[idx]
    pressure += (1 / r) * np.exp(1j * phase)

amp = np.abs(pressure)
amp_db = 20 * np.log10(amp / np.max(amp))

# --- 4. VISUALISATION 3D PLOTLY ---
fig = go.Figure()

# Dessin de la cible
fig.add_trace(go.Scatter3d(x=[fx], y=[fy], z=[fz], mode='markers', 
                           marker=dict(size=6, color='gold', symbol='cross'), name='Focale'))

# Enveloppe du faisceau
fig.add_trace(go.Isosurface(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=amp_db.flatten(),
    isomin=-6, isomax=0,
    surface_count=2, opacity=0.4, colorscale='magma', name='Faisceau (-6dB)'
))

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-30, 30], title="X"),
        yaxis=dict(range=[-30, 30], title="Y"),
        zaxis=dict(range=[60, 0], title="Z (Profondeur)"),
        aspectmode='manual', aspectratio=dict(x=1, y=1, z=1)
    ),
    height=700, margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig, use_container_width=True)
st.success("Lois focales générées avec succès par le moteur de Ray Tracing 3D.")