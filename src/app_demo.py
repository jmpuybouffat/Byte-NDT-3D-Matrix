import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from calcul_moteur import delay_laws_3d_int, t_fluid_solid, compute_beam_3d_fast

st.set_page_config(page_title="PAUT Engine Demo", layout="wide")

st.title("🧪 Jumeau Numérique / Digital Twin - Moteur de Calcul / Physics Engine")
st.markdown("Interface de test du moteur physique (Lois de Ferrari, Zoeppritz, Huygens 3D) / *Physics engine testing interface.*")

# ==========================================
# BARRE LATERALE : PARAMÈTRES (Bilingue)
# ==========================================
with st.sidebar:
    st.header("⚙️ Paramètres / Settings")
    
    st.subheader("1. Sonde / Probe (Imasonic)")
    col1, col2 = st.columns(2)
    with col1:
        Mx = st.number_input("Elements X (Mx)", min_value=1, max_value=32, value=8)
        pitch_x = st.number_input("Pitch X (mm)", min_value=0.1, max_value=5.0, value=0.43, step=0.01)
    with col2:
        My = st.number_input("Elements Y (My)", min_value=1, max_value=32, value=8)
        pitch_y = st.number_input("Pitch Y (mm)", min_value=0.1, max_value=5.0, value=0.43, step=0.01)
    freq_mhz = st.slider("Fréquence / Frequency (MHz)", 1.0, 15.0, 5.0, 0.1)

    st.subheader("2. Sabot / Wedge")
    theta_wedge = st.slider("Angle Sabot / Wedge Angle (°)", 0.0, 60.0, 36.0, 0.1)
    wedge_height = st.slider("Hauteur Centre / Height DT0 (mm)", 5.0, 50.0, 15.0, 0.5)
    c1 = st.slider("Vitesse Sabot / Vel. c1 (m/s)", 1000, 3000, 2330, 10)
    d1 = st.slider("Densité Sabot / Density d1", 0.5, 3.0, 1.18, 0.01)

    st.subheader("3. Pièce / Material (Steel)")
    cp2 = st.slider("Vitesse P Acier / P-Vel. cp2 (m/s)", 3000, 7000, 5900, 10)
    cs2 = st.slider("Vitesse S Acier / S-Vel. cs2 (m/s)", 2000, 4000, 3230, 10)
    d2 = st.slider("Densité Acier / Density d2", 5.0, 10.0, 7.85, 0.01)

    st.subheader("🎯 Cible 3D / 3D Target")
    target_x = st.slider("Cible X / Target X (mm) [Steering]", -30.0, 30.0, 0.0, 0.5)
    target_y = st.slider("Cible Y / Target Y (mm) [Skewing]", -30.0, 30.0, 0.0, 0.5)
    target_z = st.slider("Profondeur Z / Depth Z (mm)", 10.0, 150.0, 50.0, 0.5)

# ==========================================
# CALCULS VIA LE MOTEUR
# ==========================================
# 1. Calcul de la transmission d'énergie (Zoeppritz)
angle_inc = np.degrees(np.arctan2(np.sqrt(target_x**2 + target_y**2), target_z))
tpp, tps = t_fluid_solid(d1, c1, d2, cp2, cs2, angle_inc)

# 2. Calcul des lois focales (Ferrari 3D)
delays_ns = delay_laws_3d_int(Mx, My, pitch_x, pitch_y, theta_wedge, target_x, target_y, target_z, c1, cp2, wedge_height)

# 3. Calcul de la tache focale (Huygens avec Sabot)
x_grid, z_grid, pressure = compute_beam_3d_fast(Mx, My, pitch_x, pitch_y, target_x, target_y, target_z, theta_wedge, wedge_height, c1, cp2, freq_mhz)

# ==========================================
# AFFICHAGE DES RÉSULTATS
# ==========================================
col_A, col_B = st.columns([1, 1.5])

with col_A:
    st.subheader("📊 Données pour FPGA / FPGA Delay Laws (ns)")
    st.write("Matrice des retards calculée par la méthode de Ferrari (Steering + Skewing). Prêt pour injection hardware. / *Delay matrix calculated via Ferrari's method. Ready for hardware injection.*")
    st.dataframe(delays_ns, use_container_width=True)
    
    st.subheader("⚡ Énergie Transmise / Transmitted Energy (Zoeppritz)")
    st.metric("Coef. P-P (Longitudinal)", f"{tpp:.3f}")
    st.metric("Coef. P-S (Transversal / Shear)", f"{tps:.3f}")

with col_B:
    st.subheader("🔥 Profil du Faisceau / Beam Profile")
    st.write(f"Tache focale simulée autour du point Z={target_z}mm. / *Simulated focal spot around Z={target_z}mm.*")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.pcolormesh(x_grid, z_grid, pressure, cmap='jet', shading='gouraud', vmin=-15, vmax=0)
    ax.invert_yaxis() 
    ax.set_xlabel("Axe X / X-Axis (mm)")
    ax.set_ylabel("Profondeur Z / Z-Depth (mm)")
    ax.set_title(f"Focus à / at X={target_x}, Y={target_y}, Z={target_z}")
    fig.colorbar(c, ax=ax, label="Amplitude (dB)")
    
    ax.plot(target_x, target_z, 'w+', markersize=12, label="Cible / Target")
    ax.legend()
    
    st.pyplot(fig)