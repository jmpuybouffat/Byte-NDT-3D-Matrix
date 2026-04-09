import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.hardware import Probe2D, Wedge, Specimen
# CORRECTION 1 : L'import de la fonction est bien là
from src.physics import FocalLawCalculator, compute_beam_pressure_2d

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Byte NDT - Interface FPGA", layout="wide")

# --- GESTION BILINGUE ---
if 'lang' not in st.session_state: 
    st.session_state.lang = 'FR'

def toggle_lang(): 
    st.session_state.lang = 'EN' if st.session_state.lang == 'FR' else 'FR'

st.sidebar.button("🌐 Switch Language (FR/EN)", on_click=toggle_lang)
L = st.session_state.lang

T = {
    "FR": {"title": "🚀 Byte NDT : Jumeau Numérique FPGA", "mode": "⚙️ Stratégie FPGA", "probe": "💎 Sonde Matricielle", "media": "📐 Milieux (Sliders)", "beam": "🎯 Pilotage Faisceau", "dim": "📝 Dimensions & Gaps", "laws": "⏱️ Lois Focales (ns - INT)", "fmc": "📡 Séquence FMC"},
    "EN": {"title": "🚀 Byte NDT: FPGA Digital Twin", "mode": "⚙️ FPGA Strategy", "probe": "💎 Matrix Probe", "media": "📐 Media (Sliders)", "beam": "🎯 Beam Steering", "dim": "📝 Dimensions & Gaps", "laws": "⏱️ Focal Laws (ns - INT)", "fmc": "📡 FMC Sequence"}
}

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header(T[L]["mode"])
    mode = st.selectbox("Mode :", ["1. Point Unique / Single Point", "2. Balayage Sectoriel / S-Scan", "3. FMC (Full Matrix Capture)"])
    
    st.header(T[L]["probe"])
    col1, col2 = st.columns(2)
    nx = col1.number_input("Nx", 1, 64, 8)
    ny = col2.number_input("Ny", 1, 64, 8)
    pitch = st.number_input("Pitch (mm)", 0.1, 5.0, 0.43)
    
    col3, col4 = st.columns(2)
    gap_x = col3.slider("Gap X (mm)", 0.01, 0.15, 0.05, step=0.01)
    gap_y = col4.slider("Gap Y (mm)", 0.01, 0.15, 0.05, step=0.01)
    
    freq = st.number_input("Fréquence (MHz)", 1.0, 20.0, 5.0)
    
    st.header(T[L]["media"])
    v_w = st.slider("Vitesse Sabot / Wedge Vel. (m/s)", 1000, 4000, 2330, step=10)
    v_s = st.slider("Vitesse Acier / Steel Vel. (m/s)", 2000, 7000, 3330, step=10)
    w_ang = st.slider("Angle Sabot / Wedge Angle (°)", 0.0, 70.0, 36.0)

    st.header(T[L]["beam"])
    fz = st.slider("Focus Z (mm)", 5.0, 150.0, 103.0)

    if mode == "1. Point Unique / Single Point":
        theta = st.slider("Steering θ (°)", -60.0, 60.0, 15.0)
        phi = st.slider("Skew φ (°)", -180.0, 180.0, -31.0)
    elif mode == "2. Balayage Sectoriel / S-Scan":
        theta_start = st.slider("Départ Steering θ (°)", 0.0, 80.0, 35.0)
        theta_end = st.slider("Fin Steering θ (°)", 0.0, 80.0, 70.0)
        phi = st.slider("Skew φ (Fixe) (°)", -180.0, 180.0, 10.0)

# --- INITIALISATION ---
probe = Probe2D(nx=nx, ny=ny, pitch_x=pitch, pitch_y=pitch, gap_x=gap_x, gap_y=gap_y, freq_mhz=freq)
calc = FocalLawCalculator(probe, Wedge(velocity=v_w, angle_deg=w_ang), Specimen(velocity=v_s))

# --- AFFICHAGE PRINCIPAL ---
st.title(T[L]["title"]) 
c1, c2 = st.columns([2, 1])

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=probe.elements[:,0], y=probe.elements[:,1], z=np.zeros(nx*ny)-20, mode='markers', marker=dict(size=3, color='blue'), name="Sonde PA"))

with c2:
    st.markdown(f"### {T[L]['dim']}")
    st.write(f"- **Lx x Ly :** {probe.lx:.2f} x {probe.ly:.2f} mm")
    st.write(f"- **Gaps (X / Y) :** {probe.gap_x:.2f} / {probe.gap_y:.2f} mm")

if mode == "1. Point Unique / Single Point":
    tx = fz * np.tan(np.radians(theta)) * np.cos(np.radians(phi))
    ty = fz * np.tan(np.radians(theta)) * np.sin(np.radians(phi))
    # CORRECTION 2 : on appelle bien la variable d_ns
    d_ns, points_i = calc.compute_fermat_3d(tx, ty, fz) 
    
    mid = len(probe.elements) // 2
    fig.add_trace(go.Scatter3d(x=[probe.elements[mid,0], points_i[mid,0], tx], y=[probe.elements[mid,1], points_i[mid,1], ty], z=[-20, 0, fz], mode='lines+markers', line=dict(color='red', width=5), name="Beam"))
    
    with c2:
        st.markdown(f"### {T[L]['laws']}")
        st.dataframe(d_ns.reshape(nx, ny).astype(int))

elif mode == "2. Balayage Sectoriel / S-Scan":
    angles = np.linspace(theta_start, theta_end, 5)
    all_delays = {}
    
    for ang in angles:
        tx = fz * np.tan(np.radians(ang)) * np.cos(np.radians(phi))
        ty = fz * np.tan(np.radians(ang)) * np.sin(np.radians(phi))
        d_ns, p_i = calc.compute_fermat_3d(tx, ty, fz)
        all_delays[f"Steering {ang:.1f}°"] = d_ns
        
        mid = len(probe.elements) // 2
        fig.add_trace(go.Scatter3d(x=[probe.elements[mid,0], p_i[mid,0], tx], y=[probe.elements[mid,1], p_i[mid,1], ty], z=[-20, 0, fz], mode='lines+markers', line=dict(width=3), name=f"Beam {ang:.1f}°"))
        
    with c2:
        st.markdown(f"### {T[L]['laws']} (Skew: {phi}°)")
        angle_sel = st.selectbox("Angle :", list(all_delays.keys()))
        st.dataframe(all_delays[angle_sel].reshape(nx, ny).astype(int))
        # On s'assure que d_ns correspond à l'angle sélectionné pour l'affichage du faisceau
        d_ns = all_delays[angle_sel]

elif mode == "3. FMC (Full Matrix Capture)":
    with c2:
        st.markdown(f"### {T[L]['fmc']}")
        st.info("Acquisition TFM : Tx = 1 to N, Rx = All.")
        st.write(f"- **Tx :** {nx*ny}")
        st.write(f"- **A-Scans :** {nx*ny}")

# CORRECTION 3 : un seul "with c1:" et une protection pour le mode FMC
with c1:
    if mode != "3. FMC (Full Matrix Capture)":
        # --- CALCUL ET AFFICHAGE DU FAISCEAU ACOUSTIQUE ---
        with st.spinner("Calcul du champ de pression en cours..."): 
            x_grid, z_grid, pressure = compute_beam_pressure_2d(
                probe.elements, 
                d_ns,             
                v_s,              
                freq,             
                x_bounds=[-30, 80], 
                z_bounds=[0, 150], 
                resolution=1.0    
            )
            
            y_grid = np.zeros((len(z_grid), len(x_grid)))
            
            fig.add_trace(go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                surfacecolor=pressure,
                colorscale='Jet',     
                cmin=-20, cmax=0,     
                opacity=0.6,          
                showscale=False,
                name="Faisceau (dB)"
            ))

    fig.update_layout(scene=dict(zaxis=dict(range=[fz+10, -30], autorange="reversed")), height=750)
    st.plotly_chart(fig, use_container_width=True)