import numpy as np
from scipy.optimize import minimize

class FocalLawCalculator:
    def __init__(self, probe, wedge, specimen):
        self.probe = probe
        self.wedge = wedge
        self.specimen = specimen

    def compute_fermat_3d(self, tx, ty, tz):
        v1, v2 = self.wedge.velocity, self.specimen.velocity
        target = np.array([tx, ty, tz])
        tof, points_i = [], []

        for el in self.probe.elements:
            z_el = -5.0 - (el[0] + self.probe.lx/2) * np.tan(np.radians(self.wedge.angle_deg))
            el_pos = np.array([el[0], el[1], z_el])
            
            def time_func(p):
                pi = np.array([p[0], p[1], 0.0])
                return (np.linalg.norm(pi - el_pos)/v1) + (np.linalg.norm(target - pi)/v2)
            
            res = minimize(time_func, [0, 0], method='Nelder-Mead')
            tof.append(res.fun)
            points_i.append([res.x[0], res.x[1], 0.0])
            
        tof = np.array(tof)
        # Conversion stricte en nanosecondes ENTIÈRES pour le FPGA
        delays_ns = np.round((np.max(tof) - tof) * 1e9).astype(int)
        return delays_ns, np.array(points_i)


def compute_beam_pressure_2d(probe_elements, delays_ns, velocity_m_s, freq_mhz, x_bounds, z_bounds, resolution=1.0):
    """
    Calcule le champ de pression acoustique 2D (Slice XZ).
    """
    # Conversion des unités pour les mathématiques
    omega = 2 * np.pi * (freq_mhz * 1e6)  # Pulsation
    k = omega / velocity_m_s              # Nombre d'onde (k)
    
    # 1. Création de la grille d'observation (notre écran virtuel)
    x = np.arange(x_bounds[0], x_bounds[1], resolution)
    z = np.arange(z_bounds[0], z_bounds[1], resolution)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X) # On se place à Y=0 pour couper le faisceau au centre
    
    # Initialisation du champ de pression complexe (des zéros partout au début)
    P = np.zeros_like(X, dtype=np.complex128)
    
    # 2. Superposition (Principe de Huygens) : on additionne la contribution de chaque élément
    for i in range(len(probe_elements)):
        ex, ey, ez = probe_elements[i]
        delay_s = delays_ns[i] * 1e-9 # Conversion des nanosecondes en secondes
        
        # Distance géométrique entre l'élément i et chaque point de la grille
        R = np.sqrt((X - ex)**2 + (Y - ey)**2 + (Z - ez)**2)
        R[R == 0] = 1e-9 # Sécurité pour éviter la division par zéro au contact
        
        # Phase de l'onde : distance parcourue moins le retard appliqué
        phase = (k * R) - (omega * delay_s)
        
        # Addition de l'onde (Amplitude qui décroit en 1/R * exponentielle complexe de la phase)
        P += (1/R) * np.exp(1j * phase)
        
    # 3. Conversion en Décibels (dB) pour un bel affichage
    pressure_mag = np.abs(P)
    # Normalisation par rapport au point le plus fort
    pressure_db = 20 * np.log10(pressure_mag / np.max(pressure_mag))
    
    # On coupe à -20 dB pour ne pas afficher le "bruit" de fond
    pressure_db[pressure_db < -20] = -20
    
    return x, z, pressure_db


def generate_a_scan_echo(focus_z_mm, velocity_m_s, freq_mhz, sampling_rate_mhz=100):
    """
    Simule le signal A-Scan (écho) provenant d'un défaut (type EDM) situé au point focal.
    Génère un signal temporel réaliste pour l'entraînement d'une IA.
    """
    # 1. Calcul du Temps de Vol (Aller-Retour) vers la profondeur Z
    # On multiplie par 2 car l'onde fait l'aller ET le retour
    tof_round_trip_s = 2 * (focus_z_mm * 1e-3) / velocity_m_s
    
    # 2. Création de l'axe du temps (Fenêtre d'écoute de 100 microsecondes)
    fs = sampling_rate_mhz * 1e6
    time_s = np.arange(0, 100e-6, 1/fs)
    
    # 3. Création de l'impulsion ultrasonore (Onde de Gabor)
    omega = 2 * np.pi * (freq_mhz * 1e6)
    # La largeur de l'impulsion dépend de la fréquence (plus la fréquence est haute, plus l'impulsion est courte)
    sigma = 1.0 / (freq_mhz * 1e6) 
    
    # L'écho est une sinusoïde amortie centrée sur le temps de vol
    echo = np.cos(omega * (time_s - tof_round_trip_s)) * np.exp(-((time_s - tof_round_trip_s)**2) / (2 * sigma**2))
    
    # 4. Ajout d'un bruit blanc (Essentiel pour le Machine Learning !)
    noise = np.random.normal(0, 0.05, len(time_s))
    a_scan = echo + noise
    
    # 5. Conversion du temps en microsecondes (µs) pour un affichage lisible
    time_us = time_s * 1e6
    
    return time_us, a_scan