import numpy as np
from scipy.optimize import minimize

class FocalLawCalculator:
    """Moteur de calcul des retards avec réfraction (Principe de Fermat 3D)"""
    def __init__(self, probe, wedge, specimen):
        self.probe = probe
        self.wedge = wedge
        self.specimen = specimen

    def compute_fermat_3d(self, target_x, target_y, target_z):
        """
        Calcule la loi focale exacte en minimisant le temps de vol à travers 2 milieux.
        Prend en compte la réfraction (Snell-Descartes 3D).
        """
        num_elements = len(self.probe.elements)
        times_of_flight = np.zeros(num_elements)
        
        v1 = self.wedge.velocity
        v2 = self.specimen.velocity
        target = np.array([target_x, target_y, target_z])
        
        print(f"[Calculateur] Tracé de rayons 3D (Ray Tracing) vers X={target_x}, Y={target_y}, Z={target_z}...")
        
        # Hauteur virtuelle du sabot (ex: 20mm au-dessus de la pièce)
        wedge_height = 20.0 
        
        for idx, el in enumerate(self.probe.elements):
            # Position (X, Y, Z) de l'élément sur la sonde
            el_pos = np.array([el[0], el[1], -wedge_height])
            
            # Fonction de coût : Temps de parcours total selon le point d'incidence
            def time_of_flight(interface_point):
                # p_int est le point où le rayon touche la pièce (z = 0)
                p_int = np.array([interface_point[0], interface_point[1], 0.0])
                d1 = np.linalg.norm(p_int - el_pos) # Distance dans le sabot
                d2 = np.linalg.norm(target - p_int) # Distance dans la pièce
                return (d1 / v1) + (d2 / v2)
            
            # Point de départ initial pour l'algorithme
            guess = [
                el_pos[0] + (target[0] - el_pos[0]) * 0.5,
                el_pos[1] + (target[1] - el_pos[1]) * 0.5
            ]
            
            # Algorithme d'optimisation numérique (trouve le vrai point d'incidence)
            res = minimize(time_of_flight, guess, method='Nelder-Mead')
            times_of_flight[idx] = res.fun
            
        # Conversion Temps de vol -> Loi de retard
        # L'onde qui a le trajet le plus long dicte le temps T=0.
        max_time = np.max(times_of_flight)
        delays = max_time - times_of_flight
        
        return delays
