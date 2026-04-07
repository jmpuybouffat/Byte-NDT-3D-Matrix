from src.hardware import Probe2D, Wedge, Specimen
from src.physics import FocalLawCalculator

def run_tests():
    print("--- DÉMARRAGE DES TESTS DU MOTEUR BYTE NDT ---")
    
    # 1. Instanciation du matériel
    imasonic_probe = Probe2D(nx=8, ny=8, pitch_x=0.6, pitch_y=0.6)
    rexolite_wedge = Wedge(velocity=2330.0, squat_angle_deg=36.0)
    steel_part = Specimen(velocity=3240.0)
    
    print(f"[OK] Sonde créée : {len(imasonic_probe.elements)} éléments.")
    print(f"[OK] Sabot créé : Vitesse {rexolite_wedge.velocity} m/s.")
    
    # 2. Assemblage dans le calculateur
    calculator = FocalLawCalculator(imasonic_probe, rexolite_wedge, steel_part)
    
    # 3. Test de calcul vers un point 3D spécifique
    test_delays = calculator.compute_fermat_3d(target_x=10.0, target_y=5.0, target_z=40.0)
    print(f"[OK] Moteur de lois focales initialisé. Prêt pour les équations de Fermat.")

if __name__ == "__main__":
    run_tests()
