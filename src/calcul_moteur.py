import numpy as np

# ==========================================
# 1. RÉFRACTION 3D (Méthode de Ferrari)
# ==========================================
def ferrari2(cr, DF, DT, DX):
    if abs(cr - 1) < 1e-6:
        return DX * DT / (DF + DT)

    cri = 1 / cr
    A = 1 - cri**2
    B = (2 * (cri**2) * DX - 2 * DX) / DT
    C = (DX**2 + DT**2 - (cri**2) * (DX**2 + DF**2)) / (DT**2)
    D = -2 * DX * (DT**2) / (DT**3)
    E = (DX**2) * (DT**2) / (DT**4)

    alpha = -3 * B**2 / (8 * A**2) + C / A
    beta = B**3 / (8 * A**3) - B * C / (2 * A**2) + D / A
    gamma = -3 * B**4 / (256 * A**4) + C * B**2 / (16 * A**3) - B * D / (4 * A**2) + E / A

    x_roots = np.zeros(4, dtype=complex)
    
    if abs(beta) < 1e-10:
        term1 = -B / (4 * A)
        term2 = np.sqrt((-alpha + np.sqrt(alpha**2 - 4 * gamma + 0j)) / 2)
        term3 = np.sqrt((-alpha - np.sqrt(alpha**2 - 4 * gamma + 0j)) / 2)
        x_roots[0] = term1 + term2
        x_roots[1] = term1 + term3
        x_roots[2] = term1 - term2
        x_roots[3] = term1 - term3
    else:
        P = -alpha**2 / 12 - gamma
        Q = -alpha**3 / 108 + alpha * gamma / 3 - beta**2 / 8
        Rm_inner = Q**2 / 4 + P**3 / 27
        Rm = Q / 2 - np.sqrt(Rm_inner + 0j)
        U = Rm**(1/3)
        y = -5/6 * alpha - U + (P / (3 * U) if abs(U) > 1e-10 else 0)
        W = np.sqrt(alpha + 2 * y + 0j)
        
        term1 = -B / (4 * A)
        term2_plus = np.sqrt(-(3 * alpha + 2 * y + 2 * beta / W) + 0j)
        term2_minus = np.sqrt(-(3 * alpha + 2 * y - 2 * beta / W) + 0j)
        
        x_roots[0] = term1 + 0.5 * (W + term2_plus)
        x_roots[1] = term1 + 0.5 * (-W + term2_minus)
        x_roots[2] = term1 + 0.5 * (W - term2_plus)
        x_roots[3] = term1 + 0.5 * (-W - term2_minus)

    tol = 1e-6
    for root in x_roots:
        xr = root.real
        axi = DT * abs(root.imag)
        xt = xr * DT
        if (DX >= 0 and 0 <= xt <= DX + tol) or (DX < 0 and DX - tol <= xt <= 0):
            if axi < tol:
                return xr * DT
                
    # Fallback fsolve si Ferrari échoue
    from scipy.optimize import fsolve
    def snell_func(x_val):
        return x_val / np.sqrt(x_val**2 + DT**2) - cr * (DX - x_val) / np.sqrt((DX - x_val)**2 + DF**2)
    guess = DX * DT / (DF + DT)
    res = fsolve(snell_func, guess)
    return res[0]

# ==========================================
# 2. LOIS FOCALES 3D (Steering + Skewing pour cartes Lecoeur)
# ==========================================
def delay_laws_3d_int(Mx, My, pitch_x, pitch_y, theta_wedge, target_x, target_y, target_z, c1, c2, wedge_height):
    cr = c1 / c2
    t = np.zeros((Mx, My))
    Mbx, Mby = (Mx - 1) / 2, (My - 1) / 2
    ex = (np.arange(Mx) - Mbx) * pitch_x
    ey = (np.arange(My) - Mby) * pitch_y
    
    for m in range(Mx):
        for n in range(My):
            De = wedge_height + ex[m] * np.sin(np.radians(theta_wedge))
            x_proj = target_x * np.cos(np.arctan2(target_y, target_x)) 
            y_proj = target_y
            Db = np.sqrt((target_x - ex[m] * np.cos(np.radians(theta_wedge)))**2 + (target_y - ey[n])**2)
            xi = ferrari2(cr, abs(target_z), De, Db)
            t_m1 = np.sqrt(xi**2 + De**2) / (c1 * 1e-3)
            t_m2 = np.sqrt(target_z**2 + (Db - xi)**2) / (c2 * 1e-3)
            t[m, n] = t_m1 + t_m2
            
    delays_us = np.max(t) - t
    delays_ns = np.round(delays_us * 1000).astype(int) # Conversion en Nanosecondes pour FPGA
    return delays_ns

# ==========================================
# 3. AMPLITUDE (Zoeppritz / Transmission)
# ==========================================
def t_fluid_solid(d1, cp1, d2, cp2, cs2, theta1_deg):
    iang = np.radians(theta1_deg)
    sinp = (cp2 / cp1) * np.sin(iang)
    sins = (cs2 / cp1) * np.sin(iang)
    cosp = np.where(sinp >= 1, 1j * np.sqrt(sinp**2 - 1 + 0j), np.sqrt(1 - sinp**2 + 0j))
    coss = np.where(sins >= 1, 1j * np.sqrt(sins**2 - 1 + 0j), np.sqrt(1 - sins**2 + 0j))
    sin_iang2 = np.sin(iang)**2
    cos_iang = np.sqrt(1 - sin_iang2)
    term_shear = 4 * ((cs2 / cp2)**2) * (sins * coss * sinp * cosp) + 1 - 4 * (sins**2) * (coss**2)
    denom = cosp + (d2 / d1) * (cp2 / cp1) * cos_iang * term_shear
    tpp = (2 * cos_iang * (1 - 2 * (sins**2))) / denom
    tps = -(4 * cosp * sins * cos_iang) / denom
    return np.abs(tpp), np.abs(tps)

# ==========================================
# 4. GÉNÉRATEUR DE FAISCEAU 3D COMPLET
# ==========================================
def compute_beam_3d_fast(Mx, My, pitch_x, pitch_y, target_x, target_y, target_z, theta_wedge, wedge_height, c_wedge, c_steel, freq_mhz):
    x_grid = np.linspace(target_x - 30, target_x + 30, 150)
    z_grid = np.linspace(max(0, target_z - 30), target_z + 30, 150)
    X, Z = np.meshgrid(x_grid, z_grid)
    P = np.zeros_like(X, dtype=complex)
    
    omega = 2 * np.pi * freq_mhz * 1e6
    Mbx, Mby = (Mx - 1) / 2, (My - 1) / 2
    
    for m in range(Mx):
        for n in range(My):
            ex = (m - Mbx) * pitch_x
            ey = (n - Mby) * pitch_y
            
            x_elem = ex * np.cos(np.radians(theta_wedge))
            z_elem = -wedge_height - ex * np.sin(np.radians(theta_wedge))
            
            f_grid = np.abs(z_elem) / (np.abs(z_elem) + Z)
            x_int_grid = x_elem + (X - x_elem) * f_grid
            
            L1_grid = np.sqrt((x_int_grid - x_elem)**2 + (target_y - ey)**2 + z_elem**2)
            L2_grid = np.sqrt((X - x_int_grid)**2 + (target_y - ey)**2 + Z**2)
            t_grid = L1_grid / (c_wedge * 1e3) + L2_grid / (c_steel * 1e3)
            
            f_target = np.abs(z_elem) / (np.abs(z_elem) + target_z)
            x_int_target = x_elem + (target_x - x_elem) * f_target
            
            L1_target = np.sqrt((x_int_target - x_elem)**2 + (target_y - ey)**2 + z_elem**2)
            L2_target = np.sqrt((target_x - x_int_target)**2 + (target_y - ey)**2 + target_z**2)
            t_target = L1_target / (c_wedge * 1e3) + L2_target / (c_steel * 1e3)
            
            phase = omega * (t_grid - t_target)
            
            R_total = L1_grid + L2_grid
            R_total[R_total == 0] = 1e-9
            
            P += (1 / R_total) * np.exp(-1j * phase)
            
    pressure = 20 * np.log10(np.abs(P) / np.max(np.abs(P)))
    pressure[pressure < -15] = -15 
    
    return x_grid, z_grid, pressure