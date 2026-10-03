import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# =============================================================================
# 1. MODELLO DI GENT
# =============================================================================
def risolvi_gent(P_i, lambda_z, R_i, R_e, mu, Jm, C_guess=0.0, n_pts=300):
    if abs(P_i) < 1e-12:
        return R_i / np.sqrt(lambda_z), R_e / np.sqrt(lambda_z), 0.0

    def calcola_pressione_gent(C):
        if C < 0:
            return -1e10

        r_i2 = (R_i**2) / lambda_z + 2.0 * C
        r_e2 = (R_e**2) / lambda_z + 2.0 * C
        if r_i2 <= 1e-10 or r_e2 <= 1e-10:
            return -1e10

        r_i, r_e = np.sqrt(r_i2), np.sqrt(r_e2)
        r = np.linspace(r_i, r_e, n_pts)

        R2 = lambda_z * (r**2 - 2.0 * C)
        if np.any(R2 <= 1e-10):
            return -1e10

        lambda_theta = r / np.sqrt(R2)
        lambda_r = 1.0 / (lambda_theta * lambda_z)

        I1 = lambda_r**2 + lambda_theta**2 + lambda_z**2
        margine = Jm - (I1 - 3.0)

        if np.any(margine <= 1e-6):
            return -1e10  # Locking cinetico raggiunto

        dWdI1 = (mu * Jm) / (2.0 * margine)
        tau_diff = 2.0 * dWdI1 * (lambda_theta**2 - lambda_r**2)

        return trapz_func(tau_diff / r, r)

    def residuo(C):
        P_calc = calcola_pressione_gent(C)
        if P_calc < 0:
            return -1e10
        return P_calc - P_i

    # Trova limite massimo teorico di C per non superare il locking
    C_low = 0.0
    
    # Ricerca robusta di C_high
    step = 0.5 if C_guess == 0.0 else max(C_guess * 0.05, 0.1)
    C_high = max(C_guess, 0.01)
    
    found = False
    for _ in range(200):
        res = residuo(C_high)
        if res >= 0:
            found = True
            break
        elif res < -1e8: # Superato il limite di locking/validità fisica
            # Torna indietro e riduci lo step per avvicinarti alla parete di locking
            C_high -= step
            step *= 0.5
            if step < 1e-6:
                break
        C_high += step

    if not found or residuo(C_high) < 0:
        raise ValueError("Pressione oltre il limite di locking del materiale.")

    sol = root_scalar(residuo, bracket=[C_low, C_high], method='brentq', xtol=1e-7)
    C_opt = sol.root
    r_i_def = np.sqrt((R_i**2) / lambda_z + 2.0 * C_opt)
    r_e_def = np.sqrt((R_e**2) / lambda_z + 2.0 * C_opt)
    return r_i_def, r_e_def, C_opt


# =============================================================================
# 2. MODELLO DI MOONEY-RIVLIN
# =============================================================================
def risolvi_mooney_rivlin(P_i, lambda_z, R_i, R_e, C10, C01, C_guess=0.0, n_pts=300):
    if abs(P_i) < 1e-12:
        return R_i / np.sqrt(lambda_z), R_e / np.sqrt(lambda_z), 0.0

    def calcola_pressione_mr(C):
        if C < 0:
            return -1e10

        r_i2 = (R_i**2) / lambda_z + 2.0 * C
        r_e2 = (R_e**2) / lambda_z + 2.0 * C
        if r_i2 <= 1e-10 or r_e2 <= 1e-10:
            return -1e10

        r_i, r_e = np.sqrt(r_i2), np.sqrt(r_e2)
        r = np.linspace(r_i, r_e, n_pts)

        R2 = lambda_z * (r**2 - 2.0 * C)
        if np.any(R2 <= 1e-10):
            return -1e10

        lambda_theta = r / np.sqrt(R2)
        lambda_r = 1.0 / (lambda_theta * lambda_z)

        tau_diff = 2.0 * (C10 + C01 * lambda_z**2) * (lambda_theta**2 - lambda_r**2)

        return trapz_func(tau_diff / r, r)

    def residuo(C):
        P_calc = calcola_pressione_mr(C)
        if P_calc < 0:
            return -1e10
        return P_calc - P_i

    C_low = 0.0
    C_high = max(1.0, C_guess * 1.2)
    step = 0.5

    found = False
    for _ in range(100):
        if residuo(C_high) >= 0:
            found = True
            break
        C_high += step
        step *= 1.2

    if not found:
        raise ValueError("Impossibile trovare un intervallo per Mooney-Rivlin.")

    sol = root_scalar(residuo, bracket=[C_low, C_high], method='brentq', xtol=1e-7)
    C_opt = sol.root
    r_i_def = np.sqrt((R_i**2) / lambda_z + 2.0 * C_opt)
    r_e_def = np.sqrt((R_e**2) / lambda_z + 2.0 * C_opt)
    return r_i_def, r_e_def, C_opt


# =============================================================================
# 3. SWEEP DI PRESSIONE
# =============================================================================
if __name__ == "__main__":
    R_i = 10.0      # Raggio interno [mm]
    R_e = 11.0      # Raggio esterno [mm]
    lambda_z = 1.2  # Stretch assiale

    mu = 0.4        # Modulo di taglio [MPa]
    Jm = 30.0       # Parametro di locking di Gent

    C10 = 0.1       # [MPa]
    C01 = 0.1       # [MPa]

    P_i_max = 0.05  # Pressione massima target [MPa]
    num_punti = 40
    pressioni = np.linspace(0.0, P_i_max, num_punti)

    r_i_gent, r_e_gent, pressioni_gent = [], [], []
    r_i_mr, r_e_mr, pressioni_mr = [], [], []

    C_gent, C_mr = 0.0, 0.0

    print("Calcolo in corso...")

    # Risoluzione Gent
    for P_i in pressioni:
        try:
            ri, re, C_gent = risolvi_gent(P_i, lambda_z, R_i, R_e, mu, Jm, C_guess=C_gent)
            r_i_gent.append(ri)
            r_e_gent.append(re)
            pressioni_gent.append(P_i)
        except ValueError as err:
            print(f"Gent interrotto a P_i = {P_i:.3f} MPa: {err}")
            break

    # Risoluzione Mooney-Rivlin
    for P_i in pressioni:
        try:
            ri, re, C_mr = risolvi_mooney_rivlin(P_i, lambda_z, R_i, R_e, C10, C01, C_guess=C_mr)
            r_i_mr.append(ri)
            r_e_mr.append(re)
            pressioni_mr.append(P_i)
        except ValueError as err:
            print(f"Mooney-Rivlin interrotto a P_i = {P_i:.3f} MPa: {err}")
            break

    # Plot dei risultati
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(0.03876, color='g', linewidth=2, label='$p_{max}$')
    ax.plot(pressioni_gent, r_i_gent, 'b-', linewidth=2, label='Gent ($r_i$)')
    ax.plot(pressioni_mr, r_i_mr, 'r-', linewidth=2, label='Mooney-Rivlin ($r_i$)')
    ax.set_xlabel('Pressione Interna $P_i$ [MPa]')
    ax.set_ylabel('Raggio Deformato Interno $r_i$ [mm]')
    ax.set_title('Confronto modelli fino a 0.05 MPa')
    ax.grid(True, linestyle='--')
    ax.legend()
    plt.tight_layout()
    plt.show()