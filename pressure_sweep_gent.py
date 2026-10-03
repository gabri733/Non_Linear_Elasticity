import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
import warnings
import matplotlib.pyplot as plt

def calcola_tensioni_deviatriche(lambda_theta, lambda_z, mu, Jm):
    """Calcola le tensioni deviatriche in forma vettorizzata NumPy con protezione da singolarità."""
    lambda_r = 1.0 / (lambda_theta * lambda_z)
    B_r = lambda_r**2
    B_t = lambda_theta**2
    B_z = lambda_z**2

    I1 = B_r + B_t + B_z
    margine = Jm - (I1 - 3.0)

    # Evitiamo la divisione per zero con un margine minimo
    margine_safe = np.maximum(margine, 1e-12)
    dWdI1 = (mu * Jm) / (2.0 * margine_safe)
    
    tau_rr = 2.0 * dWdI1 * B_r
    tau_tt = 2.0 * dWdI1 * B_t
    tau_zz = 2.0 * dWdI1 * B_z

    return tau_rr, tau_tt, tau_zz, margine

def risolvi_cilindro_gent_FEM_consistent(P_i, lambda_z, R_i, R_e, mu, Jm, C_guess=0.0, verbose=False):
    """
    Risolve il problema del cilindro iperelastico spesso in pressione
    usando la quadratura ad alta stabilità numerica.
    """
    if abs(P_i) < 1e-12:
        r_grid = np.linspace(R_i / np.sqrt(lambda_z), R_e / np.sqrt(lambda_z), 1000)
        R_grid = r_grid * np.sqrt(lambda_z)
        sigma_rr = np.zeros_like(r_grid)
        tau_rr, tau_tt, tau_zz, margine = calcola_tensioni_deviatriche(r_grid / R_grid, lambda_z, mu, Jm)
        p_hydro = tau_rr - sigma_rr
        return crea_dizionario_risultati(r_grid, R_grid, sigma_rr, p_hydro, tau_tt, tau_zz, margine, Jm, lambda_z, 0.0, R_i, R_e, verbose)

    def integrand(r, C):
        r2 = r**2
        R2 = lambda_z * (r2 - 2.0 * C)
        if R2 <= 0:
            return 1e10 # Penalizzazione numerica invece di NaN
        
        lambda_theta = r / np.sqrt(R2)
        lambda_r = 1.0 / (lambda_theta * lambda_z)
        
        I1 = lambda_r**2 + lambda_theta**2 + lambda_z**2
        margine = Jm - (I1 - 3.0)
        
        if margine <= 1e-8:
            return 1e10 # Penalizzazione controllata vicino al locking
            
        dWdI1 = (mu * Jm) / (2.0 * margine)
        d_tau = 2.0 * dWdI1 * (lambda_theta**2 - lambda_r**2)
        return d_tau / r

    def residuo_pressione(C):
        r_i2 = (R_i**2) / lambda_z + 2.0 * C
        r_e2 = (R_e**2) / lambda_z + 2.0 * C

        if r_i2 <= 0 or r_e2 <= 0:
            return 1e10

        r_i = np.sqrt(r_i2)
        r_e = np.sqrt(r_e2)

        # Quadratura numerica con tolleranze bilanciate e gestione avvisi
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val, _ = quad(integrand, r_i, r_e, args=(C,), epsabs=1e-7, epsrel=1e-6, limit=200)
        
        return val - P_i

    # 1. Ricerca del bracket di convergenza
    C_start = max(0.0, C_guess)
    res_start = residuo_pressione(C_start)
    
    if res_start >= 1e9:
        raise ValueError("Stato di partenza invalido per estensibilità limite.")

    C_valido = C_start
    step = 0.1 if C_start == 0.0 else max(C_start * 0.05, 0.01)
    C_non_valido = None
    
    for _ in range(60):
        C_prova = C_valido + step
        res = residuo_pressione(C_prova)
        if res >= 1e9 or np.isnan(res):
            C_non_valido = C_prova
            break
        C_valido = C_prova
        step *= 1.8

    if C_non_valido is not None:
        # Bisezione rapida per stringere la soglia di locking
        for _ in range(25):
            C_mid = 0.5 * (C_valido + C_non_valido)
            if residuo_pressione(C_mid) >= 1e9:
                C_non_valido = C_mid
            else:
                C_valido = C_mid
        C_max = C_valido * (1.0 - 1e-6)
    else:
        C_max = C_valido

    res_max = residuo_pressione(C_max)
    if res_max < 0:
        raise ValueError(
            f"Pressione P_i={P_i:.4f} MPa non raggiungibile (superata estensibilità limite Jm={Jm})."
        )

    # 2. Soluzione del valore di C
    sol = root_scalar(residuo_pressione, bracket=[C_start, C_max], method='brentq', xtol=1e-8)
    if not sol.converged:
        raise ValueError("Mancata convergenza nella ricerca della radice di C.")

    C_ottimale = sol.root

    # 3. Calcolo veloce del profilo con trapezi su griglia fine (evita loop di quad)
    r_i = np.sqrt((R_i**2) / lambda_z + 2.0 * C_ottimale)
    r_e = np.sqrt((R_e**2) / lambda_z + 2.0 * C_ottimale)
    r_grid = np.linspace(r_i, r_e, 1000)
    R_grid = np.sqrt(lambda_z * (r_grid**2 - 2.0 * C_ottimale))

    # Integrazione vettoriale cumulativa
    y_integrand = np.array([integrand(r, C_ottimale) for r in r_grid])
    sigma_rr = -P_i + np.concatenate([[0], np.cumsum(0.5 * (y_integrand[:-1] + y_integrand[1:]) * np.diff(r_grid))])

    lambda_theta = r_grid / R_grid
    tau_rr, tau_tt, tau_zz, margine = calcola_tensioni_deviatriche(lambda_theta, lambda_z, mu, Jm)
    p_hydro = tau_rr - sigma_rr

    return crea_dizionario_risultati(r_grid, R_grid, sigma_rr, p_hydro, tau_tt, tau_zz, margine, Jm, lambda_z, C_ottimale, R_i, R_e, verbose)

def crea_dizionario_risultati(r_grid, R_grid, sigma_rr, p_hydro, tau_tt, tau_zz, margine, Jm, lambda_z, C_ottimale, R_i, R_e, verbose):
    lambda_theta = r_grid / R_grid
    lambda_r = 1.0 / (lambda_theta * lambda_z)

    if verbose:
        print("=" * 70)
        print(f"       RISULTATI RIGOROSI - MODELLO DI GENT (Jm={Jm})")
        print("=" * 70)
        print(f"Costante C ricavata           : {C_ottimale:.6f} mm^2")
        print(f"Raggio interno finale r_i     : {r_grid[0]:.4f} mm (iniziale R_i = {R_i:.2f} mm)")
        print(f"Raggio esterno finale r_e     : {r_grid[-1]:.4f} mm (iniziale R_e = {R_e:.2f} mm)")
        print("=" * 70)

    return {
        'r': r_grid,
        'sigma_rr': sigma_rr,
        'sigma_theta': -p_hydro + tau_tt,
        'sigma_zz': -p_hydro + tau_zz,
        'p_hydro': p_hydro,
        'epsilon_r': np.log(lambda_r),
        'epsilon_theta': np.log(lambda_theta),
        'epsilon_z': np.full_like(r_grid, np.log(lambda_z)),
        'margine_locking': margine / Jm,
        'C': C_ottimale
    }

if __name__ == "__main__":
    R_i = 10.0
    R_e = 11.0
    lambda_z = 1.2
    mu = 0.4
    Jm = 30.0

    P_i_min = 0.0
    P_i_max = 1.80
    num_punti = 20

    vett_pressioni = np.linspace(P_i_min, P_i_max, num_punti)
    r_i_deformati = []
    r_e_deformati = []

    print(f"Esecuzione dello sweep di pressione da {P_i_min} a {P_i_max} MPa ({num_punti} punti)...")

    C_last = 0.0

    for P_i in vett_pressioni:
        try:
            res = risolvi_cilindro_gent_FEM_consistent(
                P_i, lambda_z, R_i, R_e, mu, Jm, C_guess=C_last, verbose=False
            )
            r_i_deformati.append(res['r'][0])
            r_e_deformati.append(res['r'][-1])
            C_last = res['C']
        except ValueError as err:
            print(f"Interruzione sweep a P_i = {P_i:.4f} MPa: {err}")
            vett_pressioni = vett_pressioni[:len(r_i_deformati)]
            break

    plt.figure(figsize=(8, 5))
    plt.plot(vett_pressioni, r_i_deformati, '-', color='navy', linewidth=2, label='Raggio interno $r_i$ deformato')
    plt.plot(vett_pressioni, r_e_deformati, '--', color='crimson', linewidth=1.5, alpha=0.7, label='Raggio esterno $r_e$ deformato')

    plt.xlabel('Pressione Interna $P_i$ [MPa]', fontsize=11)
    plt.ylabel('Raggio Deformato $r$ [mm]', fontsize=11)
    plt.title('Andamento Raggio Deformato vs Pressione (Modello Gent)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()