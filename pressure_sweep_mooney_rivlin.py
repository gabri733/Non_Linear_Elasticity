import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

def risolvi_cilindro_mooney_rivlin_FEM_consistent(P_i, lambda_z, R_i, R_e, C10, C01, verbose=False):
    """
    Risolve analiticamente/numericamente il problema del cilindro iperelastico
    con formulazione totalmente coerente ai solutori FEM (es. Abaqus/ANSYS).
    """
    
    # 1. Definizione delle tensioni deviatrici in funzione di lambda_theta e lambda_z
    def calcola_tensioni_deviatrici(lambda_theta):
        # Incomprimibilità: lambda_r * lambda_theta * lambda_z = 1
        lambda_r = 1.0 / (lambda_theta * lambda_z)
        
        # B_rr, B_tt, B_zz
        B_r = lambda_r**2
        B_t = lambda_theta**2
        B_z = lambda_z**2
        
        # B^-1
        B_r_inv = 1.0 / B_r
        B_t_inv = 1.0 / B_t
        B_z_inv = 1.0 / B_z
        
        # Tensioni deviatrici (senza la pressione idrostatica -p)
        tau_rr = 2.0 * C10 * B_r - 2.0 * C01 * B_r_inv
        tau_tt = 2.0 * C10 * B_t - 2.0 * C01 * B_t_inv
        tau_zz = 2.0 * C10 * B_z - 2.0 * C01 * B_z_inv
        
        return tau_rr, tau_tt, tau_zz

    # 2. Integrale di equilibrio indefinito risolto numericamente lungo r per accuratezza FEM
    def calcola_profilo_sigma_rr(C, num_points=500):
        # Raggi deformati ai contorni
        r_i2 = (R_i**2) / lambda_z + 2.0 * C
        r_e2 = (R_e**2) / lambda_z + 2.0 * C
        
        if r_i2 <= 0 or r_e2 <= 0:
            return None, None, None
            
        r_i = np.sqrt(r_i2)
        r_e = np.sqrt(r_e2)
        
        r_grid = np.linspace(r_i, r_e, num_points)
        R_grid = np.sqrt(lambda_z * (r_grid**2 - 2.0 * C))
        
        lambda_theta = r_grid / R_grid
        
        tau_rr, tau_tt, _ = calcola_tensioni_deviatrici(lambda_theta)
        
        # Integrand: d(sigma_rr)/dr = (tau_tt - tau_rr) / r
        integrand = (tau_tt - tau_rr) / r_grid
        
        cum_integral = cumulative_trapezoid(integrand, r_grid, initial=0.0)
        sigma_rr = -P_i + cum_integral
        
        return r_grid, R_grid, sigma_rr

    # 3. Residuo per trovare C tale che la parete esterna sia libera da carico: sigma_rr(r_e) = 0
    def residuo_pressione(C):
        r_grid, _, sigma_rr = calcola_profilo_sigma_rr(C)
        if sigma_rr is None:
            return np.nan
        return sigma_rr[-1]

    # Ricerca radice per la costante C
    C_min = - (R_i**2) / (2.0 * lambda_z) + 1e-5
    C_max = 50.0 * (R_e**2)
    
    sol = root_scalar(residuo_pressione, bracket=[C_min, C_max], method='brentq')
    if not sol.converged:
        raise ValueError("Non converge alla soluzione fisica.")
        
    C_ottimale = sol.root
    
    # 4. Calcolo finale dei campi coerenti
    r_grid, R_grid, sigma_rr = calcola_profilo_sigma_rr(C_ottimale, num_points=1000)
    
    lambda_theta = r_grid / R_grid
    lambda_r = 1.0 / (lambda_theta * lambda_z)
    
    tau_rr, tau_tt, tau_zz = calcola_tensioni_deviatrici(lambda_theta)
    p_hydro = tau_rr - sigma_rr
    sigma_theta = -p_hydro + tau_tt
    sigma_zz = -p_hydro + tau_zz
    
    epsilon_r = np.log(lambda_r)
    epsilon_theta = np.log(lambda_theta)
    epsilon_z = np.log(lambda_z) * np.ones_like(r_grid)
    
    if verbose:
        print("=" * 70)
        print("      RISULTATI RIGOROSI (COERENTI CON SOLUTORE FEM / ABAQUS)")
        print("=" * 70)
        print(f"Costante C ricavata           : {C_ottimale:.6f} mm^2")
        print(f"Raggio interno finale r_i     : {r_grid[0]:.4f} mm (iniziale R_i = {R_i:.2f} mm)")
        print(f"Raggio esterno finale r_e     : {r_grid[-1]:.4f} mm (iniziale R_e = {R_e:.2f} mm)")
        print("=" * 70)
    
    return {
        'r': r_grid, 'sigma_rr': sigma_rr, 'sigma_theta': sigma_theta, 
        'sigma_zz': sigma_zz, 'p_hydro': p_hydro,
        'epsilon_r': epsilon_r, 'epsilon_theta': epsilon_theta, 'epsilon_z': epsilon_z
    }

if __name__ == "__main__":
    R_i = 10.0      # Raggio interno [mm]
    R_e = 11.0      # Raggio esterno [mm]
    lambda_z = 1.2  # Stretch assiale
    
    C10 = 0.1
    C01 = 0.1
    
    # ---------------------------------------------------------
    # PARAMETRI DELLO SWEEP DI PRESSIONE INTERNA
    # ---------------------------------------------------------
    P_i_min = 0.0        # Pressione minima [MPa]
    P_i_max = 0.038      # Pressione massima [MPa]
    num_punti = 30       # Numero di punti per lo sweep
    
    vett_pressioni = np.linspace(P_i_min, P_i_max, num_punti)
    r_i_deformati = []
    r_e_deformati = []

    print(f"Esecuzione dello sweep di pressione da {P_i_min} a {P_i_max} MPa ({num_punti} punti)...")
    
    for P_i in vett_pressioni:
        res = risolvi_cilindro_mooney_rivlin_FEM_consistent(
            P_i, lambda_z, R_i, R_e, C10, C01, verbose=False
        )
        r_i_deformati.append(res['r'][0])      # Raggio interno deformato r_i
        r_e_deformati.append(res['r'][-1])     # Raggio esterno deformato r_e

    # ---------------------------------------------------------
    # PLOT: PRESSIONE INTERNA VS RAGGIO INTERNO DEFORMATO
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(vett_pressioni, r_i_deformati, '-', color='navy', linewidth=2, label='Raggio interno $r_i$ deformato')
    plt.plot(vett_pressioni, r_e_deformati, '--', color='crimson', linewidth=1.5, alpha=0.7, label='Raggio esterno $r_e$ deformato')
    

    plt.xlabel('Pressione Interna $P_i$ [MPa]', fontsize=11)
    plt.ylabel('Raggio Deformato $r$ [mm]', fontsize=11)
    plt.title('Andamento del Raggio Deformato al variare della Pressione Interna', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()