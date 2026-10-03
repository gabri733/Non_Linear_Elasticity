import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

def risolvi_cilindro_mooney_rivlin_FEM_consistent(P_i, lambda_z, R_i, R_e, C10, C01):
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
        
        # Integrazione numerica cumulata partendo dal raggio interno r_i
        # sigma_rr(r) = sigma_rr(r_i) + int_{r_i}^r (tau_tt - tau_rr)/r dr
        # Imponiamo inizialmente sigma_rr(r_i) = -P_i

        cum_integral = cumulative_trapezoid(integrand, r_grid, initial=0.0)
        sigma_rr = -P_i + cum_integral
        
        return r_grid, R_grid, sigma_rr

    # 3. Residuo per trovare C tale che la parete esterna sia libera da carico: sigma_rr(r_e) = 0
    def residuo_pressione(C):
        r_grid, _, sigma_rr = calcola_profilo_sigma_rr(C)
        if sigma_rr is None:
            return np.nan
        # Vogliamo che la tensione radiale sul bordo esterno sia 0
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
    
    # Ricavo p(r) esatto da: sigma_rr = -p + tau_rr  ==>  p = tau_rr - sigma_rr
    p_hydro = tau_rr - sigma_rr
    
    # Tensioni di Cauchy totali
    sigma_theta = -p_hydro + tau_tt
    sigma_zz = -p_hydro + tau_zz
    
    # Deformazioni Reali (True Logarithmic Strain, lo standard FEM)
    epsilon_r = np.log(lambda_r)
    epsilon_theta = np.log(lambda_theta)
    epsilon_z = np.log(lambda_z) * np.ones_like(r_grid)
    
    # Deformazioni Ingegneristiche (Engineering Strain per confronto extra con FEM)
    e_r = lambda_r - 1.0
    e_theta = lambda_theta - 1.0
    e_z = lambda_z - 1.0
    
    # STAMPA RISULTATI SUL TERMINALE
    print("=" * 70)
    print("      RISULTATI RIGOROSI (COERENTI CON SOLUTORE FEM / ABAQUS)")
    print("=" * 70)
    print(f"Costante C ricavata           : {C_ottimale:.6f} mm^2")
    print(f"Raggio interno finale r_i     : {r_grid[0]:.4f} mm (iniziale R_i = {R_i:.2f} mm)")
    print(f"Raggio esterno finale r_e     : {r_grid[-1]:.4f} mm (iniziale R_e = {R_e:.2f} mm)")
    print("-" * 70)
    print(f"{'PARAMETRO / TENSIONE [MPa]':<32} | {'INTERNO (r = r_i)':<15} | {'ESTERNO (r = r_e)':<15}")
    print("-" * 70)
    print(f"{'Tensione Radiale (sigma_rr)':<32} | {sigma_rr[0]:15.6f} | {sigma_rr[-1]:15.6f}")
    print(f"{'Tensione Circonferenziale (sigma_t)':<32} | {sigma_theta[0]:15.6f} | {sigma_theta[-1]:15.6f}")
    print(f"{'Tensione Assiale (sigma_zz)':<32} | {sigma_zz[0]:15.6f} | {sigma_zz[-1]:15.6f}")
    print(f"{'Pressione Idrostatica p(r)':<32} | {p_hydro[0]:15.6f} | {p_hydro[-1]:15.6f}")
    print("-" * 70)
    print(f"{'DEFORMAZIONE REALE (LE / True)':<32} | {'INTERNO (r = r_i)':<15} | {'ESTERNO (r = r_e)':<15}")
    print("-" * 70)
    print(f"{'Deformazione Radiale (LE_r)':<32} | {epsilon_r[0]:15.6f} | {epsilon_r[-1]:15.6f}")
    print(f"{'Deformazione Circonferenziale (LE_t)':<32} | {epsilon_theta[0]:15.6f} | {epsilon_theta[-1]:15.6f}")
    print(f"{'Deformazione Assiale (LE_z)':<32} | {epsilon_z[0]:15.6f} | {epsilon_z[-1]:15.6f}")
    print("-" * 70)
    print(f"{'DEFORMAZIONE INGEGNERISTICA (E)':<32} | {'INTERNO (r = r_i)':<15} | {'ESTERNO (r = r_e)':<15}")
    print("-" * 70)
    print(f"{'Def. Ing. Radiale (E_r)':<32} | {e_r[0]:15.6f} | {e_r[-1]:15.6f}")
    print(f"{'Def. Ing. Circonferenziale (E_t)':<32} | {e_theta[0]:15.6f} | {e_theta[-1]:15.6f}")
    print("=" * 70)
    
    return {
        'r': r_grid, 'sigma_rr': sigma_rr, 'sigma_theta': sigma_theta, 
        'sigma_zz': sigma_zz, 'p_hydro': p_hydro,
        'epsilon_r': epsilon_r, 'epsilon_theta': epsilon_theta, 'epsilon_z': epsilon_z
    }

if __name__ == "__main__":
    R_i = 10.0      # Raggio interno [mm]
    R_e = 11.0      # Raggio esterno [mm]
    P_i = 0.035     # Pressione interna [MPa]
    lambda_z = 0.8  # Stretch assiale
    
    C10 = 0.1
    C01 = 0.1
    
    res = risolvi_cilindro_mooney_rivlin_FEM_consistent(P_i, lambda_z, R_i, R_e, C10, C01)

    # ---------------------------------------------------------
    # GENERAZIONE DEI GRAFICI (TENSIONI E DEFORMAZIONI REALI)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Grafico 1: Tensioni di Cauchy
    ax1.plot(res['r'], res['sigma_rr'], label=r'$\sigma_{r}$ (Radiale)', color='blue', linewidth=2)
    ax1.plot(res['r'], res['sigma_theta'], label=r'$\sigma_{\theta}$ (Circonferenziale)', color='red', linewidth=2)
    ax1.plot(res['r'], res['sigma_zz'], label=r'$\sigma_{z}$ (Assiale)', color='green', linewidth=2)
    ax1.set_xlabel('Raggio deformato $r$ [mm]', fontsize=11)
    ax1.set_ylabel('Tensione [MPa]', fontsize=11)
    ax1.set_title('Profilo delle Tensioni di Cauchy', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    # Grafico 2: Deformazioni Reali (True Logarithmic Strain)
    ax2.plot(res['r'], res['epsilon_r'], label=r'$\varepsilon_r$', color='blue', linewidth=2)
    ax2.plot(res['r'], res['epsilon_theta'], label=r'$\varepsilon_\theta$', color='red', linewidth=2)
    ax2.plot(res['r'], res['epsilon_z'], label=r'$\varepsilon_z$', color='green', linewidth=2)
    ax2.set_xlabel('Raggio deformato $r$ [mm]', fontsize=11)
    ax2.set_ylabel('Deformazione Reale [-]', fontsize=11)
    ax2.set_title('Profilo delle Deformazioni Reali (True Strain)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()