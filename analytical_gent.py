import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def risolvi_cilindro_gent_FEM_consistent(P_i, lambda_z, R_i, R_e, mu, Jm):
    """
    Risolve il problema del cilindro iperelastico spesso in pressione,
    usando il modello di Gent (estensibilità limite delle catene),
    con la stessa formulazione (equilibrio integrato in forma esatta,
    deformazioni logaritmiche) coerente con i solutori FEM.

    Parametri materiale:
      mu  : modulo di taglio iniziale (piccole deformazioni), analogo a
            2*(C10+C01) nel modello di Mooney-Rivlin
      Jm  : parametro di "locking" di Gent, legato all'estensibilità massima
            delle catene: la tensione diverge quando I1 - 3 -> Jm.
            Valori tipici per gomme: Jm ~ 20-120 (più piccolo = si irrigidisce prima)

    A differenza di Mooney-Rivlin (che dipende da I1 e I2), il modello di
    Gent dipende SOLO da I1 = tr(B); quindi la tensione deviatorica non ha
    il termine in B^-1: tau = 2*dW/dI1 * B.
    """

    # 1. Tensioni deviatoriche di Gent in funzione di lambda_theta e lambda_z
    def calcola_tensioni_deviatriche(lambda_theta):
        # Incomprimibilità: lambda_r * lambda_theta * lambda_z = 1
        lambda_r = 1.0 / (lambda_theta * lambda_z)

        B_r = lambda_r**2
        B_t = lambda_theta**2
        B_z = lambda_z**2

        I1 = B_r + B_t + B_z

        # Margine rispetto al locking: deve restare > 0
        margine = Jm - (I1 - 3.0)

        # dW/dI1 per il modello di Gent: W = -(mu*Jm/2) * ln(1 - (I1-3)/Jm)
        # dW/dI1 = (mu*Jm) / (2*margine)
        dWdI1 = np.where(margine > 0, mu * Jm / (2.0 * np.where(margine > 0, margine, np.nan)), np.nan)

        # Tensioni deviatoriche (senza la pressione idrostatica -p):
        # per Gent (funzione solo di I1): tau = 2*dW/dI1 * B  (nessun termine in B^-1)
        tau_rr = 2.0 * dWdI1 * B_r
        tau_tt = 2.0 * dWdI1 * B_t
        tau_zz = 2.0 * dWdI1 * B_z

        return tau_rr, tau_tt, tau_zz, margine

    # 2. Integrale di equilibrio, risolto come ODE con passo adattivo.
    #    NOTA IMPORTANTE rispetto alla versione Mooney-Rivlin: vicino al
    #    locking di Gent l'integrando (tau_tt - tau_rr)/r cresce ripidissimo
    #    (diverge quando margine -> 0). Una griglia uniforme con integrazione
    #    trapezoidale (come nello script Mooney-Rivlin) NON converge in
    #    quella zona: serve un integratore a passo adattivo che infittisca
    #    automaticamente dove il gradiente è ripido, quindi usiamo solve_ivp.
    def margine_a_raggio(r, C):
        R = np.sqrt(lambda_z * (r**2 - 2.0 * C))
        lambda_theta = r / R
        lambda_r = 1.0 / (lambda_theta * lambda_z)
        I1 = lambda_r**2 + lambda_theta**2 + lambda_z**2
        return Jm - (I1 - 3.0)

    def calcola_profilo_sigma_rr(C, num_points=500):
        r_i2 = (R_i**2) / lambda_z + 2.0 * C
        r_e2 = (R_e**2) / lambda_z + 2.0 * C

        if r_i2 <= 0 or r_e2 <= 0:
            return None, None, None

        r_i = np.sqrt(r_i2)
        r_e = np.sqrt(r_e2)

        # Se il materiale è già al limite di estensibilità in un estremo, scartiamo subito
        if margine_a_raggio(r_i, C) <= 1e-9 or margine_a_raggio(r_e, C) <= 1e-9:
            return None, None, None

        def rhs(r, y):
            R = np.sqrt(lambda_z * (r**2 - 2.0 * C))
            lambda_theta = r / R
            tau_rr, tau_tt, _, _ = calcola_tensioni_deviatriche(lambda_theta)
            return [(tau_tt - tau_rr) / r]

        # Evento: interrompe l'integrazione se lungo il percorso si raggiunge
        # il locking (margine -> 0) prima di arrivare a r_e
        def evento_locking(r, y):
            return margine_a_raggio(r, C) - 1e-9
        evento_locking.terminal = True
        evento_locking.direction = -1

        sol = solve_ivp(rhs, [r_i, r_e], [-P_i], method='RK45',
                         dense_output=True, events=evento_locking,
                         rtol=1e-10, atol=1e-13)

        if sol.status != 0 or not sol.success:
            # l'integrazione si è fermata prima di raggiungere r_e: il
            # materiale ha raggiunto l'estensibilità limite lungo lo spessore
            return None, None, None

        r_grid = np.linspace(r_i, r_e, num_points)
        R_grid = np.sqrt(lambda_z * (r_grid**2 - 2.0 * C))
        sigma_rr = sol.sol(r_grid)[0]

        return r_grid, R_grid, sigma_rr

    # 3. Residuo: la parete esterna deve essere libera da carico, sigma_rr(r_e) = 0
    def residuo_pressione(C):
        r_grid, _, sigma_rr = calcola_profilo_sigma_rr(C)
        if sigma_rr is None:
            return np.nan
        return sigma_rr[-1]

    # 4. Ricerca del limite cinematico C_lock: a differenza di Mooney-Rivlin,
    #    qui NON si può usare un C_max arbitrariamente grande, perché il
    #    modello di Gent smette semplicemente di essere definito quando le
    #    catene raggiungono l'estensibilità limite. Cerchiamo quindi il
    #    più grande C ammissibile con una ricerca esponenziale + bisezione.
    C_start = 0.0  # C=0 corrisponde allo stato "naturale" sotto il solo lambda_z, con P=0
    residuo_start = residuo_pressione(C_start)
    if np.isnan(residuo_start):
        raise ValueError("Anche lo stato non pressurizzato (P_i=0) supera l'estensibilità "
                          "limite del materiale: lambda_z e/o Jm non sono fisicamente compatibili.")

    # ricerca esponenziale del primo C non ammissibile
    C_valido, C_non_valido = C_start, None
    passo = 1.0
    while True:
        C_prova = C_valido + passo
        r_grid, _, sigma_rr = calcola_profilo_sigma_rr(C_prova)
        if sigma_rr is None:
            C_non_valido = C_prova
            break
        C_valido = C_prova
        passo *= 2.0
        if passo > 1e12:
            raise ValueError("Non riesco a individuare il limite di locking: controlla i parametri.")

    # bisezione per affinare C_lock tra C_valido (ammissibile) e C_non_valido (non ammissibile)
    for _ in range(60):
        C_mid = 0.5 * (C_valido + C_non_valido)
        _, _, sigma_rr = calcola_profilo_sigma_rr(C_mid)
        if sigma_rr is None:
            C_non_valido = C_mid
        else:
            C_valido = C_mid

    C_lock = C_valido
    C_max_bracket = C_lock * (1.0 - 1e-6)  # margine di sicurezza appena sotto il locking

    # 5. Ricerca radice per la costante C nel range fisicamente ammissibile
    residuo_hi = residuo_pressione(C_max_bracket)
    if np.isnan(residuo_hi) or np.sign(residuo_start - P_i * 0) == np.sign(residuo_hi):
        # NB: a C_start, sigma_rr(r_e) = -P_i (residuo_start = -P_i), sempre negativo per P_i>0.
        # Se anche al limite di locking il residuo resta negativo, vuol dire che la pressione
        # richiesta eccede quella massima sostenibile PRIMA che il materiale raggiunga
        # l'estensibilità limite delle catene: qui il limite è una vera rottura del materiale,
        # non un'instabilità geometrica come nel caso Mooney-Rivlin.
        r_i_lock = np.sqrt(R_i**2 / lambda_z + 2.0 * C_lock)
        raise ValueError(
            f"Nessuna soluzione ammissibile: per raggiungere P_i={P_i} MPa il materiale dovrebbe "
            f"superare l'estensibilità limite (Jm={Jm}) prima ancora di soddisfare l'equilibrio.\n"
            f"Pressione massima raggiungibile con questi parametri (appena sotto il locking): "
            f"~{residuo_hi + P_i:.5f} MPa (raggiunta per r_i -> {r_i_lock:.3f} mm). "
            f"Aumenta Jm (materiale più estensibile) o riduci P_i."
        )

    sol = root_scalar(residuo_pressione, bracket=[C_start, C_max_bracket], method='brentq')
    if not sol.converged:
        raise ValueError("Non converge alla soluzione fisica.")

    C_ottimale = sol.root

    # Verifica di sicurezza: vicino alla singolarità di Gent la funzione integranda
    # diverge molto rapidamente, e in doppia precisione brentq può occasionalmente
    # convergere su una radice spuria (per gestione di NaN nell'intervallo). Controlliamo
    # esplicitamente che la condizione al contorno sia soddisfatta entro una tolleranza
    # stretta, altrimenti è più onesto segnalare il problema che restituire un risultato
    # silenziosamente inaccurato.
    residuo_finale = residuo_pressione(C_ottimale)
    tolleranza = 1e-6 * max(1.0, abs(P_i))
    if not np.isfinite(residuo_finale) or abs(residuo_finale) > tolleranza:
        raise ValueError(
            f"La soluzione trovata (C={C_ottimale:.4f}) non soddisfa l'equilibrio entro tolleranza "
            f"(residuo sigma_rr(r_e) = {residuo_finale:.3e} MPa, invece di 0). Questo capita quando "
            f"P_i è così vicina al limite teorico di estensibilità (Jm) che la singolarità del "
            f"modello di Gent non è più risolvibile in doppia precisione. Riduci P_i o aumenta Jm."
        )

    # 6. Calcolo finale dei campi coerenti
    r_grid, R_grid, sigma_rr = calcola_profilo_sigma_rr(C_ottimale, num_points=1000)

    lambda_theta = r_grid / R_grid
    lambda_r = 1.0 / (lambda_theta * lambda_z)

    tau_rr, tau_tt, tau_zz, margine = calcola_tensioni_deviatriche(lambda_theta)

    # Ricavo p(r) esatto da: sigma_rr = -p + tau_rr  ==>  p = tau_rr - sigma_rr
    p_hydro = tau_rr - sigma_rr

    # Tensioni di Cauchy totali
    sigma_theta = -p_hydro + tau_tt
    sigma_zz = -p_hydro + tau_zz

    # Deformazioni Reali (True Logarithmic Strain, lo standard FEM)
    epsilon_r = np.log(lambda_r)
    epsilon_theta = np.log(lambda_theta)
    epsilon_z = np.log(lambda_z) * np.ones_like(r_grid)

    # Deformazioni Ingegneristiche
    e_r = lambda_r - 1.0
    e_theta = lambda_theta - 1.0
    e_z = lambda_z - 1.0

    # Margine residuo rispetto al locking (0 = appena sul punto di locking, 1 = stato naturale)
    margine_relativo = margine / Jm

    # STAMPA RISULTATI SUL TERMINALE
    print("=" * 70)
    print("      RISULTATI RIGOROSI - MODELLO DI GENT (locking a Jm={:.2f})".format(Jm))
    print("=" * 70)
    print(f"Costante C ricavata           : {C_ottimale:.6f} mm^2")
    print(f"Raggio interno finale r_i     : {r_grid[0]:.4f} mm (iniziale R_i = {R_i:.2f} mm)")
    print(f"Raggio esterno finale r_e     : {r_grid[-1]:.4f} mm (iniziale R_e = {R_e:.2f} mm)")
    print(f"Margine residuo al locking (min lungo lo spessore): {np.min(margine_relativo)*100:.2f} %  "
          f"(0% = fibra al limite di estensibilità)")
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
        'epsilon_r': epsilon_r, 'epsilon_theta': epsilon_theta, 'epsilon_z': epsilon_z,
        'margine_locking': margine_relativo
    }

if __name__ == "__main__":
    R_i = 10.0      # Raggio interno [mm]
    R_e = 11.0      # Raggio esterno [mm]
    P_i = 1.9   # Pressione interna [MPa]  (con Mooney-Rivlin, C10=C01=0.1, non aveva soluzione)
    lambda_z = 1.2  # Stretch assiale

    # Parametri di Gent: mu equivalente a 2*(C10+C01) del caso Mooney-Rivlin
    # per un confronto diretto a piccole deformazioni; Jm tipico di gomma naturale
    mu = 0.4
    Jm = 30.0

    res = risolvi_cilindro_gent_FEM_consistent(P_i, lambda_z, R_i, R_e, mu, Jm)