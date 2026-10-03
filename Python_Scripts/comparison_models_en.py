import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Select appropriate trapezoid function for compatibility with different numpy versions
trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# =============================================================================
# 1. GENT MODEL SOLVER
# =============================================================================
def solve_gent(P_i, lambda_z, R_i, R_e, mu, Jm, C_guess=0.0, n_pts=300):
    """
    Solves thick cylinder with Gent hyperelastic model.

    Parameters:
      P_i      : Internal pressure [MPa]
      lambda_z : Axial stretch
      R_i, R_e : Reference inner and outer radii [mm]
      mu       : Shear modulus [MPa]
      Jm       : Gent locking parameter
      C_guess  : Initial guess for integration constant
      n_pts    : Number of quadrature points

    Returns:
      r_i_deformed, r_e_deformed, C_optimal
    """
    # Zero pressure case
    if abs(P_i) < 1e-12:
        return R_i / np.sqrt(lambda_z), R_e / np.sqrt(lambda_z), 0.0

    # Compute required pressure for given C value
    def compute_gent_pressure(C):
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

        # First invariant
        I1 = lambda_r**2 + lambda_theta**2 + lambda_z**2
        margin = Jm - (I1 - 3.0)

        if np.any(margin <= 1e-6):
            return -1e10  # Kinematic locking reached

        dWdI1 = (mu * Jm) / (2.0 * margin)
        # Integrand: difference between tangential and radial stress divided by radius
        tau_diff = 2.0 * dWdI1 * (lambda_theta**2 - lambda_r**2)

        return trapz_func(tau_diff / r, r)

    # Residual function: computed pressure minus applied pressure
    def residual(C):
        P_calc = compute_gent_pressure(C)
        if P_calc < 0:
            return -1e10
        return P_calc - P_i

    # Find bracket for root search
    C_low = 0.0

    # Robust search for C_high bracket
    step = 0.5 if C_guess == 0.0 else max(C_guess * 0.05, 0.1)
    C_high = max(C_guess, 0.01)

    found = False
    for _ in range(200):
        res = residual(C_high)
        if res >= 0:
            found = True
            break
        elif res < -1e8:  # Exceeded locking/validity limit
            # Back up and reduce step to approach locking wall
            C_high -= step
            step *= 0.5
            if step < 1e-6:
                break
        C_high += step

    if not found or residual(C_high) < 0:
        raise ValueError("Pressure exceeds material locking limit.")

    # Solve for C using Brent's method
    sol = root_scalar(residual, bracket=[C_low, C_high], method='brentq', xtol=1e-7)
    C_opt = sol.root
    r_i_def = np.sqrt((R_i**2) / lambda_z + 2.0 * C_opt)
    r_e_def = np.sqrt((R_e**2) / lambda_z + 2.0 * C_opt)
    return r_i_def, r_e_def, C_opt


# =============================================================================
# 2. MOONEY-RIVLIN MODEL SOLVER
# =============================================================================
def solve_mooney_rivlin(P_i, lambda_z, R_i, R_e, C10, C01, C_guess=0.0, n_pts=300):
    """
    Solves thick cylinder with Mooney-Rivlin hyperelastic model.

    Parameters:
      P_i      : Internal pressure [MPa]
      lambda_z : Axial stretch
      R_i, R_e : Reference inner and outer radii [mm]
      C10, C01 : Mooney-Rivlin material constants [MPa]
      C_guess  : Initial guess for integration constant
      n_pts    : Number of quadrature points

    Returns:
      r_i_deformed, r_e_deformed, C_optimal
    """
    # Zero pressure case
    if abs(P_i) < 1e-12:
        return R_i / np.sqrt(lambda_z), R_e / np.sqrt(lambda_z), 0.0

    # Compute required pressure for given C value
    def compute_mr_pressure(C):
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

        # Mooney-Rivlin deviatoric stress difference
        tau_diff = 2.0 * (C10 + C01 * lambda_z**2) * (lambda_theta**2 - lambda_r**2)

        return trapz_func(tau_diff / r, r)

    # Residual function
    def residual(C):
        P_calc = compute_mr_pressure(C)
        if P_calc < 0:
            return -1e10
        return P_calc - P_i

    # Search for bracket
    C_low = 0.0
    C_high = max(1.0, C_guess * 1.2)
    step = 0.5

    found = False
    for _ in range(100):
        if residual(C_high) >= 0:
            found = True
            break
        C_high += step
        step *= 1.2

    if not found:
        raise ValueError("Cannot find interval for Mooney-Rivlin solver.")

    # Solve for C
    sol = root_scalar(residual, bracket=[C_low, C_high], method='brentq', xtol=1e-7)
    C_opt = sol.root
    r_i_def = np.sqrt((R_i**2) / lambda_z + 2.0 * C_opt)
    r_e_def = np.sqrt((R_e**2) / lambda_z + 2.0 * C_opt)
    return r_i_def, r_e_def, C_opt


# =============================================================================
# 3. PRESSURE SWEEP AND COMPARISON
# =============================================================================
if __name__ == "__main__":
    # Cylinder geometry
    R_i = 10.0      # Inner radius [mm]
    R_e = 11.0      # Outer radius [mm]
    lambda_z = 1.2  # Axial stretch

    # Gent material parameters
    mu = 0.4        # Shear modulus [MPa]
    Jm = 30.0       # Locking parameter

    # Mooney-Rivlin material parameters [MPa]
    # Note: to compare at small strains, use mu ≈ 2*(C10+C01)
    C10 = 0.1
    C01 = 0.1

    # Pressure sweep parameters
    P_i_max = 0.05  # Target maximum pressure [MPa]
    num_points = 40
    pressure_array = np.linspace(0.0, P_i_max, num_points)

    # Storage for results
    r_i_gent, r_e_gent, pressures_gent = [], [], []
    r_i_mr, r_e_mr, pressures_mr = [], [], []

    # Integration constants for continuation
    C_gent, C_mr = 0.0, 0.0

    print("Computing models...")

    # Solve Gent model for each pressure
    for P_i in pressure_array:
        try:
            ri, re, C_gent = solve_gent(P_i, lambda_z, R_i, R_e, mu, Jm, C_guess=C_gent)
            r_i_gent.append(ri)
            r_e_gent.append(re)
            pressures_gent.append(P_i)
        except ValueError as err:
            print(f"Gent stopped at P_i = {P_i:.3f} MPa: {err}")
            break

    # Solve Mooney-Rivlin model for each pressure
    for P_i in pressure_array:
        try:
            ri, re, C_mr = solve_mooney_rivlin(P_i, lambda_z, R_i, R_e, C10, C01, C_guess=C_mr)
            r_i_mr.append(ri)
            r_e_mr.append(re)
            pressures_mr.append(P_i)
        except ValueError as err:
            print(f"Mooney-Rivlin stopped at P_i = {P_i:.3f} MPa: {err}")
            break

    # Generate comparison plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(0.03876, color='g', linewidth=2, label='$p_{max}$')
    ax.plot(pressures_gent, r_i_gent, 'b-', linewidth=2, label='Gent ($r_i$)')
    ax.plot(pressures_mr, r_i_mr, 'r-', linewidth=2, label='Mooney-Rivlin ($r_i$)')
    ax.set_xlabel('Internal Pressure $P_i$ [MPa]')
    ax.set_ylabel('Deformed Inner Radius $r_i$ [mm]')
    ax.set_title('Model Comparison up to 0.05 MPa')
    ax.grid(True, linestyle='--')
    ax.legend()
    plt.tight_layout()
    plt.show()
