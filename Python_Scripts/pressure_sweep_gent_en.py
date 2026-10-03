import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
import warnings
import matplotlib.pyplot as plt

def compute_deviatoric_stresses(lambda_theta, lambda_z, mu, Jm):
    """Compute deviatoric stresses in vectorized NumPy form with singularity protection."""
    lambda_r = 1.0 / (lambda_theta * lambda_z)
    B_r = lambda_r**2
    B_t = lambda_theta**2
    B_z = lambda_z**2

    # First invariant of left Cauchy-Green tensor
    I1 = B_r + B_t + B_z
    # Margin to locking: must stay > 0
    margin = Jm - (I1 - 3.0)

    # Avoid division by zero with minimum safe margin
    margin_safe = np.maximum(margin, 1e-12)
    dWdI1 = (mu * Jm) / (2.0 * margin_safe)

    # Gent deviatoric stresses (no B^-1 term)
    tau_rr = 2.0 * dWdI1 * B_r
    tau_tt = 2.0 * dWdI1 * B_t
    tau_zz = 2.0 * dWdI1 * B_z

    return tau_rr, tau_tt, tau_zz, margin

def solve_thick_cylinder_gent_FEM_consistent(P_i, lambda_z, R_i, R_e, mu, Jm, C_guess=0.0, verbose=False):
    """
    Solves the hyperelastic thick cylinder problem under internal pressure
    using high numerical stability quadrature.

    Parameters:
      P_i      : Internal pressure [MPa]
      lambda_z : Axial stretch
      R_i, R_e : Reference inner/outer radii [mm]
      mu       : Shear modulus [MPa]
      Jm       : Gent locking parameter
      C_guess  : Initial guess for integration constant
      verbose  : Print summary if True
    """
    # Special case: zero pressure
    if abs(P_i) < 1e-12:
        r_grid = np.linspace(R_i / np.sqrt(lambda_z), R_e / np.sqrt(lambda_z), 1000)
        R_grid = r_grid * np.sqrt(lambda_z)
        sigma_rr = np.zeros_like(r_grid)
        tau_rr, tau_tt, tau_zz, margin = compute_deviatoric_stresses(r_grid / R_grid, lambda_z, mu, Jm)
        p_hydro = tau_rr - sigma_rr
        return create_results_dict(r_grid, R_grid, sigma_rr, p_hydro, tau_tt, tau_zz, margin, Jm, lambda_z, 0.0, R_i, R_e, verbose)

    # Define integrand for equilibrium equation
    def integrand(r, C):
        r2 = r**2
        R2 = lambda_z * (r2 - 2.0 * C)
        if R2 <= 0:
            return 1e10  # Numerical penalty instead of NaN

        lambda_theta = r / np.sqrt(R2)
        lambda_r = 1.0 / (lambda_theta * lambda_z)

        # First invariant
        I1 = lambda_r**2 + lambda_theta**2 + lambda_z**2
        margin = Jm - (I1 - 3.0)

        if margin <= 1e-8:
            return 1e10  # Controlled penalty near locking

        dWdI1 = (mu * Jm) / (2.0 * margin)
        # Difference in tangential and radial stress
        d_tau = 2.0 * dWdI1 * (lambda_theta**2 - lambda_r**2)
        return d_tau / r

    # Define residual function: integrated pressure minus applied pressure
    def pressure_residual(C):
        r_i2 = (R_i**2) / lambda_z + 2.0 * C
        r_e2 = (R_e**2) / lambda_z + 2.0 * C

        if r_i2 <= 0 or r_e2 <= 0:
            return 1e10

        r_i = np.sqrt(r_i2)
        r_e = np.sqrt(r_e2)

        # High-precision numerical quadrature with balanced tolerances
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val, _ = quad(integrand, r_i, r_e, args=(C,), epsabs=1e-7, epsrel=1e-6, limit=200)

        return val - P_i

    # 1. Find convergence bracket for C value
    C_start = max(0.0, C_guess)
    res_start = pressure_residual(C_start)

    if res_start >= 1e9:
        raise ValueError("Invalid starting state for extensibility limit.")

    C_valid = C_start
    step = 0.1 if C_start == 0.0 else max(C_start * 0.05, 0.01)
    C_invalid = None

    # Exponential search for bracket
    for _ in range(60):
        C_try = C_valid + step
        res = pressure_residual(C_try)
        if res >= 1e9 or np.isnan(res):
            C_invalid = C_try
            break
        C_valid = C_try
        step *= 1.8

    if C_invalid is not None:
        # Quick bisection to tighten locking threshold
        for _ in range(25):
            C_mid = 0.5 * (C_valid + C_invalid)
            if pressure_residual(C_mid) >= 1e9:
                C_invalid = C_mid
            else:
                C_valid = C_mid
        C_max = C_valid * (1.0 - 1e-6)
    else:
        C_max = C_valid

    res_max = pressure_residual(C_max)
    if res_max < 0:
        raise ValueError(
            f"Pressure P_i={P_i:.4f} MPa is not achievable (exceeded extensibility limit Jm={Jm})."
        )

    # 2. Solve for optimal C value
    sol = root_scalar(pressure_residual, bracket=[C_start, C_max], method='brentq', xtol=1e-8)
    if not sol.converged:
        raise ValueError("Failed to converge in root search for C.")

    C_optimal = sol.root

    # 3. Quick profile computation using trapezoidal integration on fine grid (avoids quad loop)
    r_i = np.sqrt((R_i**2) / lambda_z + 2.0 * C_optimal)
    r_e = np.sqrt((R_e**2) / lambda_z + 2.0 * C_optimal)
    r_grid = np.linspace(r_i, r_e, 1000)
    R_grid = np.sqrt(lambda_z * (r_grid**2 - 2.0 * C_optimal))

    # Vectorized cumulative integration
    y_integrand = np.array([integrand(r, C_optimal) for r in r_grid])
    sigma_rr = -P_i + np.concatenate([[0], np.cumsum(0.5 * (y_integrand[:-1] + y_integrand[1:]) * np.diff(r_grid))])

    lambda_theta = r_grid / R_grid
    tau_rr, tau_tt, tau_zz, margin = compute_deviatoric_stresses(lambda_theta, lambda_z, mu, Jm)
    p_hydro = tau_rr - sigma_rr

    return create_results_dict(r_grid, R_grid, sigma_rr, p_hydro, tau_tt, tau_zz, margin, Jm, lambda_z, C_optimal, R_i, R_e, verbose)

def create_results_dict(r_grid, R_grid, sigma_rr, p_hydro, tau_tt, tau_zz, margin, Jm, lambda_z, C_optimal, R_i, R_e, verbose):
    """Package results into dictionary with optional terminal output."""
    lambda_theta = r_grid / R_grid
    lambda_r = 1.0 / (lambda_theta * lambda_z)

    if verbose:
        print("=" * 70)
        print(f"       RIGOROUS RESULTS - GENT MODEL (Jm={Jm})")
        print("=" * 70)
        print(f"Computed constant C              : {C_optimal:.6f} mm^2")
        print(f"Final inner radius r_i           : {r_grid[0]:.4f} mm (initial R_i = {R_i:.2f} mm)")
        print(f"Final outer radius r_e           : {r_grid[-1]:.4f} mm (initial R_e = {R_e:.2f} mm)")
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
        'locking_margin': margin / Jm,
        'C': C_optimal
    }

if __name__ == "__main__":
    # Cylinder geometry
    R_i = 10.0
    R_e = 11.0
    lambda_z = 1.2

    # Gent material parameters
    mu = 0.4        # Shear modulus [MPa]
    Jm = 30.0       # Locking parameter

    # ---------------------------------------------------------
    # PRESSURE SWEEP
    # ---------------------------------------------------------
    P_i_min = 0.0
    P_i_max = 1.80
    num_points = 20

    pressure_array = np.linspace(P_i_min, P_i_max, num_points)
    deformed_r_i = []
    deformed_r_e = []

    print(f"Running pressure sweep from {P_i_min} to {P_i_max} MPa ({num_points} points)...")

    C_last = 0.0

    # Solve for each pressure level, using previous C as initial guess
    for P_i in pressure_array:
        try:
            res = solve_thick_cylinder_gent_FEM_consistent(
                P_i, lambda_z, R_i, R_e, mu, Jm, C_guess=C_last, verbose=False
            )
            deformed_r_i.append(res['r'][0])
            deformed_r_e.append(res['r'][-1])
            C_last = res['C']
        except ValueError as err:
            print(f"Sweep interrupted at P_i = {P_i:.4f} MPa: {err}")
            pressure_array = pressure_array[:len(deformed_r_i)]
            break

    # Plot pressure-radius relationship
    plt.figure(figsize=(8, 5))
    plt.plot(pressure_array, deformed_r_i, '-', color='navy', linewidth=2, label='Deformed inner radius $r_i$')
    plt.plot(pressure_array, deformed_r_e, '--', color='crimson', linewidth=1.5, alpha=0.7, label='Deformed outer radius $r_e$')

    plt.xlabel('Internal Pressure $P_i$ [MPa]', fontsize=11)
    plt.ylabel('Deformed Radius $r$ [mm]', fontsize=11)
    plt.title('Deformed Radius vs Pressure (Gent Model)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
