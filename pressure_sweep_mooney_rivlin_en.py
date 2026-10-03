import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

def solve_thick_cylinder_mooney_rivlin_FEM_consistent(P_i, lambda_z, R_i, R_e, C10, C01, verbose=False):
    """
    Analytically/numerically solves the hyperelastic cylinder problem
    with formulation fully consistent with FEM solvers (e.g., Abaqus/ANSYS).

    Parameters:
      P_i      : Internal pressure [MPa]
      lambda_z : Axial stretch
      R_i, R_e : Reference inner and outer radii [mm]
      C10, C01 : Mooney-Rivlin material constants [MPa]
      verbose  : Print summary if True
    """

    # 1. Deviatoric stress components as function of circumferential stretch
    def compute_deviatoric_stresses(lambda_theta):
        # Incompressibility: lambda_r * lambda_theta * lambda_z = 1
        lambda_r = 1.0 / (lambda_theta * lambda_z)

        # Left Cauchy-Green tensor components
        B_r = lambda_r**2
        B_t = lambda_theta**2
        B_z = lambda_z**2

        # Inverse tensor components
        B_r_inv = 1.0 / B_r
        B_t_inv = 1.0 / B_t
        B_z_inv = 1.0 / B_z

        # Deviatoric stresses (without hydrostatic pressure)
        tau_rr = 2.0 * C10 * B_r - 2.0 * C01 * B_r_inv
        tau_tt = 2.0 * C10 * B_t - 2.0 * C01 * B_t_inv
        tau_zz = 2.0 * C10 * B_z - 2.0 * C01 * B_z_inv

        return tau_rr, tau_tt, tau_zz

    # 2. Numerical integration of equilibrium along radial direction
    def compute_sigma_rr_profile(C, num_points=500):
        # Deformed radii at boundaries
        r_i2 = (R_i**2) / lambda_z + 2.0 * C
        r_e2 = (R_e**2) / lambda_z + 2.0 * C

        if r_i2 <= 0 or r_e2 <= 0:
            return None, None, None

        r_i = np.sqrt(r_i2)
        r_e = np.sqrt(r_e2)

        # Build radial grid in deformed configuration
        r_grid = np.linspace(r_i, r_e, num_points)
        R_grid = np.sqrt(lambda_z * (r_grid**2 - 2.0 * C))

        # Circumferential stretch at each radius
        lambda_theta = r_grid / R_grid

        # Compute deviatoric stress components
        tau_rr, tau_tt, _ = compute_deviatoric_stresses(lambda_theta)

        # Integrand: derivative of radial stress w.r.t. radius
        integrand = (tau_tt - tau_rr) / r_grid

        # Cumulative trapezoidal integration from inner to outer radius
        cum_integral = cumulative_trapezoid(integrand, r_grid, initial=0.0)
        sigma_rr = -P_i + cum_integral

        return r_grid, R_grid, sigma_rr

    # 3. Define residual for outer boundary condition: sigma_rr(r_e) = 0
    def pressure_residual(C):
        r_grid, _, sigma_rr = compute_sigma_rr_profile(C)
        if sigma_rr is None:
            return np.nan
        return sigma_rr[-1]

    # Root finding for integration constant C
    C_min = - (R_i**2) / (2.0 * lambda_z) + 1e-5
    C_max = 50.0 * (R_e**2)

    sol = root_scalar(pressure_residual, bracket=[C_min, C_max], method='brentq')
    if not sol.converged:
        raise ValueError("Did not converge to physical solution.")

    C_optimal = sol.root

    # 4. Compute final stress and strain fields
    r_grid, R_grid, sigma_rr = compute_sigma_rr_profile(C_optimal, num_points=1000)

    # Stretches along radial direction
    lambda_theta = r_grid / R_grid
    lambda_r = 1.0 / (lambda_theta * lambda_z)

    # All stress components
    tau_rr, tau_tt, tau_zz = compute_deviatoric_stresses(lambda_theta)
    p_hydro = tau_rr - sigma_rr
    sigma_theta = -p_hydro + tau_tt
    sigma_zz = -p_hydro + tau_zz

    # Strains
    epsilon_r = np.log(lambda_r)
    epsilon_theta = np.log(lambda_theta)
    epsilon_z = np.log(lambda_z) * np.ones_like(r_grid)

    if verbose:
        print("=" * 70)
        print("      RIGOROUS RESULTS (CONSISTENT WITH FEM SOLVER / ABAQUS)")
        print("=" * 70)
        print(f"Computed constant C              : {C_optimal:.6f} mm^2")
        print(f"Final inner radius r_i           : {r_grid[0]:.4f} mm (initial R_i = {R_i:.2f} mm)")
        print(f"Final outer radius r_e           : {r_grid[-1]:.4f} mm (initial R_e = {R_e:.2f} mm)")
        print("=" * 70)

    return {
        'r': r_grid, 'sigma_rr': sigma_rr, 'sigma_theta': sigma_theta,
        'sigma_zz': sigma_zz, 'p_hydro': p_hydro,
        'epsilon_r': epsilon_r, 'epsilon_theta': epsilon_theta, 'epsilon_z': epsilon_z
    }

if __name__ == "__main__":
    # Cylinder geometry
    R_i = 10.0      # Inner radius [mm]
    R_e = 11.0      # Outer radius [mm]
    lambda_z = 1.2  # Axial stretch

    # Mooney-Rivlin material parameters [MPa]
    C10 = 0.1
    C01 = 0.1

    # ---------------------------------------------------------
    # INTERNAL PRESSURE SWEEP PARAMETERS
    # ---------------------------------------------------------
    P_i_min = 0.0        # Minimum pressure [MPa]
    P_i_max = 0.038      # Maximum pressure [MPa]
    num_points = 30      # Number of pressure steps

    # Generate pressure array
    pressure_array = np.linspace(P_i_min, P_i_max, num_points)
    deformed_r_i = []
    deformed_r_e = []

    print(f"Running pressure sweep from {P_i_min} to {P_i_max} MPa ({num_points} points)...")

    # Solve for each pressure level
    for P_i in pressure_array:
        res = solve_thick_cylinder_mooney_rivlin_FEM_consistent(
            P_i, lambda_z, R_i, R_e, C10, C01, verbose=False
        )
        deformed_r_i.append(res['r'][0])      # Deformed inner radius r_i
        deformed_r_e.append(res['r'][-1])     # Deformed outer radius r_e

    # ---------------------------------------------------------
    # PLOT: INTERNAL PRESSURE VS DEFORMED INNER RADIUS
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(pressure_array, deformed_r_i, '-', color='navy', linewidth=2, label='Deformed inner radius $r_i$')
    plt.plot(pressure_array, deformed_r_e, '--', color='crimson', linewidth=1.5, alpha=0.7, label='Deformed outer radius $r_e$')

    plt.xlabel('Internal Pressure $P_i$ [MPa]', fontsize=11)
    plt.ylabel('Deformed Radius $r$ [mm]', fontsize=11)
    plt.title('Deformed Radius vs Internal Pressure', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
