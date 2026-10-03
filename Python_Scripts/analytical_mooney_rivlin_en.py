import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

def solve_thick_cylinder_mooney_rivlin_FEM_consistent(P_i, lambda_z, R_i, R_e, C10, C01):
    """
    Analytically/numerically solves the hyperelastic cylinder problem
    with formulation fully consistent with FEM solvers (e.g., Abaqus/ANSYS).

    This solver handles thick cylinders under combined internal pressure
    and axial stretch using the Mooney-Rivlin hyperelastic material model.
    """

    # 1. Definition of deviatoric stresses as a function of lambda_theta and lambda_z
    def compute_deviatoric_stresses(lambda_theta):
        # Incompressibility constraint: lambda_r * lambda_theta * lambda_z = 1
        lambda_r = 1.0 / (lambda_theta * lambda_z)

        # Components of left Cauchy-Green deformation tensor B
        B_r = lambda_r**2
        B_t = lambda_theta**2
        B_z = lambda_z**2

        # Inverse of B
        B_r_inv = 1.0 / B_r
        B_t_inv = 1.0 / B_t
        B_z_inv = 1.0 / B_z

        # Deviatoric stresses (without hydrostatic pressure -p)
        # Mooney-Rivlin: tau = 2*C10*B - 2*C01*B^-1
        tau_rr = 2.0 * C10 * B_r - 2.0 * C01 * B_r_inv
        tau_tt = 2.0 * C10 * B_t - 2.0 * C01 * B_t_inv
        tau_zz = 2.0 * C10 * B_z - 2.0 * C01 * B_z_inv

        return tau_rr, tau_tt, tau_zz

    # 2. Equilibrium integral solved numerically along r for FEM accuracy
    def compute_sigma_rr_profile(C, num_points=500):
        # Deformed radii at boundaries
        r_i2 = (R_i**2) / lambda_z + 2.0 * C
        r_e2 = (R_e**2) / lambda_z + 2.0 * C

        if r_i2 <= 0 or r_e2 <= 0:
            return None, None, None

        r_i = np.sqrt(r_i2)
        r_e = np.sqrt(r_e2)

        # Create radial grid in deformed configuration
        r_grid = np.linspace(r_i, r_e, num_points)
        # Reference radii corresponding to deformed radii
        R_grid = np.sqrt(lambda_z * (r_grid**2 - 2.0 * C))

        # Circumferential stretch at each radial position
        lambda_theta = r_grid / R_grid

        # Compute deviatoric stresses on grid
        tau_rr, tau_tt, _ = compute_deviatoric_stresses(lambda_theta)

        # Integrand of equilibrium equation: d(sigma_rr)/dr = (tau_tt - tau_rr) / r
        integrand = (tau_tt - tau_rr) / r_grid

        # Cumulative numerical integration starting from inner radius r_i
        # sigma_rr(r) = sigma_rr(r_i) + int_{r_i}^r (tau_tt - tau_rr)/r dr
        # Impose boundary condition: sigma_rr(r_i) = -P_i

        cum_integral = cumulative_trapezoid(integrand, r_grid, initial=0.0)
        sigma_rr = -P_i + cum_integral

        return r_grid, R_grid, sigma_rr

    # 3. Residual function to find C such that outer wall is stress-free: sigma_rr(r_e) = 0
    def pressure_residual(C):
        r_grid, _, sigma_rr = compute_sigma_rr_profile(C)
        if sigma_rr is None:
            return np.nan
        # We want radial stress at outer boundary to be zero
        return sigma_rr[-1]

    # Root search for constant C
    C_min = - (R_i**2) / (2.0 * lambda_z) + 1e-5
    C_max = 50.0 * (R_e**2)

    sol = root_scalar(pressure_residual, bracket=[C_min, C_max], method='brentq')
    if not sol.converged:
        raise ValueError("Did not converge to physical solution.")

    C_optimal = sol.root

    # 4. Final computation of consistent fields
    r_grid, R_grid, sigma_rr = compute_sigma_rr_profile(C_optimal, num_points=1000)

    # Compute stretches along grid
    lambda_theta = r_grid / R_grid
    lambda_r = 1.0 / (lambda_theta * lambda_z)

    # Compute all stress components
    tau_rr, tau_tt, tau_zz = compute_deviatoric_stresses(lambda_theta)

    # Obtain p(r) exactly from: sigma_rr = -p + tau_rr  ==>  p = tau_rr - sigma_rr
    p_hydro = tau_rr - sigma_rr

    # Total Cauchy stresses
    sigma_theta = -p_hydro + tau_tt
    sigma_zz = -p_hydro + tau_zz

    # True Logarithmic Strain (standard in FEM)
    epsilon_r = np.log(lambda_r)
    epsilon_theta = np.log(lambda_theta)
    epsilon_z = np.log(lambda_z) * np.ones_like(r_grid)

    # Engineering Strain (for additional FEM comparison)
    e_r = lambda_r - 1.0
    e_theta = lambda_theta - 1.0
    e_z = lambda_z - 1.0

    # PRINT RESULTS TO TERMINAL
    print("=" * 70)
    print("      RIGOROUS RESULTS (CONSISTENT WITH FEM SOLVER / ABAQUS)")
    print("=" * 70)
    print(f"Computed constant C              : {C_optimal:.6f} mm^2")
    print(f"Final inner radius r_i           : {r_grid[0]:.4f} mm (initial R_i = {R_i:.2f} mm)")
    print(f"Final outer radius r_e           : {r_grid[-1]:.4f} mm (initial R_e = {R_e:.2f} mm)")
    print("-" * 70)
    print(f"{'PARAMETER / STRESS [MPa]':<32} | {'INNER (r = r_i)':<15} | {'OUTER (r = r_e)':<15}")
    print("-" * 70)
    print(f"{'Radial Stress (sigma_rr)':<32} | {sigma_rr[0]:15.6f} | {sigma_rr[-1]:15.6f}")
    print(f"{'Hoop Stress (sigma_theta)':<32} | {sigma_theta[0]:15.6f} | {sigma_theta[-1]:15.6f}")
    print(f"{'Axial Stress (sigma_zz)':<32} | {sigma_zz[0]:15.6f} | {sigma_zz[-1]:15.6f}")
    print(f"{'Hydrostatic Pressure p(r)':<32} | {p_hydro[0]:15.6f} | {p_hydro[-1]:15.6f}")
    print("-" * 70)
    print(f"{'TRUE STRAIN (LE / True)':<32} | {'INNER (r = r_i)':<15} | {'OUTER (r = r_e)':<15}")
    print("-" * 70)
    print(f"{'Radial Strain (LE_r)':<32} | {epsilon_r[0]:15.6f} | {epsilon_r[-1]:15.6f}")
    print(f"{'Hoop Strain (LE_theta)':<32} | {epsilon_theta[0]:15.6f} | {epsilon_theta[-1]:15.6f}")
    print(f"{'Axial Strain (LE_z)':<32} | {epsilon_z[0]:15.6f} | {epsilon_z[-1]:15.6f}")
    print("-" * 70)
    print(f"{'ENGINEERING STRAIN (E)':<32} | {'INNER (r = r_i)':<15} | {'OUTER (r = r_e)':<15}")
    print("-" * 70)
    print(f"{'Eng. Radial Strain (E_r)':<32} | {e_r[0]:15.6f} | {e_r[-1]:15.6f}")
    print(f"{'Eng. Hoop Strain (E_theta)':<32} | {e_theta[0]:15.6f} | {e_theta[-1]:15.6f}")
    print("=" * 70)

    return {
        'r': r_grid, 'sigma_rr': sigma_rr, 'sigma_theta': sigma_theta,
        'sigma_zz': sigma_zz, 'p_hydro': p_hydro,
        'epsilon_r': epsilon_r, 'epsilon_theta': epsilon_theta, 'epsilon_z': epsilon_z
    }

if __name__ == "__main__":
    R_i = 10.0      # Inner radius [mm]
    R_e = 11.0      # Outer radius [mm]
    P_i = 0.035     # Internal pressure [MPa]
    lambda_z = 0.8  # Axial stretch

    # Mooney-Rivlin material parameters [MPa]
    C10 = 0.1
    C01 = 0.1

    res = solve_thick_cylinder_mooney_rivlin_FEM_consistent(P_i, lambda_z, R_i, R_e, C10, C01)

    # ---------------------------------------------------------
    # PLOT GENERATION (TRUE STRESSES AND STRAINS)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Cauchy stresses
    ax1.plot(res['r'], res['sigma_rr'], label=r'$\sigma_{r}$ (Radial)', color='blue', linewidth=2)
    ax1.plot(res['r'], res['sigma_theta'], label=r'$\sigma_{\theta}$ (Hoop)', color='red', linewidth=2)
    ax1.plot(res['r'], res['sigma_zz'], label=r'$\sigma_{z}$ (Axial)', color='green', linewidth=2)
    ax1.set_xlabel('Deformed radius $r$ [mm]', fontsize=11)
    ax1.set_ylabel('Stress [MPa]', fontsize=11)
    ax1.set_title('Cauchy Stress Profile', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    # Plot 2: True logarithmic strains
    ax2.plot(res['r'], res['epsilon_r'], label=r'$\varepsilon_r$', color='blue', linewidth=2)
    ax2.plot(res['r'], res['epsilon_theta'], label=r'$\varepsilon_\theta$', color='red', linewidth=2)
    ax2.plot(res['r'], res['epsilon_z'], label=r'$\varepsilon_z$', color='green', linewidth=2)
    ax2.set_xlabel('Deformed radius $r$ [mm]', fontsize=11)
    ax2.set_ylabel('True Strain [-]', fontsize=11)
    ax2.set_title('True Strain Profile (True Logarithmic Strain)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()
