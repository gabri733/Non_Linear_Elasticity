import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solve_thick_cylinder_gent_FEM_consistent(P_i, lambda_z, R_i, R_e, mu, Jm):
    """
    Solves the hyperelastic thick cylinder problem under internal pressure,
    using the Gent material model (strain-limiting of polymer chains),
    with formulation consistent with FEM solvers (ANSYS, Abaqus).

    Material parameters:
      mu  : shear modulus at small strains, analogous to
            2*(C10+C01) in Mooney-Rivlin model
      Jm  : Gent "locking" parameter, related to maximum chain extensibility.
            Tension diverges when I1 - 3 -> Jm.
            Typical rubber values: Jm ~ 20-120 (smaller = stiffens earlier)

    Unlike Mooney-Rivlin (depends on I1 and I2), the Gent model depends
    ONLY on I1 = tr(B); the deviatoric stress has no B^-1 term: tau = 2*dW/dI1 * B
    """

    # 1. Deviatoric stresses as a function of lambda_theta and lambda_z
    def compute_deviatoric_stresses(lambda_theta):
        # Incompressibility: lambda_r * lambda_theta * lambda_z = 1
        lambda_r = 1.0 / (lambda_theta * lambda_z)

        B_r = lambda_r**2
        B_t = lambda_theta**2
        B_z = lambda_z**2

        I1 = B_r + B_t + B_z

        # Margin before locking: must remain > 0
        margin = Jm - (I1 - 3.0)

        # dW/dI1 for Gent model: W = -(mu*Jm/2) * ln(1 - (I1-3)/Jm)
        # dW/dI1 = (mu*Jm) / (2*margin)
        dWdI1 = np.where(margin > 0, mu * Jm / (2.0 * np.where(margin > 0, margin, np.nan)), np.nan)

        # Deviatoric stresses (without hydrostatic pressure -p):
        # for Gent (depends only on I1): tau = 2*dW/dI1 * B  (no B^-1 term)
        tau_rr = 2.0 * dWdI1 * B_r
        tau_tt = 2.0 * dWdI1 * B_t
        tau_zz = 2.0 * dWdI1 * B_z

        return tau_rr, tau_tt, tau_zz, margin

    # 2. Equilibrium integral solved as ODE with adaptive stepsize.
    #    IMPORTANT NOTE vs Mooney-Rivlin version: near Gent locking,
    #    the integrand (tau_tt - tau_rr)/r grows very steeply
    #    (diverges when margin -> 0). Uniform grid with trapezoidal integration
    #    does NOT converge there: we need adaptive stepsize that refines
    #    automatically where gradient is steep, so we use solve_ivp.
    def margin_at_radius(r, C):
        R = np.sqrt(lambda_z * (r**2 - 2.0 * C))
        lambda_theta = r / R
        lambda_r = 1.0 / (lambda_theta * lambda_z)
        I1 = lambda_r**2 + lambda_theta**2 + lambda_z**2
        return Jm - (I1 - 3.0)

    def compute_sigma_rr_profile(C, num_points=500):
        r_i2 = (R_i**2) / lambda_z + 2.0 * C
        r_e2 = (R_e**2) / lambda_z + 2.0 * C

        if r_i2 <= 0 or r_e2 <= 0:
            return None, None, None

        r_i = np.sqrt(r_i2)
        r_e = np.sqrt(r_e2)

        # If material already at extensibility limit at one boundary, reject immediately
        if margin_at_radius(r_i, C) <= 1e-9 or margin_at_radius(r_e, C) <= 1e-9:
            return None, None, None

        def rhs(r, y):
            R = np.sqrt(lambda_z * (r**2 - 2.0 * C))
            lambda_theta = r / R
            tau_rr, tau_tt, _, _ = compute_deviatoric_stresses(lambda_theta)
            return [(tau_tt - tau_rr) / r]

        # Event: stops integration if locking is reached (margin -> 0)
        # before reaching r_e
        def locking_event(r, y):
            return margin_at_radius(r, C) - 1e-9
        locking_event.terminal = True
        locking_event.direction = -1

        sol = solve_ivp(rhs, [r_i, r_e], [-P_i], method='RK45',
                         dense_output=True, events=locking_event,
                         rtol=1e-10, atol=1e-13)

        if sol.status != 0 or not sol.success:
            # Integration stopped before reaching r_e:
            # material reached strain limit within thickness
            return None, None, None

        r_grid = np.linspace(r_i, r_e, num_points)
        R_grid = np.sqrt(lambda_z * (r_grid**2 - 2.0 * C))
        sigma_rr = sol.sol(r_grid)[0]

        return r_grid, R_grid, sigma_rr

    # 3. Residual: outer wall must be stress-free, sigma_rr(r_e) = 0
    def pressure_residual(C):
        r_grid, _, sigma_rr = compute_sigma_rr_profile(C)
        if sigma_rr is None:
            return np.nan
        return sigma_rr[-1]

    # 4. Find kinematic limit C_lock: unlike Mooney-Rivlin,
    #    we cannot use arbitrarily large C here, because Gent model
    #    simply becomes undefined when chains reach extensibility limit.
    #    Search for largest admissible C with exponential search + bisection.
    C_start = 0.0  # C=0 corresponds to "natural" state under just lambda_z, with P=0
    residual_start = pressure_residual(C_start)
    if np.isnan(residual_start):
        raise ValueError("Even unpressurized state (P_i=0) exceeds material extensibility limit: "
                          "lambda_z and/or Jm are not physically compatible.")

    # Exponential search for first inadmissible C
    C_valid, C_invalid = C_start, None
    step = 1.0
    while True:
        C_try = C_valid + step
        r_grid, _, sigma_rr = compute_sigma_rr_profile(C_try)
        if sigma_rr is None:
            C_invalid = C_try
            break
        C_valid = C_try
        step *= 2.0
        if step > 1e12:
            raise ValueError("Cannot identify locking limit: check parameters.")

    # Bisection to refine C_lock between C_valid (admissible) and C_invalid (not)
    for _ in range(60):
        C_mid = 0.5 * (C_valid + C_invalid)
        _, _, sigma_rr = compute_sigma_rr_profile(C_mid)
        if sigma_rr is None:
            C_invalid = C_mid
        else:
            C_valid = C_mid

    C_lock = C_valid
    C_max_bracket = C_lock * (1.0 - 1e-6)  # Safety margin just below locking

    # 5. Root search for C constant in physically admissible range
    residual_hi = pressure_residual(C_max_bracket)
    if np.isnan(residual_hi) or np.sign(residual_start - P_i * 0) == np.sign(residual_hi):
        # NOTE: at C_start, sigma_rr(r_e) = -P_i (residual_start = -P_i), always negative for P_i>0.
        # If residual is still negative at locking limit, it means the required pressure
        # exceeds the maximum sustainable BEFORE material reaches chain extensibility limit:
        # here the limit is true material failure, not geometric instability as in Mooney-Rivlin.
        r_i_lock = np.sqrt(R_i**2 / lambda_z + 2.0 * C_lock)
        raise ValueError(
            f"No admissible solution: to reach P_i={P_i} MPa material would need to "
            f"exceed extensibility limit (Jm={Jm}) before satisfying equilibrium.\n"
            f"Maximum sustainable pressure with these parameters (just before locking): "
            f"~{residual_hi + P_i:.5f} MPa (achieved at r_i -> {r_i_lock:.3f} mm). "
            f"Increase Jm (more extensible material) or reduce P_i."
        )

    sol = root_scalar(pressure_residual, bracket=[C_start, C_max_bracket], method='brentq')
    if not sol.converged:
        raise ValueError("Did not converge to physical solution.")

    C_optimal = sol.root

    # Verification: near Gent singularity the integrand diverges very rapidly,
    # and in double precision brentq can occasionally converge to a spurious root
    # (due to NaN handling in interval). Verify explicitly that boundary condition
    # is satisfied within tight tolerance, otherwise report issue rather than
    # silently return inaccurate result.
    residual_final = pressure_residual(C_optimal)
    tolerance = 1e-6 * max(1.0, abs(P_i))
    if not np.isfinite(residual_final) or abs(residual_final) > tolerance:
        raise ValueError(
            f"Solution found (C={C_optimal:.4f}) does not satisfy equilibrium within tolerance "
            f"(sigma_rr(r_e) residual = {residual_final:.3e} MPa, should be 0). This occurs when "
            f"P_i is so close to extensibility limit (Jm) that Gent model singularity is no longer "
            f"solvable in double precision. Reduce P_i or increase Jm."
        )

    # 6. Final computation of consistent fields
    r_grid, R_grid, sigma_rr = compute_sigma_rr_profile(C_optimal, num_points=1000)

    lambda_theta = r_grid / R_grid
    lambda_r = 1.0 / (lambda_theta * lambda_z)

    tau_rr, tau_tt, tau_zz, margin = compute_deviatoric_stresses(lambda_theta)

    # Obtain p(r) exactly from: sigma_rr = -p + tau_rr  ==>  p = tau_rr - sigma_rr
    p_hydro = tau_rr - sigma_rr

    # Total Cauchy stresses
    sigma_theta = -p_hydro + tau_tt
    sigma_zz = -p_hydro + tau_zz

    # True Logarithmic Strain (FEM standard)
    epsilon_r = np.log(lambda_r)
    epsilon_theta = np.log(lambda_theta)
    epsilon_z = np.log(lambda_z) * np.ones_like(r_grid)

    # Engineering Strain
    e_r = lambda_r - 1.0
    e_theta = lambda_theta - 1.0
    e_z = lambda_z - 1.0

    # Relative margin before locking (0 = at limit, 1 = natural state)
    relative_margin = margin / Jm

    # PRINT RESULTS TO TERMINAL
    print("=" * 70)
    print("      RIGOROUS RESULTS - GENT MODEL (locking at Jm={:.2f})".format(Jm))
    print("=" * 70)
    print(f"Computed constant C              : {C_optimal:.6f} mm^2")
    print(f"Final inner radius r_i           : {r_grid[0]:.4f} mm (initial R_i = {R_i:.2f} mm)")
    print(f"Final outer radius r_e           : {r_grid[-1]:.4f} mm (initial R_e = {R_e:.2f} mm)")
    print(f"Locking margin (min through thickness): {np.min(relative_margin)*100:.2f} %  "
          f"(0% = fiber at extensibility limit)")
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
        'epsilon_r': epsilon_r, 'epsilon_theta': epsilon_theta, 'epsilon_z': epsilon_z,
        'locking_margin': relative_margin
    }

if __name__ == "__main__":
    R_i = 10.0      # Inner radius [mm]
    R_e = 11.0      # Outer radius [mm]
    P_i = 1.9       # Internal pressure [MPa]  (Mooney-Rivlin with C10=C01=0.1 had no solution)
    lambda_z = 1.2  # Axial stretch

    # Gent parameters: mu equivalent to 2*(C10+C01) in Mooney-Rivlin case
    # for direct small-strain comparison; Jm typical of natural rubber
    mu = 0.4
    Jm = 30.0

    res = solve_thick_cylinder_gent_FEM_consistent(P_i, lambda_z, R_i, R_e, mu, Jm)
