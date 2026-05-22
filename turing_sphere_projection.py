"""
Projected dynamics of a free geodesic on a 2-sphere, observed in the plane.
===========================================================================

A point B moves on a great circle (a free geodesic) of a sphere S^2(R) in R^3.
An observer A lives in the equatorial plane z = 0 and can only see the
orthogonal projection B' = (x, y) of B. This script studies what apparent
dynamics A reconstructs from that projected motion.

The central question is the time map dt/dlambda: how the observer's clock t
relates to the affine parameter lambda along the geodesic. Three candidate
maps are compared; only one is physically admissible. For that map we derive,
in closed form and verify numerically, the apparent velocity, the apparent
acceleration, a conserved invariant, and the natural emergence of a
special-relativistic time-dilation factor.

Conventions:
    R       sphere radius (normalised to 1)
    omega   angular velocity along the great circle (= V / R)
    V       speed of B on the sphere (constant, = R * omega)
    i       inclination of the great circle to the equatorial plane
    phi     angular position along the great circle (phi = omega * lambda)
    f       fundamental factor f = 1 - cos^2(phi) sin^2(i) = |V_R2|^2 / V^2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import cumulative_trapezoid

# ============================================================
# PARAMETERS
# ============================================================
R = 1.0            # sphere radius (normalised units)
omega = 2 * np.pi  # angular velocity (one orbit per unit of lambda-time)
V = R * omega      # speed on the sphere (constant by the geodesic property)

N = 10000          # samples per orbit
phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
dphi = phi[1] - phi[0]
dlam = dphi / omega  # step in the affine parameter lambda

# ============================================================
# CORE FUNCTIONS
# ============================================================

def geodesic_R3(phi, i):
    """
    Position and velocity of B on the great circle of inclination i.

    The great circle is the intersection of the sphere with a plane tilted by
    the inclination i. Parametrised by phi, the embedding in R^3 is a simple
    rotation of the equatorial circle about the x-axis by the angle i.
    """
    x = R * np.cos(phi)
    y = R * np.sin(phi) * np.cos(i)
    z = R * np.sin(phi) * np.sin(i)
    vx = -V * np.sin(phi)
    vy = V * np.cos(phi) * np.cos(i)
    vz = V * np.cos(phi) * np.sin(i)
    return x, y, z, vx, vy, vz


def f_factor(phi, i):
    """
    Fundamental factor f = 1 - cos^2(phi) sin^2(i).

    This equals the squared in-plane speed |V_R2|^2 normalised by V^2, i.e.
    the fraction of B's kinetic content that remains visible to A after the
    hidden z-component of velocity is removed. It controls every projected
    quantity below.
    """
    return 1 - np.cos(phi)**2 * np.sin(i)**2


def compute_option_B(phi, i):
    """
    Physically admissible time map: dt/dlambda = sqrt(f).

    This is the special-relativistic choice: it sets the observer's proper
    time so that the hidden velocity component acts exactly as a Lorentz
    time-dilation. It is the only one of the three candidates that is positive
    everywhere (the clock never runs backwards) and that conserves the observed
    speed. Returns all quantities expressed in the observer's time t.
    """
    f = f_factor(phi, i)

    # --- Observer time t obtained by integrating dt = sqrt(f) dlambda ---
    dt_dlam = np.sqrt(f)
    t = cumulative_trapezoid(dt_dlam / omega, phi, initial=0)

    # --- Projected position in the plane (an ellipse of semi-axes R, R cos i) ---
    x_R2 = R * np.cos(phi)
    y_R2 = R * np.sin(phi) * np.cos(i)

    # --- Projected velocity: chain rule dx/dt = (dx/dlambda)/(dt/dlambda) ---
    # The factor gamma = 1/sqrt(f) is the generalised Lorentz factor.
    gamma = 1.0 / np.sqrt(f)
    ux = -V * np.sin(phi) * gamma
    uy = V * np.cos(phi) * np.cos(i) * gamma

    # --- Projected acceleration (closed form after simplification) ---
    # Differentiating the velocity once more in t and simplifying yields an
    # anisotropic restoring acceleration with a 1/f^2 modulation.
    ax = -R * omega**2 * np.cos(phi) * np.cos(i)**2 / f**2
    ay = -R * omega**2 * np.cos(i) * np.sin(phi) / f**2

    return t, x_R2, y_R2, ux, uy, ax, ay, f, gamma


def compute_option_A(phi, i):
    """
    Candidate time map A: dt/dlambda = vz / V = cos(phi) sin(i).

    Equating the observer time to the hidden coordinate's rate. Shown below to
    be unphysical: this quantity changes sign at phi = pi/2, so the clock would
    run backwards over half the orbit.
    """
    dt_dlam = np.cos(phi) * np.sin(i)
    t = cumulative_trapezoid(dt_dlam / omega, phi, initial=0)
    return t, dt_dlam


def compute_option_C(phi, i):
    """
    Candidate time map C: dt/dlambda = |vz| / V = |cos(phi)| sin(i).

    The absolute value fixes the sign problem of A but introduces zeros at the
    poles (phi = pi/2), where the clock stops and projected velocities diverge.
    Also unphysical.
    """
    dt_dlam = np.abs(np.cos(phi)) * np.sin(i)
    t = cumulative_trapezoid(dt_dlam / omega, phi, initial=0)
    return t, dt_dlam


def numerical_derivatives(t, x):
    """Central-difference numerical derivative, used to cross-check the
    closed-form expressions independently."""
    return np.gradient(x, t, edge_order=2)


# ============================================================
# FIGURE 1: OVERVIEW OF THE THREE TIME MAPS
# ============================================================

def plot_figure1():
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Projected dynamics: three candidate time maps  |  S^2(R) -> plane',
                 fontsize=14, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    i_test = np.radians(45)  # representative inclination
    x, y, z, vx, vy, vz = geodesic_R3(phi, i_test)

    # --- (a) 3D view: great circle on the sphere and its planar projection ---
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    u_s = np.linspace(0, 2 * np.pi, 50)
    v_s = np.linspace(0, np.pi, 50)
    xs = R * np.outer(np.cos(u_s), np.sin(v_s))
    ys = R * np.outer(np.sin(u_s), np.sin(v_s))
    zs = R * np.outer(np.ones_like(u_s), np.cos(v_s))
    ax3d.plot_surface(xs, ys, zs, alpha=0.1, color='skyblue')
    ax3d.plot(x, y, z, 'b-', lw=2, label='Geodesic B')
    ax3d.plot(x, y, np.zeros_like(z), 'r--', lw=1.5, label="Projection B'")
    xx_p = np.linspace(-R, R, 10)
    yy_p = np.linspace(-R, R, 10)
    XX, YY = np.meshgrid(xx_p, yy_p)
    mask = XX**2 + YY**2 <= R**2
    ZZ = np.zeros_like(XX)
    ZZ[~mask] = np.nan
    ax3d.plot_surface(XX, YY, ZZ, alpha=0.08, color='red')
    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
    ax3d.set_title('(a) Great circle on S^2(R), i = 45 deg')
    ax3d.legend(fontsize=8)

    # --- (b) The projected trajectory is an ellipse ---
    ax_ell = fig.add_subplot(gs[0, 1])
    ax_ell.plot(x, y, 'r-', lw=2)
    ax_ell.set_xlabel('x'); ax_ell.set_ylabel('y')
    ax_ell.set_aspect('equal')
    ax_ell.set_title("(b) Projected trajectory B' (ellipse)")
    ax_ell.grid(True, alpha=0.3)
    ax_ell.axhline(0, color='k', lw=0.5)
    ax_ell.axvline(0, color='k', lw=0.5)
    ax_ell.annotate(f'a = R = {R:.1f}', xy=(R, 0), fontsize=9,
                    xytext=(R + 0.1, 0.15), arrowprops=dict(arrowstyle='->', color='blue'))
    ax_ell.annotate(f'b = R cos(i) = {R*np.cos(i_test):.3f}', xy=(0, R * np.cos(i_test)),
                    fontsize=9, xytext=(0.2, R * np.cos(i_test) + 0.05),
                    arrowprops=dict(arrowstyle='->', color='blue'))

    # --- (c) The three candidate time maps dt/dlambda ---
    f = f_factor(phi, i_test)
    dtdl_A = np.cos(phi) * np.sin(i_test)
    dtdl_B = np.sqrt(f)
    dtdl_C = np.abs(np.cos(phi)) * np.sin(i_test)

    ax_dt = fig.add_subplot(gs[0, 2])
    ax_dt.plot(np.degrees(phi), dtdl_A, 'r-', lw=1.5, label='A: vz/V (sign-changing)')
    ax_dt.plot(np.degrees(phi), dtdl_B, 'b-', lw=2, label='B: sqrt(1 - vz^2/V^2)')
    ax_dt.plot(np.degrees(phi), dtdl_C, 'g--', lw=1.5, label='C: |vz|/V (vanishing)')
    ax_dt.axhline(0, color='k', lw=0.5, ls=':')
    ax_dt.set_xlabel('phi [deg]')
    ax_dt.set_ylabel('dt/dlambda')
    ax_dt.set_title('(c) Candidate time maps')
    ax_dt.legend(fontsize=9)
    ax_dt.grid(True, alpha=0.3)
    ax_dt.set_xlim(0, 360)

    # --- (d) Resulting observer time t(phi); only B is monotonic ---
    t_A, _ = compute_option_A(phi, i_test)
    t_B_data = compute_option_B(phi, i_test)
    t_B = t_B_data[0]
    t_C, _ = compute_option_C(phi, i_test)

    ax_t = fig.add_subplot(gs[1, 0])
    ax_t.plot(np.degrees(phi), t_A, 'r-', lw=1.5, label='A: t = z/V (non-monotonic)')
    ax_t.plot(np.degrees(phi), t_B, 'b-', lw=2, label='B: elliptic integral (monotonic)')
    ax_t.plot(np.degrees(phi), t_C, 'g--', lw=1.5, label='C: integral of |cos phi|')
    ax_t.set_xlabel('phi [deg]')
    ax_t.set_ylabel('t(phi)')
    ax_t.set_title('(d) Observer time vs angular position')
    ax_t.legend(fontsize=9)
    ax_t.grid(True, alpha=0.3)
    ax_t.set_xlim(0, 360)
    ax_t.axvspan(90, 270, alpha=0.08, color='red')  # region where A's clock reverses

    # --- (e) Option B: observed velocity components and conserved speed ---
    t_B, x_R2, y_R2, ux, uy, ax_acc, ay_acc, f_arr, gamma = compute_option_B(phi, i_test)
    speed = np.sqrt(ux**2 + uy**2)

    ax_v = fig.add_subplot(gs[1, 1])
    ax_v.plot(np.degrees(phi), ux / V, 'b-', lw=1.5, label='u_x / V')
    ax_v.plot(np.degrees(phi), uy / V, 'r-', lw=1.5, label='u_y / V')
    ax_v.plot(np.degrees(phi), speed / V, 'k--', lw=2, label='|V_R2| / V (constant)')
    ax_v.axhline(1, color='gray', lw=0.5, ls=':')
    ax_v.set_xlabel('phi [deg]')
    ax_v.set_ylabel('normalised velocity')
    ax_v.set_title('(e) Option B: observed velocity')
    ax_v.legend(fontsize=9)
    ax_v.grid(True, alpha=0.3)
    ax_v.set_xlim(0, 360)

    # --- (f) Option B: observed acceleration components ---
    a_norm = R * omega**2

    ax_a = fig.add_subplot(gs[1, 2])
    ax_a.plot(np.degrees(phi), ax_acc / a_norm, 'b-', lw=1.5, label='a_x / R omega^2')
    ax_a.plot(np.degrees(phi), ay_acc / a_norm, 'r-', lw=1.5, label='a_y / R omega^2')
    a_mag = np.sqrt(ax_acc**2 + ay_acc**2)
    ax_a.plot(np.degrees(phi), a_mag / a_norm, 'k--', lw=2, label='|a| / R omega^2')
    ax_a.set_xlabel('phi [deg]')
    ax_a.set_ylabel('normalised acceleration')
    ax_a.set_title('(f) Option B: observed acceleration')
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(0, 360)

    plt.savefig('fig1_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_overview.png")


# ============================================================
# FIGURE 2: DETAILED ANALYSIS OF THE ADMISSIBLE MAP (OPTION B)
# ============================================================

def plot_figure2():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Option B (dt/dlambda = sqrt(f)): detailed analysis, i = 45 deg',
                 fontsize=14, fontweight='bold')

    i_test = np.radians(45)
    t_B, x_R2, y_R2, ux, uy, ax_acc, ay_acc, f_arr, gamma = compute_option_B(phi, i_test)
    a_mag = np.sqrt(ax_acc**2 + ay_acc**2)
    r_mag = np.sqrt(x_R2**2 + y_R2**2)

    # --- (a) Trajectory with acceleration vectors overlaid ---
    ax = axes[0, 0]
    ax.plot(x_R2, y_R2, 'b-', lw=1.5)
    step = N // 20
    scale = 0.08 / np.max(a_mag)
    for k in range(0, N, step):
        ax.arrow(x_R2[k], y_R2[k], ax_acc[k] * scale, ay_acc[k] * scale,
                 head_width=0.02, head_length=0.01, fc='red', ec='red', alpha=0.7)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title('(a) Trajectory + acceleration field')
    ax.grid(True, alpha=0.3)
    ax.plot(0, 0, 'k+', ms=10, mew=2)

    # --- (b) Conserved invariant |a| * f^(3/2) = R omega^2 cos(i) ---
    # This is the key analytical result: although |a| varies strongly around
    # the orbit, the product |a| * f^(3/2) is exactly constant.
    ax = axes[0, 1]
    invariant = a_mag * f_arr**1.5 / (R * omega**2)
    expected_inv = np.cos(i_test)
    ax.plot(np.degrees(phi), invariant, 'b-', lw=2, label='|a| f^(3/2) / R omega^2')
    ax.axhline(expected_inv, color='r', ls='--', lw=2, label=f'cos(i) = {expected_inv:.4f}')
    ax.set_xlabel('phi [deg]')
    ax.set_ylabel('|a| f^(3/2) / R omega^2')
    ax.set_title('(b) Invariant: |a| f^(3/2) = R omega^2 cos(i)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    ax.set_ylim(expected_inv * 0.95, expected_inv * 1.05)

    # --- (c) Lorentz factor gamma and time-dilation dt/dtau ---
    ax = axes[0, 2]
    eta = np.sqrt(f_arr)  # dt/dtau
    ax.plot(np.degrees(phi), gamma, 'b-', lw=1.5, label='gamma = 1/sqrt(f)')
    ax.plot(np.degrees(phi), eta, 'r-', lw=1.5, label='dt/dtau = sqrt(f)')
    ax.axhline(1, color='k', lw=0.5, ls=':')
    ax.set_xlabel('phi [deg]')
    ax.set_ylabel('factor')
    ax.set_title('(c) Lorentz factor and time dilation')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)

    # --- (d) Independent check: closed-form vs numerical acceleration ---
    ax = axes[1, 0]
    ax_num = numerical_derivatives(t_B, ux)
    ay_num = numerical_derivatives(t_B, uy)
    ax.plot(np.degrees(phi), ax_acc / (R * omega**2), 'b-', lw=2, label='a_x closed form')
    ax.plot(np.degrees(phi), ax_num / (R * omega**2), 'b--', lw=1, label='a_x numerical')
    ax.plot(np.degrees(phi), ay_acc / (R * omega**2), 'r-', lw=2, label='a_y closed form')
    ax.plot(np.degrees(phi), ay_num / (R * omega**2), 'r--', lw=1, label='a_y numerical')
    ax.set_xlabel('phi [deg]')
    ax.set_ylabel('a / R omega^2')
    ax.set_title('(d) Check: closed form vs numerical')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)

    err_x = np.abs(ax_acc - ax_num) / (np.abs(ax_acc) + 1e-30)
    err_y = np.abs(ay_acc - ay_num) / (np.abs(ay_acc) + 1e-30)
    mask = (phi > 0.1) & (phi < 2 * np.pi - 0.1)
    print(f"Max relative error a_x: {np.max(err_x[mask]):.2e}")
    print(f"Max relative error a_y: {np.max(err_y[mask]):.2e}")

    # --- (e) Check: observed speed is exactly conserved ---
    ax = axes[1, 1]
    speed = np.sqrt(ux**2 + uy**2)
    speed_num = np.sqrt(numerical_derivatives(t_B, x_R2)**2 +
                        numerical_derivatives(t_B, y_R2)**2)
    ax.plot(np.degrees(phi), speed / V, 'b-', lw=2, label='|V_R2|/V closed form')
    ax.plot(np.degrees(phi), speed_num / V, 'r--', lw=1, label='|V_R2|/V numerical')
    ax.axhline(1.0, color='k', lw=0.5, ls=':')
    ax.set_ylim(0.98, 1.02)
    ax.set_xlabel('phi [deg]')
    ax.set_ylabel('|V_R2| / V')
    ax.set_title('(e) Check: |V_R2| = V = constant')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)

    # --- (f) Radial / azimuthal decomposition of the acceleration ---
    ax = axes[1, 2]
    r_hat_x = x_R2 / r_mag
    r_hat_y = y_R2 / r_mag
    a_rad = ax_acc * r_hat_x + ay_acc * r_hat_y
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x
    a_tang = ax_acc * t_hat_x + ay_acc * t_hat_y

    ax.plot(np.degrees(phi), a_rad / (R * omega**2), 'b-', lw=1.5, label='a_radial / R omega^2')
    ax.plot(np.degrees(phi), a_tang / (R * omega**2), 'r-', lw=1.5, label='a_azimuthal / R omega^2')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('phi [deg]')
    ax.set_ylabel('acceleration component')
    ax.set_title('(f) Radial / azimuthal decomposition')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)

    plt.savefig('fig2_optionB_detail.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_optionB_detail.png")


# ============================================================
# FIGURE 3: DEPENDENCE ON INCLINATION
# ============================================================

def plot_figure3():
    inclinations = [15, 30, 45, 60, 75]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Option B: dependence on inclination i', fontsize=14, fontweight='bold')

    for k, i_deg in enumerate(inclinations):
        i_rad = np.radians(i_deg)
        t_B, x_R2, y_R2, ux, uy, ax_acc, ay_acc, f_arr, gamma = compute_option_B(phi, i_rad)
        a_mag = np.sqrt(ax_acc**2 + ay_acc**2)
        r_mag = np.sqrt(x_R2**2 + y_R2**2)
        c = colors[k]
        lbl = f'i = {i_deg} deg'

        axes[0, 0].plot(x_R2, y_R2, color=c, lw=1.5, label=lbl)
        axes[0, 1].plot(np.degrees(phi), a_mag / (R * omega**2), color=c, lw=1.5, label=lbl)
        axes[0, 2].plot(np.degrees(phi), gamma, color=c, lw=1.5, label=lbl)
        axes[1, 0].plot(np.degrees(phi), np.sqrt(f_arr), color=c, lw=1.5, label=lbl)
        axes[1, 1].plot(np.degrees(phi), ay_acc / (R * omega**2), color=c, lw=1.5, label=lbl)
        # Show that the velocity factor f and the position factor g differ:
        g_arr = r_mag**2 / R**2
        axes[1, 2].plot(np.degrees(phi), f_arr, color=c, lw=1.5, ls='-')
        axes[1, 2].plot(np.degrees(phi), g_arr, color=c, lw=1.5, ls='--', alpha=0.5)

    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal'); axes[0, 0].set_title('(a) Projected ellipses')
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('phi [deg]'); axes[0, 1].set_ylabel('|a| / R omega^2')
    axes[0, 1].set_title('(b) Acceleration magnitude'); axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_xlim(0, 360)

    axes[0, 2].set_xlabel('phi [deg]'); axes[0, 2].set_ylabel('gamma')
    axes[0, 2].set_title('(c) Lorentz factor gamma'); axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3); axes[0, 2].set_xlim(0, 360)

    axes[1, 0].set_xlabel('phi [deg]'); axes[1, 0].set_ylabel('dt/dtau')
    axes[1, 0].set_title('(d) Time dilation dt/dtau'); axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].set_xlim(0, 360)

    axes[1, 1].set_xlabel('phi [deg]'); axes[1, 1].set_ylabel('a_y / R omega^2')
    axes[1, 1].set_title('(e) Cross-component a_y'); axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].set_xlim(0, 360)

    axes[1, 2].set_xlabel('phi [deg]'); axes[1, 2].set_ylabel('f (solid) / g (dashed)')
    axes[1, 2].set_title('(f) Velocity factor f vs position factor g (distinct)')
    axes[1, 2].grid(True, alpha=0.3); axes[1, 2].set_xlim(0, 360)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='k', ls='-', lw=1.5),
                    Line2D([0], [0], color='k', ls='--', lw=1.5, alpha=0.5)]
    axes[1, 2].legend(custom_lines, ['f = 1 - cos^2(phi) sin^2(i)',
                                     'g = r^2/R^2 = 1 - sin^2(phi) sin^2(i)'],
                      fontsize=8)

    plt.tight_layout()
    plt.savefig('fig3_inclinations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3_inclinations.png")


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

def print_summary():
    print("=" * 70)
    print("PROJECTED DYNAMICS ON S^2 -- NUMERICAL SUMMARY")
    print("=" * 70)

    for i_deg in [0, 15, 30, 45, 60, 75, 89]:
        i_rad = np.radians(i_deg)
        t_B, x_R2, y_R2, ux, uy, ax_acc, ay_acc, f_arr, gamma = compute_option_B(phi, i_rad)
        a_mag = np.sqrt(ax_acc**2 + ay_acc**2)
        r_mag = np.sqrt(x_R2**2 + y_R2**2)
        speed = np.sqrt(ux**2 + uy**2)

        print(f"\n--- i = {i_deg} deg ---")
        print(f"  Ellipse semi-axes: a = {R:.3f}, b = R cos(i) = {R*np.cos(i_rad):.3f}")
        print(f"  f range: [{np.min(f_arr):.4f}, {np.max(f_arr):.4f}]")
        print(f"  gamma range: [{np.min(gamma):.4f}, {np.max(gamma):.4f}]")
        print(f"  |V_R2|/V: {np.mean(speed)/V:.6f} +/- {np.std(speed)/V:.2e}  (should be 1.000)")
        print(f"  |a|_max / R omega^2: {np.max(a_mag)/(R*omega**2):.4f}")
        print(f"  |a|_min / R omega^2: {np.min(a_mag)/(R*omega**2):.4f}")

        # Conserved invariant: |a| * f^(3/2) = R omega^2 cos(i)
        product = a_mag * f_arr**1.5
        expected = R * omega**2 * np.cos(i_rad)
        print(f"  |a| f^(3/2) = {np.mean(product):.6f} +/- {np.std(product):.2e}  "
              f"(expected: {expected:.6f})")

        # Confirm that the velocity factor f differs from the position factor g
        g_arr = r_mag**2 / R**2
        diff_fg = np.max(np.abs(f_arr - g_arr))
        print(f"  max|f - g|: {diff_fg:.4f}  (f differs from r^2/R^2 for i > 0)")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print_summary()
    print("\nGenerating figures...")
    plot_figure1()
    plot_figure2()
    plot_figure3()
    print("\nAll figures generated successfully.")
