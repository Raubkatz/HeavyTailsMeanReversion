import os
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.stats import levy_stable
    HAS_LEVY_STABLE = True
except ImportError:
    HAS_LEVY_STABLE = False
    print("[WARNING] 'scipy.stats.levy_stable' not found. Lévy increments will NOT work.")


##############################################################################
# Custom Color Palette
##############################################################################

CUSTOM_PALETTE_all = [
    "#E2DBBE",  # Light
    "#D5D6AA",
    "#9DBBAE",
    "#769FB6",
    "#188FA7",  # Dark
]

CUSTOM_PALETTE = [
    "#9DBBAE",
    "#769FB6",
    "#188FA7",  # Dark
]


##############################################################################
# Basic Dimensionless OU Generator
##############################################################################

def dimensionless_ou_process(
        length,
        dt,
        theta,
        discretization="EM",
        levy=False,
        levy_alpha=1.5,
        random_seed=None
    ):
    """
    Generate a single *dimensionless* Ornstein–Uhlenbeck (OU) time series of given length,
    with an option to use Gaussian or symmetric Lévy–stable increments.

    Dimensionless Gaussian OU SDE:
        dY_t = -theta * Y_t dt + sqrt(2 * theta) dW_t,
    which has zero stationary mean and unit stationary variance.

    Lévy–stable-driven OU (symmetric, β=0) in the same dimensionless spirit uses
    σ_α = (α * theta)^{1/α} so that the stationary marginal is S(α, 0, 1, 0).
    The exact one-step transition over Δt is:
        Y_{n+1} = e^{-theta Δt} Y_n + (1 - e^{-α theta Δt})^{1/α} S_n,
    where S_n ~ S(α, 0, 1, 0) (unit-scale symmetric α-stable). See, e.g.,
    - Marquardt (2007), "Stationary Lévy-driven Ornstein–Uhlenbeck processes,"
      Stochastic Processes and their Applications.
    - Samorodnitsky & Taqqu (1994), "Stable Non-Gaussian Random Processes."

    Discretization Methods
    ----------------------
    1) Euler–Maruyama ("EM") – approximate:
         • Gaussian case:
             Y_{n+1} = Y_n - theta * Y_n * Δt + sqrt(2 * theta) * (ΔW),  ΔW ~ N(0, Δt)
         • Stable case (symmetric):
             Y_{n+1} = Y_n - theta * Y_n * Δt
                       + (α * theta)^{1/α} * (Δt)^{1/α} * S_n,  S_n ~ S(α, 0, 1, 0)

       The (Δt)^{1/α} scaling reflects the α-stable increment behavior.

    2) Exact-step ("exact") – closed-form transition:
         • Gaussian case:
             Y_{n+1} = e^{-theta Δt} Y_n + sqrt(1 - e^{-2 theta Δt}) * Z_n,  Z_n ~ N(0, 1)
         • Stable case (symmetric):
             Y_{n+1} = e^{-theta Δt} Y_n + (1 - e^{-α theta Δt})^{1/α} * S_n,
             S_n ~ S(α, 0, 1, 0).

       Implementation detail (SciPy parameterization):
       SciPy's levy_stable uses the S^0 parameterization; at α=2 with scale=1 it returns N(0, 2).
       To match Z ~ N(0,1) in the Gaussian edge when using levy_stable at α=2,
       one may rescale S by 1/sqrt(2). We do this automatically below.

    Parameters
    ----------
    length : int
        Number of samples in the generated time series.
    dt : float
        Time step in the discretization.
    theta : float
        Mean reversion rate (> 0).
    discretization : str
        Discretization method, "EM" or "exact". Default is "EM".
    levy : bool
        Whether to use Lévy–stable increments (symmetric α-stable). Default=False.
    levy_alpha : float
        Stability parameter alpha, with 0 < alpha ≤ 2. Default=1.5.
        alpha=2 recovers the Gaussian boundary.
    random_seed : int or None
        If not None, seeds NumPy and Python's random for reproducible results.

    Returns
    -------
    X : numpy.ndarray
        1D array of length 'length' representing the dimensionless OU time series.

    Additional references
    ---------------------
    - Chambers, Mallows & Stuck (1976): a classic algorithm for α-stable sampling.
    - Nolan (2020), "Stable Distributions – Models for Heavy Tailed Data" (online monograph).
    - SciPy docs: scipy.stats.levy_stable
    """
    # If a seed is provided, seed both NumPy and built-in random
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    X = np.zeros(length)
    X[0] = 0.0  # start from zero

    use_levy = levy and HAS_LEVY_STABLE
    # Tolerance for detecting α ≈ 2 in floating point
    _ALPHA2_TOL = 1e-12

    for t in range(1, length):

        if discretization.lower() == "exact":
            # Exact one-step transition

            if use_levy:
                # Unit-scale symmetric α-stable variate (no Δt inside the draw)
                S = levy_stable.rvs(alpha=levy_alpha, beta=0, loc=0, scale=1.0)
                # SciPy at α=2, scale=1 gives N(0,2); rescale to N(0,1) for the Gaussian edge.
                if abs(levy_alpha - 2.0) <= _ALPHA2_TOL:
                    S *= 1.0 / np.sqrt(2.0)

                # Y_{n+1} = e^{-θΔt} Y_n + (1 - e^{-αθΔt})^{1/α} * S
                exp_neg = np.exp(-theta * dt)
                # Use expm1 for numerical stability: 1 - e^{-a} = -expm1(-a)
                noise_scale = (-np.expm1(-levy_alpha * theta * dt)) ** (1.0 / levy_alpha)

                X[t] = exp_neg * X[t - 1] + noise_scale * S

            else:
                # Gaussian exact: Z ~ N(0, 1) (no √Δt here)
                Z = np.random.normal(0.0, 1.0)
                exp_neg = np.exp(-theta * dt)
                var_inc = -np.expm1(-2.0 * theta * dt)  # = 1 - e^{-2θΔt}
                X[t] = exp_neg * X[t - 1] + np.sqrt(var_inc) * Z

        else:
            # Euler–Maruyama (approximate)

            if use_levy:
                # Approximate stable OU increment over Δt:
                # drift: -θ X Δt
                # noise: (αθ)^{1/α} (Δt)^{1/α} S, with S ~ S(α,0,1,0)
                S = levy_stable.rvs(alpha=levy_alpha, beta=0, loc=0, scale=1.0)
                if abs(levy_alpha - 2.0) <= _ALPHA2_TOL:
                    S *= 1.0 / np.sqrt(2.0)

                X[t] = (
                    X[t - 1]
                    - theta * X[t - 1] * dt
                    + (levy_alpha * theta) ** (1.0 / levy_alpha) * (dt ** (1.0 / levy_alpha)) * S
                )

            else:
                # Gaussian EM: ΔW ~ N(0, Δt)
                dW = np.random.normal(0.0, np.sqrt(dt))
                X[t] = X[t - 1] - theta * X[t - 1] * dt + np.sqrt(2.0 * theta) * dW

    return X


##############################################################################
# Helper function for scaling
##############################################################################

def scale_to_interval(data, low=0.1, high=1.0):
    """Scale a 1D NumPy array to [low, high]."""
    dmin, dmax = np.min(data), np.max(data)
    if dmax == dmin:
        # If constant, just fill with 'low'
        return np.full_like(data, low)
    return low + (data - dmin) * (high - low) / (dmax - dmin)


##############################################################################
# Main Plotting Function
##############################################################################

def plot_dimensionless_ou_processes_custom(
    length=1000,
    dt=0.01,
    thetas=(1.0,),
    discretization="EM",
    levy=False,
    levy_alpha=1.5,
    random_seed=None,
    font_size=14,
    plot_folder="ts_data_plots",
    save_basename=None,
    plot_scale_to_unit=True,    # NEW parameter, default True
    plot_mean_line=False        # NEW parameter, default False
):
    """
    Plot multiple dimensionless Ornstein-Uhlenbeck processes with different mean-reversion
    rates (theta). Optionally uses Levy-stable increments if levy=True.

    Each theta gets its own subplot. Lines use a custom color palette in a cycling fashion.

    Parameters
    ----------
    length : int
        Number of time steps in each series (default=1000).
    dt : float
        Time increment (default=0.01).
    thetas : list/tuple of float
        Mean reversion rates to plot. One subplot per theta.
    discretization : str
        "EM" or "exact". Default="EM".
    levy : bool
        If True, use Lévy-stable increments (alpha=levy_alpha). Default=False.
    levy_alpha : float
        Stability parameter for Lévy-stable increments. Default=1.5.
    random_seed : int or None
        If not None, sets the seed for reproducibility. Default=None.
    font_size : int
        Controls *all* fonts: tick labels, legend, axis labels, titles, etc.
    plot_folder : str
        Folder name to store plots. Default="ts_data_plots".
    save_basename : str or None
        Base filename for saving plots (without extension).
        If None, just show() the plot interactively. Otherwise, saves PNG & EPS in plot_folder.
    plot_scale_to_unit : bool
        If True (default), we scale each generated OU series to [0.1, 1.0] before plotting.
    plot_mean_line : bool
        If True, we draw a dashed horizontal line at y=0. (False by default).

    Returns
    -------
    None
    """
    # 1) Ensure folder exists
    os.makedirs(plot_folder, exist_ok=True)

    # 2) Set up fonts & style
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.titlesize"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["figure.titlesize"] = font_size

    sns.set_style("whitegrid")  # a nice grid background

    # Make background transparent
    plt.rcParams['figure.facecolor']   = 'none'
    plt.rcParams['axes.facecolor']     = 'none'
    plt.rcParams['savefig.facecolor']  = 'none'
    plt.rcParams['savefig.edgecolor']  = 'none'

    # 3) Prepare subplots
    n_plots = len(thetas)
    fig, axes = plt.subplots(n_plots, 1, figsize=(9, 3.5 * n_plots))
    if n_plots == 1:
        axes = [axes]  # Make axes iterable if there's only one

    # Cycle through the palette if we have more subplots than colors
    colors = [CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)] for i in range(n_plots)]

    time = np.arange(length)

    # 4) Generate & Plot
    for i, theta in enumerate(thetas):
        Y = dimensionless_ou_process(
            length=length,
            dt=dt,
            theta=theta,
            discretization=discretization,
            levy=levy,
            levy_alpha=levy_alpha,
            random_seed=(random_seed + i if random_seed is not None else None)
        )

        # If requested, scale each OU series to [0.1, 1.0]
        if plot_scale_to_unit:
            Y = scale_to_interval(Y, low=0.1, high=1.0)

        ax = axes[i]
        ax.plot(time, Y, color=colors[i], linewidth=4,
                label=(f"Dimless OU (theta={theta})\nlevy={levy}, alpha={levy_alpha}"))

        # Plot the dashed line at y=0 only if plot_mean_line=True
        if plot_mean_line:
            ax.axhline(0.0, color="red", linestyle="--", linewidth=2, label="Mean = 0")

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("OU Value")
        ax.set_title(
            f"θ={theta}, α={levy_alpha}"
        )
        # ax.legend() # You can re-enable if you want legends visible

    plt.tight_layout()

    # 5) Save or Show
    if save_basename is not None:
        # e.g. "ts_data_plots/my_ou_plot.png" & ".eps"
        if levy:
            png_path = os.path.join(plot_folder, f"{save_basename}_alpha{str(levy_alpha)}.png")
            eps_path = os.path.join(plot_folder, f"{save_basename}_alpha{str(levy_alpha)}.eps")
        else:
            png_path = os.path.join(plot_folder, f"{save_basename}.png")
            eps_path = os.path.join(plot_folder, f"{save_basename}.eps")

        plt.savefig(png_path, dpi=200, transparent=True)
        plt.savefig(eps_path, format="eps", dpi=200, transparent=True)
        plt.close(fig)
        print(f"[INFO] Saved plot to '{png_path}' and '{eps_path}'")
    else:
        plt.show()


##############################################################################
# Example Usage
##############################################################################

if __name__ == "__main__":

    SEED = 667

    # Example: multiple thetas, Euler-Maruyama, random seed
    plot_dimensionless_ou_processes_custom(
        length=100,
        dt=0.01,
        thetas=[0.000001, 2.0, 4.0, 16.0, 32.0],
        discretization="exact",
        levy=True,
        levy_alpha=0.05,
        random_seed=SEED,
        font_size=24,
        plot_folder="ts_data_plots",
        save_basename="ou_plot_example_final"
    )

    plot_dimensionless_ou_processes_custom(
        length=100,
        dt=0.01,
        thetas=[0.000001, 2.0, 4.0, 16.0, 32.0],
        discretization="exact",
        levy=True,
        levy_alpha=0.5,
        random_seed=SEED,
        font_size=24,
        plot_folder="ts_data_plots",
        save_basename="ou_plot_example_final"
    )

    plot_dimensionless_ou_processes_custom(
        length=100,
        dt=0.01,
        thetas=[0.000001, 2.0, 4.0, 16.0, 32.0],
        discretization="exact",
        levy=True,
        levy_alpha=1.0,
        random_seed=SEED,
        font_size=24,
        plot_folder="ts_data_plots",
        save_basename="ou_plot_example_final"
    )

    plot_dimensionless_ou_processes_custom(
        length=100,
        dt=0.01,
        thetas=[0.000001, 2.0, 4.0, 16.0, 32.0],
        discretization="exact",
        levy=True,
        levy_alpha=1.5,
        random_seed=SEED,
        font_size=24,
        plot_folder="ts_data_plots",
        save_basename="ou_plot_example_final"
    )

    plot_dimensionless_ou_processes_custom(
        length=100,
        dt=0.01,
        thetas=[0.000001, 2.0, 4.0, 16.0, 32.0],
        discretization="exact",
        levy=True,
        levy_alpha=2.0,
        random_seed=SEED,
        font_size=24,
        plot_folder="ts_data_plots",
        save_basename="ou_plot_example_final"
    )


