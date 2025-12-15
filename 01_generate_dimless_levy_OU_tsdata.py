import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random  # seed the built-in Python RNG as well

# Attempt to import levy_stable from scipy.stats
try:
    from scipy.stats import levy_stable
    HAS_LEVY_STABLE = True
except ImportError:
    HAS_LEVY_STABLE = False
    print("[WARNING] 'scipy.stats.levy_stable' not found. Lévy increments will NOT work.")


def custom_list(start=0.01, step=0.25, max_value=2.0):
    """
    Generate a list of float values where:
      - The first value is initially 0.
      - The rest follow an interval defined by 'step' (e.g., 0.25 → 0.50, 0.75, ...),
        up to 'max_value' inclusive (if it divides evenly) or until just past 'max_value'.
      - After generating the list, the first value (0) is replaced with 'start' (e.g., 0.01).

    Parameters
    ----------
    start : float
        The replacement value for the initial 0. Default is 0.01.
    step : float
        The step interval for incrementing. Default is 0.25.
    max_value : float
        The maximum value. Default is 2.0.

    Returns
    -------
    list of float
        A list of floats where the initial value 0 is replaced by 'start'.
    """
    values = [0.0]  # Start with 0
    current = step  # Begin stepping

    while current <= max_value:
        values.append(round(current, 10))  # Avoid floating-point representation issues
        current += step

    values[0] = start  # Replace the first value (0) with 'start'
    return values


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


def generate_dimensionless_ou_data(
        n_length=1000,
        dt=0.01,
        theta_values=None,
        alpha_values=None,
        n_copies=3,
        output_csv="dimless_ou_data.csv",
        discretization="EM",
        levy=False,
        random_seed=None
    ):
    """
    Generate multiple realizations of the *dimensionless* Ornstein–Uhlenbeck (OU) process
    for a grid of `theta` values and (optionally) a grid of Lévy-stable alphas. Saves results to CSV.

    Dimensionless Gaussian OU:
        dY = -theta * Y dt + sqrt(2 * theta) dW,
    giving zero mean and unit stationary variance.

    Lévy–stable-driven OU (symmetric, β=0):
        Using σ_α = (α * theta)^{1/α}, the exact transition over Δt is
        Y_{n+1} = e^{-theta Δt} Y_n + (1 - e^{-α theta Δt})^{1/α} S_n,
        S_n ~ S(α, 0, 1, 0).
    For α=2 this reduces to the Gaussian OU with Z ~ N(0,1).

    Discretization options
    ----------------------
    1) Euler–Maruyama ("EM"):
       • Gaussian:   Y_{n+1} = Y_n - θ Y_n Δt + sqrt(2θ) ΔW,  ΔW ~ N(0, Δt)
       • Stable:     Y_{n+1} = Y_n - θ Y_n Δt + (αθ)^{1/α} (Δt)^{1/α} S,  S ~ S(α,0,1,0)

    2) Exact-step ("exact"):
       • Gaussian:   Y_{n+1} = e^{-θΔt} Y_n + sqrt(1 - e^{-2θΔt}) Z,  Z ~ N(0,1)
       • Stable:     Y_{n+1} = e^{-θΔt} Y_n + (1 - e^{-αθΔt})^{1/α} S,  S ~ S(α,0,1,0)

    SciPy parameterization note
    ---------------------------
    scipy.stats.levy_stable with α=2 and scale=1 returns N(0, 2).
    This code rescales such draws by 1/sqrt(2) to match N(0,1) in the α=2 boundary.

    Parameters
    ----------
    n_length : int
        Length of each generated time series. Default is 1000.
    dt : float
        Time step. Default is 0.01.
    theta_values : list of float
        Mean reversion rates. If None, defaults to np.arange(0.1, 2.1, 0.1).
    alpha_values : list of float or None
        Stability indices α if levy=True. If None and levy=True, defaults to [1.5].
        If levy=False, alpha_values is ignored and recorded as [None] for consistency.
    n_copies : int
        Number of independent realizations per (theta, alpha) pair. Default is 3.
    output_csv : str
        Output CSV path. Default is "dimless_ou_data.csv".
    discretization : str
        "EM" or "exact". Default is "EM".
    levy : bool
        Whether to use Lévy–stable increments. Default=False.
    random_seed : int or None
        If not None, base seed; each copy offsets by copy index.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns: ["theta", "alpha", "copy", "time_series"].
        "time_series" stores the series as a Python list.

    Notes
    -----
    - For very small α (e.g., 0.05), expect extremely heavy tails and large jumps; this is
      numerically delicate but sometimes informative for stress testing.
    - If levy=True but `levy_stable` is not available, normal increments are used.
    """
    if theta_values is None:
        theta_values = np.arange(0.1, 2.1, 0.1)

    if levy:
        if alpha_values is None:
            alpha_values = [1.5]
    else:
        # Keep schema consistent when levy=False
        alpha_values = [None]

    records = []

    print('Now Running')
    for theta in theta_values:
        print(f'Theta: {theta}')
        for alpha in alpha_values:
            print(f'Alpha: {alpha}')
            for copy_idx in range(n_copies):
                print(f'Copy: {copy_idx}')

                # Reproducibility across copies
                seed_for_this_copy = None
                if random_seed is not None:
                    seed_for_this_copy = random_seed + copy_idx

                ts = dimensionless_ou_process(
                    length=n_length,
                    dt=dt,
                    theta=theta,
                    discretization=discretization,
                    levy=levy,
                    levy_alpha=(alpha if alpha is not None else 1.5),
                    random_seed=seed_for_this_copy
                )
                records.append({
                    "theta": theta,
                    "alpha": alpha,
                    "copy": copy_idx,
                    "time_series": ts.tolist()
                })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    return df


def plot_dimensionless_ou_processes(
        length=1000,
        dt=0.01,
        thetas=[1.0],
        discretization="EM",
        levy=False,
        levy_alpha=1.5,
        random_seed=None
    ):
    """
    Plot multiple dimensionless Ornstein–Uhlenbeck processes with different mean-reversion
    rates (theta). Optionally, use Lévy–stable increments if levy=True (with alpha=levy_alpha).

    Under the dimensionless Gaussian OU (levy=False):
        dY = -theta * Y dt + sqrt(2 * theta) dW,
    which has zero mean and unit stationary variance.

    If levy=True, increments come from a symmetric α-stable distribution with stability parameter
    levy_alpha, using the exact or EM scheme described above.

    Parameters
    ----------
    length : int
        Number of points in each time series (default=1000).
    dt : float
        Time step between consecutive points (default=0.01).
    thetas : list of float
        Mean-reversion rates (theta) to plot (one per subplot). Default is [1.0].
    discretization : str
        "EM" (Euler–Maruyama) or "exact". Default is "EM".
    levy : bool
        Whether to use Lévy–stable increments instead of normal. Default=False.
    levy_alpha : float
        Stability parameter alpha (0 < alpha ≤ 2). Default=1.5.
    random_seed : int or None
        If not None, seeds RNGs for reproducible plots.

    Returns
    -------
    None
    """
    sns.set(style="whitegrid")

    num_plots = len(thetas)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]  # Make axes iterable if only one subplot

    time = np.arange(length) * dt

    for i, theta in enumerate(thetas):
        Y = dimensionless_ou_process(
            length=length,
            dt=dt,
            theta=theta,
            discretization=discretization,
            levy=levy,
            levy_alpha=levy_alpha,
            random_seed=random_seed
        )

        ax = axes[i]
        ax.plot(time, Y,
                label=f"OU (theta={theta}), levy={levy}, alpha={levy_alpha}")
        ax.axhline(0.0, color='r', linestyle='--', label="Mean = 0")
        ax.set_title(
            f"Dimensionless OU: discretization='{discretization}'\n"
            f"theta={theta}, dt={dt}, levy={levy}, alpha={levy_alpha}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Y(t)")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # EXAMPLE 1: Generate data (Gaussian increments) over a range of theta
    # df_gaussian = generate_dimensionless_ou_data(
    #     n_length=2000,
    #     dt=0.01,
    #     theta_values=[0.01, 1.0, 2.0],   # example
    #     alpha_values=None,              # irrelevant if levy=False
    #     n_copies=5,
    #     output_csv="dimless_ou_data_gauss.csv",
    #     discretization="exact",
    #     levy=False,
    #     random_seed=42
    # )
    # print("Generated dimensionless OU data (Gaussian). CSV: dimless_ou_data_gauss.csv")
    # print("DataFrame head:\n", df_gaussian.head())

    # EXAMPLE 2: Generate data (Lévy increments) over a range of thetas and a range of alphas
    df_levy = generate_dimensionless_ou_data(
        n_length=2000,
        dt=0.01,
        theta_values=[0.000001, 2.00, 4.00, 8.00, 16.00, 32.00],  # example
        alpha_values=[0.05, 0.5, 1.00, 1.50, 2.00],
        n_copies=2000,
        output_csv="final_ou_train_dimless_alpha_theta_ext_fin.csv",
        discretization="exact",
        levy=True,
        random_seed=137
    )
    print("Generated dimensionless OU data (Lévy), Training Seed 137. CSV: final_ou_train_dimless_alpha_theta.csv")
    print("DataFrame head:\n", df_levy.head())

    # EXAMPLE 3: Generate data (Lévy increments) for testing set
    df_levy = generate_dimensionless_ou_data(
        n_length=2000,
        dt=0.01,
        theta_values=[0.000001, 2.00, 4.00, 8.00, 16.00, 32.00],  # example
        alpha_values=[0.05, 0.5, 1.00, 1.50, 2.00],
        n_copies=2000,
        output_csv="final_ou_test_dimless_alpha_theta_ext_fin.csv",
        discretization="exact",
        levy=True,
        random_seed=3237
    )
    print("Generated dimensionless OU data (Lévy), Testing Seed 3237. CSV: final_ou_test_dimless_alpha_theta.csv")
    print("DataFrame head:\n", df_levy.head())

    import sys
    sys.exit()

