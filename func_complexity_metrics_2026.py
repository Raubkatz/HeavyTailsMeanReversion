#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
complexity_metrics_lib.py

A single-file library that collects time-series "complexity" metrics used across
your alpha/theta evaluation scripts.

Design goals
------------
- One importable file with:
  (i) optional third-party metrics (nolds, antropy, hurst, statsmodels)
  (ii) your EVT / tail metrics (Hill, Pickands, moment, POT-GPD, kurtosis, tail-rate)
  (iii) your phase-space / derivative metrics (NN-distance, mean/var of 1st/2nd derivatives)
  (iv) SVD-spectrum metrics computed on a delay-embedded matrix (default embedding_dim=3)
- Graceful fallbacks: if a package is missing, return np.nan (no hard crash).
- Short docstring per metric (what it is, key parameters, typical interpretation).
- Utility registry to standardize calling and to help your analysis scripts remain clean.

References (non-exhaustive)
---------------------------
- Extreme value theory estimators: Embrechts et al. (1997), de Haan & Ferreira (2006)
- Hurst exponent: Hurst (1951); R/S estimator; "hurst" package compute_Hc
- DFA: Peng et al. (1994); "nolds.dfa" if available
- Sample entropy: Richman & Moorman (2000); "nolds.sampen" / "antropy.sample_entropy"
- Permutation entropy: Bandt & Pompe (2002); "antropy.perm_entropy"
- Variance ratio: Lo & MacKinlay (1988)
- ADF: Dickey & Fuller (1979); statsmodels.tsa.stattools.adfuller

Notes
-----
- For SVD metrics, we form a delay-embedded matrix from the time series. By default
  we use embedding_dim=3 and then (optionally) make a square Gram matrix.
- All functions accept 1D arrays and return a scalar float (or np.nan).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Tuple, List

import numpy as np

# Core scientific stack (required)
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew
from scipy.stats import genpareto as _genpareto
from scipy.linalg import svd as _svd
import os
import pickle
from functools import lru_cache

# Optional packages (graceful fallback to NaN)
try:
    import nolds as _nolds
except Exception:
    _nolds = None

try:
    import antropy as _antropy
except Exception:
    _antropy = None

try:
    from hurst import compute_Hc as _compute_Hc
except Exception:
    _compute_Hc = None

try:
    from statsmodels.tsa.stattools import adfuller as _adfuller
except Exception:
    _adfuller = None

# CatBoost optional (only used if you explicitly call the helper)
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except Exception:
    CatBoostRegressor = None
    CatBoostClassifier = None

# Optional DFA package (Raphael Vallat). Sometimes installed as "entropy" rather than "antropy".
try:
    import entropy as _entropy
except Exception:
    _entropy = None

# =============================================================================
# CatBoost-based Hurst exponent (your ML estimator)
# =============================================================================

def _min_max_scale_01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Scale x to [0,1]. Returns zeros if constant.
    """
    z = _as_1d_float(x)
    if z.size == 0:
        return z
    mn = float(np.min(z))
    mx = float(np.max(z))
    rng = mx - mn
    if not np.isfinite(rng) or rng <= eps:
        return np.zeros_like(z, dtype=float)
    return (z - mn) / rng

def _select_hurst_model_folder_and_ws(window_size: int, stochastic_process: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Map an input window_size to the nearest supported model and its folder name.

    Supported: 10, 25, 50, 100 (same as your routine).
    stochastic_process in {"fLm","fBm","both"}.
    """
    ws = int(window_size)

    if stochastic_process not in ("fLm", "fBm", "both"):
        stochastic_process = "both"

    if ws >= 100:
        sel = 100
    elif ws >= 50:
        sel = 50
    elif ws >= 25:
        sel = 25
    elif ws >= 10:
        sel = 10
    else:
        return None, None

    folder = f"CatBoost_final_{sel}_{stochastic_process}"
    return folder, sel

@lru_cache(maxsize=32)
def _load_hurst_model_pickle_cached(model_path: str):
    """
    Cached pickle loader. The model is expected at:
      <MODELS_FOLDER>/<model_folder>/CatBoost.clf
    """
    with open(model_path, "rb") as fh:
        return pickle.load(fh)

def hurst_catboost_ml(
    x: np.ndarray,
    stochastic_process: str = "fBm",
    models_folder: str = "ML_MODELS_HURST",
) -> float:
    """
    Hurst exponent estimated by your pre-trained CatBoost model (pickled sklearn-style object).

    Behavior:
    - Selects a model based on the series length: 10/25/50/100.
    - Scales the input to [0,1] (min-max).
    - Predicts H for the single window.
    - Returns np.nan if no model fits or if loading/prediction fails.

    Parameters
    ----------
    x : array-like
        Input time series window.
    stochastic_process : {"fBm","fLm","both"}
        Which model family to use.
    models_folder : str
        Root directory containing CatBoost_final_* folders.

    Returns
    -------
    float
        Predicted Hurst exponent (np.nan if unavailable).
    """
    z = _as_1d_float(x)
    if z.size < 10:
        return np.nan

    model_folder, sel_ws = _select_hurst_model_folder_and_ws(z.size, stochastic_process)
    if model_folder is None:
        return np.nan

    model_path = os.path.join(models_folder, model_folder, "CatBoost.clf")
    if not os.path.exists(model_path):
        return np.nan

    # scale + shape to (1, sel_ws) as typical ML input
    zz = _min_max_scale_01(z)

    # If input length > selected window size, match your documented behavior:
    # use first sel_ws samples. (Alternative would be last sel_ws; pick one and keep stable.)
    if zz.size > sel_ws:
        zz = zz[:sel_ws]
    elif zz.size < sel_ws:
        # should not happen given selection, but keep safe
        return np.nan

    try:
        model = _load_hurst_model_pickle_cached(model_path)
        yhat = model.predict(zz.reshape(1, -1))
        # model.predict often returns array([val])
        return float(np.asarray(yhat).reshape(-1)[0])
    except Exception:
        return np.nan

# =============================================================================
# Small utilities
# =============================================================================

def _as_1d_float(x: Any) -> np.ndarray:
    """Convert input to finite 1D float array (drops non-finite)."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]

def _returns(x: np.ndarray) -> np.ndarray:
    """First difference of a series, with finite filtering."""
    x = _as_1d_float(x)
    if x.size < 2:
        return np.array([], dtype=float)
    r = np.diff(x)
    return r[np.isfinite(r)]

def _mad_from_median(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Absolute deviations from median, used as positive tail magnitudes."""
    x = _as_1d_float(x)
    if x.size == 0:
        return x
    return np.abs(x - np.median(x)) + eps

def choose_k(n: int, k_frac: float = 0.10, k_min: int = 5, k_max_frac: float = 0.30) -> int:
    """
    Choose tail parameter k for Hill/Pickands/moment estimators.

    Parameters
    ----------
    n : int
        Sample size.
    k_frac : float
        Fraction of n to use as tail size (default 0.10).
    k_min : int
        Minimum k (default 5).
    k_max_frac : float
        Upper cap fraction for k (default 0.30).

    Returns
    -------
    int
        Tail size k in [k_min, floor(k_max_frac*n)] and < n.
    """
    n = int(n)
    if n <= 2:
        return 1
    k0 = max(k_min, int(np.floor(k_frac * n)))
    kmax = max(k_min, int(np.floor(k_max_frac * n)))
    return int(min(k0, kmax, n - 1))


# =============================================================================
# EVT / heavy-tail metrics (alpha-related)
# =============================================================================

def hill_index(x: np.ndarray, k: Optional[int] = None) -> float:
    """
    Hill estimator for tail index (on positive magnitudes).

    Interprets heavier tails as larger Hill index (xi ~ Hill for Pareto-type tails).
    Use alpha ≈ 1/xi (with your convention) if desired downstream.

    Parameters
    ----------
    x : array-like
        Data (levels or returns). Internally uses abs(x - median).
    k : int, optional
        Tail size. If None: choose_k(n).

    Returns
    -------
    float
        Hill index estimate (np.nan if invalid).
    """
    z = _mad_from_median(x)
    n = z.size
    if n < 20:
        return np.nan
    if k is None:
        k = choose_k(n)
    k = int(k)
    if k < 2 or k >= n:
        return np.nan

    zs = np.sort(z)
    tail = zs[-k:]
    # avoid log(0)
    tail = tail + 1e-12
    x0 = tail[0]
    if not np.isfinite(x0) or x0 <= 0:
        return np.nan
    denom = np.sum(np.log(tail) - np.log(x0))
    if denom <= 0 or (not np.isfinite(denom)):
        return np.nan
    return float(k / denom)

def xi_pickands(x: np.ndarray, k: Optional[int] = None) -> float:
    """
    Pickands estimator of the extreme value index xi.

    Requires 4k < n and positive spacings in order statistics.

    Parameters
    ----------
    x : array-like
        Data; internally uses abs(x - median).
    k : int, optional
        Tail size. If None: choose_k(n).

    Returns
    -------
    float
        xi estimate (np.nan if invalid).
    """
    z = _mad_from_median(x)
    n = z.size
    if n < 20:
        return np.nan
    if k is None:
        k = choose_k(n)
    k = int(k)
    if k < 2 or (4 * k) >= n:
        return np.nan

    zs = np.sort(z)
    x_nk  = zs[-k]
    x_n2k = zs[-2 * k]
    x_n4k = zs[-4 * k]

    num = x_nk - x_n2k
    den = x_n2k - x_n4k
    if not (np.isfinite(num) and np.isfinite(den)) or num <= 0 or den <= 0:
        return np.nan

    return float((1.0 / np.log(2.0)) * np.log(num / den))

def xi_moment(x: np.ndarray, k: Optional[int] = None) -> float:
    """
    Dekkers–Einmahl–de Haan moment estimator of xi.

    Parameters
    ----------
    x : array-like
        Data; internally uses abs(x - median).
    k : int, optional
        Tail size. If None: choose_k(n).

    Returns
    -------
    float
        xi estimate (np.nan if invalid).
    """
    z = _mad_from_median(x)
    n = z.size
    if n < 20:
        return np.nan
    if k is None:
        k = choose_k(n)
    k = int(k)
    if k < 2 or k >= n:
        return np.nan

    zs = np.sort(z)
    thresh = zs[-k]
    if not np.isfinite(thresh) or thresh <= 0:
        return np.nan

    tail = zs[-k:]
    y = np.log(tail / thresh)
    if not np.all(np.isfinite(y)):
        return np.nan

    M1 = float(np.mean(y))
    M2 = float(np.mean(y ** 2))
    if M2 <= 0 or (not np.isfinite(M2)):
        return np.nan

    denom = (1.0 - (M1 * M1) / M2)
    if denom <= 0 or (not np.isfinite(denom)):
        return np.nan

    return float(M1 + 1.0 - 0.5 * (1.0 / denom))

def xi_pot_gpd(x: np.ndarray, q: float = 0.95, min_exc: int = 10) -> float:
    """
    Peaks-over-threshold (POT) using a GPD fit on exceedances.

    Parameters
    ----------
    x : array-like
        Data; internally uses abs(x - median).
    q : float
        Quantile for threshold u (default 0.95).
    min_exc : int
        Minimum number of exceedances required.

    Returns
    -------
    float
        xi estimate (GPD shape parameter), np.nan if invalid.
    """
    z = _mad_from_median(x)
    n = z.size
    if n < 50:
        return np.nan

    u = float(np.quantile(z, q))
    exc = z[z > u] - u
    exc = exc[np.isfinite(exc)]
    if exc.size < int(min_exc):
        return np.nan

    try:
        c, loc, scale = _genpareto.fit(exc, floc=0.0)
        if not np.isfinite(c):
            return np.nan
        return float(c)
    except Exception:
        return np.nan

def tail_rate(x: np.ndarray, q: float = 0.95) -> float:
    """
    Tail exceedance rate: P(|x - median| > quantile_q).

    This is a simple heavy-tail proxy. Larger values indicate more mass in the tail.

    Parameters
    ----------
    x : array-like
        Data.
    q : float
        Tail quantile threshold (default 0.95).

    Returns
    -------
    float
        Exceedance fraction in [0,1] (np.nan if invalid).
    """
    z = _mad_from_median(x)
    if z.size < 8:
        return np.nan
    thr = float(np.quantile(z, q))
    return float(np.mean(z > thr))

def skewness(x: np.ndarray) -> float:
    """Skewness of x (scipy.stats.skew)."""
    z = _as_1d_float(x)
    if z.size < 8:
        return np.nan
    return float(_skew(z, bias=False))

def kurtosis_excess(x: np.ndarray) -> float:
    """Excess kurtosis of x (scipy.stats.kurtosis with fisher=True)."""
    z = _as_1d_float(x)
    if z.size < 8:
        return np.nan
    return float(_kurtosis(z, fisher=True, bias=False))

# =============================================================================
# Mean-reversion metrics (theta-related)
# =============================================================================

def hurst_h(
    x: np.ndarray,
    kind: str = "auto",
    use_returns_when_auto: bool = True,
) -> float:
    """
    Hurst exponent H via hurst.compute_Hc.

    IMPORTANT: The hurst package requires selecting a consistent "kind":
      - 'change': raw changes
      - 'random_walk': cumulative sum of changes
      - 'price': cumulative product of positive multiplicative changes
    See package README for details. :contentReference[oaicite:2]{index=2}

    For OU-type windows (levels) and for scaled windows, 'random_walk' is typically the safest choice.
    For returns/increments, 'change' is typically appropriate.

    Parameters
    ----------
    kind : {'auto','change','random_walk','price'}
        If 'auto':
          - if use_returns_when_auto=True: run on diff(x) with kind='change'
          - else: run on x with kind='random_walk'
    use_returns_when_auto : bool
        Only used when kind='auto'.

    Returns
    -------
    float
        H estimate, or np.nan.
    """
    if _compute_Hc is None:
        return np.nan

    z = _as_1d_float(x)
    if z.size < 40:
        return np.nan

    # auto selection tuned for OU analysis:
    if kind == "auto":
        if use_returns_when_auto:
            r = _returns(z)
            if r.size < 40:
                return np.nan
            z_use = r
            kind_use = "change"
        else:
            z_use = z
            kind_use = "random_walk"
    else:
        z_use = z
        kind_use = kind

    # Guard against constant/near-constant series (common after scaling in short windows)
    if np.nanstd(z_use) <= 0:
        return np.nan

    try:
        H, c, data = _compute_Hc(z_use, kind=kind_use, simplified=True)
        if not np.isfinite(H):
            return np.nan
        return float(H)
    except Exception:
        return np.nan

def dfa_alpha(x: np.ndarray) -> float:
    """
    Detrended fluctuation analysis (DFA) exponent.

    Preferred implementation:
      - entropy.detrended_fluctuation (Raphael Vallat) if available :contentReference[oaicite:4]{index=4}
    Fallback:
      - nolds.dfa if available

    Returns
    -------
    float
        DFA exponent, or np.nan.
    """
    z = _as_1d_float(x)
    if z.size < 40:
        return np.nan

    # Constant windows cause many DFA implementations to fail.
    if np.nanstd(z) <= 0:
        return np.nan

    # Prefer Vallat DFA if present (installed as entropy or antropy depending on environment)
    try:
        if _entropy is not None and hasattr(_entropy, "detrended_fluctuation"):
            v = _entropy.detrended_fluctuation(z)
            return float(v) if np.isfinite(v) else np.nan
    except Exception:
        pass

    try:
        if _antropy is not None and hasattr(_antropy, "detrended_fluctuation"):
            v = _antropy.detrended_fluctuation(z)
            return float(v) if np.isfinite(v) else np.nan
    except Exception:
        pass

    # Fallback: nolds
    if _nolds is None:
        return np.nan
    try:
        v = _nolds.dfa(z)
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def lag1_autocorr(x: np.ndarray) -> float:
    """
    Lag-1 autocorrelation (levels): corr(x_t, x_{t-1}).

    More negative / smaller values indicate stronger mean reversion.
    """
    z = _as_1d_float(x)
    if z.size < 3:
        return np.nan
    x0 = z[:-1]
    x1 = z[1:]
    v0 = float(np.var(x0))
    if v0 <= 0 or (not np.isfinite(v0)):
        return np.nan
    c01 = float(np.cov(x0, x1, ddof=0)[0, 1])
    return float(c01 / v0)

def ar1_phi(x: np.ndarray) -> float:
    """
    AR(1) phi estimate by OLS slope:
      phi = cov(x_{t-1}, x_t) / var(x_{t-1})

    phi closer to 0 implies faster mean reversion (in AR(1)/OU mapping).
    """
    z = _as_1d_float(x)
    if z.size < 5:
        return np.nan
    x_prev = z[:-1]
    x_curr = z[1:]
    var = float(np.var(x_prev))
    if var <= 0 or (not np.isfinite(var)):
        return np.nan
    cov = float(np.cov(x_prev, x_curr, ddof=0)[0, 1])
    return float(cov / var)

def theta_from_phi(x: np.ndarray, dt: float = 1.0) -> float:
    """
    OU-like theta proxy using phi ≈ exp(-theta*dt) => theta ≈ -ln(phi)/dt.

    Valid only for phi in (0,1).
    """
    phi = ar1_phi(x)
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return np.nan
    return float(-np.log(phi) / float(dt))

def half_life_from_phi(x: np.ndarray, dt: float = 1.0) -> float:
    """
    Half-life from phi:
      half_life = ln(2) / theta  with theta from theta_from_phi.

    Returns np.nan if invalid.
    """
    th = theta_from_phi(x, dt=dt)
    if not np.isfinite(th) or th <= 0:
        return np.nan
    return float(np.log(2.0) / th)

def variance_ratio(x: np.ndarray, q: int = 2) -> float:
    """
    Variance ratio VR(q) on increments:
      r_t = diff(x)
      VR(q) = Var(sum_{i=1..q} r_{t-i+1}) / (q * Var(r_t))

    VR(q) < 1 suggests negative serial correlation (mean reversion).
    """
    z = _as_1d_float(x)
    if z.size < (q + 3):
        return np.nan
    r = np.diff(z)
    r = r[np.isfinite(r)]
    if r.size < (q + 2):
        return np.nan
    var_r = float(np.var(r, ddof=0))
    if var_r <= 0 or (not np.isfinite(var_r)):
        return np.nan
    rq = np.convolve(r, np.ones(int(q), dtype=float), mode="valid")
    var_rq = float(np.var(rq, ddof=0))
    return float(var_rq / (float(q) * var_r))

def adf_stat(x: np.ndarray, autolag: str = "AIC") -> float:
    """
    Augmented Dickey-Fuller test statistic via statsmodels.adfuller.

    More negative values suggest stronger evidence of stationarity.
    """
    if _adfuller is None:
        return np.nan
    z = _as_1d_float(x)
    if z.size < 30:
        return np.nan
    try:
        res = _adfuller(z, autolag=autolag)
        return float(res[0])
    except Exception:
        return np.nan

# =============================================================================
# Phase-space embedding / derivative metrics (your code, consolidated)
# =============================================================================

from sklearn.neighbors import NearestNeighbors
from copy import deepcopy as dc

def delay_embed(data: np.ndarray, tau: int = 10, max_dim: int = 3) -> np.ndarray:
    """
    Delay embedding of a 1D time series.

    Parameters
    ----------
    data : array-like
        Input series.
    tau : int
        Delay (samples).
    max_dim : int
        Embedding dimension.

    Returns
    -------
    np.ndarray
        Embedded trajectory, shape (n_samples, max_dim).
    """
    x = _as_1d_float(data)
    if type(tau) is not int:
        tau = int(tau)
    if type(max_dim) is not int:
        max_dim = int(max_dim)
    if x.size <= tau * (max_dim - 1):
        return np.empty((0, max_dim), dtype=float)

    num_samples = x.size - tau * (max_dim - 1)
    return np.array([x[dim * tau:num_samples + dim * tau] for dim in range(max_dim)]).T[:, ::-1]

def hessian(x: np.ndarray) -> np.ndarray:
    """Second derivative along embedded trajectory."""
    return dc(np.gradient(np.gradient(x, axis=0), axis=0))

def grad_ps(x: np.ndarray) -> np.ndarray:
    """First derivative along embedded trajectory."""
    return dc(np.gradient(x, axis=0))

def mean_2nd_derivative(data: np.ndarray, tau: int = 10, fixed_dim: int = 3, cut: bool = False) -> float:
    """Mean Euclidean norm of 2nd derivative along embedded trajectory."""
    data_vec = delay_embed(data, tau=tau, max_dim=fixed_dim)
    if data_vec.shape[0] < 3:
        return np.nan
    if cut:
        data_vec = dc(data_vec[tau * fixed_dim:])
        if data_vec.shape[0] < 3:
            return np.nan
    v = np.sqrt(np.sum(np.square(hessian(data_vec)), axis=1))
    return float(np.mean(v)) if v.size > 0 else np.nan

def var_2nd_derivative(data: np.ndarray, tau: int = 1, fixed_dim: int = 3, cut: bool = False) -> float:
    """Variance of Euclidean norm of 2nd derivative along embedded trajectory."""
    data_vec = delay_embed(data, tau=tau, max_dim=fixed_dim)
    if data_vec.shape[0] < 3:
        return np.nan
    if cut:
        data_vec = dc(data_vec[tau * fixed_dim:])
        if data_vec.shape[0] < 3:
            return np.nan
    v = np.sqrt(np.sum(np.square(hessian(data_vec)), axis=1))
    return float(np.var(v)) if v.size > 0 else np.nan

def mean_1st_derivative(data: np.ndarray, tau: int = 10, fixed_dim: int = 3, cut: bool = False) -> float:
    """Mean Euclidean norm of 1st derivative along embedded trajectory."""
    data_vec = delay_embed(data, tau=tau, max_dim=fixed_dim)
    if data_vec.shape[0] < 3:
        return np.nan
    if cut:
        data_vec = dc(data_vec[tau * fixed_dim:])
        if data_vec.shape[0] < 3:
            return np.nan
    v = np.sqrt(np.sum(np.square(grad_ps(data_vec)), axis=1))
    return float(np.mean(v)) if v.size > 0 else np.nan

def var_1st_derivative(data: np.ndarray, tau: int = 10, fixed_dim: int = 3, cut: bool = False) -> float:
    """Variance of Euclidean norm of 1st derivative along embedded trajectory."""
    data_vec = delay_embed(data, tau=tau, max_dim=fixed_dim)
    if data_vec.shape[0] < 3:
        return np.nan
    if cut:
        data_vec = dc(data_vec[tau * fixed_dim:])
        if data_vec.shape[0] < 3:
            return np.nan
    v = np.sqrt(np.sum(np.square(grad_ps(data_vec)), axis=1))
    return float(np.var(v)) if v.size > 0 else np.nan


# =============================================================================
# Package-based complexity metrics (nolds / antropy)
# =============================================================================

def _is_effectively_constant(z: np.ndarray, eps: float = 1e-12) -> bool:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return True
    return (np.nanmax(z) - np.nanmin(z)) <= eps

def antropy_sample_entropy(
    x: np.ndarray,
    order: int = 2,
    metric: str = "chebyshev",
    normalize_input: bool = True,
    min_n: int = 50,
) -> float:
    """
    Sample entropy via antropy.sample_entropy.

    Notes
    -----
    - AntroPy exposes parameters such as `order` and `metric` for sample entropy. :contentReference[oaicite:3]{index=3}
    - For OU windows of length ~100, SampEn can be unstable on near-constant series.
      We return 0.0 for effectively constant windows.

    Parameters
    ----------
    order : int
        Embedding dimension m.
    metric : str
        Distance metric used internally (AntroPy uses this in its API).
    normalize_input : bool
        If True: z-score normalize before computing (reduces scale sensitivity).
    min_n : int
        Minimum length required to attempt computation.

    Returns
    -------
    float
        Sample entropy, or np.nan if unavailable/invalid.
    """
    if _antropy is None:
        return np.nan

    z = _as_1d_float(x)
    if z.size < int(min_n):
        return np.nan

    if _is_effectively_constant(z):
        return 0.0

    if normalize_input:
        mu = float(np.mean(z))
        sd = float(np.std(z))
        if not np.isfinite(sd) or sd <= 0:
            return 0.0
        z = (z - mu) / sd

    try:
        # AntroPy signature includes order and metric. :contentReference[oaicite:4]{index=4}
        v = _antropy.sample_entropy(z, order=int(order), metric=str(metric))
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def antropy_perm_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
    normalize_input: bool = False,
    min_n: int = 30,
) -> float:
    """
    Permutation entropy via antropy.perm_entropy. :contentReference[oaicite:5]{index=5}

    Key constraint
    --------------
    Embedded vectors require enough samples: n - (order-1)*delay must be >= 2 (practically more).
    AntroPy defines the embedding explicitly in its documentation. :contentReference[oaicite:6]{index=6}

    Parameters
    ----------
    order : int
        Permutation order.
    delay : int
        Time delay (lag).
    normalize : bool
        If True: normalize by log2(order!) to [0,1]. :contentReference[oaicite:7]{index=7}
    normalize_input : bool
        Optional z-score normalization (usually not necessary for ordinal patterns).
    min_n : int
        Minimum length required to attempt computation.

    Returns
    -------
    float
        Permutation entropy, or np.nan if invalid.
    """
    if _antropy is None:
        return np.nan

    z = _as_1d_float(x)
    if z.size < int(min_n):
        return np.nan

    if _is_effectively_constant(z):
        # For strictly monotone / constant sequences, perm entropy can be ~0. :contentReference[oaicite:8]{index=8}
        return 0.0

    if normalize_input:
        mu = float(np.mean(z))
        sd = float(np.std(z))
        if np.isfinite(sd) and sd > 0:
            z = (z - mu) / sd

    order = int(order)
    delay = int(delay)

    # Ensure feasible embedding for this window length
    max_delay = int((z.size - 2) // max(1, (order - 1)))
    if max_delay < 1:
        return np.nan
    if delay > max_delay:
        delay = max_delay

    # AntroPy supports delay as an int or list; we keep an int for stability. :contentReference[oaicite:9]{index=9}
    try:
        v = _antropy.perm_entropy(z, order=order, delay=delay, normalize=bool(normalize))
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def antropy_spectral_entropy(
    x: np.ndarray,
    sf: float = 1.0,
    method: str = "welch",
    normalize: bool = True,
    min_n: int = 64,
) -> float:
    """
    Spectral entropy via antropy.spectral_entropy. :contentReference[oaicite:10]{index=10}

    Practical handling for short windows
    -----------------------------------
    For short windows, Welch can be numerically awkward. For n < 128, switching to FFT is often
    more stable. AntroPy exposes `method` and `sf` in its API. :contentReference[oaicite:11]{index=11}

    Parameters
    ----------
    sf : float
        Sampling frequency (use 1.0 if unknown and you only compare within-dataset).
    method : str
        "welch" or "fft" (AntroPy API). :contentReference[oaicite:12]{index=12}
    normalize : bool
        Normalize entropy to [0,1] if True.
    min_n : int
        Minimum length required to attempt computation.

    Returns
    -------
    float
        Spectral entropy, or np.nan if invalid.
    """
    if _antropy is None:
        return np.nan

    z = _as_1d_float(x)
    if z.size < int(min_n):
        return np.nan

    if _is_effectively_constant(z):
        return 0.0

    # Use FFT for short windows to avoid Welch edge cases
    m = str(method).lower()
    if z.size < 128 and m == "welch":
        m = "fft"

    try:
        v = _antropy.spectral_entropy(z, sf=float(sf), method=m, normalize=bool(normalize))
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


# =============================================================================
# SVD-spectrum metrics on delay-embedded matrices (time-series -> matrix -> SVD)
# =============================================================================

def _make_square_matrix(M: np.ndarray) -> np.ndarray:
    """Convert rectangular matrix to square Gram matrix, or keep if already square."""
    if M.ndim != 2:
        return M
    if M.shape[0] < M.shape[1]:
        return M @ M.T
    if M.shape[1] < M.shape[0]:
        return M.T @ M
    return M

def embedded_matrix(
    x: np.ndarray,
    embedding_dim: int = 3,
    tau: int = 1,
    make_square: bool = True
) -> np.ndarray:
    """
    Create delay-embedded matrix for SVD metrics.

    Parameters
    ----------
    embedding_dim : int
        Embedding dimension (default 3).
    tau : int
        Delay.
    make_square : bool
        If True, use Gram matrix to make it square.

    Returns
    -------
    np.ndarray
        Matrix (possibly square). Empty if insufficient data.
    """
    M = delay_embed(x, tau=tau, max_dim=embedding_dim)
    if M.size == 0:
        return M
    return _make_square_matrix(M) if make_square else M

def _singular_values_of_embedded(
    x: np.ndarray,
    embedding_dim: int = 3,
    tau: int = 1,
    make_square: bool = True
) -> np.ndarray:
    M = embedded_matrix(x, embedding_dim=embedding_dim, tau=tau, make_square=make_square)
    if M.size == 0:
        return np.array([], dtype=float)
    s = _svd(M, compute_uv=False)
    s = s[np.isfinite(s)]
    return s

def sv_entropy(x: np.ndarray, embedding_dim: int = 3, tau: int = 1, make_square: bool = True) -> float:
    """
    Shannon entropy of singular values (normalized to a probability vector).
    """
    s = _singular_values_of_embedded(x, embedding_dim, tau, make_square)
    if s.size < 2:
        return np.nan
    p = s / np.sum(s) if np.sum(s) > 0 else None
    if p is None or np.any(p <= 0):
        return np.nan
    return float(-np.sum(p * np.log(p)))

def sv_condition_number(x: np.ndarray, embedding_dim: int = 3, tau: int = 1, make_square: bool = True) -> float:
    """Condition number max(s)/min(s) for embedded matrix."""
    s = _singular_values_of_embedded(x, embedding_dim, tau, make_square)
    if s.size < 2:
        return np.nan
    smin = float(np.min(s))
    smax = float(np.max(s))
    if smin <= 0 or (not np.isfinite(smin)) or (not np.isfinite(smax)):
        return np.nan
    return float(smax / smin)

def sv_spectral_flatness(x: np.ndarray, embedding_dim: int = 3, tau: int = 1, make_square: bool = True) -> float:
    """
    Spectral flatness (geometric mean / arithmetic mean) of singular values.
    """
    s = _singular_values_of_embedded(x, embedding_dim, tau, make_square)
    if s.size < 2:
        return np.nan
    s = s[s > 0]
    if s.size < 2:
        return np.nan
    gm = float(np.exp(np.mean(np.log(s))))
    am = float(np.mean(s))
    if am <= 0 or (not np.isfinite(gm)) or (not np.isfinite(am)):
        return np.nan
    return float(gm / am)

def sv_effective_rank(x: np.ndarray, embedding_dim: int = 3, tau: int = 1, make_square: bool = True) -> float:
    """
    Effective rank: exp(H(p)) where p = s/sum(s), H is Shannon entropy.
    """
    s = _singular_values_of_embedded(x, embedding_dim, tau, make_square)
    if s.size < 2:
        return np.nan
    p = s / np.sum(s) if np.sum(s) > 0 else None
    if p is None or np.any(p <= 0):
        return np.nan
    H = float(-np.sum(p * np.log(p)))
    return float(np.exp(H))


# =============================================================================
# CatBoost helper (optional)
# =============================================================================

def load_catboost_model(model_path: str):
    """
    Load a CatBoost model from disk.

    Returns
    -------
    object
        CatBoostClassifier or CatBoostRegressor instance.

    Notes
    -----
    This helper exists so the alpha/theta scripts can import a single library.
    """
    if (CatBoostClassifier is None) and (CatBoostRegressor is None):
        raise ImportError("catboost is not installed in this environment.")

    # Try classifier first (works for .cbm classifiers in your pipeline)
    try:
        m = CatBoostClassifier()
        m.load_model(model_path)
        return m
    except Exception:
        m = CatBoostRegressor()
        m.load_model(model_path)
        return m


# =============================================================================
# Metric registry
# =============================================================================

@dataclass(frozen=True)
class MetricSpec:
    name: str
    func: Callable[..., float]
    family: str
    doc: str
    defaults: Dict[str, Any]

def get_metric_registry() -> Dict[str, MetricSpec]:
    """
    Registry of metrics used by your analysis scripts.

    Returns
    -------
    dict
        name -> MetricSpec
    """
    reg: Dict[str, MetricSpec] = {}

    # Heavy-tail / EVT / moments
    reg["hill_index"] = MetricSpec(
        name="hill_index",
        func=hill_index,
        family="tail",
        doc="Hill tail index on abs(x - median). Larger => heavier tail.",
        defaults={}
    )
    reg["xi_pickands"] = MetricSpec(
        name="xi_pickands",
        func=xi_pickands,
        family="tail",
        doc="Pickands EVI xi on abs(x - median).",
        defaults={}
    )
    reg["xi_moment"] = MetricSpec(
        name="xi_moment",
        func=xi_moment,
        family="tail",
        doc="Moment EVI xi on abs(x - median).",
        defaults={}
    )
    reg["xi_pot_gpd"] = MetricSpec(
        name="xi_pot_gpd",
        func=xi_pot_gpd,
        family="tail",
        doc="POT GPD xi (shape) on abs(x - median).",
        defaults={"q": 0.95, "min_exc": 10}
    )
    reg["tail_rate"] = MetricSpec(
        name="tail_rate",
        func=tail_rate,
        family="tail",
        doc="Exceedance rate of abs(x - median) above quantile q.",
        defaults={"q": 0.95}
    )
    reg["skewness"] = MetricSpec(
        name="skewness",
        func=skewness,
        family="moments",
        doc="Skewness of the series.",
        defaults={}
    )
    reg["kurtosis_excess"] = MetricSpec(
        name="kurtosis_excess",
        func=kurtosis_excess,
        family="moments",
        doc="Excess kurtosis (Fisher).",
        defaults={}
    )
    reg["hurst_h"] = MetricSpec(
        name="hurst_h",
        func=hurst_h,
        family="mean_reversion",
        doc="Hurst exponent via hurst.compute_Hc. Use kind='random_walk' for OU-like level windows or kind='change' for returns. :contentReference[oaicite:6]{index=6}",
        defaults={"kind": "random_walk"}  # or {"kind":"auto","use_returns_when_auto":False}
    )
    reg["dfa_alpha"] = MetricSpec(
        name="dfa_alpha",
        func=dfa_alpha,
        family="mean_reversion",
        doc="DFA exponent via entropy/antropy detrended_fluctuation when available, else nolds.dfa. :contentReference[oaicite:7]{index=7}",
        defaults={}
    )
    reg["lag1_autocorr"] = MetricSpec(
        name="lag1_autocorr",
        func=lag1_autocorr,
        family="mean_reversion",
        doc="Lag-1 autocorrelation of levels.",
        defaults={}
    )
    reg["ar1_phi"] = MetricSpec(
        name="ar1_phi",
        func=ar1_phi,
        family="mean_reversion",
        doc="AR(1) phi from OLS slope.",
        defaults={}
    )
    reg["theta_from_phi"] = MetricSpec(
        name="theta_from_phi",
        func=theta_from_phi,
        family="mean_reversion",
        doc="OU-like theta proxy from AR(1) phi.",
        defaults={"dt": 1.0}
    )
    reg["half_life_from_phi"] = MetricSpec(
        name="half_life_from_phi",
        func=half_life_from_phi,
        family="mean_reversion",
        doc="Half-life derived from theta_from_phi.",
        defaults={"dt": 1.0}
    )
    reg["variance_ratio_q2"] = MetricSpec(
        name="variance_ratio_q2",
        func=lambda x: variance_ratio(x, q=2),
        family="mean_reversion",
        doc="Variance ratio VR(2) on increments.",
        defaults={}
    )
    reg["variance_ratio_q5"] = MetricSpec(
        name="variance_ratio_q5",
        func=lambda x: variance_ratio(x, q=5),
        family="mean_reversion",
        doc="Variance ratio VR(5) on increments.",
        defaults={}
    )
    reg["adf_stat"] = MetricSpec(
        name="adf_stat",
        func=adf_stat,
        family="mean_reversion",
        doc="ADF test statistic via statsmodels (if available).",
        defaults={"autolag": "AIC"}
    )
    reg["antropy_sample_entropy"] = MetricSpec(
        name="antropy_sample_entropy",
        func=antropy_sample_entropy,
        family="entropy",
        doc="Sample entropy via antropy.sample_entropy (if available).",
        defaults={}
    )
    reg["antropy_sample_entropy"] = MetricSpec(
        name="antropy_sample_entropy",
        func=antropy_sample_entropy,
        family="entropy",
        doc="Sample entropy via antropy.sample_entropy with stable defaults for short windows.",
        defaults={"order": 2, "metric": "chebyshev", "normalize_input": True, "min_n": 50}
    )

    reg["antropy_perm_entropy"] = MetricSpec(
        name="antropy_perm_entropy",
        func=antropy_perm_entropy,
        family="entropy",
        doc="Permutation entropy via antropy.perm_entropy with feasibility checks for order/delay.",
        defaults={"order": 3, "delay": 1, "normalize": True, "normalize_input": False, "min_n": 30}
    )
    reg["antropy_spectral_entropy"] = MetricSpec(
        name="antropy_spectral_entropy",
        func=antropy_spectral_entropy,
        family="entropy",
        doc="Spectral entropy via antropy.spectral_entropy; uses FFT fallback for short windows.",
        defaults={"sf": 1.0, "method": "welch", "normalize": True, "min_n": 64}
    )
    reg["mean_1st_derivative"] = MetricSpec(
        name="mean_1st_derivative",
        func=mean_1st_derivative,
        family="phase_space",
        doc="Mean norm of 1st derivative along embedded trajectory.",
        defaults={"tau": 10, "fixed_dim": 3, "cut": False}
    )
    reg["var_1st_derivative"] = MetricSpec(
        name="var_1st_derivative",
        func=var_1st_derivative,
        family="phase_space",
        doc="Variance of norm of 1st derivative along embedded trajectory.",
        defaults={"tau": 10, "fixed_dim": 3, "cut": False}
    )
    reg["mean_2nd_derivative"] = MetricSpec(
        name="mean_2nd_derivative",
        func=mean_2nd_derivative,
        family="phase_space",
        doc="Mean norm of 2nd derivative along embedded trajectory.",
        defaults={"tau": 10, "fixed_dim": 3, "cut": False}
    )
    reg["var_2nd_derivative"] = MetricSpec(
        name="var_2nd_derivative",
        func=var_2nd_derivative,
        family="phase_space",
        doc="Variance of norm of 2nd derivative along embedded trajectory.",
        defaults={"tau": 1, "fixed_dim": 3, "cut": False}
    )
    reg["sv_entropy"] = MetricSpec(
        name="sv_entropy",
        func=lambda x: sv_entropy(x, embedding_dim=3, tau=1, make_square=True),
        family="svd",
        doc="Shannon entropy of singular values on embedded matrix.",
        defaults={}
    )
    reg["sv_condition_number"] = MetricSpec(
        name="sv_condition_number",
        func=lambda x: sv_condition_number(x, embedding_dim=3, tau=1, make_square=True),
        family="svd",
        doc="Condition number of embedded matrix.",
        defaults={}
    )
    reg["sv_spectral_flatness"] = MetricSpec(
        name="sv_spectral_flatness",
        func=lambda x: sv_spectral_flatness(x, embedding_dim=3, tau=1, make_square=True),
        family="svd",
        doc="Spectral flatness of singular values on embedded matrix.",
        defaults={}
    )
    reg["sv_effective_rank"] = MetricSpec(
        name="sv_effective_rank",
        func=lambda x: sv_effective_rank(x, embedding_dim=3, tau=1, make_square=True),
        family="svd",
        doc="Effective rank of embedded matrix.",
        defaults={}
    )
    reg["hurst_catboost_ml"] = MetricSpec(
        name="hurst_catboost_ml",
        func=hurst_catboost_ml,
        family="ml",
        doc="Hurst exponent via pre-trained CatBoost model. Params: stochastic_process, models_folder.",
        defaults={"stochastic_process": "fBm", "models_folder": "ML_MODELS_HURST"}
    )


    return reg


def compute_metric(name: str, x: np.ndarray, **kwargs) -> float:
    """
    Compute a single metric by registry name.

    Parameters
    ----------
    name : str
        Metric name (see get_metric_registry()).
    x : array-like
        Input series.
    kwargs :
        Overrides defaults.

    Returns
    -------
    float
        Metric value (np.nan if invalid).
    """
    reg = get_metric_registry()
    if name not in reg:
        raise KeyError(f"Unknown metric '{name}'.")
    spec = reg[name]
    params = dict(spec.defaults)
    params.update(kwargs)
    try:
        return float(spec.func(x, **params))
    except TypeError:
        # for lambda-wrapped metrics without kwargs
        return float(spec.func(x))
    except Exception:
        return np.nan


def compute_metrics(
    x: np.ndarray,
    metric_names: Optional[List[str]] = None,
    on_returns: bool = False,
    **kwargs
) -> Dict[str, float]:
    """
    Compute multiple metrics for a single series.

    Parameters
    ----------
    x : array-like
        Input series.
    metric_names : list[str], optional
        Which metrics to compute. If None: compute all registry metrics.
    on_returns : bool
        If True, compute metrics on diff(x) instead of x.
    kwargs :
        Global kwargs passed to compute_metric (only applies if compatible).

    Returns
    -------
    dict
        metric_name -> value
    """
    reg = get_metric_registry()
    names = list(reg.keys()) if metric_names is None else list(metric_names)

    data = _returns(x) if on_returns else _as_1d_float(x)

    out: Dict[str, float] = {}
    for name in names:
        out[name] = compute_metric(name, data, **kwargs)
    return out
