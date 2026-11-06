"""
Self-Consistent Agreement Score (SCAS) and comparison metrics.

Notation (matches the revised manuscript)
-----------------------------------------
y      : observed values (length N)
yhat   : predicted values (length N)
eps_i  = |y_i - yhat_i|           absolute residuals
sigma  : scale of the observed values; by default the sample standard
         deviation (ddof=1), optionally the normalised MAD.

SCAS is the crossing point
    tau* = sup{ tau in [0,1] : f(tau) >= tau },
    f(tau) = (1/N) * #{ i : eps_i < (1 - tau) * sigma }.

Exact finite-sample algorithm (order statistics)
------------------------------------------------
Sort eps_(1) <= ... <= eps_(N) and put c_j = 1 - eps_(j)/sigma.
Let J = max{ j : eps_(j) < (1 - (j-1)/N) * sigma } (J = 0 if empty).
Then
    tau* = min( J/N , c_J ) ,            tau* = 0 if J = 0,
and the bracketing  eps_(J) <= (1 - tau*) sigma <= eps_(J+1)  holds
(with eps_(N+1) = +inf).  The fixed point f(tau*) = tau* holds exactly
iff c_J > J/N (strict); otherwise f(tau*) < tau* <= f(tau*-).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy import stats

# --------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------

def mad_scale(x: np.ndarray) -> float:
    """Normalised median absolute deviation (consistent for the SD under normality)."""
    x = np.asarray(x, dtype=float)
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def scale_of(y: np.ndarray, scale: str = "sd") -> float:
    """Scale of the observed values. 'sd' = sample SD (ddof=1), 'mad' = 1.4826*MAD."""
    y = np.asarray(y, dtype=float)
    if scale == "sd":
        s = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    elif scale == "mad":
        s = mad_scale(y)
    else:
        raise ValueError(f"unknown scale '{scale}'")
    if not np.isfinite(s) or s <= 0:
        # degenerate target: fall back to the other estimator, then to 1
        s = mad_scale(y) if scale == "sd" else float(np.std(y, ddof=1))
        if not np.isfinite(s) or s <= 0:
            s = 1.0
    return s


# --------------------------------------------------------------------------
# SCAS
# --------------------------------------------------------------------------

def f_tau(tau, eps: np.ndarray, sigma: float):
    """Right-hand side f(tau) of the fixed-point equation (vectorised in tau)."""
    tau = np.asarray(tau, dtype=float)
    thr = (1.0 - tau) * sigma
    return np.mean(eps[None, ...] < thr[..., None], axis=-1) if tau.ndim else float(np.mean(eps < thr))


def scas_exact(y, yhat, scale: str = "sd", sigma: float | None = None, return_details: bool = False):
    """
    Exact SCAS by the order-statistic algorithm (O(N log N)).

    Parameters
    ----------
    y, yhat : array-like
        Observed and predicted values.
    scale : 'sd' | 'mad'
        Scale estimator applied to y (ignored if sigma is given).
    sigma : float, optional
        Externally supplied scale (e.g. a known population SD).
    return_details : bool
        If True also return a dict with J, sigma, the threshold and whether
        the fixed point holds exactly.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError("y and yhat must have the same shape")
    N = y.size
    eps = np.sort(np.abs(y - yhat))
    sig = scale_of(y, scale) if sigma is None else float(sigma)

    j = np.arange(1, N + 1)
    admissible = eps < (1.0 - (j - 1) / N) * sig         # eps_(j) < (1-(j-1)/N) sigma
    if not admissible.any():
        tau = 0.0
        J = 0
    else:
        J = int(np.max(j[admissible]))                    # admissible set is {1,...,J}
        cJ = 1.0 - eps[J - 1] / sig
        tau = float(min(J / N, cJ))
    if not return_details:
        return tau
    thr = (1.0 - tau) * sig
    details = {
        "J": J,
        "sigma": sig,
        "threshold": thr,
        "k_star": 1.0 - tau,                               # implied tolerance k* = 1 - tau*
        "exact_fixed_point": bool(J > 0 and (1.0 - eps[J - 1] / sig) > J / N),
        "eps_J": eps[J - 1] if J > 0 else np.nan,
        "eps_J1": eps[J] if J < N else np.inf,
    }
    return tau, details


def scas_brent(y, yhat, scale: str = "sd", sigma: float | None = None, xtol: float = 1e-10):
    """
    SCAS via Brent's method on g(tau) = f(tau) - tau (the procedure used in the
    first submission). Because g is a step function, Brent converges to the
    jump location; it is kept only as a numerical cross-check of scas_exact.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    eps = np.abs(y - yhat)
    sig = scale_of(y, scale) if sigma is None else float(sigma)

    def g(t):
        thr = (1.0 - t) * sig
        return float(np.mean(eps < thr)) - t

    if g(0.0) < 0:
        return 0.0
    if g(1.0) >= 0:                                         # only if g(1)=0, impossible (eps<0)
        return 1.0
    # g(1) = -1 < 0 always; g(0) >= 0. Brent locates the sign change.
    return float(brentq(g, 0.0, 1.0, xtol=xtol))


def scas(y, yhat, scale: str = "sd", sigma: float | None = None) -> float:
    """SCAS (exact algorithm). Alias of scas_exact without details."""
    return scas_exact(y, yhat, scale=scale, sigma=sigma)


def scas_curve(y, yhat, scale: str = "sd", n_grid: int = 1001):
    """Grid of (tau, f(tau)) for plotting the fixed-point diagram."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    eps = np.abs(y - yhat)
    sig = scale_of(y, scale)
    taus = np.linspace(0.0, 1.0, n_grid)
    fs = np.array([np.mean(eps < (1.0 - t) * sig) for t in taus])
    return taus, fs


# --------------------------------------------------------------------------
# Comparison metrics
# --------------------------------------------------------------------------

def tolerance_accuracy(y, yhat, k: float, scale: str = "sd", sigma: float | None = None) -> float:
    """A(k) = fraction of |y - yhat| < k * sigma."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    sig = scale_of(y, scale) if sigma is None else float(sigma)
    return float(np.mean(np.abs(y - yhat) < k * sig))


def r2(y, yhat) -> float:
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot, SS_tot about the mean of y."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else -np.inf
    return 1.0 - ss_res / ss_tot


def willmott_d(y, yhat) -> float:
    """Willmott (1981) index of agreement d in [0, 1]."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ybar = y.mean()
    num = float(np.sum((y - yhat) ** 2))
    den = float(np.sum((np.abs(yhat - ybar) + np.abs(y - ybar)) ** 2))
    return 1.0 if den == 0.0 and num == 0.0 else 1.0 - num / den


def willmott_dr(y, yhat, c: float = 2.0) -> float:
    """Willmott, Robeson & Matsuura (2012) refined index of agreement d_r in [-1, 1]."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    a = float(np.sum(np.abs(yhat - y)))
    b = c * float(np.sum(np.abs(y - y.mean())))
    if b == 0.0:
        return 1.0 if a == 0.0 else -1.0
    return 1.0 - a / b if a <= b else b / a - 1.0


def theil_u1(y, yhat) -> float:
    """
    Theil's U1 inequality coefficient in [0, 1]; 0 = perfect agreement.
    U1 = sqrt(mean (y - yhat)^2) / ( sqrt(mean y^2) + sqrt(mean yhat^2) ).
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    num = np.sqrt(np.mean((y - yhat) ** 2))
    den = np.sqrt(np.mean(y ** 2)) + np.sqrt(np.mean(yhat ** 2))
    return 0.0 if den == 0.0 else float(num / den)


KS_DEFAULT = (0.05, 0.1, 0.2, 0.5)


def all_metrics(y, yhat, ks=KS_DEFAULT, scale: str = "sd", extended: bool = True) -> dict:
    """Dictionary with SCAS, R2, A(k) for each k and (optionally) d, d_r, U1."""
    out = {"SCAS": scas(y, yhat, scale=scale), "R2": r2(y, yhat)}
    for k in ks:
        out[f"A({k})"] = tolerance_accuracy(y, yhat, k, scale=scale)
    if extended:
        out["d"] = willmott_d(y, yhat)
        out["d_r"] = willmott_dr(y, yhat)
        out["U1"] = theil_u1(y, yhat)
    return out


# --------------------------------------------------------------------------
# Population (large-sample) SCAS and asymptotic variance
# --------------------------------------------------------------------------

def population_scas(G, sigma: float = 1.0) -> float:
    """
    Population SCAS: the unique root of  G((1 - tau) sigma) = tau  on [0, 1],
    where G is the CDF of the absolute residual |eps| (a callable).
    G is non-decreasing and tau -> G((1-tau)sigma) is non-increasing, so
    h(tau) = G((1-tau)sigma) - tau is strictly decreasing; h(0) >= 0, h(1) <= 0.
    """
    def h(t):
        return float(G((1.0 - t) * sigma)) - t
    if h(0.0) <= 0.0:
        return 0.0
    if h(1.0) >= 0.0:
        return 1.0
    return float(brentq(h, 0.0, 1.0, xtol=1e-12))


def abs_cdf_from_dist(dist) -> callable:
    """CDF of |X| for a scipy.stats frozen distribution symmetric about 0."""
    return lambda x: np.where(np.asarray(x) <= 0, 0.0, 2.0 * dist.cdf(np.asarray(x)) - 1.0)


def abs_pdf_from_dist(dist) -> callable:
    """Density of |X| for a scipy.stats frozen distribution symmetric about 0."""
    return lambda x: np.where(np.asarray(x) < 0, 0.0, 2.0 * dist.pdf(np.asarray(x)))


def population_scas_gaussian(s_over_sigma: float) -> float:
    """Population SCAS for Gaussian residuals with SD s, given the ratio s/sigma."""
    s = float(s_over_sigma)
    if s <= 0:
        return 1.0
    return population_scas(abs_cdf_from_dist(stats.norm(0, s)), sigma=1.0)


def population_scas_laplace(s_over_sigma: float) -> float:
    """Population SCAS for Laplace residuals with SD s (scale b = s/sqrt 2)."""
    s = float(s_over_sigma)
    if s <= 0:
        return 1.0
    return population_scas(abs_cdf_from_dist(stats.laplace(0, s / np.sqrt(2.0))), sigma=1.0)


def population_scas_t(s_over_sigma: float, df: float = 3.0) -> float:
    """Population SCAS for Student-t residuals (df>2) scaled to SD s."""
    s = float(s_over_sigma)
    if s <= 0:
        return 1.0
    scale = s / np.sqrt(df / (df - 2.0))
    return population_scas(abs_cdf_from_dist(stats.t(df, 0, scale)), sigma=1.0)


def asymptotic_variance(tau: float, p_at_threshold: float, sigma: float = 1.0,
                        kurtosis_y: float | None = None) -> float:
    """
    Asymptotic variance V such that sqrt(N) (tau_hat - tau) -> N(0, V).

    With sigma known:
        V = tau (1 - tau) / (1 + sigma * p)^2,
    where p is the density of |eps| at the threshold (1 - tau) sigma.

    If sigma is the sample SD of y (kurtosis_y = kurtosis of y, 3 for Gaussian)
    and the residuals are independent of y, an extra term is added:
        V = [ tau(1-tau) + (1-tau)^2 (sigma p)^2 (kappa - 1)/4 ] / (1 + sigma p)^2 .
    """
    sp = sigma * p_at_threshold
    num = tau * (1.0 - tau)
    if kurtosis_y is not None:
        num += (1.0 - tau) ** 2 * sp ** 2 * (kurtosis_y - 1.0) / 4.0
    return num / (1.0 + sp) ** 2


# --------------------------------------------------------------------------
# Plot style shared by all study scripts
# --------------------------------------------------------------------------

STYLE = {
    "SCAS":    dict(color="#d62728", marker="o", linestyle="-",  label="SCAS"),
    "R2":      dict(color="#000000", marker="x", linestyle="-",  label=r"$R^2$"),
    "A(0.05)": dict(color="#6a3d9a", marker="^", linestyle="--", label="A(0.05)"),
    "A(0.1)":  dict(color="#1f78b4", marker="v", linestyle="--", label="A(0.1)"),
    "A(0.2)":  dict(color="#008080", marker="D", linestyle="--", label="A(0.2)"),
    "A(0.5)":  dict(color="#4c9bdc", marker="P", linestyle="--", label="A(0.5)"),
    "d":       dict(color="#ff7f0e", marker="s", linestyle="-.", label=r"$d$"),
    "d_r":     dict(color="#8c564b", marker="*", linestyle="-.", label=r"$d_r$"),
    "U1":      dict(color="#7f7f7f", marker="h", linestyle=":",  label=r"$U_1$"),
    "1-U1":    dict(color="#7f7f7f", marker="h", linestyle=":",  label=r"$1-U_1$"),
    "SCAS_mad": dict(color="#e377c2", marker="o", linestyle="--", label="SCAS (MAD scale)"),
    "theory":  dict(color="#d62728", marker=None, linestyle=":",  label="SCAS (population)"),
}


def style(name: str) -> dict:
    return dict(STYLE.get(name, dict(color="gray", marker="o", linestyle="--", label=name)))
