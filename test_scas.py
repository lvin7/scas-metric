"""
Sanity checks for scas.py.

1. The exact order-statistic algorithm agrees with Brent's method on g(tau)=f(tau)-tau
   (to within Brent's tolerance) on many random problems, including ties and
   degenerate cases.
2. The bracketing  eps_(J) <= (1-tau*) sigma <= eps_(J+1)  holds.
3. tau* = sup{tau : f(tau) >= tau} verified on a dense grid.
4. Population formulas: Monte-Carlo SCAS converges to population_scas.
"""
import numpy as np
from scipy import stats

import scas as S

rng = np.random.default_rng(0)

# ---- 1-3: exact vs Brent, bracketing, sup definition ---------------------
max_abs_diff = 0.0
n_exact_fixed = 0
n_cases = 0
for trial in range(3000):
    N = int(rng.integers(1, 60))
    kind = rng.integers(0, 4)
    y = rng.normal(size=N) * rng.uniform(0.5, 5)
    if kind == 0:
        yhat = y + rng.normal(size=N) * rng.uniform(0.01, 2)
    elif kind == 1:                      # heavy tails + exact zeros (ties at 0)
        yhat = y + rng.standard_t(2, size=N) * rng.uniform(0.01, 2)
        m = rng.random(N) < 0.3
        yhat[m] = y[m]
    elif kind == 2:                      # discretised residuals -> many ties
        yhat = y + np.round(rng.normal(size=N), 1)
    else:                                # perfect prediction
        yhat = y.copy()
    if N < 2:
        continue
    t_exact, det = S.scas_exact(y, yhat, return_details=True)
    t_brent = S.scas_brent(y, yhat, xtol=1e-12)
    d = abs(t_exact - t_brent)
    max_abs_diff = max(max_abs_diff, d)
    assert d < 1e-8, (trial, N, t_exact, t_brent)

    # bracketing
    thr = det["threshold"]
    assert det["eps_J"] <= thr + 1e-12 or det["J"] == 0, (trial, det)
    assert thr <= det["eps_J1"] + 1e-12, (trial, det)

    # sup definition on a dense grid
    taus, fs = S.scas_curve(y, yhat, n_grid=20001)
    ok = taus[fs >= taus]
    t_grid = ok.max() if ok.size else 0.0
    assert abs(t_grid - t_exact) < 1e-4, (trial, t_grid, t_exact)

    n_exact_fixed += det["exact_fixed_point"]
    n_cases += 1

print(f"exact vs Brent: {n_cases} cases, max |diff| = {max_abs_diff:.2e}")
print(f"fraction of cases where f(tau*) = tau* holds exactly: {n_exact_fixed / n_cases:.3f}")

# worked example from the docstring
y = np.array([0.0, 1.0]); yhat = np.array([0.0, 1.0 - 0.3 * np.std(y, ddof=1)])
t, det = S.scas_exact(y, yhat, return_details=True)
print("N=2 example (eps = 0, 0.3 sigma): tau* =", round(t, 4), "(expected 0.7), J =", det["J"])

# ---- 4: Monte-Carlo convergence to population value ---------------------
for name, dist, pop in [
    ("gaussian s=0.3", stats.norm(0, 0.3), S.population_scas_gaussian(0.3)),
    ("laplace  s=0.3", stats.laplace(0, 0.3 / np.sqrt(2)), S.population_scas_laplace(0.3)),
    ("t3       s=0.3", stats.t(3, 0, 0.3 / np.sqrt(3)), S.population_scas_t(0.3, 3)),
]:
    N = 200_000
    y = rng.normal(size=N)              # sigma_y ~ 1
    yhat = y - dist.rvs(size=N, random_state=rng)
    t = S.scas(y, yhat)
    print(f"{name}: MC SCAS (N=2e5) = {t:.4f}   population = {pop:.4f}")

# ---- asymptotic variance check (Gaussian residuals, sigma known vs estimated) --
s = 0.3
pop = S.population_scas_gaussian(s)
p = S.abs_pdf_from_dist(stats.norm(0, s))((1 - pop) * 1.0)
for N in (50, 200, 1000):
    vals_known, vals_est = [], []
    for _ in range(4000):
        y = rng.normal(size=N)
        yhat = y - rng.normal(scale=s, size=N)
        vals_known.append(S.scas(y, yhat, sigma=1.0))
        vals_est.append(S.scas(y, yhat))
    v_known = S.asymptotic_variance(pop, p, 1.0)
    v_est = S.asymptotic_variance(pop, p, 1.0, kurtosis_y=3.0)
    print(f"N={N:5d}: SD(tau_hat) sigma known  MC={np.std(vals_known):.4f} theory={np.sqrt(v_known/N):.4f} | "
          f"sigma estimated MC={np.std(vals_est):.4f} theory={np.sqrt(v_est/N):.4f}")
print("all checks passed")
