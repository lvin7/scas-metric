"""
Simulation study 3: systematic bias.

Observed y ~ N(100, 10^2), N = 100, R = 1000 replications.
Additive bias       : yhat = y + beta * s,  beta in [-1, 1] (19 levels), s = SD(y).
Multiplicative bias : yhat = m * y,         m in [0.85, 1.15] (19 levels).

Expected SCAS under pure additive bias: all residuals equal |beta| s, so
f(tau) = 1 if |beta| < 1 - tau, else 0; hence tau* = 1 - |beta|  (exactly).
Under multiplicative bias eps_i = |m-1| |y_i|; with y ~ N(mu, s^2) the
population SCAS solves P(|m-1||Y| < (1-tau) s) = tau.

Outputs: Figure 5 of the paper (median metric vs bias level, additive and multiplicative),
now including Willmott d, d_r and Theil U1 (plotted as 1 - U1).
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import scas as S
from common import savefig, save_data, seeds

R = 1000
N = 100
MU, SD = 100.0, 10.0
ADD = np.linspace(-1, 1, 19)
MUL = np.linspace(0.85, 1.15, 19)
KS = S.KS_DEFAULT
METRICS = ["R2", "SCAS", "A(0.05)", "A(0.1)", "A(0.2)", "A(0.5)", "d", "d_r", "1-U1"]


def one_rep(seed):
    rng = np.random.default_rng(seed)
    y = rng.normal(MU, SD, size=N)
    s = S.scale_of(y)
    rows = []
    for beta in ADD:
        m = S.all_metrics(y, y + beta * s, ks=KS)
        rows.append({"bias_type": "add", "bias_level": beta, **m})
    for mult in MUL:
        m = S.all_metrics(y, mult * y, ks=KS)
        rows.append({"bias_type": "mul", "bias_level": mult, **m})
    return rows


def population_mult(mult):
    """Population SCAS for multiplicative bias m with Y ~ N(MU, SD^2)."""
    a = abs(mult - 1.0)
    if a == 0:
        return 1.0
    Y = stats.norm(MU, SD)
    G = lambda x: Y.cdf(x / a) - Y.cdf(-x / a) if x > 0 else 0.0
    return S.population_scas(G, sigma=SD)


def main():
    print("running bias study ...")
    rows = []
    with ProcessPoolExecutor() as ex:
        for r in ex.map(one_rep, seeds(7, R), chunksize=20):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df["1-U1"] = 1 - df["U1"]
    save_data(df, "study3_bias")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for ax, bt in zip(axes, ("add", "mul")):
        sub = df[df["bias_type"] == bt]
        for m in METRICS:
            med = sub.groupby("bias_level")[m].median()
            ax.plot(med.index, med.values, **S.style(m), ms=4)
        levels = ADD if bt == "add" else MUL
        th = (1 - np.abs(levels)) if bt == "add" else [population_mult(m) for m in levels]
        ax.plot(levels, th, **S.style("theory"))
        ax.axvline(0 if bt == "add" else 1, color="k", ls=":", lw=1)
        ax.set_xlabel(r"additive bias $\beta$ ($\times s_y$)" if bt == "add" else "multiplicative factor $m$")
        ax.set_ylabel("metric value (median of 1000 runs)")
        ax.set_title("(a) Additive bias" if bt == "add" else "(b) Multiplicative bias")
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=9, loc="outside lower center")
    savefig(fig, "fig6_bias_sensitivity")


if __name__ == "__main__":
    main()
