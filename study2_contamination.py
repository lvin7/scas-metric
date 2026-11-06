"""
Simulation study 2: contamination of the predictions.

Observed signal y (N = 100) ~ N(0,1), t(3) or Laplace(0,1); R = 1000 replications.
Prediction yhat = y, then a random fraction c in {10%, 50%, 100%} of the
predictions is perturbed by additive noise  b * s * D,  with s = SD(y),
b in {0.5, 1, 1.5, 2} and D unit-SD Gaussian or Laplace.
Real-world counterpart: the four Stackloss columns (N = 21), same mechanism.

Expected (population) SCAS: the residual |eps| is 0 with probability 1-c and
|b s D| with probability c, so  G(x) = (1-c) + c * P(|b s D| < x)  and the
population SCAS solves G((1-tau) s) = tau.  (For N(0,1) signals s -> 1.)

Outputs: curve plots (median metric vs b, with the population SCAS dashed),
box plots at b = 2, heat maps of median SCAS and R2, and the t(3)/Laplace table and
curves. Only the curve figure (Fig. 4 of the paper) is shown in the manuscript; the
box plots, heat maps, full table and t(3)/Laplace curves are supplementary outputs
(their findings are summarized in the text of Section 4.3).
"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import scas as S
from common import savefig, save_table, save_data, load_stackloss, seeds

R = 1000
N = 100
DISTS = ("normal", "t3", "laplace")
FRACS = (0.10, 0.50, 1.00)
NOISES = ("gauss", "laplace")
SCALES = (0.5, 1.0, 1.5, 2.0)
KS = S.KS_DEFAULT
METRICS = ["R2", "SCAS", "A(0.05)", "A(0.1)", "A(0.2)", "A(0.5)"]


def sample_signal(dist, n, rng):
    if dist == "normal":
        return rng.normal(size=n)
    if dist == "t3":
        return rng.standard_t(3, size=n)
    if dist == "laplace":
        return rng.laplace(scale=1.0, size=n)
    raise ValueError(dist)


def draw_noise(kind, n, rng):
    if kind == "gauss":
        return rng.normal(size=n)
    if kind == "laplace":
        return rng.laplace(scale=1 / np.sqrt(2), size=n)   # unit SD
    raise ValueError(kind)


def contaminate(y, frac, kind, b, rng):
    n = y.size
    m = max(1, int(np.floor(frac * n))) if frac > 0 else 0
    idx = np.arange(n) if frac >= 1.0 else rng.choice(n, size=m, replace=False)
    s = S.scale_of(y, "sd")
    yhat = y.copy()
    yhat[idx] += b * s * draw_noise(kind, idx.size, rng)
    return yhat


def population_scas(frac, kind, b, s=1.0):
    """
    Population SCAS for the mixture residual distribution (s = SD of y).
    Because the noise is scaled by the SD of y, the result expressed in units
    of sigma does not depend on s nor on the signal distribution.
    """
    if kind == "gauss":
        D = stats.norm(0, b * s)
    else:
        D = stats.laplace(0, b * s / np.sqrt(2))
    G = lambda x: (1 - frac) + frac * (2 * D.cdf(x) - 1) if x > 0 else 0.0
    return S.population_scas(G, sigma=s)


def one_config(args):
    dataset, dist_or_col, frac, kind, b, seed_list = args
    rows = []
    y_fixed = None
    if dataset == "stackloss":
        y_fixed = load_stackloss()[dist_or_col].to_numpy(dtype=float)
    for sd in seed_list:
        rng = np.random.default_rng(sd)
        y = sample_signal(dist_or_col, N, rng) if y_fixed is None else y_fixed
        yhat = contaminate(y, frac, kind, b, rng)
        m = S.all_metrics(y, yhat, ks=KS, extended=False)
        rows.append({"dataset": dataset, "signal": dist_or_col, "N": y.size, "contam_frac": frac,
                     "contam_noise": kind, "scale_mult": b, **m})
    return rows


def run_all():
    configs = []
    sd = seeds(42, 10_000_000)
    i = 0
    for dist in DISTS:
        for frac, kind, b in itertools.product(FRACS, NOISES, SCALES):
            configs.append(("synthetic", dist, frac, kind, b, sd[i:i + R])); i += R
    for col in ("STACKLOSS", "AIRFLOW", "WATERTEMP", "ACIDCONC"):
        for frac, kind, b in itertools.product(FRACS, NOISES, SCALES):
            configs.append(("stackloss", col, frac, kind, b, sd[i:i + R])); i += R
    rows = []
    with ProcessPoolExecutor() as ex:
        for r in ex.map(one_config, configs):
            rows.extend(r)
    return pd.DataFrame(rows)


def curve_plot(ax, sub, title, s_for_theory):
    for m in METRICS:
        med = sub.groupby("scale_mult")[m].median()
        ax.plot(med.index, med.values, **S.style(m))
    frac = sub["contam_frac"].iloc[0]; kind = sub["contam_noise"].iloc[0]
    th = [population_scas(frac, kind, b, s=s_for_theory) for b in SCALES]
    ax.plot(SCALES, th, **S.style("theory"))
    ax.set_xlabel(r"noise intensity $b$ ($\times s_y$)")
    ax.set_ylabel("metric value (median of 1000 runs)")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main():
    print("running contamination study ...")
    df = run_all()
    save_data(df, "study2_contamination")

    # ---- Figure 4 of the paper: curves (normal N=100, 50 % Gaussian) and (Stackloss AIRFLOW, 50 % Gaussian)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    sub = df.query("dataset=='synthetic' & signal=='normal' & contam_frac==0.5 & contam_noise=='gauss'")
    curve_plot(axes[0], sub, "(a) synthetic N(0,1), $N=100$, 50% contaminated", 1.0)
    y_air = load_stackloss()["AIRFLOW"].to_numpy(dtype=float)
    sub = df.query("dataset=='stackloss' & signal=='AIRFLOW' & contam_frac==0.5 & contam_noise=='gauss'")
    curve_plot(axes[1], sub, "(b) Stackloss air flow, $N=21$, 50% contaminated", S.scale_of(y_air))
    axes[1].legend(ncol=2, fontsize=8)
    savefig(fig, "fig3_contamination_curves")

    # ---- Supplementary (not in the paper): box plots, Stackloss AIRFLOW, 10 %, b = 2, Gaussian vs Laplace
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, kind in zip(axes, NOISES):
        sub = df.query(f"dataset=='stackloss' & signal=='AIRFLOW' & contam_frac==0.1 & contam_noise=='{kind}' & scale_mult==2.0")
        ax.boxplot([sub[m].values for m in METRICS], tick_labels=[S.style(m)["label"] for m in METRICS])
        ax.set_title(f"({'a' if kind=='gauss' else 'b'}) 10% contaminated, $b=2$, {'Gaussian' if kind=='gauss' else 'Laplace'} noise")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("metric value (1000 runs)")
    savefig(fig, "fig4_contamination_boxplots")

    # ---- Supplementary (not in the paper): heat maps of median SCAS and R2 (Stackloss AIRFLOW, Gaussian noise)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    sub = df.query("dataset=='stackloss' & signal=='AIRFLOW' & contam_noise=='gauss'")
    for ax, m in zip(axes, ("SCAS", "R2")):
        piv = sub.groupby(["contam_frac", "scale_mult"])[m].median().unstack("scale_mult")
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis",
                       vmin=min(0, piv.values.min()), vmax=1)
        ax.set_xticks(range(len(SCALES))); ax.set_xticklabels([f"{b}" for b in SCALES])
        ax.set_yticks(range(len(FRACS))); ax.set_yticklabels([f"{int(100*f)}%" for f in FRACS])
        ax.set_xlabel(r"noise intensity $b$ ($\times s_y$)"); ax.set_ylabel("contaminated fraction")
        ax.set_title(f"({'a' if m=='SCAS' else 'b'}) median {S.style(m)['label']} (1000 runs)")
        for (i, j), v in np.ndenumerate(piv.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 else "k", fontsize=9)
        fig.colorbar(im, ax=ax)
    savefig(fig, "fig5_contamination_heatmaps")

    # ---- Supplementary (not in the paper): median (IQR) SCAS and R2 for all three signal distributions
    rows = []
    for dist in DISTS:
        for frac, kind, b in itertools.product(FRACS, NOISES, SCALES):
            sub = df.query(f"dataset=='synthetic' & signal=='{dist}' & contam_frac=={frac} & contam_noise=='{kind}' & scale_mult=={b}")
            rows.append({"signal": dist, "fraction": f"{int(100*frac)}\\%", "noise": kind, "$b$": b,
                         "SCAS median": sub["SCAS"].median(), "SCAS IQR": sub["SCAS"].quantile(.75) - sub["SCAS"].quantile(.25),
                         "SCAS population": population_scas(frac, kind, b, 1.0),
                         "$R^2$ median": sub["R2"].median(), "$R^2$ IQR": sub["R2"].quantile(.75) - sub["R2"].quantile(.25)})
    save_table(pd.DataFrame(rows), "tableA1_contamination_all_distributions")

    # ---- Supplementary (not in the paper): curves for t(3) and Laplace signals
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, dist in zip(axes, ("t3", "laplace")):
        sub = df.query(f"dataset=='synthetic' & signal=='{dist}' & contam_frac==0.5 & contam_noise=='gauss'")
        curve_plot(ax, sub, f"({'a' if dist=='t3' else 'b'}) synthetic {'$t(3)$' if dist=='t3' else 'Laplace(0,1)'}, $N=100$, 50% contaminated", 1.0)
    axes[1].legend(ncol=2, fontsize=8)
    savefig(fig, "figA1_contamination_curves_t3_laplace")


if __name__ == "__main__":
    main()
