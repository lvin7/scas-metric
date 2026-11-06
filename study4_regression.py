"""
Simulation study 4: regression with parameter estimation and out-of-sample evaluation.

Data-generating process:  x ~ U(0, 10),  y = 1 + 2 x + e.
A training sample of size N_TRAIN is drawn, an ordinary least-squares line is
fitted, and all metrics are evaluated on an independent test sample of size
N_TEST from the same process (the scale sigma is the sample SD of the observed
test values).  R = 1000 replications per setting.

Scenarios (one sweep parameter each):
  A  clean      : e ~ N(0, s^2); sweep the population R^2 = var(2x)/(var(2x)+s^2)
                  over {0.5, 0.7, 0.9, 0.95, 0.99}.
  B  Huber      : e ~ (1-delta) N(0, s^2) + delta N(0, (5 s)^2), s set so that the
                  clean population R^2 = 0.9; sweep delta in {0, .05, .1, .2, .3}
                  (contamination present in both training and test data).
  C  misspec.   : y = 1 + 2x + gamma (x-5)^2 + e (linear model fitted);
                  sweep gamma in {0, 0.05, 0.1, 0.2, 0.4}.
  D  leverage   : clean data plus ONE bad leverage point in the training sample at
                  x_L in {10, 15, 20, 30, 50} with y_L = 1 + 0 (the line value at x=0);
                  metrics reported both in-sample (training) and out-of-sample (test).

Population SCAS references are drawn for A and B (Gaussian / mixture residuals
with the OLS prediction-error inflation s_eff^2 = s^2 (1 + 2/N_TRAIN)).

Outputs: Figure 6 of the paper (four panels), Table 3 (median and IQR at the main setting).
"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import scas as S
from common import savefig, save_table, save_data, seeds

R = 1000
N_TRAIN = 100
N_TEST = 100
KS = S.KS_DEFAULT
VAR_SIGNAL = 4.0 * 100.0 / 12.0          # var(2x), x ~ U(0,10)
R2_MAIN = 0.9
S_MAIN = np.sqrt(VAR_SIGNAL * (1 - R2_MAIN) / R2_MAIN)
CONTAM_MULT = 5.0

SWEEPS = {
    "A": ("population $R^2$ (clean)", [0.5, 0.7, 0.9, 0.95, 0.99]),
    "B": (r"contamination fraction $\delta$ (Huber mixture)", [0.0, 0.05, 0.1, 0.2, 0.3]),
    "C": (r"curvature $\gamma$ (misspecification)", [0.0, 0.05, 0.1, 0.2, 0.4]),
    "D": (r"leverage point position $x_L$", [10, 15, 20, 30, 50]),
}
MAIN = {"A": 0.9, "B": 0.1, "C": 0.2, "D": 30}
METRICS = ["R2", "SCAS", "A(0.05)", "A(0.1)", "A(0.2)", "A(0.5)", "d", "d_r", "1-U1"]


def noise_sd(r2pop):
    return np.sqrt(VAR_SIGNAL * (1 - r2pop) / r2pop)


def draw(scenario, level, n, rng, train=False):
    x = rng.uniform(0, 10, size=n)
    if scenario == "A":
        e = rng.normal(0, noise_sd(level), size=n)
        y = 1 + 2 * x + e
    elif scenario == "B":
        s = S_MAIN
        bad = rng.random(n) < level
        e = rng.normal(0, s, size=n)
        e[bad] = rng.normal(0, CONTAM_MULT * s, size=bad.sum())
        y = 1 + 2 * x + e
    elif scenario == "C":
        e = rng.normal(0, S_MAIN, size=n)
        y = 1 + 2 * x + level * (x - 5) ** 2 + e
    elif scenario == "D":
        e = rng.normal(0, S_MAIN, size=n)
        y = 1 + 2 * x + e
        if train:
            x = np.append(x, level)
            y = np.append(y, 1.0 + rng.normal(0, S_MAIN))      # bad leverage point
    else:
        raise ValueError(scenario)
    return x, y


def ols_fit(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def one_setting(args):
    scenario, level, seed_list = args
    rows = []
    for sd in seed_list:
        rng = np.random.default_rng(sd)
        xtr, ytr = draw(scenario, level, N_TRAIN, rng, train=True)
        xte, yte = draw(scenario, level, N_TEST, rng, train=False)
        b = ols_fit(xtr, ytr)
        yhat_te = b[0] + b[1] * xte
        m = S.all_metrics(yte, yhat_te, ks=KS)
        row = {"scenario": scenario, "level": level, "sample": "test", **m,
               "beta0": b[0], "beta1": b[1]}
        rows.append(row)
        if scenario == "D":
            yhat_tr = b[0] + b[1] * xtr
            m = S.all_metrics(ytr, yhat_tr, ks=KS)
            rows.append({"scenario": scenario, "level": level, "sample": "train", **m,
                         "beta0": b[0], "beta1": b[1]})
    return rows


def population_reference(scenario, level):
    """Population SCAS for scenarios A and B (None otherwise)."""
    infl = np.sqrt(1 + 2.0 / N_TRAIN)
    if scenario == "A":
        s = noise_sd(level) * infl
        sigma = np.sqrt(VAR_SIGNAL + noise_sd(level) ** 2)
        return S.population_scas(S.abs_cdf_from_dist(stats.norm(0, s)), sigma=sigma)
    if scenario == "B":
        s = S_MAIN * infl
        d = level
        sigma = np.sqrt(VAR_SIGNAL + (1 - d) * S_MAIN ** 2 + d * (CONTAM_MULT * S_MAIN) ** 2)
        G1, G2 = stats.norm(0, s), stats.norm(0, CONTAM_MULT * s)
        G = lambda x: (1 - d) * (2 * G1.cdf(x) - 1) + d * (2 * G2.cdf(x) - 1) if x > 0 else 0.0
        return S.population_scas(G, sigma=sigma)
    return None


def main():
    print("running regression study ...")
    sd = seeds(4, 10_000_000)
    configs, i = [], 0
    for sc, (_, levels) in SWEEPS.items():
        for lv in levels:
            configs.append((sc, lv, sd[i:i + R])); i += R
    rows = []
    with ProcessPoolExecutor() as ex:
        for r in ex.map(one_setting, configs):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df["1-U1"] = 1 - df["U1"]
    save_data(df, "study4_regression")

    # ---- Figure 6 of the paper: four panels, medians with IQR band for SCAS and R2
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (sc, (xlabel, levels)) in zip(axes.ravel(), SWEEPS.items()):
        sub = df[(df["scenario"] == sc) & (df["sample"] == "test")]
        for m in METRICS:
            g = sub.groupby("level")[m]
            med = g.median()
            ax.plot(med.index, med.values, **S.style(m), ms=4)
            if m in ("SCAS", "R2"):
                ax.fill_between(med.index, g.quantile(.25).values, g.quantile(.75).values,
                                color=S.style(m)["color"], alpha=0.15, lw=0)
        if sc == "D":
            g = df[(df["scenario"] == sc) & (df["sample"] == "train")].groupby("level")
            for m, mk in (("SCAS", "o"), ("R2", "x")):
                ax.plot(g[m].median().index, g[m].median().values, color=S.style(m)["color"],
                        marker=mk, ls="--", ms=4, alpha=0.6, label=S.style(m)["label"] + " (in-sample)")
        ref = [population_reference(sc, lv) for lv in levels]
        if ref[0] is not None:
            ax.plot(levels, ref, **S.style("theory"))
        ax.set_xlabel(xlabel); ax.set_ylabel("metric value (median, IQR band)")
        ax.set_title({"A": "(A) clean Gaussian errors", "B": "(B) Huber-contaminated errors",
                      "C": "(C) misspecified mean (quadratic truth)", "D": "(D) one bad leverage point in training"}[sc])
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
        if sc == "A":
            ax.set_xlim(0.45, 1.0)
    h, l = axes[0, 1].get_legend_handles_labels()
    h2, l2 = axes[1, 1].get_legend_handles_labels()
    extra = [(hh, ll) for hh, ll in zip(h2, l2) if "in-sample" in ll]
    fig.legend(h + [e[0] for e in extra], l + [e[1] for e in extra], ncol=6, fontsize=9,
               loc="outside lower center")
    savefig(fig, "fig8_regression_scenarios")

    # ---- Table 3: median (IQR) of every metric at the main setting of each scenario
    rows = []
    for sc, lv in MAIN.items():
        for sample in (("test",) if sc != "D" else ("train", "test")):
            sub = df[(df["scenario"] == sc) & (df["level"] == lv) & (df["sample"] == sample)]
            row = {"Scenario": f"{sc} ({sample})", "level": lv}
            for m in ["SCAS", "R2"] + [f"A({k})" for k in KS] + ["d", "d_r", "U1"]:
                q = sub[m].quantile([.25, .5, .75]).values
                row[m] = f"{q[1]:.3f} ({q[0]:.3f}, {q[2]:.3f})"
            ref = population_reference(sc, lv)
            row["SCAS (population)"] = f"{ref:.3f}" if ref is not None else "--"
            rows.append(row)
    tab = pd.DataFrame(rows)
    print(tab.to_string())
    save_table(tab, "table3_regression_main")


if __name__ == "__main__":
    main()
