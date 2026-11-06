"""
Simulation study 6: contamination of the OBSERVED TARGET and the choice of scale.

The sample standard deviation s_y of the observations is not a robust scale estimator, so
gross errors in the observed target inflate the tolerance band.  Here the
predictions are realistic (yhat = y_clean + N(0, s^2), s = 0.3) while a fraction
c of the OBSERVED values is corrupted:  y = y_clean + b * D  on the corrupted
points, D ~ N(0,1), b in {0, 1, 2, 5, 10, 20}.  y_clean ~ N(0,1), N = 100, R = 1000.

Compared: SCAS with sigma = sample SD (default), SCAS with sigma = 1.4826 MAD,
R2, A(0.2) with both scales.  Clean reference: population SCAS for s = 0.3.

Expected: the SD-scaled SCAS *increases* with b because the band widens with
the inflated SD, whereas the MAD-scaled version stays at approximately
(1 - c) x clean value.

Output: Figure 8 of the paper, Table 5.
"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scas as S
from common import savefig, save_table, save_data, seeds

R = 1000
N = 100
S_PRED = 0.3
FRACS = (0.05, 0.10, 0.20)
BS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)


def one_setting(args):
    c, b, seed_list = args
    rows = []
    for sd in seed_list:
        rng = np.random.default_rng(sd)
        y_clean = rng.normal(size=N)
        yhat = y_clean + rng.normal(0, S_PRED, size=N)
        y = y_clean.copy()
        m = max(1, int(round(c * N)))
        idx = rng.choice(N, size=m, replace=False)
        y[idx] += b * rng.normal(size=m)
        rows.append({"frac": c, "b": b,
                     "SCAS": S.scas(y, yhat, scale="sd"), "SCAS_mad": S.scas(y, yhat, scale="mad"),
                     "R2": S.r2(y, yhat),
                     "A(0.2)": S.tolerance_accuracy(y, yhat, 0.2, scale="sd"),
                     "A(0.2)_mad": S.tolerance_accuracy(y, yhat, 0.2, scale="mad"),
                     "sd_y": S.scale_of(y, "sd"), "mad_y": S.scale_of(y, "mad")})
    return rows


def main():
    print("running scale-variant study ...")
    sd = seeds(6, 10_000_000)
    configs, i = [], 0
    for c, b in itertools.product(FRACS, BS):
        configs.append((c, b, sd[i:i + R])); i += R
    rows = []
    with ProcessPoolExecutor() as ex:
        for r in ex.map(one_setting, configs):
            rows.extend(r)
    df = pd.DataFrame(rows)
    save_data(df, "study6_scale_variant")
    clean_ref = S.population_scas_gaussian(S_PRED)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, constrained_layout=True)
    for ax, c in zip(axes, FRACS):
        sub = df[df["frac"] == c]
        for m in ("SCAS", "SCAS_mad", "R2", "A(0.2)"):
            g = sub.groupby("b")[m]
            ax.plot(g.median().index, g.median().values, **S.style(m), ms=4)
            ax.fill_between(g.median().index, g.quantile(.25).values, g.quantile(.75).values,
                            color=S.style(m)["color"], alpha=0.12, lw=0)
        ax.axhline(clean_ref, color="k", ls=":", lw=1, label="SCAS, no target contamination")
        ax.set_xlabel(r"magnitude $b$ of target contamination ($\times\sigma_c$)")
        ax.set_title(f"({chr(97+list(FRACS).index(c))}) {int(100*c)}% of observed values corrupted")
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    axes[0].set_ylabel("metric value (median, IQR band)")
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=9, loc="outside lower center")
    savefig(fig, "fig10_target_contamination_scale")

    rows = []
    for c, b in itertools.product(FRACS, BS):
        sub = df[(df["frac"] == c) & (df["b"] == b)]
        rows.append({"fraction": f"{int(100*c)}\\%", "$b$": b,
                     "SD inflation": sub["sd_y"].median(), "MAD inflation": sub["mad_y"].median(),
                     "SCAS (SD)": sub["SCAS"].median(), "SCAS (MAD)": sub["SCAS_mad"].median(),
                     "$R^2$": sub["R2"].median(), "A(0.2) (SD)": sub["A(0.2)"].median(),
                     "A(0.2) (MAD)": sub["A(0.2)_mad"].median()})
    tab = pd.DataFrame(rows)
    print(tab.to_string())
    save_table(tab, "table5_target_contamination_scale")


if __name__ == "__main__":
    main()
