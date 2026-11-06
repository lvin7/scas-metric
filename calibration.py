"""
Calibration of SCAS: what does a given value mean?

For i.i.d. residuals eps with absolute-value CDF G, the population SCAS solves
    G((1 - tau) sigma) = tau.
With Gaussian residuals of SD s:   2 Phi((1-tau) sigma / s) - 1 = tau,
so tau depends only on the noise-to-signal ratio r = s / sigma, and for an
unbiased predictor with residuals uncorrelated with yhat, R^2 = 1 - r^2.

Outputs:
  Table 1 of the paper: SCAS -> r = s/sigma -> R^2 (Gaussian), plus Laplace and t(3) residuals.
  Figure 1 of the paper: population SCAS vs r for the three residual laws; SCAS vs R^2.
  Also: the implied tolerance k* = 1 - tau* and the fraction of Gaussian
  residuals falling outside the band although no outliers exist.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import scas as S
from common import savefig, save_table


def main():
    rs = np.concatenate([np.linspace(0.001, 0.1, 50), np.linspace(0.1, 2.0, 150)])
    laws = {
        "Gaussian": S.population_scas_gaussian,
        "Laplace": S.population_scas_laplace,
        "Student t(3)": lambda r: S.population_scas_t(r, 3),
    }
    curves = {k: np.array([f(r) for r in rs]) for k, f in laws.items()}

    # ---- Table 1 of the paper: target SCAS values -> noise ratio -> R2 (Gaussian), and other laws
    rows = []
    for target in (0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        row = {"SCAS": target, "$k^*=1-\\tau^*$": 1 - target}
        for name, f in laws.items():
            r = brentq(lambda r: f(r) - target, 1e-6, 50.0)
            row[f"$s/\\sigma$ ({name})"] = r
            row[f"$R^2$ ({name})"] = f"{1 - r ** 2:.3f}" if r < 1 else "$<0$"
        rows.append(row)
    tab = pd.DataFrame(rows)
    print(tab.to_string())
    save_table(tab, "table2_calibration")

    # ---- Figure 1 of the paper
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, t in curves.items():
        axes[0].plot(rs, t, label=name)
        axes[1].plot(1 - rs ** 2, t, label=name)
    axes[0].set_xlabel(r"noise-to-signal ratio $\sigma_e/\sigma_y$"); axes[0].set_ylabel("population SCAS")
    axes[0].set_xlim(0, 2); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel(r"$R^2 = 1-(\sigma_e/\sigma_y)^2$ (unbiased predictor)"); axes[1].set_ylabel("population SCAS")
    axes[1].set_xlim(0, 1); axes[1].grid(alpha=0.3); axes[1].legend()
    axes[0].set_title("(a) SCAS as a function of residual scale"); axes[1].set_title(r"(b) SCAS versus $R^2$")
    savefig(fig, "fig7_calibration")


if __name__ == "__main__":
    main()
