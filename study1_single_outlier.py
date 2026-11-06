"""
Simulation study 1: a single outlier.

Synthetic signal : y = sin(x), x = 20 equispaced points in [0, 2 pi].
Prediction       : yhat = sin(x + u), u ~ U(-0.1 s, 0.1 s), s = SD(y),
                   plus ONE outlier: yhat[10] += a * s, a in {3, 6, 10}.
Real-world signal: Stackloss (Brownlee 1965) STACKLOSS column (N = 21),
                   yhat = y + u, u ~ U(-0.1 s, 0.1 s), outlier yhat[7] += a * s.

(The first submission described the noise as U(-a s, a s); that was a typo in
the text. The code always used +-0.1 s phase/additive noise; see the response.)

Outputs: Table 2 of the paper (metrics vs outlier severity), Figure 2 (signals),
         Figure 3 (fixed-point diagram with the exact algorithm's solution).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scas as S
from common import savefig, save_table, load_stackloss

SEED = 2026
SEVERITIES = (3, 6, 10)
KS = S.KS_DEFAULT


def synthetic(rng):
    x = np.linspace(0, 2 * np.pi, 20)
    y = np.sin(x)
    s = np.std(y, ddof=1)
    u = rng.uniform(-0.1 * s, 0.1 * s, size=x.size)
    base = np.sin(x + u)
    return x, y, base, s


def real(rng):
    y = load_stackloss()["STACKLOSS"].to_numpy(dtype=float)
    x = np.arange(y.size)
    s = np.std(y, ddof=1)
    base = y + rng.uniform(-0.1 * s, 0.1 * s, size=y.size)
    return x, y, base, s


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    figs = {}
    for name, gen, idx in (("Synthetic", synthetic, 10), ("Real-world", real, 7)):
        x, y, base, s = gen(rng)
        for a in SEVERITIES:
            yhat = base.copy()
            yhat[idx] += a * s
            m = S.all_metrics(y, yhat, ks=KS, extended=True)
            tau, det = S.scas_exact(y, yhat, return_details=True)
            rows.append({"Dataset": name, "Outlier severity": f"${a}s_y$", **m,
                         "J": det["J"], "exact_fixed_point": det["exact_fixed_point"]})
            if a == 3:
                figs[name] = (x, y, yhat)

    df = pd.DataFrame(rows)
    print(df.to_string())
    save_table(df[["Dataset", "Outlier severity", "SCAS", "R2"] + [f"A({k})" for k in KS] + ["d", "d_r", "U1"]],
               "table1_single_outlier")
    df.to_csv("results/tables/table1_single_outlier_full.csv", index=False)

    # Figure 2 of the paper: signals with a 3 s_y outlier
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (name, (x, y, yhat)) in zip(axes, figs.items()):
        ax.plot(x, yhat, label="predicted $\\hat{y}$")
        ax.plot(x, y, "--", label="observed $y$")
        ax.set_xlabel("x" if name == "Synthetic" else "observation index")
        ax.set_ylabel("y" if name == "Synthetic" else "stack loss")
        ax.set_title(f"({'a' if name=='Synthetic' else 'b'}) {name} data, outlier of $3s_y$")
        ax.legend()
    savefig(fig, "fig1_single_outlier_signals")

    # Figure 3 of the paper: fixed-point diagram with the exact solution marked
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (name, (x, y, yhat)) in zip(axes, figs.items()):
        taus, fs = S.scas_curve(y, yhat, n_grid=2001)
        tau, det = S.scas_exact(y, yhat, return_details=True)
        ax.step(taus, fs, where="post", label=r"$f(\tau)$")
        ax.plot([0, 1], [0, 1], "r--", label=r"identity $\tau$")
        ax.axvline(tau, color="k", ls=":", lw=1)
        ax.plot(tau, tau, "ko", ms=5, label=rf"$\tau^*={tau:.3f}$ (J={det['J']})")
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$f(\tau)$")
        ax.set_title(f"({'a' if name=='Synthetic' else 'b'}) {name} data: crossing point")
        ax.legend(loc="lower left")
    savefig(fig, "fig2_fixed_point_diagram")


if __name__ == "__main__":
    main()
