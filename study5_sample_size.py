"""
Simulation study 5: sampling variability of SCAS across sample sizes.

Same regression process as study 4 (y = 1 + 2x + e, OLS fit on a training
sample, metrics on an independent test sample of the SAME size N), for
N in {20, 50, 100, 200, 500, 1000} and R = 2000 replications, in two settings:
  clean        : e ~ N(0, s^2) with population R^2 = 0.9;
  contaminated : Huber mixture with delta = 0.1 (as in study 4 B).

The Monte-Carlo SD of SCAS is compared with the asymptotic approximation
    SD(tau_hat) ~ sqrt(V / N),
    V = [tau(1-tau) + (1-tau)^2 (sigma p)^2 (kappa-1)/4] / (1 + sigma p)^2,
where p is the density of |eps| at the threshold (1-tau)sigma and kappa the
kurtosis of y (the second term accounts for sigma being the sample SD of y).

Outputs: Figure 7 of the paper (box plots of SCAS and R2 vs N; SD vs N on log-log with the
asymptotic line), Table 4 (mean, SD, 95% interval per N).
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import scas as S
from common import savefig, save_table, save_data, seeds
from study4_regression import draw, ols_fit, S_MAIN, VAR_SIGNAL, CONTAM_MULT, R2_MAIN

R = 2000
NS = (20, 50, 100, 200, 500, 1000)
SETTINGS = {"clean": ("A", R2_MAIN), "contaminated": ("B", 0.1)}


def one_setting(args):
    name, N, seed_list = args
    sc, lv = SETTINGS[name]
    rows = []
    for sd in seed_list:
        rng = np.random.default_rng(sd)
        xtr, ytr = draw(sc, lv, N, rng, train=True)
        xte, yte = draw(sc, lv, N, rng, train=False)
        b = ols_fit(xtr, ytr)
        yhat = b[0] + b[1] * xte
        rows.append({"setting": name, "N": N, "SCAS": S.scas(yte, yhat), "R2": S.r2(yte, yhat),
                     "A(0.2)": S.tolerance_accuracy(yte, yhat, 0.2)})
    return rows


def theory(name, N):
    """Population SCAS and asymptotic SD for the given setting and N."""
    sc, lv = SETTINGS[name]
    infl = np.sqrt(1 + 2.0 / N)
    if sc == "A":
        s_clean = S_MAIN
        s = s_clean * infl
        sigma = np.sqrt(VAR_SIGNAL + s_clean ** 2)
        D = stats.norm(0, s)
        G = S.abs_cdf_from_dist(D); p = S.abs_pdf_from_dist(D)
        # kurtosis of y = 2x + e: x uniform (kurt 1.8), e normal
        var_x, var_e = VAR_SIGNAL, s_clean ** 2
        m4 = 1.8 * var_x ** 2 + 6 * var_x * var_e + 3 * var_e ** 2
    else:
        d = lv
        s = S_MAIN * infl
        G1, G2 = stats.norm(0, s), stats.norm(0, CONTAM_MULT * s)
        G = lambda x: (1 - d) * (2 * G1.cdf(x) - 1) + d * (2 * G2.cdf(x) - 1) if x > 0 else 0.0
        p = lambda x: (1 - d) * 2 * G1.pdf(x) + d * 2 * G2.pdf(x)
        var_e = (1 - d) * S_MAIN ** 2 + d * (CONTAM_MULT * S_MAIN) ** 2
        m4_e = 3 * ((1 - d) * S_MAIN ** 4 + d * (CONTAM_MULT * S_MAIN) ** 4)
        var_x = VAR_SIGNAL
        sigma = np.sqrt(var_x + var_e)
        m4 = 1.8 * var_x ** 2 + 6 * var_x * var_e + m4_e
    kappa = m4 / (var_x + var_e) ** 2
    tau = S.population_scas(G, sigma=sigma)
    V = S.asymptotic_variance(tau, float(p((1 - tau) * sigma)), sigma, kurtosis_y=kappa)
    return tau, np.sqrt(V / N)


def main():
    print("running sample-size study ...")
    sd = seeds(5, 10_000_000)
    configs, i = [], 0
    for name in SETTINGS:
        for N in NS:
            configs.append((name, N, sd[i:i + R])); i += R
    rows = []
    with ProcessPoolExecutor() as ex:
        for r in ex.map(one_setting, configs):
            rows.extend(r)
    df = pd.DataFrame(rows)
    save_data(df, "study5_sample_size")

    # ---- Table 4
    rows = []
    for name in SETTINGS:
        for N in NS:
            sub = df[(df["setting"] == name) & (df["N"] == N)]
            tau, sd_th = theory(name, N)
            q = sub["SCAS"].quantile([.025, .975]).values
            qr = sub["R2"].quantile([.025, .975]).values
            rows.append({"setting": name, "$N$": N,
                         "SCAS mean": sub["SCAS"].mean(), "SCAS SD": sub["SCAS"].std(),
                         "SCAS 95\\% interval": f"[{q[0]:.3f}, {q[1]:.3f}]",
                         "population SCAS": tau, "asymptotic SD": sd_th,
                         "$R^2$ mean": sub["R2"].mean(), "$R^2$ SD": sub["R2"].std(),
                         "$R^2$ 95\\% interval": f"[{qr[0]:.3f}, {qr[1]:.3f}]"})
    tab = pd.DataFrame(rows)
    print(tab.to_string())
    save_table(tab, "table4_sample_size")

    # ---- Figure 7 of the paper
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, name in zip(axes[:2], SETTINGS):
        sub = df[df["setting"] == name]
        pos = np.arange(len(NS))
        bs = ax.boxplot([sub[sub["N"] == N]["SCAS"].values for N in NS], positions=pos - 0.18, widths=0.3,
                        patch_artist=True, showfliers=False)
        br = ax.boxplot([sub[sub["N"] == N]["R2"].values for N in NS], positions=pos + 0.18, widths=0.3,
                        patch_artist=True, showfliers=False)
        for b_, c in ((bs, S.style("SCAS")["color"]), (br, "#777777")):
            for patch in b_["boxes"]:
                patch.set_facecolor(c); patch.set_alpha(0.5)
        for N, x0 in zip(NS, pos):
            tau, _ = theory(name, N)
            ax.plot([x0 - 0.35, x0 - 0.01], [tau, tau], color="k", ls=":", lw=1.2)
        ax.set_xticks(pos); ax.set_xticklabels(NS)
        ax.set_xlabel("sample size $N$ (train = test)"); ax.set_ylabel("metric value (2000 runs)")
        ax.set_title(f"({'a' if name=='clean' else 'b'}) {name}: SCAS (red), $R^2$ (grey)")
        ax.grid(alpha=0.3)
    ax = axes[2]
    for name, mk in zip(SETTINGS, ("o", "s")):
        sub = df[df["setting"] == name]
        sd_mc = [sub[sub["N"] == N]["SCAS"].std() for N in NS]
        sd_th = [theory(name, N)[1] for N in NS]
        sd_r2 = [sub[sub["N"] == N]["R2"].std() for N in NS]
        ax.loglog(NS, sd_mc, marker=mk, color=S.style("SCAS")["color"], ls="-", label=f"SCAS, {name} (MC)")
        ax.loglog(NS, sd_th, marker=None, color=S.style("SCAS")["color"], ls=":", label=f"SCAS, {name} (asymptotic)")
        ax.loglog(NS, sd_r2, marker=mk, color="#777777", ls="-", label=f"$R^2$, {name} (MC)")
    ax.set_xlabel("sample size $N$"); ax.set_ylabel("standard deviation over runs")
    ax.set_title("(c) SD over runs vs $N$"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    savefig(fig, "fig9_sample_size")

    # ---- Check: is the small downward bias under contamination at small N due to the estimation step?
    #      Evaluate the same contaminated test samples with the TRUE line y = 1 + 2x.
    rng = np.random.default_rng(99)
    for N in (20, 50, 100):
        fit, true = [], []
        for _ in range(3000):
            xtr, ytr = draw("B", 0.1, N, rng, train=True)
            xte, yte = draw("B", 0.1, N, rng, train=False)
            b = ols_fit(xtr, ytr)
            fit.append(S.scas(yte, b[0] + b[1] * xte))
            true.append(S.scas(yte, 1 + 2 * xte))
        tau, _ = theory("contaminated", N)
        print(f"true-line check, contaminated, N={N}: mean SCAS fitted line = {np.mean(fit):.4f}, "
              f"true line = {np.mean(true):.4f}, population = {tau:.4f}")


if __name__ == "__main__":
    main()
