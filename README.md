# SCAS: Self-Consistent Agreement Score

Reference implementation and simulation studies for the manuscript
*A Self-Consistent Agreement Score (SCAS) for Regression Assessment*
(Statistics and Computing, revised version).

## Definition

For observed values `y`, predictions `yhat`, absolute residuals `eps_i = |y_i - yhat_i|`
and scale `sigma` (the sample standard deviation `s_y` of `y` by default; written `s_y` in the paper), SCAS is the crossing point

    tau* = sup{ tau in [0,1] : f(tau) >= tau },   f(tau) = (1/N) #{ i : eps_i < (1 - tau) sigma }.

`scas.py` computes it exactly from the sorted residuals (no root finder):

    c_j = 1 - eps_(j) / sigma,   J = max{ j : eps_(j) < (1 - (j-1)/N) sigma },
    tau* = min( J/N , c_J )      (tau* = 0 if J = 0).

The implied tolerance is `k* = 1 - tau*`, i.e. SCAS = A(k*) where `A(k)` is the
tolerance accuracy at tolerance `k sigma`.

```python
import scas
tau = scas.scas(y, yhat)                       # default: sigma = sample SD of y
tau_mad = scas.scas(y, yhat, scale="mad")      # robust scale 1.4826 * MAD
tau, info = scas.scas_exact(y, yhat, return_details=True)   # J, threshold, k*, ...
```

`scas.py` also provides `r2`, `tolerance_accuracy`, `willmott_d`, `willmott_dr`,
`theil_u1`, the population SCAS for a given residual distribution
(`population_scas*`) and the asymptotic variance (`asymptotic_variance`).

## Installation

Python 3.11 or newer (developed with 3.14):

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the paper; the code only needs
numpy, scipy, pandas, matplotlib, statsmodels and Jinja2 (for the LaTeX tables).

## Reproducing the paper

Each simulation study is a separate script; all outputs go to `results/`
(`figs/`, `tables/`, `data/`). Seeds are fixed.

| script | paper | output files (`results/`) | content |
|---|---|---|---|
| `test_scas.py` | Sec. 3, App. A | -- | checks: exact algorithm vs Brent, bracketing, population limits, asymptotic SD |
| `calibration.py` | Table 1, Fig. 1 | `table2_calibration`, `fig7_calibration` | population SCAS vs noise-to-signal ratio and vs R^2 (Gaussian, Laplace, t(3)) |
| `study1_single_outlier.py` | Table 2, Figs. 2-3 | `table1_single_outlier`, `fig1_single_outlier_signals`, `fig2_fixed_point_diagram` | single outlier, sine and Stackloss, fixed-point diagram |
| `study2_contamination.py` | Fig. 4 | `fig3_contamination_curves` (the script also writes box plots, heat maps and the t(3)/Laplace variants, which are not shown in the paper) | contamination of predictions, with population SCAS reference |
| `study3_bias.py` | Fig. 5 | `fig6_bias_sensitivity` | additive and multiplicative bias, with d, d_r, Theil U1 |
| `study4_regression.py` | Fig. 6, Table 3 | `fig8_regression_scenarios`, `table3_regression_main` | OLS estimation + out-of-sample evaluation: clean, Huber contamination, misspecification, leverage |
| `study5_sample_size.py` | Fig. 7, Table 4 | `fig9_sample_size`, `table4_sample_size` | sampling variability of SCAS vs N, compared with the asymptotic SD |
| `study6_scale_variant.py` | Fig. 8, Table 5 | `fig10_target_contamination_scale`, `table5_target_contamination_scale` | contamination of the observed target: SD scale vs MAD scale |

The output file numbers are internal and differ from the figure numbers in the paper;
the mapping is given in the table above.

Run everything with

```
for s in test_scas calibration study1_single_outlier study2_contamination study3_bias \
         study4_regression study5_sample_size study6_scale_variant; do .venv/bin/python $s.py; done
```

Studies 2, 4, 5 and 6 use all available CPU cores (a few minutes on a 16-core machine).

## Data

The Stackloss data (Brownlee, 1965; 21 daily observations of a plant oxidising
ammonia to nitric acid) are loaded from `statsmodels.datasets.stackloss`.
All other data are generated from the distributions stated in the scripts.
