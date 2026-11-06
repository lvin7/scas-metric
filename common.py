"""Shared helpers for the simulation-study scripts: output paths, data, LaTeX tables."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(RESULTS, "figs")
TABLES = os.path.join(RESULTS, "tables")
DATA = os.path.join(RESULTS, "data")
for _d in (RESULTS, FIGS, TABLES, DATA):
    os.makedirs(_d, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
})


def savefig(fig, name: str):
    """Save figure as PNG (and PDF) under results/figs."""
    fig.savefig(os.path.join(FIGS, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.join("results/figs", name + ".png"))


def save_table(df: pd.DataFrame, name: str, **kw):
    """Save a DataFrame as CSV and as a booktabs LaTeX table under results/tables."""
    df.to_csv(os.path.join(TABLES, name + ".csv"), index=kw.pop("index", False))
    tex = df.to_latex(index=False, escape=False, float_format=lambda v: f"{v:.3f}", **kw)
    with open(os.path.join(TABLES, name + ".tex"), "w") as fh:
        fh.write(tex)
    print("  wrote", os.path.join("results/tables", name + ".tex"))


def save_data(df: pd.DataFrame, name: str):
    df.to_csv(os.path.join(DATA, name + ".csv.gz"), index=False, compression="gzip")
    print("  wrote", os.path.join("results/data", name + ".csv.gz"))


def load_stackloss() -> pd.DataFrame:
    """
    Brownlee's (1965) stack-loss data: 21 daily observations of a plant oxidising
    ammonia to nitric acid. Columns: AIRFLOW, WATERTEMP, ACIDCONC, STACKLOSS.
    """
    import statsmodels.api as sm
    return sm.datasets.stackloss.load_pandas().data


def seeds(master: int, n: int) -> np.ndarray:
    """Independent child seeds from a master seed (reproducible across runs)."""
    return np.random.default_rng(master).integers(0, 2**31 - 1, size=n)
