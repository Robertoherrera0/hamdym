import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
EPS = 1e-6
CS = [0.5, 1, 2, 4, 8]


def save(df, name):
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, name + ".csv"), index=False)
    with open(os.path.join(RESULTS, name + "_table.tex"), "w") as fh:
        fh.write(df.to_latex(index=False, float_format=lambda v: f"{v:.3g}"))


def plot(df, name, group):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (key, method), part in df.groupby([group, "method"]):
        ax.plot(part["c"], part["gradient evals"], marker="o", linestyle="-" if method == "leapfrog" else "--", label=f"{group} {key:g} {method}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("K divided by sqrt(kappa)")
    ax.set_ylabel("gradient evaluations to reach the target")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, name + ".png"), dpi=150)
    plt.close(fig)


def row(base, runs, K):
    ratios = [r for run in runs for r in run.ratios]
    return {
        **base,
        "K": K,
        "moves": float(np.median([r.moves for r in runs])),
        "gradient evals": float(np.median([r.grads for r in runs])),
        "function evals": float(np.median([r.funs for r in runs])),
        "worst move ratio": max(ratios) if ratios else np.nan,
        "reached": sum(r.reached for r in runs),
        "stalled": sum(r.stalled for r in runs),
    }


def checkpoint_table(df, name="checkpoint_table"):
    sub = df[(df["spectrum"] == "log") & (df["method"] == "leapfrog") & (df["c"].isin([1, 2, 4]))]
    sub = sub[sub["kappa"].isin([sub["kappa"].min(), sub["kappa"].max()])]
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$\kappa$ & $K/\sqrt{\kappa}$ & $K$ & moves & gradient evaluations & worst contraction & proved bound \\",
        r"\midrule",
    ]
    for _, r in sub.iterrows():
        bound = "none" if r["leapfrog bound"] >= 1 else f"{r['leapfrog bound']:.3f}"
        lines.append(
            f"{int(r['kappa'])} & {r['c']:g} & {int(r['K'])} & {int(r['moves'])} & {int(r['gradient evals'])} & {r['worst move ratio']:.3f} & {bound} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(RESULTS, name + ".tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
