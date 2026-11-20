import sys

import numpy as np
import pandas as pd

from hamls.problems import multiwell
from hamls.search import search
from hamls.theory import exact_factor, leapfrog_factor

from .common import CS, EPS, plot, row, save

KAPPAS = (10.0, 100.0, 1000.0)
SEEDS = range(5)


def main(quick=False):
    rows = []
    wells = [multiwell(KAPPAS, seed=s) for s in SEEDS]
    for j, kappa in enumerate(KAPPAS):
        probs = [w[j] for w in wells]
        p0 = probs[0]
        for method in ["leapfrog", "extragradient"]:
            for c in CS:
                K = int(-(-c * p0.kappa**0.5 // 1))
                runs = [search(p, K, method, eps=EPS) for p in probs]
                dist = max(np.linalg.norm(r.x - p.mode) / p.width for r, p in zip(runs, probs))
                base = {"basin kappa": kappa, "local kappa": p0.kappa, "method": method, "c": c}
                r = row(base, runs, K)
                r["leapfrog bound"] = leapfrog_factor(K, p0.eta, p0.alpha) if method == "leapfrog" else float("nan")
                r["exact path bound"] = exact_factor(K, p0.eta, p0.alpha)
                r["max final distance in widths"] = dist
                rows.append(r)
    df = pd.DataFrame(rows)
    save(df, "nonconvex")
    plot(df, "nonconvex", "basin kappa")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main("--quick" in sys.argv)