import sys

import pandas as pd

from hamls.problems import logistic
from hamls.search import search
from hamls.theory import exact_factor

from .common import CS, EPS, plot, row, save

ALPHAS = [1e-1, 1e-2, 1e-3]
SEEDS = range(5)


def main(quick=False):
    alphas = ALPHAS[:2] if quick else ALPHAS
    rows = []
    for alpha in alphas:
        probs = [logistic(alpha=alpha, seed=s) for s in SEEDS]
        p0 = probs[0]
        for method in ["leapfrog", "extragradient"]:
            for c in CS:
                K = int(-(-c * p0.kappa**0.5 // 1))
                runs = [search(p, K, method, eps=EPS) for p in probs]
                base = {"alpha": alpha, "kappa": p0.kappa, "method": method, "c": c}
                r = row(base, runs, K)
                r["exact path bound"] = exact_factor(K, p0.eta, p0.alpha)
                rows.append(r)
    df = pd.DataFrame(rows)
    save(df, "convex")
    plot(df, "convex", "alpha")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main("--quick" in sys.argv)