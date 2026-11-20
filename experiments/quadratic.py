import sys

import pandas as pd

from hamls.problems import quadratic
from hamls.search import search
from hamls.theory import exact_factor, leapfrog_factor, moves_needed

from .common import CS, EPS, plot, row, save

KAPPAS = [1e2, 1e3, 1e4]
SPECTRA = ["log", "uniform", "two"]
SEEDS = range(5)


def main(quick=False):
    kappas = KAPPAS[:2] if quick else KAPPAS
    rows = []
    for spectrum in SPECTRA:
        for kappa in kappas:
            probs = [quadratic(kappa, seed=s, spectrum=spectrum) for s in SEEDS]
            p0 = probs[0]
            for method in ["leapfrog", "extragradient"]:
                for c in CS:
                    K = int(-(-c * p0.kappa**0.5 // 1))
                    runs = [search(p, K, method, eps=EPS) for p in probs]
                    lf = leapfrog_factor(K, p0.eta, p0.alpha) if method == "leapfrog" else float("nan")
                    ex = exact_factor(K, p0.eta, p0.alpha)
                    base = {"spectrum": spectrum, "kappa": kappa, "method": method, "c": c}
                    r = row(base, runs, K)
                    r["leapfrog bound"] = lf
                    r["exact path bound"] = ex
                    r["moves by leapfrog bound"] = moves_needed(lf, EPS) if method == "leapfrog" else float("nan")
                    rows.append(r)
    df = pd.DataFrame(rows)
    save(df, "quadratic")
    plot(df[df["spectrum"] == "log"], "quadratic", "kappa")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main("--quick" in sys.argv)