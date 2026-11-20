# Hamiltonian trajectories as a continuous local search

Code for the CMP_SC 8050 final project. The local search starts at a point x, releases a particle from rest, follows K steps of an integrator, and moves to the lowest point on that path (the start included). Cost is counted in gradient and function evaluations.

## Setup

    pip install -r requirements.txt

Run every command from this folder.

## Tests

    python -m pytest tests

## Experiments

    python -m experiments.quadratic
    python -m experiments.convex
    python -m experiments.nonconvex

Add `--quick` for a smaller run. Each experiment writes a csv, a LaTeX table and a figure to `results/`.

| Experiment | Problem | Question |
|---|---|---|
| quadratic | quadratics with log, uniform and two eigenvalue spectra | does the per move bound of theory/theory.tex hold, and how does cost scale with K and kappa |
| convex | L2 regularized logistic regression | where no leapfrog proof exists, how close is the cost to the exact path number |
| nonconvex | mixture of wells with different local kappa | does the cost inside a basin follow the local kappa |

## Layout

    hamls/integrators.py   leapfrog and extragradient paths
    hamls/search.py        the move (lowest point on the path) and the search loop with counters
    hamls/problems.py      instance generators
    hamls/theory.py        bounds from theory/theory.tex
    experiments/           one script per experiment
    tests/                 closed form, counters and bound checks
    theory/theory.tex      proofs

## Conventions

- Step size is 1 divided by the square root of L, which keeps leapfrog stable.
- K is set as c times the square root of kappa, for c in 0.5, 1, 2, 4, 8.
- Cost is the median over 5 seeds. The target is a relative gap of 1e-6.
- Gradients at a cached point are reused, so a move costs K gradient and K function evaluations, and the search costs one extra of each at the start. Extragradient costs two gradients per step.
- The leapfrog bound is proved for quadratics only. In the convex and nonconvex tables it is a reference, not a guarantee.
