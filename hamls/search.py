from dataclasses import dataclass, field

import numpy as np

from .integrators import INTEGRATORS


@dataclass
class Run:
    moves: int
    grads: int
    funs: int
    ratios: list
    reached: bool
    stalled: bool
    x: np.ndarray
    trace: list = field(default_factory=list)


def move(problem, x, fx, g, K, eta, method):
    best = (x, fx, g)
    for xk, gk in INTEGRATORS[method](problem, x, g, eta, K):
        fk = problem.value(xk)
        if fk < best[1]:
            best = (xk, fk, gk)
    return best


def search(problem, K, method="leapfrog", eta=None, eps=1e-6, max_moves=5000):
    eta = problem.eta if eta is None else eta
    problem.reset()
    x = problem.x0.copy()
    g = problem.gradient(x)
    fx = problem.value(x)
    gap0 = fx - problem.f_star
    ratios, trace = [], [(problem.ngrad, gap0)]
    moves, stalled = 0, False
    while fx - problem.f_star > eps * gap0 and moves < max_moves:
        nx, nf, ng = move(problem, x, fx, g, K, eta, method)
        if nf >= fx:
            stalled = True
            break
        ratios.append((nf - problem.f_star) / (fx - problem.f_star))
        x, fx, g = nx, nf, ng
        moves += 1
        trace.append((problem.ngrad, fx - problem.f_star))
    reached = fx - problem.f_star <= eps * gap0
    return Run(moves, problem.ngrad, problem.nfun, ratios, reached, stalled, x, trace)