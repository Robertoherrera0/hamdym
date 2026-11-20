import numpy as np

from hamls.integrators import extragradient, leapfrog
from hamls.problems import multiwell, quadratic
from hamls.search import move, search
from hamls.theory import exact_factor, leapfrog_factor, theta


def test_leapfrog_matches_closed_form():
    p = quadratic(200.0, d=12, seed=1)
    lam = p.grad(np.ones(12))
    x0 = p.x0
    pts = [x0] + [x for x, _ in leapfrog(p, x0, p.grad(x0), p.eta, 30)]
    th = theta(p.eta, lam)
    for k, x in enumerate(pts):
        assert np.allclose(x, np.cos(k * th) * x0, atol=1e-9)


def test_extragradient_first_step():
    p = quadratic(50.0, d=5, seed=2)
    x0 = p.x0
    x1, _ = next(extragradient(p, x0, p.grad(x0), p.eta, 3))
    assert np.allclose(x1, x0 - p.eta**2 * p.grad(x0))


def test_move_counts():
    p = quadratic(100.0, d=10, seed=3)
    p.reset()
    x = p.x0
    g = p.gradient(x)
    fx = p.value(x)
    move(p, x, fx, g, 17, p.eta, "leapfrog")
    assert p.ngrad == 1 + 17
    assert p.nfun == 1 + 17


def test_per_move_bound_on_quadratics():
    worst = 0.0
    for kappa in [1e2, 1e3]:
        for spectrum in ["log", "uniform", "two"]:
            for seed in range(5):
                p = quadratic(kappa, d=30, seed=seed, spectrum=spectrum)
                for c in [1, 2, 4]:
                    K = int(np.ceil(c * np.sqrt(kappa)))
                    run = search(p, K, "leapfrog", eps=1e-4)
                    bound = leapfrog_factor(K, p.eta, p.alpha)
                    if run.ratios:
                        assert max(run.ratios) <= bound + 1e-12
                        worst = max(worst, max(run.ratios))
    assert worst < 1.0


def test_four_sqrt_kappa_gives_two_thirds():
    for kappa in [1e2, 1e4]:
        p = quadratic(kappa)
        K = int(np.ceil(4 * np.sqrt(kappa)))
        assert leapfrog_factor(K, p.eta, p.alpha) <= 2 / 3


def test_exact_factor_at_two_over_sqrt_alpha():
    alpha, eta = 1e-2, 1.0
    K = 20
    assert abs(exact_factor(K, eta, alpha) - 5 / 6) < 1e-12


def test_multiwell_modes_are_local_minima():
    for p in multiwell(seed=0):
        assert np.linalg.norm(p.grad(p.mode)) < 1e-6
        assert p.alpha > 0
