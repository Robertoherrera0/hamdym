import numpy as np


def theta(eta, lam):
    return 2 * np.arcsin(eta * np.sqrt(lam) / 2)


def leapfrog_factor(K, eta, alpha):
    return min(1.0, 0.5 + 1 / (2 * K * np.sin(theta(eta, alpha))))


def exact_factor(K, eta, alpha):
    T = K * eta
    return min(1.0, 2 / 3 + 2 / (3 * T**2 * alpha))


def moves_needed(factor, eps):
    if factor >= 1:
        return np.inf
    return float(np.ceil(np.log(1 / eps) / np.log(1 / factor)))