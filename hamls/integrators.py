import numpy as np


def leapfrog(problem, x, g, eta, K):
    yh = -0.5 * eta * g
    for _ in range(K):
        x = x + eta * yh
        g = problem.gradient(x)
        yield x, g
        yh = yh - eta * g


def extragradient(problem, x, g, eta, K):
    y = np.zeros_like(x)
    first = True
    for _ in range(K):
        xh = x + eta * y
        gh = g if first else problem.gradient(xh)
        first = False
        x = xh - eta**2 * gh
        g = problem.gradient(x)
        y = y - eta * g
        yield x, g


INTEGRATORS = {"leapfrog": leapfrog, "extragradient": extragradient}
