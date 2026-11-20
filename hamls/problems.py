import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


class Problem:
    def __init__(self, name, f, grad, x0, f_star, alpha, L):
        self.name = name
        self.f = f
        self.grad = grad
        self.x0 = x0
        self.f_star = f_star
        self.alpha = alpha
        self.L = L
        self.ngrad = 0
        self.nfun = 0

    def value(self, x):
        self.nfun += 1
        return self.f(x)

    def gradient(self, x):
        self.ngrad += 1
        return self.grad(x)

    def reset(self):
        self.ngrad = 0
        self.nfun = 0

    @property
    def eta(self):
        return 1.0 / np.sqrt(self.L)

    @property
    def kappa(self):
        return self.L / self.alpha


def quadratic(kappa, d=50, seed=0, spectrum="log"):
    L = 1.0
    alpha = L / kappa
    if spectrum == "log":
        lam = np.exp(np.linspace(np.log(alpha), np.log(L), d))
    elif spectrum == "uniform":
        lam = np.linspace(alpha, L, d)
    elif spectrum == "two":
        lam = np.where(np.arange(d) < d // 2, alpha, L)
    else:
        raise ValueError(spectrum)
    x0 = np.random.default_rng(seed).normal(size=d)
    return Problem(
        f"quadratic_{spectrum}",
        lambda x: 0.5 * np.sum(lam * x * x),
        lambda x: lam * x,
        x0,
        0.0,
        alpha,
        L,
    )


def logistic(n=500, d=20, alpha=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = np.sign(X @ w_true + 0.5 * rng.normal(size=n))

    def f(w):
        return np.mean(np.logaddexp(0.0, -y * (X @ w))) + 0.5 * alpha * (w @ w)

    def grad(w):
        s = expit(-y * (X @ w))
        return -(X.T @ (y * s)) / n + alpha * w

    L = np.linalg.eigvalsh(X.T @ X)[-1] / (4 * n) + alpha
    res = minimize(f, np.zeros(d), jac=grad, method="BFGS", options={"gtol": 1e-12, "maxiter": 10000})
    x0 = rng.normal(size=d)
    return Problem("logistic", f, grad, x0, res.fun, alpha, L)


def multiwell(kappas=(10.0, 100.0, 1000.0), dim=4, spread=30.0, seed=0):
    rng = np.random.default_rng(seed)
    m = len(kappas)
    centers = np.zeros((m, dim))
    for j in range(m):
        centers[j, 0] = spread * j
    precs = [np.exp(np.linspace(0.0, np.log(k), dim)) for k in kappas]
    logw = np.full(m, -np.log(m))

    def parts(x):
        diff = x[None, :] - centers
        v = np.array([precs[j] * diff[j] for j in range(m)])
        q = np.array([diff[j] @ v[j] for j in range(m)])
        a = logw - 0.5 * q
        return v, a

    def f(x):
        _, a = parts(x)
        return -logsumexp(a)

    def grad(x):
        v, a = parts(x)
        r = np.exp(a - logsumexp(a))
        return r @ v

    def hess(x):
        v, a = parts(x)
        r = np.exp(a - logsumexp(a))
        mean = r @ v
        H = sum(r[j] * np.diag(precs[j]) - r[j] * np.outer(v[j], v[j]) for j in range(m))
        return H + np.outer(mean, mean)

    problems = []
    for j in range(m):
        res = minimize(f, centers[j], jac=grad, method="BFGS", options={"gtol": 1e-12})
        mode = res.x
        ev = np.linalg.eigvalsh(hess(mode))
        z = rng.normal(size=dim)
        x0 = mode + z / np.sqrt(precs[j])
        p = Problem(f"multiwell_{j}", f, grad, x0, res.fun, ev[0], ev[-1])
        p.mode = mode
        p.width = 1.0 / np.sqrt(precs[j][0])
        problems.append(p)
    return problems