"""Stage D: temperature, novelty posterior, prior-shift EM, beta offset."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


def softmax_groups(z, gid, T):
    """Softmax of z/T within each group id (gid sorted contiguous not required)."""
    z = z / T
    mx = np.full(gid.max() + 1, -np.inf); np.maximum.at(mx, gid, z)
    e = np.exp(z - mx[gid])
    s = np.zeros(gid.max() + 1); np.add.at(s, gid, e)
    return e / s[gid]


def fit_temperature(z, gid, y):
    """NLL of the true candidate, over groups that contain a positive."""
    has = np.zeros(gid.max() + 1, bool); has[gid[y == 1]] = True
    m = has[gid]
    zz, gg, yy = z[m], np.unique(gid[m], return_inverse=True)[1], y[m]

    def nll(T):
        p = softmax_groups(zz, gg, T)
        return -np.log(np.clip(p[yy == 1], 1e-12, 1)).mean()
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def prior_shift(pi, pi_train, pi_target):
    r1 = pi_target / pi_train; r0 = (1 - pi_target) / (1 - pi_train)
    return r1 * pi / (r1 * pi + r0 * (1 - pi))


def em_prior(pi, pi_train, tol=1e-5, it=200):
    """Saerens et al. 2002: estimate the target prior of 'new' from unlabelled posteriors."""
    t = pi_train
    for _ in range(it):
        t_new = prior_shift(pi, pi_train, t).mean()
        if abs(t_new - t) < tol:
            break
        t = t_new
    return t


def shift_logit(p, beta):
    return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) + beta)
