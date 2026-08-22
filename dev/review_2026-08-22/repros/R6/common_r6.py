"""Shared helpers for the R6 lane: asymmetric frames/tangents and dense oracles."""
import itertools
import math
import numpy as np

import t3toolbox.backend.tv_operations as tvo
import t3toolbox.backend.sampling_derivatives as pd

# Asymmetric structure family: distinct N, distinct nU, nD != nU, rL != rR, non-palindromic.
# Index d -> (N, nU, nD, rL(len d+1), rR(len d+1)); boundary bonds 1 unless stated.
ASYM = {
    1: ((5,), (3,), (2,), (1, 1), (1, 1)),
    2: ((5, 3), (3, 2), (2, 3), (1, 2, 1), (1, 3, 1)),
    3: ((3, 5, 4), (2, 3, 2), (3, 2, 4), (1, 2, 4, 1), (1, 3, 2, 1)),
    4: ((3, 5, 4, 6), (2, 3, 2, 4), (3, 2, 4, 2), (1, 2, 4, 3, 1), (1, 3, 2, 4, 1)),
}


def make_frame(d, C=(), rng=None, orth=False):
    rng = rng or np.random.default_rng(0)
    N, nU, nD, rL, rR = ASYM[d]
    R = lambda *s: rng.standard_normal(s)
    up = tuple(R(*C, nU[i], N[i]) for i in range(d))
    down = tuple(R(*C, rL[i], nD[i], rR[i + 1]) for i in range(d))
    left = tuple(R(*C, rL[i], nU[i], rL[i + 1]) for i in range(d))
    right = tuple(R(*C, rR[i], nU[i], rR[i + 1]) for i in range(d))
    return (up, down, left, right)


def make_var(d, K=(), C=(), rng=None):
    rng = rng or np.random.default_rng(1)
    N, nU, nD, rL, rR = ASYM[d]
    R = lambda *s: rng.standard_normal(s)
    dU = tuple(R(*K, *C, nD[i], N[i]) for i in range(d))
    dG = tuple(R(*K, *C, rL[i], nU[i], rR[i + 1]) for i in range(d))
    return (dU, dG)


def make_t3(d, C=(), rng=None):
    """Plain T3 with asymmetric ranks: tucker (nU_i, N_i), tt (r_i, nU_i, r_{i+1}) with r = rL."""
    rng = rng or np.random.default_rng(2)
    N, nU, nD, rL, rR = ASYM[d]
    R = lambda *s: rng.standard_normal(s)
    tucker = tuple(R(*C, nU[i], N[i]) for i in range(d))
    tt = tuple(R(*C, rL[i], nU[i], rL[i + 1]) for i in range(d))
    return (tucker, tt)


def t3_dense(x):
    tucker, tt = x
    d = len(tucker)
    C = tucker[0].shape[:-2]
    nc = len(C)
    # contract tt chain then tucker lift; loop over C elements for simplicity
    def one(cores_t, cores_tt):
        T = cores_tt[0][0]                      # (n0, r1)
        for i in range(1, d):
            T = np.tensordot(T, cores_tt[i], axes=([-1], [0]))
        T = T[..., 0]
        for i in range(d):
            T = np.moveaxis(np.tensordot(T, cores_t[i], axes=([i], [0])), -1, i)
        return T
    if nc == 0:
        return one(tucker, tt)
    out = np.zeros(C + tuple(u.shape[-1] for u in tucker))
    for c in itertools.product(*[range(n) for n in C]):
        out[c] = one([u[c] for u in tucker], [g[c] for g in tt])
    return out


def tangent_dense(frame, var):
    """Dense K+C+(N...) tensor of the tangent (frame, var); loops over K (frame has C only)."""
    dU, dG = var
    C = frame[0][0].shape[:-2]
    K = dU[0].shape[:dU[0].ndim - 2 - len(C)]
    if not K:
        return tvo.tv_to_dense(frame, var)
    N = tuple(u.shape[-1] for u in frame[0])
    out = np.zeros(K + C + N)
    for k in itertools.product(*[range(n) for n in K]):
        out[k] = tvo.tv_to_dense(frame, ([u[k] for u in dU], [g[k] for g in dG]))
    return out


def relerr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    den = np.linalg.norm(b)
    return np.linalg.norm(a - b) / den if den > 0 else np.linalg.norm(a)


def poly_jets(f, ww, pp, order, d):
    """Exact derivatives of s -> f(ww + s pp) (a polynomial of degree <= d) via a Vandermonde solve
    at d+1 nodes. Returns stack over t of f^(t)(0) = t! * coeff_t. Independent of the dense oracle."""
    nodes = np.linspace(-1.0, 1.0, d + 1) if d > 0 else np.array([0.0])
    vals = [np.asarray(f([w + s * p for w, p in zip(ww, pp)])) for s in nodes]
    V = np.vander(nodes, N=d + 1, increasing=True)          # vals[j] = sum_t c_t nodes[j]^t
    shp = vals[0].shape
    A = np.stack([v.reshape(-1) for v in vals], axis=0)      # (d+1, M)
    coef = np.linalg.solve(V, A)                             # (d+1, M)
    out = []
    for t in range(order + 1):
        out.append((math.factorial(t) * coef[t]).reshape(shp) if t <= d else np.zeros(shp))
    return np.stack(out, axis=0)
