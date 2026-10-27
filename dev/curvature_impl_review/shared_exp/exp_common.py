"""Shared vs unshared exact-Newton experiment: problem builders, run summaries, final-point spectra.

Run from the snapshot root with PYTHONPATH=$PWD. Numpy only, eager. See run_case.py / aggregate.py."""
import time
import numpy as np
import scipy.linalg as sla

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.sharing as sharing

MANIFOLD = bgeo.ManifoldGeometryOps()

SHAPE = (6, 6, 6, 6)
import os
W = int(os.environ.get('T3EXP_W', '60'))
NOISE = 0.01
TARGET = ((5, 5, 5, 5), (1, 3, 4, 3, 1))          # misspecified target (modes 0,3: ceiling 3 < 5)
LEVEL1 = ((2, 2, 2, 2), (1, 2, 2, 2, 1))          # continuation level 1
RESTART_SHARED = ((3, 3, 3, 3), (1, 2, 3, 2, 1))  # group rank 3 > modes 0,3 own ceiling 1*2 = 2 (group-ceiling)
RESTART_UNSHARED = ((2, 3, 3, 2), (1, 2, 3, 2, 1))  # same TT ranks, Tucker clipped to the per-mode ceilings
LEVEL1_STEPS = 40
WARM_STEPS = 10
EXACT_MAX = 40
GN_MAX = 72          # rows = states x_0..x_71; restart runs are read to x_60, warm = x_10..x_70
GTOL = 1e-8


def make_problem(seed, labels):
    """labels=None: untied target, unshared geometry only. Else a tied target and BOTH geometries on the same data."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    target = t3.TuckerTensorTrain.randn(SHAPE, *TARGET)
    if labels is not None:
        target = t3.TuckerTensorTrain(*sharing.t3_tie_tucker_factors(target.data, labels))
    ww = [rng.standard_normal((W, N)) for N in SHAPE]
    pp = [rng.standard_normal((W, N)) for N in SHAPE]
    kind = bfit.probe_derivatives_kind(2, np.array([1.0, 0.5, 0.25]))
    data = [z + NOISE * np.linalg.norm(z) / z.size ** 0.5 * rng.standard_normal(z.shape)
            for z in kind.point_forward(target.data, (ww, pp))]
    x0 = t3.TuckerTensorTrain.randn(SHAPE, *LEVEL1).data
    if labels is not None:
        x0 = sharing.t3_tie_tucker_factors(x0, labels)
    probs = {'unshared': bopt.least_squares_problem(MANIFOLD, kind, (ww, pp), data)}
    if labels is not None:
        probs['shared'] = bopt.least_squares_problem(MANIFOLD.with_sharing(labels, SHAPE), kind, (ww, pp), data)
    return probs, x0


def restart_point(problem, x0, labels):
    """Level-1 GN fit, then the zero-padded resize restart (shared resize when labels is not None)."""
    x1, st = bopt.newton_cg(problem, x0, max_newton=LEVEL1_STEPS, gtol_rel=1e-10)
    if labels is None:
        r = t3.TuckerTensorTrain(*x1).resize(SHAPE, *RESTART_UNSHARED).data
    else:
        r = t3.TuckerTensorTrain(*x1).resize(SHAPE, *RESTART_SHARED, sharing=labels).data
    return r, st


def _e(h):
    return [row['gnorm'] / row['g0norm'] for row in h]


def summarize(h, data_norm, cut_e=None, cap=None):
    """A run summary from history rows. For GN runs read to a cap, `cut_e` truncates at the first row with
    e <= cut_e (that row = the converged line, as a gtol_rel=cut_e run would stop)."""
    h = list(h)
    stop = h[-1].get('stop_reason')
    if cut_e is not None:            # h = states x_0..x_cap (row i = the point before step i)
        e = _e(h)
        k = next((i for i, v in enumerate(e) if v <= cut_e), None)
        if k is not None:
            h, stop = h[:k + 1], 'gradient'
            steps = h[:k]
        else:
            steps = h[:-1] if len(h) > 1 else h
            stop = stop if stop not in (None, 'gradient') else 'max_newton'
    else:
        steps = [r for r in h if not r['converged']]
    e = _e(h)
    tail = e[-4:]
    ratios = [b / a for a, b in zip(tail, tail[1:])]
    acc = [r for r in steps if r.get('tr_accepted') is not False and r.get('ls_failed') is not True]
    kappa = [r['pKp'] / (r['pHp'] - r['pKp']) for r in steps
             if r.get('pKp') is not None and r.get('tr_accepted')]
    return dict(outer=len(steps), accepted=len(acc), cg=int(sum(r['cg_iters'] for r in steps)),
                stop=stop, final_e=e[-1], final_g=h[-1]['gnorm'],
                rel_misfit=(2 * h[-1]['misfit']) ** 0.5 / data_norm,
                wall=float(sum(r['wall_time'] or 0.0 for r in steps)), tail_e=tail, tail_ratios=ratios,
                kappa_steps=kappa, rho=[r['rho'] for r in steps])


def gn_steps_to(h, gtarget, cap=60):
    for i, r in enumerate(h[:cap + 1]):
        if r['gnorm'] <= gtarget:
            return i
    return None


def gn_tail_rate(h, lo=1e-9, hi=1e-2):
    e = _e(h)
    return [b / a for a, b in zip(e, e[1:]) if lo < a < hi and b > 0]


# ---- final-point curvature spectrum ------------------------------------------------------------------
def _flat(t, K):
    return np.concatenate([np.asarray(a).reshape(K, -1) for a in t[0] + t[1]], axis=1)


def _unflat(M, shapes):
    """M: (K, n) rows -> a K-stacked tangent with the variation shapes."""
    K = M.shape[0]
    out, i = [], 0
    for grp in shapes:
        cores = []
        for s in grp:
            n = int(np.prod(s))
            cores.append(M[:, i:i + n].reshape((K,) + tuple(s)))
            i += n
        out.append(tuple(cores))
    return tuple(out)


def final_point_curvature(problem, x, n_random=10, seed=0):
    """At x: the curvature-to-GN ratios on random gauged tangents, and the pencil (K, H_GN) on the whole
    (tied) horizontal space -- max |lambda| = rho(H_GN^-1 K), the GN linear-rate constant at a minimizer."""
    geom = problem.geom
    lmx = problem.local_model(x, hessian='exact')
    lmg = problem.local_model(x)
    frame, aux = lmx.frame, lmx.geom_aux
    shapes = fvo.fv_variation_shapes(frame)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    P = geom.project(frame, fvo.fv_variations_randn(shapes, (n_random,)), aux=aux)
    Kp, Gp = lmx.curvature_hvp(P), lmg.hvp(P)
    kq = np.asarray(geom.inner(P, Kp))
    gq = np.asarray(geom.inner(P, Gp))
    nk = np.sqrt(np.asarray(geom.inner(Kp, Kp)))
    ng = np.sqrt(np.asarray(geom.inner(Gp, Gp)))
    # horizontal-space basis: project n_coord random tangents, orthonormalize, keep the range
    ncoord = sum(int(np.prod(s)) for grp in shapes for s in grp)
    R = geom.project(frame, fvo.fv_variations_randn(shapes, (ncoord,)), aux=aux)
    U, s, _ = np.linalg.svd(_flat(R, ncoord).T, full_matrices=False)
    r = int(np.sum(s > 1e-10 * s[0]))
    B = U[:, :r].T                                         # (r, ncoord), orthonormal rows in the coordinate dot
    Bt = geom.project(frame, _unflat(B, shapes), aux=aux)  # already in range; project as a safeguard
    KB, GB = _flat(lmx.curvature_hvp(Bt), r), _flat(lmg.hvp(Bt), r)
    Bf = _flat(Bt, r)
    Km, Gm = Bf @ KB.T, Bf @ GB.T
    asym = max(np.abs(Km - Km.T).max() / np.abs(Km).max(), np.abs(Gm - Gm.T).max() / np.abs(Gm).max())
    Km, Gm = 0.5 * (Km + Km.T), 0.5 * (Gm + Gm.T)
    lam = sla.eigh(Km, Gm, eigvals_only=True)
    gev = np.linalg.eigvalsh(Gm)
    return dict(rayleigh=(kq / gq).tolist(), norm_ratio=(nk / ng).tolist(), horiz_dim=r,
                lam_min=float(lam.min()), lam_max=float(lam.max()), rho_gn=float(np.abs(lam).max()),
                lam_top=sorted(np.abs(lam).tolist())[-3:], lam_all=lam.tolist(), asym=float(asym),
                gn_cond=float(gev.max() / gev.min()),
                lift_sigma_min=float(np.min(np.asarray(lmx.curvature.geometry_data.sigma_min))),
                rel_misfit=None)


def manifold_dim(labels, shared):
    if shared:
        return t3m.manifold_dim((SHAPE,) + RESTART_SHARED, sharing=labels)
    return t3m.manifold_dim((SHAPE,) + RESTART_UNSHARED)


def data_norm(problem):
    return float(problem.kind.sumsq(problem.data, problem.kind.w_axes(problem.sample))) ** 0.5
