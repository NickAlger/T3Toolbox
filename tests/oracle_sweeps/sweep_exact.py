"""Exact-Hessian oracle sweep (2026.3.0; new, not a copy of the 2026-08-22 review's sweeps): the Hessian action
of ``fitting.LocalModel`` built with ``hessian='exact'`` against a DENSE EXTRINSIC oracle, over representation x
geometry x sharing x kind x order x weight x regularizer (and ``d``; the shared ``d = 3`` point is group-ceiling, so
the tied lift is discriminated from a per-mode one). No library sampling, jet, lift or curvature
code is used by the oracle -- only ``oracle.py``'s dense sampling maps and a hand-written einsum of the cores.

The oracle (Absil, Mahony & Trumpf 2013, "An extrinsic look at the Riemannian Hessian"):

* ``f(X) = 1/2 ||omega (.) (A X - y)||^2`` is quadratic in the dense ``X``, with ``A`` the kind's dense sampling
  matrix (one ``oracle.S`` per basis tensor), so ``grad f = A_w^T r_w`` and ``nabla^2 f = A_w^T A_w``.
* The tangent space at ``X = Phi(theta)`` is the range of the parametrization's dense Jacobian ``J`` (``Phi`` the
  multilinear core map, ``Phi o T`` -- one factor per sharing group -- on a shared geometry), ``P = J J^+``. Along
  ``theta + t theta_dot`` (``J theta_dot = xi``) the rank is constant, so ``DP[xi] = P_perp DJ J^+ + (.)^T`` with
  ``DJ = d/dt J`` EXACT (``Phi`` is multilinear), and the Riemannian Hessian is
  ``Hess f [xi] = P nabla^2 f xi + (J^+)^T DJ^T P_perp grad f`` (the Weingarten term) ``+ lambda xi`` (the
  identity regularizer's ``1/2 lambda ||X||^2``: ``X`` lies in its own tangent space, so no normal part).
* On a corewise geometry the exact Hessian is the plain Hessian of ``f o Phi`` in the raw cores,
  ``J^T nabla^2 f J p + DJ[p]^T grad f + lambda p``, tied on a shared one by the per-group mean.

The library side runs at ``curvature_damping=0.0`` (the undamped Hessian; the default ``1e-6`` would move it by
that much), on a random point with a random high-residual target, so the curvature term is a sizable part of the
action (asserted, ``> 1e-3`` of it, except on the flat ``d = 1`` manifold, where it vanishes and the library model
is Gauss-Newton). Every case seeds its own global and ``default_rng`` streams. Checks: ``hvp`` (dense on the
manifold, raw core coordinates on corewise), the quadratic form ``quadratic(p)``, and symmetry
``<q, H p> = <p, H q>``."""
import math, sys, time
import numpy as np
sys.path.insert(0, '.')
import oracle as O
import sweep_models as SM
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.shared_geometry as sg
import t3toolbox.fitting as fitting
from t3toolbox.backend.regularization import IdentityRegularizer

LAM = 0.3
TOL = 1e-9          # observed <= ~1e-12 (the oracle's pinv conditions the error, not the library)
CURV_MIN = 1e-3     # the curvature term's share of the action: the check is discriminating
failures, rows = [], []


# ---- the dense parametrization Phi and its exact Jacobians (einsum only) -------------------------------------
def phi(slots, d):
    """Dense X from the 2d raw cores ``slots = tucker[0..d-1] + tt[0..d-1]`` (``tucker[i]: (n_i, N_i)``,
    ``tt[i]: (r_i, n_i, r_{i+1})``)."""
    G = slots[d]
    for i in range(1, d):
        G = np.tensordot(G, slots[d + i], axes=([G.ndim - 1], [0]))
    G = G[0, ..., 0]
    for i in range(d):
        G = np.tensordot(G, slots[i], axes=([0], [0]))       # contracts n_i, appends N_i
    return G


def _subst(slots, *pairs):
    out = list(slots)
    for s, a in pairs:
        out[s] = a
    return out


def _units(shape):
    n = math.prod(shape)
    for e in range(n):
        E = np.zeros(n)
        E[e] = 1.0
        yield E.reshape(shape)


def jacobian(slots, params, d):
    """``J[:, k] = d Phi / d theta_k`` for the parameters ``params`` (each a tuple of the slots it fills: a tied
    group's Tucker slots share one parameter)."""
    cols = []
    for pslots in params:
        for E in _units(slots[pslots[0]].shape):
            cols.append(sum(phi(_subst(slots, (s, E)), d) for s in pslots).ravel())
    return np.stack(cols, axis=1)


def jacobian_derivative(slots, params, d, dot):
    """``d/dt J(theta + t theta_dot)`` at 0, exactly: ``dot`` is ``theta_dot`` per SLOT (tied copies repeated)."""
    cols = []
    for pslots in params:
        for E in _units(slots[pslots[0]].shape):
            col = 0.0
            for s in pslots:
                for s2 in range(2 * d):
                    if s2 != s:
                        col = col + phi(_subst(slots, (s, E), (s2, dot[s2])), d)
            cols.append(np.ravel(col))
    return np.stack(cols, axis=1)


def unpack(vec, slots, params):
    """A parameter vector -> per-slot arrays (a tied group's direction copied to each of its slots)."""
    out, k = [None] * len(slots), 0
    for pslots in params:
        shape = slots[pslots[0]].shape
        n = math.prod(shape)
        for s in pslots:
            out[s] = vec[k:k + n].reshape(shape)
        k += n
    return out


def make_params(d, sharing):
    """Slot tuples: per Tucker sharing group (or per mode), then one per TT core."""
    if sharing is None:
        tucker = [(i,) for i in range(d)]
    else:
        groups = {}
        for i, g in enumerate(sharing):
            groups.setdefault(g, []).append(i)
        tucker = [tuple(v) for v in groups.values()]
    return tucker + [(d + i,) for i in range(d)]


def mean_project(vec, slots, d, sharing):
    """The shared corewise geometry's tie: each group's Tucker slots replaced by their mean (expanded coordinates)."""
    if sharing is None:
        return vec
    arrs = unpack(vec, slots, [(s,) for s in range(2 * d)])
    for g in set(sharing):
        modes = [i for i, h in enumerate(sharing) if h == g]
        m = sum(arrs[i] for i in modes) / len(modes)
        for i in modes:
            arrs[i] = m
    return np.concatenate([a.ravel() for a in arrs])


# ---- the dense objective -----------------------------------------------------------------------------------
def _flat(z):
    if isinstance(z, (list, tuple)):
        return np.concatenate([np.ravel(np.asarray(a)) for a in z])
    return np.ravel(np.asarray(z))


def _zeros_like_output(z):
    return [np.zeros_like(np.asarray(a)) for a in z] if isinstance(z, (list, tuple)) else np.zeros_like(np.asarray(z))


def sampling_matrix(kind, shape, sample, order, weight):
    """``A_w``: column ``k`` is ``omega (.) S(E_k)`` for the dense basis tensor ``E_k``."""
    cols = []
    for E in _units(shape):
        Se = O.S(kind, E, sample, order)
        cols.append(_flat(O.weighted_residual(kind, Se, _zeros_like_output(Se), weight, order)))
    return np.stack(cols, axis=1)


# ---- one case ----------------------------------------------------------------------------------------------
def build_exact_model(geom, x, kind, sample, residual, order, weight, reg, damping):
    kw = dict(regularizer=reg, hessian='exact', curvature_damping=damping)
    if kind == 'apply':
        return fitting.apply_model(geom, x, sample, residual, **kw)
    if kind == 'entries':
        return fitting.entries_model(geom, x, sample, residual, **kw)
    if kind == 'probe':
        return fitting.probe_model(geom, x, sample, residual, weight=weight, **kw)
    if kind == 'apply_derivatives':
        return fitting.apply_derivatives_model(geom, x, sample[0], sample[1], order, residual, weight=weight, **kw)
    if kind == 'entries_derivatives':
        return fitting.entries_derivatives_model(geom, x, sample[0], sample[1], order, residual, weight=weight, **kw)
    return fitting.probe_derivatives_model(geom, x, sample[0], sample[1], order, residual, weight=weight, **kw)


def _ragged_tangent(v):
    return v.to_t3tangent() if isinstance(v, ut3m.UT3Tangent) else v


def _coords(v):
    v = _ragged_tangent(v)
    return np.concatenate([np.ravel(np.asarray(a)) for a in v.variations.data[0] + v.variations.data[1]])


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / (np.linalg.norm(np.asarray(b)) + 1e-300))


def build_point(d, shared, seed):
    """``sweep_models.build_point``, except the shared ``d = 3`` point is GROUP-CEILING: the group rank 4 exceeds
    mode 0's own ceiling ``r_0 r_1 = 2`` (its ``S_0^T`` is wide), the structure on which a per-mode lift differs
    from the tied one -- on ``sweep_models``' shared point every ``S_i^T`` is square and the two lifts coincide."""
    if not (shared and d == 3):
        return SM.build_point(d, shared, seed)
    rng = np.random.default_rng(seed)
    x = t3.TuckerTensorTrain.randn((6, 6, 4), (4, 4, 2), (1, 2, 2, 1))
    tk = [rng.standard_normal(c.shape) for c in x.data[0]]
    tk[1] = tk[0]
    return t3.TuckerTensorTrain(tuple(tk), tuple(rng.standard_normal(c.shape) for c in x.data[1])), (0, 0, 1)


def run_case(d, rep, geom_name, shared, kind, order, wspec, regspec, seed=13, W=(12,)):
    np.random.seed(seed)
    x, sharing = build_point(d, shared, seed)
    rng = np.random.default_rng(seed + 1)
    shape = x.shape
    sample = SM.make_sample(kind, shape, W, rng)
    weight = SM.make_weight(kind, d, order or 0, wspec)
    reg = IdentityRegularizer(LAM) if regspec else None
    A0 = t3.TuckerTensorTrain.randn(shape, x.tucker_ranks, x.tt_ranks)
    A = t3.TuckerTensorTrain(tuple(rng.standard_normal(c.shape) for c in A0.data[0]),
                             tuple(rng.standard_normal(c.shape) for c in A0.data[1]))
    y = O.S(kind, A.to_dense(), sample, order)
    Sx = SM.lib_S(x, kind, sample, order)
    residual = ([np.asarray(a) - np.asarray(b) for a, b in zip(Sx, y)] if kind.startswith('probe')
                else np.asarray(Sx) - np.asarray(y))

    if rep == 'ragged':
        base, pt = (t3m.MANIFOLD if geom_name == 'manifold' else t3m.COREWISE), x
    else:
        base = ut3m.UNIFORM_MANIFOLD if geom_name == 'manifold' else ut3m.UNIFORM_COREWISE
        pt = ut3.UniformTuckerTensorTrain.from_t3(x)
    geom = sg.shared(base, sharing) if shared else base
    model = build_exact_model(geom, pt, kind, sample, residual, order, weight, reg,
                              0.0 if geom_name == 'manifold' else None)
    p = geom.project(geom.randn(model.frame))
    q = geom.project(geom.randn(model.frame))

    # the dense objective at X
    slots = list(x.data[0]) + list(x.data[1])
    assert np.allclose(phi(slots, d), x.to_dense())
    Aw = sampling_matrix(kind, shape, sample, order, weight)
    rw = _flat(O.weighted_residual(kind, O.S(kind, x.to_dense(), sample, order), y, weight, order))
    grad = Aw.T @ rw
    lam = LAM if regspec else 0.0
    res = {}
    if geom_name == 'manifold':
        params = make_params(d, sharing)
        J = jacobian(slots, params, d)
        Uj, sj, Vtj = np.linalg.svd(J, full_matrices=False)
        rank = int(np.sum(sj > 1e-10 * sj[0]))
        dim = t3m.manifold_dim((shape, x.tucker_ranks, x.tt_ranks), sharing=sharing)
        res['tangent_dim'] = 0.0 if rank == dim else 1.0          # the oracle's tangent space is the manifold's
        Q = Uj[:, :rank]
        Jpinv = Vtj[:rank].T @ np.diag(1.0 / sj[:rank]) @ Q.T
        u = grad - Q @ (Q.T @ grad)                             # the normal part of the Euclidean gradient

        def oracle_hess(v):
            xi = np.ravel(np.asarray(_ragged_tangent(v).to_dense()))     # (uniform d = 1 converts first)
            dot = unpack(Jpinv @ xi, slots, params)
            res.setdefault('lift', rel(J @ (Jpinv @ xi), xi))   # xi is a tangent: the lift is exact
            DJ = jacobian_derivative(slots, params, d, dot)
            weingarten = Jpinv.T @ (DJ.T @ u)
            return Q @ (Q.T @ (Aw.T @ (Aw @ xi))) + weingarten + lam * xi, weingarten, xi

        Hp_or, Wp, xp = oracle_hess(p)
        Hq_or, _, xq = oracle_hess(q)
        Hp_lib = np.ravel(np.asarray(_ragged_tangent(model.hessian(p)).to_dense()))
        Hq_lib = np.ravel(np.asarray(_ragged_tangent(model.hessian(q)).to_dense()))
        res['hvp'] = rel(Hp_lib, Hp_or)
        res['quadratic'] = abs(float(model.quadratic(p)) - float(xp @ Hp_or)) / (np.linalg.norm(xp) *
                                                                                 np.linalg.norm(Hp_or))
        res['symmetry'] = abs(float(xq @ Hp_lib) - float(xp @ Hq_lib)) / (np.linalg.norm(xp) * np.linalg.norm(Hq_lib))
        share = float(np.linalg.norm(Wp) / np.linalg.norm(Hp_or))
        if d > 1:
            res['curvature_share'] = 0.0 if share > CURV_MIN else 1.0     # pass/fail flag (0 = discriminating)
        else:
            res['flat_d1'] = share                                        # the flat manifold: no normal part
    else:
        params = [(s,) for s in range(2 * d)]                             # raw cores: every slot its own
        J = jacobian(slots, params, d)

        def oracle_hess(v):
            pv = mean_project(_coords(v), slots, d, sharing)
            DJ = jacobian_derivative(slots, params, d, unpack(pv, slots, params))
            curv = DJ.T @ grad
            Hc = J.T @ (Aw.T @ (Aw @ (J @ pv))) + curv + lam * pv
            return mean_project(Hc, slots, d, sharing), curv, pv

        Hp_or, Cp, pv = oracle_hess(p)
        Hq_or, _, qv = oracle_hess(q)
        Hp_lib, Hq_lib = _coords(model.hessian(p)), _coords(model.hessian(q))
        res['hvp'] = rel(Hp_lib, Hp_or)
        res['quadratic'] = abs(float(model.quadratic(p)) - float(pv @ Hp_or)) / (np.linalg.norm(pv) *
                                                                                 np.linalg.norm(Hp_or))
        res['symmetry'] = abs(float(qv @ Hp_lib) - float(pv @ Hq_lib)) / (np.linalg.norm(pv) * np.linalg.norm(Hq_lib))
        share = float(np.linalg.norm(mean_project(Cp, slots, d, sharing)) / np.linalg.norm(Hp_or))
        res['curvature_share'] = 0.0 if share > CURV_MIN else 1.0
    res['_share'] = share
    return res


def cases(orders=(1, 2)):
    """Every case key ``(d, rep, geometry, shared, kind, order, wspec, regularized)``: d in {3, 2} with every
    geometry x sharing, plus the flat d = 1 (unshared) -- every kind, the derivative kinds at ``orders``,
    every weight shape the kind takes, with and without the identity regularizer, ragged and uniform."""
    kinds = ['apply', 'entries', 'probe', 'apply_derivatives', 'entries_derivatives', 'probe_derivatives']
    for d in (3, 2, 1):
        geoms = ([('manifold', False), ('corewise', False), ('manifold', True), ('corewise', True)] if d > 1
                 else [('manifold', False), ('corewise', False)])
        for geom_name, shared in geoms:
            for kind in kinds:
                for order in (orders if kind.endswith('_derivatives') else [None]):
                    wspecs = ['none']
                    if kind == 'probe': wspecs.append('mode')
                    if kind.endswith('_derivatives'): wspecs.append('order')
                    if kind == 'probe_derivatives': wspecs.append('matrix')
                    for wspec in wspecs:
                        for regspec in (False, True):
                            for rep in ('ragged', 'uniform'):
                                yield (d, rep, geom_name, shared, kind, order, wspec, regspec)


def main(case_filter=None, seeds=(13,), Ws=((12,),), orders=(1, 2)):    # case_filter(key) -> bool
    """The always-on tier is the default call (one seed, ``W = (12,)``, orders 1-2); the slow tier
    (``T3TOOLBOX_SLOW_TESTS=1``) adds seeds, a multi-axis sample stack ``W = (3, 4)`` and order 3."""
    t0 = time.time()
    n = 0
    for key in cases(orders):
        if case_filter is not None and not case_filter(key):
            continue
        for seed in seeds:
            for W in Ws:
                n += 1
                try:
                    res = run_case(*key, seed=seed, W=W)
                except Exception as e:
                    failures.append((key + (seed, W), 'EXC', f'{type(e).__name__}: {str(e)[:200]}'))
                    continue
                rows.append((key + (seed, W), res))
                bad = {k: v for k, v in res.items() if not k.startswith('_') and not (v < TOL)}
                if bad:
                    failures.append((key + (seed, W), 'NUM', bad))
    print(f'{n} exact-Hessian cases in {time.time()-t0:.0f}s; {len(failures)} failures')
    for f in failures:
        print(' FAIL', f)


if __name__ == '__main__':
    main()
