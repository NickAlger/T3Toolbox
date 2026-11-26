# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The ``probe`` sampling type: contract a T3 with vectors in all but one mode, giving ``d`` vectors (T4S §6).

The probe of a point and of a tangent, their transposes (ambient, corewise, tangent), the corewise forward
derivative and the corewise Hessian (the exact Hessian's sampling term), each but the ambient transpose one call
of the engine :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`; and the dense reference
:py:func:`dense_probe`, whose dense jets ``apply`` builds on (probe ⊃ apply ⊃ entries). Reference module
for the signature style (``docs/contributor/signature_style.md``).
"""
import itertools
import math
import typing as typ

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.sampling_variables as sampling_variables
from t3toolbox.backend.common import *

__all__ = [
    # Probe a Tucker tensor train
    't3_probe',
    # Probe a tangent vector, and its transpose
    'tv_probe',
    'tv_probe_transpose',
    # Ambient / corewise probe transposes, the corewise tangent and Hessian
    't3_probe_ambient_transpose',
    't3_probe_corewise_tangent',
    't3_probe_corewise_transpose',
    't3_probe_corewise_hessian',
    # Probe a dense tensor
    'dense_probe',
]

# NOTE: probing is intentionally UNWEIGHTED. In the typical regime (many probes at once) it is
# cheaper to absorb any edge weights into the cores once, up front, then probe the weighted cores
# with the plain functions below (rather than threading weights through every probe). The up-front
# weighting helper is ``t3_absorb_weights`` (see ``docs/weighting.md``).


#####################################################
########    Probing a Tucker Tensor Train    ########
#####################################################

def t3_probe(
        x:                 typ.Tuple,                    # the point: X, (tucker_cores, tt_cores), elm_shapes
                                                         #   C+(nDi,Ni), C+(ri,nDi,r(i+1))
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> typ.Tuple[NDArray, ...]:               # z: P⊗V_i*, len=d, elm_shape=F+W+C+(Ni,)
    '''Probe a Tucker tensor train: contract every mode but one with the input jets, ``d`` vectors.

    One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``z`` on the point ``x``
    (a corewise call). Taylor coefficients on the jet shape ``order`` (``()`` for none).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> z = probing.t3_probe(x, ww)
    >>> print([zi.shape for zi in z])
    [(3, 4), (3, 5), (3, 6)]
    '''
    (z,) = sampling_variables.sampling_variables('probe', x, ww, ('z',), order=order)
    return z


#####################################################
###########    Probing a tangent vector    ##########
#####################################################

def tv_probe(
        frame:             typ.Tuple,                    # the base point: (D, U, L, R)
        variations:        typ.Tuple,                    # X, (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> typ.Tuple[NDArray, ...]:               # Jp: P⊗V_i*, len=d, elm_shape=F+W+K+C+(Ni,)
    '''The probe of a tangent vector: ``J p``, the derivative along the direction ``variations`` at ``frame``.

    One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``Jp``.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> Jp = probing.tv_probe(frame, variations, ww)
    >>> print([a.shape for a in Jp])
    [(3, 4), (3, 5), (3, 6)]
    '''
    (Jp,) = sampling_variables.sampling_variables('probe', frame, ww, ('Jp',), order=order, variations=variations)
    return Jp


###############################################################
######    Transposes of the probe; the corewise tangent    ######
###############################################################

def tv_probe_transpose(
        frame:             typ.Tuple,                    # the base point: (D, U, L, R)
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        c:                 typ.Sequence[NDArray],        # P*⊗V_i, len=d, elm_shape=F+W+[K]+C+(Ni,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # V blocks: X*, len=d, elm_shape=[W]+[K]+C+(nUi,Ni)
        typ.Tuple[NDArray, ...],                         # H blocks: X*, len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))
]:
    '''The transpose of :py:func:`tv_probe`: ``J^T`` applied to ``c``, a tangent's blocks at ``frame``.

    ``c`` is an element of ``Y`` (``P*⊗V_i`` per mode per sample; a stack ``K`` of them allowed), paired with ``z`` slot
    by slot. It carries no metric, and none is applied here: for the gradient of a weighted least-squares fit pass
    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weight and
    no scale. One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``JtJp`` with
    ``c`` held as ``Jp`` (the route of a held ``Jp``: ``J^T`` of the vector held; the engine turns ``c`` into its
    dual-jet form inside). The blocks are summed over the samples by default. For repeated calls at one frame (inside an
    iterative solver), call :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the
    frame's sweeps are then computed once.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> Jp = probing.tv_probe(frame, variations, ww)
    >>> V, H = probing.tv_probe_transpose(frame, ww, Jp)
    >>> pair = lambda a, b: sum(float(np.sum(u * v)) for u, v in zip(a[0] + a[1], tuple(b[0]) + tuple(b[1])))
    >>> print(bool(np.isclose(pair((V, H), variations), sum(float(np.sum(a * a)) for a in Jp))))       # <p, J^T J p> = |J p|^2
    True
    '''
    held = sampling_variables.HeldVariables(Jp=c)
    (blocks,) = sampling_variables.sampling_variables('probe', frame, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def t3_probe_ambient_transpose(
        ww:                typ.Sequence[NDArray],  # probe vectors: V_i, len=d, elm_shape=W+(Ni,)
        c:                 typ.Sequence[NDArray],  # V_i, len=d, elm_shape=W+C+(Ni,)
        sum_over_samples:  bool = False,           # True: W folds into the CP rank
) -> typ.Sequence[NDArray]:  # canonical (CP) factors: E*, len=d, ith elm_shape=stack_shape+(R, Ni)
    '''Ambient transpose of :py:func:`t3_probe`: back-project ``c`` into CP factors.

    The *ambient* adjoint -- the transpose of ``probe`` as a linear map on the **full tensor space**. Probe returns
    ``d`` vectors (one free mode each), so ``c`` (an element of ``Y``: ``V_i`` per mode and sample, paired with
    ``probe``'s output, with no metric) is ``d`` vectors; the back-projection is the rank-``d`` tensor

        sum_i  w0 (x) ... (x) w_{i-1} (x) c_i (x) w_{i+1} (x) ... (x) w_{d-1}

    (term ``i`` has ``c_i`` in slot ``i`` and the probe vectors elsewhere), whose natural representation is a
    **canonical (CP) decomposition** of rank ``d``. Frame-free. Distinct from the *corewise* transpose (gradient w.r.t.
    a frame's cores) and the *tangent* transpose (Riemannian gradient); see ``docs/transposes.md``. The
    ``apply``/``entries`` analog is the rank-1 (or rank-``|W|``) :py:func:`t3_apply_ambient_transpose`.

    - ``sum_over_samples=False`` (primary): ``W`` is a passthrough stacking axis -- a ``W (+ C)`` stack
      of rank-``d`` CP tensors.
    - ``sum_over_samples=True``: ``W`` folds into the CP rank -- one rank-``d|W|`` CP tensor
      ``sum_W sum_i (...)``. Cheap as CP (``O(d |W| N)``).

    Returns CP ``factors`` (factor ``k`` has the diagonal structure: rank slot ``k`` = ``c_k``,
    the others = ``ww_k``), in the layout :py:func:`t3_conversions.t3_from_canonical` consumes.

    Examples
    --------
    The adjoint of ``probe`` on the full tensor space, ``<J^T c, T> = <c, dense_probe(T, ww)>``. **The argument order**
    (changed in this release: the sample, then ``c``): ``c`` and the sample are both ``d`` vectors, so the old order
    ``(c, ww)`` runs and returns the transpose of a different problem:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> np.random.seed(0)
    >>> T = np.random.randn(4, 5, 6)
    >>> ww, r = ([np.random.randn(3, n) for n in T.shape] for _ in range(2))
    >>> lhs = sum(float(np.sum(a * b)) for a, b in zip(r, probing.dense_probe(T, ww)))
    >>> pair = lambda f: float(np.einsum('rA,rB,rC,ABC->', *f, T))           # <CP(f), T>, the rank axis summed
    >>> right = probing.t3_probe_ambient_transpose(ww, r, sum_over_samples=True)
    >>> wrong = probing.t3_probe_ambient_transpose(r, ww, sum_over_samples=True)   # swapped: runs, and is wrong
    >>> print(abs(lhs - pair(right)) < 1e-12 * abs(lhs), abs(lhs - pair(wrong)) < 1e-12 * abs(lhs))
    True False
    '''
    use_jax = tree_contains_jax((c, ww))
    xnp, _, _ = get_backend(False, use_jax)
    d = len(ww)
    nW = ww[0].ndim - 1
    W  = ww[0].shape[:nW]                 # probe stack
    C  = c[0].shape[nW:-1]          # frame stack (c[i] is W + C + (Ni,))
    nC = len(C)
    mW = math.prod(W)

    factors = []
    for k in range(d):
        Nk = ww[k].shape[-1]
        w_bc = xnp.broadcast_to(ww[k].reshape(W + (1,) * nC + (1, Nk)), W + C + (d, Nk))
        diag = (xnp.arange(d) == k).reshape((d, 1))            # rank slot k is the "diagonal"
        Fk = xnp.where(diag, c[k][..., None, :], w_bc)   # W + C + (d, Nk)
        if sum_over_samples:                                    # fold W into the CP rank: rank d -> d|W|
            Fk = xnp.moveaxis(Fk, tuple(range(nW)), tuple(range(nC, nC + nW)))  # C + W + (d, Nk)
            Fk = Fk.reshape(C + (mW * d, Nk))
        factors.append(Fk)
    return tuple(factors)


def t3_probe_corewise_tangent(
        x:                 typ.Tuple,                    # the point: X, (B, G), whose cores the directions vary
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        core_directions:   typ.Tuple,                    # X, (B_dot, G_dot), the shapes of B, G with K+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> typ.Tuple[NDArray, ...]:               # J x_dot: P⊗V_i*, len=d, elm_shape=F+W+K+C+(Ni,)
    '''The corewise forward derivative: ``J x_dot``, the derivative of ``z(x)`` (:py:func:`t3_probe`) along
    ``core_directions = (B_dot, G_dot)``, of the type of ``z`` with the stack of directions ``K`` after ``W``.

    The trio at a point: this is the tangent, :py:func:`t3_probe_corewise_transpose` its transpose and
    :py:func:`t3_probe_corewise_hessian` the Hessian. Per sample, with no sum over the samples (no
    ``sum_over_samples``). One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the
    corewise point ``x`` wanting ``Jp`` with ``core_directions`` (its raw route), each sweep once; the substitution
    ``tv_probe(t3_corewise_frame(x), core_directions, ...)`` computes the same at the cost of the frame's sweeps on
    ``(B, G, G, G)``. For repeated calls at one point, call
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the point's sweeps are then
    computed once.

    Examples
    --------
    ``z`` is homogeneous of degree ``2d`` in the cores, so the derivative along ``x`` itself is ``2d z`` (Euler):

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> z = probing.t3_probe(x, ww)
    >>> Jx = probing.t3_probe_corewise_tangent(x, ww, x)
    >>> print([a.shape for a in Jx])                                         # W + (Ni,)
    [(3, 4), (3, 5), (3, 6)]
    >>> print(all(np.allclose(a, 6 * b) for a, b in zip(Jx, z)))        # Euler: J x = 2d z
    True
    '''
    (Jp,) = sampling_variables.sampling_variables('probe', x, ww, ('Jp',), order=order,
                                                   core_directions=core_directions)
    return Jp


def t3_probe_corewise_transpose(
        x:                 typ.Tuple,                    # the point: X, (tucker_cores, tt_cores)
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        c:                 typ.Sequence[NDArray],        # P*⊗V_i, len=d, elm_shape=F+W+[K]+C+(Ni,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker-core blocks: X*, len=d, elm_shape=[W]+[K]+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT-core blocks:     X*, len=d,
                                                         #   elm_shape=[W]+[K]+C+(ri,nDi,r(i+1))
]:
    '''The corewise transpose: ``J^T`` applied to ``c`` in the cores of the point ``x``, the gradient of
    ``phi(x) = <c, z(x)>`` (the pairing of ``Y`` with ``Y*``, slot by slot: no metric).

    ``c`` is an element of ``Y`` (``P*⊗V_i`` per mode per sample; a stack ``K`` of them allowed), paired with ``z`` slot
    by slot. It carries no metric, and none is applied here: for the gradient of a weighted least-squares fit pass
    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weight and
    no scale. One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point
    ``x`` wanting ``JtJp`` with ``c`` held as ``Jp``. Summed over the samples by default. For repeated calls at one
    frame (inside an iterative solver), call :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with
    ``held=``: the frame's sweeps are then computed once.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> c = probing.t3_probe(x, ww)
    >>> B, G = probing.t3_probe_corewise_transpose(x, ww, c)
    >>> print([b.shape for b in B], [g.shape for g in G])
    [(2, 4), (3, 5), (2, 6)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]

    **The argument order** (changed in this release: the point, the sample, then ``c``). ``c`` and the sample are both
    ``d`` vectors, so ``(x, c, ww)`` runs and returns the transpose of a different problem. Checked against the forward
    derivative at the intended sample:

    >>> x_dot = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> r = [np.random.randn(3, n) for n in (4, 5, 6)]
    >>> lhs = sum(float(np.sum(a * b)) for a, b in zip(r, probing.t3_probe_corewise_tangent(x, ww, x_dot)))
    >>> pair = lambda g: sum(float(np.sum(a * b)) for p, q in zip(g, x_dot) for a, b in zip(p, q))
    >>> right = probing.t3_probe_corewise_transpose(x, ww, r)
    >>> wrong = probing.t3_probe_corewise_transpose(x, r, ww)       # c and the sample swapped
    >>> print(abs(lhs - pair(right)) < 1e-12 * abs(lhs), abs(lhs - pair(wrong)) < 1e-12 * abs(lhs))
    True False
    '''
    held = sampling_variables.HeldVariables(Jp=c)
    (blocks,) = sampling_variables.sampling_variables('probe', x, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


###############################################################
######    Corewise two-hole: the probe Hessian           ######
###############################################################

def t3_probe_corewise_hessian(
        x:                 typ.Tuple,                    # the point: X, (B, G), whose cores the directions vary
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        c:                 typ.Sequence[NDArray],        # P*⊗V_i, len=d, elm_shape=F+W+C+(Ni,)
        core_directions:   typ.Tuple,                    # X, (B_dot, G_dot), the shapes of B, G with K+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker block: X*, len=d, elm_shape=[W]+K+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT block:     X*, len=d, elm_shape=[W]+K+C+(ri,nDi,r(i+1))
]:
    '''The corewise Hessian: the second derivative of ``phi(x) = <c, z(x)>`` in the cores, applied to
    ``core_directions`` (the derivative of :py:func:`t3_probe_corewise_transpose` along them).

    ``c`` is an element of ``Y`` (``P*⊗V_i`` per mode per sample), paired with ``z`` slot by slot. It carries no metric,
    and none is applied here: for the curvature term of a weighted least-squares fit (``Kp`` at ``c = M r``), pass
    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weight and
    no scale. One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point
    ``x`` wanting ``Kp`` with ``c`` held as ``r``. Summed over the samples (``sum_over_samples=False``: per sample,
    ``W`` leading the blocks' stack). For repeated calls at one frame (inside an iterative solver), call
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the frame's sweeps are then
    computed once.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> c = probing.t3_probe(x, ww)
    >>> KB, KG = probing.t3_probe_corewise_hessian(x, ww, c, x)    # the direction x itself
    >>> print([b.shape for b in KB], [g.shape for g in KG])
    [(2, 4), (3, 5), (2, 6)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]
    '''
    held = sampling_variables.HeldVariables(r=c)
    (blocks,) = sampling_variables.sampling_variables('probe', x, ww, ('Kp',), order=order, held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks


###############################################
##########    Probe dense tensor    ###########
###############################################

def _dense_contract_one_mode(
        A:          NDArray,    # shape=[W]+C+(the modes not yet contracted)
        has_W:      bool,       # does A lead with the sample stack W?
        vector:     NDArray,    # shape=W+(N_mode,): one coefficient of the input jet of the mode
        n_C:        int,        # the number of axes of C
        axis:       int,        # the place of the mode among the modes not yet contracted
) -> NDArray:                   # shape=W+C+(the modes not yet contracted, but this one)
    '''Contract one mode of a dense tensor (partly contracted already) with one vector: one einsum.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((A, vector)))
    n_W = vector.ndim - 1
    n_modes = A.ndim - n_C - (n_W if has_W else 0)
    # We are going to construct an einsum string from letters.
    # A dense 2x2x..x2 tensor exhausting these letters would have 4e15 entries
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    W_letters = letters[:n_W]
    C_letters = letters[n_W:n_W + n_C]
    mode_letters = letters[n_W + n_C:n_W + n_C + n_modes]
    kept = mode_letters[:axis] + mode_letters[axis + 1:]
    subscripts = ((W_letters if has_W else '') + C_letters + mode_letters + ',' + W_letters + mode_letters[axis]
                  + '->' + W_letters + C_letters + kept)
    return xnp.einsum(subscripts, A, vector)


def _dense_jet(
        T:          NDArray,                    # shape=C+(N0,...,N(d-1))
        ww:         typ.Sequence[NDArray],      # len=d, elm_shape=F_in_i+W+(Ni,): the input jets
        order:      typ.Tuple[int, ...],        # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        open_mode:  typ.Optional[int],          # the mode left open, or None: every mode is contracted
        name:       str,                        # the public function, for the error messages
) -> NDArray:                                   # shape=F+W+C+(N_open,), or F+W+C if open_mode is None
    '''The Taylor coefficients, on the box ``F = order + 1``, of ``T`` contracted with the input jets in every mode
    but ``open_mode``. The input jet of mode ``j`` is the polynomial ``w_j(s) = sum_k ww[j][k] s^k`` (``k`` a
    multi-index over its own jet sizes, zero beyond them), and the result is the product of the ``w_j(s)``
    through ``T``, truncated to the box. Computed one mode at a time: the partly contracted tensor is a polynomial
    whose coefficients are kept by multi-index, and contracting mode ``j`` multiplies it by ``w_j(s)``, by explicit
    loops over the pairs of multi-indices whose sum lies in the box, one einsum per pair. Exact: sums of
    contractions, with no step size and no interpolation. The one place the jets of a dense tensor are computed
    (the three public dense functions rest on it), and it shares nothing with the library's sweeps or with
    ``jet_contract``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((T, ww)))
    order = contractions._read_order(order, name)
    m = len(order)
    box = tuple(n + 1 for n in order)
    d = len(ww)
    if T.ndim < d:
        raise ValueError('%s: %d vectors for a tensor with %d axes' % (name, d, T.ndim))
    shape = T.shape[T.ndim - d:]
    n_C = T.ndim - d
    for ii, w in enumerate(ww):
        if w.ndim < m + 1:
            raise ValueError('%s: ww[%d] has %d axes, fewer than the %d jet axes of the jet shape %s and a mode axis'
                             % (name, ii, w.ndim, m, order))
    W = tuple(ww[0].shape[m:-1])
    for ii, w in enumerate(ww):
        if min(w.shape[:m], default=1) < 1:
            raise ValueError('%s: ww[%d] has the jet sizes %s: every jet axis has at least one slot'
                             % (name, ii, tuple(w.shape[:m])))
        if tuple(w.shape[m:-1]) != W:
            raise ValueError('%s: ww[%d] has sample stack %s once its %d jet axes are read, expected W = %s'
                             % (name, ii, tuple(w.shape[m:-1]), m, W))
        if w.shape[-1] != shape[ii]:
            raise ValueError('%s: ww[%d] has length %d, expected N_%d = %d' % (name, ii, w.shape[-1], ii, shape[ii]))

    coefficients = {(0,) * m: T}                # the tensor: a polynomial of degree 0, layout C + modes
    has_W = False                               # after the first contraction: W + C + the modes left
    left = list(range(d))
    for jj in range(d):
        if jj == open_mode:
            continue
        input_slots = list(itertools.product(*(range(min(size, f)) for size, f in zip(ww[jj].shape[:m], box))))
        product = {}
        for r, A in coefficients.items():
            for k in input_slots:               # slots of the input beyond the box are never read
                t = tuple(a + b for a, b in zip(r, k))
                if any(a > n for a, n in zip(t, order)):
                    continue
                term = _dense_contract_one_mode(A, has_W, ww[jj][k], n_C, left.index(jj))
                product[t] = term if t not in product else product[t] + term
        coefficients, has_W = product, True
        left.remove(jj)
    if not has_W:
        # d == 1 and its one mode is open: no vector is contracted, and the result is the tensor itself, broadcast
        # over W (the W letters would otherwise appear only in an einsum output, which numpy rejects)
        coefficients = {r: xnp.broadcast_to(A, W + tuple(A.shape)) for r, A in coefficients.items()}

    first = coefficients[(0,) * m]
    if m == 0:
        return first
    zero = xnp.zeros_like(first)                # a coefficient no product reaches: exactly zero
    slots = [coefficients.get(r, zero) for r in itertools.product(*(range(f) for f in box))]
    return xnp.reshape(xnp.stack(slots), box + tuple(first.shape))


def dense_probe(
        T:      NDArray,                                # E, shape=C+(N0,...,N(d-1))
        ww:     typ.Sequence[NDArray],                  # the input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        order:  typ.Tuple[int, ...] = (),               # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
) -> typ.Tuple[NDArray, ...]:                           # z: P⊗V_i*, len=d, elm_shape=F+W+C+(Ni,)
    """Probe a dense tensor: contract it with a vector in every mode but one, for each mode in turn.

    The dense ground truth of ``probe``: ``z[i]`` is the tensor ``T`` contracted with ``ww[j]`` in every mode
    ``j != i``.

    **Jets.** With a jet shape ``order = (n_1, ..., n_m)`` the inputs are input jets in ``m`` variables
    ``s_1 ... s_m``: ``ww[j]`` has ``m`` leading jet axes of sizes of its own, ``F_in_j``, and its slot at the
    multi-index ``k`` is the Taylor coefficient of ``s_1^k_1 ... s_m^k_m`` of the input vector ``w_j(s)`` (zero
    beyond its sizes). ``z[i]`` is then the polynomial ``T`` contracted with ``w_j(s)`` for ``j != i``, given by
    its Taylor coefficients (**not** derivatives) on the box ``F = (n_1 + 1, ..., n_m + 1)``, on ``m`` leading jet
    axes. Input slots beyond the box are never read. The coefficients are exact: sums of contractions, computed
    one mode at a time. A line ``w + s p`` is the input jet of two slots, :py:func:`~t3toolbox.backend.jets.affine_jet`.
    ``order`` is never read off the inputs: with the default ``order = ()`` an input jet's leading axes are read
    as sample axes.

    In the shape comments ``F`` is the full jet sizes, ``F_in_i`` the jet sizes of the input ``ww[i]``; the
    argument ``T`` is the tensor.

    Parameters
    ----------
    T: NDArray
        The tensor to be probed. shape=C+(N0,...,N(d-1)); ``d`` is the number of vectors.
    ww: typ.Sequence[NDArray]
        The input jets. len=d, elm_shape=F_in_i+W+(Ni,); with ``order = ()``, W+(Ni,).
    order: typ.Tuple[int, ...]
        The jet shape: the maximum order in each jet variable. ``()`` (the default) for no jets.

    Returns
    -------
    typ.Tuple[NDArray, ...]
        The probes ``z``. len=d, elm_shape=F+W+C+(Ni,)

    Raises
    ------
    TypeError
        If ``order`` is not a tuple of integers (an integer is not a jet shape).
    ValueError
        If an entry of ``order`` is negative, or an input does not have the jet axes of the jet shape (each of
        at least one slot), the sample stack of ``ww[0]`` or the length of its mode.

    See Also
    --------
    t3toolbox.backend.apply.dense_apply
    t3toolbox.backend.entries.dense_entries
    t3toolbox.backend.jets.affine_jet

    Examples
    --------

    Probe with one set of vectors; value-match each mode against a hand-written einsum:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> u0, u1, u2 = np.random.randn(10), np.random.randn(11), np.random.randn(12)
    >>> yy = t3p.dense_probe(T, (u0, u1, u2))
    >>> y0 = np.einsum('ijk,j,k', T, u1, u2)   # contract all modes but 0
    >>> y1 = np.einsum('ijk,i,k', T, u0, u2)
    >>> y2 = np.einsum('ijk,i,j', T, u0, u1)
    >>> print([y.shape for y in yy])           # one probe per mode, elm_shape=(Ni,)
    [(10,), (11,), (12,)]
    >>> print([bool(np.allclose(y, ref)) for y, ref in zip(yy, (y0, y1, y2))])
    [True, True, True]

    Vectorize over probing vectors: a probe stack ``W`` rides through, ``elm_shape = W + (Ni,)``:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> u0, u1, u2 = np.random.randn(2, 3, 10), np.random.randn(2, 3, 11), np.random.randn(2, 3, 12)
    >>> yy = t3p.dense_probe(T, (u0, u1, u2))
    >>> y0 = np.einsum('ijk,uvj,uvk->uvi', T, u1, u2)
    >>> y1 = np.einsum('ijk,uvi,uvk->uvj', T, u0, u2)
    >>> y2 = np.einsum('ijk,uvi,uvj->uvk', T, u0, u1)
    >>> print(yy[0].shape)                      # W=(2,3) outer, then N0=10
    (2, 3, 10)
    >>> print([bool(np.allclose(y, ref)) for y, ref in zip(yy, (y0, y1, y2))])
    [True, True, True]

    Vectorize over probing vectors AND a stacked (big) tensor: frame-inner ``elm_shape = W + C + (Ni,)``:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> T = np.random.randn(4, 5, 6, 10, 11, 12)   # C=(4,5,6) stack on the tensor
    >>> u0, u1, u2 = np.random.randn(2, 3, 10), np.random.randn(2, 3, 11), np.random.randn(2, 3, 12)
    >>> yy = t3p.dense_probe(T, (u0, u1, u2))
    >>> y0 = np.einsum('xyzijk,uvj,uvk->uvxyzi', T, u1, u2)
    >>> y1 = np.einsum('xyzijk,uvi,uvk->uvxyzj', T, u0, u2)
    >>> y2 = np.einsum('xyzijk,uvi,uvj->uvxyzk', T, u0, u1)
    >>> print(yy[0].shape)                      # W=(2,3) outer, C=(4,5,6) inner, then N0=10
    (2, 3, 4, 5, 6, 10)
    >>> print([bool(np.allclose(y, ref)) for y, ref in zip(yy, (y0, y1, y2))])
    [True, True, True]

    Jets along a line ``ww + s pp``: the input jet of two slots, made by ``affine_jet``. The probes are
    polynomials in ``s`` of degree ``d - 1 = 2``, so slot 1 is the sum over the one mode that takes ``pp``, and
    the three coefficients sum to the probes at ``s = 1``:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> from t3toolbox.backend.jets import affine_jet
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> ww = [np.random.randn(2, N) for N in (10, 11, 12)]      # W=(2,)
    >>> pp = [np.random.randn(2, N) for N in (10, 11, 12)]
    >>> zz = t3p.dense_probe(T, affine_jet(ww, pp), order=(2,))
    >>> print(zz[0].shape)                      # F=(3,) outermost, then W=(2,), then N0=10
    (3, 2, 10)
    >>> print(bool(np.allclose(zz[0][0], t3p.dense_probe(T, ww)[0])))             # slot 0 is the plain probe
    True
    >>> print(bool(np.allclose(zz[0][1], np.einsum('ijk,wj,wk->wi', T, pp[1], ww[2])
    ...                                  + np.einsum('ijk,wj,wk->wi', T, ww[1], pp[2]))))   # slot 1: one mode takes pp
    True
    >>> moved = [w + p for w, p in zip(ww, pp)]                                    # the inputs at s = 1
    >>> print(bool(np.allclose(zz[0].sum(axis=0), t3p.dense_probe(T, moved)[0])))  # degree d-1 = 2: nothing above
    True

    An input of three slots, a curve ``w + s p + s^2 q`` in one variable: slot 2 of a probe has the terms where
    one mode takes ``q`` and those where two modes take ``p``:

    >>> qq = [np.random.randn(2, N) for N in (10, 11, 12)]
    >>> curve = [np.stack([w, p, q]) for w, p, q in zip(ww, pp, qq)]               # F_in = (3,): slots 0, 1, 2
    >>> zz = t3p.dense_probe(T, curve, order=(2,))
    >>> slot2 = (np.einsum('ijk,wj,wk->wi', T, qq[1], ww[2]) + np.einsum('ijk,wj,wk->wi', T, ww[1], qq[2])
    ...          + np.einsum('ijk,wj,wk->wi', T, pp[1], pp[2]))
    >>> print(bool(np.allclose(zz[0][2], slot2)))
    True

    Gotcha -- ``order`` is a tuple, the jet shape; an integer is not accepted:

    >>> t3p.dense_probe(T, affine_jet(ww, pp), 2)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: dense_probe: order is the jet shape, a tuple of maximum orders, ...
    """
    return tuple(_dense_jet(T, ww, order, ii, 'dense_probe') for ii in range(len(ww)))
