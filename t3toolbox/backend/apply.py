# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The ``apply`` sampling type: contract a T3 with vectors in ALL modes -> one scalar per sample.

Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise forward derivative and the
corewise Hessian (the exact Hessian's sampling term), each but the ambient transpose one call of the engine,
and the dense reference :py:func:`dense_apply` (containment probe ⊃ apply ⊃ entries; this module imports from
``probing``, never the reverse). Costs and the role in Riemannian least-squares:
``docs/entries_apply_probe.md``.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.sampling_variables as sampling_variables
from t3toolbox.backend.common import *
import t3toolbox.backend.probing as probing

__all__ = [
    't3_apply',
    't3_apply_ambient_transpose',
    'tv_apply',
    'tv_apply_transpose',
    't3_apply_corewise_tangent',
    't3_apply_corewise_transpose',
    't3_apply_corewise_hessian',
    # Apply a dense tensor
    'dense_apply',
]


def t3_apply(
        x:                 typ.Tuple,                    # the point: X, (tucker_cores, tt_cores), elm_shapes
                                                         #   C+(nDi,Ni), C+(ri,nDi,r(i+1))
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # z: P, shape=F+W+C
    '''Apply a Tucker tensor train in every mode to the input jets: one number per sample and slot.

    One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``z`` on the point ``x``
    (a corewise call). Taylor coefficients on the jet shape ``order`` (``()`` for none).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.apply as apply
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> z = apply.t3_apply(x, ww)
    >>> print(z.shape)
    (3,)
    '''
    (z,) = sampling_variables.sampling_variables('apply', x, ww, ('z',), order=order)
    return z


def t3_apply_ambient_transpose(
        ww:                 typ.Sequence[NDArray],  # apply vectors: V_i, len=d, elm_shape=W+(Ni,)
        c:                  NDArray,                # ℝ, shape=W+C
        sum_over_samples:   bool = False,           # True: W becomes the CP rank (ambient J^T c)
) -> typ.Sequence[NDArray]:  # canonical (CP) factors: E*, len=d, ith elm_shape=stack_shape+(R, Ni)
    '''Ambient transpose of :py:func:`t3_apply`: back-project ``c`` into CP factors.

    The *ambient* adjoint -- the transpose of ``apply`` as a linear map on the **full tensor space**
    (``X -> ( <X, w0^W (x) ... (x) w_{d-1}^W> )_W``). Frame-free; the back-projection ``c * (w0 (x) ... (x) w_{d-1})``
    is rank-1, whose natural representation is a **canonical (CP) decomposition** (``apply`` consumes one vector per
    mode; its adjoint emits one scaled vector per mode). This is distinct from the *corewise* transpose (gradient w.r.t.
    a base point's cores) and the *tangent* transpose (Riemannian gradient) -- see ``docs/transposes.md`` for the full
    taxonomy. ``c`` is an element of ``Y`` (``ℝ`` per sample), paired with ``apply``'s output:
    ``<J^T c, T> = <c, apply(T, ww)>``, with no metric (for a weighted fit's, pass
    ``data_metric(r, weight, minibatch_scale)``).

    - ``sum_over_samples=False`` (primary): ``W`` is a passthrough stacking axis -- a ``W (+ C)`` stack
      of rank-1 CP tensors (CP rank ``R=1``).
    - ``sum_over_samples=True``: ``W`` becomes the CP **rank** -- one rank-``|W|`` CP tensor
      ``sum_W c_W * (w0^W (x) ...)`` (the ambient ``J^T c``). Cheap as CP (``O(d |W| N)``, the shared
      rank index stays implicit); the ``|W|^2`` cost of a *dense* Tucker tensor train is incurred only
      if you convert with ``t3_conversions.t3_from_canonical``.

    Returns the CP ``factors`` (``c`` folded into the first), in the layout
    ``t3_conversions.t3_from_canonical`` consumes.
    '''
    use_jax = tree_contains_jax((c, ww))
    xnp, _, _ = get_backend(False, use_jax)
    c = xnp.asarray(c)

    nW = ww[0].ndim - 1     # probe stack rank (ww[i] is W + (Ni,))
    W  = ww[0].shape[:nW]   # probe stack
    C  = c.shape[nW:]       # frame stack (c is W + C)
    nC = len(C)

    if sum_over_samples:
        # canonical rank |W|, stack C: w_i flattened over W into the rank axis, broadcast over C;
        # c folded into F_0 as  F_0[C, s, n] = c_flat[s, C] * w0_flat[s, n].
        m = int(np.prod(W, dtype=int))                       # |W|
        c_flat = xnp.moveaxis(c.reshape((m,) + C), 0, nC)    # (m,) + C  ->  C + (m,)
        factors = []
        for i, w in enumerate(ww):
            w_flat = w.reshape((1,) * nC + (m, w.shape[-1]))  # broadcastable to C + (m, Ni)
            if i == 0:
                factors.append(c_flat[..., None] * w_flat)               # C + (m, N0)
            else:
                factors.append(w_flat * xnp.ones(C + (1, 1)))            # materialize C + (m, Ni)
    else:
        # canonical rank 1, stack W + C: c folded into F_0 as  F_0[W, C, 0, n] = c[W, C] * w0[W, n].
        c_exp = c.reshape(W + C + (1, 1))                    # W + C + (1, 1)
        factors = []
        for i, w in enumerate(ww):
            w_exp = w.reshape(W + (1,) * nC + (1, w.shape[-1]))  # broadcastable to W + C + (1, Ni)
            if i == 0:
                factors.append(c_exp * w_exp)                            # W + C + (1, N0)
            else:
                factors.append(w_exp * xnp.ones(C + (1, 1)))             # materialize W + C + (1, Ni)

    return tuple(factors)


def tv_apply(
        frame:             typ.Tuple,                    # the base point: (D, U, L, R)
        variations:        typ.Tuple,                    # X, (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # Jp: P, shape=F+W+K+C
    '''The apply of a tangent vector: ``J p``, the derivative along the direction ``variations`` at ``frame``.

    One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``Jp``.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.apply as apply
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> Jp = apply.tv_apply(frame, variations, ww)
    >>> print(Jp.shape)
    (3,)
    '''
    (Jp,) = sampling_variables.sampling_variables('apply', frame, ww, ('Jp',), order=order, variations=variations)
    return Jp


def tv_apply_transpose(
        frame:             typ.Tuple,                    # the base point: (D, U, L, R)
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        c:                 NDArray,                      # P*, shape=F+W+[K]+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # V blocks: X*, len=d, elm_shape=[W]+[K]+C+(nUi,Ni)
        typ.Tuple[NDArray, ...],                         # H blocks: X*, len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))
]:
    '''The transpose of :py:func:`tv_apply`: ``J^T`` applied to ``c``, a tangent's blocks at ``frame``.

    ``c`` is an element of ``Y`` (``P*`` per sample; a stack ``K`` of them allowed), paired with ``z`` slot by slot. It
    carries no metric, and none is applied here: for the gradient of a weighted least-squares fit pass
    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weight and
    no scale. One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``JtJp`` with
    ``c`` held as ``Jp`` (the route of a held ``Jp``: ``J^T`` of the vector held; the engine turns ``c`` into its
    dual-jet form inside). The blocks are summed over the samples by default. For repeated calls at one frame (inside an
    iterative solver), call :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the
    frame's sweeps are then computed once.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.apply as apply
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> Jp = apply.tv_apply(frame, variations, ww)
    >>> V, H = apply.tv_apply_transpose(frame, ww, Jp)
    >>> pair = lambda a, b: sum(float(np.sum(u * v)) for u, v in zip(a[0] + a[1], tuple(b[0]) + tuple(b[1])))
    >>> print(bool(np.isclose(pair((V, H), variations), float(np.sum(Jp * Jp)))))       # <p, J^T J p> = |J p|^2
    True
    '''
    held = sampling_variables.HeldVariables(Jp=c)
    (blocks,) = sampling_variables.sampling_variables('apply', frame, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def t3_apply_corewise_tangent(
        x:                 typ.Tuple,                    # the point: X, (B, G), whose cores the directions vary
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        core_directions:   typ.Tuple,                    # X, (B_dot, G_dot), the shapes of B, G with K+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # J x_dot: P, shape=F+W+K+C
    '''The corewise forward derivative: ``J x_dot``, the derivative of ``z(x)`` (:py:func:`t3_apply`) along
    ``core_directions = (B_dot, G_dot)``, of the type of ``z`` with the stack of directions ``K`` after ``W``.

    The trio at a point: this is the tangent, :py:func:`t3_apply_corewise_transpose` its transpose and
    :py:func:`t3_apply_corewise_hessian` the Hessian. Per sample, with no sum over the samples (no
    ``sum_over_samples``). One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the
    corewise point ``x`` wanting ``Jp`` with ``core_directions`` (its raw route), each sweep once; the substitution
    ``tv_apply(t3_corewise_frame(x), core_directions, ...)`` computes the same at the cost of the frame's sweeps on
    ``(B, G, G, G)``. For repeated calls at one point, call
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the point's sweeps are then
    computed once.

    Examples
    --------
    ``z`` is homogeneous of degree ``2d`` in the cores, so the derivative along ``x`` itself is ``2d z`` (Euler):

    >>> import numpy as np
    >>> import t3toolbox.backend.apply as apply
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> z = apply.t3_apply(x, ww)
    >>> Jx = apply.t3_apply_corewise_tangent(x, ww, x)
    >>> print(Jx.shape)
    (3,)
    >>> print(np.allclose(Jx, 6 * z))                                     # Euler: J x = 2d z
    True
    '''
    (Jp,) = sampling_variables.sampling_variables('apply', x, ww, ('Jp',), order=order,
                                                   core_directions=core_directions)
    return Jp


def t3_apply_corewise_transpose(
        x:                 typ.Tuple,                    # the point: X, (tucker_cores, tt_cores)
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        c:                 NDArray,                      # P*, shape=F+W+[K]+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker-core blocks: X*, len=d, elm_shape=[W]+[K]+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT-core blocks:     X*, len=d,
                                                         #   elm_shape=[W]+[K]+C+(ri,nDi,r(i+1))
]:
    '''The corewise transpose: ``J^T`` applied to ``c`` in the cores of the point ``x``, the gradient of
    ``phi(x) = <c, z(x)>`` (the pairing of ``Y`` with ``Y*``, slot by slot: no metric).

    ``c`` is an element of ``Y`` (``P*`` per sample; a stack ``K`` of them allowed), paired with ``z`` slot by slot. It
    carries no metric, and none is applied here: for the gradient of a weighted least-squares fit pass
    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weight and
    no scale. One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point
    ``x`` wanting ``JtJp`` with ``c`` held as ``Jp``. Summed over the samples by default. For repeated calls at one
    frame (inside an iterative solver), call :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with
    ``held=``: the frame's sweeps are then computed once.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.apply as apply
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> c = apply.t3_apply(x, ww)
    >>> B, G = apply.t3_apply_corewise_transpose(x, ww, c)
    >>> print([b.shape for b in B], [g.shape for g in G])
    [(2, 4), (3, 5), (2, 6)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]
    '''
    held = sampling_variables.HeldVariables(Jp=c)
    (blocks,) = sampling_variables.sampling_variables('apply', x, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def t3_apply_corewise_hessian(
        x:                 typ.Tuple,                    # the point: X, (B, G), whose cores the directions vary
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        c:                 NDArray,                      # P*, shape=F+W+C
        core_directions:   typ.Tuple,                    # X, (B_dot, G_dot), the shapes of B, G with K+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker block: X*, len=d, elm_shape=[W]+K+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT block:     X*, len=d, elm_shape=[W]+K+C+(ri,nDi,r(i+1))
]:
    '''The corewise Hessian: the second derivative of ``phi(x) = <c, z(x)>`` in the cores, applied to
    ``core_directions`` (the derivative of :py:func:`t3_apply_corewise_transpose` along them).

    ``c`` is an element of ``Y`` (``P*`` per sample), paired with ``z`` slot by slot. It carries no metric, and none is
    applied here: for the curvature term of a weighted least-squares fit (``Kp`` at ``c = M r``), pass
    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weight and
    no scale. One call of
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point ``x`` wanting ``Kp`` with
    ``c`` held as ``r``. Summed over the samples (``sum_over_samples=False``: per sample, ``W`` leading the blocks'
    stack). For repeated calls at one frame (inside an iterative solver), call
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the frame's sweeps are then
    computed once.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.apply as apply
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> c = apply.t3_apply(x, ww)
    >>> KB, KG = apply.t3_apply_corewise_hessian(x, ww, c, x)    # the direction x itself
    >>> print([b.shape for b in KB], [g.shape for g in KG])
    [(2, 4), (3, 5), (2, 6)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]
    '''
    held = sampling_variables.HeldVariables(r=c)
    (blocks,) = sampling_variables.sampling_variables('apply', x, ww, ('Kp',), order=order, held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks


##########################################
########    Dense ground truth    ########
##########################################

def dense_apply(
        T:      NDArray,                                # E, shape=C+(N0,...,N(d-1))
        ww:     typ.Sequence[NDArray],                  # the input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        order:  typ.Tuple[int, ...] = (),               # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
) -> NDArray:                                           # z: P, shape=F+W+C
    """Apply a dense tensor: contract it with a vector in every mode, one number per sample.

    The dense ground truth of ``apply``: ``z`` is the tensor ``T`` contracted with ``ww[j]`` in every mode ``j``.

    **Jets.** With a jet shape ``order = (n_1, ..., n_m)`` the inputs are input jets in ``m`` variables, as for
    :py:func:`~t3toolbox.backend.probing.dense_probe`: ``ww[j]`` has ``m`` leading jet axes of sizes of its own,
    its slot at the multi-index ``k`` the Taylor coefficient of ``s_1^k_1 ... s_m^k_m`` of the input ``w_j(s)``.
    ``z`` is the polynomial ``T`` contracted with every ``w_j(s)``, by its Taylor coefficients (**not**
    derivatives) on the box ``F = (n_1 + 1, ..., n_m + 1)``, on ``m`` leading jet axes; input slots beyond the box
    are never read. Exact: sums of contractions, one mode at a time. The entries of a tensor along a curve that
    starts at a grid point are this, at the one-hot vectors of
    :py:func:`~t3toolbox.backend.jets.index_to_one_hot`.

    In the shape comments ``F`` is the full jet sizes, ``F_in_i`` the jet sizes of the input ``ww[i]``; the
    argument ``T`` is the tensor.

    Parameters
    ----------
    T: NDArray
        The tensor to be applied. shape=C+(N0,...,N(d-1)); ``d`` is the number of vectors.
    ww: typ.Sequence[NDArray]
        The input jets. len=d, elm_shape=F_in_i+W+(Ni,); with ``order = ()``, W+(Ni,).
    order: typ.Tuple[int, ...]
        The jet shape: the maximum order in each jet variable. ``()`` (the default) for no jets.

    Returns
    -------
    NDArray
        The applies ``z``. shape=F+W+C

    Raises
    ------
    TypeError
        If ``order`` is not a tuple of integers (an integer is not a jet shape).
    ValueError
        If an entry of ``order`` is negative, or an input does not have the jet axes of the jet shape (each of
        at least one slot), the sample stack of ``ww[0]`` or the length of its mode.

    See Also
    --------
    t3toolbox.backend.probing.dense_probe
    t3toolbox.backend.entries.dense_entries
    t3toolbox.backend.jets.affine_jet

    Examples
    --------

    Apply to a stack of vectors; value-match against a hand-written einsum, and against the probe contracted with
    the vector of its open mode:

    >>> import numpy as np
    >>> import t3toolbox.backend.apply as t3a
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> ww = [np.random.randn(2, 3, N) for N in (10, 11, 12)]      # W=(2,3)
    >>> z = t3a.dense_apply(T, ww)
    >>> print(z.shape)                           # one number per sample: W
    (2, 3)
    >>> print(bool(np.allclose(z, np.einsum('ijk,uvi,uvj,uvk->uv', T, *ww))))
    True
    >>> print(bool(np.allclose(z, np.einsum('uvk,uvk->uv', t3p.dense_probe(T, ww)[2], ww[2]))))
    True

    A stacked tensor: ``W`` outer, ``C`` inner, as for :py:func:`~t3toolbox.backend.probing.dense_probe`:

    >>> import numpy as np
    >>> import t3toolbox.backend.apply as t3a
    >>> np.random.seed(0)
    >>> T = np.random.randn(4, 5, 10, 11, 12)    # C=(4,5) stack on the tensor
    >>> ww = [np.random.randn(2, 3, N) for N in (10, 11, 12)]
    >>> z = t3a.dense_apply(T, ww)
    >>> print(z.shape)                           # W + C
    (2, 3, 4, 5)
    >>> print(bool(np.allclose(z, np.einsum('xyijk,uvi,uvj,uvk->uvxy', T, *ww))))
    True

    Jets in two variables: the inputs ``ww + s_1 pp + s_2 qq``, of jet sizes ``(2, 2)`` (``affine_jet`` with two
    directions), at the jet shape ``(1, 2)``. The coefficient of ``s_1 s_2`` is the sum over the ordered pairs of
    distinct modes, one taking ``pp`` and the other ``qq``; that of ``s_2^2`` the pairs that take ``qq``:

    >>> import numpy as np
    >>> import t3toolbox.backend.apply as t3a
    >>> from t3toolbox.backend.jets import affine_jet
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> ww = [np.random.randn(2, N) for N in (10, 11, 12)]
    >>> pp = [np.random.randn(2, N) for N in (10, 11, 12)]
    >>> qq = [np.random.randn(2, N) for N in (10, 11, 12)]
    >>> z = t3a.dense_apply(T, affine_jet(ww, pp, qq), order=(1, 2))
    >>> print(z.shape)                           # F=(2, 3) outermost, then W=(2,)
    (2, 3, 2)
    >>> print(bool(np.allclose(z[0, 0], t3a.dense_apply(T, ww))))                # slot (0, 0) is the plain apply
    True
    >>> with_ = lambda a, b, c: np.einsum('ijk,wi,wj,wk->w', T, a, b, c)
    >>> (w0, w1, w2), (p0, p1, p2), (q0, q1, q2) = ww, pp, qq
    >>> s1s2 = (with_(p0, q1, w2) + with_(q0, p1, w2) + with_(p0, w1, q2) + with_(q0, w1, p2)
    ...         + with_(w0, p1, q2) + with_(w0, q1, p2))
    >>> print(bool(np.allclose(z[1, 1], s1s2)))
    True
    >>> print(bool(np.allclose(z[0, 2], with_(q0, q1, w2) + with_(q0, w1, q2) + with_(w0, q1, q2))))
    True
    """
    return probing._dense_jet(T, ww, order, None, 'dense_apply')

