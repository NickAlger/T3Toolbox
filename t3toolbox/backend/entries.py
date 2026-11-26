# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The ``entries`` sampling type: evaluate individual T3 entries (one-hot probes) -> scalars.

Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise forward derivative and the
corewise Hessian (the exact Hessian's sampling term), each but the ambient transpose one call of the engine,
and the dense reference :py:func:`dense_entries`. The most-special case of probe ⊃ apply ⊃ entries; imports
from ``apply`` (its ambient transpose), never the reverse. See ``docs/entries_apply_probe.md``.

Index semantics are numpy's throughout (forward and all transposes): a negative index wraps; an
out-of-range index raises under numpy but **clamps silently under jax** (gather semantics) --
validate indices yourself on the jax path.
"""
import typing as typ

import t3toolbox.backend.apply as apply
import t3toolbox.backend.sampling_variables as sampling_variables
from t3toolbox.backend.common import *

__all__ = [
    't3_entries',
    't3_entries_ambient_transpose',
    'tv_entries',
    'tv_entries_transpose',
    't3_entries_corewise_tangent',
    't3_entries_corewise_transpose',
    't3_entries_corewise_hessian',
    # Entries of a dense tensor
    'dense_entries',
]


def t3_entries(
        x:                 typ.Tuple,                    # the point: X, (tucker_cores, tt_cores), elm_shapes
                                                         #   C+(nDi,Ni), C+(ri,nDi,r(i+1))
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
) -> NDArray:                               # z: ℝ, shape=W+C
    '''The entries of a Tucker tensor train at the grid points ``index``.

    One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``z`` on the point ``x``
    (a corewise call). No jets (entry 120: along a curve, use the apply kind at ``index_to_one_hot``).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.entries as entries
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> index = np.array([[0, 3, 1], [4, 0, 2], [5, 1, 0]])                     # (d,) + W, W = (3,)
    >>> z = entries.t3_entries(x, index)
    >>> print(z.shape)
    (3,)
    '''
    (z,) = sampling_variables.sampling_variables('entries', x, index, ('z',))
    return z


def t3_entries_ambient_transpose(
        index:              NDArray,                # index^d, shape=(d,)+W, int
        shape:              typ.Sequence[int],      # ambient dims (N0,...,N(d-1)) -- to size the one-hots
        c:                  NDArray,                # ℝ, shape=W+C
        sum_over_samples:   bool = False,           # True: W becomes the CP rank (scatter-adds collisions)
) -> typ.Sequence[NDArray]:  # canonical (CP) factors: E*, len=d, ith elm_shape=stack_shape+(R, Ni)
    '''Ambient transpose of :py:func:`t3_entries`: scatter ``c`` at ``index`` into CP factors.

    The ``entries`` counterpart of :py:func:`t3_apply_ambient_transpose` -- identical with the apply vectors replaced by
    the unit vectors ``e_{index_k}``, so the CP factors are one-hots and the back-projection is
    ``c * e_{idx_0} (x) ... (x) e_{idx_{d-1}}`` (``c`` an element of ``Y``, ``ℝ`` per sample, as there).
    ``sum_over_samples=True`` makes ``W`` the CP rank (scatter-adding colliding indices -- the ``J^T c`` for entry
    sampling). ``shape`` supplies the ambient dims, which (unlike the apply case, where ``ww`` carries them) ``c`` and
    ``index`` alone do not determine. Returns CP factors (see the apply version).
    '''
    use_jax = tree_contains_jax((c, index))
    xnp, _, _ = get_backend(False, use_jax)
    index = xnp.array(index)
    # one-hot CP factors e_{index_i}, elm_shape = W + (Ni,). eye(N)[index] follows numpy index semantics
    # (a negative index wraps), exactly like the forward t3_entries and the tangent/corewise transposes; the
    # former `arange(N) == index` scatter matched nothing for a negative index and silently returned zero factors.
    ww = tuple(xnp.eye(N)[index[i]] for i, N in enumerate(shape))
    return apply.t3_apply_ambient_transpose(ww, c, sum_over_samples=sum_over_samples)


def tv_entries(
        frame:             typ.Tuple,                    # the base point: (D, U, L, R)
        variations:        typ.Tuple,                    # X, (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
) -> NDArray:                               # Jp: ℝ, shape=W+K+C
    '''The entries of a tangent vector: ``J p``, the derivative along the direction ``variations`` at ``frame``.

    One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``Jp``.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.entries as entries
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> index = np.array([[0, 3, 1], [4, 0, 2], [5, 1, 0]])                     # (d,) + W, W = (3,)
    >>> Jp = entries.tv_entries(frame, variations, index)
    >>> print(Jp.shape)
    (3,)
    '''
    (Jp,) = sampling_variables.sampling_variables('entries', frame, index, ('Jp',), variations=variations)
    return Jp


def tv_entries_transpose(
        frame:             typ.Tuple,                    # the base point: (D, U, L, R)
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
        c:                 NDArray,                      # ℝ, shape=W+[K]+C
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # V blocks: X*, len=d, elm_shape=[W]+[K]+C+(nUi,Ni)
        typ.Tuple[NDArray, ...],                         # H blocks: X*, len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))
]:
    '''The transpose of :py:func:`tv_entries`: ``J^T`` applied to ``c``, a tangent's blocks at ``frame``.

    ``c`` is an element of ``Y`` (``ℝ`` per sample; a stack ``K`` of them allowed), paired with ``z`` slot by slot. It
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
    >>> import t3toolbox.backend.entries as entries
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> index = np.array([[0, 3, 1], [4, 0, 2], [5, 1, 0]])                     # (d,) + W, W = (3,)
    >>> Jp = entries.tv_entries(frame, variations, index)
    >>> V, H = entries.tv_entries_transpose(frame, index, Jp)
    >>> pair = lambda a, b: sum(float(np.sum(u * v)) for u, v in zip(a[0] + a[1], tuple(b[0]) + tuple(b[1])))
    >>> print(bool(np.isclose(pair((V, H), variations), float(np.sum(Jp * Jp)))))       # <p, J^T J p> = |J p|^2
    True
    '''
    held = sampling_variables.HeldVariables(Jp=c)
    (blocks,) = sampling_variables.sampling_variables('entries', frame, index, ('JtJp',), held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def t3_entries_corewise_tangent(
        x:                 typ.Tuple,                    # the point: X, (B, G), whose cores the directions vary
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
        core_directions:   typ.Tuple,                    # X, (B_dot, G_dot), the shapes of B, G with K+C
) -> NDArray:                               # J x_dot: ℝ, shape=W+K+C
    '''The corewise forward derivative: ``J x_dot``, the derivative of ``z(x)`` (:py:func:`t3_entries`) along
    ``core_directions = (B_dot, G_dot)``, of the type of ``z`` with the stack of directions ``K`` after ``W``.

    The trio at a point: this is the tangent, :py:func:`t3_entries_corewise_transpose` its transpose and
    :py:func:`t3_entries_corewise_hessian` the Hessian. Per sample, with no sum over the samples (no
    ``sum_over_samples``). One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the
    corewise point ``x`` wanting ``Jp`` with ``core_directions`` (its raw route), each sweep once; the substitution
    ``tv_entries(t3_corewise_frame(x), core_directions, ...)`` computes the same at the cost of the frame's sweeps on
    ``(B, G, G, G)``. For repeated calls at one point, call
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the point's sweeps are then
    computed once.

    Examples
    --------
    ``z`` is homogeneous of degree ``2d`` in the cores, so the derivative along ``x`` itself is ``2d z`` (Euler):

    >>> import numpy as np
    >>> import t3toolbox.backend.entries as entries
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> index = np.array([[0, 3, 1], [4, 0, 2], [5, 1, 0]])                     # (d,) + W, W = (3,)
    >>> z = entries.t3_entries(x, index)
    >>> Jx = entries.t3_entries_corewise_tangent(x, index, x)
    >>> print(Jx.shape)
    (3,)
    >>> print(np.allclose(Jx, 6 * z))                                     # Euler: J x = 2d z
    True
    '''
    (Jp,) = sampling_variables.sampling_variables('entries', x, index, ('Jp',),
                                                   core_directions=core_directions)
    return Jp


def t3_entries_corewise_transpose(
        x:                 typ.Tuple,                    # the point: X, (tucker_cores, tt_cores)
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
        c:                 NDArray,                      # ℝ, shape=W+[K]+C
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker-core blocks: X*, len=d, elm_shape=[W]+[K]+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT-core blocks:     X*, len=d,
                                                         #   elm_shape=[W]+[K]+C+(ri,nDi,r(i+1))
]:
    '''The corewise transpose: ``J^T`` applied to ``c`` in the cores of the point ``x``, the gradient of
    ``phi(x) = <c, z(x)>`` (the pairing of ``Y`` with ``Y*``, slot by slot: no metric).

    ``c`` is an element of ``Y`` (``ℝ`` per sample; a stack ``K`` of them allowed), paired with ``z`` slot by slot. It
    carries no metric, and none is applied here: for the gradient of a weighted least-squares fit pass
    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weight and
    no scale. One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point
    ``x`` wanting ``JtJp`` with ``c`` held as ``Jp``. Summed over the samples by default. For repeated calls at one
    frame (inside an iterative solver), call :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with
    ``held=``: the frame's sweeps are then computed once.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.entries as entries
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> index = np.array([[0, 3, 1], [4, 0, 2], [5, 1, 0]])                     # (d,) + W, W = (3,)
    >>> c = entries.t3_entries(x, index)
    >>> B, G = entries.t3_entries_corewise_transpose(x, index, c)
    >>> print([b.shape for b in B], [g.shape for g in G])
    [(2, 4), (3, 5), (2, 6)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]
    '''
    held = sampling_variables.HeldVariables(Jp=c)
    (blocks,) = sampling_variables.sampling_variables('entries', x, index, ('JtJp',), held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def t3_entries_corewise_hessian(
        x:                 typ.Tuple,                    # the point: X, (B, G), whose cores the directions vary
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
        c:                 NDArray,                      # ℝ, shape=W+C
        core_directions:   typ.Tuple,                    # X, (B_dot, G_dot), the shapes of B, G with K+C
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker block: X*, len=d, elm_shape=[W]+K+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT block:     X*, len=d, elm_shape=[W]+K+C+(ri,nDi,r(i+1))
]:
    '''The corewise Hessian: the second derivative of ``phi(x) = <c, z(x)>`` in the cores, applied to
    ``core_directions`` (the derivative of :py:func:`t3_entries_corewise_transpose` along them).

    ``c`` is an element of ``Y`` (``ℝ`` per sample), paired with ``z`` slot by slot. It carries no metric, and none is
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
    >>> import t3toolbox.backend.entries as entries
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> index = np.array([[0, 3, 1], [4, 0, 2], [5, 1, 0]])                     # (d,) + W, W = (3,)
    >>> c = entries.t3_entries(x, index)
    >>> KB, KG = entries.t3_entries_corewise_hessian(x, index, c, x)    # the direction x itself
    >>> print([b.shape for b in KB], [g.shape for g in KG])
    [(2, 4), (3, 5), (2, 6)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]
    '''
    held = sampling_variables.HeldVariables(r=c)
    (blocks,) = sampling_variables.sampling_variables('entries', x, index, ('Kp',), held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks


##########################################
########    Dense ground truth    ########
##########################################

def dense_entries(
        T:      NDArray,                # E, shape=C+(N0,...,N(d-1))
        index:  NDArray,                # index^d, shape=(d,)+W, int
) -> NDArray:                           # z: ℝ, shape=W+C
    """Entries of a dense tensor: ``T[index[0], ..., index[d-1]]``, one number per sample.

    The dense ground truth of ``entries``. The tensor is indexed, with no arithmetic: an entry comes back as it is
    stored, whatever the other entries hold. The entries kind has no jets: an index is a point of the grid, and a
    tensor on a grid has no derivative until it is extended off it. Along a curve of input vectors that starts at
    the grid point (one such extension) the jets are those of :py:func:`~t3toolbox.backend.apply.dense_apply` at the
    one-hot vectors :py:func:`~t3toolbox.backend.jets.index_to_one_hot` (example below).

    The argument ``T`` is the tensor. Index semantics are numpy's: a negative index wraps; an out-of-range index
    raises under numpy and **clamps silently under jax**.

    Parameters
    ----------
    T: NDArray
        The tensor. shape=C+(N0,...,N(d-1)); ``d`` is the length of ``index``.
    index: NDArray
        The grid points, one row per mode. dtype=int, shape=(d,)+W

    Returns
    -------
    NDArray
        The entries ``z``. shape=W+C

    Raises
    ------
    ValueError
        If the index has more rows than the tensor has axes.

    See Also
    --------
    t3toolbox.backend.probing.dense_probe
    t3toolbox.backend.apply.dense_apply
    t3toolbox.backend.jets.index_to_one_hot

    Examples
    --------

    Entries at a stack of grid points; value-match against indexing:

    >>> import numpy as np
    >>> import t3toolbox.backend.entries as t3e
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> index = np.array([[3, 0, 9, 9], [1, 10, 4, 4], [0, 11, 2, 5]])     # (d,)+W, W=(4,)
    >>> z = t3e.dense_entries(T, index)
    >>> print(z.shape)                           # one number per sample: W
    (4,)
    >>> print(bool(np.allclose(z, T[index[0], index[1], index[2]])))
    True

    A stacked tensor: ``W`` outer, ``C`` inner:

    >>> import numpy as np
    >>> import t3toolbox.backend.entries as t3e
    >>> np.random.seed(0)
    >>> T = np.random.randn(2, 10, 11, 12)       # C=(2,) stack on the tensor
    >>> index = np.array([[3, 0, 9, 9], [1, 10, 4, 4], [0, 11, 2, 5]])
    >>> z = t3e.dense_entries(T, index)
    >>> print(z.shape)                           # W + C
    (4, 2)
    >>> print(bool(np.allclose(z, T[:, index[0], index[1], index[2]].T)))
    True

    Off the grid, along ``e_index + s pp``: the apply at the one-hot vectors moved in the direction ``pp``. Slot 0
    is the entry; slot 1 the sum over the modes of ``pp`` contracted into that mode, at the grid point in the
    others:

    >>> import numpy as np
    >>> import t3toolbox.backend.apply as t3a
    >>> import t3toolbox.backend.entries as t3e
    >>> from t3toolbox.backend.jets import affine_jet, index_to_one_hot
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> index = np.array([[3, 0, 9, 9], [1, 10, 4, 4], [0, 11, 2, 5]])
    >>> pp = [np.random.randn(4, N) for N in (10, 11, 12)]
    >>> z = t3a.dense_apply(T, affine_jet(index_to_one_hot(index, T.shape), pp), order=(2,))
    >>> print(z.shape)                           # F=(3,) outermost, then W=(4,)
    (3, 4)
    >>> print(bool(np.allclose(z[0], t3e.dense_entries(T, index))))
    True
    >>> slope = (np.einsum('wi,iw->w', pp[0], T[:, index[1], index[2]])
    ...          + np.einsum('wj,wj->w', pp[1], T[index[0], :, index[2]])
    ...          + np.einsum('wk,wk->w', pp[2], T[index[0], index[1], :]))
    >>> print(bool(np.allclose(z[1], slope)))
    True
    """
    xnp, _, _ = get_backend(True, tree_contains_jax((T, index)))
    T = xnp.asarray(T)
    index = xnp.asarray(index)
    d = index.shape[0]
    if T.ndim < d:
        raise ValueError('dense_entries: an index with %d rows for a tensor with %d axes' % (d, T.ndim))
    n_C, n_W = T.ndim - d, index.ndim - 1
    z_CW = T[(Ellipsis,) + tuple(index[ii] for ii in range(d))]                 # indexing: no arithmetic
    return xnp.moveaxis(z_CW, tuple(range(n_C, n_C + n_W)), tuple(range(n_W)))  # C + W -> W + C

