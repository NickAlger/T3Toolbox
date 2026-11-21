# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The ``entries`` sampling type: evaluate individual T3 entries (one-hot probes) -> scalars.

Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise transpose's derivative in the
cores (the exact Hessian's sampling term), and the frame sweeps for entry sampling. The most-special case of
probe ⊃ apply ⊃ entries; imports the general machinery from ``probing``, never the reverse. See
``docs/entries_apply_probe.md``.

Index semantics are numpy's throughout (forward and all transposes): a negative index wraps; an
out-of-range index raises under numpy but **clamps silently under jax** (gather semantics) --
validate indices yourself on the jax path.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.apply as apply
import t3toolbox.backend.sampling_variables as sampling_variables
from t3toolbox.backend.common import *
import math
import t3toolbox.backend.probing as probing
from t3toolbox.backend.probing import compute_mu
from t3toolbox.backend.apply import _apply_from_xis, _apply_transpose_adjoint, _apply_corewise_hessian_adjoint

__all__ = [
    't3_entries',
    't3_entries_ambient_transpose',
    'tv_entries',
    'tv_entries_transpose',
    'tv_precompute_entries_frame_sweep',
    'tv_entries_jacobian_from_sweep',
    'tv_entries_transpose_from_sweep',
    't3_entries_corewise_transpose',
    't3_entries_corewise_hessian',
    't3_entries_corewise_hessian_from_sweep',
    # Entries of a dense tensor
    'dense_entries',
]


def _t3_entries_step(
        mu_WCa:  NDArray,                                # carry: W+C+(rLi,)
        ind_B_G: typ.Tuple[NDArray, NDArray, NDArray],   # (ind, B, G): int W ; C+(ni,Ni) ; C+(rLi,ni,rL(i+1))
) -> typ.Tuple[NDArray, typ.Tuple[int]]:   # (next carry, (0,) -- nothing emitted per mode)
    '''One mode of the all-modes fiber contraction of :py:func:`t3_entries`. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((mu_WCa, ind_B_G)))   # only xnp; it ignores the flag
    ind, B_Cpo, G_Capb = ind_B_G
    n_idx = ind.ndim                                           # this mode's index carries exactly the W axes
    xi_CpW = B_Cpo[..., ind]                                   # C + (p,) + W (index batch trails)
    xi_WCp = xnp.moveaxis(                                     # -> W + C + (p,) = WCi
        xi_CpW, tuple(range(-n_idx, 0)), tuple(range(n_idx)),
    )

    mu_WCb = contractions.contract('WCa,Caib,WCi->WCb', 
        mu_WCa, G_Capb, xi_WCp,
    )

    return mu_WCb, (0,)


def t3_entries(
        x:                 typ.Tuple,                    # (tucker_cores, tt_cores), elm_shapes C+(nDi,Ni), C+(ri,nDi,r(i+1))
        index:             NDArray,                      # int, shape=(d,)+W: the grid points
) -> NDArray:                               # z. shape=W+C
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
        index:              NDArray,                # int, shape=(d,)+W
        shape:              typ.Sequence[int],      # ambient dims (N0,...,N(d-1)) -- to size the one-hots
        residual:           NDArray,                # shape=W+C
        sum_over_samples:   bool = False,           # True: W becomes the CP rank (scatter-adds collisions)
) -> typ.Sequence[NDArray]:  # canonical (CP) factors. len=d, ith elm_shape=stack_shape+(R, Ni)
    '''Ambient transpose of :py:func:`t3_entries`: scatter ``residual`` at ``index`` into CP factors.

    The ``entries`` counterpart of :py:func:`t3_apply_ambient_transpose` -- identical
    with the apply vectors replaced by the unit vectors ``e_{index_k}``, so the CP factors are one-hots
    and the back-projection is ``residual * e_{idx_0} (x) ... (x) e_{idx_{d-1}}``. ``sum_over_samples=True``
    makes ``W`` the CP rank (scatter-adding colliding indices -- the ``J^T r`` for entry sampling).
    ``shape`` supplies the ambient dims, which (unlike the apply case, where ``ww`` carries them) the
    residual and index alone do not determine. Returns CP factors (see the apply version).
    '''
    use_jax = tree_contains_jax((residual, index))
    xnp, _, _ = get_backend(False, use_jax)
    index = xnp.array(index)
    # one-hot CP factors e_{index_i}, elm_shape = W + (Ni,). eye(N)[index] follows numpy index semantics
    # (a negative index wraps), exactly like the forward t3_entries and the tangent/corewise transposes
    # (_onehot_vectors); the former `arange(N) == index` scatter matched nothing for a negative index and
    # silently returned zero factors.
    ww = tuple(xnp.eye(N)[index[i]] for i, N in enumerate(shape))
    return apply.t3_apply_ambient_transpose(ww, residual, sum_over_samples=sum_over_samples)


def tv_entries(
        frame:             typ.Tuple,                    # (D, U, L, R)
        variations:        typ.Tuple,                    # (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        index:             NDArray,                      # int, shape=(d,)+W: the grid points
) -> NDArray:                               # Jp. shape=W+K+C
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
        frame:             typ.Tuple,                    # (D, U, L, R)
        index:             NDArray,                      # int, shape=(d,)+W: the grid points
        residual:          NDArray,                       # of the type of z: shape=W+[K]+C
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # V blocks. len=d, elm_shape=[W]+[K]+C+(nUi,Ni)
        typ.Tuple[NDArray, ...],                         # H blocks. len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))
]:
    '''The transpose of :py:func:`tv_entries`: ``J^T`` applied to ``residual``, a tangent's blocks at ``frame``.

    ``residual`` is a vector of the type of ``z`` (a stack ``K`` of them allowed), with no weight; it is turned inside
    (a dual jet). One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``JtJp`` with
    ``residual`` held as ``Jp`` (the route of a held ``Jp``: ``J^T`` of the vector held). The blocks are summed over
    the samples by default. For repeated calls at one frame (inside an iterative solver), call
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
    >>> Jp = entries.tv_entries(frame, variations, index)
    >>> V, H = entries.tv_entries_transpose(frame, index, Jp)
    >>> pair = lambda a, b: sum(float(np.sum(u * v)) for u, v in zip(a[0] + a[1], tuple(b[0]) + tuple(b[1])))
    >>> print(bool(np.isclose(pair((V, H), variations), float(np.sum(Jp * Jp)))))       # <p, J^T J p> = |J p|^2
    True
    '''
    held = sampling_variables.HeldVariables(Jp=residual)
    (blocks,) = sampling_variables.sampling_variables('entries', frame, index, ('JtJp',), held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def tv_precompute_entries_frame_sweep(
        frame:  typ.Tuple[
            typ.Sequence[NDArray],          # down_tucker_cores  D. len=d
            typ.Sequence[NDArray],          # up_tt_cores        U. len=d
            typ.Sequence[NDArray],          # left_tt_cores      L. len=d
            typ.Sequence[NDArray],          # right_tt_cores     R. len=d
        ],                                  # frame order = T3Frame.data = (down, up, left, right)
        index:  NDArray,                    # int, shape=(d,)+W -- the grid points
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xis. len=d, elm_shape=W+C+(nDi,) -- the FIBER-SLICED seed (not contracted)
    typ.Sequence[NDArray],  # mus. len=d, elm_shape=W+C+(rLi,)
]:                                          # lean frame sweep -- (xis, mus) only (entries seed)
    '''The all-modes **entries** frame sweep (lean): identical to :py:func:`tv_precompute_apply_frame_sweep`
    but the ``xi-hat`` seed comes from slicing the Tucker-core fibers at ``index`` (``_entry_xis``)
    instead of contracting with probe vectors. Like apply, entries uses the adjoint-state transpose, so
    only ``(xis, mus)`` are needed (no ``nu``/``eta``). Reused by the entries forward/transpose (the
    reuse hook for ``fitting.py``).

    See Also
    --------
    tv_precompute_apply_frame_sweep
    tv_entries_jacobian_from_sweep
    tv_entries_transpose_from_sweep
    '''
    down_tucker_cores, _, left_tt_cores, _ = frame
    xis = _entry_xis(down_tucker_cores, index)
    mus = compute_mu(left_tt_cores, xis)
    return xis, mus


def tv_entries_jacobian_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nUi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nDi,rR(i+1))
        ],
        index:      NDArray,                # int, shape=(d,)+W -- for the variation's fiber-sliced dxis
        frame:      typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (D, U, L, R); uses R (right) and U (up)
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = tv_precompute_entries_frame_sweep(frame, index)  (lean)
) -> NDArray:                               # entries of the dense tangent at ``index``; shape = W + K + C
    '''Forward all-modes entries of a tangent vector reusing a precomputed frame sweep -- the bare ``𝒥``
    (entries) with the frame edge variables injected. Equivalent to :py:func:`tv_entries`, but takes
    the lean ``(xis, mus)`` from ``frame_sweep``; only the fiber-sliced ``dxis`` is computed here. No
    gauge projector ``Π``.'''
    var_tucker_cores, var_tt_cores = variation
    _, up_tt_cores, _, right_tt_cores = frame
    xis, mus = frame_sweep
    dxis = _entry_xis(var_tucker_cores, index)    # fiber slice; variation-dependent, not in the frame sweep
    return _apply_from_xis(xis, dxis, mus, right_tt_cores, up_tt_cores, var_tt_cores)


def tv_entries_transpose_from_sweep(
        c:          NDArray,                # residual, shape = W + C (or W + K + C)
        index:      NDArray,                # int, shape=(d,)+W -- the indices c weights (-> one-hot vectors)
        frame:      typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (D, U, L, R); uses D (one-hot), U, R
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = tv_precompute_entries_frame_sweep(frame, index)  (lean: no nu/eta)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (V_tildes, H_tildes) = T3Variations.data
    '''Transpose of the all-modes entries reusing a precomputed frame sweep -- the bare ``𝒥ᵀ`` (entries),
    by the **adjoint-state** method (see :py:func:`_apply_transpose_adjoint`). Takes the **lean**
    ``(xis, mus)`` sweep (the reuse hook for ``fitting.py``). Identical to
    :py:func:`tv_apply_transpose_from_sweep` with the one-hot vectors ``e_{index}`` as the apply vectors.
    Full ``W + K + C``; no gauge projector ``Π``.'''
    down_tucker_cores, up_tt_cores, _, right_tt_cores = frame
    ww = _onehot_vectors(index, down_tucker_cores)
    xis, mus = frame_sweep
    return _apply_transpose_adjoint(c, ww, xis, mus, up_tt_cores, right_tt_cores, sum_over_probes)


def t3_entries_corewise_transpose(
        x:                 typ.Tuple,                    # (tucker_cores, tt_cores): the point
        index:             NDArray,                      # int, shape=(d,)+W: the grid points
        residual:          NDArray,                       # of the type of z: shape=W+[K]+C
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker-core blocks. len=d, elm_shape=[W]+[K]+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT-core blocks.     len=d, elm_shape=[W]+[K]+C+(ri,nDi,r(i+1))
]:
    '''The corewise transpose: ``J^T`` applied to ``residual`` in the cores of the point ``x``, the gradient of
    ``phi(x) = <residual, z(x)>`` (plain inner product).

    ``residual`` is a vector of the type of ``z`` (a stack ``K`` allowed), with no weight, turned inside. One call of
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point ``x`` wanting ``JtJp``
    with ``residual`` held as ``Jp``. Summed over the samples by default. For repeated calls at one frame (inside an iterative solver), call
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
    >>> B, G = entries.t3_entries_corewise_transpose(x, index, c)
    >>> print([b.shape for b in B], [g.shape for g in G])
    [(2, 4), (3, 5), (2, 6)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]
    '''
    held = sampling_variables.HeldVariables(Jp=residual)
    (blocks,) = sampling_variables.sampling_variables('entries', x, index, ('JtJp',), held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def t3_entries_corewise_hessian_from_sweep(
        c:                  NDArray,                # residual (omega^2 applied), shape=W+C
        index:              NDArray,                # int, shape=(d,)+W
        core_pair:          typ.Tuple[
            typ.Sequence[NDArray],                  # tucker_cores of theta. len=d, elm_shape=C+(ni,Ni)
            typ.Sequence[NDArray],                  # tt_cores of theta.     len=d, elm_shape=C+(ri,ni,r(i+1))
        ],                                          # = TuckerTensorTrain.data
        core_directions:    typ.Tuple[
            typ.Sequence[NDArray],                  # Tucker directions theta_hat. len=d, elm_shape=K+C+(ni,Ni)
            typ.Sequence[NDArray],                  # TT directions theta_hat.     len=d, elm_shape=K+C+(ri,ni,r(i+1))
        ],

        frame_sweep:        typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],   # (xis, mus) of theta
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                        # Tucker block of K theta_hat. len=d, elm_shape=K+C+(ni,Ni)
    typ.Tuple[NDArray, ...],                        # TT block of K theta_hat.     len=d, elm_shape=K+C+(ri,ni,r(i+1))
]:
    ''':py:func:`t3_entries_corewise_hessian` reusing a precomputed lean ``frame_sweep``
    (:py:func:`tv_precompute_entries_frame_sweep`, the same at the manifold frame and at the corewise frame of its
    base point ``theta = (D, L)``) -- the reuse hook for a local model; it must be ``core_pair``'s own.'''
    tucker_cores, tt_cores = core_pair
    tucker_dirs, tt_dirs = core_directions
    xis, mus = frame_sweep
    xis_dot = _entry_xis(tucker_dirs, index)                                     # xi_dot = Bdot[..., index] (fiber slice)
    mus_dot = probing.compute_sigma(tt_dirs, tt_cores, tt_cores, xis, xis_dot, mus)
    ww = _onehot_vectors(index, tucker_cores)                                    # the one-hot "apply vectors"
    return _apply_corewise_hessian_adjoint(c, ww, xis, xis_dot, mus, mus_dot, tt_cores, tt_dirs)


def t3_entries_corewise_hessian(
        x:                 typ.Tuple,                    # (B, G): the point whose cores the directions vary
        index:             NDArray,                      # int, shape=(d,)+W: the grid points
        residual:          NDArray,                       # c, of the type of z: shape=W+C; no weight
        core_directions:   typ.Tuple,                    # (B_dot, G_dot), the shapes of B, G with K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker block. len=d, elm_shape=K+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT block.     len=d, elm_shape=K+C+(ri,nDi,r(i+1))
]:
    '''The corewise Hessian: the second derivative of ``phi(x) = <residual, z(x)>`` in the cores, applied to
    ``core_directions`` (the derivative of :py:func:`t3_entries_corewise_transpose` along them).

    ``residual`` is a vector of the type of ``z``, no weight, turned inside. One call of
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point ``x`` wanting ``Kp``
    with ``residual`` held as ``r``. Summed over the samples. For repeated calls at one frame (inside an iterative solver), call
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
    held = sampling_variables.HeldVariables(r=residual)
    (blocks,) = sampling_variables.sampling_variables('entries', x, index, ('Kp',), held=held,
                                                       core_directions=core_directions)
    return blocks


##########################################
########    Dense ground truth    ########
##########################################

def dense_entries(
        T:      NDArray,                # shape=C+(N0,...,N(d-1))
        index:  NDArray,                # int, shape=(d,)+W
) -> NDArray:                           # z. shape=W+C
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


def _entry_xis(tucker_cores, index):
    '''Upward edge variables by slicing Tucker-core fibers at ``index`` (no contraction).

    tucker_cores[i].shape = C + (p_i, Ni); index is an int array of shape (d,) + W (index stack W).
    Returns xis with elm_shape = W + C + (p_i,) -- the same layout compute_xi produces, so the
    downstream mu/sigma sweeps are identical to apply.
    '''
    use_jax = tree_contains_jax((tucker_cores,))
    is_uniform = is_ndarray(tucker_cores)
    xnp, _, _ = get_backend(is_uniform, use_jax)
    index = xnp.array(index)

    if is_uniform:
        # Vectorized fiber slice over the core index d: a single advanced-indexing gather, NOT a per-core
        # Python loop (which would unroll under jit) and NOT a one-hot contraction (which would add the N
        # factor entries exists to avoid). The supercore is (d,)+C+(p,N); advanced indices on axis 0 (d)
        # and axis -1 (N) broadcast to (d,)+W, the C/p slice rides between -> (d,)+W+C+(p,), matching the
        # ragged W+C+(p,) layout. index_i < Ni <= N stays in the real region.
        n_cores = tucker_cores.shape[0]
        n_W = index.ndim - 1
        d_idx = xnp.arange(n_cores).reshape((n_cores,) + (1,) * n_W)          # (d,) + (1,)*W, broadcasts with index
        return tucker_cores[d_idx, ..., index]                   # (d,) + W + C + (p,)

    n_idx = len(index.shape[1:])                                  # number of index-stack (W) axes
    xis = []
    for i, B in enumerate(tucker_cores):
        xi_CpW = B[..., index[i]]                                 # C + (p,) + W (index batch trails)
        xi_WCp = xnp.moveaxis(xi_CpW, tuple(range(-n_idx, 0)), tuple(range(n_idx)))  # -> W + C + (p,)
        xis.append(xi_WCp)
    return tuple(xis)


def _onehot_vectors(index, down_tucker_cores):
    '''Unit vectors e_{index_k} (shape W + (Nk,)) -- the "apply vectors" whose adjoint is the entry
    scatter, so that tv_entries_transpose is tv_apply_transpose with these one-hot vectors.'''
    use_jax = tree_contains_jax((index, down_tucker_cores))
    is_uniform = is_ndarray(down_tucker_cores)
    xnp, _, _ = get_backend(is_uniform, use_jax)
    index = xnp.array(index)
    if is_uniform:
        # packed one-hots (d,)+W+(N,): eye(N) indexed by the (d,)+W index array. N is the padded mode dim;
        # each real index_i < Ni <= N puts the 1 in the real region (the padding stays zero -> contracts
        # to zero against the masked supercore).
        N = down_tucker_cores.shape[-1]
        return xnp.eye(N)[index]
    return tuple(xnp.eye(B.shape[-1])[index[i]] for i, B in enumerate(down_tucker_cores))
