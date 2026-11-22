# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The ``apply`` sampling type: contract a T3 with vectors in ALL modes -> one scalar per sample.

Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise transpose's derivative in the
cores (the exact Hessian's sampling term), and the frame sweeps for applies.
Specializes the general probing machinery (containment probe ⊃ apply ⊃ entries; this module
imports from ``probing``, never the reverse). Costs and the role in Riemannian least-squares:
``docs/entries_apply_probe.md``.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.sampling_variables as sampling_variables
from t3toolbox.backend.common import *
import math
import t3toolbox.backend.probing as probing
from t3toolbox.backend.probing import compute_xi, compute_mu, compute_dxi, compute_sigma_hat, _sigma_step

__all__ = [
    't3_apply',
    't3_apply_ambient_transpose',
    'tv_apply',
    'tv_apply_transpose',
    'tv_precompute_apply_frame_sweep',
    'tv_apply_jacobian_from_sweep',
    'tv_apply_transpose_from_sweep',
    't3_apply_corewise_tangent',
    't3_apply_corewise_transpose',
    't3_apply_corewise_hessian',
    't3_apply_corewise_hessian_from_sweep',
    # Apply a dense tensor
    'dense_apply',
]

def _t3_apply_step(
        mu_WCa: NDArray,                                # carry: W+C+(rLi,)
        v_B_G:  typ.Tuple[NDArray, NDArray, NDArray],   # (v, B, G): W+(Ni,) ; C+(ni,Ni) ; C+(rLi,ni,rL(i+1))
) -> typ.Tuple[NDArray, typ.Tuple[int]]:   # (next carry, (0,) -- nothing emitted per mode)
    '''One mode of the all-modes contraction of :py:func:`t3_apply`. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    v_Wo, B_Cpo, G_Capb = v_B_G
    mu_WCb = contractions.contract('WCa,Caib,Wo,Cio->WCb', 
        mu_WCa, G_Capb, v_Wo, B_Cpo,
    )
    return mu_WCb, (0,)


def t3_apply(
        x:                 typ.Tuple,                    # (tucker_cores, tt_cores), elm_shapes C+(nDi,Ni), C+(ri,nDi,r(i+1))
        ww:                typ.Sequence[NDArray],        # len=d, elm_shape=F_in_i+W+(Ni,): input jets
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # z. shape=F+W+C
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
        ww:                 typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)
        residual:           NDArray,                # shape=W+C
        sum_over_samples:   bool = False,           # True: W becomes the CP rank (ambient J^T r)
) -> typ.Sequence[NDArray]:  # canonical (CP) factors. len=d, ith elm_shape=stack_shape+(R, Ni)
    '''Ambient transpose of :py:func:`t3_apply`: back-project ``residual`` into CP factors.

    The *ambient* adjoint -- the transpose of ``apply`` as a linear map on the **full tensor space**
    (``X -> ( <X, w0^W (x) ... (x) w_{d-1}^W> )_W``). Frame-free; the back-projection
    ``residual * (w0 (x) ... (x) w_{d-1})`` is rank-1, whose natural representation is a **canonical (CP)
    decomposition** (``apply`` consumes one vector per mode; its adjoint emits one scaled vector per
    mode). This is distinct from the *corewise* transpose (gradient w.r.t. a base point's cores) and
    the *tangent* transpose (Riemannian gradient) -- see ``docs/transposes.md`` for the full taxonomy.

    - ``sum_over_samples=False`` (primary): ``W`` is a passthrough stacking axis -- a ``W (+ C)`` stack
      of rank-1 CP tensors (CP rank ``R=1``).
    - ``sum_over_samples=True``: ``W`` becomes the CP **rank** -- one rank-``|W|`` CP tensor
      ``sum_W residual_W * (w0^W (x) ...)`` (the ambient ``J^T r``). Cheap as CP (``O(d |W| N)``, the shared
      rank index stays implicit); the ``|W|^2`` cost of a *dense* Tucker tensor train is incurred only
      if you convert with ``t3_conversions.t3_from_canonical``.

    Returns the CP ``factors`` (``residual`` folded into the first), in the layout
    ``t3_conversions.t3_from_canonical`` consumes.
    '''
    use_jax = tree_contains_jax((residual, ww))
    xnp, _, _ = get_backend(False, use_jax)
    c = xnp.asarray(residual)

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
        frame:             typ.Tuple,                    # (D, U, L, R)
        variations:        typ.Tuple,                    # (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        ww:                typ.Sequence[NDArray],        # len=d, elm_shape=F_in_i+W+(Ni,): input jets
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # Jp. shape=F+W+K+C
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
        frame:             typ.Tuple,                    # (D, U, L, R)
        ww:                typ.Sequence[NDArray],        # len=d, elm_shape=F_in_i+W+(Ni,): input jets
        residual:          NDArray,                       # of the type of z: shape=F+W+[K]+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # V blocks. len=d, elm_shape=[W]+[K]+C+(nUi,Ni)
        typ.Tuple[NDArray, ...],                         # H blocks. len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))
]:
    '''The transpose of :py:func:`tv_apply`: ``J^T`` applied to ``residual``, a tangent's blocks at ``frame``.

    ``residual`` is a vector of the type of ``z`` (a stack ``K`` of them allowed), with no weight; it is turned inside
    (a dual jet). One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``JtJp`` with
    ``residual`` held as ``Jp`` (the route of a held ``Jp``: ``J^T`` of the vector held). The blocks are summed over
    the samples by default. For repeated calls at one frame (inside an iterative solver), call
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
    >>> Jp = apply.tv_apply(frame, variations, ww)
    >>> V, H = apply.tv_apply_transpose(frame, ww, Jp)
    >>> pair = lambda a, b: sum(float(np.sum(u * v)) for u, v in zip(a[0] + a[1], tuple(b[0]) + tuple(b[1])))
    >>> print(bool(np.isclose(pair((V, H), variations), float(np.sum(Jp * Jp)))))       # <p, J^T J p> = |J p|^2
    True
    '''
    held = sampling_variables.HeldVariables(Jp=residual)
    (blocks,) = sampling_variables.sampling_variables('apply', frame, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def tv_precompute_apply_frame_sweep(
        frame:  typ.Tuple[
            typ.Sequence[NDArray],          # down_tucker_cores  D. len=d
            typ.Sequence[NDArray],          # up_tt_cores        U. len=d
            typ.Sequence[NDArray],          # left_tt_cores      L. len=d
            typ.Sequence[NDArray],          # right_tt_cores     R. len=d
        ],                                  # frame order = T3Frame.data = (down, up, left, right)
        ww:     typ.Sequence[NDArray],      # apply vectors, len=d, elm_shape=W+(Ni,)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xis. len=d, elm_shape=W+C+(nDi,)
    typ.Sequence[NDArray],  # mus. len=d, elm_shape=W+C+(rLi,)
]:                                          # lean frame sweep -- (xis, mus) only
    '''The all-modes apply **frame sweep** (lean): the frame edge variables ``(xi-hat, mu-hat)`` that
    depend only on the frame frame and the apply vectors ``ww`` -- NOT on the tangent or residual.
    Computing them is the expensive, ``W``-scaled part of the apply Jacobian; precomputed **once per
    frame** and reused across every ``J`` / ``Jᵀ`` of an inner solve (the reuse hook for ``fitting.py``).

    **Lean ``(xis, mus)`` only** (not the right ``nu`` / downward ``eta`` sweeps): the all-modes apply
    forward AND its adjoint-state transpose use only ``(xi, mu)`` -- the transpose recomputes the right
    context as ``sigma_hat`` from the residual rather than storing ``nu``/``eta``, halving the
    ``W``-scaling memory (apply on the manifold, §6.2.2 of Alger et al. (2026)). Probe, which leaves a
    mode free, needs the full sweep -- :py:func:`tv_precompute_probe_frame_sweep`.

    See Also
    --------
    tv_apply_jacobian_from_sweep
    tv_apply_transpose_from_sweep
    tv_precompute_probe_frame_sweep
    '''
    down_tucker_cores, up_tt_cores, left_tt_cores, right_tt_cores = frame
    xis = compute_xi(down_tucker_cores, ww)
    mus = compute_mu(left_tt_cores, xis)
    return xis, mus


def tv_apply_jacobian_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nUi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nDi,rR(i+1))
        ],
        ww:         typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,) -- for the variation's dxis
        frame:      typ.Tuple[
            typ.Sequence[NDArray],          # down_tucker_cores  D. len=d  (unused; for a uniform call signature)
            typ.Sequence[NDArray],          # up_tt_cores        U. len=d
            typ.Sequence[NDArray],          # left_tt_cores      L. len=d  (unused)
            typ.Sequence[NDArray],          # right_tt_cores     R. len=d
        ],                                  # frame order = T3Frame.data = (down, up, left, right)
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray],          # xis
            typ.Sequence[NDArray],          # mus
        ],                                  # = tv_precompute_apply_frame_sweep(frame, ww)  (lean)
) -> NDArray:                               # the scalar apply(v, ww), one per stack element; shape = W + K + C
    '''Forward all-modes apply of a tangent vector reusing a precomputed frame sweep -- the bare ``𝒥`` with
    the frame edge variables injected. Equivalent to :py:func:`tv_apply`, but it takes the lean
    ``(xis, mus)`` from ``frame_sweep`` instead of recomputing them (the reuse hook for ``fitting.py``).
    Only the variation-dependent ``dxis`` is computed here. No gauge projector ``Π``.

    See Also
    --------
    tv_precompute_apply_frame_sweep
    tv_apply
    '''
    var_tucker_cores, var_tt_cores = variation
    _, up_tt_cores, _, right_tt_cores = frame
    xis, mus = frame_sweep
    dxis = compute_dxi(var_tucker_cores, ww)     # variation-dependent; not part of the frame sweep
    return _apply_from_xis(xis, dxis, mus, right_tt_cores, up_tt_cores, var_tt_cores)


def tv_apply_transpose_from_sweep(
        c:          NDArray,                # residual, shape = W + K + C (K optional)
        ww:         typ.Sequence[NDArray],  # apply vectors (one-hot e_index for entries), len=d, elm_shape=W+(Ni,)
        frame:      typ.Tuple[
            typ.Sequence[NDArray],          # down_tucker_cores  D. len=d  (unused; uniform call signature)
            typ.Sequence[NDArray],          # up_tt_cores        U. len=d
            typ.Sequence[NDArray],          # left_tt_cores      L. len=d  (unused)
            typ.Sequence[NDArray],          # right_tt_cores     R. len=d
        ],                                  # frame order = T3Frame.data = (down, up, left, right)
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray],          # xis
            typ.Sequence[NDArray],          # mus
        ],                                  # = tv_precompute_apply_frame_sweep(frame, ww)  (lean: no nu/eta)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (V_tildes, H_tildes) = T3Variations.data
    '''Transpose of the all-modes apply reusing a precomputed frame sweep -- the bare ``𝒥ᵀ``, by the
    **adjoint-state** method (the scalar residual ``c`` seeds one reverse ``sigma_hat`` sweep; see
    :py:func:`_apply_transpose_adjoint`). Takes the **lean** ``(xis, mus)`` sweep + the frame cores ``U,
    R`` (it recomputes the right context rather than storing ``nu``/``eta`` -- half the memory). Reuse
    hook for ``fitting.py`` (one frame sweep feeds the forward and this transpose). Full ``W + K + C``
    (the residual ``c`` may carry the tangent stack ``K``). No gauge projector ``Π``.

    See Also
    --------
    tv_precompute_apply_frame_sweep
    tv_apply_transpose
    '''
    _, up_tt_cores, _, right_tt_cores = frame
    xis, mus = frame_sweep
    return _apply_transpose_adjoint(c, ww, xis, mus, up_tt_cores, right_tt_cores, sum_over_probes)


def t3_apply_corewise_tangent(
        x:                 typ.Tuple,                    # (B, G): the point whose cores the directions vary
        ww:                typ.Sequence[NDArray],        # len=d, elm_shape=F_in_i+W+(Ni,): input jets
        core_directions:   typ.Tuple,                    # (B_dot, G_dot), the shapes of B, G with K+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # J x_dot. shape=F+W+K+C
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
        x:                 typ.Tuple,                    # (tucker_cores, tt_cores): the point
        ww:                typ.Sequence[NDArray],        # len=d, elm_shape=F_in_i+W+(Ni,): input jets
        residual:          NDArray,                       # of the type of z: shape=F+W+[K]+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
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
    held = sampling_variables.HeldVariables(Jp=residual)
    (blocks,) = sampling_variables.sampling_variables('apply', x, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def t3_apply_corewise_hessian_from_sweep(
        c:                  NDArray,                # residual (omega^2 applied), shape=W+C

        ww:                 typ.Union[typ.Sequence[NDArray], NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)

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
    ''':py:func:`t3_apply_corewise_hessian` reusing a precomputed lean ``frame_sweep`` -- the reuse hook for a local
    model. :py:func:`tv_precompute_apply_frame_sweep` is the same at the manifold frame ``(D, U, L, R)`` and at the
    corewise frame ``(D, L, L, L)`` of its base point ``theta = (D, L)`` (``xi`` through ``D``, ``mu`` through
    ``L``), so the model's own sweep is reused; it must be ``core_pair``'s own, which nothing here checks.'''
    _, tt_cores = core_pair
    tucker_dirs, tt_dirs = core_directions
    xis, mus = frame_sweep
    xis_dot = compute_dxi(tucker_dirs, ww)                                       # xi_dot = Bdot w
    mus_dot = probing.compute_sigma(tt_dirs, tt_cores, tt_cores, xis, xis_dot, mus)   # mu_dot, through G
    return _apply_corewise_hessian_adjoint(c, ww, xis, xis_dot, mus, mus_dot, tt_cores, tt_dirs)


def t3_apply_corewise_hessian(
        x:                 typ.Tuple,                    # (B, G): the point whose cores the directions vary
        ww:                typ.Sequence[NDArray],        # len=d, elm_shape=F_in_i+W+(Ni,): input jets
        residual:          NDArray,                       # c, of the type of z: shape=F+W+C; no weight
        core_directions:   typ.Tuple,                    # (B_dot, G_dot), the shapes of B, G with K+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker block. len=d, elm_shape=[W]+K+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT block.     len=d, elm_shape=[W]+K+C+(ri,nDi,r(i+1))
]:
    '''The corewise Hessian: the second derivative of ``phi(x) = <residual, z(x)>`` in the cores, applied to
    ``core_directions`` (the derivative of :py:func:`t3_apply_corewise_transpose` along them).

    ``residual`` is a vector of the type of ``z``, no weight, turned inside. One call of
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point ``x`` wanting ``Kp``
    with ``residual`` held as ``r``. Summed over the samples (``sum_over_samples=False``: per sample, ``W``
    leading the blocks' stack). For repeated calls at one frame (inside an iterative solver), call
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
    held = sampling_variables.HeldVariables(r=residual)
    (blocks,) = sampling_variables.sampling_variables('apply', x, ww, ('Kp',), order=order, held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks


##########################################
########    Dense ground truth    ########
##########################################

def dense_apply(
        T:      NDArray,                                # shape=C+(N0,...,N(d-1))
        ww:     typ.Sequence[NDArray],                  # len=d, elm_shape=F_in_i+W+(Ni,): the input jets
        order:  typ.Tuple[int, ...] = (),               # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
) -> NDArray:                                           # z. shape=F+W+C
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


def _apply_from_xis_step(
        sigma: NDArray,   # carry: W+K+C+(rRi,)
        x:     typ.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray],  # (R, U, H, xi, dxi, mu)
) -> typ.Tuple[NDArray, typ.Tuple[int]]:   # (next carry, (0,) -- only the terminal carry is used)
    '''One edge of the sweep of :py:func:`_apply_from_xis`, emitting nothing (the sequence-keeping twin
    is ``probing._sigma_sweep_step``). Closure-free scan body -- ``docs/contributor/scan_body_principles.md``.'''
    R, U, H, xi, dxi, mu = x
    return _sigma_step(sigma, R, U, H, xi, dxi, mu), (0,)


def _apply_from_xis(xis, dxis, mus, right_tt_cores, up_tt_cores, var_tt_cores):
    '''Run the perturbation sigma sweep (via R) to its TERMINAL carry and contract the final bond.

    Shared tail of tv_apply and tv_entries (they differ only in how xis/dxis are formed).
    '''
    use_jax = tree_contains_jax((xis, dxis, mus, right_tt_cores))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    # carry sigma is W+K+C; take the leading stack from dxis (carries K), not xis (W+C only).
    rR0 = right_tt_cores[0].shape[-3]
    init = xnp.zeros(dxis[0].shape[:-1] + (rR0,))
    sigma_terminal, _ = xscan(_apply_from_xis_step, init, (right_tt_cores, up_tt_cores, var_tt_cores, xis, dxis, mus))
    return xnp.sum(sigma_terminal, axis=-1)   # contract the terminal bond -> W + K + C


def _apply_transpose_adjoint(c, ww, xis, mus, up_tt_cores, right_tt_cores, sum_over_probes):
    '''Adjoint-state assembly shared by apply/tv_entries_transpose_from_sweep -- the **low-memory,
    K-aware** transpose (replaces the old scatter). The scalar residual ``c`` seeds one reverse
    ``sigma_hat`` sweep (recomputing the right context, so ``nu``/``eta`` are never stored); then

        dxi-hat_k  = mu-hat_{k-1} . U_k . sigma-hat_k                 # over the up mode nU
        H-tilde_k  = mu-hat_{k-1} (x) xi-hat_k (x) sigma-hat_k        # over (rL, nD, rR)
        V-tilde_k  = dxi-hat_k (x) w_k                                # over (nU, N)

    Uses only the lean frame sweep ``(xis, mus)`` (half the memory of the ``(xis, mus, nus, etas)`` the
    scatter stored -- the ``W``-scaling ``nu``/``eta`` are gone), at the cost of the ``sigma_hat`` sweep
    per transpose. Full ``W + K + C``: the residual ``c`` may carry the tangent stack ``K`` (the output
    space of a ``K``-stacked forward ``tv_apply``), which rides into the variation gradient -- the
    capability the scatter lacked. ``sum_over_probes=True`` sums the probe stack ``W`` (the ``J^T r``
    back-projection); ``False`` keeps it as the output tangent stack. ``K``/``C`` always kept.'''
    is_uniform = is_ndarray(mus)
    sigma_hats = compute_sigma_hat(right_tt_cores, xis, c)               # polymorphic reverse sweep

    if is_uniform:
        # d-prefixed WKC, vectorized over the core index d (NOT a per-core loop -> no jit
        # unroll). ww is the packed apply/one-hot probe supercore (d,)+W+(N,) -> n_probe = len(W). The
        # ragged loop below is the oracle.
        n_probe = ww.ndim - 2
        dxi_hats = contractions.contract('dWCa,dCaib,dWKCb->dWKCi', mus, up_tt_cores, sigma_hats)
        if sum_over_probes:
            H_tildes = contractions.contract('dWCa,dWCi,dWKCb->dKCaib', mus, xis, sigma_hats, len_W=n_probe)
            V_tildes = contractions.contract('dWo,dWKCa->dKCao', ww, dxi_hats)
        else:
            H_tildes = contractions.contract('dWCa,dWCi,dWKCb->dWKCaib', mus, xis, sigma_hats, len_W=n_probe)
            V_tildes = contractions.contract('dWo,dWKCa->dWKCao', ww, dxi_hats)
        return V_tildes, H_tildes

    n_probe = ww[0].ndim - 1
    dxi_hats = tuple(contractions.contract('WCa,Caib,WKCb->WKCi', mu, U, sig_hat)        # dxi_hat = mu . U . sigma_hat
                     for mu, U, sig_hat in zip(mus, up_tt_cores, sigma_hats))
    if sum_over_probes:
        H_tildes = tuple(contractions.contract('WCa,WCi,WKCb->KCaib', mu, xi, sig_hat, len_W=n_probe)
                          for mu, xi, sig_hat in zip(mus, xis, sigma_hats))
        V_tildes = tuple(contractions.contract('Wo,WKCa->KCao', w, dxh) for w, dxh in zip(ww, dxi_hats))
    else:
        H_tildes = tuple(contractions.contract('WCa,WCi,WKCb->WKCaib', mu, xi, sig_hat, len_W=n_probe)
                          for mu, xi, sig_hat in zip(mus, xis, sigma_hats))
        V_tildes = tuple(contractions.contract('Wo,WKCa->WKCao', w, dxh) for w, dxh in zip(ww, dxi_hats))
    return V_tildes, H_tildes   # (var_tucker, var_tt) = T3Variations.data


def _apply_corewise_hessian_adjoint(c, ww, xis, xis_dot, mus, mus_dot, tt_cores, tt_dirs):
    '''The derivative of :py:func:`_apply_transpose_adjoint` (``sum_over_probes=True``) at the corewise frame
    ``(B, G, G, G)`` along ``(Bdot, Gdot)``, shared by :py:func:`t3_apply_corewise_hessian_from_sweep` and
    :py:func:`~t3toolbox.backend.entries.t3_entries_corewise_hessian_from_sweep` (which differ only in how
    ``xi`` / ``xi_dot`` and the apply vectors ``ww`` are formed). With the residual ``c`` fixed (``W + C``; no
    ``K``):

        sigma_hat       = the c-seeded reverse sweep through G                (compute_sigma_hat)
        sigma_hat_dot   = its derivative: the reverse GN tangent sweep        (compute_tau(Gdot, G, G, xi, xi_dot, sigma_hat))
        dxi_hat_dot     = d(mu G sigma_hat)                                    (compute_deta(Gdot, G, G, mu, sigma_hat, mu_dot, sigma_hat_dot))
        dB              = sum_W  w (x) dxi_hat_dot
        dG              = sum_W  mu_dot . E + mu . E_dot,  E = xi (x) sigma_hat,  E_dot = xi_dot (x) sigma_hat + xi (x) sigma_hat_dot

    -- the single-term assemblies' product rule, ``dG``'s three terms grouped by their common factor ``mu`` (two
    W-summed finals). Polymorphic: ``d``-prefixed contractions on the uniform path, a per-core loop on ragged.'''
    is_uniform = is_ndarray(mus)
    sigma_hats = compute_sigma_hat(tt_cores, xis, c)                                   # W+C (c has no K)
    sigma_hats_dot = probing.compute_tau(tt_dirs, tt_cores, tt_cores, xis, xis_dot, sigma_hats)
    dxi_hats_dot = probing.compute_deta(tt_dirs, tt_cores, tt_cores, mus, sigma_hats, mus_dot, sigma_hats_dot)

    if is_uniform:
        n_probe = ww.ndim - 2                     # ww is the W-only packed apply / one-hot supercore (d,)+W+(N,)
        dB = contractions.contract('dWo,dWKCa->dKCao', ww, dxi_hats_dot)
        E = contractions.contract('dWCi,dWCb->dWCib', xis, sigma_hats)
        E_dot = (contractions.contract('dWKCi,dWCb->dWKCib', xis_dot, sigma_hats, len_W=n_probe)
                 + contractions.contract('dWCi,dWKCb->dWKCib', xis, sigma_hats_dot, len_W=n_probe))
        dG = (contractions.contract('dWKCa,dWCib->dKCaib', mus_dot, E, len_W=n_probe)
              + contractions.contract('dWCa,dWKCib->dKCaib', mus, E_dot, len_W=n_probe))
        return dB, dG

    n_probe = ww[0].ndim - 1
    dB = tuple(contractions.contract('Wo,WKCa->KCao', w, dxd) for w, dxd in zip(ww, dxi_hats_dot))
    dG = []
    for mu, mud, xi, xid, sig_hat, sig_hat_d in zip(mus, mus_dot, xis, xis_dot, sigma_hats, sigma_hats_dot):
        E = contractions.contract('WCi,WCb->WCib', xi, sig_hat)
        E_dot = (contractions.contract('WKCi,WCb->WKCib', xid, sig_hat, len_W=n_probe)
                 + contractions.contract('WCi,WKCb->WKCib', xi, sig_hat_d, len_W=n_probe))
        dG.append(contractions.contract('WKCa,WCib->KCaib', mud, E, len_W=n_probe)
                  + contractions.contract('WCa,WKCib->KCaib', mu, E_dot, len_W=n_probe))
    return dB, tuple(dG)
