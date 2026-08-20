# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Shared Tucker factors (SF-T3): the sharing partition, the tied-factors checkers, and the
per-frame companion.

``validate_sharing`` canonicalizes a per-mode label spec into the static ``groups`` form;
``t3_sharing_residual`` / ``t3_tucker_factors_shared`` are the non-enforcing tied-factors
checkers (the safe-mode precondition behind the shared operations); ``t3_share_tucker_cores``
is the plain per-group mean (drift repair for nearly-tied POINTS -- never the metric
projection of a tangent, which is geometry-specific and lives with the shared geometry).
``fv_shared_frame_data`` derives the :py:class:`T3SharedFrameData` companion from an
orthogonal frame -- the per-group center cores and the thin SVD of the stacked ``S`` factors
that the shared geometry's projection, retraction, and spectrum diagnostics consume.

A shared T3 is an ordinary Tucker tensor train whose Tucker factors are equal within
user-specified groups of modes -- the SF-ETT decomposition of Molozhavenko & Rakhuba (2026),
"Optimization on the extended tensor-train manifold with shared factors" (Comput. Appl. Math.
45:221), generalized to an arbitrary partition of the modes into sharing groups.
"""
import numpy as np
import typing as typ
from dataclasses import dataclass

import t3toolbox.backend.tt_orthogonalization as tt_orthogonalization
from t3toolbox.backend.common import *

__all__ = [
    'validate_sharing',
    'nontrivial_groups',
    't3_sharing_residual',
    't3_tucker_factors_shared',
    't3_share_tucker_cores',
    'T3SharedFrameData',
    'fv_shared_frame_data',
    'fv_share_tucker_variations',
    'fv_mean_tucker_variations',
]


def validate_sharing(
        sharing:    typ.Sequence,       # len=d, static; one hashable group label per mode, e.g. (0, 1, 1)
        shape:      typ.Sequence[int],  # (N0, ..., N(d-1)); modes sharing a label need equal sizes
) -> typ.Tuple[typ.Tuple[int, ...], ...]:  # groups, static; mode indices per group (canonical form)
    '''Validate a sharing partition and return its canonical ``groups`` form.

    ``sharing`` assigns one hashable group label per mode; modes with equal labels share one
    Tucker factor. The canonical form lists each group's mode indices (ascending), groups
    ordered by first mode, singleton groups included -- static structure (a jit aux / closure
    constant), never traced. Structural problems -- wrong length, unhashable labels, unequal
    mode sizes within a group -- raise unconditionally (both safety modes).

    Examples
    --------
    Labels are arbitrary hashables; groups come back as mode-index tuples, ordered by first
    mode, singletons included:

    >>> import t3toolbox.backend.sharing as sharing
    >>> sharing.validate_sharing((0, 1, 1, 2, 2, 2), (4, 5, 5, 6, 6, 6))
    ((0,), (1, 2), (3, 4, 5))
    >>> sharing.validate_sharing(('in', 'out', 'in'), (7, 5, 7))   # non-adjacent groups are fine
    ((0, 2), (1,))

    Gotcha: modes in a group must have equal mode sizes (a shared factor needs one ambient
    dimension) -- a mismatch is a structural error:

    >>> sharing.validate_sharing((0, 0), (4, 5))   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError
    '''
    sharing = tuple(sharing)
    shape = tuple(int(N) for N in shape)
    if len(sharing) != len(shape):
        raise ValueError(
            'sharing must assign one group label per mode: len(sharing) = %d != %d = number of modes'
            % (len(sharing), len(shape)))

    modes_by_label = {}                       # insertion-ordered -> groups ordered by first mode
    for ii, label in enumerate(sharing):
        try:
            hash(label)
        except TypeError:
            raise ValueError(
                'sharing labels must be hashable; got %r at mode %d' % (label, ii))
        modes_by_label.setdefault(label, []).append(ii)

    groups = tuple(tuple(modes) for modes in modes_by_label.values())
    for group in groups:
        sizes = tuple(shape[ii] for ii in group)
        if len(set(sizes)) > 1:
            raise ValueError(
                'modes in a sharing group must have equal mode sizes (one shared factor needs '
                'one ambient dimension); group %r has sizes %r' % (group, sizes))
    return groups


def nontrivial_groups(
        groups: typ.Tuple[typ.Tuple[int, ...], ...],  # static; canonical form (validate_sharing)
) -> typ.Tuple[typ.Tuple[int, ...], ...]:             # static; the groups with >= 2 modes
    '''The sharing groups that actually tie anything (two or more modes), in canonical order.

    An all-singleton result means the partition is trivial -- shared operations dispatch to
    their unshared code paths in that case.

    Examples
    --------
    >>> import t3toolbox.backend.sharing as sharing
    >>> groups = sharing.validate_sharing((0, 1, 1, 2), (4, 5, 5, 6))
    >>> sharing.nontrivial_groups(groups)
    ((1, 2),)
    >>> sharing.nontrivial_groups(sharing.validate_sharing((0, 1, 2), (4, 5, 6)))
    ()
    '''
    return tuple(group for group in groups if len(group) > 1)


def _validate_group_tucker_ranks(
        tucker_cores:   typ.Sequence[NDArray],              # len=d, elm_shape=stack_shape+(ni, Ni)
        groups:         typ.Tuple[typ.Tuple[int, ...], ...],  # static; from validate_sharing
) -> None:
    '''Structural check: tied factors need equal Tucker ranks within each group (always raises).'''
    for group in groups:
        ranks = tuple(tucker_cores[ii].shape[-2] for ii in group)
        if len(set(ranks)) > 1:
            raise ValueError(
                'modes in a sharing group must have equal Tucker ranks (their factors are one '
                'shared array); group %r has ranks %r' % (group, ranks))


def t3_sharing_residual(
        x:          typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores. len=d, elm_shape=stack_shape+(ni, Ni)
            typ.Sequence[NDArray],  # tt_cores.     len=d, elm_shape=stack_shape+(ri, ni, r(i+1))
        ],
        sharing:    typ.Sequence,   # len=d, static; one hashable group label per mode
) -> NDArray:  # shape = stack_shape; max relative factor deviation per stack element (0 == exactly tied)
    '''Non-enforcing check of the shared-factors property, **per stack element**.

    Returns the max over groups and group modes of ``||B_i - B_ref||_F / ||B_ref||_F``
    (``B_ref`` = the group's first factor), reduced over the non-stack axes. Exactly-tied
    factors give 0; the measure is relative (the factors' overall scale cancels), so a caller
    thresholds against a relative tolerance (``<= rtol``). A zero reference with a nonzero
    other factor gives ``inf``. This is the residual behind the shared operations' safe-mode
    tied-factors precondition (paired with ``safety.effective_rtol`` at the frontend check
    sites). Structural problems -- invalid partition, unequal Tucker ranks within a group --
    raise unconditionally.

    Examples
    --------
    Exactly-tied factors (one array used at both modes) give exactly 0; independent random
    factors do not:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.sharing as sharing
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 2, 2, 1))
    >>> tk, tt = x.data
    >>> tied = ((tk[0], tk[0], tk[2]), tt)                       # tie modes 0, 1
    >>> print(float(sharing.t3_sharing_residual(tied, (0, 0, 1))))
    0.0
    >>> print(bool(sharing.t3_sharing_residual(x.data, (0, 0, 1)) > 0.1))
    True

    Stack-aware -- one verdict per stack element (perturb one element of a tied stack):

    >>> xs = t3.TuckerTensorTrain.randn((6, 6), (3, 3), (1, 2, 1), stack_shape=(2,))
    >>> tks, tts = xs.data
    >>> B2 = np.asarray(tks[0]).copy()
    >>> B2[1] += 1e-3 * np.random.randn(3, 6)                    # perturb stack element 1 only
    >>> r = sharing.t3_sharing_residual(((tks[0], B2), tts), (0, 0))
    >>> print(r.shape, bool(r[0] == 0.0), bool(r[1] > 1e-5))
    (2,) True True
    '''
    tucker_cores, tt_cores = x
    shape = tuple(B.shape[-1] for B in tucker_cores)
    groups = validate_sharing(sharing, shape)
    _validate_group_tucker_ranks(tucker_cores, groups)

    use_jax = tree_contains_jax(x)
    xnp, _, _ = get_backend(False, use_jax)

    devs = []
    for group in groups:
        if len(group) < 2:
            continue
        B_ref = tucker_cores[group[0]]
        denom = xnp.sqrt(xnp.sum(B_ref * B_ref, axis=(-2, -1)))   # keep stack
        for ii in group[1:]:
            diff = tucker_cores[ii] - B_ref
            num = xnp.sqrt(xnp.sum(diff * diff, axis=(-2, -1)))
            pos = denom > 0.0
            devs.append(xnp.where(pos, num / xnp.where(pos, denom, 1.0),   # branch-free zero guard
                                  xnp.where(num > 0.0, xnp.inf, 0.0)))
    if not devs:                                                  # all-singleton: trivially tied
        return xnp.zeros(tucker_cores[0].shape[:-2])
    return xnp.max(xnp.stack(devs), axis=0)   # max over the checks, keep stack_shape


def t3_tucker_factors_shared(
        x:          typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores. len=d, elm_shape=stack_shape+(ni, Ni)
            typ.Sequence[NDArray],  # tt_cores.     len=d, elm_shape=stack_shape+(ri, ni, r(i+1))
        ],
        sharing:    typ.Sequence,   # len=d, static; one hashable group label per mode
        rtol:       float = 1e-9,   # relative tolerance on the factor deviation
) -> NDArray:  # bool array, shape = stack_shape (scalar/0-d when unstacked)
    '''True (per stack element) where the Tucker factors are tied within every sharing group.

    The boolean form of :py:func:`t3_sharing_residual` (``residual <= rtol``) -- a
    non-enforcing checker with an explicit tolerance (the backend is check-free; frontend
    safe-mode sites pair the residual with ``safety.effective_rtol`` instead). Reduce with
    ``.all()`` for a single verdict on a stacked ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.sharing as sharing
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 2, 2, 1))
    >>> tk, tt = x.data
    >>> print(bool(sharing.t3_tucker_factors_shared(((tk[0], tk[0], tk[2]), tt), (0, 0, 1))))
    True
    >>> print(bool(sharing.t3_tucker_factors_shared(x.data, (0, 0, 1))))
    False
    '''
    return t3_sharing_residual(x, sharing) <= rtol


def t3_share_tucker_cores(
        x:          typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores. len=d, elm_shape=stack_shape+(ni, Ni)
            typ.Sequence[NDArray],  # tt_cores.     len=d, elm_shape=stack_shape+(ri, ni, r(i+1))
        ],
        sharing:    typ.Sequence,   # len=d, static; one hashable group label per mode
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],        # new_tucker_cores. len=d; ONE shared array per group
    typ.Tuple[NDArray, ...],        # tt_cores, untouched
]:
    '''Tie the Tucker factors exactly, by per-group arithmetic averaging.

    Each group's factor is replaced by the group mean, and the SAME array is assigned to every
    mode of the group -- the tie is exact by construction, never floating-point agreement. TT
    cores are untouched. The represented tensor changes unless the factors were already tied
    (drift repair for NEARLY-tied points, e.g. insurance after an operation that guarantees
    ties only to roundoff). The mean is computed as ``B_ref + mean(B_i - B_ref)``, so an
    exactly-tied group is a **bitwise fixed point** for any group size (the plain
    ``sum(B_i)/k`` would perturb the last ulp already at ``k = 3``).

    This is a repair of a POINT's representation, **not** the metric projection of a tangent
    onto the tied tangent space -- tangent tying is geometry-specific (the manifold geometry
    weights coordinates by the frame's ``S`` factors; the corewise geometry averages raw core
    perturbations) and lives with the shared geometry.

    Examples
    --------
    The group factor becomes the mean, assigned as one array (identity, not just equality),
    and the result passes the tied-factors check exactly:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.sharing as sharing
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 2, 2, 1))
    >>> tk, tt = x.data
    >>> tk2, tt2 = sharing.t3_share_tucker_cores(x.data, (0, 0, 1))
    >>> print(tk2[0] is tk2[1], tt2 is tt)
    True True
    >>> print(bool(np.allclose(np.asarray(tk2[0]), (np.asarray(tk[0]) + np.asarray(tk[1])) / 2)))
    True
    >>> print(float(sharing.t3_sharing_residual((tk2, tt2), (0, 0, 1))))
    0.0

    Already-tied input comes back with unchanged factor values:

    >>> tk3, _ = sharing.t3_share_tucker_cores(((tk[0], tk[0], tk[2]), tt), (0, 0, 1))
    >>> print(bool(np.array_equal(np.asarray(tk3[0]), np.asarray(tk[0]))))
    True
    '''
    tucker_cores, tt_cores = x
    shape = tuple(B.shape[-1] for B in tucker_cores)
    groups = validate_sharing(sharing, shape)
    _validate_group_tucker_ranks(tucker_cores, groups)

    new_tucker_cores = list(tucker_cores)
    for group in groups:
        if len(group) < 2:
            continue
        # mean = B_ref + mean(B_i - B_ref): algebraically the group mean, but an exactly-tied
        # group is a bitwise fixed point (the differences are exactly zero) for ANY group size
        # -- the plain (sum B_i)/k rounds twice and perturbs the last ulp already at k=3.
        B_ref = tucker_cores[group[0]]
        drift = tucker_cores[group[1]] - B_ref
        for ii in group[2:]:
            drift = drift + (tucker_cores[ii] - B_ref)
        mean = B_ref + drift / len(group)
        for ii in group:
            new_tucker_cores[ii] = mean       # the SAME array object at every group mode
    return tuple(new_tucker_cores), tuple(tt_cores)


@dataclass(frozen=True, eq=False)  # eq=False -> identity hash/eq (array fields; value-eq is ambiguous)
class T3SharedFrameData:
    '''The per-frame companion of the shared geometry: everything the tied projection,
    the shared retraction, and the group-spectrum diagnostics need, derived from an
    orthogonal frame by :py:func:`fv_shared_frame_data` (never stored inside a frame).

    All array fields carry the frame stack ``C`` leading; ``groups`` / ``row_splits`` are
    static structure (jax aux). One entry per NONTRIVIAL group (>= 2 modes), in canonical
    order; ``svd_*`` is the thin SVD of the stacked matrix
    ``M_g = concat_i(S_i^T)`` -- deliberately an SVD, never a Cholesky/Gram: the solve gets
    the intrinsic least-squares sensitivity, ``svd_s`` IS the group spectrum ``s_g`` at full
    (non-squared) accuracy, and the clipped pseudoinverse is well-defined at the
    rank-deficient points rank continuation visits.
    '''
    groups:     tuple  # static; the FULL canonical partition (validate_sharing form)
    row_splits: tuple  # static; per nontrivial group: cumulative row offsets of the stacked S^T blocks, len=k+1
    centers:    tuple  # per nontrivial group: tuple of center cores H_i, elm_shape=C+(rLi, nUi, rR(i+1))
    svd_U:      tuple  # per nontrivial group: left factor of thin SVD of M_g,  shape=C+(sum_nD, q)
    svd_s:      tuple  # per nontrivial group: singular values of M_g = the group spectrum s_g, C+(q,)
    svd_Vt:     tuple  # per nontrivial group: right factor of thin SVD of M_g, shape=C+(q, n_g)


def fv_shared_frame_data(
        frame_data: typ.Tuple[
            typ.Sequence[NDArray],  # up_tucker_cores. len=d, elm_shape=C+(nUi, Ni)
            typ.Sequence[NDArray],  # down_tt_cores.   len=d, elm_shape=C+(rLi, nDi, rR(i+1))
            typ.Sequence[NDArray],  # left_tt_cores.   len=d, elm_shape=C+(rLi, nUi, rL(i+1))
            typ.Sequence[NDArray],  # right_tt_cores.  len=d, elm_shape=C+(rRi, nUi, rR(i+1))
        ],
        groups:     typ.Tuple[typ.Tuple[int, ...], ...],  # static; canonical partition (validate_sharing)
) -> T3SharedFrameData:
    '''Derive the shared-geometry companion from an **orthogonal** frame.

    Three steps, all exact by construction rather than by tolerance:

    1. **Re-sweep** the frame's stored left chain
       (``tt_right_orthogonalize(left_tt_cores, return_variation_cores=True)``) -- the SAME
       computation, on the SAME arrays, that produced the center cores ``H_i`` at frame
       construction, so the centers here reproduce the constructed ones exactly.
    2. Per mode of each nontrivial group, ``S_i^T = <O_i, H_i>`` against the frame's STORED
       down core (``S_i S_i^T = Gamma_i`` and ``W2_i = S_i O2_i`` hold by the construction's
       own factorization; no re-SVD, so no sign/degenerate-block hazards).
    3. Per nontrivial group, one thin (batched) SVD of the stacked
       ``M_g = concat_i(S_i^T)``, shape ``C + (sum_i nD_i, n_g)``.

    Requires an orthogonal frame (the identities above presume the frame's gauges); the
    shared geometry enforces that in safe mode at its check sites -- this backend function is
    check-free. ``svd_s`` is the group spectrum: the singular values of the concatenated
    matricizations ``[X_(i1) | ... | X_(ik)]`` of the represented tensor. Stack-aware (frame
    stack ``C`` rides every array); ragged path only (uniform twin deferred to the uniform
    slices). Design + measurements: ``dev/shared_t3_math.tex`` (the tilted subspace and its
    SVD-not-normal-equations remark).

    Examples
    --------
    The companion of a shared frame: the centers reproduce the construction's own centers
    exactly, and ``svd_s`` is the concatenated-matricization spectrum of the tensor:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.frame_variations_format as bvf
    >>> import t3toolbox.backend.sharing as sharing
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 6, 6), (3, 3, 3), (1, 2, 2, 1))
    >>> tk, tt = x.data
    >>> x = t3.TuckerTensorTrain((tk[0],) * 3, tt)                # tie all three modes
    >>> frame, variations = bvf.t3_orthogonal_representations(x)
    >>> groups = sharing.validate_sharing((0, 0, 0), x.shape)
    >>> sfd = sharing.fv_shared_frame_data(frame.data, groups)
    >>> print(len(sfd.centers[0]), sfd.svd_s[0].shape, sfd.row_splits[0])
    3 (3,) (0, 2, 5, 7)
    >>> print(all(np.array_equal(np.asarray(H), np.asarray(V))
    ...           for H, V in zip(sfd.centers[0], variations.tt_variations)))
    True
    >>> Xd = np.asarray(x.to_dense())
    >>> mats = [np.moveaxis(Xd, ii, 0).reshape(6, -1) for ii in range(3)]
    >>> s_dense = np.linalg.svd(np.concatenate(mats, axis=1), compute_uv=False)
    >>> print(bool(np.allclose(np.asarray(sfd.svd_s[0]), s_dense[:3])))
    True
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame_data
    use_jax = tree_contains_jax(frame_data)
    xnp, _, _ = get_backend(False, use_jax)

    # Re-sweep: the centers H_i, by the construction's own computation on the frame's stored
    # left chain (exact reproduction; the zipper H_i = Z_i R_i is a measured-viable GEMM-only
    # alternative if the SVD-based sweep ever shows up in a GPU profile).
    _, HH = tt_orthogonalization.tt_right_orthogonalize(left_tt_cores, return_variation_cores=True)

    centers, row_splits, svd_U, svd_s, svd_Vt = [], [], [], [], []
    for group in nontrivial_groups(groups):
        SS_gT = tuple(
            xnp.einsum('...axb,...aub->...xu', down_tt_cores[ii], HH[ii])  # S_i^T, C+(nDi, nU)
            for ii in group)
        splits = (0,)
        for S_T in SS_gT:
            splits = splits + (splits[-1] + S_T.shape[-2],)
        M = xnp.concatenate(SS_gT, axis=-2)                                # C+(sum nD, nU)
        U_M, s_g, Vt_M = xnp.linalg.svd(M, full_matrices=False)
        centers.append(tuple(HH[ii] for ii in group))
        row_splits.append(splits)
        svd_U.append(U_M)
        svd_s.append(s_g)
        svd_Vt.append(Vt_M)

    return T3SharedFrameData(groups=tuple(groups), row_splits=tuple(row_splits),
                             centers=tuple(centers), svd_U=tuple(svd_U),
                             svd_s=tuple(svd_s), svd_Vt=tuple(svd_Vt))


def fv_share_tucker_variations(
        variations_data: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations. len=d, elm_shape=(K+C)+(nDi, Ni)
            typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=(K+C)+(rLi, nUi, rRi)
        ],
        shared_data:    'T3SharedFrameData',  # the frame's companion (fv_shared_frame_data)
        rcond:          float = None,         # relative clip on the group spectrum; None -> dtype eps * max dim
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_variations, tied within groups (exact row blocks of one solve)
    typ.Tuple[NDArray, ...],  # tt_variations, untouched
]:
    '''The tied post-pass of the shared MANIFOLD geometry: orthogonally project gauged Tucker
    variation coordinates onto the tied subspace ``{V_i = S_i^T Udot, common gauged Udot}``.

    Per nontrivial group, one clipped least-squares solve against the companion's stacked-``S``
    SVD (``Udot = M_g^+ [V_{i_1}; ...; V_{i_k}]``, sensitivity ``kappa_g`` -- never the
    ``kappa_g^2`` normal equations) followed by the exact redistribution ``V_i <- S_i^T Udot``
    (the row blocks of ``M_g Udot``). Gauge is preserved identically (each ``S_i^T Udot`` is
    gauged when ``Udot`` is, and ``Udot`` inherits the gauge from gauged inputs); the
    projection is idempotent and fixes exactly-tied inputs. TT variations are untouched
    (sharing constrains only the Tucker factors). The clip makes the solve well-defined
    (minimum-norm) at rank-deficient points -- which zero-padded continuation restarts visit by
    construction, where the gated directions correctly receive zero.

    Broadcasting: the companion carries the frame stack ``C``; the variations carry ``K + C``
    -- the library-wide frame-inner layout makes the solve broadcast for free. Verified against
    the dense orthogonal projection onto the tied tangent subspace (design round, 1.6e-13;
    promoted to the permanent tests). Ragged path only (uniform twin deferred).

    Examples
    --------
    Gauge a raw direction at a tied frame, then tie it; the post-pass is idempotent and the
    result stays gauged:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.fv_conversions as fvc
    >>> import t3toolbox.backend.tv_operations as tvo
    >>> import t3toolbox.backend.sharing as sharing
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 6), (3, 3), (1, 2, 1))
    >>> tk, tt = x.data
    >>> frame_d, _ = fvc.t3_orthogonal_representations(((tk[0], tk[0]), tt))
    >>> groups = sharing.validate_sharing((0, 0), (6, 6))
    >>> sfd = sharing.fv_shared_frame_data(frame_d, groups)
    >>> raw = (tuple(np.random.randn(O.shape[-2], U.shape[-1])
    ...              for O, U in zip(frame_d[1], frame_d[0])),
    ...        tuple(np.random.randn(*H.shape[-3:]) for H in frame_d[2]))
    >>> tied = tvo.tv_orthogonal_gauge_projection(frame_d, raw, shared_data=sfd)
    >>> tied2 = sharing.fv_share_tucker_variations(tied, sfd)
    >>> print(bool(np.allclose(np.asarray(tied[0][0]), np.asarray(tied2[0][0]))))   # idempotent
    True
    >>> print(float(tvo.tv_gauge_residual(frame_d, tied)) < 1e-12)                  # still gauged
    True
    '''
    tucker_variations, tt_variations = variations_data
    use_jax = tree_contains_jax(variations_data)
    xnp, _, _ = get_backend(False, use_jax)

    new_tucker = list(tucker_variations)
    for gi, group in enumerate(nontrivial_groups(shared_data.groups)):
        U_M, s_g, Vt_M = shared_data.svd_U[gi], shared_data.svd_s[gi], shared_data.svd_Vt[gi]
        splits = shared_data.row_splits[gi]
        eff_rcond = (float(np.finfo(s_g.dtype).eps) * max(splits[-1], Vt_M.shape[-1])
                     if rcond is None else rcond)
        keep = s_g > eff_rcond * s_g[..., :1]                 # relative clip against s_{g,1}
        s_inv = xnp.where(keep, 1.0 / xnp.where(keep, s_g, 1.0), 0.0)   # branch-free clipped pinv

        Vstack = xnp.concatenate([tucker_variations[ii] for ii in group], axis=-2)  # (K+C)+(sum nD, N)
        t = xnp.einsum('...xw,...xn->...wn', U_M, Vstack)     # U_M^T Vstack, (K+C)+(q, N)
        t = xnp.einsum('...w,...wn->...wn', s_inv, t)         # clipped pinv apply
        Udot = xnp.einsum('...wy,...wn->...yn', Vt_M, t)      # the tied ambient direction, (K+C)+(n_g, N)
        # R = M_g @ Udot: the tied coordinates of every group mode in one stack
        r = xnp.einsum('...wy,...yn->...wn', Vt_M, Udot)
        r = xnp.einsum('...w,...wn->...wn', s_g, r)
        R = xnp.einsum('...xw,...wn->...xn', U_M, r)          # (K+C)+(sum nD, N)
        for jj, ii in enumerate(group):
            new_tucker[ii] = R[..., splits[jj]:splits[jj + 1], :]
    return tuple(new_tucker), tuple(tt_variations)


def fv_mean_tucker_variations(
        variations_data: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations. len=d, elm_shape=(K+C)+(ni, Ni)
            typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=(K+C)+(rLi, ni, rRi)
        ],
        groups:         typ.Tuple[typ.Tuple[int, ...], ...],  # static; canonical (validate_sharing)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_variations; ONE mean array per group
    typ.Tuple[NDArray, ...],  # tt_variations, untouched
]:
    '''The tied post-pass of the shared COREWISE geometry: orthogonally project raw core
    perturbations onto the tied subspace ``{dU_i all equal within each group}``.

    On the corewise geometry the coordinates are raw factor copies and the metric is Euclidean
    on the core entries, so the projection is the per-group arithmetic mean, assigned as ONE
    array per group (the additive corewise retraction then preserves tying exactly). Computed
    in the drift form (``ref + mean of differences``) so an exactly-tied group is a bitwise
    fixed point. TT variations untouched. This is NOT the manifold geometry's post-pass -- the
    two geometries tie by orthogonal projection in their OWN metrics on their OWN coordinates,
    and the formulas differ (see :py:func:`fv_share_tucker_variations`).
    '''
    tucker_variations, tt_variations = variations_data
    new_tucker = list(tucker_variations)
    for group in nontrivial_groups(groups):
        ref = tucker_variations[group[0]]
        drift = tucker_variations[group[1]] - ref
        for ii in group[2:]:
            drift = drift + (tucker_variations[ii] - ref)
        mean = ref + drift / len(group)
        for ii in group:
            new_tucker[ii] = mean             # the SAME array object at every group mode
    return tuple(new_tucker), tuple(tt_variations)


if jax_available:
    import jax

    # The companion is a jax pytree: the per-group arrays (centers + SVD factors) are LEAVES
    # (they flow as traced data beside the frame), the partition and the static row offsets are
    # aux_data -- so a companion crossing a jit boundary keeps a stable, static structure key.
    jax.tree_util.register_pytree_node(
        T3SharedFrameData,
        lambda s: ((s.centers, s.svd_U, s.svd_s, s.svd_Vt), (s.groups, s.row_splits)),
        lambda aux, ch: T3SharedFrameData(groups=aux[0], row_splits=aux[1], centers=ch[0],
                                          svd_U=ch[1], svd_s=ch[2], svd_Vt=ch[3]),
    )
