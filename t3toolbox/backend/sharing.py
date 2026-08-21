# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Shared Tucker factors (SF-T3): the sharing partition, the tied-factors checkers, and the
per-frame companion.

``validate_sharing`` canonicalizes a per-mode label spec into the static ``groups`` form;
``t3_sharing_residual`` / ``t3_tucker_factors_shared`` are the non-enforcing tied-factors
checkers (the safe-mode precondition behind the shared operations); ``t3_tie_tucker_factors``
is the plain per-group mean (drift repair for nearly-tied POINTS -- never the metric
projection of a tangent, which is geometry-specific and lives with the shared geometry).
``fv_shared_frame_data`` derives the :py:class:`SharedFrameData` companion from an
orthogonal frame -- the per-group center cores and the thin SVD of the stacked ``S`` factors
that the shared geometry's projection, retraction, and spectrum diagnostics consume.

A shared T3 is an ordinary Tucker tensor train whose Tucker factors are equal within
user-specified groups of modes -- the SF-ETT decomposition of Molozhavenko & Rakhuba (2026),
"Optimization on the extended tensor-train manifold with shared factors" (Comput. Appl. Math.
45:221), generalized to an arbitrary partition of the modes into sharing groups. The
shared-factor format originates with SF-Tucker: Peshekhonov, Arzhantsev & Rakhuba (2024),
"Training a Tucker model with shared factors: a Riemannian optimization approach"
(AISTATS, PMLR 238).
"""
import numpy as np
import typing as typ
from dataclasses import dataclass

import t3toolbox.backend.tt_orthogonalization as tt_orthogonalization
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ufv_masking as ufv_masking
from t3toolbox.backend.common import *

__all__ = [
    'validate_sharing',
    'nontrivial_groups',
    'canonical_groups',
    'groups_to_labels',
    't3_sharing_residual',
    't3_tucker_factors_shared',
    't3_tie_tucker_factors',
    'ut3_tie_tucker_factors',
    'SharedFrameData',
    'fv_shared_frame_data',
    'fv_share_tucker_variations',
    'fv_share_tucker_variations_corewise',
    'fv_tied_variations_residual',
    'fv_tied_ambient_directions',
    'ut3_sharing_residual',
    'ut3_tucker_factors_shared',
    'ufv_shared_frame_data',
    'ufv_share_tucker_variations',
    'ufv_share_tucker_variations_corewise',
    'ufv_tied_variations_residual',
    't3_tucker_weights_sharing_residual',
    't3_tucker_weights_shared',
    'ut3_tucker_weights_sharing_residual',
    'ut3_tucker_weights_shared',
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
    shape = tuple(int(N) for N in shape)
    groups = _groups_from_labels(sharing, len(shape))
    for group in groups:
        sizes = tuple(shape[ii] for ii in group)
        if len(set(sizes)) > 1:
            raise ValueError(
                'modes in a sharing group must have equal mode sizes (one shared factor needs '
                'one ambient dimension); group %r has sizes %r' % (group, sizes))
    return groups


def _groups_from_labels(
        sharing:    typ.Sequence,   # len=d, static; one hashable group label per mode
        d:          int,            # number of modes
) -> typ.Tuple[typ.Tuple[int, ...], ...]:  # canonical groups (no mode-size check -- for shape-free data)
    '''The label-grouping core of :py:func:`validate_sharing`, without the equal-mode-sizes check --
    for objects that carry no mode sizes (the edge-weight checkers: weights live on internal edges
    only). Length and hashability are still structural errors.'''
    sharing = tuple(sharing)
    if len(sharing) != d:
        raise ValueError(
            'sharing must assign one group label per mode: len(sharing) = %d != %d = number of modes'
            % (len(sharing), d))
    modes_by_label = {}                       # insertion-ordered -> groups ordered by first mode
    for ii, label in enumerate(sharing):
        try:
            hash(label)
        except TypeError:
            raise ValueError(
                'sharing labels must be hashable; got %r at mode %d' % (label, ii))
        modes_by_label.setdefault(label, []).append(ii)
    return tuple(tuple(modes) for modes in modes_by_label.values())


def canonical_groups(
        sharing:  typ.Optional[typ.Sequence],   # len=d group labels, or None
        shape:    typ.Tuple[int, ...],          # the mode sizes (group members must agree)
) -> typ.Tuple[typ.Tuple[int, ...], ...]:       # canonical partition, () when there is nothing to tie
    """Normalize a ``sharing`` spec to a canonical partition, collapsing "nothing is tied" to ``()``.

    ``None`` and an all-singleton partition both give ``()``, so a consumer that stores the partition
    has exactly ONE representation of the unshared case -- which matters when the partition is part of
    an object's value identity (a jit ``aux_data`` key). :py:func:`validate_sharing` +
    :py:func:`nontrivial_groups` do the work; this just names the pairing, which callers otherwise
    open-code."""
    if sharing is None:
        return ()
    all_groups = validate_sharing(sharing, shape)
    return all_groups if nontrivial_groups(all_groups) else ()


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


def groups_to_labels(
        groups: typ.Tuple[typ.Tuple[int, ...], ...],  # static; canonical form (validate_sharing)
) -> typ.Tuple[int, ...]:                             # len=d, static; the per-mode label spec
    '''The inverse of :py:func:`validate_sharing`: per-mode integer labels for a canonical
    partition (``validate_sharing(groups_to_labels(g), shape) == g``).

    Examples
    --------
    >>> import t3toolbox.backend.sharing as sharing
    >>> sharing.groups_to_labels(((0, 2), (1,)))
    (0, 1, 0)
    '''
    d = sum(len(group) for group in groups)
    labels = [None] * d
    for gi, group in enumerate(groups):
        for ii in group:
            labels[ii] = gi
    return tuple(labels)


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


def t3_tie_tucker_factors(
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
    >>> tk2, tt2 = sharing.t3_tie_tucker_factors(x.data, (0, 0, 1))
    >>> print(tk2[0] is tk2[1], tt2 is tt)
    True True
    >>> print(bool(np.allclose(np.asarray(tk2[0]), (np.asarray(tk[0]) + np.asarray(tk[1])) / 2)))
    True
    >>> print(float(sharing.t3_sharing_residual((tk2, tt2), (0, 0, 1))))
    0.0

    Already-tied input comes back with unchanged factor values:

    >>> tk3, _ = sharing.t3_tie_tucker_factors(((tk[0], tk[0], tk[2]), tt), (0, 0, 1))
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


def ut3_tie_tucker_factors(
        data:       typ.Tuple[
            NDArray,             # tucker_supercore, shape=(d,)+stack+(n,N)
            NDArray,             # tt_supercore,     shape=(d,)+stack+(r,n,r)
            typ.Sequence[int],   # shape, static int tuple
            typ.Tuple[NDArray, NDArray],  # (tucker_edge_mask, tt_edge_mask), HOST bool, static
        ],
        sharing:    typ.Sequence,   # len=d, static; one hashable group label per mode
) -> typ.Tuple[
    NDArray,                        # tucker_supercore with each group's slices set to the group mean
    NDArray,                        # tt_supercore, untouched
    typ.Sequence[int],              # shape, untouched
    typ.Tuple[NDArray, NDArray],    # masks, untouched (the tie changes values, never ranks)
]:
    '''The uniform twin of :py:func:`t3_tie_tucker_factors`: tie the Tucker factors exactly, by
    per-group arithmetic averaging of the supercore slices.

    Use this to repair **numerical drift** away from equal factors without round-tripping through the
    ragged layer -- e.g. after many low-precision first-order steps, where an exactly-tied start can
    creep apart. TT cores, shape and masks are untouched: averaging changes factor *values*, never
    ranks, and a group's Tucker rank masks are required equal anyway (structural, raises otherwise).

    **Garbage-transparent, so no masking is needed.** A group's rank masks are equal, so every real
    slot is real at every mode of the group and the mean of the real content uses only real values;
    the padding averages to other padding, which is don't-care either way
    (``docs/uniform_equivalence_contract.md``).

    As in the ragged twin the mean is computed as ``B_ref + mean(B_i - B_ref)``, so an exactly-tied
    group is a **bitwise fixed point** for any group size. Unlike the ragged twin there is no array
    identity to preserve -- a supercore holds one slice per mode -- so ties are exact by *value*, which
    is what the uniform checkers compare.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.backend.sharing as sharing
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 2, 2, 1))
    >>> u = ut3.UniformTuckerTensorTrain.from_t3(x)                      # untied
    >>> print(bool(sharing.ut3_sharing_residual(u.data, (0, 0, 1)) > 0.1))
    True
    >>> tied = sharing.ut3_tie_tucker_factors(u.data, (0, 0, 1))
    >>> print(float(sharing.ut3_sharing_residual(tied, (0, 0, 1))))
    0.0

    Masks and TT cores come back untouched, and re-tying is a bitwise fixed point:

    >>> print(bool(np.array_equal(tied[1], u.data[1])), tied[2] == u.data[2])
    True True
    >>> again = sharing.ut3_tie_tucker_factors(tied, (0, 0, 1))
    >>> print(bool(np.array_equal(np.asarray(again[0]), np.asarray(tied[0]))))
    True
    '''
    tucker_supercore, tt_supercore, shape, masks = data
    groups = validate_sharing(sharing, shape)
    _validate_group_tucker_rank_masks(masks[0], groups)

    xnp, _, _ = get_backend(True, tree_contains_jax(data[:2]))
    slices = [tucker_supercore[ii] for ii in range(len(shape))]
    for group in groups:
        if len(group) < 2:
            continue
        B_ref = slices[group[0]]
        drift = slices[group[1]] - B_ref
        for ii in group[2:]:
            drift = drift + (slices[ii] - B_ref)
        mean = B_ref + drift / len(group)
        for ii in group:
            slices[ii] = mean
    return xnp.stack(slices), tt_supercore, shape, masks


@dataclass(frozen=True, eq=False)  # eq=False -> identity hash/eq (array fields; value-eq is ambiguous)
class SharedFrameData:
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

    **What ``s_g`` is** (representation-independent -- a property of the represented tensor
    ``T`` and the partition alone): the singular values of the concatenated matricization
    ``[T_(i1) | ... | T_(ik)]`` over the group's modes; equivalently
    ``s_g^2 = eig(sum_i Gamma_i)`` (the summed mode Grams), equivalently the singular values
    of the Jacobian of ``T`` with respect to a gauged tied motion of the shared factor -- the
    exact analog of what a per-mode Tucker spectrum is to an unshared factor. Note the scale:
    every mode carries the full norm, so ``sum_j s_gj^2 = k * ||T||^2`` (a group of ``k``
    modes inflates the spectrum by ``sqrt(k)``; the factor cancels in every condition-number
    ratio). Cf. Peshekhonov, Arzhantsev & Rakhuba (2024, SF-Tucker) and Molozhavenko &
    Rakhuba (2026, SF-ETT), whose algorithms compute this same object.
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
) -> SharedFrameData:
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
    stack ``C`` rides every array). The uniform twin is
    :py:func:`ufv_shared_frame_data`. Design + measurements: ``docs/contributor/sharing_internals.md``
    (the tilted subspace and the SVD-not-normal-equations measurement).

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

    return SharedFrameData(groups=tuple(groups), row_splits=tuple(row_splits),
                             centers=tuple(centers), svd_U=tuple(svd_U),
                             svd_s=tuple(svd_s), svd_Vt=tuple(svd_Vt))


def fv_share_tucker_variations(
        variations_data: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations. len=d, elm_shape=(K+C)+(nDi, Ni)
            typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=(K+C)+(rLi, nUi, rRi)
        ],
        shared_data:    'SharedFrameData',  # the frame's companion (fv_shared_frame_data)
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
    promoted to the permanent tests). The uniform twin is :py:func:`ufv_share_tucker_variations`.

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
        Udot = _tied_solve(shared_data, gi, tucker_variations, rcond, xnp)
        # R = M_g @ Udot: the tied coordinates of every group mode in one stack
        U_M, s_g, Vt_M = shared_data.svd_U[gi], shared_data.svd_s[gi], shared_data.svd_Vt[gi]
        splits = shared_data.row_splits[gi]
        r = xnp.einsum('...wy,...yn->...wn', Vt_M, Udot)
        r = xnp.einsum('...w,...wn->...wn', s_g, r)
        R = xnp.einsum('...xw,...wn->...xn', U_M, r)          # (K+C)+(sum nD, N)
        for jj, ii in enumerate(group):
            new_tucker[ii] = R[..., splits[jj]:splits[jj + 1], :]
    return tuple(new_tucker), tuple(tt_variations)


def _tied_solve(
        shared_data:        'SharedFrameData',
        gi:                 int,      # index into the companion's nontrivial-group tuples
        tucker_variations:  typ.Sequence[NDArray],  # len=d, elm_shape=(K+C)+(nDi, Ni)
        rcond:              typ.Optional[float],   # relative clip; None -> dtype eps * max dim
        xnp,                                        # the array backend (numpy / jax.numpy)
) -> NDArray:  # Udot: the common gauged ambient direction, (K+C)+(n_g, N)
    '''The clipped least-squares solve ``Udot = M_g^+ [stacked V_i]`` against the companion's
    stacked-``S`` SVD (minimum-norm at rank-deficient points; sensitivity ``kappa_g``).'''
    group = nontrivial_groups(shared_data.groups)[gi]
    U_M, s_g, Vt_M = shared_data.svd_U[gi], shared_data.svd_s[gi], shared_data.svd_Vt[gi]
    splits = shared_data.row_splits[gi]
    eff_rcond = (float(np.finfo(s_g.dtype).eps) * max(splits[-1], Vt_M.shape[-1])
                 if rcond is None else rcond)
    keep = s_g > eff_rcond * s_g[..., :1]                     # relative clip against s_{g,1}
    s_inv = xnp.where(keep, 1.0 / xnp.where(keep, s_g, 1.0), 0.0)   # branch-free clipped pinv

    Vstack = xnp.concatenate([tucker_variations[ii] for ii in group], axis=-2)  # (K+C)+(sum nD, N)
    t = xnp.einsum('...xw,...xn->...wn', U_M, Vstack)         # U_M^T Vstack, (K+C)+(q, N)
    t = xnp.einsum('...w,...wn->...wn', s_inv, t)             # clipped pinv apply
    return xnp.einsum('...wy,...wn->...yn', Vt_M, t)          # Udot, (K+C)+(n_g, N)


def fv_tied_ambient_directions(
        variations_data: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations. len=d, elm_shape=(K+C)+(nDi, Ni)
            typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=(K+C)+(rLi, nUi, rRi)
        ],
        shared_data:    'SharedFrameData',  # the frame's companion (fv_shared_frame_data)
        rcond:          float = None,         # relative clip on the group spectrum; None -> dtype eps * max dim
) -> typ.Tuple[NDArray, ...]:  # per NONTRIVIAL group: Udot, elm_shape=(K+C)+(n_g, N)
    '''Recover each group's common gauged ambient direction ``Udot`` from (tied) Tucker
    variation coordinates -- the clipped least-squares solve of
    :py:func:`fv_share_tucker_variations`, returning ``Udot`` itself instead of the
    redistributed coordinates. Exact on exactly-tied input (``V_i = S_i^T Udot``); on untied
    input it returns the least-squares fit (the tied projection's ambient direction). The
    shared retraction uses this to build the TIED doubled-rank embedding
    (``[U_g | Udot]`` per group, with the center cores as the paired blocks).'''
    tucker_variations, _ = variations_data
    use_jax = tree_contains_jax(variations_data)
    xnp, _, _ = get_backend(False, use_jax)
    return tuple(_tied_solve(shared_data, gi, tucker_variations, rcond, xnp)
                 for gi in range(len(nontrivial_groups(shared_data.groups))))


def fv_share_tucker_variations_corewise(
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


############################################################
def fv_tied_variations_residual(
        variations:     typ.Tuple[
            typ.Sequence[NDArray],  # tucker_variations. len=d, elm_shape=K+C+(nDi, Ni)
            typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=K+C+(rLi, nUi, rR(i+1))
        ],
        shared_data:    'SharedFrameData',  # the frame's companion (fv_shared_frame_data)
        rcond:          typ.Optional[float] = None,  # relative clip on the group spectrum; None -> dtype eps
) -> NDArray:  # shape = K + C; relative deviation per stack element (0 == already tied)
    '''How far a tangent's coordinates are from the TIED tangent subspace, per stack element.

    One **global Frobenius** ratio: ``||Pi_sh(V) - V||_F / ||V||_F``, with both norms taken over all
    ``d`` Tucker variation cores at once (sum of squares, then one square root) and the stack axes
    ``K + C`` kept. Only the Tucker variations can be untied -- the TT variations are unrestricted --
    so they alone enter the norm. Zero reference with a nonzero deviation gives ``inf``, branch-free,
    matching :py:func:`t3_sharing_residual`.

    This is the non-enforcing checker behind the shared geometry's TIED-tangent precondition. It costs
    one tied projection, which is strictly cheaper than the retraction it guards.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.shared_geometry as sg
    >>> import t3toolbox.backend.sharing as sharing
    >>> np.random.seed(0)
    >>> sh = (0, 0, 1)
    >>> x = t3.TuckerTensorTrain.randn((6, 6, 5), (2, 2, 2), (1, 2, 2, 1)).share(sh)
    >>> geom = sg.shared_manifold(sh)
    >>> frame = geom.frame(x)
    >>> companion = geom.shared_frame_data(frame)

    A tangent produced by the shared geometry is already tied; a raw one from the base geometry is not:

    >>> tied = geom.randn(frame)
    >>> print(bool(sharing.fv_tied_variations_residual(tied.variations.data, companion) < 1e-12))
    True
    >>> raw = t3m.MANIFOLD.randn(frame)
    >>> print(bool(sharing.fv_tied_variations_residual(raw.variations.data, companion) > 0.1))
    True
    '''
    tucker_variations, _ = variations
    tied, _ = fv_share_tucker_variations(variations, shared_data, rcond=rcond)
    xnp, _, _ = get_backend(False, tree_contains_jax((tucker_variations, shared_data.svd_s)))

    num_sq = den_sq = None
    for V, T in zip(tucker_variations, tied):
        diff = T - V
        n_i = xnp.sum(diff * diff, axis=(-2, -1))     # sum the CORE axes, keep the K+C stack
        d_i = xnp.sum(V * V, axis=(-2, -1))
        num_sq = n_i if num_sq is None else num_sq + n_i
        den_sq = d_i if den_sq is None else den_sq + d_i
    num, den = xnp.sqrt(num_sq), xnp.sqrt(den_sq)
    pos = den > 0.0
    return xnp.where(pos, num / xnp.where(pos, den, 1.0),   # branch-free zero guard
                     xnp.where(num > 0.0, xnp.inf, 0.0))


############################################################
##########    Uniform-layer twins (supercores)    ##########
############################################################


def _validate_group_tucker_rank_masks(
        tucker_mask:    NDArray,                              # HOST bool, shape=(d,)+stack+(n,)
        groups:         typ.Tuple[typ.Tuple[int, ...], ...],  # static; canonical (validate_sharing)
):
    '''Structural: tied factors need equal Tucker rank masks within each group (per stack element).'''
    for group in nontrivial_groups(groups):
        for jj in group[1:]:
            if not np.array_equal(tucker_mask[group[0]], tucker_mask[jj]):
                raise ValueError(
                    'Tucker rank masks must be equal within a sharing group (tied factors have one '
                    'shared rank); group %r differs at modes %d and %d' % (group, group[0], jj))


def ut3_sharing_residual(
        data:       typ.Tuple[
            NDArray,             # tucker_supercore, shape=(d,)+stack+(n,N)
            NDArray,             # tt_supercore,     shape=(d,)+stack+(r,n,r)
            typ.Sequence[int],   # shape, static int tuple
            typ.Tuple[NDArray, NDArray],  # (tucker_edge_mask, tt_edge_mask), HOST bool, static
        ],
        sharing:    typ.Sequence,   # len=d, static; one hashable group label per mode
) -> NDArray:  # shape = stack_shape; max relative factor deviation per stack element (0 == exactly tied)
    '''The uniform twin of :py:func:`t3_sharing_residual`: per stack element, the max over groups and
    group modes of ``||B_i - B_ref||_F / ||B_ref||_F`` on the MASKED factor content (padding is
    don't-care garbage, so it is zeroed before comparing -- two elements tied on their real content
    are tied regardless of their padding). Structural problems -- invalid partition, unequal Tucker
    rank masks within a group -- raise unconditionally.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.backend.sharing as sharing
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 2, 2, 1))
    >>> tk, tt = x.data
    >>> tied = t3.TuckerTensorTrain((tk[0], tk[0], tk[2]), tt)   # tie modes 0, 1
    >>> u = ut3.UniformTuckerTensorTrain.from_t3(tied)
    >>> print(float(sharing.ut3_sharing_residual(u.data, (0, 0, 1))))
    0.0
    >>> u2 = ut3.UniformTuckerTensorTrain.from_t3(x)             # untied
    >>> print(bool(sharing.ut3_sharing_residual(u2.data, (0, 0, 1)) > 0.1))
    True
    '''
    shape = data[2]
    tucker_mask, _tt_mask = data[3]
    groups = validate_sharing(sharing, shape)
    _validate_group_tucker_rank_masks(tucker_mask, groups)

    masked_tucker, _ = ut3_masking.ut3_apply_masks(data)
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    devs = []
    for group in groups:
        if len(group) < 2:
            continue
        B_ref = masked_tucker[group[0]]
        denom = xnp.sqrt(xnp.sum(B_ref * B_ref, axis=(-2, -1)))   # keep stack
        for ii in group[1:]:
            diff = masked_tucker[ii] - B_ref
            num = xnp.sqrt(xnp.sum(diff * diff, axis=(-2, -1)))
            pos = denom > 0.0
            devs.append(xnp.where(pos, num / xnp.where(pos, denom, 1.0),   # branch-free zero guard
                                  xnp.where(num > 0.0, xnp.inf, 0.0)))
    if not devs:                                                  # all-singleton: trivially tied
        return xnp.zeros(masked_tucker.shape[1:-2])
    return xnp.max(xnp.stack(devs), axis=0)   # max over the checks, keep stack_shape


def ut3_tucker_factors_shared(
        data:       typ.Tuple[
            NDArray,             # tucker_supercore, shape=(d,)+stack+(n,N)
            NDArray,             # tt_supercore,     shape=(d,)+stack+(r,n,r)
            typ.Sequence[int],   # shape, static int tuple
            typ.Tuple[NDArray, NDArray],  # (tucker_edge_mask, tt_edge_mask), HOST bool, static
        ],
        sharing:    typ.Sequence,   # len=d, static; one hashable group label per mode
        rtol:       float = 1e-9,   # relative tolerance on the factor deviation
) -> NDArray:  # bool array, shape = stack_shape (scalar/0-d when unstacked)
    '''True (per stack element) where the MASKED Tucker factors are tied within every sharing group --
    the uniform twin of :py:func:`t3_tucker_factors_shared` (the boolean form of
    :py:func:`ut3_sharing_residual`; a non-enforcing checker).'''
    return ut3_sharing_residual(data, sharing) <= rtol


def ufv_shared_frame_data(
        frame_data,  # UT3Frame .data: (up_sc, down_sc, left_sc, right_sc, shape, (4 masks)); stack = C
        groups:     typ.Tuple[typ.Tuple[int, ...], ...],  # static; canonical partition (validate_sharing)
) -> 'SharedFrameData':
    '''The uniform twin of :py:func:`fv_shared_frame_data`: the IDENTICAL polymorphic derivation
    (the right re-sweep, ``S_i^T = <O_i, H_i>`` batched over the mode axis, one batched thin SVD per
    group of the statically-gathered concatenation) on the frame's supercores.

    The re-sweep runs on the frame's stored left supercore AS STORED -- deliberately NOT re-masked:
    the companion's exactness rests on reproducing the construction's own sweep on the SAME arrays
    (bit-identical, so the ``<O_i, H_i>`` pairing inherits the construction's SVD gauge). The stored
    padding slots carry the construction's arbitrary orthonormal completions (the equivalence
    contract's honest caveat); masking them first changes the sweep's SVD sign choices and BREAKS the
    pairing (measured: a flipped bond column destroys the group spectrum). The padded rows of each
    ``S_i^T`` vanish to roundoff anyway (completion rows are orthogonal to the center's row space),
    so every consumer stays padding-transparent. Contract: the frame's chains must be the polymorphic
    sweep's own output at these padded dims (a frame built by ``ut3_orthogonal_representations``); a
    ``t3frame_to_ut3frame``-packed RAGGED frame is NOT guaranteed (the padded re-sweep can flip signs
    against the unpadded construction).

    Container reuse: the returned :py:class:`SharedFrameData` holds SUPERCORE SLICES at the padded
    dims (frame stack ``C`` leading; ``row_splits`` are multiples of the padded ``nD``).
    '''
    up_sc, down_sc, left_sc, right_sc = frame_data[0], frame_data[1], frame_data[2], frame_data[3]
    return fv_shared_frame_data((up_sc, down_sc, left_sc, right_sc), groups)


def ufv_share_tucker_variations(
        variations_data,   # UT3Variations .data: (tkv_sc, ttv_sc, shape, (4 masks)); stack = K + C
        shared_data:    'SharedFrameData',  # the frame's UNIFORM companion (ufv_shared_frame_data)
        rcond:          float = None,         # relative clip on the group spectrum; None -> dtype eps * max dim
):  # -> tied variations .data (same masks; padded rows come back exactly zero)
    '''The uniform twin of :py:func:`fv_share_tucker_variations` (the manifold tied post-pass):
    mask the variation supercores, delegate to the polymorphic ragged solve on the mode slices,
    re-stack. The companion's masked ``U_M`` zeroes every padded row of the redistribution, so the
    output is masked-clean and the masks are unchanged.'''
    tkv, ttv = ufv_masking.ufv_apply_variations_masks(variations_data)
    shape, masks = variations_data[2], variations_data[3]
    use_jax = tree_contains_jax(variations_data[:2])
    xnp, _, _ = get_backend(True, use_jax)
    new_tk_slices, _ = fv_share_tucker_variations((tkv, ttv), shared_data, rcond=rcond)
    return xnp.stack(list(new_tk_slices), axis=0), ttv, shape, masks


def ufv_share_tucker_variations_corewise(
        variations_data,   # UT3Variations .data at the corewise (U,G,G,G) frame; stack = K + C
        groups:         typ.Tuple[typ.Tuple[int, ...], ...],  # static; canonical (validate_sharing)
):  # -> tied variations .data (per-group drift-form mean; same masks)
    '''The uniform twin of :py:func:`fv_share_tucker_variations_corewise` (the corewise tied post-pass):
    mask, delegate to the polymorphic per-group drift-form mean on the mode slices, re-stack.'''
    tkv, ttv = ufv_masking.ufv_apply_variations_masks(variations_data)
    shape, masks = variations_data[2], variations_data[3]
    use_jax = tree_contains_jax(variations_data[:2])
    xnp, _, _ = get_backend(True, use_jax)
    new_tk_slices, _ = fv_share_tucker_variations_corewise((tkv, ttv), groups)
    return xnp.stack(list(new_tk_slices), axis=0), ttv, shape, masks


def ufv_tied_variations_residual(
        variations_data,    # UT3Variations .data: (tkv_sc, ttv_sc, shape, (4 masks)); stack = K + C
        shared_data:    'SharedFrameData',  # the frame's UNIFORM companion (ufv_shared_frame_data)
        rcond:          typ.Optional[float] = None,  # relative clip on the group spectrum; None -> dtype eps
) -> NDArray:  # shape = K + C; relative deviation per stack element (0 == already tied)
    '''The uniform twin of :py:func:`fv_tied_variations_residual`, on the MASKED variation content.

    Same single global Frobenius ratio with the ``K + C`` stack kept -- masked first, because padding is
    don't-care garbage and two tangents tied on their real content are tied whatever their padding
    holds. The core axes summed over are the supercore's trailing ``(nDi, Ni)``; the leading ``d`` axis
    is summed as well, since the norm spans all ``d`` cores.
    '''
    tkv, ttv = ufv_masking.ufv_apply_variations_masks(variations_data)
    tied_sc = ufv_share_tucker_variations(variations_data, shared_data, rcond=rcond)[0]
    xnp, _, _ = get_backend(True, tree_contains_jax((variations_data[:2], shared_data.svd_s)))

    diff = tied_sc - tkv
    num = xnp.sqrt(xnp.sum(diff * diff, axis=(0, -2, -1)))   # over d and the core axes; keep K+C
    den = xnp.sqrt(xnp.sum(tkv * tkv, axis=(0, -2, -1)))
    pos = den > 0.0
    return xnp.where(pos, num / xnp.where(pos, den, 1.0),    # branch-free zero guard
                     xnp.where(num > 0.0, xnp.inf, 0.0))


############################################################
##########    Edge-weight sharing compatibility    #########
############################################################


def _weights_deviation(vectors, groups, xnp, stack_of):
    '''Max relative deviation of per-mode weight VECTORS within each group (the shared core of the
    two weight checkers): ``||w_i - w_ref|| / ||w_ref||`` per stack element, branch-free zero guard
    (zero reference + nonzero other -> inf). ``vectors[i]`` = stack+(rank_i,); all-singleton -> 0.'''
    devs = []
    for group in groups:
        if len(group) < 2:
            continue
        w_ref = vectors[group[0]]
        denom = xnp.sqrt(xnp.sum(w_ref * w_ref, axis=-1))
        for ii in group[1:]:
            diff = vectors[ii] - w_ref
            num = xnp.sqrt(xnp.sum(diff * diff, axis=-1))
            pos = denom > 0.0
            devs.append(xnp.where(pos, num / xnp.where(pos, denom, 1.0),
                                  xnp.where(num > 0.0, xnp.inf, 0.0)))
    if not devs:
        return xnp.zeros(stack_of)
    return xnp.max(xnp.stack(devs), axis=0)


def t3_tucker_weights_sharing_residual(
        weights:    typ.Tuple[
            typ.Sequence[NDArray],  # tucker_weights. len=d,   elm_shape=stack_shape+(ni,)
            typ.Sequence[NDArray],  # tt_weights.     len=d+1, elm_shape=stack_shape+(ri,)
        ],
        sharing:    typ.Sequence,   # len=d, static; one hashable group label per mode
) -> NDArray:  # shape = stack_shape; max relative Tucker-weight deviation per stack element
    '''Non-enforcing check that edge weights are COMPATIBLE with a sharing partition, per stack
    element: the max over groups and group modes of ``||w_i - w_ref|| / ||w_ref||`` on the TUCKER
    weight vectors. TT-bond weights are unconstrained (they are absorbed into the TT cores and
    never touch the factors -- only equal group Tucker weights keep ``absorb_weights`` on a tied
    T3 tied). Weights carry no mode sizes, so the size check of :py:func:`validate_sharing` does
    not apply; unequal weight LENGTHS within a group (unequal Tucker ranks) raise (structural).
    ``T3Weights.from_t3svd(x, sharing=...)`` produces group-equal weights by construction (the
    group spectrum at every group mode), and ``concatenate``/``kronecker``/``reciprocal``/``sqrt``
    all preserve group-equality.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.sharing as sharing
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 3, 2, 1))
    >>> tk, tt = x.data
    >>> xs = t3.TuckerTensorTrain((tk[0], tk[0], tk[2]), tt)          # a tied point
    >>> W = t3.T3Weights.from_t3svd(xs, sharing=(0, 0, 1))            # grouped svals: group-equal
    >>> print(float(sharing.t3_tucker_weights_sharing_residual(W.data, (0, 0, 1))))
    0.0
    >>> W2 = t3.T3Weights.from_t3svd(xs)                              # per-mode svals: NOT group-equal
    >>> print(bool(sharing.t3_tucker_weights_sharing_residual(W2.data, (0, 0, 1)) > 1e-3))
    True
    '''
    tucker_weights, _tt_weights = weights
    d = len(tucker_weights)
    groups = _groups_from_labels(sharing, d)
    for group in groups:
        lengths = tuple(tucker_weights[ii].shape[-1] for ii in group)
        if len(set(lengths)) > 1:
            raise ValueError(
                'Tucker weights in a sharing group must have equal lengths (one shared rank per '
                'group); group %r has lengths %r' % (group, lengths))
    use_jax = tree_contains_jax(weights)
    xnp, _, _ = get_backend(False, use_jax)
    return _weights_deviation(tucker_weights, groups, xnp, tucker_weights[0].shape[:-1])


def t3_tucker_weights_shared(
        weights:    typ.Tuple[
            typ.Sequence[NDArray],  # tucker_weights. len=d,   elm_shape=stack_shape+(ni,)
            typ.Sequence[NDArray],  # tt_weights.     len=d+1, elm_shape=stack_shape+(ri,)
        ],
        sharing:    typ.Sequence,   # len=d, static; one hashable group label per mode
        rtol:       float = 1e-9,   # relative tolerance on the Tucker-weight deviation
) -> NDArray:  # bool array, shape = stack_shape (scalar/0-d when unstacked)
    '''True (per stack element) where the Tucker weights are equal within every sharing group --
    the boolean form of :py:func:`t3_tucker_weights_sharing_residual`. A non-enforcing checker:
    absorbing group-UNEQUAL weights into a tied T3 is legitimate (it just unties the result --
    repair with :py:func:`t3_tie_tucker_factors` or re-enter with ``share`` if wanted).'''
    return t3_tucker_weights_sharing_residual(weights, sharing) <= rtol


def ut3_tucker_weights_sharing_residual(
        weights_data:   typ.Tuple[
            NDArray,             # tucker_weight_supercore, shape=(d,)+stack+(n,)
            NDArray,             # tt_weight_supercore,     shape=(d+1,)+stack+(r,)
            typ.Tuple[NDArray, NDArray],  # (tucker_edge_mask, tt_edge_mask), HOST bool, static
        ],
        sharing:        typ.Sequence,   # len=d, static; one hashable group label per mode
) -> NDArray:  # shape = stack_shape; max relative MASKED Tucker-weight deviation per stack element
    '''The uniform twin of :py:func:`t3_tucker_weights_sharing_residual`: the comparison runs on the
    MASKED Tucker weight vectors (padding is don't-care), and unequal group rank masks raise
    (structural -- unequal ranks cannot carry one shared weight).'''
    tucker_w_sc, _tt_w_sc, (tucker_mask, _tt_mask) = weights_data
    d = tucker_w_sc.shape[0]
    groups = _groups_from_labels(sharing, d)
    _validate_group_tucker_rank_masks(tucker_mask, groups)
    use_jax = tree_contains_jax(weights_data[:2])
    xnp, _, _ = get_backend(True, use_jax)
    masked = tucker_w_sc * tucker_mask
    return _weights_deviation(masked, groups, xnp, masked.shape[1:-1])


def ut3_tucker_weights_shared(
        weights_data:   typ.Tuple[
            NDArray,             # tucker_weight_supercore, shape=(d,)+stack+(n,)
            NDArray,             # tt_weight_supercore,     shape=(d+1,)+stack+(r,)
            typ.Tuple[NDArray, NDArray],  # (tucker_edge_mask, tt_edge_mask), HOST bool, static
        ],
        sharing:        typ.Sequence,   # len=d, static; one hashable group label per mode
        rtol:           float = 1e-9,   # relative tolerance on the Tucker-weight deviation
) -> NDArray:  # bool array, shape = stack_shape (scalar/0-d when unstacked)
    '''True (per stack element) where the MASKED Tucker weights are equal within every sharing
    group -- the boolean form of :py:func:`ut3_tucker_weights_sharing_residual` (non-enforcing).'''
    return ut3_tucker_weights_sharing_residual(weights_data, sharing) <= rtol


if jax_available:
    import jax

    # The companion is a jax pytree: the per-group arrays (centers + SVD factors) are LEAVES
    # (they flow as traced data beside the frame), the partition and the static row offsets are
    # aux_data -- so a companion crossing a jit boundary keeps a stable, static structure key.
    jax.tree_util.register_pytree_node(
        SharedFrameData,
        lambda s: ((s.centers, s.svd_U, s.svd_s, s.svd_Vt), (s.groups, s.row_splits)),
        lambda aux, ch: SharedFrameData(groups=aux[0], row_splits=aux[1], centers=ch[0],
                                          svd_U=ch[1], svd_s=ch[2], svd_Vt=ch[3]),
    )
