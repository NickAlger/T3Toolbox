# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Shared Tucker factors (SF-T3): the sharing partition and the tied-factors checkers.

``validate_sharing`` canonicalizes a per-mode label spec into the static ``groups`` form;
``t3_sharing_residual`` / ``t3_tucker_factors_shared`` are the non-enforcing tied-factors
checkers (the safe-mode precondition behind the shared operations); ``t3_share_tucker_cores``
is the plain per-group mean (drift repair for nearly-tied POINTS -- never the metric
projection of a tangent, which is geometry-specific and lives with the shared geometry).

A shared T3 is an ordinary Tucker tensor train whose Tucker factors are equal within
user-specified groups of modes -- the SF-ETT decomposition of Molozhavenko & Rakhuba (2026),
"Optimization on the extended tensor-train manifold with shared factors" (Comput. Appl. Math.
45:221), generalized to an arbitrary partition of the modes into sharing groups.
"""
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'validate_sharing',
    't3_sharing_residual',
    't3_tucker_factors_shared',
    't3_share_tucker_cores',
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
