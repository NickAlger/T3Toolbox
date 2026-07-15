# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Structural operations on uniform supercore data, and the uniform weighted-layer core ops.

``ut3_squash_tails``/``ut3_reverse``, stack/unstack + leaf structure, and the packing seam
(``pack_vectors``/``unpack_vectors``/``is_packed``/``pack_if_ragged``) behind the
packedness-mirror convention (user-facing ops mirror the input's packedness).

The weighted layer (uniform twins of the ragged ``t3_*_weights`` ops, same module split):
``ut3_absorb_weights`` / ``ut3_weights_consistent`` / ``ut3_reciprocal_weights`` /
``ut3_sqrt_weights`` / ``ut3_concatenate_weights`` / ``ut3_kronecker_weights``. Note this module does **not** import the masking layer: weighting and
masking are kept apart, and the shared mechanics they both need (``prefix_mask``,
``require_concrete_masks``) live in ``common`` (``dev/uniform_weighting_design.md`` §2).
"""
import numpy as np
import typing as typ

import t3toolbox.backend.stacking as stacking
from t3toolbox.backend.common import *
from t3toolbox.backend.tt_operations import tt_reverse, tt_squash_tails

__all__ = [
    'ut3_squash_tails',
    'ut3_reverse',
    'pack_vectors',
    'unpack_vectors',
    'is_packed',
    'pack_if_ragged',
    'ut3_unstack',
    'ut3_stack',
    'ut3_leaf_structure',
    'ut3_absorb_weights',
    'ut3_weights_consistent',
    'ut3_reciprocal_weights',
    'ut3_sqrt_weights',
    'ut3_concatenate_weights',
    'ut3_kronecker_weights',
]

# A uniform-T3 .data tuple: (tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask)).
# `shape` is a static int tuple (N0,...,N(d-1)); the two rank masks are HOST bool, static structure
# (numpy, never traced); the supercores are xnp data.
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]]

# A uniform-T3-WEIGHTS .data tuple: (tucker_weight_supercore, tt_weight_supercore, (2 edge masks)).
# NO `shape`: weights live only on the INTERNAL edges -- a weight has no physical mode legs (external
# weights are out of scope; dev/weighted_layer_design.md §2). Otherwise it mirrors UT3Data: the weight
# supercores are xnp data, the masks are HOST bool static structure and are the SAME masks as the object
# the weight pairs with (a weight's edges are the tensor's edges).
UT3WeightsData = typ.Tuple[NDArray, NDArray, typ.Tuple[NDArray, NDArray]]


def ut3_leaf_structure(d: int):  # leaf-structure template for stacking.apply_func_to_leaf_subtrees
    """Template marking one uniform-T3 ``.data`` leaf for the tree machinery in ``stacking.py``.

    The leaf is ``(tucker_supercore, tt_supercore, shape, (tucker_mask, tt_mask))``. The ``shape``
    int tuple has ``d`` int leaves, so the template must encode its length: a bare ``None`` there
    fails to match (an int tuple is a ``Sequence``, unlike the ndarray leaves, so the walker would
    recurse into it)."""
    return (None, None, (None,) * d, (None, None))


def _first_data_leaf(xx):  # drill to the first .data leaf without recursing into the int-tuple `shape`
    # A .data leaf has an ndarray (tucker_supercore) at [0]; a nesting node has a subtree (tuple) there.
    # (get_first_leaf can't be used here: it would drill into `shape`, which is itself a Sequence.)
    while not is_ndarray(xx[0]):
        xx = xx[0]
    return xx


def ut3_squash_tails(data: UT3Data) -> UT3Data:
    """Sum the leading/trailing TT bonds down to rank 1 (preserves the tensor), updating those edge
    masks to rank 1. Operates on the full .data tuple."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    tk, tt, shape, (tkm, ttm) = data
    require_concrete_masks(tkm, ttm)  # masks are host, not traced
    new_tt = tt_squash_tails(tt)
    r = tt.shape[-1]
    stack = tt.shape[1:-3]
    # np (host): the rank-1 boundary masks are static structure, not supercore data. Intentional.
    rank1 = np.broadcast_to(prefix_mask(1, r), stack + (r,))                   # [True, False, ...]
    new_ttm = np.concatenate([rank1[None], ttm[1:-1], rank1[None]], axis=0)
    return tk, new_tt, shape, (tkm, new_ttm)


def ut3_reverse(data: UT3Data) -> UT3Data:
    """Reverse the mode order (supercores, shape, and masks). Operates on the full .data tuple."""
    tk, tt, shape, (tkm, ttm) = data
    return tk[::-1], tt_reverse(tt), shape[::-1], (tkm[::-1], ttm[::-1])


def pack_vectors(
        unpacked_vectors: typ.Sequence[NDArray],  # len=d, ith elm_shape=stack_shape+(Ni,)
        N: int = None,                            # padded length (default max(Ni))
) -> NDArray:                                     # packed, shape=(d,)+stack_shape+(N,)
    """Zero-pad and stack a sequence of (ragged-length) vectors into one supercore-shaped tensor.

    The pad fill is zeros, and must stay FINITE: masking works by multiplication, and
    ``0 * nan = nan`` -- a ``nan``/``inf`` fill would poison masked reductions downstream
    (``docs/uniform_equivalence_contract.md``). Shape information always travels alongside the
    packed array; the fill is never used to infer shape.
    """
    if not unpacked_vectors:
        return np.array(())
    use_jax = tree_contains_jax(unpacked_vectors)
    xnp, _, _ = get_backend(False, use_jax)

    stack_shape = unpacked_vectors[0].shape[:-1]
    if N is None:
        N = max(v.shape[-1] for v in unpacked_vectors)

    padded = []
    for v in unpacked_vectors:
        pad = ((0, 0),) * len(stack_shape) + ((0, N - v.shape[-1]),)
        padded.append(xnp.pad(v, pad))
    return xnp.stack(padded)


def unpack_vectors(
        packed_vectors:  NDArray,            # shape=(d,)+stack_shape+(N,)
        unpacking_shape: typ.Sequence[int],  # (N0,...,N(d-1))
) -> typ.Tuple[NDArray, ...]:                # len=d, ith elm_shape=stack_shape+(Ni,)
    """Slice a packed supercore-shaped tensor back into a tuple of (ragged-length) vectors.
    """
    return tuple(
        packed_vectors[ii, ..., :unpacking_shape[ii]]
        for ii in range(len(unpacking_shape))
    )


def is_packed(
        vectors,  # a packed supercore array (single ndarray) OR a ragged len=d sequence of per-mode vectors
) -> bool:        # True if already packed (one array), False if a ragged list/tuple of d per-mode arrays
    """Whether mode ``vectors`` are packed (a single supercore-shaped ndarray) or ragged (a ``len=d``
    sequence of per-mode arrays of differing widths). The uniform sampling ops infer packedness from this
    and **mirror** it -- packed in ``->`` packed out, ragged in ``->`` ragged out."""
    return not isinstance(vectors, (list, tuple))


def pack_if_ragged(
        vectors,        # packed array (returned as-is) OR ragged len=d sequence, ith elm_shape=stack+(Ni,)
        N:  int = None,  # padded length (default max Ni); ignored when ``vectors`` is already packed
) -> NDArray:            # packed, shape=(d,)+stack_shape+(N,)
    """Pack ``vectors`` iff ragged (a ``len=d`` sequence); an already-packed array is returned unchanged.
    The input side of the sampling-op packedness mirror (:py:func:`is_packed`)."""
    return vectors if is_packed(vectors) else pack_vectors(vectors, N)


def ut3_unstack(
        x: typ.Tuple[
            NDArray,                          # tucker_supercore
            NDArray,                          # tt_supercore
            typ.Tuple[int, ...],              # shape
            typ.Tuple[NDArray, NDArray],      # (tucker_edge_mask, tt_edge_mask)
        ],
):  # -> nested tuple (shaped like stack_shape) of unstacked uniform-T3 .data leaves
    """Unstack a uniform Tucker tensor train into an array-like tree of unstacked ones.

    The stack lives at axes ``1 .. len(stack_shape)`` (axis 0 is the mode index ``d``). The supercores
    and the rank masks unstack along it; ``shape`` is shared and replicated onto every leaf (the
    ndarray-only ``(tk, tt, tkm, ttm)`` go through the tree machinery; ``shape`` is woven in after).
    """
    tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask) = x
    stack_shape = tucker_supercore.shape[1:-2]
    axes = tuple(range(1, 1 + len(stack_shape)))

    tree = stacking.unstack((tucker_supercore, tt_supercore, tucker_edge_mask, tt_edge_mask), axes=axes)

    return stacking.apply_func_to_leaf_subtrees(
        tree,
        lambda leaf: (leaf[0], leaf[1], shape, (leaf[2], leaf[3])),
        (None, None, None, None),
    )


def ut3_stack(
        xx,  # nested tuple (shaped like stack_shape) of unstacked uniform-T3 .data leaves
) -> typ.Tuple[
    NDArray,                          # tucker_supercore
    NDArray,                          # tt_supercore
    typ.Tuple[int, ...],              # shape
    typ.Tuple[NDArray, NDArray],      # (tucker_edge_mask, tt_edge_mask)
]:
    """Stack an array-like tree of uniform Tucker tensor trains into one.

    Inverse of :py:func:`ut3_unstack`: stacks the supercores and rank masks onto axes
    ``1 .. num_levels`` (after the mode index), keeping the shared ``shape`` unstacked. Only the four
    ndarray components go through ``stacking.stack``; ``shape`` (a ``Sequence`` the walker would recurse
    into) is read once from the first leaf and re-attached.
    """
    first = _first_data_leaf(xx)        # shape is shared across the stack -> read once (manual drill)
    shape = first[2]
    d = first[0].shape[0]

    # Stack the supercores and the masks via SEPARATE stacking.stack calls. stacking.stack infers ONE
    # backend per call (tree_contains_jax over the whole tree), so a mixed (jax supercore + host mask) call
    # would promote the masks to jax -- breaking the masks-are-host-numpy invariant. The mask-only call has
    # no jax inputs, so the masks stay host numpy; the supercores follow xnp as usual.
    sc_tree   = stacking.apply_func_to_leaf_subtrees(
        xx, lambda leaf: (leaf[0], leaf[1]), ut3_leaf_structure(d))
    mask_tree = stacking.apply_func_to_leaf_subtrees(
        xx, lambda leaf: (leaf[3][0], leaf[3][1]), ut3_leaf_structure(d))

    num_levels = tree_depth_of_tree_over_leaf(sc_tree)
    axes = tuple(range(1, 1 + num_levels))

    tucker_supercore, tt_supercore = stacking.stack(sc_tree, axes)
    tucker_edge_mask, tt_edge_mask = stacking.stack(mask_tree, axes)
    return tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask)


def tree_depth_of_tree_over_leaf(
        flat_tree,  # tree whose leaves are flat tuples (tk, tt, tucker_mask, tt_mask) of arrays
) -> int:           # number of stacking levels (tree nesting depth above the leaf tuple)
    """Number of stack levels in a tree of 4-array leaves = total nesting depth minus the 1 leaf level."""
    return stacking.tree_depth(flat_tree) - 1


def ut3_absorb_weights(
        x:       UT3Data,         # (tucker_supercore, tt_supercore, shape, masks)
        weights: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)
) -> UT3Data:                     # the weighted train: same shape, same masks (absorb preserves both)
    """Contract diagonal edge weights into a uniform Tucker tensor train's supercores (shape-preserving).

    The uniform twin of ``t3_absorb_weights``, with the same side-convention, but vectorized over the
    leading ``(d,)`` instead of looping the cores:

    - **Tucker weights -> the Tucker supercore** (its rank leg ``n``).
    - **TT bond weights leftward**: bond ``r(k+1)`` into core ``k``'s right leg; the leftmost boundary
      bond ``r0`` (which has no left neighbour) goes **rightward** into core ``0``'s left leg instead. So
      each of the ``d+1`` bonds is absorbed exactly once.

    **No entry masking, deliberately -- absorb is garbage-transparent.** It is a *pointwise scale along
    each edge axis*, not a reduction: real slot ``i`` of the output depends only on slot ``i`` of the core
    and slot ``i`` of the weight, so garbage can never mix into a real slot. Garbage padding in gives
    garbage padding out, which the equivalence contract declares don't-care
    (``docs/uniform_equivalence_contract.md``). Do not add a defensive entry-mask: it would be dead work,
    and masking is a separate concern from weighting (``dev/uniform_weighting_design.md`` §2).

    **Precondition (structural; NOT enforced here):** ``weights``' masks must equal ``x``'s masks --
    :py:func:`ut3_weights_consistent`. Ragged catches a rank mismatch as a loud einsum shape error, but
    uniform pads both to the common ``(n, r)``, so a mismatch is **silent and corrupting** (a weight whose
    mask calls slot ``i`` padding carries a canonical zero there, and absorbing it would zero a *real*
    slot of ``x``). The frontend enforces it; a raw-``.data`` user should call the predicate first. This
    is the same precondition uniform adds to variation add/sub (``docs/uniform_masks_vs_ranks.md``).
    """
    xnp, _, _ = get_backend(True, tree_contains_jax((x[:2], weights[:2])))
    tucker_supercore, tt_supercore, shape, masks = x
    tucker_weight_supercore, tt_weight_supercore, _ = weights

    weighted_tucker = xnp.einsum('d...n,d...nN->d...nN', tucker_weight_supercore, tucker_supercore)

    # Bond r(k+1) leftward into core k's right leg -- every core at once (tt weights 1..d).
    weighted_tt = xnp.einsum('d...lnr,d...r->d...lnr', tt_supercore, tt_weight_supercore[1:])
    # The boundary bond r0 has no left neighbour -> rightward into core 0's left leg. Slice-and-rejoin on
    # the d axis (the ut3_scale idiom) keeps the rest untouched; the [:1] slices keep d in the einsum.
    G0 = xnp.einsum('d...lnr,d...l->d...lnr', weighted_tt[:1], tt_weight_supercore[:1])
    weighted_tt = xnp.concatenate([G0, weighted_tt[1:]], axis=0)

    return weighted_tucker, weighted_tt, shape, masks


def ut3_weights_consistent(
        x:       UT3Data,         # (tucker_supercore, tt_supercore, shape, masks)
        weights: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)
) -> bool:                        # True iff `weights` can be absorbed into `x`
    """True iff ``weights`` fits ``x``: the padded weight shapes match, **and the edge masks are equal**.

    Mask equality is the substance, and it is what the ragged twin (``t3_weights_consistent``, which
    compares lengths/ranks/stack) gets for free from shapes. A weight's edges *are* the tensor's edges, so
    it declares the same ranks; ragged enforces that structurally (a length-``n`` weight vector against a
    rank-``n`` core -- a mismatch is an einsum shape error). Uniform pads both to the common ``(n, r)``, so
    a mismatched mask is invisible to the shapes and would silently corrupt: a weight whose mask calls slot
    ``i`` padding carries a canonical zero there, so absorbing it **zeroes a real slot** of ``x``. Hence an
    explicit structural predicate -- the same precondition uniform adds to variation add/sub
    (``docs/uniform_masks_vs_ranks.md``). Non-raising (the frontend raises).
    """
    tucker_supercore, tt_supercore, _, (tucker_edge_mask, tt_edge_mask) = x
    tucker_weight_supercore, tt_weight_supercore, (weight_tucker_mask, weight_tt_mask) = weights

    d     = tucker_supercore.shape[0]
    stack = tucker_supercore.shape[1:-2]
    n     = tucker_supercore.shape[-2]
    r     = tt_supercore.shape[-1]

    if tuple(tucker_weight_supercore.shape) != (d,) + stack + (n,):
        return False
    if tuple(tt_weight_supercore.shape) != (d + 1,) + stack + (r,):
        return False

    # np (host): masks are static structure, so this is a cheap host-side compare -- valid under jit.
    return (np.array_equal(weight_tucker_mask, tucker_edge_mask)
            and np.array_equal(weight_tt_mask, tt_edge_mask))


def _ut3_map_real_weights(
        weights: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)
        fn,                       # (xnp, w) -> w, applied elementwise to the REAL slots only
) -> UT3WeightsData:              # fn on the real slots; padding forced to a canonical, finite 0
    """Apply ``fn`` to the real slots of both weight supercores, forcing the padding to a finite ``0``.

    The padding is **neutralized to 1 before ``fn`` runs**, then overwritten with ``0`` -- the standard
    double-``where``. Both halves earn their keep: ``fn`` may be undefined (``1/0``) or
    non-differentiable (``sqrt`` at ``0``) at the padding's canonical zero, and a single outer ``where``
    would not save you -- ``nan`` from the dead branch still propagates through the **gradient**. It also
    neutralizes large-finite *garbage* padding, so no separate entry-masking is needed.

    Why the padding must end finite at all: masking downstream works by multiplication, and
    ``0 * inf = nan`` would poison every masked reduction (``docs/uniform_equivalence_contract.md``).
    """
    xnp, _, _ = get_backend(True, tree_contains_jax(weights[:2]))
    tucker_weight_supercore, tt_weight_supercore, (tucker_edge_mask, tt_edge_mask) = weights
    require_concrete_masks(tucker_edge_mask, tt_edge_mask)  # masks are host, not traced

    def go(w, m):
        neutral = xnp.where(m, w, 1.0)         # padding (canonical 0 OR garbage) -> 1: fn safe, grad finite
        return xnp.where(m, fn(xnp, neutral), 0.0)   # real slots: fn(w). padding: the canonical finite 0

    return (go(tucker_weight_supercore, tucker_edge_mask),
            go(tt_weight_supercore, tt_edge_mask),
            (tucker_edge_mask, tt_edge_mask))


def ut3_reciprocal_weights(
        weights: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)
) -> UT3WeightsData:              # 1/w on the real slots; padding a canonical, finite 0; masks unchanged
    """Elementwise ``1/w`` on the real slots (masks unchanged) -- e.g. to form inverse-singular-value
    weights.

    **Not just ``1/w``**, because a canonical weight's padding is zero and ``1/0 = inf``, which then
    poisons every masked reduction downstream (``0 * inf = nan``). That is the headline path, not a corner
    case: the Grasedyck-Kramer metric *is* ``from_ut3svd(x).reciprocal()``. The real slots are left alone
    (see :py:func:`_ut3_map_real_weights` for how the padding is handled).

    **The real slots are deliberately unprotected**: a genuinely zero singular value gives ``inf`` here,
    exactly as in the ragged layer. It is a real weight, not a padding artifact, so it is the caller's to
    avoid -- silently clamping it would hide a rank-deficient point.
    """
    return _ut3_map_real_weights(weights, lambda xnp, w: 1.0 / w)


def ut3_sqrt_weights(
        weights: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)
) -> UT3WeightsData:              # sqrt(w) on the real slots; padding a canonical, finite 0; masks unchanged
    """Elementwise ``sqrt`` on the real slots (masks unchanged). The padding is neutralized rather than
    square-rooted: ``sqrt`` is fine at the canonical ``0`` but not differentiable there (an ``inf``
    gradient), and garbage padding can be negative (``nan``). See :py:func:`_ut3_map_real_weights`."""
    return _ut3_map_real_weights(weights, lambda xnp, w: xnp.sqrt(w))


def _weight_pair_backend(weights_A: UT3WeightsData, weights_B: UT3WeightsData):
    """``(xnp, A supercores, B supercores, A masks, B masks)`` for the two-weight combines."""
    xnp, _, _ = get_backend(True, tree_contains_jax((weights_A[:2], weights_B[:2])))
    require_concrete_masks(*weights_A[2], *weights_B[2])  # masks are host, not traced
    return xnp, weights_A[:2], weights_B[:2], weights_A[2], weights_B[2]


def ut3_concatenate_weights(
        weights_A: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)
        weights_B: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)
) -> UT3WeightsData:                # per-edge concatenation: padded widths ADD, masks concatenate
    """Per-edge concatenation of two uniform weights -- the ``+`` / direct-sum combine, where ranks add.
    Uniform twin of ``t3_concatenate_weights``, and the weight-side partner of ``ut3_add`` (which
    concatenates the *object's* rank masks in exactly the same way).

    Both the padded supercores and the masks concatenate on the last axis, so the output's padded width
    is ``nA + nB`` and its real rank is ``rank_A + rank_B``.

    **The output mask can go gappy, and that is correct** (``docs/uniform_masks_vs_ranks.md``): if ``A``'s
    mask has slack (real rank below its padded width), the concatenation leaves a hole between ``A``'s
    real slots and ``B``'s block. The data stays put -- no compaction -- and ``ut3weights_to_t3weights``
    gathers through the holes. Re-canonicalize with the SVD if a prefix form is wanted.

    Neither operand's real slots move, so no masking is needed: the supercores are copied, not reduced,
    and each output slot comes from exactly one input slot (garbage padding stays in the padding).
    """
    xnp, (tkA, ttA), (tkB, ttB), (tkmA, ttmA), (tkmB, ttmB) = _weight_pair_backend(weights_A, weights_B)
    tucker = xnp.concatenate([tkA, tkB], axis=-1)
    tt     = xnp.concatenate([ttA, ttB], axis=-1)
    # np (host): masks are static structure -- concatenation IS the closure of the mask algebra under +.
    masks = (np.concatenate([tkmA, tkmB], axis=-1), np.concatenate([ttmA, ttmB], axis=-1))
    return tucker, tt, masks


def ut3_kronecker_weights(
        weights_A: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)
        weights_B: UT3WeightsData,  # (tucker_weight_supercore, tt_weight_supercore, masks)
) -> UT3WeightsData:                # per-edge Kronecker: padded widths MULTIPLY, masks Kronecker
    """Per-edge Kronecker product of two uniform weights -- the Hadamard (``⊙``) combine, where ranks
    multiply. Uniform twin of ``t3_kronecker_weights``.

    **Kronecker the weights, Kronecker the masks -- that is the whole operation.** Treat the mask as just
    another weight that happens to hold 0s and 1s: what we need is
    ``weight_AB * mask_AB == kron(weight_A * mask_A, weight_B * mask_B)``, and it is satisfied by
    ``weight_AB = kron(weight_A, weight_B)`` and ``mask_AB = kron(mask_A, mask_B)``, because elementwise
    multiply **commutes** with the Kronecker product: ``kron(a∘p, b∘q) = kron(a,b) ∘ kron(p,q)`` (the
    mixed-product property, for any vectors -- nothing here is special to booleans). So there is no mask
    cleverness: the same ``kron_last`` runs on both operands.

    Each edge is a **last-axis outer product, then reshape**: ``(wA ⊗ wB)[..., a*nB + b] = wA[..., a] ·
    wB[..., b]`` -- **A-major**, over the PADDED widths, with the shared ``(d,)+stack`` prefix broadcast.
    **This is the one real trap**: not ``np.kron``, which would Kronecker the mode/stack axes too (the
    ragged build hit exactly that). A-major must also agree with whatever core-combine pairs with it.

    The resulting mask is **not an interval** -- the real set is ``{a*nB + b : mask_A[a] and mask_B[b]}``,
    strided over the *padded* width ``nB``, so even two prefix inputs give holes (``{0,1}`` of 3 times
    ``{0}`` of 2 gives ``{0, 2}``; ``docs/uniform_masks_vs_ranks.md``). That is a description of the
    output, not a difficulty: it costs nothing here, and only obliges *consumers* to read the mask rather
    than slice a prefix. It cannot be flattened to a prefix of rank ``rA*rB``: slot ``a*nB + b`` with
    ``b >= rank_B`` holds ``wA[a] * <padding>``, so a prefix mask would claim padding as real data
    (phantom rank).

    Note there is currently **no uniform Hadamard** (``ut3_add`` exists; ``ut3_mult`` does not), so this
    op ships verified against the ragged oracle but without a uniform core-combine partner -- see
    ``dev/uniform_weighting_design.md`` §3.
    """
    xnp, (tkA, ttA), (tkB, ttB), (tkmA, ttmA), (tkmB, ttmB) = _weight_pair_backend(weights_A, weights_B)

    def kron_last(a, b, np_module):  # (...,pA),(...,pB) -> (...,pA*pB), A-major, shared prefix broadcast
        prefix = np_module.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        return (a[..., :, None] * b[..., None, :]).reshape(prefix + (a.shape[-1] * b.shape[-1],))

    tucker = kron_last(tkA, tkB, xnp)
    tt     = kron_last(ttA, ttB, xnp)
    # np (host): masks are static structure. The SAME outer-product-then-reshape, on booleans -- the
    # closure of the mask algebra under x. `*` on bools is `and`.
    masks = (kron_last(tkmA, tkmB, np), kron_last(ttmA, ttmB, np))
    return tucker, tt, masks
