# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Named grouped-block einsum contractions (``inputs_to_output``, e.g. ``WCa_Caib_WCi_to_WCb``).

The machinery for TWO independent batch blocks on different operand subsets -- what a single
leading ``'...'`` einsum cannot express. Capital letters are index BLOCKS, each reshaped to one
flat axis (``W`` probe stack, ``K`` tangent stack, ``C`` frame stack; = 1 when empty); lowercase
letters are single axes; a leading ``d`` is the stacked jet/derivative axis. Full design:
``docs/batching_and_stacking.md``.
"""
import functools
import math
import typing as typ
import numpy as np

from t3toolbox.backend.common import *

# This is the grouped-block contraction toolkit for the case of TWO independent batch blocks on
# DIFFERENT operand subsets (the core/frame stack C vs the probe/tangent stack W/K), which a single
# '...' cannot express. For the full design -- why base-inner (W/K outer, C inner), the naming
# scheme, and when to use this vs a plain '...' einsum -- see docs/batching_and_stacking.md (esp. §4).
#
###############################################################################
# SHARDING: FLATTEN ONLY WHAT EINSUM FORCES YOU TO.  (Read before writing a contraction.)
#
# THE RULE, stated once:  *** No named block may be FUSED (flattened together) with another. A block
# that is SHARED across operands needs an einsum letter, so it must flatten its own axes; a block that
# is PASSIVE (lives on ONE operand and rides to the output) needs no letter and rides as '...', with
# every axis intact. ***
#
# WHY IT MATTERS. Flattening axes is a pure reinterpretation -- there is no transpose anywhere in this
# file, by design -- but a reshape moves nothing on a device; it reindexes which logical elements live
# where. So it is SHARDING-SAFE ONLY IF THE SHARDED AXIS IS THE MAJOR (leftmost) MEMBER of the group
# being flattened. With (t=2, W=4) sharded on W, row-major flat index t*4 + W:
#
#   dev0 holds W in {0,1} -> flat {0,1,4,5};  dev1 holds W in {2,3} -> flat {2,3,6,7}
#   a contiguous 2-way tiling is {0..3}/{4..7} -- neither matches, so XLA inserts a collective.
#   Reversed, (W=4, K=2): dev0 holds W in {0,1} -> flat {0,1,2,3} = exactly tile 0. Free.
#
# Every flatten therefore FREEZES a sharding choice: within a flattened group only the leftmost member
# is free. The fix is not to order the fusions cleverly -- it is to not fuse. Each flatten must be
# forced by einsum, and einsum only forces one for a block that needs a LETTER.
#
# SHARED vs PASSIVE -- the whole decision:
#
#   * SHARED (needs a letter, so it flattens its own axes).  C in 'WCi,Cio->WCo': it is on both
#     operands, and einsum must know they are the same block. A fixed subscript cannot spell a variable
#     number of axes, so the block flattens to one axis (size_C = math.prod(C_shape)) -- and only C's
#     LEADING axis is shardable. Measured on C=(4,4): major 0, minor 6. This is inherent and ACCEPTED
#     (the only escape is generating the subscript per call, which is rejected -- see
#     docs/contributor/batching_internals.md).
#
#   * PASSIVE (needs no letter, so it flattens NOTHING).  W+K in '...Ci,Cio->...Co': both live on one
#     operand and ride to the output untouched. Einsum's '...' spells a variable number of axes
#     natively, so they keep their individual axes: W and K are each shardable, on ANY axis, at zero
#     cost. Measured 0 for K, and 0 for W's MINOR axis with W=(4,4) -- the case the letter form cannot
#     fix (it costs 3).
#
# W, K AND C ARE ALL SHARDABLE TODAY. W is the sample axis and the usual data-parallel target, but no
# block is second-class. The only remaining limit is the block-internal one above, and it now applies
# ONLY to a block that a contraction genuinely needs a letter for.
#
# A PASSIVE BLOCK'S SPLIT IS NOT KNOWABLE -- AND THAT IS THE POINT. In WKCi_Cio_to_WKCo, Cio pins
# len(C), but NOTHING pins len(W) vs len(K): W=(2,3),K=() and W=(2,),K=(3,) and W=(),K=(2,3) all give
# operand shape (2,3,C,i) and the same output shape. The function cannot know the split, so it cannot
# give W and K separate letters -- which is exactly why these once fused. '...' dissolves the problem:
# the split is never needed, because a passive block is never flattened. Do NOT add an n_probe/n_frame
# parameter to recover a split a contraction does not need. (Contrast tWCi_KCio_to_tWKCo, where K IS
# pinned -- it is on the other operand -- and n_frame is genuinely required.)
#
# THE DETECTOR. Every function infers its group boundaries POSITIONALLY from its own signature (e.g.
# "everything left of C is W"), so a mis-grouping is invisible in the shapes. Check the RULE instead:
#
#   Every block in a function's NAME must appear in its einsum subscript either as its OWN letter or
#   inside the '...'. Two named blocks sharing one letter -- or a block flattened into another's letter
#   before the call -- is a FUSION, and is a bug.
#
# Fusion hides in two forms, and a grep for delegations only sees the first:
#   1. DELEGATION -- 'return <callee>(...)' where the callee's name drops a block the caller names
#      (it got folded into a neighbour):   awk '/^def /{n=$0} /^    return [a-zA-Z_]+\(/{print n" -> "$0}'
#   2. INTERNAL -- two named blocks merged into one letter in the subscript itself (the old 'Wo,WXa->WXao',
#      X = K+C). No callee to name-check; only reading the subscript against the name finds it.
# Prefer a static check of the rule over either grep: tests/test_contractions_naming.py.
#
# None of this is visible to the numerical suite: a fusion is bit-identical to the unfused form (a pure
# reinterpretation), and nothing else in the library shards. The only instrument that can see it is the
# compiler -- tests/test_contractions_sharding.py compiles under 4 virtual devices and counts collectives.
# Rationale + evidence: docs/contributor/batching_internals.md.
###############################################################################

__all__ = [
    'Wa_Caib_Wi_to_WCb',
    'CWa_Caib_Wo_Cio_to_CWb',
    'WCa_Caib_Wo_Cio_to_WCb',
    'CWa_Caib_CiW_to_CWb',
    'WCa_Caib_WCi_to_WCb',
    'Cio_Wo_to_WCi',
    'dCio_dWo_to_dWCi',
    'WCa_Caib_WCb_to_WCi',
    'dWCa_dCaib_dWCb_to_dWCi',
    'WCi_Cio_to_WCo',
    'dWCi_dCio_to_dWCo',
    'WCo_Cio_to_WCi',
    'WCo_WCa_to_Cao',
    'Wo_WCa_to_Cao',
    'WCi_WCa_WCj_to_Ciaj',
    # Symmetric probe-derivative contractions (the t derivative-order axis + binomial tensor).
    'trs_rWCa_Caib_sWCi_to_tWCb',
    'trs_rWCa_Caib_sWCb_to_tWCi',
    'tWCi_Cio_to_tWCo',
    # Transpose (adjoint) of the symmetric probe derivatives: the adjoint lift (the 2-block adjoint
    # sweeps + order-less assembly are superseded by the K-stacked 3-group versions below, which
    # reduce to the 2-block case at K=()).
    'tWCo_Cio_to_tWCi',
    # Three-group (W probe, K tangent, C frame) contractions for probing a K-stacked tangent.
    'WKCa_Caib_WCi_to_WKCb',
    'WCa_Caib_WKCi_to_WKCb',
    'WKCa_Caib_WCb_to_WKCi',
    'WCa_Caib_WKCb_to_WKCi',
    'WCa_KCaib_WCi_to_WKCb',
    'WCa_KCaib_WCb_to_WKCi',
    'WCi_KCio_to_WKCo',
    'WKCi_Cio_to_WKCo',
    # Transpose-assemble three-group (W,K,C) outer products that build variation cores.
    'WKCo_WCa_to_WKCao',
    'WKCo_WCa_to_KCao',
    'Wo_WKCa_to_WKCao',
    'Wo_WKCa_to_KCao',
    'WCi_WCa_WKCj_to_WKCiaj',
    'WCi_WCa_WKCj_to_KCiaj',
    'WKCi_WCa_WCj_to_WKCiaj',
    'WKCi_WCa_WCj_to_KCiaj',
    'WCi_WKCa_WCj_to_WKCiaj',
    'WCi_WKCa_WCj_to_KCiaj',
    'trs_rWKCa_Caib_sWCi_to_tWKCb',
    'trs_rWCa_Caib_sWKCi_to_tWKCb',
    'trs_rWCa_KCaib_sWCi_to_tWKCb',
    'trs_rWKCa_Caib_sWCb_to_tWKCi',
    'trs_rWCa_Caib_sWKCb_to_tWKCi',
    'trs_rWCa_KCaib_sWCb_to_tWKCi',
    'tWKCi_Cio_to_tWKCo',
    'tWCi_KCio_to_tWKCo',
    'trs_tWKCa_Caib_rWCi_to_sWKCb',
    'trs_rWCa_Caib_tWKCi_to_sWKCb',
    'trs_tWKCa_Caib_sWCb_to_rWKCi',
    'trs_rWCa_Caib_tWKCb_to_sWKCi',
    'tWKCo_Cio_to_tWKCi',
    'trs_rWCa_sWCi_tWKCb_to_WKCaib',
    'trs_rWCa_sWCi_tWKCb_to_KCaib',
    'trs_tWKCa_rWCi_sWCb_to_WKCaib',
    'trs_tWKCa_rWCi_sWCb_to_KCaib',
    'trs_rWCa_tWKCi_sWCb_to_WKCaib',
    'trs_rWCa_tWKCi_sWCb_to_KCaib',
    'tWCa_tWKCo_to_WKCao',
    'tWCa_tWKCo_to_KCao',
    'uWKCa_uWo_to_WKCao',
    'uWKCa_uWo_to_KCao',
    # Adjoint-state apply/entries transpose dG assemble: mu (x) xi (x) sigma_hat (tt-core order a,i,b).
    'WCa_WCi_WKCb_to_WKCaib',
    'WCa_WCi_WKCb_to_KCaib',
    # d-prefixed uniform WKC contractions (3b-6a): each ragged WKC twin with the core index d prepended,
    # vectorized over d in one einsum (the uniform-tangent map-style probing branches).
    'dWKCa_dCaib_dWCb_to_dWKCi',
    'dWCa_dCaib_dWKCb_to_dWKCi',
    'dWCa_dKCaib_dWCb_to_dWKCi',
    'dWKCi_dCio_to_dWKCo',
    'dWCi_dKCio_to_dWKCo',
    'dWCo_dCio_to_dWCi',
    'dWKCo_dWCa_to_dWKCao',
    'dWKCo_dWCa_to_dKCao',
    'dWo_dWKCa_to_dWKCao',
    'dWo_dWKCa_to_dKCao',
    'dWCi_dWCa_dWKCj_to_dWKCiaj',
    'dWCi_dWCa_dWKCj_to_dKCiaj',
    'dWKCi_dWCa_dWCj_to_dWKCiaj',
    'dWKCi_dWCa_dWCj_to_dKCiaj',
    'dWCi_dWKCa_dWCj_to_dWKCiaj',
    'dWCi_dWKCa_dWCj_to_dKCiaj',
    'dWCa_dWCi_dWKCb_to_dWKCaib',
    'dWCa_dWCi_dWKCb_to_dKCaib',
    # d-prefixed uniform JET contractions (3b-6'a): each binomial-jet trs_* twin with the core index d
    # prepended (d leads, then the order axis, then W/K/C). The map-style jet branches of sampling_derivatives.
    'trs_drWCa_dCaib_dsWCb_to_dtWCi',
    'dtWCi_dCio_to_dtWCo',
    'dtWKCi_dCio_to_dtWKCo',
    'dtWCi_dKCio_to_dtWKCo',
    'trs_drWKCa_dCaib_dsWCb_to_dtWKCi',
    'trs_drWCa_dKCaib_dsWCb_to_dtWKCi',
    'trs_drWCa_dCaib_dsWKCb_to_dtWKCi',
    'dtWCo_dCio_to_dtWCi',
    'dtWKCo_dCio_to_dtWKCi',
    'trs_dtWKCa_dCaib_dsWCb_to_drWKCi',
    'trs_drWCa_dCaib_dtWKCb_to_dsWKCi',
    'trs_drWCa_dsWCi_dtWKCb_to_dWKCaib',
    'trs_drWCa_dsWCi_dtWKCb_to_dKCaib',
    'trs_dtWKCa_drWCi_dsWCb_to_dWKCaib',
    'trs_dtWKCa_drWCi_dsWCb_to_dKCaib',
    'trs_drWCa_dtWKCi_dsWCb_to_dWKCaib',
    'trs_drWCa_dtWKCi_dsWCb_to_dKCaib',
    'dtWCa_dtWKCo_to_dWKCao',
    'dtWCa_dtWKCo_to_dKCao',
    'duWKCa_duWo_to_dWKCao',
    'duWKCa_duWo_to_dKCao',
]



# --------------------------------------------------------------------------------------------------
# The einsum dispatcher (numpy: forced pairwise BLAS path; jax: one big einsum)
# --------------------------------------------------------------------------------------------------
# Every grouped contraction below routes its einsum through `_grouped_einsum`. The numpy/jax split is
# NOT cosmetic -- the two backends want OPPOSITE things, and getting it wrong is a silent 10-55x perf hit:
#
#   * numpy. `np.einsum(..., optimize=True)` minimizes FLOP *count*, not wall-clock. On a FLOP-tie it
#     runs a single multi-operand contraction as one `c_einsum` nested loop -- which skips BLAS. For the
#     high-dimensional order-COMBINE contractions (the `trs_*_to_tWCi` jet combines: 4 operands, the full
#     r,s order convolution) that naive path is ~55x slower than splitting into 2-operand `tensordot`
#     (BLAS) steps. (`optimize=False` / a bare einsum is just as bad.) So for numpy we FORCE a greedy
#     pairwise path -- each step a 2-operand BLAS contraction. The path depends only on the subscript
#     string (index sharing is size-independent), so it is cached.
#   * jax. `jnp.einsum` uses opt_einsum (BLAS-aware) + XLA fusion, which beats any path we force -- a
#     single big einsum measured FASTER than manual pairwise. So for jax we pass ONE einsum, no `optimize`.
#
# (Historically the toolkit hand-picked pairwise paths -- correct, but the later `trs` jet contractions
# used `optimize=True`/bare and silently regressed; this dispatcher unifies + fixes them. The greedy path
# reproduces the old hand-picked `[(0,1),(0,1)]` exactly, so no regression.) See docs/batching_and_stacking.md.


@functools.lru_cache(maxsize=None)
def _pairwise_path(
        subscripts: str,    # the einsum string, e.g. 'trs,rWCa,Caib,sWCb->tWCi'
) -> tuple:                 # a numpy `optimize=` path: ('einsum_path', (i,j), ...) -- all 2-operand steps
    '''A greedy pairwise contraction path for numpy: at each step contract the operand pair sharing the
    MOST indices (so no outer products), 2 operands at a time (every step BLAS-eligible). Keyed only on
    the subscript string (index sharing is size-independent), so it is computed once per distinct
    contraction and cached. Reproduces the toolkit's old hand-picked `[(0,1),(0,1)]` paths.'''
    terms = [set(t) for t in subscripts.split('->')[0].split(',')]
    path = []
    while len(terms) > 2:
        n = len(terms)
        i, j = max(((a, b) for a in range(n) for b in range(a + 1, n)),
                   key=lambda ab: len(terms[ab[0]] & terms[ab[1]]))
        path.append((i, j))
        merged = terms[i] | terms[j]
        terms = [t for k, t in enumerate(terms) if k not in (i, j)] + [merged]
    path.append((0, 1))
    return tuple(['einsum_path'] + path)


def _grouped_einsum(
        xnp,                            # numpy or jax.numpy (from get_backend)
        use_jax:    bool,               # is the computation on jax arrays?
        subscripts: str,                # the einsum string
        *operands:  NDArray,
) -> NDArray:
    '''Dispatched einsum for the grouped contractions: jax -> one big einsum (XLA optimizes); numpy ->
    a forced greedy-pairwise BLAS path (numpy's own optimizer runs FLOP-tied multi-operand contractions
    as a single non-BLAS `c_einsum`). 2-operand contractions are already BLAS, so they pass straight
    through on both. See the module note above.'''
    if use_jax or len(operands) <= 2:
        return xnp.einsum(subscripts, *operands)
    return xnp.einsum(subscripts, *operands, optimize=list(_pairwise_path(subscripts)))


def Wa_Caib_Wi_to_WCb(
        Wa: NDArray,
        Caib: NDArray,
        Wi: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((Wa, Caib, Wi))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wa.shape[:-1]
    C_shape = Caib.shape[:-3]

    a_shape = Wa.shape[-1:]
    aib_shape = Caib.shape[-3:]
    b_shape = Caib.shape[-1:]
    i_shape = Wi.shape[-1:]

    Wa      = Wa.reshape((-1,)      + a_shape)
    Caib    = Caib.reshape((-1,)    + aib_shape)
    Wi      = Wi.reshape((-1,)      + i_shape)

    WCb = _grouped_einsum(xnp, use_jax, 'Wa,Caib,Wi->WCb', Wa, Caib, Wi)

    WCb = WCb.reshape(W_shape + C_shape + b_shape)
    return WCb


def CWa_Caib_Wo_Cio_to_CWb(
        CWa: NDArray,
        Caib: NDArray,
        Wo: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((CWa, Caib, Wo, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wo.shape[:-1]
    C_shape = Caib.shape[:-3]
    a_shape = (Caib.shape[-3],)

    aib_shape = Caib.shape[-3:]
    io_shape = Cio.shape[-2:]
    b_shape = Caib.shape[-1:]
    o_shape = Wo.shape[-1:]

    size_C = math.prod(C_shape)
    size_W = math.prod(W_shape)

    CWa     = CWa.reshape((size_C,) + (size_W,)      + a_shape)
    Caib    = Caib.reshape((size_C,) + aib_shape)
    Wo      = Wo.reshape((size_W,) + o_shape)
    Cio     = Cio.reshape((size_C,) + io_shape)

    CWb = _grouped_einsum(xnp, use_jax, 'CWa,Caib,Wo,Cio->CWb', CWa, Caib, Wo, Cio)

    CWb = CWb.reshape(C_shape + W_shape + b_shape)
    return CWb


def WCa_Caib_Wo_Cio_to_WCb(
        WCa: NDArray,
        Caib: NDArray,
        Wo: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost. WC twin of
    CWa_Caib_Wo_Cio_to_CWb, used by the (base-inner) apply.
    """
    use_jax = tree_contains_jax((WCa, Caib, Wo, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wo.shape[:-1]
    C_shape = Caib.shape[:-3]
    a_shape = (Caib.shape[-3],)

    aib_shape = Caib.shape[-3:]
    io_shape = Cio.shape[-2:]
    b_shape = Caib.shape[-1:]
    o_shape = Wo.shape[-1:]

    size_C = math.prod(C_shape)
    size_W = math.prod(W_shape)

    WCa     = WCa.reshape((size_W,) + (size_C,)      + a_shape)
    Caib    = Caib.reshape((size_C,) + aib_shape)
    Wo      = Wo.reshape((size_W,) + o_shape)
    Cio     = Cio.reshape((size_C,) + io_shape)

    WCb = _grouped_einsum(xnp, use_jax, 'WCa,Caib,Wo,Cio->WCb', WCa, Caib, Wo, Cio)

    WCb = WCb.reshape(W_shape + C_shape + b_shape)
    return WCb


def CWa_Caib_CiW_to_CWb(
        CWa: NDArray,
        Caib: NDArray,
        CiW: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((CWa, Caib, CiW))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    i_shape = (Caib.shape[-2],)
    W_shape = CWa.shape[len(C_shape):-1]

    a_shape = CWa.shape[-1:]
    aib_shape = Caib.shape[-3:]
    b_shape = Caib.shape[-1:]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    CWa     = CWa.reshape((size_C,) + (size_W,) + a_shape)
    Caib    = Caib.reshape((size_C,) + aib_shape)
    CiW     = CiW.reshape((size_C,) + i_shape + (size_W,))

    CWb = _grouped_einsum(xnp, use_jax, 'CWa,Caib,CiW->CWb', CWa, Caib, CiW)

    CWb = CWb.reshape(C_shape + W_shape + b_shape)
    return CWb


def WCa_Caib_WCi_to_WCb(
        WCa: NDArray,
        Caib: NDArray,
        WCi: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((WCa, Caib, WCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    i_shape = (Caib.shape[-2],)
    W_shape = WCa.shape[:-(len(C_shape) + 1)]

    a_shape = WCa.shape[-1:]
    aib_shape = Caib.shape[-3:]
    b_shape = Caib.shape[-1:]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    WCa     = WCa.reshape((size_W,) + (size_C,) + a_shape)
    Caib    = Caib.reshape((size_C,) + aib_shape)
    WCi     = WCi.reshape((size_W,) + (size_C,) + i_shape)

    WCb = _grouped_einsum(xnp, use_jax, 'WCa,Caib,WCi->WCb', WCa, Caib, WCi)

    WCb = WCb.reshape(W_shape + C_shape + b_shape)
    return WCb


def Cio_Wo_to_WCi(
        Cio: NDArray,
        Wo: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((Cio, Wo))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    W_shape = Wo.shape[:-1]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    Cio = Cio.reshape((size_C,) + i_shape + o_shape)
    Wo  = Wo.reshape((size_W,) + o_shape)

    WCi = _grouped_einsum(xnp, use_jax, 'Cio,Wo->WCi', Cio, Wo)

    WCi = WCi.reshape(W_shape + C_shape + i_shape)
    return WCi


def dCio_dWo_to_dWCi(
        dCio: NDArray,
        dWo: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((dCio, dWo))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCio.shape[0],)
    C_shape = dCio.shape[1:-2]
    i_shape = (dCio.shape[-2],)
    o_shape = (dCio.shape[-1],)
    W_shape = dWo.shape[1:-1]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    dCio = dCio.reshape(d_shape + (size_C,) + i_shape + o_shape)
    dWo  = dWo.reshape(d_shape + (size_W,) + o_shape)

    dWCi = _grouped_einsum(xnp, use_jax, 'dCio,dWo->dWCi', dCio, dWo)

    dWCi = dWCi.reshape(d_shape + W_shape + C_shape + i_shape)
    return dWCi


def WCa_Caib_WCb_to_WCi(
        WCa: NDArray,
        Caib: NDArray,
        WCb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((WCa, Caib, WCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)
    W_shape = WCa.shape[:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    WCa     = WCa.reshape((size_W,) + (size_C,) + a_shape)
    Caib    = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    WCb     = WCb.reshape((size_W,) + (size_C,) + b_shape)

    WCi = _grouped_einsum(xnp, use_jax, 'WCa,Caib,WCb->WCi', WCa, Caib, WCb)

    WCi = WCi.reshape(W_shape + C_shape + i_shape)
    return WCi


def dWCa_dCaib_dWCb_to_dWCi(
        dWCa: NDArray,
        dCaib: NDArray,
        dWCb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((dWCa, dCaib, dWCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCaib.shape[0],)
    C_shape = dCaib.shape[1:-3]
    a_shape = (dCaib.shape[-3],)
    i_shape = (dCaib.shape[-2],)
    b_shape = (dCaib.shape[-1],)
    W_shape = dWCa.shape[1:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    dWCa    = dWCa.reshape(d_shape + (size_W,) + (size_C,) + a_shape)
    dCaib   = dCaib.reshape(d_shape + (size_C,) + a_shape + i_shape + b_shape)
    dWCb    = dWCb.reshape(d_shape + (size_W,) + (size_C,) + b_shape)

    dWCi = _grouped_einsum(xnp, use_jax, 'dWCa,dCaib,dWCb->dWCi', dWCa, dCaib, dWCb)

    dWCi = dWCi.reshape(d_shape + W_shape + C_shape + i_shape)
    return dWCi


def WCi_Cio_to_WCo(
        WCi: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((WCi, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    W_shape = WCi.shape[:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    Cio = Cio.reshape((size_C,) + i_shape + o_shape)
    WCi = WCi.reshape((size_W,) + (size_C,) + i_shape)

    WCo = _grouped_einsum(xnp, use_jax, 'WCi,Cio->WCo', WCi, Cio)

    WCo = WCo.reshape(W_shape + C_shape + o_shape)
    return WCo


def dWCi_dCio_to_dWCo(
        dWCi: NDArray,
        dCio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((dWCi, dCio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCio.shape[0],)
    C_shape = dCio.shape[1:-2]
    i_shape = (dCio.shape[-2],)
    o_shape = (dCio.shape[-1],)
    W_shape = dWCi.shape[1:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    dCio = dCio.reshape(d_shape + (size_C,) + i_shape + o_shape)
    dWCi  = dWCi.reshape(d_shape + (size_W,) + (size_C,) + i_shape)

    dWCo = _grouped_einsum(xnp, use_jax, 'dWCi,dCio->dWCo', dWCi, dCio)

    dWCo = dWCo.reshape(d_shape + W_shape + C_shape + o_shape)
    return dWCo


def WCo_Cio_to_WCi(
        WCo: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Unlike Cio_Wo_to_WCi (which forms an outer product over the two stacks), here C is a *shared*
    batch on both operands: Cio carries the T3 stack C only, WCo carries the probe stack W and C.
    Base-inner convention: W outermost, C innermost.
    """
    use_jax = tree_contains_jax((WCo, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    W_shape = WCo.shape[:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    Cio = Cio.reshape((size_C,) + i_shape + o_shape)
    WCo = WCo.reshape((size_W,) + (size_C,) + o_shape)

    WCi = _grouped_einsum(xnp, use_jax, 'WCo,Cio->WCi', WCo, Cio)

    WCi = WCi.reshape(W_shape + C_shape + i_shape)
    return WCi


def WCo_WCa_to_Cao(
        WCo: NDArray,
        WCa: NDArray,
        n_probe: int,
) -> NDArray:
    """Computes named contraction, summing over the probe stack W (kept on both operands, dropped
    from the output). Capital letters indicate grouped indices, which may be empty. n_probe is the
    number of leading (probe-stack) batch axes to sum over (base-inner: W outermost, C innermost).
    """
    use_jax = tree_contains_jax((WCo, WCa))
    xnp, _, _ = get_backend(True, use_jax)

    prefix = WCo.shape[:-1]
    W_shape = prefix[:n_probe]
    C_shape = prefix[n_probe:]
    o_shape = (WCo.shape[-1],)
    a_shape = (WCa.shape[-1],)

    size_C = math.prod(C_shape)
    size_W = math.prod(W_shape)

    WCo = WCo.reshape((size_W,) + (size_C,) + o_shape)
    WCa = WCa.reshape((size_W,) + (size_C,) + a_shape)

    Cao = _grouped_einsum(xnp, use_jax, 'WCo,WCa->Cao', WCo, WCa)

    Cao = Cao.reshape(C_shape + a_shape + o_shape)
    return Cao


def Wo_WCa_to_Cao(
        Wo: NDArray,
        WCa: NDArray,
) -> NDArray:
    """Computes named contraction, summing over the probe stack W (with Wo broadcast over the T3
    stack C). Capital letters indicate grouped indices, which may be empty (base-inner: W outer, C inner).
    """
    use_jax = tree_contains_jax((Wo, WCa))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wo.shape[:-1]
    o_shape = (Wo.shape[-1],)
    a_shape = (WCa.shape[-1],)
    C_shape = WCa.shape[len(W_shape):-1]

    size_C = math.prod(C_shape)
    size_W = math.prod(W_shape)

    Wo = Wo.reshape((size_W,) + o_shape)
    WCa = WCa.reshape((size_W,) + (size_C,) + a_shape)

    Cao = _grouped_einsum(xnp, use_jax, 'Wo,WCa->Cao', Wo, WCa)

    Cao = Cao.reshape(C_shape + a_shape + o_shape)
    return Cao


def WCi_WCa_WCj_to_Ciaj(
        WCi: NDArray,
        WCa: NDArray,
        WCj: NDArray,
        n_probe: int,
) -> NDArray:
    """Computes named contraction, summing over the probe stack W (kept on all operands, dropped from
    the output). Capital letters indicate grouped indices, which may be empty. n_probe is the number
    of leading (probe-stack) batch axes to sum over (base-inner: W outermost, C innermost).
    """
    use_jax = tree_contains_jax((WCi, WCa, WCj))
    xnp, _, _ = get_backend(True, use_jax)

    prefix = WCi.shape[:-1]
    W_shape = prefix[:n_probe]
    C_shape = prefix[n_probe:]
    i_shape = (WCi.shape[-1],)
    a_shape = (WCa.shape[-1],)
    j_shape = (WCj.shape[-1],)

    size_C = math.prod(C_shape)
    size_W = math.prod(W_shape)

    WCi = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WCa = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WCj = WCj.reshape((size_W,) + (size_C,) + j_shape)

    Ciaj = _grouped_einsum(xnp, use_jax, 'WCi,WCa,WCj->Ciaj', WCi, WCa, WCj)

    Ciaj = Ciaj.reshape(C_shape + i_shape + a_shape + j_shape)
    return Ciaj


###############################################################################
###############################################################################
# Symmetric probe-derivative contractions (the t derivative-order axis).
#
# The derivative order is a single leading axis t (lowercase: a single axis, not
# a grouped block). Each input vector carries a trivial jet -- value at order 0,
# direction at order 1, zero above -- because x + s*p is affine in s; so every
# product of jets is a binomial convolution driven by the static tensor
#     trs:  trs[t, r, s] = C(t, r) if r + s == t else 0   (binomial_combine_tensor).
# The pushthrough and combine are then the derivative-order analogs of probing's
# two core contractions WCa_Caib_WCi_to_WCb and WCa_Caib_WCb_to_WCi, with the
# binomial tensor threading the order axis (r, s contracted -> output order t).
# W (sample stack) and C (frame stack) ride exactly as in plain probing.
###############################################################################


def trs_rWCa_Caib_sWCi_to_tWCb(
        trs: NDArray,
        rWCa: NDArray,
        Caib: NDArray,
        sWCi: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Derivative-order pushthrough: the binomial jet-product of the left jet with the input jet through
    one core. The derivative-order analog of WCa_Caib_WCi_to_WCb, with the binomial tensor trs
    convolving the order axis (mu order r and input-jet order s -> output order t). Since the input
    jet is zero above order 1, s may be size 2 (slice trs[:, :, :2]).
    """
    use_jax = tree_contains_jax((trs, rWCa, Caib, sWCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)
    W_shape = rWCa.shape[1:-(len(C_shape) + 1)]

    t_shape = (trs.shape[0],)
    r_shape = (trs.shape[1],)
    s_shape = (trs.shape[2],)

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    rWCa = rWCa.reshape(r_shape + (size_W, size_C) + a_shape)
    Caib = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    sWCi = sWCi.reshape(s_shape + (size_W, size_C) + i_shape)

    tWCb = _grouped_einsum(xnp, use_jax, 'trs,rWCa,Caib,sWCi->tWCb', trs, rWCa, Caib, sWCi)

    tWCb = tWCb.reshape(t_shape + W_shape + C_shape + b_shape)
    return tWCb


def trs_rWCa_Caib_sWCb_to_tWCi(
        trs: NDArray,
        rWCa: NDArray,
        Caib: NDArray,
        sWCb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Derivative-order combine: the binomial jet-product of the left and right jets through one core,
    leaving mode i free. The derivative-order analog of WCa_Caib_WCb_to_WCi, with the binomial tensor
    trs convolving the order axis (mu order r and nu order s -> output order t).
    """
    use_jax = tree_contains_jax((trs, rWCa, Caib, sWCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)
    W_shape = rWCa.shape[1:-(len(C_shape) + 1)]

    t_shape = (trs.shape[0],)
    r_shape = (trs.shape[1],)
    s_shape = (trs.shape[2],)

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    rWCa = rWCa.reshape(r_shape + (size_W, size_C) + a_shape)
    Caib = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    sWCb = sWCb.reshape(s_shape + (size_W, size_C) + b_shape)

    tWCi = _grouped_einsum(xnp, use_jax, 'trs,rWCa,Caib,sWCb->tWCi', trs, rWCa, Caib, sWCb)

    tWCi = tWCi.reshape(t_shape + W_shape + C_shape + i_shape)
    return tWCi


def tWCi_Cio_to_tWCo(
        tWCi: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Derivative-order assemble: lift the combined jet to the ambient mode through the Tucker core. The
    derivative-order analog of WCi_Cio_to_WCo; the order axis t rides as a leading broadcast batch
    (the Tucker core is order-independent).
    """
    use_jax = tree_contains_jax((tWCi, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    W_shape = tWCi.shape[1:-(len(C_shape) + 1)]
    t_shape = (tWCi.shape[0],)

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    Cio  = Cio.reshape((size_C,) + i_shape + o_shape)
    tWCi = tWCi.reshape(t_shape + (size_W, size_C) + i_shape)

    tWCo = _grouped_einsum(xnp, use_jax, 'tWCi,Cio->tWCo', tWCi, Cio)

    tWCo = tWCo.reshape(t_shape + W_shape + C_shape + o_shape)
    return tWCo


###############################################################################
# Transpose (adjoint) of the symmetric probe derivatives -- the adjoint lift.
#
# The jet-ified adjoint sweeps and the order-less gradient assembly are the K-stacked 3-group
# (W,K,C) versions further down (trs_tWKCa_Caib_rWCi_to_sWKCb etc., which reduce to the 2-block case
# when K=()). Only the adjoint lift (contract the ambient mode, order t broadcast) lives here; it is
# delegated to from the 3-group tWKCo_Cio_to_tWKCi.
###############################################################################


def tWCo_Cio_to_tWCi(
        tWCo: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Adjoint lift: contract the ambient mode, order t rides through."""
    use_jax = tree_contains_jax((tWCo, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    W_shape = tWCo.shape[1:-(len(C_shape) + 1)]
    t_shape = (tWCo.shape[0],)

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    Cio  = Cio.reshape((size_C,) + i_shape + o_shape)
    tWCo = tWCo.reshape(t_shape + (size_W, size_C) + o_shape)

    tWCi = _grouped_einsum(xnp, use_jax, 'tWCo,Cio->tWCi', tWCo, Cio)

    return tWCi.reshape(t_shape + W_shape + C_shape + i_shape)


###############################################################################
# Three-group contractions (probing a K-stacked tangent).
#
# A third independent batch block K (a stack of tangent vectors sharing one base
# point) joins the probe stack W and frame stack C. Base-inner output order is
# W + K + C (W outer, K middle, C inner -- see docs/batching_and_stacking.md).
#
# The split is recovered from shapes, never passed from the frontend. A function
# self-infers when its operands include a C-only frame core (pins len(C)) and an
# W+C edge variable (pins len(W)). When the only "core" operand is a variation
# core (K+C) with no C-only operand present, len(C) is underdetermined by the
# operands alone, so n_frame is supplied (computed locally in the sweep _func from
# a C-only frame core it already holds -- the same precedent as n_probe above).
###############################################################################


def WKCa_Caib_WCi_to_WKCb(
        WKCa: NDArray,
        Caib: NDArray,
        WCi: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction. Self-infers the split: Caib (C-only) pins len(C),
    WCi (W+C) pins len(W), and K is the remainder of WKCa. K rides on WKCa and the output and
    broadcasts over the operands that lack it.
    """
    use_jax = tree_contains_jax((WKCa, Caib, WCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = WCi.shape[:-(len(C_shape) + 1)]
    K_shape = WKCa.shape[len(W_shape):-(len(C_shape) + 1)]

    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCa = WKCa.reshape((size_W,) + (size_K,) + (size_C,) + a_shape)
    Caib = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)

    WKCb = _grouped_einsum(xnp, use_jax, 'WKCa,Caib,WCi->WKCb', WKCa, Caib, WCi)

    WKCb = WKCb.reshape(W_shape + K_shape + C_shape + b_shape)
    return WKCb


def WCa_Caib_WKCi_to_WKCb(
        WCa: NDArray,
        Caib: NDArray,
        WKCi: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction. Self-infers the split: Caib (C-only) pins len(C),
    WCa (W+C) pins len(W), and K is the remainder of WKCi.
    """
    use_jax = tree_contains_jax((WCa, Caib, WKCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = WCa.shape[:-(len(C_shape) + 1)]
    K_shape = WKCi.shape[len(W_shape):-(len(C_shape) + 1)]

    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    Caib = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    WKCi = WKCi.reshape((size_W,) + (size_K,) + (size_C,) + i_shape)

    WKCb = _grouped_einsum(xnp, use_jax, 'WCa,Caib,WKCi->WKCb', WCa, Caib, WKCi)

    WKCb = WKCb.reshape(W_shape + K_shape + C_shape + b_shape)
    return WKCb


def WKCa_Caib_WCb_to_WKCi(
        WKCa: NDArray,
        Caib: NDArray,
        WCb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction. Self-infers the split: Caib (C-only) pins len(C),
    WCb (W+C) pins len(W), and K is the remainder of WKCa.
    """
    use_jax = tree_contains_jax((WKCa, Caib, WCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = WCb.shape[:-(len(C_shape) + 1)]
    K_shape = WKCa.shape[len(W_shape):-(len(C_shape) + 1)]

    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCa = WKCa.reshape((size_W,) + (size_K,) + (size_C,) + a_shape)
    Caib = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    WCb  = WCb.reshape((size_W,) + (size_C,) + b_shape)

    WKCi = _grouped_einsum(xnp, use_jax, 'WKCa,Caib,WCb->WKCi', WKCa, Caib, WCb)

    WKCi = WKCi.reshape(W_shape + K_shape + C_shape + i_shape)
    return WKCi


def WCa_Caib_WKCb_to_WKCi(
        WCa: NDArray,
        Caib: NDArray,
        WKCb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction. Self-infers the split: Caib (C-only) pins len(C),
    WCa (W+C) pins len(W), and K is the remainder of WKCb.
    """
    use_jax = tree_contains_jax((WCa, Caib, WKCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = WCa.shape[:-(len(C_shape) + 1)]
    K_shape = WKCb.shape[len(W_shape):-(len(C_shape) + 1)]

    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    Caib = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    WKCb = WKCb.reshape((size_W,) + (size_K,) + (size_C,) + b_shape)

    WKCi = _grouped_einsum(xnp, use_jax, 'WCa,Caib,WKCb->WKCi', WCa, Caib, WKCb)

    WKCi = WKCi.reshape(W_shape + K_shape + C_shape + i_shape)
    return WKCi


def WCa_KCaib_WCi_to_WKCb(
        WCa: NDArray,
        KCaib: NDArray,
        WCi: NDArray,
        n_frame: int,     # len(C). The only core operand (KCaib) is K+C, so len(C) cannot be
                         # recovered from these operands -- it is supplied (the n_probe precedent).
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction whose only core operand is a variation core (K+C).
    The operands {W+C, K+C, W+C} do not pin len(C), so it is passed as ``n_frame``.
    """
    use_jax = tree_contains_jax((WCa, KCaib, WCi))
    xnp, _, _ = get_backend(True, use_jax)

    KC_shape = KCaib.shape[:-3]
    C_shape = KC_shape[len(KC_shape) - n_frame:]
    K_shape = KC_shape[:len(KC_shape) - n_frame]
    W_shape = WCa.shape[:len(WCa.shape) - 1 - n_frame]

    a_shape = (KCaib.shape[-3],)
    i_shape = (KCaib.shape[-2],)
    b_shape = (KCaib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCa   = WCa.reshape((size_W,) + (size_C,) + a_shape)
    KCaib = KCaib.reshape((size_K,) + (size_C,) + a_shape + i_shape + b_shape)
    WCi   = WCi.reshape((size_W,) + (size_C,) + i_shape)

    WKCb = _grouped_einsum(xnp, use_jax, 'WCa,KCaib,WCi->WKCb', WCa, KCaib, WCi)

    WKCb = WKCb.reshape(W_shape + K_shape + C_shape + b_shape)
    return WKCb


def WCa_KCaib_WCb_to_WKCi(
        WCa: NDArray,
        KCaib: NDArray,
        WCb: NDArray,
        n_frame: int,     # len(C) (supplied; KCaib is K+C with no C-only operand to pin it).
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction whose only core operand is a variation core (K+C).
    The operands {W+C, K+C, W+C} do not pin len(C), so it is passed as ``n_frame``.
    """
    use_jax = tree_contains_jax((WCa, KCaib, WCb))
    xnp, _, _ = get_backend(True, use_jax)

    KC_shape = KCaib.shape[:-3]
    C_shape = KC_shape[len(KC_shape) - n_frame:]
    K_shape = KC_shape[:len(KC_shape) - n_frame]
    W_shape = WCa.shape[:len(WCa.shape) - 1 - n_frame]

    a_shape = (KCaib.shape[-3],)
    i_shape = (KCaib.shape[-2],)
    b_shape = (KCaib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCa   = WCa.reshape((size_W,) + (size_C,) + a_shape)
    KCaib = KCaib.reshape((size_K,) + (size_C,) + a_shape + i_shape + b_shape)
    WCb   = WCb.reshape((size_W,) + (size_C,) + b_shape)

    WKCi = _grouped_einsum(xnp, use_jax, 'WCa,KCaib,WCb->WKCi', WCa, KCaib, WCb)

    WKCi = WKCi.reshape(W_shape + K_shape + C_shape + i_shape)
    return WKCi


def WCi_KCio_to_WKCo(
        WCi: NDArray,
        KCio: NDArray,
        n_frame: int,     # len(C) (supplied; KCio is K+C with no C-only operand to pin it).
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction whose only core operand is a variation core (K+C).
    The operands {W+C, K+C} do not pin len(C), so it is passed as ``n_frame``.
    """
    use_jax = tree_contains_jax((WCi, KCio))
    xnp, _, _ = get_backend(True, use_jax)

    KC_shape = KCio.shape[:-2]
    C_shape = KC_shape[len(KC_shape) - n_frame:]
    K_shape = KC_shape[:len(KC_shape) - n_frame]
    W_shape = WCi.shape[:len(WCi.shape) - 1 - n_frame]

    i_shape = (KCio.shape[-2],)
    o_shape = (KCio.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    KCio = KCio.reshape((size_K,) + (size_C,) + i_shape + o_shape)

    WKCo = _grouped_einsum(xnp, use_jax, 'WCi,KCio->WKCo', WCi, KCio)

    WKCo = WKCo.reshape(W_shape + K_shape + C_shape + o_shape)
    return WKCo


def WKCi_Cio_to_WKCo(
        WKCi: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Lifts the down edge variable through the C-only frame Tucker core, contracting the TT-rank index
    i. C is SHARED (it is on both operands, so Cio pins len(C)) and gets a letter; W and K are
    PASSIVE (they live on WKCi alone and ride to the output untouched), so they ride as ``'...'``
    with every axis intact -- neither fused with the other nor flattened. See the SHARDING block in
    the module docstring.
    """
    use_jax = tree_contains_jax((WKCi, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    WK_shape = WKCi.shape[:-(len(C_shape) + 1)]     # W + K, kept as individual axes under '...'

    size_C = math.prod(C_shape)

    Cio  = Cio.reshape((size_C,) + i_shape + o_shape)
    WKCi = WKCi.reshape(WK_shape + (size_C,) + i_shape)

    WKCo = _grouped_einsum(xnp, use_jax, '...Ci,Cio->...Co', WKCi, Cio)

    WKCo = WKCo.reshape(WK_shape + C_shape + o_shape)
    return WKCo


###############################################################################
# Transpose-assemble three-group contractions (K-stacked tangent transpose).
#
# These build variation cores by OUTER-PRODUCTINC edge variables (the indices a/i/j/o are free, not
# contracted), the adjoint analogue of the forward assembly. The tangent stack K rides on the
# residual-derived operand; the frame edge vars stay W+C. Each comes in a keep-W (output W+K+C+...)
# and a sum-W (output K+C+..., the probe stack summed) form; the sum-W forms generalize the existing
# WCo_WCa_to_Cao / Wo_WCa_to_Cao / WCi_WCa_WCj_to_Ciaj (their K=() case).
#
# len(W) (n_probe) is supplied where no operand pins it; the w-bearing ones self-infer W from the
# W-only probe vector. See docs/batching_and_stacking.md and docs/entries_apply_probe.md section 8
# (the probing paper<->code map).
###############################################################################


def WKCo_WCa_to_WKCao(
        WKCo: NDArray,
        WCa: NDArray,
        n_probe: int,     # len(W); {W+K+C, W+C} do not pin it, so it is supplied
) -> NDArray:
    """Computes named contraction (outer product over a, o). Capitals are grouped indices, may be empty.

    Transpose-assemble (z-tilde (x) eta-hat), keeping the probe stack W.
    """
    use_jax = tree_contains_jax((WKCo, WCa))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCa.shape[:n_probe]
    C_shape = WCa.shape[n_probe:-1]
    o_shape = (WKCo.shape[-1],)
    a_shape = (WCa.shape[-1],)
    K_shape = WKCo.shape[n_probe:len(WKCo.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCo = WKCo.reshape((size_W,) + (size_K,) + (size_C,) + o_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)

    WKCao = _grouped_einsum(xnp, use_jax, 'WKCo,WCa->WKCao', WKCo, WCa)

    WKCao = WKCao.reshape(W_shape + K_shape + C_shape + a_shape + o_shape)
    return WKCao


def WKCo_WCa_to_KCao(
        WKCo: NDArray,
        WCa: NDArray,
        n_probe: int,     # len(W), summed out
) -> NDArray:
    """Computes named contraction (outer product over a, o; probe stack W summed out).

    Transpose-assemble (z-tilde (x) eta-hat), summing over the probe stack W.
    """
    use_jax = tree_contains_jax((WKCo, WCa))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCa.shape[:n_probe]
    C_shape = WCa.shape[n_probe:-1]
    o_shape = (WKCo.shape[-1],)
    a_shape = (WCa.shape[-1],)
    K_shape = WKCo.shape[n_probe:len(WKCo.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCo = WKCo.reshape((size_W,) + (size_K,) + (size_C,) + o_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)

    KCao = _grouped_einsum(xnp, use_jax, 'WKCo,WCa->KCao', WKCo, WCa)

    KCao = KCao.reshape(K_shape + C_shape + a_shape + o_shape)
    return KCao


def Wo_WKCa_to_WKCao(
        Wo: NDArray,
        WKCa: NDArray,
) -> NDArray:
    """Computes named contraction (outer product over a, o). Capitals are grouped indices, may be empty.

    Transpose-assemble (w (x) delta-xi-tilde), keeping the probe stack W. W is SHARED (on both operands,
    self-inferred from the W-only probe vector Wo) and gets a letter. K and C are PASSIVE (they live on
    WKCa alone and ride to the output), so they ride as ``'...'`` with every axis intact -- neither fused
    with each other nor flattened. This is also why no ``n_frame`` is needed: the K|C boundary is not
    pinned by the operands, and this contraction never needs it. See the SHARDING block in the module
    docstring.
    """
    use_jax = tree_contains_jax((Wo, WKCa))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wo.shape[:-1]
    o_shape = (Wo.shape[-1],)
    a_shape = (WKCa.shape[-1],)
    KC_shape = WKCa.shape[len(W_shape):-1]   # K + C, individual axes under '...'

    size_W = math.prod(W_shape)

    Wo   = Wo.reshape((size_W,) + o_shape)
    WKCa = WKCa.reshape((size_W,) + KC_shape + a_shape)

    WKCao = _grouped_einsum(xnp, use_jax, 'Wo,W...a->W...ao', Wo, WKCa)

    WKCao = WKCao.reshape(W_shape + KC_shape + a_shape + o_shape)
    return WKCao


def Wo_WKCa_to_KCao(
        Wo: NDArray,
        WKCa: NDArray,
) -> NDArray:
    """Computes named contraction (outer product over a, o; probe stack W summed out).

    Transpose-assemble (w (x) delta-xi-tilde), summing over the probe stack W. W is SHARED (on both
    operands, self-inferred from the W-only probe vector Wo) and gets a letter -- here it is summed, so it
    is absent from the output. K and C are PASSIVE (on WKCa alone, riding to the output), so they ride as
    ``'...'`` with every axis intact -- neither fused with each other nor flattened. See the SHARDING block
    in the module docstring.
    """
    use_jax = tree_contains_jax((Wo, WKCa))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wo.shape[:-1]
    o_shape = (Wo.shape[-1],)
    a_shape = (WKCa.shape[-1],)
    KC_shape = WKCa.shape[len(W_shape):-1]   # K + C, individual axes under '...'

    size_W = math.prod(W_shape)

    Wo   = Wo.reshape((size_W,) + o_shape)
    WKCa = WKCa.reshape((size_W,) + KC_shape + a_shape)

    KCao = _grouped_einsum(xnp, use_jax, 'Wo,W...a->...ao', Wo, WKCa)

    KCao = KCao.reshape(KC_shape + a_shape + o_shape)
    return KCao


def WCi_WCa_WKCj_to_WKCiaj(
        WCi: NDArray,
        WCa: NDArray,
        WKCj: NDArray,
        n_probe: int,     # len(W); {W+C, W+C, W+K+C} do not pin it, so it is supplied
) -> NDArray:
    """Computes named contraction (triple outer product over i, a, j). Capitals may be empty.

    Transpose tt-assemble term with K on the third (j) edge var, keeping the probe stack W.
    """
    use_jax = tree_contains_jax((WCi, WCa, WKCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCi.shape[:n_probe]
    C_shape = WCi.shape[n_probe:-1]
    i_shape = (WCi.shape[-1],)
    a_shape = (WCa.shape[-1],)
    j_shape = (WKCj.shape[-1],)
    K_shape = WKCj.shape[n_probe:len(WKCj.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WKCj = WKCj.reshape((size_W,) + (size_K,) + (size_C,) + j_shape)

    WKCiaj = _grouped_einsum(xnp, use_jax, 'WCi,WCa,WKCj->WKCiaj', WCi, WCa, WKCj)

    WKCiaj = WKCiaj.reshape(W_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return WKCiaj


def WCi_WCa_WKCj_to_KCiaj(
        WCi: NDArray,
        WCa: NDArray,
        WKCj: NDArray,
        n_probe: int,     # len(W), summed out
) -> NDArray:
    """Computes named contraction (triple outer product over i, a, j; probe stack W summed out).

    Transpose tt-assemble term with K on the third (j) edge var, summing over the probe stack W.
    """
    use_jax = tree_contains_jax((WCi, WCa, WKCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCi.shape[:n_probe]
    C_shape = WCi.shape[n_probe:-1]
    i_shape = (WCi.shape[-1],)
    a_shape = (WCa.shape[-1],)
    j_shape = (WKCj.shape[-1],)
    K_shape = WKCj.shape[n_probe:len(WKCj.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WKCj = WKCj.reshape((size_W,) + (size_K,) + (size_C,) + j_shape)

    KCiaj = _grouped_einsum(xnp, use_jax, 'WCi,WCa,WKCj->KCiaj', WCi, WCa, WKCj)

    KCiaj = KCiaj.reshape(K_shape + C_shape + i_shape + a_shape + j_shape)
    return KCiaj


def WKCi_WCa_WCj_to_WKCiaj(
        WKCi: NDArray,
        WCa: NDArray,
        WCj: NDArray,
        n_probe: int,     # len(W); supplied
) -> NDArray:
    """Computes named contraction (triple outer product over i, a, j). Capitals may be empty.

    Transpose tt-assemble term with K on the first (i) edge var, keeping the probe stack W.
    """
    use_jax = tree_contains_jax((WKCi, WCa, WCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCa.shape[:n_probe]
    C_shape = WCa.shape[n_probe:-1]
    i_shape = (WKCi.shape[-1],)
    a_shape = (WCa.shape[-1],)
    j_shape = (WCj.shape[-1],)
    K_shape = WKCi.shape[n_probe:len(WKCi.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCi = WKCi.reshape((size_W,) + (size_K,) + (size_C,) + i_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WCj  = WCj.reshape((size_W,) + (size_C,) + j_shape)

    WKCiaj = _grouped_einsum(xnp, use_jax, 'WKCi,WCa,WCj->WKCiaj', WKCi, WCa, WCj)

    WKCiaj = WKCiaj.reshape(W_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return WKCiaj


def WKCi_WCa_WCj_to_KCiaj(
        WKCi: NDArray,
        WCa: NDArray,
        WCj: NDArray,
        n_probe: int,     # len(W), summed out
) -> NDArray:
    """Computes named contraction (triple outer product over i, a, j; probe stack W summed out).

    Transpose tt-assemble term with K on the first (i) edge var, summing over the probe stack W.
    """
    use_jax = tree_contains_jax((WKCi, WCa, WCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCa.shape[:n_probe]
    C_shape = WCa.shape[n_probe:-1]
    i_shape = (WKCi.shape[-1],)
    a_shape = (WCa.shape[-1],)
    j_shape = (WCj.shape[-1],)
    K_shape = WKCi.shape[n_probe:len(WKCi.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCi = WKCi.reshape((size_W,) + (size_K,) + (size_C,) + i_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WCj  = WCj.reshape((size_W,) + (size_C,) + j_shape)

    KCiaj = _grouped_einsum(xnp, use_jax, 'WKCi,WCa,WCj->KCiaj', WKCi, WCa, WCj)

    KCiaj = KCiaj.reshape(K_shape + C_shape + i_shape + a_shape + j_shape)
    return KCiaj


def WCi_WKCa_WCj_to_WKCiaj(
        WCi: NDArray,
        WKCa: NDArray,
        WCj: NDArray,
        n_probe: int,     # len(W); supplied
) -> NDArray:
    """Computes named contraction (triple outer product over i, a, j). Capitals may be empty.

    Transpose tt-assemble term with K on the middle (a) edge var, keeping the probe stack W.
    """
    use_jax = tree_contains_jax((WCi, WKCa, WCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCi.shape[:n_probe]
    C_shape = WCi.shape[n_probe:-1]
    i_shape = (WCi.shape[-1],)
    a_shape = (WKCa.shape[-1],)
    j_shape = (WCj.shape[-1],)
    K_shape = WKCa.shape[n_probe:len(WKCa.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WKCa = WKCa.reshape((size_W,) + (size_K,) + (size_C,) + a_shape)
    WCj  = WCj.reshape((size_W,) + (size_C,) + j_shape)

    WKCiaj = _grouped_einsum(xnp, use_jax, 'WCi,WKCa,WCj->WKCiaj', WCi, WKCa, WCj)

    WKCiaj = WKCiaj.reshape(W_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return WKCiaj


def WCi_WKCa_WCj_to_KCiaj(
        WCi: NDArray,
        WKCa: NDArray,
        WCj: NDArray,
        n_probe: int,     # len(W), summed out
) -> NDArray:
    """Computes named contraction (triple outer product over i, a, j; probe stack W summed out).

    Transpose tt-assemble term with K on the middle (a) edge var, summing over the probe stack W.
    """
    use_jax = tree_contains_jax((WCi, WKCa, WCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCi.shape[:n_probe]
    C_shape = WCi.shape[n_probe:-1]
    i_shape = (WCi.shape[-1],)
    a_shape = (WKCa.shape[-1],)
    j_shape = (WCj.shape[-1],)
    K_shape = WKCa.shape[n_probe:len(WKCa.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WKCa = WKCa.reshape((size_W,) + (size_K,) + (size_C,) + a_shape)
    WCj  = WCj.reshape((size_W,) + (size_C,) + j_shape)

    KCiaj = _grouped_einsum(xnp, use_jax, 'WCi,WKCa,WCj->KCiaj', WCi, WKCa, WCj)

    KCiaj = KCiaj.reshape(K_shape + C_shape + i_shape + a_shape + j_shape)
    return KCiaj


###############################################################################
# Order-threaded three-group contractions (K-stacked derivative probing).
#
# The order-threaded jet pushthrough/combine (trs_rWCa_Caib_sWCi_to_tWCb etc.)
# with the tangent stack K added, exactly as WKCa_Caib_WCi_to_WKCb extends
# WCa_Caib_WCi_to_WCb: K rides on the one variation-derived operand (the swept
# jet, or the variation core), broadcasting over the others, base-inner output
# t + W + K + C. The split self-infers (C-only core pins len(C), a W+C jet pins
# len(W), K is the remainder); a variation-core-only term takes n_frame. Each
# reduces to its 2-group trs_... when K=(). Used by the K-aware perturbation
# sweep + assembly of the derivative-probe forward (sampling_derivatives.py).
###############################################################################


def trs_rWKCa_Caib_sWCi_to_tWKCb(
        trs: NDArray,
        rWKCa: NDArray,
        Caib: NDArray,
        sWCi: NDArray,
) -> NDArray:
    """Computes named contraction. Order-threaded 3-group (W,K,C) pushthrough; the K-stacked analog of
    trs_rWCa_Caib_sWCi_to_tWCb (K on the swept jet). Self-infers: Caib pins C, sWCi pins W, K=remainder."""
    use_jax = tree_contains_jax((trs, rWKCa, Caib, sWCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = sWCi.shape[1:-(len(C_shape) + 1)]
    K_shape = rWKCa.shape[1 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (Caib.shape[-3],), (Caib.shape[-2],), (Caib.shape[-1],)
    t_shape = (trs.shape[0],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    rWKCa = rWKCa.reshape((trs.shape[1], size_W, size_K, size_C) + a_shape)
    Caib  = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    sWCi  = sWCi.reshape((trs.shape[2], size_W, size_C) + i_shape)

    tWKCb = _grouped_einsum(xnp, use_jax, 'trs,rWKCa,Caib,sWCi->tWKCb', trs, rWKCa, Caib, sWCi)

    return tWKCb.reshape(t_shape + W_shape + K_shape + C_shape + b_shape)


def trs_rWCa_Caib_sWKCi_to_tWKCb(
        trs: NDArray,
        rWCa: NDArray,
        Caib: NDArray,
        sWKCi: NDArray,
) -> NDArray:
    """Computes named contraction. Order-threaded 3-group (W,K,C) pushthrough; the K-stacked analog of
    trs_rWCa_Caib_sWCi_to_tWCb (K on the input jet). Self-infers: Caib pins C, rWCa pins W, K=remainder."""
    use_jax = tree_contains_jax((trs, rWCa, Caib, sWKCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = rWCa.shape[1:-(len(C_shape) + 1)]
    K_shape = sWKCi.shape[1 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (Caib.shape[-3],), (Caib.shape[-2],), (Caib.shape[-1],)
    t_shape = (trs.shape[0],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    rWCa  = rWCa.reshape((trs.shape[1], size_W, size_C) + a_shape)
    Caib  = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    sWKCi = sWKCi.reshape((trs.shape[2], size_W, size_K, size_C) + i_shape)

    tWKCb = _grouped_einsum(xnp, use_jax, 'trs,rWCa,Caib,sWKCi->tWKCb', trs, rWCa, Caib, sWKCi)

    return tWKCb.reshape(t_shape + W_shape + K_shape + C_shape + b_shape)


def trs_rWCa_KCaib_sWCi_to_tWKCb(
        trs: NDArray,
        rWCa: NDArray,
        KCaib: NDArray,
        sWCi: NDArray,
        n_frame: int,     # len(C). Only core operand is K+C, so len(C) is supplied (n_probe precedent).
) -> NDArray:
    """Computes named contraction. Order-threaded 3-group (W,K,C) pushthrough; the K-stacked analog of
    trs_rWCa_Caib_sWCi_to_tWCb (K on the variation core). Operands {W+C, K+C, W+C} do not pin len(C) -> n_frame."""
    use_jax = tree_contains_jax((trs, rWCa, KCaib, sWCi))
    xnp, _, _ = get_backend(True, use_jax)

    KC_shape = KCaib.shape[:-3]
    C_shape = KC_shape[len(KC_shape) - n_frame:]
    K_shape = KC_shape[:len(KC_shape) - n_frame]
    W_shape = rWCa.shape[1:len(rWCa.shape) - 1 - n_frame]
    a_shape, i_shape, b_shape = (KCaib.shape[-3],), (KCaib.shape[-2],), (KCaib.shape[-1],)
    t_shape = (trs.shape[0],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    rWCa  = rWCa.reshape((trs.shape[1], size_W, size_C) + a_shape)
    KCaib = KCaib.reshape((size_K, size_C) + a_shape + i_shape + b_shape)
    sWCi  = sWCi.reshape((trs.shape[2], size_W, size_C) + i_shape)

    tWKCb = _grouped_einsum(xnp, use_jax, 'trs,rWCa,KCaib,sWCi->tWKCb', trs, rWCa, KCaib, sWCi)

    return tWKCb.reshape(t_shape + W_shape + K_shape + C_shape + b_shape)


def trs_rWKCa_Caib_sWCb_to_tWKCi(
        trs: NDArray,
        rWKCa: NDArray,
        Caib: NDArray,
        sWCb: NDArray,
) -> NDArray:
    """Computes named contraction. Order-threaded 3-group (W,K,C) combine; the K-stacked analog of
    trs_rWCa_Caib_sWCb_to_tWCi (K on the left jet). Self-infers: Caib pins C, sWCb pins W, K=remainder."""
    use_jax = tree_contains_jax((trs, rWKCa, Caib, sWCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = sWCb.shape[1:-(len(C_shape) + 1)]
    K_shape = rWKCa.shape[1 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (Caib.shape[-3],), (Caib.shape[-2],), (Caib.shape[-1],)
    t_shape = (trs.shape[0],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    rWKCa = rWKCa.reshape((trs.shape[1], size_W, size_K, size_C) + a_shape)
    Caib  = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    sWCb  = sWCb.reshape((trs.shape[2], size_W, size_C) + b_shape)

    tWKCi = _grouped_einsum(xnp, use_jax, 'trs,rWKCa,Caib,sWCb->tWKCi', trs, rWKCa, Caib, sWCb)

    return tWKCi.reshape(t_shape + W_shape + K_shape + C_shape + i_shape)


def trs_rWCa_Caib_sWKCb_to_tWKCi(
        trs: NDArray,
        rWCa: NDArray,
        Caib: NDArray,
        sWKCb: NDArray,
) -> NDArray:
    """Computes named contraction. Order-threaded 3-group (W,K,C) combine; the K-stacked analog of
    trs_rWCa_Caib_sWCb_to_tWCi (K on the right jet). Self-infers: Caib pins C, rWCa pins W, K=remainder."""
    use_jax = tree_contains_jax((trs, rWCa, Caib, sWKCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = rWCa.shape[1:-(len(C_shape) + 1)]
    K_shape = sWKCb.shape[1 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (Caib.shape[-3],), (Caib.shape[-2],), (Caib.shape[-1],)
    t_shape = (trs.shape[0],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    rWCa  = rWCa.reshape((trs.shape[1], size_W, size_C) + a_shape)
    Caib  = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    sWKCb = sWKCb.reshape((trs.shape[2], size_W, size_K, size_C) + b_shape)

    tWKCi = _grouped_einsum(xnp, use_jax, 'trs,rWCa,Caib,sWKCb->tWKCi', trs, rWCa, Caib, sWKCb)

    return tWKCi.reshape(t_shape + W_shape + K_shape + C_shape + i_shape)


def trs_rWCa_KCaib_sWCb_to_tWKCi(
        trs: NDArray,
        rWCa: NDArray,
        KCaib: NDArray,
        sWCb: NDArray,
        n_frame: int,     # len(C). Only core operand is K+C, so len(C) is supplied (n_probe precedent).
) -> NDArray:
    """Computes named contraction. Order-threaded 3-group (W,K,C) combine; the K-stacked analog of
    trs_rWCa_Caib_sWCb_to_tWCi (K on the variation core). Operands {W+C, K+C, W+C} do not pin len(C) -> n_frame."""
    use_jax = tree_contains_jax((trs, rWCa, KCaib, sWCb))
    xnp, _, _ = get_backend(True, use_jax)

    KC_shape = KCaib.shape[:-3]
    C_shape = KC_shape[len(KC_shape) - n_frame:]
    K_shape = KC_shape[:len(KC_shape) - n_frame]
    W_shape = rWCa.shape[1:len(rWCa.shape) - 1 - n_frame]
    a_shape, i_shape, b_shape = (KCaib.shape[-3],), (KCaib.shape[-2],), (KCaib.shape[-1],)
    t_shape = (trs.shape[0],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    rWCa  = rWCa.reshape((trs.shape[1], size_W, size_C) + a_shape)
    KCaib = KCaib.reshape((size_K, size_C) + a_shape + i_shape + b_shape)
    sWCb  = sWCb.reshape((trs.shape[2], size_W, size_C) + b_shape)

    tWKCi = _grouped_einsum(xnp, use_jax, 'trs,rWCa,KCaib,sWCb->tWKCi', trs, rWCa, KCaib, sWCb)

    return tWKCi.reshape(t_shape + W_shape + K_shape + C_shape + i_shape)


def tWKCi_Cio_to_tWKCo(
        tWKCi: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Order-threaded lift of the combined variation jet through the
    C-only frame Tucker core (no trs -- the order axis t is a passive broadcast).

    C is SHARED (on both operands; Cio pins len(C)) and gets a letter. t is a single passive axis
    and keeps its own letter. W and K are PASSIVE (on tWKCi alone, riding to the output), so they
    ride as ``'...'`` with every axis intact -- nothing is fused or flattened across them. See the
    SHARDING block in the module docstring."""
    use_jax = tree_contains_jax((tWKCi, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    t_shape = (tWKCi.shape[0],)
    WK_shape = tWKCi.shape[1:-(len(C_shape) + 1)]   # W + K, individual axes under '...'

    size_C = math.prod(C_shape)

    Cio   = Cio.reshape((size_C,) + i_shape + o_shape)
    tWKCi = tWKCi.reshape(t_shape + WK_shape + (size_C,) + i_shape)

    tWKCo = _grouped_einsum(xnp, use_jax, 't...Ci,Cio->t...Co', tWKCi, Cio)

    tWKCo = tWKCo.reshape(t_shape + WK_shape + C_shape + o_shape)
    return tWKCo


def tWCi_KCio_to_tWKCo(
        tWCi: NDArray,
        KCio: NDArray,
        n_frame: int,    # len(C). Only core operand is K+C, so len(C) is supplied (n_probe precedent).
) -> NDArray:
    """Computes named contraction. Order-threaded lift through a variation core; the lift has no trs, so
    the order axis t rides as a passive broadcast -- but it is kept as its OWN einsum letter rather than
    folded into the W block. Folding would flatten (t, W) with t MAJOR, forcing a reshard of a W-sharded
    array; see the SHARDING block in the module docstring."""
    use_jax = tree_contains_jax((tWCi, KCio))
    xnp, _, _ = get_backend(True, use_jax)

    KC_shape = KCio.shape[:-2]
    C_shape = KC_shape[len(KC_shape) - n_frame:]
    K_shape = KC_shape[:len(KC_shape) - n_frame]
    i_shape = (KCio.shape[-2],)
    o_shape = (KCio.shape[-1],)
    t_shape = (tWCi.shape[0],)
    W_shape = tWCi.shape[1:len(tWCi.shape) - 1 - n_frame]   # t held OUT of the W block

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    tWCi = tWCi.reshape(t_shape + (size_W, size_C) + i_shape)
    KCio = KCio.reshape((size_K, size_C) + i_shape + o_shape)

    tWKCo = _grouped_einsum(xnp, use_jax, 'tWCi,KCio->tWKCo', tWCi, KCio)

    tWKCo = tWKCo.reshape(t_shape + W_shape + K_shape + C_shape + o_shape)
    return tWKCo



###############################################################################
# Order-threaded three-group ADJOINT contractions (K-stacked derivative-probe
# transpose).
#
# The transpose's adjoint sweeps + gradient assembly with the tangent stack K
# added. The residual jets carry K, so the residual-derived adjoint variables
# (deta_tilde, tau_tilde, sigma_tilde, dxi_tilde) carry K; the frame edge vars
# (xi, mu, nu, eta) stay W+C. These are the order-threaded 3-block versions of
# the adjoint-hooked sweep contractions (trs_tWCa_Caib_uWCi_to_sWCb etc.) and
# the order-less assembly outer products (the dG / dU builders), each with K on
# the one residual-derived operand. Each reduces to its 2-block form when K=().
###############################################################################


def trs_tWKCa_Caib_rWCi_to_sWKCb(
        trs: NDArray,
        tWKCa: NDArray,
        Caib: NDArray,
        rWCi: NDArray,
) -> NDArray:
    """Computes named contraction. Order-threaded 3-group adjoint-hooked pushthrough (sweep
    propagation, output order s); the K-stacked analog of trs_tWCa_Caib_uWCi_to_sWCb (K on the swept
    adjoint jet). Self-infers: Caib pins C, rWCi pins W, K=remainder."""
    use_jax = tree_contains_jax((trs, tWKCa, Caib, rWCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = rWCi.shape[1:-(len(C_shape) + 1)]
    K_shape = tWKCa.shape[1 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (Caib.shape[-3],), (Caib.shape[-2],), (Caib.shape[-1],)
    s_shape = (trs.shape[2],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    tWKCa = tWKCa.reshape((trs.shape[0], size_W, size_K, size_C) + a_shape)
    Caib  = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    rWCi  = rWCi.reshape((trs.shape[1], size_W, size_C) + i_shape)

    sWKCb = _grouped_einsum(xnp, use_jax, 'trs,tWKCa,Caib,rWCi->sWKCb', trs, tWKCa, Caib, rWCi)

    return sWKCb.reshape(s_shape + W_shape + K_shape + C_shape + b_shape)


def trs_rWCa_Caib_tWKCi_to_sWKCb(
        trs: NDArray,
        rWCa: NDArray,
        Caib: NDArray,
        tWKCi: NDArray,
) -> NDArray:
    """Computes named contraction. Order-threaded 3-group adjoint-hooked deta_tilde source (output
    order s); the K-stacked analog of trs_rWCa_Caib_tWCi_to_sWCb (K on deta_tilde). Self-infers:
    Caib pins C, rWCa pins W, K=remainder."""
    use_jax = tree_contains_jax((trs, rWCa, Caib, tWKCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = rWCa.shape[1:-(len(C_shape) + 1)]
    K_shape = tWKCi.shape[1 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (Caib.shape[-3],), (Caib.shape[-2],), (Caib.shape[-1],)
    s_shape = (trs.shape[2],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    rWCa  = rWCa.reshape((trs.shape[1], size_W, size_C) + a_shape)
    Caib  = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    tWKCi = tWKCi.reshape((trs.shape[0], size_W, size_K, size_C) + i_shape)

    sWKCb = _grouped_einsum(xnp, use_jax, 'trs,rWCa,Caib,tWKCi->sWKCb', trs, rWCa, Caib, tWKCi)

    return sWKCb.reshape(s_shape + W_shape + K_shape + C_shape + b_shape)


def trs_tWKCa_Caib_sWCb_to_rWKCi(
        trs: NDArray,
        tWKCa: NDArray,
        Caib: NDArray,
        sWCb: NDArray,
) -> NDArray:
    """Computes named contraction. Order-threaded 3-group dxi_tilde-from-tau combine (output order u);
    the K-stacked analog of trs_tWCa_Caib_sWCb_to_uWCi (K on tau_tilde). Self-infers: Caib pins C,
    sWCb pins W, K=remainder."""
    use_jax = tree_contains_jax((trs, tWKCa, Caib, sWCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = sWCb.shape[1:-(len(C_shape) + 1)]
    K_shape = tWKCa.shape[1 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (Caib.shape[-3],), (Caib.shape[-2],), (Caib.shape[-1],)
    r_shape = (trs.shape[1],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    tWKCa = tWKCa.reshape((trs.shape[0], size_W, size_K, size_C) + a_shape)
    Caib  = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    sWCb  = sWCb.reshape((trs.shape[2], size_W, size_C) + b_shape)

    rWKCi = _grouped_einsum(xnp, use_jax, 'trs,tWKCa,Caib,sWCb->rWKCi', trs, tWKCa, Caib, sWCb)

    return rWKCi.reshape(r_shape + W_shape + K_shape + C_shape + i_shape)


def trs_rWCa_Caib_tWKCb_to_sWKCi(
        trs: NDArray,
        rWCa: NDArray,
        Caib: NDArray,
        tWKCb: NDArray,
) -> NDArray:
    """Computes named contraction. Order-threaded 3-group dxi_tilde-from-sigma combine (output order u);
    the K-stacked analog of trs_rWCa_Caib_tWCb_to_uWCi (K on sigma_tilde). Self-infers: Caib pins C,
    rWCa pins W, K=remainder."""
    use_jax = tree_contains_jax((trs, rWCa, Caib, tWKCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = rWCa.shape[1:-(len(C_shape) + 1)]
    K_shape = tWKCb.shape[1 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (Caib.shape[-3],), (Caib.shape[-2],), (Caib.shape[-1],)
    s_shape = (trs.shape[2],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    rWCa  = rWCa.reshape((trs.shape[1], size_W, size_C) + a_shape)
    Caib  = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    tWKCb = tWKCb.reshape((trs.shape[0], size_W, size_K, size_C) + b_shape)

    sWKCi = _grouped_einsum(xnp, use_jax, 'trs,rWCa,Caib,tWKCb->sWKCi', trs, rWCa, Caib, tWKCb)

    return sWKCi.reshape(s_shape + W_shape + K_shape + C_shape + i_shape)


def tWKCo_Cio_to_tWKCi(
        tWKCo: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Order-threaded adjoint lift: contract the ambient mode o against
    the C-only frame Tucker core, order t riding through as a passive broadcast.

    C is SHARED (on both operands; Cio pins len(C)) and gets a letter. t is a single passive axis
    and keeps its own letter. W and K are PASSIVE (on tWKCo alone, riding to the output), so they
    ride as ``'...'`` with every axis intact -- nothing is fused or flattened across them. See the
    SHARDING block in the module docstring."""
    use_jax = tree_contains_jax((tWKCo, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    t_shape = (tWKCo.shape[0],)
    WK_shape = tWKCo.shape[1:-(len(C_shape) + 1)]   # W + K, individual axes under '...'

    size_C = math.prod(C_shape)

    Cio   = Cio.reshape((size_C,) + i_shape + o_shape)
    tWKCo = tWKCo.reshape(t_shape + WK_shape + (size_C,) + o_shape)

    tWKCi = _grouped_einsum(xnp, use_jax, 't...Co,Cio->t...Ci', tWKCo, Cio)

    tWKCi = tWKCi.reshape(t_shape + WK_shape + C_shape + i_shape)
    return tWKCi


# ---- order-threaded 3-block gradient assembly (order summed; K on the residual-derived operand) ----

def _assemble_dG_jet3(trs, opA, opI, opB, einsum_str, k_op, n_probe, keep_W):
    '''Shared body for the six order-threaded 3-block dG_tilde core-adjoint outer products. opA/opI/opB
    carry the (a, i, b) legs; exactly one (k_op in {'A','I','B'}) carries the tangent stack K. C is
    pinned by a non-K operand (W+C); n_probe = len(W). The order axes are summed via trs; W kept or
    summed per keep_W. K=() recovers the 2-block trs_..._to_[W]Caib.'''
    use_jax = tree_contains_jax((trs, opA, opI, opB))
    xnp, _, _ = get_backend(True, use_jax)
    ops = {'A': opA, 'I': opI, 'B': opB}
    nonk = next(k for k in 'AIB' if k != k_op)
    W_shape = ops[nonk].shape[1:1 + n_probe]
    C_shape = ops[nonk].shape[1 + n_probe:-1]
    K_shape = ops[k_op].shape[1 + n_probe:-(1 + len(C_shape))]
    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    rs = {}
    for k in 'AIB':
        op = ops[k]
        if k == k_op:
            rs[k] = op.reshape((op.shape[0], size_W, size_K, size_C, op.shape[-1]))
        else:
            rs[k] = op.reshape((op.shape[0], size_W, size_C, op.shape[-1]))
    out = _grouped_einsum(xnp, use_jax, einsum_str, trs, rs['A'], rs['I'], rs['B'])
    tail = (opA.shape[-1], opI.shape[-1], opB.shape[-1])
    return out.reshape((W_shape if keep_W else ()) + K_shape + C_shape + tail)


def trs_rWCa_sWCi_tWKCb_to_WKCaib(trs, rWCa, sWCi, tWKCb, n_probe):
    """Computes named contraction. dG_tilde sigma-term (mu (x) xi (x) sigma_tilde), K on sigma_tilde, keep W."""
    return _assemble_dG_jet3(trs, rWCa, sWCi, tWKCb, 'trs,rWCa,sWCi,tWKCb->WKCaib', 'B', n_probe, True)


def trs_rWCa_sWCi_tWKCb_to_KCaib(trs, rWCa, sWCi, tWKCb, n_probe):
    """Computes named contraction. dG_tilde sigma-term (mu (x) xi (x) sigma_tilde), K on sigma_tilde, sum W."""
    return _assemble_dG_jet3(trs, rWCa, sWCi, tWKCb, 'trs,rWCa,sWCi,tWKCb->KCaib', 'B', n_probe, False)


def trs_tWKCa_rWCi_sWCb_to_WKCaib(trs, tWKCa, rWCi, sWCb, n_probe):
    """Computes named contraction. dG_tilde tau-term (tau_tilde (x) xi (x) nu), K on tau_tilde, keep W."""
    return _assemble_dG_jet3(trs, tWKCa, rWCi, sWCb, 'trs,tWKCa,rWCi,sWCb->WKCaib', 'A', n_probe, True)


def trs_tWKCa_rWCi_sWCb_to_KCaib(trs, tWKCa, rWCi, sWCb, n_probe):
    """Computes named contraction. dG_tilde tau-term (tau_tilde (x) xi (x) nu), K on tau_tilde, sum W."""
    return _assemble_dG_jet3(trs, tWKCa, rWCi, sWCb, 'trs,tWKCa,rWCi,sWCb->KCaib', 'A', n_probe, False)


def trs_rWCa_tWKCi_sWCb_to_WKCaib(trs, rWCa, tWKCi, sWCb, n_probe):
    """Computes named contraction. dG_tilde deta-term (mu (x) deta_tilde (x) nu), K on deta_tilde, keep W."""
    return _assemble_dG_jet3(trs, rWCa, tWKCi, sWCb, 'trs,rWCa,tWKCi,sWCb->WKCaib', 'I', n_probe, True)


def trs_rWCa_tWKCi_sWCb_to_KCaib(trs, rWCa, tWKCi, sWCb, n_probe):
    """Computes named contraction. dG_tilde deta-term (mu (x) deta_tilde (x) nu), K on deta_tilde, sum W."""
    return _assemble_dG_jet3(trs, rWCa, tWKCi, sWCb, 'trs,rWCa,tWKCi,sWCb->KCaib', 'I', n_probe, False)


def _assemble_dU_eta(tWCa, tWKCo, n_probe, keep_W):
    '''dU_tilde eta (x) r term (order-diagonal sum over t): eta-hat is W+C, the residual r carries K
    (W+K+C). C pinned by eta; K from r. K=() recovers tWCa_tWCo_to_[W]Cao.'''
    use_jax = tree_contains_jax((tWCa, tWKCo))
    xnp, _, _ = get_backend(True, use_jax)
    W_shape = tWCa.shape[1:1 + n_probe]
    C_shape = tWCa.shape[1 + n_probe:-1]
    a_shape, o_shape = (tWCa.shape[-1],), (tWKCo.shape[-1],)
    K_shape = tWKCo.shape[1 + n_probe:-(1 + len(C_shape))]
    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    tWCa  = tWCa.reshape((tWCa.shape[0], size_W, size_C) + a_shape)
    tWKCo = tWKCo.reshape((tWKCo.shape[0], size_W, size_K, size_C) + o_shape)
    out = _grouped_einsum(xnp, use_jax, 'tWCa,tWKCo->WKCao' if keep_W else 'tWCa,tWKCo->KCao', tWCa, tWKCo)
    return out.reshape((W_shape if keep_W else ()) + K_shape + C_shape + a_shape + o_shape)


def tWCa_tWKCo_to_WKCao(tWCa, tWKCo, n_probe):
    """Computes named contraction. dU_tilde eta (x) r term, K on the residual, keep W."""
    return _assemble_dU_eta(tWCa, tWKCo, n_probe, True)


def tWCa_tWKCo_to_KCao(tWCa, tWKCo, n_probe):
    """Computes named contraction. dU_tilde eta (x) r term, K on the residual, sum W."""
    return _assemble_dU_eta(tWCa, tWKCo, n_probe, False)


def _assemble_dU_dxi(uWKCa, uWo, keep_W):
    '''dU_tilde dxi_tilde (x) w_jet term (order-diagonal sum over u): dxi_tilde carries K (W+K+C), the
    w_jet is W-only (self-pins W, no C). K=() recovers uWCa_uWo_to_[W]Cao.

    W is SHARED (on both operands, self-pinned by the W-only w_jet) and gets a letter, as does the
    order axis u (summed). K and C are PASSIVE (on uWKCa alone, riding to the output in exactly the
    output dU core's K+C ordering), so they ride as ``'...'`` with every axis intact -- neither fused
    with each other nor flattened, which is also why their unpinnable split is never needed. See the
    SHARDING block in the module docstring.'''
    use_jax = tree_contains_jax((uWKCa, uWo))
    xnp, _, _ = get_backend(True, use_jax)
    W_shape = uWo.shape[1:-1]
    o_shape, a_shape = (uWo.shape[-1],), (uWKCa.shape[-1],)
    KC_shape = uWKCa.shape[1 + len(W_shape):-1]   # K + C, individual axes under '...'
    size_W = math.prod(W_shape)
    uWKCa = uWKCa.reshape((uWKCa.shape[0], size_W) + KC_shape + a_shape)
    uWo   = uWo.reshape((uWo.shape[0], size_W) + o_shape)
    out = _grouped_einsum(xnp, use_jax,
                          'uW...a,uWo->W...ao' if keep_W else 'uW...a,uWo->...ao', uWKCa, uWo)
    return out.reshape((W_shape if keep_W else ()) + KC_shape + a_shape + o_shape)


def uWKCa_uWo_to_WKCao(uWKCa, uWo):
    """Computes named contraction. dU_tilde dxi_tilde (x) w_jet term, K on dxi_tilde, keep W (self-pins W)."""
    return _assemble_dU_dxi(uWKCa, uWo, True)


def uWKCa_uWo_to_KCao(uWKCa, uWo):
    """Computes named contraction. dU_tilde dxi_tilde (x) w_jet term, K on dxi_tilde, sum W (self-pins W)."""
    return _assemble_dU_dxi(uWKCa, uWo, False)


def WCa_WCi_WKCb_to_WKCaib(
        WCa: NDArray,
        WCi: NDArray,
        WKCb: NDArray,
        n_probe: int,     # len(W); {W+C, W+C, W+K+C} do not pin it, so it is supplied
) -> NDArray:
    """Computes named contraction (triple outer product over a, i, b). Capitals may be empty.

    Adjoint-state apply/entries dG assemble ``mu (x) xi (x) sigma_hat`` (the order-0 strip of
    :py:func:`trs_rWCa_sWCi_tWKCb_to_WKCaib`; same structure as
    :py:func:`WCi_WCa_WKCj_to_WKCiaj` relabeled to the tt-core order (a, i, b)). K on the third
    (sigma_hat) operand, keeping the probe stack W.
    """
    use_jax = tree_contains_jax((WCa, WCi, WKCb))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCa.shape[:n_probe]
    C_shape = WCa.shape[n_probe:-1]
    a_shape = (WCa.shape[-1],)
    i_shape = (WCi.shape[-1],)
    b_shape = (WKCb.shape[-1],)
    K_shape = WKCb.shape[n_probe:len(WKCb.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WKCb = WKCb.reshape((size_W,) + (size_K,) + (size_C,) + b_shape)

    WKCaib = _grouped_einsum(xnp, use_jax, 'WCa,WCi,WKCb->WKCaib', WCa, WCi, WKCb)

    WKCaib = WKCaib.reshape(W_shape + K_shape + C_shape + a_shape + i_shape + b_shape)
    return WKCaib


def WCa_WCi_WKCb_to_KCaib(
        WCa: NDArray,
        WCi: NDArray,
        WKCb: NDArray,
        n_probe: int,     # len(W), summed out
) -> NDArray:
    """Computes named contraction (triple outer product over a, i, b; probe stack W summed out).

    Sum-over-probes form of :py:func:`WCa_WCi_WKCb_to_WKCaib` (the J^T r back-projection): the
    adjoint-state apply/entries dG assemble ``mu (x) xi (x) sigma_hat``, K on the third (sigma_hat)
    operand, summing over the probe stack W.
    """
    use_jax = tree_contains_jax((WCa, WCi, WKCb))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCa.shape[:n_probe]
    C_shape = WCa.shape[n_probe:-1]
    a_shape = (WCa.shape[-1],)
    i_shape = (WCi.shape[-1],)
    b_shape = (WKCb.shape[-1],)
    K_shape = WKCb.shape[n_probe:len(WKCb.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WKCb = WKCb.reshape((size_W,) + (size_K,) + (size_C,) + b_shape)

    KCaib = _grouped_einsum(xnp, use_jax, 'WCa,WCi,WKCb->KCaib', WCa, WCi, WKCb)

    KCaib = KCaib.reshape(K_shape + C_shape + a_shape + i_shape + b_shape)
    return KCaib


###############################################################################
# d-prefixed uniform WKC contractions (3b-6a). Each is the ragged WKC contraction (above) with the core
# index ``d`` prepended to every operand and the output, vectorized over ``d`` in a SINGLE einsum (NOT a
# per-core Python loop -- that would unroll under jit). The W/K/C block-flattening is identical; ``d`` rides
# as a leading batch axis (it appears in every operand and the output, never contracted), so the only change
# from the ragged twin is: a leading ``d_shape`` on every reshape, and the W/K/C front-counted slices shift
# by +1 to skip ``d``. These are the uniform-tangent map-style probing branches' contractions (the
# is_ndarray-dispatched path of compute_deta / assemble_tangent_z / compute_*_tildes / assemble_*).
#
# Order-0 note for 3b-6' (the jet/derivative slice): the binomial-jet ``trs_*`` analogs add the order axis
# ``t`` on top of this same ``d`` batch, and at order 0 reduce to exactly these contractions.
###############################################################################


def dWKCa_dCaib_dWCb_to_dWKCi(
        dWKCa: NDArray,
        dCaib: NDArray,
        dWCb: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WKCa_Caib_WCb_to_WKCi` (compute_deta term1)."""
    use_jax = tree_contains_jax((dWKCa, dCaib, dWCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCaib.shape[0],)
    C_shape = dCaib.shape[1:-3]
    W_shape = dWCb.shape[1:-(len(C_shape) + 1)]
    K_shape = dWKCa.shape[1 + len(W_shape):-(len(C_shape) + 1)]

    a_shape = (dCaib.shape[-3],)
    i_shape = (dCaib.shape[-2],)
    b_shape = (dCaib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWKCa = dWKCa.reshape(d_shape + (size_W, size_K, size_C) + a_shape)
    dCaib = dCaib.reshape(d_shape + (size_C,) + a_shape + i_shape + b_shape)
    dWCb  = dWCb.reshape(d_shape + (size_W, size_C) + b_shape)

    dWKCi = _grouped_einsum(xnp, use_jax, 'dWKCa,dCaib,dWCb->dWKCi', dWKCa, dCaib, dWCb)

    dWKCi = dWKCi.reshape(d_shape + W_shape + K_shape + C_shape + i_shape)
    return dWKCi


def dWCa_dCaib_dWKCb_to_dWKCi(
        dWCa: NDArray,
        dCaib: NDArray,
        dWKCb: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WCa_Caib_WKCb_to_WKCi` (compute_deta term3, compute_dxi_tilde)."""
    use_jax = tree_contains_jax((dWCa, dCaib, dWKCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCaib.shape[0],)
    C_shape = dCaib.shape[1:-3]
    W_shape = dWCa.shape[1:-(len(C_shape) + 1)]
    K_shape = dWKCb.shape[1 + len(W_shape):-(len(C_shape) + 1)]

    a_shape = (dCaib.shape[-3],)
    i_shape = (dCaib.shape[-2],)
    b_shape = (dCaib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWCa  = dWCa.reshape(d_shape + (size_W, size_C) + a_shape)
    dCaib = dCaib.reshape(d_shape + (size_C,) + a_shape + i_shape + b_shape)
    dWKCb = dWKCb.reshape(d_shape + (size_W, size_K, size_C) + b_shape)

    dWKCi = _grouped_einsum(xnp, use_jax, 'dWCa,dCaib,dWKCb->dWKCi', dWCa, dCaib, dWKCb)

    dWKCi = dWKCi.reshape(d_shape + W_shape + K_shape + C_shape + i_shape)
    return dWKCi


def dWCa_dKCaib_dWCb_to_dWKCi(
        dWCa: NDArray,
        dKCaib: NDArray,
        dWCb: NDArray,
        n_frame: int,      # len(C) (supplied; dKCaib is K+C with no C-only operand to pin it)
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WCa_KCaib_WCb_to_WKCi` (compute_deta term2)."""
    use_jax = tree_contains_jax((dWCa, dKCaib, dWCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dKCaib.shape[0],)
    KC_shape = dKCaib.shape[1:-3]
    C_shape = KC_shape[len(KC_shape) - n_frame:]
    K_shape = KC_shape[:len(KC_shape) - n_frame]
    W_shape = dWCa.shape[1:-(1 + n_frame)]

    a_shape = (dKCaib.shape[-3],)
    i_shape = (dKCaib.shape[-2],)
    b_shape = (dKCaib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWCa   = dWCa.reshape(d_shape + (size_W, size_C) + a_shape)
    dKCaib = dKCaib.reshape(d_shape + (size_K, size_C) + a_shape + i_shape + b_shape)
    dWCb   = dWCb.reshape(d_shape + (size_W, size_C) + b_shape)

    dWKCi = _grouped_einsum(xnp, use_jax, 'dWCa,dKCaib,dWCb->dWKCi', dWCa, dKCaib, dWCb)

    dWKCi = dWKCi.reshape(d_shape + W_shape + K_shape + C_shape + i_shape)
    return dWKCi


def dWKCi_dCio_to_dWKCo(
        dWKCi: NDArray,
        dCio: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WKCi_Cio_to_WKCo` (assemble_tangent_z term1). Lifts the down
    edge variable through the C-only frame Tucker supercore.

    ``C`` is SHARED (on both operands; ``dCio`` pins ``len(C)``) and gets a letter, as does the core index
    ``d``. ``W`` and ``K`` are PASSIVE (on ``dWKCi`` alone, riding to the output), so they ride as ``'...'``
    with every axis intact -- nothing fused or flattened across them. See the SHARDING block in the module
    docstring."""
    use_jax = tree_contains_jax((dWKCi, dCio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCio.shape[0],)
    C_shape = dCio.shape[1:-2]
    i_shape = (dCio.shape[-2],)
    o_shape = (dCio.shape[-1],)
    WK_shape = dWKCi.shape[1:-(len(C_shape) + 1)]   # W + K, individual axes under '...'

    size_C = math.prod(C_shape)

    dCio  = dCio.reshape(d_shape + (size_C,) + i_shape + o_shape)
    dWKCi = dWKCi.reshape(d_shape + WK_shape + (size_C,) + i_shape)

    dWKCo = _grouped_einsum(xnp, use_jax, 'd...Ci,dCio->d...Co', dWKCi, dCio)

    dWKCo = dWKCo.reshape(d_shape + WK_shape + C_shape + o_shape)
    return dWKCo


def dWCi_dKCio_to_dWKCo(
        dWCi: NDArray,
        dKCio: NDArray,
        n_frame: int,      # len(C) (supplied; dKCio is K+C with no C-only operand to pin it)
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WCi_KCio_to_WKCo` (assemble_tangent_z term2)."""
    use_jax = tree_contains_jax((dWCi, dKCio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dKCio.shape[0],)
    KC_shape = dKCio.shape[1:-2]
    C_shape = KC_shape[len(KC_shape) - n_frame:]
    K_shape = KC_shape[:len(KC_shape) - n_frame]
    W_shape = dWCi.shape[1:-(1 + n_frame)]

    i_shape = (dKCio.shape[-2],)
    o_shape = (dKCio.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWCi  = dWCi.reshape(d_shape + (size_W, size_C) + i_shape)
    dKCio = dKCio.reshape(d_shape + (size_K, size_C) + i_shape + o_shape)

    dWKCo = _grouped_einsum(xnp, use_jax, 'dWCi,dKCio->dWKCo', dWCi, dKCio)

    dWKCo = dWKCo.reshape(d_shape + W_shape + K_shape + C_shape + o_shape)
    return dWKCo


def dWCo_dCio_to_dWCi(
        dWCo: NDArray,
        dCio: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WCo_Cio_to_WCi` (compute_deta_tilde). C is a SHARED batch on
    both operands (the shared-C contraction, NOT the outer-product :py:func:`dCio_dWo_to_dWCi`)."""
    use_jax = tree_contains_jax((dWCo, dCio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCio.shape[0],)
    C_shape = dCio.shape[1:-2]
    i_shape = (dCio.shape[-2],)
    o_shape = (dCio.shape[-1],)
    W_shape = dWCo.shape[1:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    dCio = dCio.reshape(d_shape + (size_C,) + i_shape + o_shape)
    dWCo = dWCo.reshape(d_shape + (size_W, size_C) + o_shape)

    dWCi = _grouped_einsum(xnp, use_jax, 'dWCo,dCio->dWCi', dWCo, dCio)

    dWCi = dWCi.reshape(d_shape + W_shape + C_shape + i_shape)
    return dWCi


def dWKCo_dWCa_to_dWKCao(
        dWKCo: NDArray,
        dWCa: NDArray,
        n_probe: int,      # len(W); {W+K+C, W+C} do not pin it, so supplied
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WKCo_WCa_to_WKCao` (assemble_tucker_variations, keep W)."""
    use_jax = tree_contains_jax((dWKCo, dWCa))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWCa.shape[0],)
    W_shape = dWCa.shape[1:1 + n_probe]
    C_shape = dWCa.shape[1 + n_probe:-1]
    o_shape = (dWKCo.shape[-1],)
    a_shape = (dWCa.shape[-1],)
    K_shape = dWKCo.shape[1 + n_probe:len(dWKCo.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWKCo = dWKCo.reshape(d_shape + (size_W, size_K, size_C) + o_shape)
    dWCa  = dWCa.reshape(d_shape + (size_W, size_C) + a_shape)

    dWKCao = _grouped_einsum(xnp, use_jax, 'dWKCo,dWCa->dWKCao', dWKCo, dWCa)

    dWKCao = dWKCao.reshape(d_shape + W_shape + K_shape + C_shape + a_shape + o_shape)
    return dWKCao


def dWKCo_dWCa_to_dKCao(
        dWKCo: NDArray,
        dWCa: NDArray,
        n_probe: int,      # len(W), summed out
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WKCo_WCa_to_KCao` (assemble_tucker_variations, sum W)."""
    use_jax = tree_contains_jax((dWKCo, dWCa))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWCa.shape[0],)
    W_shape = dWCa.shape[1:1 + n_probe]
    C_shape = dWCa.shape[1 + n_probe:-1]
    o_shape = (dWKCo.shape[-1],)
    a_shape = (dWCa.shape[-1],)
    K_shape = dWKCo.shape[1 + n_probe:len(dWKCo.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWKCo = dWKCo.reshape(d_shape + (size_W, size_K, size_C) + o_shape)
    dWCa  = dWCa.reshape(d_shape + (size_W, size_C) + a_shape)

    dKCao = _grouped_einsum(xnp, use_jax, 'dWKCo,dWCa->dKCao', dWKCo, dWCa)

    dKCao = dKCao.reshape(d_shape + K_shape + C_shape + a_shape + o_shape)
    return dKCao


def dWo_dWKCa_to_dWKCao(
        dWo: NDArray,
        dWKCa: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`Wo_WKCa_to_WKCao` (assemble_tucker_variations, keep W).

    ``W`` is SHARED (on both operands, self-inferred from the W-only probe supercore ``dWo``) and gets a
    letter, as does the core index ``d``. ``K`` and ``C`` are PASSIVE (on ``dWKCa`` alone, riding to the
    output), so they ride as ``'...'`` with every axis intact -- neither fused with each other nor
    flattened. See the SHARDING block in the module docstring."""
    use_jax = tree_contains_jax((dWo, dWKCa))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWo.shape[0],)
    W_shape = dWo.shape[1:-1]
    o_shape = (dWo.shape[-1],)
    a_shape = (dWKCa.shape[-1],)
    KC_shape = dWKCa.shape[1 + len(W_shape):-1]   # K + C, individual axes under '...'

    size_W = math.prod(W_shape)

    dWo   = dWo.reshape(d_shape + (size_W,) + o_shape)
    dWKCa = dWKCa.reshape(d_shape + (size_W,) + KC_shape + a_shape)

    dWKCao = _grouped_einsum(xnp, use_jax, 'dWo,dW...a->dW...ao', dWo, dWKCa)

    dWKCao = dWKCao.reshape(d_shape + W_shape + KC_shape + a_shape + o_shape)
    return dWKCao


def dWo_dWKCa_to_dKCao(
        dWo: NDArray,
        dWKCa: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`Wo_WKCa_to_KCao` (assemble_tucker_variations, sum W).

    ``W`` is SHARED (on both operands, self-inferred from the W-only probe supercore ``dWo``) and gets a
    letter -- here it is summed, so it is absent from the output; the core index ``d`` keeps its letter.
    ``K`` and ``C`` are PASSIVE (on ``dWKCa`` alone, riding to the output), so they ride as ``'...'`` with
    every axis intact -- neither fused with each other nor flattened. See the SHARDING block in the module
    docstring."""
    use_jax = tree_contains_jax((dWo, dWKCa))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWo.shape[0],)
    W_shape = dWo.shape[1:-1]
    o_shape = (dWo.shape[-1],)
    a_shape = (dWKCa.shape[-1],)
    KC_shape = dWKCa.shape[1 + len(W_shape):-1]   # K + C, individual axes under '...'

    size_W = math.prod(W_shape)

    dWo   = dWo.reshape(d_shape + (size_W,) + o_shape)
    dWKCa = dWKCa.reshape(d_shape + (size_W,) + KC_shape + a_shape)

    dKCao = _grouped_einsum(xnp, use_jax, 'dWo,dW...a->d...ao', dWo, dWKCa)

    dKCao = dKCao.reshape(d_shape + KC_shape + a_shape + o_shape)
    return dKCao


def dWCi_dWCa_dWKCj_to_dWKCiaj(
        dWCi: NDArray,
        dWCa: NDArray,
        dWKCj: NDArray,
        n_probe: int,      # len(W); supplied
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WCi_WCa_WKCj_to_WKCiaj` (assemble_tt_variations, K on j, keep W)."""
    use_jax = tree_contains_jax((dWCi, dWCa, dWKCj))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWCi.shape[0],)
    W_shape = dWCi.shape[1:1 + n_probe]
    C_shape = dWCi.shape[1 + n_probe:-1]
    i_shape = (dWCi.shape[-1],)
    a_shape = (dWCa.shape[-1],)
    j_shape = (dWKCj.shape[-1],)
    K_shape = dWKCj.shape[1 + n_probe:len(dWKCj.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWCi  = dWCi.reshape(d_shape + (size_W, size_C) + i_shape)
    dWCa  = dWCa.reshape(d_shape + (size_W, size_C) + a_shape)
    dWKCj = dWKCj.reshape(d_shape + (size_W, size_K, size_C) + j_shape)

    dWKCiaj = _grouped_einsum(xnp, use_jax, 'dWCi,dWCa,dWKCj->dWKCiaj', dWCi, dWCa, dWKCj)

    dWKCiaj = dWKCiaj.reshape(d_shape + W_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return dWKCiaj


def dWCi_dWCa_dWKCj_to_dKCiaj(
        dWCi: NDArray,
        dWCa: NDArray,
        dWKCj: NDArray,
        n_probe: int,      # len(W), summed out
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WCi_WCa_WKCj_to_KCiaj` (assemble_tt_variations, K on j, sum W)."""
    use_jax = tree_contains_jax((dWCi, dWCa, dWKCj))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWCi.shape[0],)
    W_shape = dWCi.shape[1:1 + n_probe]
    C_shape = dWCi.shape[1 + n_probe:-1]
    i_shape = (dWCi.shape[-1],)
    a_shape = (dWCa.shape[-1],)
    j_shape = (dWKCj.shape[-1],)
    K_shape = dWKCj.shape[1 + n_probe:len(dWKCj.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWCi  = dWCi.reshape(d_shape + (size_W, size_C) + i_shape)
    dWCa  = dWCa.reshape(d_shape + (size_W, size_C) + a_shape)
    dWKCj = dWKCj.reshape(d_shape + (size_W, size_K, size_C) + j_shape)

    dKCiaj = _grouped_einsum(xnp, use_jax, 'dWCi,dWCa,dWKCj->dKCiaj', dWCi, dWCa, dWKCj)

    dKCiaj = dKCiaj.reshape(d_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return dKCiaj


def dWKCi_dWCa_dWCj_to_dWKCiaj(
        dWKCi: NDArray,
        dWCa: NDArray,
        dWCj: NDArray,
        n_probe: int,      # len(W); supplied
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WKCi_WCa_WCj_to_WKCiaj` (assemble_tt_variations, K on i, keep W)."""
    use_jax = tree_contains_jax((dWKCi, dWCa, dWCj))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWCa.shape[0],)
    W_shape = dWCa.shape[1:1 + n_probe]
    C_shape = dWCa.shape[1 + n_probe:-1]
    i_shape = (dWKCi.shape[-1],)
    a_shape = (dWCa.shape[-1],)
    j_shape = (dWCj.shape[-1],)
    K_shape = dWKCi.shape[1 + n_probe:len(dWKCi.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWKCi = dWKCi.reshape(d_shape + (size_W, size_K, size_C) + i_shape)
    dWCa  = dWCa.reshape(d_shape + (size_W, size_C) + a_shape)
    dWCj  = dWCj.reshape(d_shape + (size_W, size_C) + j_shape)

    dWKCiaj = _grouped_einsum(xnp, use_jax, 'dWKCi,dWCa,dWCj->dWKCiaj', dWKCi, dWCa, dWCj)

    dWKCiaj = dWKCiaj.reshape(d_shape + W_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return dWKCiaj


def dWKCi_dWCa_dWCj_to_dKCiaj(
        dWKCi: NDArray,
        dWCa: NDArray,
        dWCj: NDArray,
        n_probe: int,      # len(W), summed out
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WKCi_WCa_WCj_to_KCiaj` (assemble_tt_variations, K on i, sum W)."""
    use_jax = tree_contains_jax((dWKCi, dWCa, dWCj))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWCa.shape[0],)
    W_shape = dWCa.shape[1:1 + n_probe]
    C_shape = dWCa.shape[1 + n_probe:-1]
    i_shape = (dWKCi.shape[-1],)
    a_shape = (dWCa.shape[-1],)
    j_shape = (dWCj.shape[-1],)
    K_shape = dWKCi.shape[1 + n_probe:len(dWKCi.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWKCi = dWKCi.reshape(d_shape + (size_W, size_K, size_C) + i_shape)
    dWCa  = dWCa.reshape(d_shape + (size_W, size_C) + a_shape)
    dWCj  = dWCj.reshape(d_shape + (size_W, size_C) + j_shape)

    dKCiaj = _grouped_einsum(xnp, use_jax, 'dWKCi,dWCa,dWCj->dKCiaj', dWKCi, dWCa, dWCj)

    dKCiaj = dKCiaj.reshape(d_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return dKCiaj


def dWCi_dWKCa_dWCj_to_dWKCiaj(
        dWCi: NDArray,
        dWKCa: NDArray,
        dWCj: NDArray,
        n_probe: int,      # len(W); supplied
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WCi_WKCa_WCj_to_WKCiaj` (assemble_tt_variations, K on a, keep W)."""
    use_jax = tree_contains_jax((dWCi, dWKCa, dWCj))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWCi.shape[0],)
    W_shape = dWCi.shape[1:1 + n_probe]
    C_shape = dWCi.shape[1 + n_probe:-1]
    i_shape = (dWCi.shape[-1],)
    a_shape = (dWKCa.shape[-1],)
    j_shape = (dWCj.shape[-1],)
    K_shape = dWKCa.shape[1 + n_probe:len(dWKCa.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWCi  = dWCi.reshape(d_shape + (size_W, size_C) + i_shape)
    dWKCa = dWKCa.reshape(d_shape + (size_W, size_K, size_C) + a_shape)
    dWCj  = dWCj.reshape(d_shape + (size_W, size_C) + j_shape)

    dWKCiaj = _grouped_einsum(xnp, use_jax, 'dWCi,dWKCa,dWCj->dWKCiaj', dWCi, dWKCa, dWCj)

    dWKCiaj = dWKCiaj.reshape(d_shape + W_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return dWKCiaj


def dWCi_dWKCa_dWCj_to_dKCiaj(
        dWCi: NDArray,
        dWKCa: NDArray,
        dWCj: NDArray,
        n_probe: int,      # len(W), summed out
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WCi_WKCa_WCj_to_KCiaj` (assemble_tt_variations, K on a, sum W)."""
    use_jax = tree_contains_jax((dWCi, dWKCa, dWCj))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWCi.shape[0],)
    W_shape = dWCi.shape[1:1 + n_probe]
    C_shape = dWCi.shape[1 + n_probe:-1]
    i_shape = (dWCi.shape[-1],)
    a_shape = (dWKCa.shape[-1],)
    j_shape = (dWCj.shape[-1],)
    K_shape = dWKCa.shape[1 + n_probe:len(dWKCa.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWCi  = dWCi.reshape(d_shape + (size_W, size_C) + i_shape)
    dWKCa = dWKCa.reshape(d_shape + (size_W, size_K, size_C) + a_shape)
    dWCj  = dWCj.reshape(d_shape + (size_W, size_C) + j_shape)

    dKCiaj = _grouped_einsum(xnp, use_jax, 'dWCi,dWKCa,dWCj->dKCiaj', dWCi, dWKCa, dWCj)

    dKCiaj = dKCiaj.reshape(d_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return dKCiaj


def dWCa_dWCi_dWKCb_to_dWKCaib(
        dWCa: NDArray,
        dWCi: NDArray,
        dWKCb: NDArray,
        n_probe: int,      # len(W); supplied
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WCa_WCi_WKCb_to_WKCaib` (apply/entries adjoint dG assemble, keep W)."""
    use_jax = tree_contains_jax((dWCa, dWCi, dWKCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWCa.shape[0],)
    W_shape = dWCa.shape[1:1 + n_probe]
    C_shape = dWCa.shape[1 + n_probe:-1]
    a_shape = (dWCa.shape[-1],)
    i_shape = (dWCi.shape[-1],)
    b_shape = (dWKCb.shape[-1],)
    K_shape = dWKCb.shape[1 + n_probe:len(dWKCb.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWCa  = dWCa.reshape(d_shape + (size_W, size_C) + a_shape)
    dWCi  = dWCi.reshape(d_shape + (size_W, size_C) + i_shape)
    dWKCb = dWKCb.reshape(d_shape + (size_W, size_K, size_C) + b_shape)

    dWKCaib = _grouped_einsum(xnp, use_jax, 'dWCa,dWCi,dWKCb->dWKCaib', dWCa, dWCi, dWKCb)

    dWKCaib = dWKCaib.reshape(d_shape + W_shape + K_shape + C_shape + a_shape + i_shape + b_shape)
    return dWKCaib


def dWCa_dWCi_dWKCb_to_dKCaib(
        dWCa: NDArray,
        dWCi: NDArray,
        dWKCb: NDArray,
        n_probe: int,      # len(W), summed out
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`WCa_WCi_WKCb_to_KCaib` (apply/entries adjoint dG assemble, sum W)."""
    use_jax = tree_contains_jax((dWCa, dWCi, dWKCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dWCa.shape[0],)
    W_shape = dWCa.shape[1:1 + n_probe]
    C_shape = dWCa.shape[1 + n_probe:-1]
    a_shape = (dWCa.shape[-1],)
    i_shape = (dWCi.shape[-1],)
    b_shape = (dWKCb.shape[-1],)
    K_shape = dWKCb.shape[1 + n_probe:len(dWKCb.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dWCa  = dWCa.reshape(d_shape + (size_W, size_C) + a_shape)
    dWCi  = dWCi.reshape(d_shape + (size_W, size_C) + i_shape)
    dWKCb = dWKCb.reshape(d_shape + (size_W, size_K, size_C) + b_shape)

    dKCaib = _grouped_einsum(xnp, use_jax, 'dWCa,dWCi,dWKCb->dKCaib', dWCa, dWCi, dWKCb)

    dKCaib = dKCaib.reshape(d_shape + K_shape + C_shape + a_shape + i_shape + b_shape)
    return dKCaib


###############################################################################
# d-prefixed uniform JET contractions (3b-6'a). The binomial-jet ``trs_*`` analogs (above) with the core
# index ``d`` prepended to every jet/core operand and the output, vectorized over ``d`` in a SINGLE einsum
# (NOT a per-core map -- that would unroll under jit). Uniform jet layout is ``(d,) + (order,) + W + K + C
# + (...)`` -- ``d`` leads, THEN the order axis (from build_input_jets' axis-1 stack / the xscan's d-stack),
# then the passive W/K/C blocks. So each is its ragged jet twin with: a leading ``d_shape`` on every reshape,
# and the W/K/C front-counted slices shifted by +1 (to skip ``d``, on top of the +1 the order axis already
# takes). ``trs`` (the binomial tensor) is shared across cores -> it carries NO ``d``. The order axes
# (r,s,t,u) are contracted through ``trs`` (pushthrough/combine/assembly) or summed (order-diagonal dU
# assembly), so -- unlike the passive-broadcast Tucker LIFTS below -- they cannot fold into the W block.
#
# Order-0 anchor: at order 0 every one reduces to the plain ``dWKC`` contraction verified in 3b-6a.
# These back the is_ndarray-dispatched uniform branches of the map-style jet fns in sampling_derivatives.py.
###############################################################################


# ---- forward combine + lift (mode i free; the eta / z jet builders) ----

def trs_drWCa_dCaib_dsWCb_to_dtWCi(
        trs: NDArray,
        drWCa: NDArray,
        dCaib: NDArray,
        dsWCb: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`trs_rWCa_Caib_sWCb_to_tWCi` (plain compute_eta_jets)."""
    use_jax = tree_contains_jax((trs, drWCa, dCaib, dsWCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCaib.shape[0],)
    C_shape = dCaib.shape[1:-3]
    W_shape = drWCa.shape[2:-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (dCaib.shape[-3],), (dCaib.shape[-2],), (dCaib.shape[-1],)
    t_shape = (trs.shape[0],)

    size_W, size_C = math.prod(W_shape), math.prod(C_shape)
    drWCa = drWCa.reshape(d_shape + (trs.shape[1], size_W, size_C) + a_shape)
    dCaib = dCaib.reshape(d_shape + (size_C,) + a_shape + i_shape + b_shape)
    dsWCb = dsWCb.reshape(d_shape + (trs.shape[2], size_W, size_C) + b_shape)

    dtWCi = _grouped_einsum(xnp, use_jax, 'trs,drWCa,dCaib,dsWCb->dtWCi', trs, drWCa, dCaib, dsWCb)

    return dtWCi.reshape(d_shape + t_shape + W_shape + C_shape + i_shape)


def dtWCi_dCio_to_dtWCo(
        dtWCi: NDArray,
        dCio: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`tWCi_Cio_to_tWCo` (plain assemble_z_jets). The Tucker lift has
    no ``trs``, so the order axis ``t`` rides as a passive broadcast -- but it is kept as its OWN einsum
    letter rather than folded into the W block. Folding would flatten ``(t, W)`` with ``t`` MAJOR, forcing
    a reshard of a W-sharded array; see the SHARDING block in the module docstring."""
    use_jax = tree_contains_jax((dtWCi, dCio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCio.shape[0],)
    C_shape = dCio.shape[1:-2]
    i_shape = (dCio.shape[-2],)
    o_shape = (dCio.shape[-1],)
    t_shape = (dtWCi.shape[1],)
    W_shape = dtWCi.shape[2:-(len(C_shape) + 1)]   # t held OUT of the W block

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    dCio  = dCio.reshape(d_shape + (size_C,) + i_shape + o_shape)
    dtWCi = dtWCi.reshape(d_shape + t_shape + (size_W, size_C) + i_shape)

    dtWCo = _grouped_einsum(xnp, use_jax, 'dtWCi,dCio->dtWCo', dtWCi, dCio)

    dtWCo = dtWCo.reshape(d_shape + t_shape + W_shape + C_shape + o_shape)
    return dtWCo


def dtWKCi_dCio_to_dtWKCo(
        dtWKCi: NDArray,
        dCio: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`tWKCi_Cio_to_tWKCo` (assemble_tangent_z_jets term1). Passive
    order-threaded Tucker lift (no ``trs``) through the C-only frame Tucker supercore.

    ``C`` is SHARED (on both operands; ``dCio`` pins ``len(C)``) and gets a letter, as do the single passive
    axes ``d`` and ``t``. ``W`` and ``K`` are PASSIVE (on ``dtWKCi`` alone, riding to the output), so they
    ride as ``'...'`` with every axis intact -- nothing fused or flattened across them. See the SHARDING
    block in the module docstring."""
    use_jax = tree_contains_jax((dtWKCi, dCio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCio.shape[0],)
    C_shape = dCio.shape[1:-2]
    i_shape = (dCio.shape[-2],)
    o_shape = (dCio.shape[-1],)
    t_shape = (dtWKCi.shape[1],)
    WK_shape = dtWKCi.shape[2:-(len(C_shape) + 1)]   # W + K, individual axes under '...'

    size_C = math.prod(C_shape)

    dCio   = dCio.reshape(d_shape + (size_C,) + i_shape + o_shape)
    dtWKCi = dtWKCi.reshape(d_shape + t_shape + WK_shape + (size_C,) + i_shape)

    dtWKCo = _grouped_einsum(xnp, use_jax, 'dt...Ci,dCio->dt...Co', dtWKCi, dCio)

    dtWKCo = dtWKCo.reshape(d_shape + t_shape + WK_shape + C_shape + o_shape)
    return dtWKCo


def dtWCi_dKCio_to_dtWKCo(
        dtWCi: NDArray,
        dKCio: NDArray,
        n_frame: int,      # len(C) (dKCio is K+C with no C-only operand to pin it)
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`tWCi_KCio_to_tWKCo` (assemble_tangent_z_jets term2). The lift
    has no ``trs``, so ``t`` rides as a passive broadcast -- but it is kept as its OWN einsum letter rather
    than folded into the W block, which would flatten ``(t, W)`` with ``t`` MAJOR and force a reshard of a
    W-sharded array; see the SHARDING block in the module docstring."""
    use_jax = tree_contains_jax((dtWCi, dKCio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dKCio.shape[0],)
    KC_shape = dKCio.shape[1:-2]
    C_shape = KC_shape[len(KC_shape) - n_frame:]
    K_shape = KC_shape[:len(KC_shape) - n_frame]
    i_shape = (dKCio.shape[-2],)
    o_shape = (dKCio.shape[-1],)
    t_shape = (dtWCi.shape[1],)
    W_shape = dtWCi.shape[2:-(1 + n_frame)]   # t held OUT of the W block

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    dtWCi = dtWCi.reshape(d_shape + t_shape + (size_W, size_C) + i_shape)
    dKCio = dKCio.reshape(d_shape + (size_K, size_C) + i_shape + o_shape)

    dtWKCo = _grouped_einsum(xnp, use_jax, 'dtWCi,dKCio->dtWKCo', dtWCi, dKCio)

    dtWKCo = dtWKCo.reshape(d_shape + t_shape + W_shape + K_shape + C_shape + o_shape)
    return dtWKCo


def trs_drWKCa_dCaib_dsWCb_to_dtWKCi(
        trs: NDArray,
        drWKCa: NDArray,
        dCaib: NDArray,
        dsWCb: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`trs_rWKCa_Caib_sWCb_to_tWKCi` (compute_deta_jets term1)."""
    use_jax = tree_contains_jax((trs, drWKCa, dCaib, dsWCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCaib.shape[0],)
    C_shape = dCaib.shape[1:-3]
    W_shape = dsWCb.shape[2:-(len(C_shape) + 1)]
    K_shape = drWKCa.shape[2 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (dCaib.shape[-3],), (dCaib.shape[-2],), (dCaib.shape[-1],)
    t_shape = (trs.shape[0],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    drWKCa = drWKCa.reshape(d_shape + (trs.shape[1], size_W, size_K, size_C) + a_shape)
    dCaib  = dCaib.reshape(d_shape + (size_C,) + a_shape + i_shape + b_shape)
    dsWCb  = dsWCb.reshape(d_shape + (trs.shape[2], size_W, size_C) + b_shape)

    dtWKCi = _grouped_einsum(xnp, use_jax, 'trs,drWKCa,dCaib,dsWCb->dtWKCi', trs, drWKCa, dCaib, dsWCb)

    return dtWKCi.reshape(d_shape + t_shape + W_shape + K_shape + C_shape + i_shape)


def trs_drWCa_dKCaib_dsWCb_to_dtWKCi(
        trs: NDArray,
        drWCa: NDArray,
        dKCaib: NDArray,
        dsWCb: NDArray,
        n_frame: int,      # len(C) (dKCaib is K+C with no C-only operand to pin it)
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`trs_rWCa_KCaib_sWCb_to_tWKCi` (compute_deta_jets term2)."""
    use_jax = tree_contains_jax((trs, drWCa, dKCaib, dsWCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dKCaib.shape[0],)
    KC_shape = dKCaib.shape[1:-3]
    C_shape = KC_shape[len(KC_shape) - n_frame:]
    K_shape = KC_shape[:len(KC_shape) - n_frame]
    W_shape = drWCa.shape[2:len(drWCa.shape) - 1 - n_frame]
    a_shape, i_shape, b_shape = (dKCaib.shape[-3],), (dKCaib.shape[-2],), (dKCaib.shape[-1],)
    t_shape = (trs.shape[0],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    drWCa  = drWCa.reshape(d_shape + (trs.shape[1], size_W, size_C) + a_shape)
    dKCaib = dKCaib.reshape(d_shape + (size_K, size_C) + a_shape + i_shape + b_shape)
    dsWCb  = dsWCb.reshape(d_shape + (trs.shape[2], size_W, size_C) + b_shape)

    dtWKCi = _grouped_einsum(xnp, use_jax, 'trs,drWCa,dKCaib,dsWCb->dtWKCi', trs, drWCa, dKCaib, dsWCb)

    return dtWKCi.reshape(d_shape + t_shape + W_shape + K_shape + C_shape + i_shape)


def trs_drWCa_dCaib_dsWKCb_to_dtWKCi(
        trs: NDArray,
        drWCa: NDArray,
        dCaib: NDArray,
        dsWKCb: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`trs_rWCa_Caib_sWKCb_to_tWKCi` (compute_deta_jets term3)."""
    use_jax = tree_contains_jax((trs, drWCa, dCaib, dsWKCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCaib.shape[0],)
    C_shape = dCaib.shape[1:-3]
    W_shape = drWCa.shape[2:-(len(C_shape) + 1)]
    K_shape = dsWKCb.shape[2 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (dCaib.shape[-3],), (dCaib.shape[-2],), (dCaib.shape[-1],)
    t_shape = (trs.shape[0],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    drWCa  = drWCa.reshape(d_shape + (trs.shape[1], size_W, size_C) + a_shape)
    dCaib  = dCaib.reshape(d_shape + (size_C,) + a_shape + i_shape + b_shape)
    dsWKCb = dsWKCb.reshape(d_shape + (trs.shape[2], size_W, size_K, size_C) + b_shape)

    dtWKCi = _grouped_einsum(xnp, use_jax, 'trs,drWCa,dCaib,dsWKCb->dtWKCi', trs, drWCa, dCaib, dsWKCb)

    return dtWKCi.reshape(d_shape + t_shape + W_shape + K_shape + C_shape + i_shape)


# ---- transpose (adjoint) sweeps + assembly ----

def dtWCo_dCio_to_dtWCi(
        dtWCo: NDArray,
        dCio: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`tWCo_Cio_to_tWCi`: adjoint lift, contract the ambient mode,
    order ``t`` rides through as a passive broadcast.

    ``t`` is kept as its OWN einsum letter rather than folded into the W block -- folding would flatten
    ``(t, W)`` with ``t`` MAJOR and force a reshard of a W-sharded array; see "Sharding: W must stay
    major" in the module docstring. (This twin was missing: uniform had no explicit ``t``-carrying adjoint
    lift, which is why :py:func:`dtWKCo_dCio_to_dtWKCi` reached past it to the t-folding
    :py:func:`dWCo_dCio_to_dWCi`.)
    """
    use_jax = tree_contains_jax((dtWCo, dCio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCio.shape[0],)
    C_shape = dCio.shape[1:-2]
    i_shape = (dCio.shape[-2],)
    o_shape = (dCio.shape[-1],)
    t_shape = (dtWCo.shape[1],)
    W_shape = dtWCo.shape[2:-(len(C_shape) + 1)]   # t held OUT of the W block

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    dCio  = dCio.reshape(d_shape + (size_C,) + i_shape + o_shape)
    dtWCo = dtWCo.reshape(d_shape + t_shape + (size_W, size_C) + o_shape)

    dtWCi = _grouped_einsum(xnp, use_jax, 'dtWCo,dCio->dtWCi', dtWCo, dCio)

    dtWCi = dtWCi.reshape(d_shape + t_shape + W_shape + C_shape + i_shape)
    return dtWCi


def dtWKCo_dCio_to_dtWKCi(
        dtWKCo: NDArray,
        dCio: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`tWKCo_Cio_to_tWKCi` (compute_deta_tilde_jets). Passive adjoint
    lift: contract the ambient mode ``o`` against the C-only frame Tucker supercore, order ``t`` riding
    through.

    ``C`` is SHARED (on both operands; ``dCio`` pins ``len(C)``) and gets a letter, as do the single passive
    axes ``d`` and ``t``. ``W`` and ``K`` are PASSIVE (on ``dtWKCo`` alone, riding to the output), so they
    ride as ``'...'`` with every axis intact -- nothing fused or flattened across them. See the SHARDING
    block in the module docstring."""
    use_jax = tree_contains_jax((dtWKCo, dCio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCio.shape[0],)
    C_shape = dCio.shape[1:-2]
    i_shape = (dCio.shape[-2],)
    o_shape = (dCio.shape[-1],)
    t_shape = (dtWKCo.shape[1],)
    WK_shape = dtWKCo.shape[2:-(len(C_shape) + 1)]   # W + K, individual axes under '...'

    size_C = math.prod(C_shape)

    dCio   = dCio.reshape(d_shape + (size_C,) + i_shape + o_shape)
    dtWKCo = dtWKCo.reshape(d_shape + t_shape + WK_shape + (size_C,) + o_shape)

    dtWKCi = _grouped_einsum(xnp, use_jax, 'dt...Co,dCio->dt...Ci', dtWKCo, dCio)

    dtWKCi = dtWKCi.reshape(d_shape + t_shape + WK_shape + C_shape + i_shape)
    return dtWKCi


def trs_dtWKCa_dCaib_dsWCb_to_drWKCi(
        trs: NDArray,
        dtWKCa: NDArray,
        dCaib: NDArray,
        dsWCb: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`trs_tWKCa_Caib_sWCb_to_rWKCi` (compute_dxi_tilde_jets, tau term)."""
    use_jax = tree_contains_jax((trs, dtWKCa, dCaib, dsWCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCaib.shape[0],)
    C_shape = dCaib.shape[1:-3]
    W_shape = dsWCb.shape[2:-(len(C_shape) + 1)]
    K_shape = dtWKCa.shape[2 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (dCaib.shape[-3],), (dCaib.shape[-2],), (dCaib.shape[-1],)
    r_shape = (trs.shape[1],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    dtWKCa = dtWKCa.reshape(d_shape + (trs.shape[0], size_W, size_K, size_C) + a_shape)
    dCaib  = dCaib.reshape(d_shape + (size_C,) + a_shape + i_shape + b_shape)
    dsWCb  = dsWCb.reshape(d_shape + (trs.shape[2], size_W, size_C) + b_shape)

    drWKCi = _grouped_einsum(xnp, use_jax, 'trs,dtWKCa,dCaib,dsWCb->drWKCi', trs, dtWKCa, dCaib, dsWCb)

    return drWKCi.reshape(d_shape + r_shape + W_shape + K_shape + C_shape + i_shape)


def trs_drWCa_dCaib_dtWKCb_to_dsWKCi(
        trs: NDArray,
        drWCa: NDArray,
        dCaib: NDArray,
        dtWKCb: NDArray,
) -> NDArray:
    """d-prefixed uniform twin of :py:func:`trs_rWCa_Caib_tWKCb_to_sWKCi` (compute_dxi_tilde_jets sigma term;
    also the apply/entries adjoint dxi_hat combine)."""
    use_jax = tree_contains_jax((trs, drWCa, dCaib, dtWKCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCaib.shape[0],)
    C_shape = dCaib.shape[1:-3]
    W_shape = drWCa.shape[2:-(len(C_shape) + 1)]
    K_shape = dtWKCb.shape[2 + len(W_shape):-(len(C_shape) + 1)]
    a_shape, i_shape, b_shape = (dCaib.shape[-3],), (dCaib.shape[-2],), (dCaib.shape[-1],)
    s_shape = (trs.shape[2],)

    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    drWCa  = drWCa.reshape(d_shape + (trs.shape[1], size_W, size_C) + a_shape)
    dCaib  = dCaib.reshape(d_shape + (size_C,) + a_shape + i_shape + b_shape)
    dtWKCb = dtWKCb.reshape(d_shape + (trs.shape[0], size_W, size_K, size_C) + b_shape)

    dsWKCi = _grouped_einsum(xnp, use_jax, 'trs,drWCa,dCaib,dtWKCb->dsWKCi', trs, drWCa, dCaib, dtWKCb)

    return dsWKCi.reshape(d_shape + s_shape + W_shape + K_shape + C_shape + i_shape)


# ---- d-prefixed order-threaded 3-block dG gradient assembly (order summed; K on the residual operand) ----

def _assemble_dG_jet3_d(trs, opA, opI, opB, einsum_str, k_op, n_probe, keep_W):
    '''d-prefixed twin of :py:func:`_assemble_dG_jet3`: shared body for the six d-prefixed order-threaded
    dG_tilde outer products. Every operand carries ``d`` (shape[0]) then an order axis (shape[1]); exactly
    one (k_op) also carries K. C is pinned by a non-K operand (W+C); n_probe = len(W). The order axes are
    summed via trs; W kept or summed per keep_W; ``d`` rides through.'''
    use_jax = tree_contains_jax((trs, opA, opI, opB))
    xnp, _, _ = get_backend(True, use_jax)
    ops = {'A': opA, 'I': opI, 'B': opB}
    nonk = next(k for k in 'AIB' if k != k_op)
    d_shape = (ops[nonk].shape[0],)
    W_shape = ops[nonk].shape[2:2 + n_probe]
    C_shape = ops[nonk].shape[2 + n_probe:-1]
    K_shape = ops[k_op].shape[2 + n_probe:-(1 + len(C_shape))]
    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    rs = {}
    for k in 'AIB':
        op = ops[k]
        if k == k_op:
            rs[k] = op.reshape(d_shape + (op.shape[1], size_W, size_K, size_C, op.shape[-1]))
        else:
            rs[k] = op.reshape(d_shape + (op.shape[1], size_W, size_C, op.shape[-1]))
    out = _grouped_einsum(xnp, use_jax, einsum_str, trs, rs['A'], rs['I'], rs['B'])
    tail = (opA.shape[-1], opI.shape[-1], opB.shape[-1])
    return out.reshape(d_shape + (W_shape if keep_W else ()) + K_shape + C_shape + tail)


def trs_drWCa_dsWCi_dtWKCb_to_dWKCaib(trs, drWCa, dsWCi, dtWKCb, n_probe):
    """d-prefixed uniform twin of :py:func:`trs_rWCa_sWCi_tWKCb_to_WKCaib` (dG sigma-term, keep W)."""
    return _assemble_dG_jet3_d(trs, drWCa, dsWCi, dtWKCb, 'trs,drWCa,dsWCi,dtWKCb->dWKCaib', 'B', n_probe, True)


def trs_drWCa_dsWCi_dtWKCb_to_dKCaib(trs, drWCa, dsWCi, dtWKCb, n_probe):
    """d-prefixed uniform twin of :py:func:`trs_rWCa_sWCi_tWKCb_to_KCaib` (dG sigma-term, sum W)."""
    return _assemble_dG_jet3_d(trs, drWCa, dsWCi, dtWKCb, 'trs,drWCa,dsWCi,dtWKCb->dKCaib', 'B', n_probe, False)


def trs_dtWKCa_drWCi_dsWCb_to_dWKCaib(trs, dtWKCa, drWCi, dsWCb, n_probe):
    """d-prefixed uniform twin of :py:func:`trs_tWKCa_rWCi_sWCb_to_WKCaib` (dG tau-term, keep W)."""
    return _assemble_dG_jet3_d(trs, dtWKCa, drWCi, dsWCb, 'trs,dtWKCa,drWCi,dsWCb->dWKCaib', 'A', n_probe, True)


def trs_dtWKCa_drWCi_dsWCb_to_dKCaib(trs, dtWKCa, drWCi, dsWCb, n_probe):
    """d-prefixed uniform twin of :py:func:`trs_tWKCa_rWCi_sWCb_to_KCaib` (dG tau-term, sum W)."""
    return _assemble_dG_jet3_d(trs, dtWKCa, drWCi, dsWCb, 'trs,dtWKCa,drWCi,dsWCb->dKCaib', 'A', n_probe, False)


def trs_drWCa_dtWKCi_dsWCb_to_dWKCaib(trs, drWCa, dtWKCi, dsWCb, n_probe):
    """d-prefixed uniform twin of :py:func:`trs_rWCa_tWKCi_sWCb_to_WKCaib` (dG deta-term, keep W)."""
    return _assemble_dG_jet3_d(trs, drWCa, dtWKCi, dsWCb, 'trs,drWCa,dtWKCi,dsWCb->dWKCaib', 'I', n_probe, True)


def trs_drWCa_dtWKCi_dsWCb_to_dKCaib(trs, drWCa, dtWKCi, dsWCb, n_probe):
    """d-prefixed uniform twin of :py:func:`trs_rWCa_tWKCi_sWCb_to_KCaib` (dG deta-term, sum W)."""
    return _assemble_dG_jet3_d(trs, drWCa, dtWKCi, dsWCb, 'trs,drWCa,dtWKCi,dsWCb->dKCaib', 'I', n_probe, False)


# ---- d-prefixed order-diagonal dU gradient assembly (order summed on the diagonal; no trs) ----

def _assemble_dU_eta_d(dtWCa, dtWKCo, n_probe, keep_W):
    '''d-prefixed twin of :py:func:`_assemble_dU_eta`: dU_tilde eta (x) r term (order-diagonal sum over t).
    Both operands carry ``d`` (shape[0]) then order t (shape[1]); the residual r carries K. C pinned by
    eta; K from r; ``d`` rides through.'''
    use_jax = tree_contains_jax((dtWCa, dtWKCo))
    xnp, _, _ = get_backend(True, use_jax)
    d_shape = (dtWCa.shape[0],)
    W_shape = dtWCa.shape[2:2 + n_probe]
    C_shape = dtWCa.shape[2 + n_probe:-1]
    a_shape, o_shape = (dtWCa.shape[-1],), (dtWKCo.shape[-1],)
    K_shape = dtWKCo.shape[2 + n_probe:-(1 + len(C_shape))]
    size_W, size_K, size_C = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)
    dtWCa  = dtWCa.reshape(d_shape + (dtWCa.shape[1], size_W, size_C) + a_shape)
    dtWKCo = dtWKCo.reshape(d_shape + (dtWKCo.shape[1], size_W, size_K, size_C) + o_shape)
    out = _grouped_einsum(xnp, use_jax,
                          'dtWCa,dtWKCo->dWKCao' if keep_W else 'dtWCa,dtWKCo->dKCao', dtWCa, dtWKCo)
    return out.reshape(d_shape + (W_shape if keep_W else ()) + K_shape + C_shape + a_shape + o_shape)


def dtWCa_dtWKCo_to_dWKCao(dtWCa, dtWKCo, n_probe):
    """d-prefixed uniform twin of :py:func:`tWCa_tWKCo_to_WKCao` (dU eta (x) r term, keep W)."""
    return _assemble_dU_eta_d(dtWCa, dtWKCo, n_probe, True)


def dtWCa_dtWKCo_to_dKCao(dtWCa, dtWKCo, n_probe):
    """d-prefixed uniform twin of :py:func:`tWCa_tWKCo_to_KCao` (dU eta (x) r term, sum W)."""
    return _assemble_dU_eta_d(dtWCa, dtWKCo, n_probe, False)


def _assemble_dU_dxi_d(duWKCa, duWo, keep_W):
    '''d-prefixed twin of :py:func:`_assemble_dU_dxi`: dU_tilde dxi_tilde (x) w_jet term (order-diagonal sum
    over u). Both operands carry ``d`` (shape[0]) then order u (shape[1]); dxi_tilde carries K; the w_jet is
    W-only (self-pins W, no C).

    ``d``, the order axis ``u`` (summed) and the SHARED ``W`` (self-pinned by the W-only w_jet) each keep
    their own letter. ``K`` and ``C`` are PASSIVE (on ``duWKCa`` alone, riding to the output), so they ride
    as ``'...'`` with every axis intact -- neither fused with each other nor flattened. See the SHARDING
    block in the module docstring.'''
    use_jax = tree_contains_jax((duWKCa, duWo))
    xnp, _, _ = get_backend(True, use_jax)
    d_shape = (duWo.shape[0],)
    W_shape = duWo.shape[2:-1]
    o_shape, a_shape = (duWo.shape[-1],), (duWKCa.shape[-1],)
    KC_shape = duWKCa.shape[2 + len(W_shape):-1]   # K + C, individual axes under '...'
    size_W = math.prod(W_shape)
    duWKCa = duWKCa.reshape(d_shape + (duWKCa.shape[1], size_W) + KC_shape + a_shape)
    duWo   = duWo.reshape(d_shape + (duWo.shape[1], size_W) + o_shape)
    out = _grouped_einsum(xnp, use_jax,
                          'duW...a,duWo->dW...ao' if keep_W else 'duW...a,duWo->d...ao', duWKCa, duWo)
    return out.reshape(d_shape + (W_shape if keep_W else ()) + KC_shape + a_shape + o_shape)


def duWKCa_duWo_to_dWKCao(duWKCa, duWo):
    """d-prefixed uniform twin of :py:func:`uWKCa_uWo_to_WKCao` (dU dxi_tilde (x) w_jet term, keep W)."""
    return _assemble_dU_dxi_d(duWKCa, duWo, True)


def duWKCa_duWo_to_dKCao(duWKCa, duWo):
    """d-prefixed uniform twin of :py:func:`uWKCa_uWo_to_KCao` (dU dxi_tilde (x) w_jet term, sum W)."""
    return _assemble_dU_dxi_d(duWKCa, duWo, False)
