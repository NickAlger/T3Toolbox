# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The grouped-einsum interpreter: ``contract('WCa,Caib,WCi->WCb', *operands, len_W=...)``.

An einsum whose UPPERCASE letters each stand for a GROUP of zero or more axes (``W`` probe stack,
``K`` tangent stack, ``C`` frame stack, ...); lowercase letters are single axes as usual. This is
the machinery for independent batch blocks living on DIFFERENT operand subsets -- what a single
leading ``'...'`` einsum cannot express. Group sizes are solved from the operand ndims; each group
expands into fresh single-axis letters; ONE ordinary einsum runs on the operands exactly as given.
Full design: ``docs/batching_and_stacking.md`` (esp. §4).

``jet_contract`` is ``contract`` with one more rule: the jet group ``J`` (the jet axes, one per jet
variable, with sizes of each operand's own) multiplies as truncated polynomials, and the output's jet
shape is an argument. It takes the axis counts of all its groups as one record, ``GroupLengths``, so
that an operand of the wrong rank raises.

(Until 2026-07-17 this module instead enumerated ~104 named contraction functions --
``WCa_Caib_WCi_to_WCb``-style, each a hand-written flatten + fixed-subscript einsum. The
interpreter replaced them: the old name is the string argument, and the old ``n_probe``/``n_frame``
parameters are ``len_W=``/``len_C=``.)
"""
import dataclasses as dc
import functools
import itertools
import math
import numbers
import operator
import string
import typing as typ
from fractions import Fraction

from t3toolbox.backend.common import *

###############################################################################
# SHARDING (the invariant is now BY CONSTRUCTION -- kept short; history in
# docs/contributor/batching_internals.md).
#
# The enumerated named contractions this module used to hold flattened each shared block to one
# einsum letter. A flatten is numerically exact but reindexes which logical elements live where, so
# only the flattened group's LEADING axis could be sharded without collectives -- and delegations
# that silently FUSED two blocks into one flatten cost real all-gathers (found only by compiling;
# no numerical test can see an exact reinterpretation).
#
# The interpreter removes the mechanism behind both problems: it never reshapes anything. Every
# group sub-axis is an honest einsum axis, so EVERY sub-axis of EVERY group is shardable, and
# fusing two groups is not expressible at all. This is checked by compiled evidence anyway
# (tests/test_contractions_sharding.py: shard each sub-axis of each group of every subscripts
# string in the library, 4 virtual devices, assert zero all-gathers) -- measured, not argued,
# because that is how the original fusions were caught.
###############################################################################

__all__ = [
    'contract',
    'GroupLengths',
    'jet_contract',
]

# (The ~104 named contraction functions that used to fill this module -- 'Wa_Caib_Wi_to_WCb'
# through 'duWKCa_duWo_to_dKCao' -- were REMOVED 2026-07-17 in favor of `contract`. Migration:
# `X_Y_to_Z(a, b, n_probe)` -> `contract('X,Y->Z', a, b, len_W=n_probe)`; `n_frame` -> `len_C`.)



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


# --------------------------------------------------------------------------------------------------
# The grouped-einsum interpreter: contract('WCa,Caib,WCi->WCb', ...)
# --------------------------------------------------------------------------------------------------
# The subscripts use the standard einsum format with one extension: an UPPERCASE letter is a GROUP
# of zero or more axes (W probe stack, K tangent stack, C frame stack, ...); lowercase letters are
# single axes as usual. The interpreter solves each group's axis count from the operand ndims (a
# small exact linear system; see `_solve_group_lengths`), expands every group into fresh
# single-axis letters, and runs ONE ordinary einsum on the operands EXACTLY AS GIVEN.
#
# No reshape ever happens (see the SHARDING note above): every axis of every group is shardable,
# and no fusion of two groups is even expressible. The numpy pairwise path is computed on the
# GROUPED string (one symbol per group -- size-independent, cached once per distinct contraction)
# and applied to the expanded einsum; jax gets the single expanded einsum, as always.


_LETTERS = frozenset(string.ascii_letters)


class _Underdetermined(Exception):
    '''Internal: the operand ndims do not pin some group split (caught to build the user error).'''


def _parse_grouped_subscripts(
        subscripts: str,    # grouped einsum string, e.g. 'WCo,WCa->Cao' (whitespace allowed)
) -> tuple:                 # (input_terms, output_term): tuple of str, str
    '''Validates and splits a grouped-einsum string. Uppercase = group, lowercase = single axis;
    explicit '->' required; '...' forbidden (an uppercase group subsumes it); no symbol may repeat
    within one term; every output symbol must appear in some input term.'''
    s = subscripts.replace(' ', '')
    if '...' in s:
        raise ValueError(f"contract({subscripts!r}): '...' is not part of the grouped-einsum "
                         f"format -- spell the riding axes as an uppercase group instead.")
    if s.count('->') != 1:
        raise ValueError(f"contract({subscripts!r}): an explicit '->' output is required.")
    lhs, out = s.split('->')
    terms = tuple(lhs.split(','))
    for t in terms:
        if t == '' or any(ch not in _LETTERS for ch in t):
            raise ValueError(f"contract({subscripts!r}): invalid term {t!r} -- input terms are "
                             f"nonempty strings of ascii letters (uppercase = group, lowercase = axis).")
    if any(ch not in _LETTERS for ch in out):
        raise ValueError(f"contract({subscripts!r}): invalid output term {out!r} -- the output is "
                         f"a (possibly empty) string of ascii letters.")
    for t in terms + (out,):
        if len(set(t)) != len(t):
            raise ValueError(f"contract({subscripts!r}): symbol repeated within term {t!r} -- "
                             f"each group/axis may appear at most once per term.")
    missing = set(out) - set().union(*(set(t) for t in terms))
    if missing:
        raise ValueError(f"contract({subscripts!r}): output symbol(s) {sorted(missing)} do not "
                         f"appear in any input term.")
    return terms, out


def _co_travel_runs(
        terms: typ.Tuple[str, ...],     # input terms + output, e.g. ('WKCi', 'Cio', 'WKCo')
        groups: typ.Tuple[str, ...],    # the group letters, in first-appearance order
) -> typ.Tuple[typ.Tuple[str, ...], ...]:   # partition of `groups` into maximal co-traveling runs
    '''Groups X, Y merge into one run iff every occurrence of X is immediately followed by Y and
    every occurrence of Y is immediately preceded by X, across ALL terms including the output. The
    boundary inside such a run is unobservable: any split of the run's total axis count yields the
    identical expanded einsum, so only the run TOTAL ever needs to be determined. (This is the
    interpreter's form of the old rule "do not demand a split a contraction does not need".)'''
    follower = {g: set() for g in groups}
    preceder = {g: set() for g in groups}
    for t in terms:
        for i, ch in enumerate(t):
            if ch.isupper():
                follower[ch].add(t[i + 1] if i + 1 < len(t) else None)
                preceder[ch].add(t[i - 1] if i > 0 else None)
    nxt = {x: y for x in groups for y in groups
           if follower[x] == {y} and preceder[y] == {x}}
    has_prev = set(nxt.values())
    runs = []
    for g in groups:
        if g in has_prev:
            continue
        run = [g]
        while run[-1] in nxt:
            run.append(nxt[run[-1]])
        runs.append(tuple(run))
    return tuple(runs)


def _solve_group_lengths(
        subscripts: str,                    # the original string (for error messages)
        terms: typ.Tuple[str, ...],         # input terms
        out: str,                           # output term
        ndims: typ.Tuple[int, ...],         # ndim of each operand
        lens: typ.Dict[str, int],           # supplied len_<G> values (may be empty / redundant)
) -> typ.Dict[str, int]:                    # axis count for every group letter
    '''Solves each group's axis count from the operand ndims plus any supplied lengths.

    Each input term contributes one equation (sum of its group lengths = ndim minus its lowercase
    count); each supplied len_<G> contributes another. The system is solved exactly (Fractions).
    Identifiability is judged from the SUBSCRIPTS (rank), never from instance-specific values, so a
    call site either always needs a given len_<G> or never does. Groups that always travel together
    (`_co_travel_runs`) only need their run TOTAL determined -- the split inside a run is
    unobservable and an arbitrary valid split is used.'''
    groups = []
    for t in terms + (out,):
        for ch in t:
            if ch.isupper() and ch not in groups:
                groups.append(ch)
    unknown = set(lens) - set(groups)
    if unknown:
        raise ValueError(f"contract({subscripts!r}): len_{unknown.pop()} was supplied but that "
                         f"group does not appear in the subscripts.")
    n = len(groups)
    col = {g: j for j, g in enumerate(groups)}
    if not groups:
        for t, nd in zip(terms, ndims):
            if nd != len(t):
                raise ValueError(f"contract({subscripts!r}): operand for term {t!r} has ndim "
                                 f"{nd}, expected {len(t)}.")
        return {}

    rows = []   # (coeffs: list of Fraction, rhs: Fraction)
    for t, nd in zip(terms, ndims):
        coeffs = [Fraction(0)] * n
        n_single = 0
        for ch in t:
            if ch.isupper():
                coeffs[col[ch]] = Fraction(1)
            else:
                n_single += 1
        rows.append((coeffs, Fraction(nd - n_single)))
    for g, v in lens.items():
        coeffs = [Fraction(0)] * n
        coeffs[col[g]] = Fraction(1)
        rows.append((coeffs, Fraction(v)))

    inconsistent = ValueError(
        f"contract({subscripts!r}): the operand ndims (and any supplied len_*) are inconsistent "
        f"with the subscripts -- check the operand order against the terms.")

    # exact RREF; then a functional (a coefficient vector over groups) is determined iff it reduces
    # to zero against the pivot rows, with its value accumulating from the right-hand sides.
    mat = [coeffs + [rhs] for coeffs, rhs in rows]
    pivots = []     # (row, col)
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, len(mat)) if mat[i][c] != 0), None)
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        f = mat[r][c]
        mat[r] = [x / f for x in mat[r]]
        for i in range(len(mat)):
            if i != r and mat[i][c] != 0:
                k = mat[i][c]
                mat[i] = [x - k * y for x, y in zip(mat[i], mat[r])]
        pivots.append((r, c))
        r += 1
    for row in mat:
        if all(x == 0 for x in row[:n]) and row[n] != 0:
            raise inconsistent

    def reduce_functional(f):   # -> (determined: bool, value: Fraction)
        f = list(f)
        val = Fraction(0)
        for rr, cc in pivots:
            if f[cc] != 0:
                k = f[cc]
                val += k * mat[rr][n]
                f = [x - k * y for x, y in zip(f, mat[rr][:n])]
        return all(x == 0 for x in f), val

    def as_length(value):       # a solved value must be a nonnegative integer
        if value.denominator != 1 or value < 0:
            raise inconsistent
        return int(value)

    runs = _co_travel_runs(terms + (out,), tuple(groups))
    lengths = {}
    undetermined_runs = []
    for run in runs:
        total_f = [Fraction(1 if g in run else 0) for g in groups]
        det, total = reduce_functional(total_f)
        if not det:
            undetermined_runs.append(run)
            continue
        total = as_length(total)
        known = {}
        for g in run:
            det_g, val_g = reduce_functional([Fraction(1 if h == g else 0) for h in groups])
            if det_g:
                known[g] = as_length(val_g)
        remainder = total - sum(known.values())
        if remainder < 0:
            raise inconsistent
        free = [g for g in run if g not in known]
        for j, g in enumerate(free):    # any split of the run remainder gives the same einsum
            lengths[g] = remainder if j == 0 else 0
        lengths.update(known)
    if undetermined_runs:
        raise _Underdetermined(undetermined_runs)
    return lengths


@functools.lru_cache(maxsize=None)
def _expanded_subscripts(
        subscripts: str,                # grouped einsum string (whitespace already stripped is NOT required)
        ndims: typ.Tuple[int, ...],     # ndim of each operand
        lens_items: tuple,              # sorted ((group, length), ...) from the len_<G> kwargs
) -> str:                               # ordinary einsum string, every group expanded to fresh letters
    '''Expands a grouped-einsum string to an ordinary one for given operand ndims. Groups expand,
    in first-appearance order, into consecutive fresh letters (singles keep their letters, so the
    expansion stays readable); cached, so per-call work is one dict lookup plus the einsum.'''
    terms, out = _parse_grouped_subscripts(subscripts)
    if len(terms) != len(ndims):
        raise ValueError(f"contract({subscripts!r}): {len(terms)} input term(s) but "
                         f"{len(ndims)} operand(s).")
    try:
        lengths = _solve_group_lengths(subscripts, terms, out, ndims, dict(lens_items))
    except _Underdetermined as e:
        (runs,) = e.args
        candidates = ', '.join(f'len_{g}' for run in runs for g in run)
        raise ValueError(
            f"contract({subscripts!r}): the operand ndims do not determine the axis counts of "
            f"group(s) {', '.join('|'.join(run) for run in runs)} -- only combined totals are "
            f"pinned. Supply one of: {candidates} (repeat with another if still underdetermined). "
            f"Note this is decided from the subscripts alone, so a call site either always needs "
            f"it or never does.") from None

    singles = {ch for t in terms + (out,) for ch in t if ch.islower()}
    pool = ([ch for ch in string.ascii_lowercase if ch not in singles]
            + [ch for ch in string.ascii_uppercase if ch not in lengths])
    if sum(lengths.values()) > len(pool):
        raise ValueError(f"contract({subscripts!r}): the expansion needs more than "
                         f"{len(pool)} distinct einsum letters; reduce the group sizes.")
    letters = {}
    i = 0
    for t in terms + (out,):
        for ch in t:
            if ch.isupper() and ch not in letters:
                letters[ch] = ''.join(pool[i:i + lengths[ch]])
                i += lengths[ch]

    def expand(term):
        return ''.join(letters[ch] if ch.isupper() else ch for ch in term)

    return ','.join(expand(t) for t in terms) + '->' + expand(out)


def contract(
        subscripts: str,        # grouped einsum, e.g. 'WCa,Caib,WCi->WCb': UPPERCASE = a group of
                                # ZERO OR MORE axes; lowercase = one axis. Explicit '->' required.
        *operands:  NDArray,    # one array per comma-separated input term
        **group_lens: int,      # len_<G> (e.g. len_W=2) = axis count of group <G>. Required only
                                # when the subscripts cannot pin it; verified when redundant.
) -> NDArray:
    """Grouped einsum: an einsum whose UPPERCASE letters each stand for a GROUP of zero or more axes.

    The group sizes are solved from the operand ndims, every group is expanded into ordinary
    single-axis letters, and one einsum runs on the operands exactly as given -- no reshape, no
    data movement, every axis of every group independently shardable. Dispatch follows the house
    convention (inferred numpy/jax; numpy uses the greedy pairwise BLAS path, jax one fused einsum).

    Full usage guide with examples: ``docs/grouped_contractions.md``; block-letter conventions
    (``W`` probe stack, ``K`` tangent stack, ``C`` frame stack, frame-inner order) are in
    ``docs/batching_and_stacking.md``; implementation internals:
    ``docs/contributor/contractions_internals.md``. Group axes follow einsum's usual extent rules:
    mismatched sizes for the same group axis raise, except that a size-1 axis broadcasts (standard
    einsum semantics on both backends).

    A batched matvec where the matrix batch ``C`` is shared and the vector batch ``W`` rides:

    >>> import numpy as np
    >>> from t3toolbox.backend.contractions import contract
    >>> A = np.random.default_rng(0).standard_normal((2, 4, 3))         # C=(2,), axes (i, o)
    >>> x = np.random.default_rng(1).standard_normal((5, 2, 3))         # W=(5,), C=(2,), axis o
    >>> y = contract('Cio,WCo->WCi', A, x)
    >>> y.shape
    (5, 2, 4)
    >>> bool(np.allclose(y, np.einsum('cio,wco->wci', A, x)))           # |W|=|C|=1 by hand
    True

    The SAME string handles any group sizes, including empty (a group of zero axes just vanishes):

    >>> contract('Cio,WCo->WCi', np.ones((4, 3)), np.ones(3)).shape     # W=(), C=()
    (4,)
    >>> contract('Cio,WCo->WCi', np.ones((2, 2, 4, 3)), np.ones((5, 6, 2, 2, 3))).shape
    (5, 6, 2, 2, 4)

    When the ndims cannot pin a split, say which axes are which via ``len_<G>`` -- the error says
    exactly what is missing (and this is decided from the subscripts alone, never from the shapes,
    so a call site either always needs the argument or never does):

    >>> zo = np.ones((5, 2, 3)); za = np.ones((5, 2, 6))                # W=(5,), C=(2,)
    >>> contract('WCo,WCa->Cao', zo, za)                                # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: contract('WCo,WCa->Cao'): the operand ndims do not determine ...
    >>> contract('WCo,WCa->Cao', zo, za, len_W=1).shape                 # sum over the probe stack W
    (2, 6, 3)
    """
    lens = {}
    for key, value in group_lens.items():
        if not (len(key) == 5 and key.startswith('len_') and key[4].isupper()):
            raise TypeError(f"contract() got an unexpected keyword argument {key!r} -- group "
                            f"lengths are passed as len_<GROUP LETTER>, e.g. len_W=2.")
        lens[key[4]] = operator.index(value)
    ndims = tuple(op.ndim for op in operands)
    expanded = _expanded_subscripts(subscripts, ndims, tuple(sorted(lens.items())))

    use_jax = tree_contains_jax(operands)
    xnp, _, _ = get_backend(True, use_jax)
    if use_jax or len(operands) <= 2:
        return xnp.einsum(expanded, *operands)
    # the pairwise path is computed on the GROUPED string: one symbol per group, size-independent,
    # cached once -- identical to the path the equivalent named contraction uses.
    return xnp.einsum(expanded, *operands, optimize=list(_pairwise_path(subscripts.replace(' ', ''))))



# --------------------------------------------------------------------------------------------------
# jet_contract: contract, with the jet group J multiplying as truncated polynomials
# --------------------------------------------------------------------------------------------------
# A jet in m variables s_1 .. s_m stacks a variable over its Taylor coefficients: the m leading axes of
# its array (the jet group J) hold the coefficient of s_1^k_1 ... s_m^k_m at the multi-index
# (k_1, ..., k_m). Every operand has jet sizes of its own, and its coefficients beyond them are zero (an
# axis of size 1 is a variable it does not depend on); an operand whose term has no J is a constant.
# `jet_contract` computes ONE thing: the product of its operands, contracted as `contract` contracts the
# string with J deleted, and then
#
#   J in the output       its coefficients on the box F = order + 1: the truncated product
#   no J in the output    its one coefficient at the corner `order`: a block readout (the pairing of a
#                         jet with a dual jet stored turned is the corner coefficient of their product)
#
# Which operand does what is read off the operands' jet sizes, which are static, each operand counted
# after it is clipped to the box: slots beyond the box never reach the result and are never read.
#
#   one jet operand, J in the output    a ride: ONE contraction, the operand sliced to the box and the
#                                       result padded with zeros where the box is larger
#   two or more, J in the output        a product: the operand with the most slots (the carrier) keeps J,
#                                       and every other one (a partner) is taken one multi-index at a
#                                       time: one contraction per combination of the partners' indices
#                                       that lies in the box, of the carrier sliced to the box above that
#                                       offset, added into the result at the offset
#   no J in the output                  a readout: the two with the most slots (the pair) are folded in
#                                       ONE contraction along all their jet axes, both sliced to the
#                                       ranges that meet at the corner and the second flipped on every jet
#                                       axis (sum over k of a[k] b[c-k]); every other one is taken one
#                                       multi-index at a time, the corner reduced by its index. A jet
#                                       operand alone: its corner coefficient, one slice. A corner no
#                                       product reaches: zeros, by one contraction
#
# Ties go to the first in the string. With ONE jet variable (m = 1), `over_orders` says how a FULL partner
# is taken (one with as many slots as the box, once it is sliced to it):
#
#   'scan'   a scan over its coefficients. The carrier (in a readout, the first of the pair) rides the
#            carry and is shifted one slot up per step, so every step has the same shapes and the zeros
#            the shift brings in do the truncation: one program size at every order. In a readout each
#            step meets the carried jet and the turned one along the jet axis (carried[t] = a[t-k] meets
#            turned[t] = b[n-t]: the orders sum to n-k, the coefficient supplies k). With several partners
#            the first full one is scanned and the others are looped, the carrier starting as far up as
#            the orders they took.
#   'loop'   a Python loop over the same coefficients, with slices. The program grows with the order.
#
# A partner smaller than the box (a leaf, an input) is looped under both forms.
#
# PART 2 (row 5.4, jets in several variables): with two or more jet variables both forms loop over the
# partners' multi-indices, so the compiled program grows with the box there (the answer to O-38). A scan
# over more than one jet axis, the order in which a product's multi-indices are taken, and transforms are
# Part 2's, behind this signature. The place is marked again where m >= 2 takes the loop under 'scan'.
#
# PART 2 (row 5.5): the order of the contractions INSIDE one step is `contract`'s default, and on numpy
# that default is the costly one for most product steps (see the note at the product step below).
#
# No contraction has a table over the orders, and no array carries two sets of jet axes. Nothing is
# reshaped and only jet axes are sliced, flipped or padded, so `contract`'s sharding property (every
# sub-axis of every group shards free) carries over; tests/test_contractions_sharding.py compiles it.
#
# The strings `contract` is handed are the literal with J deleted from the operands taken one coefficient
# at a time, so every one of them is an ordinary grouped string. The groups of these strings are J, W, K,
# C and no others. The site axis of the uniform layout is NOT one of them: a term is written for one
# site's arrays, which is what a step sees on both layouts.


_JET_GROUPS = frozenset('JWKC')        # the groups a jet_contract string may name


@dc.dataclass(frozen=True)
class GroupLengths:
    """The axis counts of the groups of one ``jet_contract`` call: a hashable static.

    One record is passed whole to every ``jet_contract`` call of a computation, so that every group of
    every string is pinned and an operand of the wrong rank raises. The fields are plain Python integers, so
    two records with the same counts are equal and hash alike: a record can key a memoized scan body
    and ride as jax static data. ``J`` is the number of jet variables ``m``, any count; the output's jet
    shape is not a field, since it differs between the calls of one computation.

    >>> from t3toolbox.backend.contractions import GroupLengths
    >>> lens = GroupLengths(J=2, W=1, K=0, C=1)
    >>> lens.J
    2
    >>> lens == GroupLengths(J=2, W=1, K=0, C=1), hash(lens) == hash(GroupLengths(J=2, W=1, K=0, C=1))
    (True, True)
    >>> GroupLengths(J=-1, W=1, K=0, C=1)                          # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: GroupLengths: J must be a nonnegative int, got -1
    """
    J:  int     # jet axes: the number of jet variables m = len(order), any count >= 0
    W:  int     # sample axes
    K:  int     # direction-stack axes
    C:  int     # base-point-stack axes

    def __post_init__(self):
        for field in dc.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"GroupLengths: {field.name} must be a nonnegative int, got {value!r}")


def _read_order(
        order:  typ.Any,            # the jet shape as the caller gave it
        what:   str,                # who reads it, for the messages
) -> typ.Tuple[int, ...]:           # the jet shape: a tuple of Python ints >= 0, one per jet variable
    '''Reads a jet shape: a tuple of maximum orders, ``()`` for no jets, ``(n,)`` for one variable. A
    tuple only, so an integer or a list raises (O-36); each entry a nonnegative integer, a Python ``int``
    or a numpy integer scalar (an order computed from arrays arrives as one), never a ``bool``. Returned
    as Python ints, the form that keys the caches and rides as jax static data. Structural: raises.'''
    if not isinstance(order, tuple):
        raise TypeError(f"{what}: order is the jet shape, a tuple of maximum orders, one per jet variable "
                        f"(() for none, (n,) for one), got {order!r}.")
    for n in order:
        if not isinstance(n, numbers.Integral) or isinstance(n, bool):
            raise TypeError(f"{what}: every entry of the jet shape is an integer, got {order!r}.")
        if n < 0:
            raise ValueError(f"{what}: every entry of the jet shape is >= 0, got {order!r}.")
    return tuple(int(n) for n in order)


class _JetString(typ.NamedTuple):
    '''A jet_contract string as read off the string alone, by `_read_jet_string`.'''
    terms:  typ.Tuple[str, ...]     # the input terms
    out:    str                     # the output term
    jets:   typ.Tuple[int, ...]     # the operands whose term has J, in the order of the string


@functools.lru_cache(maxsize=None)
def _read_jet_string(
        subscripts: str,    # a jet_contract string, whitespace stripped
) -> _JetString:
    '''Validates a jet_contract string and reads which terms have the jet group. From the string alone, so
    a call site is wrong at every order or at none. Cached: per-call work is one dict lookup.'''
    terms, out = _parse_grouped_subscripts(subscripts)
    what = f"jet_contract({subscripts!r})"
    unknown = sorted({ch for t in terms + (out,) for ch in t if ch.isupper()} - _JET_GROUPS)
    if unknown:
        raise ValueError(f"{what}: group(s) {unknown} are not groups of a jet_contract string -- the "
                         f"groups are J, W, K, C, the ones GroupLengths counts.")
    for t in terms + (out,):
        if 'J' in t and t[0] != 'J':
            raise ValueError(f"{what}: the jet group J must lead its term, got {t!r}.")
    if 'J' in terms:        # taken one coefficient at a time, it would leave `contract` an empty term
        raise ValueError(f"{what}: an input term of J alone -- give it a group beside J (a group of zero axes "
                         f"will do, e.g. 'JW' with W=0).")
    jets = tuple(p for p, t in enumerate(terms) if 'J' in t)
    if not jets:
        raise ValueError(f"{what}: no operand has the jet group J -- a string with no J is a `contract` "
                         f"string.")
    return _JetString(terms, out, jets)


class _JetRoles(typ.NamedTuple):
    '''Which operand does what in one jet_contract call. Static: read off the string, the operands' jet
    sizes and the output's jet shape, by `_jet_roles`.'''
    kind:             str                       # 'ride' | 'product' | 'readout'
    step_subscripts:  str                       # what `contract` is handed: the literal with J deleted from `taken`
    carrier:          typ.Optional[int]         # 'ride': the jet operand; 'product': the operand that keeps J;
                                                #   'readout': the first of the pair (the one carried under 'scan')
    turned:           typ.Optional[int]         # 'readout': the second of the pair, flipped on every jet axis
    taken:            typ.Tuple[int, ...]       # the operands taken one multi-index at a time (a lone readout's too)
    scannable:        typ.Optional[int]         # m = 1: the first full operand of `taken`, which 'scan' scans


@functools.lru_cache(maxsize=None)
def _jet_roles(
        subscripts: str,        # a jet_contract string, whitespace stripped

        jet_sizes:  typ.Tuple[typ.Optional[typ.Tuple[int, ...]], ...],     # per operand: its jet sizes; None if no J
        order:      typ.Tuple[int, ...],                                    # the output's jet shape, len=m >= 1
) -> _JetRoles:
    '''Chooses the carrier, the pair and the operands taken one multi-index at a time from the operands'
    jet sizes, each counted after it is clipped to the box (``order + 1``; for a readout the box of the
    corner, the same), since slots beyond it are never read; the first in the string among equals. Cached
    on statics only.'''
    reading = _read_jet_string(subscripts)
    box = tuple(n + 1 for n in order)

    def n_slots(p):         # the slots of operand p that can be read
        return math.prod(min(size, f) for size, f in zip(jet_sizes[p], box))

    def stepped(taken):     # the literal with J deleted from the operands in `taken`
        return ','.join(t.replace('J', '') if p in taken else t
                        for p, t in enumerate(reading.terms)) + '->' + reading.out

    def first_full(taken):  # with one jet variable: the first operand of `taken` as large as the box
        if len(order) != 1:
            return None
        return next((p for p in taken if min(jet_sizes[p][0], box[0]) == box[0]), None)

    ranked = sorted(reading.jets, key=lambda p: (-n_slots(p), p))
    if 'J' in reading.out:
        if len(reading.jets) == 1:
            return _JetRoles('ride', subscripts, reading.jets[0], None, (), None)
        carrier = ranked[0]
        taken = tuple(p for p in reading.jets if p != carrier)
        return _JetRoles('product', stepped(taken), carrier, None, taken, first_full(taken))
    if len(reading.jets) == 1:
        return _JetRoles('readout', stepped(reading.jets), None, None, reading.jets, None)
    first, second = sorted(ranked[:2])
    taken = tuple(p for p in reading.jets if p not in (first, second))
    return _JetRoles('readout', stepped(taken), first, second, taken, first_full(taken))


def _jet_group_lens(
        subscripts: str,            # a grouped string over the groups of GroupLengths
        lens:       GroupLengths,
) -> typ.Dict[str, int]:            # the len_<G> keywords of `contract`, for exactly the groups the string names
    '''`contract` raises on a len_<G> whose group is not in its string, so it is handed the lengths of
    the groups its string names -- which is all of them, and that is the strictness.'''
    return {'len_' + g: getattr(lens, g) for g in sorted(set(subscripts) & _JET_GROUPS)}


def _multi_indices(
        jet_sizes:  typ.Tuple[int, ...],    # an operand's jet sizes
        box:        typ.Tuple[int, ...],    # order + 1
) -> typ.List[typ.Tuple[int, ...]]:         # its multi-indices inside the box, row-major
    return list(itertools.product(*(range(min(size, f)) for size, f in zip(jet_sizes, box))))


def _placed_in_box(
        part:   NDArray,                    # shape=J+..., J no larger than box - offset on any axis
        offset: typ.Tuple[int, ...],        # where slot (0, ..., 0) of part goes
        box:    typ.Tuple[int, ...],        # order + 1
        xnp,
) -> NDArray:                               # shape=box+...: part at offset, zeros elsewhere
    m = len(box)
    widths = [(o, f - o - size) for o, f, size in zip(offset, box, part.shape[:m])]
    if not any(before or after for before, after in widths):
        return part
    return xnp.pad(part, widths + [(0, 0)] * (part.ndim - m))


def _shifted_up(
        x_jet:  NDArray,    # shape=(n_slots,)+...
        shift:  int,        # 0 <= shift <= n_slots
        xnp,
) -> NDArray:               # same shape. Slot t holds what slot t-shift held; the first `shift` slots hold zero
    if shift == 0:
        return x_jet
    return xnp.concatenate([xnp.zeros_like(x_jet[:shift]), x_jet[:x_jet.shape[0] - shift]], axis=0)


@functools.lru_cache(maxsize=None)
def _jet_scan_body(
        step_subscripts:    str,            # the string of one step: the literal with J deleted from the taken operands
        lens:               GroupLengths,
        carrier:            int,            # the position of the carried jet among the step's operands
        partner:            int,            # the position of the scanned operand's coefficient
) -> typ.Callable:                          # the body of the scan over the partner's coefficients
    '''The scan body of `over_orders='scan'`, for a product and for a readout alike: one coefficient of
    the partner per step, contracted with the carried jet, which rides the carry and is shifted one slot
    per step. Memoized on the hashable statics it reads, so each one has ONE body object and an eager jax
    scan compiles once (docs/contributor/scan_body_principles.md, rule 4). The body closes over strings
    and ints, never arrays: every operand rides the carry or the scanned inputs.'''
    step_lens = _jet_group_lens(step_subscripts, lens)
    n_operands = step_subscripts.count(',') + 1

    def _jet_scan_step(
            carry:  typ.Tuple,              # (the sum so far, the carried jet shifted, (every other operand, ...))
            xs:     typ.Tuple[NDArray],     # (one coefficient of the partner,)
    ) -> typ.Tuple[
        typ.Tuple,                          # the next carry: the sum with this coefficient's part added,
                                            # the carried jet shifted one more slot, the other operands as they came
        typ.Tuple[()],                      # nothing is stacked
    ]:
        total, carried, others = carry
        (coefficient,) = xs
        xnp, _, _ = get_backend(True, tree_contains_jax((carry, xs)))   # only xnp; it ignores the flag
        carried = _shifted_up(carried, 1, xnp)
        rest = iter(others)
        operands = [carried if p == carrier else (coefficient if p == partner else next(rest))
                    for p in range(n_operands)]
        return (total + contract(step_subscripts, *operands, **step_lens), carried, others), ()

    return _jet_scan_step


def _jet_scan(
        step_subscripts:    str,                    # the string of one step
        lens:               GroupLengths,
        step_operands:      typ.List[NDArray],      # the operands of one step, the carried jet as it starts;
                                                    #   the partner's entry is not read
        carrier:            int,
        partner:            int,
        partner_jet:        NDArray,                # shape=(n+1,)+...: scanned along its jet axis
        xscan,                                      # the scan of get_backend(True, use_jax)
) -> NDArray:                                       # the sum over the partner's coefficients of the step's contraction
    '''`over_orders='scan'`: the partner's coefficient 0 against the carried jet as it starts, by one
    `contract` call (which gives the sum its shape and its dtype); then a scan over the other coefficients,
    the carried jet shifted one slot before each.'''
    step_operands = list(step_operands)
    step_operands[partner] = partner_jet[0]
    total = contract(step_subscripts, *step_operands, **_jet_group_lens(step_subscripts, lens))
    if partner_jet.shape[0] == 1:                   # order (0,): nothing to scan
        return total
    others = tuple(x for p, x in enumerate(step_operands) if p not in (carrier, partner))
    (total, _, _), _ = xscan(_jet_scan_body(step_subscripts, lens, carrier, partner),
                             (total, step_operands[carrier], others), (partner_jet[1:],))
    return total


def jet_contract(
        subscripts:   str,                  # a literal grouped string over J, W, K, C, e.g. 'JWCa,Caib,JWCi->JWCb';
                                            # J (the jet axes) leads every term that has it
        *operands:    NDArray,              # one array per input term; each with its own sizes along J
        lens:         GroupLengths,         # the axis counts of J, W, K, C: the whole record, at every call
        order:        typ.Tuple[int, ...],  # the output's jet shape, len=lens.J; the box is order + 1. No default
        over_orders:  str,                  # 'scan' | 'loop': how a full partner is taken one coefficient at a time
) -> NDArray:                               # the output term's shape; its J axes, if it has them, are order + 1
    """``contract``, with the jet group ``J`` multiplying as truncated polynomials.

    A jet in ``m`` variables (``m = lens.J``) stacks a variable over its Taylor coefficients: the ``m``
    leading axes of an operand whose term has ``J`` hold the coefficient of ``s_1^k_1 ... s_m^k_m`` at
    the multi-index ``k``. Each operand has jet sizes of its own and is zero beyond them (an axis of size 1
    is a variable it does not depend on); an operand whose term has no ``J`` is a constant. The result is
    the product of the operands, contracted as ``contract`` contracts the string with ``J`` deleted:

    - ``J`` in the output: the product's coefficients on the box ``order + 1``, the truncated product.
    - no ``J`` in the output: its one coefficient at the corner ``order``. This is a block readout: the
      pairing of a jet with a dual jet stored flipped is the corner coefficient of their product.

    At ``order = ()`` (``lens.J == 0``) the call is exactly one ``contract`` call on the string as
    written. Slots beyond the box are never read. The truncation is the caller's: every operand is taken
    as an exact polynomial, so the output's box must not exceed that of an operand that is itself a
    truncation (a full jet), which this function cannot see.

    How it is computed, from the operands' jet sizes (each counted after clipping to the box; the first
    in the string among equals): with one jet operand and ``J`` in the output, one ``contract`` call. In a
    product the operand with the most slots keeps ``J`` and the others are taken one multi-index at a
    time, one ``contract`` call each. In a readout the two with the most slots are folded along their jet
    axes in one call, and the others are taken one multi-index at a time.

    ``over_orders`` says how a full partner (one with as many slots as the box, once sliced to it) is
    taken when there is one jet variable. Both forms give the same value, to rounding:

    - ``'scan'``: a scan over its coefficients. The carried jet is shifted one slot per step, so every
      step has the same shapes: one compiled program size at every order. It is the ``xscan`` of
      ``get_backend(True, use_jax)``: a real ``jax.lax.scan`` when an operand is a jax array, a Python
      loop on numpy. With several partners the first full one is scanned and the others are looped.
    - ``'loop'``: a Python loop over the same coefficients, with slices; the program grows with the order.

    A partner smaller than the box (a leaf, an input) is looped under both, and with two or more jet
    variables both loop, so the program grows with the box there (Part 2 may change that, behind this
    signature).

    ``lens`` is one record of the axis counts of every group, the same at every call of a computation.
    Every group of the string is pinned by it, so an operand with a stray or a missing axis raises
    rather than being read as a longer stack. ``order`` is a tuple (an integer raises). A string must
    have ``J`` on some operand, and no input term may be ``J`` alone (give it a group beside ``J``, such
    as ``W`` of zero axes). Numpy or jax is inferred from the operands; nothing is reshaped, so every
    sub-axis of ``W``, ``K`` and ``C`` shards free, as with ``contract``.

    One step of a sweep with one jet variable: the full jet ``mu`` through a core ``G`` and a leaf ``xi``
    of two slots (an input line ``w + s p``).

    >>> import numpy as np
    >>> from t3toolbox.backend.contractions import GroupLengths, jet_contract
    >>> rng = np.random.default_rng(0)
    >>> mu = rng.standard_normal((4, 5, 2))                             # J=(4,), W=(5,), axis a
    >>> G = rng.standard_normal((2, 3, 4))                              # C=(), axes (a, i, b)
    >>> xi = rng.standard_normal((2, 5, 3))                             # J=(2,), W=(5,), axis i
    >>> lens = GroupLengths(J=1, W=1, K=0, C=0)
    >>> out = jet_contract('JWCa,Caib,JWCi->JWCb', mu, G, xi, lens=lens, order=(3,), over_orders='loop')
    >>> out.shape
    (4, 5, 4)
    >>> by_hand = (np.einsum('wa,aib,wi->wb', mu[2], G, xi[0])          # s^2: orders 2+0 and 1+1
    ...            + np.einsum('wa,aib,wi->wb', mu[1], G, xi[1]))
    >>> bool(np.allclose(out[2], by_hand))
    True

    Two full jets: the two forms over the orders agree to rounding.

    >>> nu = rng.standard_normal((4, 5, 4))                             # J=(4,), W=(5,), axis b
    >>> eta_scan = jet_contract('JWCa,Caib,JWCb->JWCi', mu, G, nu, lens=lens, order=(3,), over_orders='scan')
    >>> eta_loop = jet_contract('JWCa,Caib,JWCb->JWCi', mu, G, nu, lens=lens, order=(3,), over_orders='loop')
    >>> eta_scan.shape, bool(np.allclose(eta_scan, eta_loop))
    ((4, 5, 3), True)

    No ``J`` in the output: the corner coefficient, here summed over the samples (a block readout).

    >>> block = jet_contract('JWCa,JWCi,JWCb->Caib', mu, xi, nu, lens=lens, order=(3,), over_orders='loop')
    >>> block.shape
    (2, 3, 4)

    Two jet variables, with the jet shape ``(1, 2)``: a jet of sizes ``(2, 3)`` times a line in each
    variable, ``w + s_1 p_1 + s_2 p_2``, of sizes ``(2, 2)`` (its slot ``(1, 1)`` is zero).

    >>> lens2 = GroupLengths(J=2, W=1, K=0, C=0)
    >>> mu2 = rng.standard_normal((2, 3, 5, 2))                         # J=(2, 3), W=(5,), axis a
    >>> xi2 = rng.standard_normal((2, 2, 5, 3))                         # J=(2, 2), W=(5,), axis i
    >>> xi2[1, 1] = 0.0
    >>> out2 = jet_contract('JWCa,Caib,JWCi->JWCb', mu2, G, xi2, lens=lens2, order=(1, 2), over_orders='loop')
    >>> out2.shape
    (2, 3, 5, 4)
    >>> by_hand = (np.einsum('wa,aib,wi->wb', mu2[1, 2], G, xi2[0, 0])  # s_1 s_2^2: (1,2)+(0,0) ...
    ...            + np.einsum('wa,aib,wi->wb', mu2[0, 2], G, xi2[1, 0])  # ... (0,2)+(1,0) ...
    ...            + np.einsum('wa,aib,wi->wb', mu2[1, 1], G, xi2[0, 1])) # ... and (1,1)+(0,1)
    >>> bool(np.allclose(out2[1, 2], by_hand))
    True

    No jets, ``order = ()``: the same string with ``J=0``, no jet axis anywhere, and one ``contract`` call.

    >>> lens0 = GroupLengths(J=0, W=1, K=0, C=0)
    >>> jet_contract('JWCa,Caib,JWCi->JWCb', mu[0], G, xi[0], lens=lens0, order=(), over_orders='loop').shape
    (5, 4)

    The jet shape is a tuple, never an integer, and an operand of the wrong rank raises, wherever the extra
    axis is (structural gotchas):

    >>> jet_contract('JWCa,Caib,JWCi->JWCb', mu, G, xi, lens=lens, order=3, over_orders='loop')    # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: jet_contract('JWCa,Caib,JWCi->JWCb'): order is the jet shape, a tuple of maximum orders, ...
    >>> jet_contract('JWCa,Caib,JWCi->JWCb', mu[:, None], G, xi, lens=lens, order=(3,), over_orders='loop')   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: jet_contract('JWCa,Caib,JWCi->JWCb'): operand 0 (term 'JWCa') has ndim 4, but ...
    """
    if not isinstance(lens, GroupLengths):
        raise TypeError(f"jet_contract(): lens must be a GroupLengths, got {type(lens).__name__}.")
    if over_orders not in ('scan', 'loop'):
        raise ValueError(f"jet_contract(): over_orders must be 'scan' or 'loop', got {over_orders!r}.")
    subscripts = subscripts.replace(' ', '')
    what = f"jet_contract({subscripts!r})"
    order = _read_order(order, what)                # before any cache lookup: a list would be unhashable
    if len(order) != lens.J:
        raise ValueError(f"{what}: the jet shape {order} has {len(order)} entries, but lens.J = {lens.J}.")
    reading = _read_jet_string(subscripts)
    terms = reading.terms
    if len(operands) != len(terms):
        raise ValueError(f"{what}: {len(terms)} input term(s) but {len(operands)} operand(s).")
    for p, (term, operand) in enumerate(zip(terms, operands)):
        ndim = sum(getattr(lens, ch) if ch.isupper() else 1 for ch in term)
        if operand.ndim != ndim:
            raise ValueError(f"{what}: operand {p} (term {term!r}) has ndim {operand.ndim}, but the term has "
                             f"ndim {ndim} with {lens}.")

    m = lens.J
    if m == 0:                                      # no jets: exactly one contraction
        return contract(subscripts, *operands, **_jet_group_lens(subscripts, lens))

    jet_sizes = tuple(tuple(operands[p].shape[:m]) if p in reading.jets else None for p in range(len(terms)))
    for p in reading.jets:
        if min(jet_sizes[p]) < 1:
            raise ValueError(f"{what}: operand {p} (term {terms[p]!r}) has the jet sizes {jet_sizes[p]} -- every "
                             f"jet axis has at least one slot.")
    roles = _jet_roles(subscripts, jet_sizes, order)
    box = tuple(n + 1 for n in order)
    in_box = tuple(slice(0, f) for f in box)
    xnp, _, xscan = get_backend(True, tree_contains_jax(operands))

    if roles.kind == 'ride':                        # one contraction: the jet sliced to the box, the result padded
        step_operands = list(operands)
        step_operands[roles.carrier] = operands[roles.carrier][in_box]
        part = contract(subscripts, *step_operands, **_jet_group_lens(subscripts, lens))
        return _placed_in_box(part, (0,) * m, box, xnp)

    step_subscripts = roles.step_subscripts
    step_lens = _jet_group_lens(step_subscripts, lens)
    step_operands = list(operands)
    # PART 2 (row 5.4, jets in several variables): with m >= 2 `scannable` is None, so 'scan' takes the loop
    # below, and the compiled program grows with the box.
    scanned = roles.scannable if over_orders == 'scan' else None
    looped = tuple(p for p in roles.taken if p != scanned)
    combinations = itertools.product(*(_multi_indices(jet_sizes[p], box) for p in looped))

    if roles.kind == 'product':
        # PART 2 (row 5.5), the contraction order inside a step -- NOT fixed here, on purpose. A step
        # contracts three things: the carried jet with all its slots, the operands with no jet group (a
        # core), and ONE coefficient of the partner. The cheap order takes the coefficient with the core
        # first (the expensive contraction, once) and then meets every slot of the carried jet.
        #   jax:    `contract` runs one `jnp.einsum`, whose default optimizer picks that order.
        #   numpy:  `contract` forces `_pairwise_path`, which pairs the operands sharing the most
        #           letters; here that is a three-way tie, the tie goes to the first two operands, and
        #           wherever those are the carried jet and the core, the expensive contraction is done
        #           on EVERY slot, at every step. The values are the same; the cost is not.
        # The order of a term's contractions is the contraction-order decision of Part 2. Until then
        # tests/test_jet_contract.py (TestOrderInsideAStep) holds an expected failure that turns into
        # a reported failure the day the numpy order changes, so that this note cannot be forgotten.
        carrier_jet = operands[roles.carrier][in_box]
        product = None
        for taken in combinations:                  # one multi-index of every looped partner
            offset = tuple(sum(r[j] for r in taken) for j in range(m))
            if any(o > n for o, n in zip(offset, order)):
                continue
            for p, r in zip(looped, taken):
                step_operands[p] = operands[p][r]
            if scanned is None:                     # the carrier sliced to the box above the offset
                step_operands[roles.carrier] = carrier_jet[tuple(slice(0, f - o) for f, o in zip(box, offset))]
                part = _placed_in_box(contract(step_subscripts, *step_operands, **step_lens), offset, box, xnp)
            else:                                   # m = 1: the carrier is as large as the box (it has at least
                                                    # the scanned partner's slots), and starts `offset` slots up
                step_operands[roles.carrier] = _shifted_up(carrier_jet, offset[0], xnp)
                part = _jet_scan(step_subscripts, lens, step_operands, roles.carrier, scanned,
                                 operands[scanned][in_box], xscan)
            product = part if product is None else product + part
        return product

    # a readout: the corner coefficient of the product
    if roles.carrier is None:                       # one jet operand: its corner, or zeros where it has none
        (alone,) = roles.taken
        if all(n < size for n, size in zip(order, jet_sizes[alone])):
            step_operands[alone] = operands[alone][order]
        else:
            step_operands[alone] = xnp.zeros_like(operands[alone][(0,) * m])
        return contract(step_subscripts, *step_operands, **step_lens)

    first, second = roles.carrier, roles.turned
    flip = (slice(None, None, -1),) * m
    top = None
    for taken in combinations:                      # one multi-index of every looped operand beyond the pair
        corner = tuple(n - sum(r[j] for r in taken) for j, n in enumerate(order))
        if min(corner) < 0:
            continue
        for p, r in zip(looped, taken):
            step_operands[p] = operands[p][r]
        if scanned is None:                         # sum over k of first[k] second[corner - k]: the k in range
            ranges = [(max(0, c - size_b + 1), min(c, size_a - 1))
                      for c, size_a, size_b in zip(corner, jet_sizes[first], jet_sizes[second])]
            if any(lo > hi for lo, hi in ranges):
                continue
            step_operands[first] = operands[first][tuple(slice(lo, hi + 1) for lo, hi in ranges)]
            step_operands[second] = operands[second][tuple(slice(c - hi, c - lo + 1)
                                                           for c, (lo, hi) in zip(corner, ranges))][flip]
            part = contract(step_subscripts, *step_operands, **step_lens)
        else:                                       # m = 1: the pair is as large as the box; the carried jet
                                                    # starts as far up as the looped operands' orders took
            step_operands[first] = _shifted_up(operands[first][in_box], order[0] - corner[0], xnp)
            step_operands[second] = operands[second][in_box][flip]
            part = _jet_scan(step_subscripts, lens, step_operands, first, scanned,
                             operands[scanned][in_box], xscan)
        top = part if top is None else top + part
    if top is None:                                 # no product reaches the corner: zeros, by one contraction
        for p in looped:
            step_operands[p] = xnp.zeros_like(operands[p][(0,) * m])
        empty = (slice(0, 0),) * m                  # the pair along empty jet axes: an empty sum
        step_operands[first], step_operands[second] = operands[first][empty], operands[second][empty]
        top = contract(step_subscripts, *step_operands, **step_lens)
    return top
