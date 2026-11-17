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

``jet_contract`` is ``contract`` with one more rule: the order groups ``T`` (a full jet) and ``A`` (an
affine variable) multiply as truncated polynomials. It takes the axis counts of all its groups as one
record, ``GroupLengths``, so that an operand of the wrong rank raises.

(Until 2026-07-17 this module instead enumerated ~104 named contraction functions --
``WCa_Caib_WCi_to_WCb``-style, each a hand-written flatten + fixed-subscript einsum. The
interpreter replaced them: the old name is the string argument, and the old ``n_probe``/``n_frame``
parameters are ``len_W=``/``len_C=``.)
"""
import dataclasses as dc
import functools
import itertools
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
# jet_contract: contract, with the order groups T and A multiplying as truncated polynomials
# --------------------------------------------------------------------------------------------------
# A jet is a variable stacked over its Taylor coefficients: slot t of the order axis holds the
# coefficient of s^t. A full jet has order+1 slots (group T); an affine variable has two (group A).
# A term over the orders is the truncated product of its operands' jets, and `jet_contract` carries
# it out with ordinary `contract` calls on slices of the order axis:
#
#   one operand has an order group, and so has the output    the axis rides: ONE contraction
#   two operands have one, and the output has T              the truncated product. One jet (the
#                                                            carrier) keeps T through the contraction,
#                                                            the other (the partner) is taken one
#                                                            coefficient at a time: two contractions
#                                                            for an affine partner; `over_orders`
#                                                            says how for a full one
#   the output has no order group                            the top coefficient of the product (the
#                                                            block readouts). Two full jets are
#                                                            contracted along the order axis, one of
#                                                            them turned: ONE contraction. A further
#                                                            operand is taken one coefficient at a
#                                                            time: two contractions when it is affine;
#                                                            `over_orders` says how when it is full
#
# The two forms of `over_orders`, wherever a full jet is taken one coefficient at a time:
#
#   'scan'   a scan over its coefficients. The carrier rides the carry and is shifted one slot up
#            the order axis per step, so every step has the same shapes: the zeros the shift brings
#            in do the truncation. In a product the step contracts the coefficient against the whole
#            carried jet; in a readout, against the carried jet and the turned one, summed along
#            the order axis (carried[t] = a[t-k] meets turned[t] = b[order-t]: the orders sum to
#            order-k, the coefficient supplies k). One program size at every order.
#   'loop'   a Python loop over the same coefficients, with slices: only the pairs of orders in
#            range are contracted. The program grows with the order.
#
# PART 2 (row 5.5): the order of the contractions INSIDE one step is `contract`'s default, and on numpy
# that default is the costly one for most product steps (see the note at the product step below).
#
# There is no dense order-by-order form: no contraction ever has an (order+1)^2 intermediate.
# Nothing is reshaped and only the order axis is ever sliced, so `contract`'s sharding property
# (every sub-axis of every group shards free) carries over; tests/test_contractions_sharding.py
# compiles it.
#
# The strings `contract` is handed are the literal with T or A deleted from the operands taken one
# coefficient at a time (`_jet_string_case`), so every one of them is an ordinary grouped string.
#
# The groups of these strings are T, A, W, K, C and no others. The site axis of the uniform layout
# is NOT one of them: a term is written for one site's arrays, which is what a step sees on both
# layouts.


_JET_GROUPS = frozenset('TAWKC')        # the groups a jet_contract string may name


@dc.dataclass(frozen=True)
class GroupLengths:
    """The axis counts of the groups of one ``jet_contract`` call: a hashable static.

    One record is passed whole to every ``jet_contract`` call of a computation, so that every group of
    every string is pinned and an operand of the wrong rank raises. The fields are plain Python integers, so
    two records with the same counts are equal and hash alike: a record can key a memoized scan body
    and ride as jax static data.

    >>> from t3toolbox.backend.contractions import GroupLengths
    >>> lens = GroupLengths(T=1, W=2, K=0, C=1)
    >>> lens.W
    2
    >>> lens == GroupLengths(T=1, W=2, K=0, C=1), hash(lens) == hash(GroupLengths(T=1, W=2, K=0, C=1))
    (True, True)
    >>> GroupLengths(T=2, W=2, K=0, C=1)                           # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: GroupLengths: T is 1 if order else 0, got 2
    """
    T:  int     # 1 if order else 0. The axis count of the order group T, and of the affine group A
    W:  int     # sample axes
    K:  int     # direction-stack axes
    C:  int     # base-point-stack axes

    def __post_init__(self):
        for field in dc.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"GroupLengths: {field.name} must be a nonnegative int, got {value!r}")
        if self.T > 1:
            raise ValueError(f"GroupLengths: T is 1 if order else 0, got {self.T}")


class _JetStringCase(typ.NamedTuple):
    '''How the order groups of one jet_contract string multiply when there is an order axis. Static:
    read off the string alone, by `_jet_string_case`.'''
    kind:             str                           # 'ride' | 'product' | 'top'
    terms:            typ.Tuple[str, ...]           # the input terms of the literal
    step_subscripts:  str                           # what `contract` is handed ('ride': the literal)
    carrier:          typ.Optional[int]             # the full jet that keeps T and is shifted under 'scan':
                                                    #   'product': the first full jet; 'top': the first of the pair
    partner:          typ.Optional[int]             # the operand `over_orders` is about, taken one coefficient
                                                    #   at a time: 'product': the other jet; 'top': the first
                                                    #   full jet beyond the pair, if there is one
    partner_is_full:  bool                          # the partner has T (else A, or there is none)
    pair:             typ.Optional[typ.Tuple[int, int]]     # 'top': two full jets contracted along the order axis
    alone:            typ.Optional[int]             # 'top': the one full jet when there is no pair (one slot)
    looped:           typ.Tuple[int, ...]           # 'top': the operands taken one coefficient at a time


def _order_group(
        term: str,          # one term of a grouped string
) -> str:                   # 'T', 'A' or ''
    return 'T' if 'T' in term else ('A' if 'A' in term else '')


@functools.lru_cache(maxsize=None)
def _jet_string_case(
        subscripts: str,    # a jet_contract string, whitespace stripped
) -> _JetStringCase:
    '''Validates a jet_contract string and decides, from the string alone, how its order groups multiply.
    The string `contract` is handed is the literal with T or A deleted from every operand that is taken
    one coefficient at a time. Cached: per-call work is one dict lookup.'''
    terms, out = _parse_grouped_subscripts(subscripts)
    what = f"jet_contract({subscripts!r})"
    unknown = sorted({ch for t in terms + (out,) for ch in t if ch.isupper()} - _JET_GROUPS)
    if unknown:
        raise ValueError(f"{what}: group(s) {unknown} are not groups of a jet_contract string -- the "
                         f"groups are T, A, W, K, C, the ones GroupLengths counts.")
    for t in terms + (out,):
        if 'T' in t and 'A' in t:
            raise ValueError(f"{what}: term {t!r} has both order groups T and A.")
        if _order_group(t) and t[0] not in 'TA':
            raise ValueError(f"{what}: the order group must lead its term, got {t!r}.")

    jets = tuple(p for p, t in enumerate(terms) if _order_group(t))
    full = tuple(p for p in jets if 'T' in terms[p])

    def stepped(taken):     # the literal with the order group deleted from the operands in `taken`
        return ','.join(t.replace('T', '').replace('A', '') if p in taken else t
                        for p, t in enumerate(terms)) + '->' + out

    if not jets or (len(jets) == 1 and _order_group(out)):
        return _JetStringCase('ride', terms, subscripts, None, None, False, None, None, ())
    if 'T' in out:
        if len(jets) != 2:
            raise ValueError(f"{what}: a product with T in the output has exactly two operands with "
                             f"an order group, got {len(jets)}.")
        # Which of two full jets carries is free: the product is the same either way, to rounding, and
        # only the split of the work differs (the carrier meets every coefficient of the partner). The
        # first is taken so that the choice is one rule, read off the string by whoever writes it.
        carrier = full[0]
        (partner,) = (p for p in jets if p != carrier)
        return _JetStringCase('product', terms, stepped((partner,)), carrier, partner,
                              'T' in terms[partner], None, None, ())
    if 'A' in out:
        raise ValueError(f"{what}: a product of jets is a full jet -- the output cannot have A.")
    if not full:
        raise ValueError(f"{what}: the top coefficient of a product needs an operand with T (it is "
                         f"what says how many coefficients there are).")
    pair = full[:2] if len(full) >= 2 else None     # as for the carrier: which two is free; the first two
    alone = None if pair else full[0]
    looped = tuple(p for p in jets if p not in (pair or (alone,)))
    step_subscripts = stepped(looped + (() if pair else (alone,)))
    if len(full) >= 3:
        return _JetStringCase('top', terms, step_subscripts, pair[0], full[2], True, pair, alone, looped)
    return _JetStringCase('top', terms, step_subscripts, None, None, False, pair, alone, looped)


def _jet_group_lens(
        subscripts: str,            # a grouped string over the groups of GroupLengths
        lens:       GroupLengths,
) -> typ.Dict[str, int]:            # the len_<G> keywords of `contract`, for exactly the groups the string names
    '''`contract` raises on a len_<G> whose group is not in its string, so it is handed the lengths of
    the groups its string names -- which is all of them, and that is the strictness.'''
    return {'len_' + g: getattr(lens, 'T' if g == 'A' else g)
            for g in sorted(set(subscripts) & _JET_GROUPS)}


def _shifted_one_slot(
        x_jet:  NDArray,    # shape=(n_slots,)+...
        xnp,
) -> NDArray:               # same shape. Slot t holds what slot t-1 held; slot 0 holds zero
    return xnp.concatenate([xnp.zeros_like(x_jet[:1]), x_jet[:-1]], axis=0)


@functools.lru_cache(maxsize=None)
def _jet_scan_body(
        subscripts: str,            # a jet_contract string whose case has a full partner, whitespace stripped
        lens:       GroupLengths,
) -> typ.Callable:                  # the body of the scan over the partner's coefficients
    '''The scan body of `over_orders='scan'`, for a product and for a readout alike: one coefficient of
    the partner per step, contracted with the carrier, which rides the carry and is shifted one slot per
    step. Memoized on hashable statics only, so each (string, lengths) has ONE body object and an eager
    jax scan compiles once (docs/contributor/scan_body_principles.md, rule 4). The body closes over
    strings and ints, never arrays: every operand rides the carry or the scanned inputs.'''
    case = _jet_string_case(subscripts)
    step_subscripts, carrier, partner = case.step_subscripts, case.carrier, case.partner
    step_lens = _jet_group_lens(step_subscripts, lens)
    n_operands = len(case.terms)

    def _jet_scan_step(
            carry:  typ.Tuple,              # (the sum so far, the carrier jet shifted, (every other operand but the partner, ...))
            xs:     typ.Tuple[NDArray],     # (one coefficient of the partner,)
    ) -> typ.Tuple[
        typ.Tuple,                          # the next carry: the sum with this coefficient's part added,
                                            # the carrier shifted one more slot, the other operands as they came
        typ.Tuple[()],                      # nothing is stacked
    ]:
        total, carried, others = carry
        (coefficient,) = xs
        xnp, _, _ = get_backend(True, tree_contains_jax((carry, xs)))   # only xnp; it ignores the flag
        carried = _shifted_one_slot(carried, xnp)
        rest = iter(others)
        operands = [carried if p == carrier else (coefficient if p == partner else next(rest))
                    for p in range(n_operands)]
        return (total + contract(step_subscripts, *operands, **step_lens), carried, others), ()

    return _jet_scan_step


def _jet_scan(
        subscripts:     str,                    # the jet_contract string, whitespace stripped
        lens:           GroupLengths,
        step_operands:  typ.List[NDArray],      # the operands of one step, with the carrier whole and unshifted;
                                                # the partner's entry is not read
        partner_jet:    NDArray,                # shape=(order+1,)+...: scanned along its order axis
        xscan,                                  # the scan of get_backend(True, use_jax)
) -> NDArray:                                   # the sum over the partner's coefficients of the step's contraction
    '''`over_orders='scan'`: the partner's coefficient 0 against the carrier as it is, by one `contract`
    call (which gives the sum its shape and its dtype); then a scan over the other coefficients, the
    carrier shifted one slot before each.'''
    case = _jet_string_case(subscripts)
    step_operands = list(step_operands)
    step_operands[case.partner] = partner_jet[0]
    total = contract(case.step_subscripts, *step_operands, **_jet_group_lens(case.step_subscripts, lens))
    others = tuple(x for p, x in enumerate(step_operands) if p not in (case.carrier, case.partner))
    (total, _, _), _ = xscan(_jet_scan_body(subscripts, lens),
                             (total, step_operands[case.carrier], others), (partner_jet[1:],))
    return total


def jet_contract(
        subscripts:   str,              # a literal grouped string, e.g. 'TWCa,Caib,AWCi->TWCb'. T: the order
                                        # group of a full jet; A: of an affine variable. Each leads its term.
        *operands:    NDArray,          # one array per input term. A T axis has order+1 slots, an A axis 2
        lens:         GroupLengths,     # the axis counts of T, W, K, C: the whole record, at every call
        over_orders:  str,              # 'scan' | 'loop': how a full jet is taken one coefficient at a time. No default
) -> NDArray:                           # the output term's shape. Its T axis, if it has one, has order+1 slots
    """``contract``, with the order groups ``T`` and ``A`` multiplying as truncated polynomials.

    A jet stacks a variable over its Taylor coefficients: slot ``t`` of its leading order axis is the
    coefficient of ``s^t``. A full jet has ``order+1`` slots and is written with the group ``T``; an
    affine variable has two, and is written with ``A``. A term over the orders is the truncated product
    of its operands' jets, ``c[t] = sum over i+r=t of contract(a[i], ..., b[r])``, and the string says
    which case it is:

    - one operand has an order group, and so has the output: the axis rides. One ``contract`` call.
    - two operands have one, and the output has ``T``: the truncated product. The first full jet
      keeps ``T`` through the contraction and the other is taken one coefficient at a time: two
      contractions when it is affine; when it is full, the form ``over_orders`` names.
    - the output has no order group: the top coefficient of the product (the orders of the operands
      sum to ``order``), which is what a block readout is. Two full jets are contracted along the
      order axis in one ``contract`` call, one of them turned; a further operand is taken one
      coefficient at a time: two contractions when it is affine; when it is full, the form
      ``over_orders`` names.

    At order 0 (``lens.T == 0``) both groups are empty and the call is exactly one ``contract`` call
    on the string as written.

    ``over_orders`` names the form wherever a full jet is taken one coefficient at a time: in a
    product of two full jets, and in a readout of three. Both forms give the same value, to rounding:

    - ``'scan'``: a scan over that jet's coefficients. The other jet is carried and shifted one slot
      up the order axis per step, so every step has the same shapes and the zeros the shift brings in
      do the truncation. In a product each coefficient is contracted against the whole carried jet;
      in a readout, against the carried jet and the turned one along the order axis. The scan is the
      ``xscan`` of ``get_backend(True, use_jax)``: a real ``jax.lax.scan`` when an operand is a jax
      array (a compiled program of one size at every order), a Python loop on numpy.
    - ``'loop'``: a Python loop over the same coefficients, with slices; it contracts only the pairs
      of orders in range, and its program grows with the order.

    Order 0, a ride, a product with an affine operand and a readout with no third full jet have no
    such loop, and are the same under both.

    The order group leads its term, and every ``T`` axis has the same ``order+1 >= 2`` slots (an
    ``A`` axis 2).

    ``lens`` is one record of the axis counts of every group, the same at every call of a computation.
    Every group of the string is pinned by it, so an operand with a stray or a missing axis raises
    rather than being read as a longer stack. Numpy or jax is inferred from the operands; nothing is
    reshaped, so every sub-axis of ``W``, ``K`` and ``C`` shards free, as with ``contract``.

    One step of a sweep: the jet ``mu`` through a core ``G`` and an affine ``xi``.

    >>> import numpy as np
    >>> from t3toolbox.backend.contractions import GroupLengths, jet_contract
    >>> rng = np.random.default_rng(0)
    >>> order = 3
    >>> mu = rng.standard_normal((order + 1, 5, 2))                     # T=(4,), W=(5,), axis a
    >>> G = rng.standard_normal((2, 3, 4))                              # C=(), axes (a, i, b)
    >>> xi = rng.standard_normal((2, 5, 3))                             # A=(2,), W=(5,), axis i
    >>> lens = GroupLengths(T=1, W=1, K=0, C=0)
    >>> out = jet_contract('TWCa,Caib,AWCi->TWCb', mu, G, xi, lens=lens, over_orders='loop')
    >>> out.shape
    (4, 5, 4)
    >>> by_hand = (np.einsum('wa,aib,wi->wb', mu[2], G, xi[0])          # coefficient 2: orders 2+0 and 1+1
    ...            + np.einsum('wa,aib,wi->wb', mu[1], G, xi[1]))
    >>> bool(np.allclose(out[2], by_hand))
    True

    Two full jets: the two forms over the orders agree to rounding.

    >>> nu = rng.standard_normal((order + 1, 5, 4))                     # T=(4,), W=(5,), axis b
    >>> eta_scan = jet_contract('TWCa,Caib,TWCb->TWCi', mu, G, nu, lens=lens, over_orders='scan')
    >>> eta_loop = jet_contract('TWCa,Caib,TWCb->TWCi', mu, G, nu, lens=lens, over_orders='loop')
    >>> eta_scan.shape, bool(np.allclose(eta_scan, eta_loop))
    ((4, 5, 3), True)

    No order group in the output: the top coefficient, here summed over the samples (a block readout).

    >>> block = jet_contract('TWCa,AWCi,TWCb->Caib', mu, xi, nu, lens=lens, over_orders='loop')
    >>> block.shape
    (2, 3, 4)

    Order 0 is the same string with ``T=0``: no order axis anywhere, and one ``contract`` call.

    >>> lens0 = GroupLengths(T=0, W=1, K=0, C=0)
    >>> jet_contract('TWCa,Caib,AWCi->TWCb', mu[0], G, xi[0], lens=lens0, over_orders='loop').shape
    (5, 4)

    An operand of the wrong rank raises, wherever the extra axis is (structural gotcha):

    >>> jet_contract('TWCa,Caib,AWCi->TWCb', mu[:, None], G, xi, lens=lens, over_orders='loop')   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: jet_contract('TWCa,Caib,AWCi->TWCb'): operand 0 (term 'TWCa') has ndim 4, but ...
    """
    if not isinstance(lens, GroupLengths):
        raise TypeError(f"jet_contract(): lens must be a GroupLengths, got {type(lens).__name__}.")
    if over_orders not in ('scan', 'loop'):
        raise ValueError(f"jet_contract(): over_orders must be 'scan' or 'loop', got {over_orders!r}.")
    subscripts = subscripts.replace(' ', '')
    case = _jet_string_case(subscripts)
    terms = case.terms
    if len(operands) != len(terms):
        raise ValueError(f"jet_contract({subscripts!r}): {len(terms)} input term(s) but "
                         f"{len(operands)} operand(s).")
    for p, (term, operand) in enumerate(zip(terms, operands)):
        ndim = sum(getattr(lens, 'T' if ch == 'A' else ch) if ch.isupper() else 1 for ch in term)
        if operand.ndim != ndim:
            raise ValueError(f"jet_contract({subscripts!r}): operand {p} (term {term!r}) has ndim "
                             f"{operand.ndim}, but the term has ndim {ndim} with {lens}.")

    n_slots = None                              # order + 1: the length of every T axis
    if lens.T:
        for p, (term, operand) in enumerate(zip(terms, operands)):
            group = _order_group(term)
            if group == 'T' and n_slots is None:
                n_slots = operand.shape[0]
            if group and (operand.shape[0] < 2 or operand.shape[0] != (2 if group == 'A' else n_slots)):
                raise ValueError(
                    f"jet_contract({subscripts!r}): operand {p} (term {term!r}) has {operand.shape[0]} "
                    f"slot(s) on its order axis -- where there is an order axis (lens.T == 1) every full "
                    f"jet (T) has the same order+1 >= 2 slots and an affine variable (A) has 2.")

    if lens.T == 0 or case.kind == 'ride':      # no order axis, or it rides: exactly one contraction
        return contract(subscripts, *operands, **_jet_group_lens(subscripts, lens))

    step_subscripts = case.step_subscripts
    step_lens = _jet_group_lens(step_subscripts, lens)
    step_operands = list(operands)
    xnp, _, xscan = get_backend(True, tree_contains_jax(operands))

    if case.kind == 'product':
        # PART 2 (row 5.5), the contraction order inside a step -- NOT fixed here, on purpose. A step
        # contracts three things: the carried jet with all its slots, the operands with no order group
        # (a core), and ONE coefficient of the partner. The cheap order takes the coefficient with the
        # core first (the expensive contraction, once) and then meets every slot of the carried jet.
        #   jax:    `contract` runs one `jnp.einsum`, whose default optimizer picks that order.
        #   numpy:  `contract` forces `_pairwise_path`, which pairs the operands sharing the most
        #           letters; here that is a three-way tie, the tie goes to the first two operands, and
        #           wherever those are the carried jet and the core, the expensive contraction is done
        #           on EVERY slot, at every step. The values are the same; the cost is not.
        # The order of a term's contractions is the contraction-order decision of Part 2. Until then
        # tests/test_jet_contract.py (TestOrderInsideAStep) holds an expected failure that turns into
        # a reported failure the day the numpy order changes, so that this note cannot be forgotten.
        carrier_jet, partner_jet = operands[case.carrier], operands[case.partner]
        if case.partner_is_full and over_orders == 'scan':
            return _jet_scan(subscripts, lens, step_operands, partner_jet, xscan)
        product = None
        for r in range(partner_jet.shape[0]):       # an affine partner (two coefficients), or 'loop'
            step_operands[case.carrier] = carrier_jet[:n_slots - r]
            step_operands[case.partner] = partner_jet[r]
            part = contract(step_subscripts, *step_operands, **step_lens)       # slots r .. of the product
            product = part if r == 0 else product + xnp.concatenate(
                [xnp.zeros((r,) + part.shape[1:], part.dtype), part], axis=0)
        return product

    # the top coefficient of the product: the orders of the operands sum to order
    scanned = case.partner if over_orders == 'scan' else None       # a full jet beyond the pair, under 'scan'
    looped = tuple(p for p in case.looped if p != scanned)          # taken one coefficient at a time, here
    top = None
    for taken in itertools.product(*(range(operands[p].shape[0]) for p in looped)):
        left = n_slots - 1 - sum(taken)             # what the remaining operands' orders must sum to
        if left < 0:
            continue
        for p, r in zip(looped, taken):
            step_operands[p] = operands[p][r]
        if scanned is not None:                     # the pair along the whole order axis, at every step
            carried = operands[case.pair[0]]
            if left < n_slots - 1:                  # the orders already taken: the carrier starts that far up
                carried = xnp.concatenate([xnp.zeros_like(carried[:n_slots - 1 - left]), carried[:left + 1]], axis=0)
            step_operands[case.pair[0]] = carried
            step_operands[case.pair[1]] = operands[case.pair[1]][::-1]
            part = _jet_scan(subscripts, lens, step_operands, operands[scanned], xscan)
        else:
            if case.pair:                           # contracted along the order axis, one of them turned
                step_operands[case.pair[0]] = operands[case.pair[0]][:left + 1]
                step_operands[case.pair[1]] = operands[case.pair[1]][:left + 1][::-1]
            else:
                step_operands[case.alone] = operands[case.alone][left]
            part = contract(step_subscripts, *step_operands, **step_lens)
        top = part if top is None else top + part
    return top
