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

(Until 2026-07-17 this module instead enumerated ~104 named contraction functions --
``WCa_Caib_WCi_to_WCb``-style, each a hand-written flatten + fixed-subscript einsum. The
interpreter replaced them: the old name is the string argument, and the old ``n_probe``/``n_frame``
parameters are ``len_W=``/``len_C=``.)
"""
import functools
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

    Block-letter conventions (``W`` probe stack, ``K`` tangent stack, ``C`` frame stack,
    frame-inner order) are in ``docs/batching_and_stacking.md``. Group axes follow einsum's usual
    extent rules: mismatched sizes for the same group axis raise, except that a size-1 axis
    broadcasts (standard einsum semantics on both backends).

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
