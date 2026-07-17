# The grouped-einsum interpreter — implementation internals

How `backend.contractions.contract('WCa,Caib,WCi->WCb', *operands, len_W=...)` works, and why each
piece is the way it is. The user-facing story (dialect, usage, guarantees) is
[`../grouped_contractions`](../grouped_contractions.md); the design history that led here — the
named-contraction era, its drift and fusion bugs, the un-rejection of per-call subscript generation
— is [`batching_internals`](batching_internals.md). This note is for reading or modifying the
implementation.

## 1. The pipeline

A call runs four stages; the first three are pure string/integer work, cached as a unit:

```
parse            'WCo,WCa->Cao'  ->  input terms ('WCo','WCa'), output 'Cao'; validation
solve lengths    operand ndims + len_* kwargs  ->  {W: 1, C: 1}   (exact linear algebra)
expand           groups -> fresh single-axis letters  ->  'uvo,uva->auv'-style ordinary einsum
dispatch         numpy: greedy pairwise path (computed on the GROUPED string); jax: one einsum
```

The cache key is `(subscripts, tuple of operand ndims, sorted len_* items)` — everything the
expansion depends on. A warm call is one dictionary lookup plus the einsum: **no reshape, no shape
arithmetic, no per-call string work**. Under `jit`, `.ndim` of a tracer is static, so all of this
happens once at trace time and the compiled program contains only the einsum.

## 2. Parsing and the dialect's hard edges

`_parse_grouped_subscripts` enforces: ascii letters only; exactly one `->` (explicit output);
nonempty input terms (the output may be empty = scalar); no symbol repeated within a term; every
output symbol present in some input; no `'...'` (groups subsume it). Whitespace is stripped, so
`'WCo, WCa -> Cao'` and `'WCo,WCa->Cao'` are the same contraction (and share a cache entry —
callers see the *stripped* string in error messages).

## 3. Group lengths as an exact linear system

Unknowns: `x_G` = the number of axes of group `G`. Every input term contributes one equation —

```
sum of x_G over the groups in the term  =  ndim(operand) - (number of lowercase letters in the term)
```

— and every supplied `len_G` contributes a unit-row equation `x_G = value`. The system is solved by
row reduction over exact `Fraction`s (`_solve_group_lengths`); float arithmetic is never involved.
Three checks fall out of the reduced form:

- **consistency** — a zero row with nonzero right-hand side means the ndims (or a supplied length)
  contradict the subscripts;
- **integrality and nonnegativity** — a 0/1 coefficient matrix can still produce fractional or
  negative solutions for wrong operands (e.g. `'Wab,c->Wabc'` on a 1-d first operand forces
  `x_W = -1`); both raise the same "inconsistent" error;
- **identifiability** — the part with actual design content, below.

### Identifiability is a rank condition on the SUBSCRIPTS

Whether the solution is unique depends only on the coefficient matrix — i.e. only on the string —
never on the right-hand side (the ndims). Two worked examples:

```
'WCo,WCa->Cao'      rows (1 1), (1 1) over (x_W, x_C)      rank 1 < 2
                    kernel spanned by (1, -1): the sum x_W + x_C is pinned, the split is free
                    -> underdetermined; any ONE of len_W, len_C completes it

'WKCa,Caib,WCi->WKCb'   rows (1 1 1), (0 0 1), (1 0 1)     rank 3 = 3
                    -> fully determined, no supplement, always
```

A group's length is determined iff its indicator vector lies in the row space; the check is
performed by reducing the indicator against the pivot rows (`reduce_functional`), accumulating the
value from the right-hand sides as it goes. When something is undetermined, the kernel *is* the
diagnosis — the error message lists exactly the confounded groups and the `len_*` candidates.

Supplements compose greedily: adding `len_G` for any currently-undetermined `G` raises the rank by
exactly 1 (its indicator was outside the row space), so any sequence of choices among undetermined
groups terminates after nullity-many steps — which is why the error message can honestly say
"supply one of …, repeat with another if still underdetermined" without prescribing a particular
set.

**Design decision — identifiability is judged from the string alone, never from the instance.** A
particular call's shapes can *accidentally* pin an underdetermined split (a zero-length prefix, a
nonnegativity corner). The solver deliberately refuses to use such accidents: if the string is
rank-deficient, the supplement is required on every call. Otherwise a call site could work
correctly on unstacked inputs and silently reinterpret axes the first time a stack appears — the
exact silent-misgrouping class that motivated the interpreter. The user-visible corollary is a
stability guarantee: whether a call site needs `len_<G>` is decidable by reading its source.

## 4. Co-traveling runs — the splits that never matter

For `'WKCi,Cio->WKCo'` the `W|K` split is not just underdetermined — it is *unobservable*: `W` and
`K` always appear adjacent, in order, everywhere. Any split of their combined axes produces the
identical expanded einsum, so demanding one would be pure bureaucracy (the old named-toolkit rule
"do not demand a split a contraction does not need", now mechanical).

`_co_travel_runs` merges `X` and `Y` into one run iff **every occurrence of `X` is immediately
followed by `Y`, and every occurrence of `Y` is immediately preceded by `X`, across all input terms
AND the output.** Two details carry the correctness:

- **The output must be included.** In `'WCo,WCa->Cao'`, `W` is always followed by `C` in the input
  terms — but the output contains `C` *without* a preceding `W`. The boundary is observable there
  (`C` survives, `W` is summed), so the merge must not happen, and it doesn't.
- **The merge edges form chains, never cycles.** An edge `X→Y` requires `follower(X) = {Y}` and
  `preceder(Y) = {X}` exactly; a cycle `X→Y→X` would force a term containing `…XYX…`, i.e. a
  repeated symbol in one term, which the parser rejects. So runs are read off by walking maximal
  chains — no fixpoint iteration.

Only each run's **total** must be determined by the linear system. Members whose individual lengths
happen to be pinned (including by a redundant `len_*`) keep their values; the remainder is assigned
arbitrarily to the first free member — legitimate precisely because the expansion is invariant to
the split (the members' fresh letters concatenate identically for every split; this is pinned
bitwise by `TestSplitInvariance`).

## 5. Letter assignment and expansion

Groups expand, in first-appearance order (scanning input terms left to right, then the output),
into consecutive fresh letters from a pool: lowercase letters not used as singles in the string,
then uppercase letters not used as group names. Singles keep their own letters, which keeps
generated strings readable in a debugger. Real examples (via `_expanded_subscripts`):

```
'Cio,WCo->WCi'              |W|=1, |C|=1   ->   'aio,bao->bai'
'Cio,WCo->WCi'              |W|=2, |C|=2   ->   'abio,cdabo->cdabi'
'WKCa,Caib,WCi->WKCb'       1 axis each    ->   'cdea,eaib,cei->cdeb'
'trs,rWCa,Caib,sWCi->tWCb'  |W|=1, |C|=0   ->   'trs,rca,aib,sci->tcb'
```

Note the last line: an **empty group simply vanishes** — no size-1 placeholder axis. (This is why
the interpreter measured slightly *faster* than the old flatten-based bodies on some numpy paths:
the flatten kept empty blocks around as size-1 einsum axes.) The assignment is deterministic, so
equal calls produce byte-equal expanded strings — cache-friendly and reproducible. The 52-letter
einsum alphabet is the ceiling; the largest realistic library contraction uses ~16 symbols, and the
bound is asserted with a clear error.

## 6. Dispatch, and why the pairwise path is computed on the grouped string

The backend split follows the module's long-standing dispatcher policy (see the comment block above
`_grouped_einsum`): jax gets ONE einsum (XLA's optimizer beats any forced order); numpy at 3+
operands gets a forced greedy pairwise path, because numpy's own optimizer runs FLOP-tied
multi-operand contractions as a single non-BLAS `c_einsum` (measured up to ~55× slower on the jet
combines).

The greedy path, however, is computed on the **grouped** string, not the expanded one:

- the path is just a sequence of operand-index pairs, valid for any spelling of the same
  contraction, so it can be computed on one form and applied to the other;
- keyed on the grouped string it stays **size-independent** (one symbol per group), so it is
  computed once per distinct contraction rather than once per batch configuration;
- and it reproduces exactly the paths the named functions used, so the migration changed no numpy
  contraction orders.

Computing greedy on the *expanded* string would also skew the heuristic itself: a 3-axis group
would count as three shared indices where it used to count as one.

## 7. What einsum now validates for free

Because every group axis is a named index in the expanded einsum, extent agreement is checked
**per axis, per operand** by einsum itself. The old flatten-based bodies could not do this: a
transposed batch (`W=(2,3)` on one operand, `(3,2)` on another) flattens to the same size 6, so the
reshape succeeded and the contraction silently computed garbage. Expanded, the axes carry sizes
2,3 vs 3,2 and einsum raises. One deliberate inheritance from einsum semantics: a size-1 group axis
**broadcasts** against the matching axis of other operands (both numpy generations and jax agree);
the old bodies raised there, so this is a strict widening of the accepted-input set, documented in
the `contract` docstring.

The interpreter's own errors are designed to name the fix: underdetermined → which `len_*` to pass
(and that the need is string-level, not shape-level); inconsistent/negative/fractional solve →
check operand order against the terms; unknown `len_X` → the group is not in the subscripts;
malformed kwarg / non-integer length → `TypeError` (via `operator.index`, so `len_W=1.5` cannot
truncate silently).

## 8. Sharding, by construction and by compiler

There is no reshape anywhere in the pipeline, so no flatten ever freezes a layout: every group
sub-axis is an independent einsum axis, sharding a batch/free axis costs zero collectives wherever
it sits, and *fusing* two groups (the old drift class) is not expressible. The invariant is still
verified with the compiler — shard each sub-axis of each group of every library subscripts string
under virtual devices and count all-gathers (`tests/test_contractions_sharding.py`) — because
compiled evidence is how every violation in this module's history was found. The history and the
measurements are in [`batching_internals`](batching_internals.md).

## 9. The evidence structure (and the one maintenance duty)

The interpreter is a single implementation serving every grouped contraction in the library, so its
test suite (`tests/test_contractions_interpreter.py`) carries the weight the hand-written
per-function oracle tests used to. The five lines of evidence, each independent of the
implementation's mechanism:

1. **the definitional loop oracle** — the lowercase-only contraction literally mapped over every
   group index tuple, over the full vocabulary × an empty/single/multi-axis shape matrix;
2. **the frozen identifiability table** (`HISTORICAL`) — for every string, whether a supplement is
   demanded, pinned as data (seeded from the deleted named functions' `n_probe`/`n_frame`
   signatures, which the rank analysis was verified to reproduce exactly, all ~100 cases);
3. **call-site consistency** — every literal `contract(...)` call in the library supplies a
   sufficient supplement set;
4. **split invariance** — all valid splits of a co-traveling run give bitwise-identical results;
5. **the error contract**, plus generic strings beyond the library's letters.

The vocabulary for (1), (3) and the sharding sweep is **AST-scanned from the library source**, so a
new call site is covered the moment it is written — there is no inventory to maintain (four
hand-maintained inventories of the old module were found wrong; that lesson is load-bearing).

**The one duty when adding a contraction:** write the string at the call site, then record its
expected supplement in `HISTORICAL` — *deciding it by hand first*, so the table pins the solver
against your analysis rather than the solver blessing itself. `TestVocabularyCompleteness` fails
until the entry exists, with instructions in the failure message.
