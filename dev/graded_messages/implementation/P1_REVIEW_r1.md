NOT YET

# P1 review, round 1: `jet_contract` and `GroupLengths`

Reviewer, 2026-11-17. One should-fix finding (a test gap, no code defect), eight minor ones, and three points
that need a decision before a later slice. The code itself I could not fault: it does what rows 5.4 and 2.9, the
answer to O-10 and the plan's sections 2.4 and P1 say, the mathematics checks out independently, and the whole
gate is green on the tree as it stands.

## What is confirmed

- **Additions only.** `git diff` of `t3toolbox/backend/contractions.py` has no removed line; `contract` and
  everything under it are untouched. The three existing test files have additions only.
- **Against the decisions.** Call form `jet_contract(subscripts, *operands, lens, over_orders)`, both keyword-only,
  neither with a default; `over_orders` in `'scan'` | `'loop'`, checked at every order; `GroupLengths` with the
  five fields `S, T, W, K, C` in that order; the three behaviours; order 0 is one `contract` call on the string as
  written with the length of every group it names; no layout knowledge; `'scan'` is the `xscan` of
  `get_backend(True, tree_contains_jax(operands))`; `contract` is handed only the lengths of the groups each
  derived string names; no dense order-by-order form; a wrong rank raises.
- **The mathematics, by my own check** (`/tmp/p1rev/check_math.py`, not the implementer's reference): every
  operand built as a polynomial in `s`, a plain lowercase `np.einsum` evaluated at Chebyshev nodes, the
  coefficients recovered by a Vandermonde solve. All 41 strings and 13 more of my own (affine operand first, the
  pair not in the first two places, two affine operands with one full jet, a single jet, a readout with the
  samples kept, four jets), orders 1 to 4, `W` in `()`, `(3,)`, `(2,2)`, `K` in `()`, `(2,)`, `(2,2)`, `C` in
  `()`, `(2,)`, both forms: 2,688 cases, largest relative error 1.6e-13. The same on eager jax (x64) at orders 1
  and 4: 9.3e-14. Order 0 is bit-equal to `contract` and equal to the plain einsum, for all 54 strings.
- **The top coefficient reads only what it needs.** Spied: for `'TWCa,TWCi,TWCb->Caib'` at order 3 the four calls
  get pair slices of lengths 4, 3, 2, 1; no slot beyond `left` is touched.
- **jit, grad, vmap.** All 41 strings and four extras, orders 0 and 2, both forms, compile under `jax.jit` in
  float32 and return jax float32 (180 compiles, 1.3e-7 against numpy float64); `jax.grad` through `'scan'` works on
  each; `vmap` works; a numpy float64 core among jax float32 jets with x64 off scans without a carry type error.
- **The form of the code.** The scan body comes from a factory memoized on `(subscripts, lens)` (a string and a
  frozen record of ints); it closes over a string, a dict of ints and three ints; every array rides the carry or
  the scanned input. The key cannot collide (the plan is a function of the string, the lengths of the record) and
  grows only with distinct (literal, record) pairs. The body re-derives `xnp` from all its arguments. No `np.`
  touches an operand. Zeros are made with `zeros_like` or the part's own dtype. Doctests run and pass (3 in the
  module; outputs match a real run).
- **The sensitivity claims in the handover** hold where I repeated them: the sharding test is sensitive (my own
  wrapper that flattens `W`: 0 all-gathers as written, 8 under `'scan'` and 10 under `'loop'` with the minor axis
  sharded); the 12 strings needing more than one supplement are 12.

## Findings

**1. Should fix. The dispatch tests do not notice jax-ness inferred from a subset of the operands.**
Place: `tests/test_dispatch.py`, `test_jit_jet_contract`, the line
`self.assert_eager_jax(fn, arrays[0], *jops[1:])`.
Evidence: the mutant `get_backend(True, tree_contains_jax(operands[:1]))` in `jet_contract` passes all four
slice test files. With a numpy first operand it picks `np`, runs `np.concatenate` on jax parts (a silent pull to
the host, the case `scan_body_principles.md` rule 3 names) and still returns a jax array, because `product + part`
re-absorbs. The same mutant under `jax.jit` with the numpy operand closed over raises
`TracerArrayConversionError`, so the gap is one call away from closed.
Resolves it: in the same test, jit with one numpy operand held outside the traced arguments, for each operand
position (or at least the carrier and the partner), e.g. `jax.jit(lambda *rest: fn(arrays[0], *rest))(*jops[1:])`.

**2. Minor. The closure assertion is vacuous.**
Place: `tests/test_jet_contract.py`, `test_an_equal_record_gets_the_same_scan_body`, last statement.
Evidence: a factory memoized on a string and a record never sees a call's arrays, and the assertion looks only for
`np.ndarray` cells. It cannot fail for any body this factory could build.
Resolves it: assert the cell contents are of a whitelist of static types (`str`, `int`, `dict` of ints), or drop
the claim.

**3. Minor. The ride case is not in the jit test.**
Place: `tests/test_dispatch.py`, `JET_CASES`. The plan's "three cases" reads most naturally as the three rules of
2.4 (ride, product, top); the test has affine product, full product, readout. My sweep compiled every ride
string, so nothing is wrong; one ride string in `JET_CASES` would close it.

**4. Minor. Which jet carries is not pinned.**
Place: `_jet_plan`, `carrier = full[0]`; handover choice 7. The mutant `full[-1]` passes everything (the values
are equal by commutativity). The handover says the choice fixes the cost split per string; if that is meant to
hold until 5.5 is decided, one exact assertion on the derived string of a `nu` product would pin it.

**5. Minor. `_JetPlan`, `_jet_plan`, the local `plan`.**
"Plan" is this arc's word for the Part 2 execution plan, and the plan says Part 1 has none. A private record of
how one string is read would be clearer under another name (`_JetStringCase` or similar).

**6. Minor. No shape comment on the return of `jet_contract`.**
The plan's sketch has one; `signature_style.md` asks for the contract the type cannot say. Something like
`# the output term's shape; T: order+1 slots`.

**7. Minor. The order-axis check is uneven.**
A full jet with one slot raises in a product and a readout, and rides without complaint
(`jet_contract('TWa,ab->TWb', x[(1,3,2)], ...)` with `T=1` returns shape `(1,3,4)`). Harmless; say so in the
docstring or check it in the ride case too.

**8. Minor. Operands must be arrays.** A list operand gives `AttributeError: 'list' object has no attribute
'ndim'`. `contract`'s own behaviour was not compared; note only.

**9. Minor. The site group is accepted after the order group, and rejected before it.**
Place: `_jet_plan` ("the order group must lead its term"), and
`TestStructure.test_the_site_group_is_one_more_group`, which pins `'TSWCi,SCio->TSWCo'`.
Evidence: row 5.4 and `SAMPLE_VARIABLES.md` give the layout `[S] + T + W + [K] + C + (size,)`, with `S` first;
the prototype served that. `jet_contract('STWa,Sab->STWb', ...)` raises at every order. No Part 1 string has
`S`, so no Part 1 slice trips over it, and the handover discloses it (choice 3). But the one test about `S`
blesses the layout the row does not have.
Resolves it: reword or drop that test so that it does not pin `T` before `S`, and record the restriction for
Part 2 (see "Decisions" below).

## The implementer's twelve choices

| # | Choice | Whose | Sound? |
| --- | --- | --- | --- |
| 1 | plain frozen dataclass, not `ValueHashedFields` | implementer's; it is the plan's sketch | yes: int fields hash by value |
| 2 | `GroupLengths` validates its fields | implementer's | yes; rejecting numpy integers is strict but `len()` and `.ndim` give Python ints |
| 3 | the order group leads its term | implementer's in Part 1; **record for Part 2** | sound for the 41 strings; finding 9 |
| 4 | only `S T A W K C` | implementer's | yes; can be loosened later without breaking a call |
| 5 | rank check in `jet_contract` itself | implementer's | yes; it is what 5.4 asks, with a usable message |
| 6 | order axes checked where they multiply | implementer's | yes; finding 7 |
| 7 | the first full jet carries | implementer's until 5.5 | yes; finding 4 |
| 8 | first scan step outside the scan | implementer's | yes: one `lax.scan`, 53 jaxpr lines at orders 1, 4, 16 (measured) |
| 9 | operands with no order group ride the carry | implementer's | yes; it is the prototype's form and rule 4 |
| 10 | a readout is one program under both forms | the plan's sentence; **the consequence goes to the maintainer** | see below |
| 11 | strings and reference in a test helper module | implementer's | yes |
| 12 | names and tolerances | implementer's | yes |

## Decisions needed before a later slice

**A. A readout of three full jets under `'scan'` (choice 10). To the maintainer, before P6.**
Real, measured: the jaxpr of `'TWCa,TWCi,TWCb->Caib'` has 45, 126 and 450 lines at orders 1, 4 and 16 under
`'scan'`, the same as under `'loop'`. The other readouts do not grow (two full jets: 7 lines at every order; an
affine leaf: 45). So four of the 41 strings (the `H` strings with a full leaf) grow linearly with the order under
jit whatever `over_orders` says. Related: a readout contracts two full jets along the order axis in one
`contract`, so its intermediate holds all order slots at once and not one coefficient at a time.
The plan's sentence ("the block readouts are the same under both forms") is followed. The note to 2.9 of
2026-11-15 19:08 gives the maintainer's reason for the scan as a program that does not grow with the order and
lists "the tilde and tilde-dot steps and their readouts" as where it is needed; the note of 19:32 says the
readouts "differ only in the form of the program" under the two settings. These do not agree with the plan's
sentence. No signature changes either way, so P1 does not wait on it.

**B. `HISTORICAL` cannot record the 41 literals (handover 5.1). To Claude, before P5.**
Real: as plain `contract` strings, 12 of the 41 need two or three supplements (my count, by trying every subset
of lengths: `'TWKCa,Caib,AWCi->TWKCb'` needs `A` and `K`; `'TWCa,AWKCi,TWCb->KCaib'` needs `A`, `C` and `K`).
`scan_vocabulary` will pick the literals up at the first call site, `TestVocabularyCompleteness` will ask for an
entry each, and the entry format is one letter or none. It is a test format, not a decided row. Also to expect:
the sharding sweep over the vocabulary will compile the 41 with `T` and `A` as ordinary groups, which adds time.

**C. `S` before `T` (finding 9). To Claude to record for Part 2; the maintainer only for information.**
Part 2's strings with `S` in the decided place need the order axis off axis 0 in `jet_contract`'s slicing and
scan. No interface changes.

**D. `W` and `K` traded in the record (handover 5.3). No decision.**
Real and inherent: the ranks are the same, and the contraction is the same (the test asserts bit equality).

## Mutations (in a scratch copy under /tmp; 42 tried)

Caught by `tests/test_jet_contract.py` unless said otherwise: the scan without its shift; shifted the wrong way;
the scan or the loop dropping the partner's last coefficient; the scan counting coefficient 0 twice; the loop's
part in the low slots; the affine partner's second coefficient against the wrong slice; the top coefficient one
order low; the pair not turned; a single jet's slot fixed at the top; the third jet's top coefficient dropped;
no rank check; the rank check on the first operand only; `len_C`, `len_W` or `len_T` not passed (also the
interpreter test); zeros in the default dtype in the loop and in the shift (also dispatch); `np` for `xnp` in the
shift and in the loop (every file, numpy tests included: the module does not bind the name `np`, so a stray `np.` is a `NameError` on first use); `over_orders` unchecked;
`lens` type unchecked; order 0 on a string with `T`/`A` stripped; order axes unchecked; `GroupLengths`
unvalidated; `T` up to 2; the lead check, the unknown-group check, the arity check, the `T`-and-`A` check and the
operand count dropped; the body not memoized (also dispatch).
Caught by `tests/test_dispatch.py` only: `over_orders` ignored (always loop; always scan); an affine partner
sent through the scan; `'scan'` as a Python loop on jax (`get_backend(False, ...)`).

Not caught:
- jax-ness read from the first operand only: finding 1.
- the last full jet carries instead of the first: equal values; finding 4.
- the scan body dispatching on the product alone: equivalent in effect (the product is jax whenever anything is).
- the rank check skipped at order 0: `contract` raises there with a message that also says `ndim`; the
  behaviour is the same.

## The gate

The implementer's full run was on a tree one docstring sentence older than the final one. File times agree with
that account (the test files last changed before that run began; `contractions.py` changed 12 s after it ended),
but I ran the whole gate again on the tree as it stands (`run_gate.sh --detach -n 2 -l /tmp/gate_P1_review`):

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1438 passed, 5 skipped, 1 warning, 60650 subtests passed     1071.7      3392       4477
    doctest-modules     PASS   243 passed                                                     29.4       887       6813
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 1.7       235       7395
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.0       476       7212
    examples            PASS   14 examples, 0 failed                                           228       368       7336
    sphinx              PASS   0 warning/error line(s)                                       210.4       689       6886

Against the baseline: 31 more tests, 11,592 more subtests, the same 5 skips and 1 warning, no failure.

## What I ran

- `pytest --doctest-modules t3toolbox/backend/contractions.py` (3 passed) and `pytest tests/test_jet_contract.py`
  (26 passed, 11,148 subtests) on the working tree.
- `/tmp/p1rev/check_math.py` (interpolation check), `edges.py` (malformed strings, mixed dtypes, caches, the
  slices a readout reads), `jit_sweep.py` (jit and grad over every string), `supp.py` (the supplements the 41
  need as `contract` strings), `shard_sens.py` (the sharding test's sensitivity), `growth.py` (jaxpr sizes; mixed
  numpy and jax under the scan).
- `/tmp/p1rev/mutate.py`: 42 mutations of a copy of the tree in `/tmp/p1rev/scratch`, each against
  `tests/test_jet_contract.py`, the `JetContract` class of the interpreter test, the `jet_contract` tests of
  `test_dispatch.py`, and the sharding test where it bears.
- The whole gate, above. Logs in `/tmp/gate_P1_review`.
- The working tree is as I found it apart from this file; the scratch copy is removed; nothing is left running.
