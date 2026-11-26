NOT YET: one finding, `y_block_sumsq` and `f_resolution` read the packed probe `y`'s mode padding (a uniform-contract break), plus two small test gaps. Everything else is right.

# P14 review, round 1: the oracle and the batch

Reviewer, 2026-11-21. This reviews the uncommitted P14 on top of `3769bad4`, against the plan's 2.2 and P14, my
design review, and the rulings in `LOG.md` ("P14 designed"). Everything was run single-process. The mutant copies have
been deleted, and the working tree is untouched apart from this file.

## What I ran

| Check | Result |
| --- | --- |
| `test_least_squares_oracle`, and the oracle's dispatch row (current env) | 15 passed |
| On the floor: the new file and `test_jets` | 58 passed |
| My `p14_check.py`, both envs: unbiasedness, `held` reuse, `f_resolution` against today | all pass, worst 2.5e-16 |
| My `p14_check.py`: the uniform probe `y` with garbage padding | finding 1 |
| 4 mutants of my own | 3 killed, 1 missed (finding 2) |

**What `p14_check.py` checks:**
- **Unbiasedness, my own enumeration:**
  - probe at `(1,)` with a jet sample and a per-(mode, slot) weight;
  - apply at `(2,)`;
  - entries;
  - each over a **two-axis `W = (2, 3)`**, every subset of size 2 of the 6 measurements.

  The mean over subsets of the drawn `f`, `g`, `JtJp`, `Jp_norm_sq` and `Kp` equals the full batch's, to 2.5e-16.
- **A nested `take`** (4 of 6, then 2 of 4) gives the scale 3.
- **`held` reuse:** a query of `f` returns a record within the hold list. A later query of `g`, `JtJp`, `Kp` and
  `r_block_sumsq` with that record equals a fresh query to 1e-14, for the three kinds.
- **`f_resolution`** equals today's `optimizers._objective_resolution(eps, |w y|, misfit, f)` on a full batch, to
  1e-12.

## The settled design, point by point

**`keep=`.**
- The record is the `held` passed in (the same objects), plus each kept name in `{rule.name for rule in rules}`, the
  closure's computed names. Kept values on uniform are masked like outputs.
- So it holds exactly what the call computed, and nothing is computed for it.
- The filled slots come from the closure, so they are static under `jit`; the dispatch row asserts this.
- An unknown name raises, and so does a bare string.

**`frame_form`.** It reads the layout first, then the count, and the engine now reads its call type through it. That
is one reader, as ruled.

**The oracle.**
- It makes one engine call, with `keep=` set to the hold list.
- *The corewise direction's three cases:* `core_directions` if given, else `variations` renamed; both given must be
  the same arrays leaf by leaf, else a `ValueError`.
- *The scale* reaches exactly O-21's six outputs (`_SCALED` plus `f_resolution`).
- *`f_resolution`* is computed from the unscaled values and scaled once.

**`take`.** It is a draw: the jets are skipped, ragged and packed, and nested scales multiply (my enumeration and nested
take above).

**`check_batch`, `pack_batch`, the protocol and reachability.**
- `check_batch` composes the three checkers on the ragged form.
- `pack_batch` packs the jet axes with `len_J`.
- The `Oracle` protocol is documentation in `optimizers.py`, with an unchanged import graph.
- `LeastSquaresOracle` and `LeastSquaresBatch` are re-exported from `t3toolbox.fitting`.

**The changed P3 test.**
- P3's purpose was "checkers, not guarantees": no computational path in the backend calls a checker, and checking is
  the caller's (O-18).
- `check_batch` is itself a checker, built from the three. So allowing them inside it, and only there, keeps the
  purpose, **provided nothing in the backend calls `check_batch`**.
- The test asserts that for `fitting.py` only (finding 3).

## Findings

**1. (Must fix; the uniform equivalence contract.) `y_block_sumsq` and `f_resolution` sum the packed probe `y`'s
mode padding.**
- *The cause.* `query` computes both locally through `data_block_norms(kind, order, batch.y)`. Its docstring says it
  takes the ragged type, "not an array packed for the uniform layout". On a packed probe `y` it sums every column,
  padding included.
- *The engine path is safe:* it masks `y` on `N` on entry (P11), so `f` and `r_block_sumsq` are unaffected.
- *Reproduced.* A uniform frame, with a `pack_batch`-ed probe batch whose `y` padding I filled with garbage (1e3):
  - `f` and `r_block_sumsq` equal ragged (0 and 3e-16);
  - **`y_block_sumsq` is off by a relative 8.7e5, and `f_resolution` by 14.**
  - With `pack_batch`'s zero padding all are exact, so the suite's tests (which pack cleanly) pass.
- *Why it matters.* Padding is don't-care on every input (`docs/uniform_equivalence_contract.md`; P11's Finding B made
  held variables follow the same rule). `f_resolution` sets the trust region's `objective_floor` (P16), and
  `y_block_sumsq` the display's relative-error table, so a hand-packed `y` would mislead both, silently.
- *Fix.* Reduce the real part only: on a packed probe `y`, mask its mode axis with the frame's shape prefix mask before
  `data_block_norms`, or reduce `unpack_vectors(y, shape)`. The shape is static, so this is traceable. Apply and
  entries `y` have no mode axis.
- *Test:* the uniform query with garbage in `y`'s padding equals ragged for `y_block_sumsq` and `f_resolution`.

**2. (Should fix, one assertion.) `f_resolution`'s `eps` in float32 is untested.**
- My mutant `eps = np.finfo(np.float64).eps` (ignoring `f.dtype`) **survives** the oracle file and its dispatch row.
- On a jax float32 run that understates the resolution by about 1e9, so P16's objective floor would never trigger.
- *Test:* on float32 inputs, `f_resolution` uses float32's `eps`. A numpy float32 batch is enough.

**3. (Should fix, small.) The P3 test's "nothing calls `check_batch`" should cover every backend module.** Today it
checks `fitting.py` only. A backend optimizer calling `oracle.check_batch` would break the tier unnoticed. Extend the
attribute check over all of `backend/`; the frontend (O-29) is where it may be called.

## Mutants (4, mine)

**Killed:**
- the scale missing `Jp_norm_sq` (7 subtests);
- the resolution's weight not squared (21);
- the hold list without `r` (16).

**Missed:** the resolution's `eps` fixed to float64 (finding 2).
