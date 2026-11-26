# M2 design review: the implementer's note on the switch to the rule

Reviewer, 2026-11-22, on `M2_DESIGN_NOTE.md` at `b32fdb33` (`m2-wip`), read only. I read it against:
- plan §2 to §4 and the M2 row;
- the maintainer's notes of 2026-11-22;
- `M_ARC_DECISIONS.md`;
- my plan reviews r1 and r7 (the two strengthenings of M2's gate);
- M1's records;
- today's code: `sampling_steps` (`f_step`, `Jp_norm_sq_step`, `w_tilde_step`, `r_block_sumsq_step`),
  `sampling_variables` (`_UNIFORM_MASKS`, the packing of `y`), and `backend/fitting.py` (`f_resolution`).

I checked corrections 1 to 3 by script on today's engine: `scratchpad/m2rev/check_decisions.py`.

**Verdict: the design is sound, the order of the parts is right, and I agree with all six corrections.** Two
should-fix points tighten the gate's definition (findings 1 and 2); the rest are minor. None needs another design
round.

## The five parts and their order

- **Part 1 first is right.** The reference's full-form reading and the dense pins (`np.ndindex` loops; `g` by a
  central difference, which is exact because `z` is affine along one variation's core, so `f` is quadratic in `t`;
  `JtJp` by `Σ ω² Jp Jq`) are built and signed off before any library line changes. That is r7's point 5(b): the
  reference and the engine change in one slice, so their errors must not be allowed to correlate. Recording the
  pre-M2 battery in part 1, on the untouched tree, is also right.
- **The engine before `jets`.** Once `w_tilde_step` seeds with `flip(data_metric(…))`, the engine no longer calls
  `jet_to_dual_jet`, so the converters can switch afterwards without breaking it. Each part is then reviewable
  against layers already switched. A block weight reaching the switched engine raises by rank (M1's `data_metric`),
  so the intermediate red is loud, not silent.
- **Parts 3 to 5** follow the dependency order: the checkers and converters, then `check_batch`/`take`/packing/
  `f_resolution`, then the frontend, the translation, the pages and the full gate with the slow tier. Committing the
  parts on `m2-wip` and squashing is P16's method, and it keeps the silent-wrong intermediate the plan names off
  `graded-messages`.

## The gate's split: bitwise against to rounding

**The split is right.** From today's code:
- **`f` and `Jp_norm_sq`.** `f_step` and `Jp_norm_sq_step` compute `(ω·ω) · (r·r)`, then sum. The new form is
  `r · ((ω·ω)·r)`, a different grouping of the same product, so they agree to rounding.
- **`f_resolution`.** Today it is `eps·(√Σ(ω²·y_block_sumsq)·√(2f) + |f|)`, with block sums first. The new
  `<y, M y>` sums over the full broadcast shape, so it agrees to rounding.
- **`w_tilde`, `g`, `JtJp`, `Kp`.** These go through the seed, and the seed is the elementwise product
  `(ω·ω)·r`, cast-then-square, in the same order. M1 proved this bitwise, mixed precision included. The sweeps
  downstream are unchanged, so these are bitwise on numpy.
- **`r`.** Probe `y` is masked today too, and `y·1 = y` exactly, so `r` is bitwise.

See finding 2 for what the split should add, and for the jitted run and the optimizer runs.

## The re-pinned bridge test

**Agree.** After part 3 the converters go through `data_metric`, so M1's bridge would compare `data_metric` with
itself. Writing out the old formula in the test keeps it independent. It must keep:
- today's cast-then-square order: `ω` cast to the lower precision of the pair, squared, reshaped to
  `F + (1,)*rest`, times `r` cast the same way, then flipped;
- the four float32/float64 pairings.

Rerun M1's L4 mutant against the re-pinned test; it must still fail.

## The six corrections

**1. The engine checks `y` and `ω` structurally: agree, verified.**
- Today, `apply` with `C = W = (2,)` and a shared `y` of shape `W`: `r_step` aligns it (`r == z - y[:, None]`), and
  the plain right-aligned `z - y` is **not** equal to it. That is a silent wrong answer once the alignment goes.
- Plan §2 says "a stacked `y` checked without its stack fails by rank", but that is the checkers. The engine is
  public and reachable directly, so it needs its own check: rank equal to `z`'s, each axis 1 or the setting's size,
  exact mode count. This is the structural-stack principle the maintainer kept ("strongly agree", TARGET_DECISIONS
  O-18's reading).

**2. Mask `y` on the uniform layout whatever its edge: agree, verified.**
- On today's engine: a uniform probe with a packed edge-1 `y` gives `f = 29953.7977100615`, equal to ragged; with
  `'y'` taken out of `_UNIFORM_MASKS`, `29958.9735251771`.
- Masking broadcasts the edge to `N` with zero padding, which is right. `ω` can stay conditional (`ω²·0 = 0`).
- Say "probe `y`": the apply and entries `y` have no `N` edge. Today's code skips them (`sampling_variables.py:1841`),
  and that must stay.

**3. `f_resolution` over `z`'s shape: agree, verified.**
- With `y` the same for every sample (`W` axis of 1) and `ω` per slot, `<y, M y>` over `y`'s own shape is 2.23 and
  over `z`'s shape 8.92: a factor of exactly `|W|`. The resolution is a floor on `f`'s rounding, and `f` sums every
  sample, so the data's norm must count every sample too.
- An `ω` fuller than `y` raises in `data_metric` (M1's axis check), as the note says.
- Broadcasting `y` to `r`'s shape and summing over all but `C` is right.
- The same holds for M3's `y_block_sumsq`. Today, on a stacked model with a shared `y`, `r_block_sumsq` sums over
  `C` while `y_block_sumsq` counts the shared `y` once, so the display's ratio is already off by `√|C|` there. M3's
  design note should know that.

**4. The checkers accept a `d`-leading array: agree.** `blocks_to_weight` returns one, and the recipe of plan §4
runs it through the checkers; refusing it would break the documented recipe. Accept it iff:
- every mode's slice obeys the rule;
- the first axis is exactly `d`, so a mode axis of 1 is refused, as in M1's `data_metric`.

A padded packed array then fails unless every `N_i = N`. That is correct, since the checkers take the unpadded form.

**5. `check_batch(batch, shape, stack_shape=())`, always passed by the frontend: agree.** It is plan §2's signature.
The `Oracle` protocol's optional member is unreleased (P16), so user oracles break nothing released. State the
keyword in the protocol's docstring.

**6. The "weight's shape" wording fixed where M2 makes it false: agree.** The display's text stays M3's, as the plan
says.

## Findings

### 1. SHOULD FIX: the expected-red lists should be recorded by a run, not written by hand, and they miss files

The lists name the oracle-level test files, but the hand-written lists already miss some:
- **`tests/backend/test_zero_reference.py:415`** runs a weighted derivative fit through `_fits.derivative_fit(…,
  weight=omega)`, so it is red from part 2.
- **`tests/test_jax_absent.py:178-183`** calls `check_weight` and the converters with a block weight, so it is red
  from part 3. Only `test_dispatch` and `test_jax_absent`'s jets rows are named, in part 3's tests, not in a red
  list.
- Part 3's own changes to `test_jets` (the rejection table) and to M1's `test_data_metric` (the re-pin) belong in
  that part's account.

**Resolve: do as P16 did** (`P16_part*_EXPECTED_RED.txt`). At the start of each part, after its library change and
before the test conversions, run the suite and record the red nodes in `M2_part<k>_EXPECTED_RED.txt`. The part's
review then checks node by node that:
- everything red is explained by the part's change;
- everything not converted in the part stays as red as recorded, or green.

This also catches a test that turns green for a wrong reason.

### 2. SHOULD FIX: complete the battery's expectations

1. **Add the outputs M2 does not touch to the bitwise list:** `z`, `Jp`, `r_block_sumsq` (its step is unchanged and
   reads `r`, which is bitwise), and `y_block_sumsq`. The last is `data_block_norms(y)²` on a `y` that now carries
   size-1 `C` axes. A sum over an extra size-1 axis should be bitwise; if it is not, say why.
2. **The jitted run: to rounding, not bitwise.** The ops are the same, but XLA's fusion of a changed graph (the `f`
   and `Jp_norm_sq` reductions sit in the same program) is not promised to keep every reduction bitwise. Bitwise is
   a numpy claim.
3. **The short optimizer runs** (`newton_cg`, `mc_sgd`, new form and translated) need their own stated expectation.
   Rounding-level changes in `f` and in CG's `pHp` (`Jp_norm_sq`) propagate through the iterations. State:
   - the same number of iterations and the same stop reason;
   - the iterates to a tolerance chosen with a reason (for example `rtol` 1e-10 after 3 Newton steps);
   - and if a run sits on a knife-edge (an Armijo or a stop test decided at rounding level), choose another run.
     Do not loosen the tolerance.

### 3. MINOR

1. **`_with_K`'s placement wording.** "after `F + W` (after the mode axis for a `d`-leading array)" can be read as
   "right after the mode axis". Write: the `K` axes go at position `lead + m + len(W)`, with `lead` 1 for a
   `d`-leading array and 0 otherwise.
2. **The `ω = 0` oracle** should drop a sample whose `y` and `z` differ, so that the deletion is visible in `f`. A
   sample with `r = 0` would pass trivially. The same holds for "stacked equals separate": use different data per
   element, so that a mix-up of `C` elements shows.
3. **Garbage padding.** "Finite garbage" is the house rule (`pack_vectors`: the fill must stay finite; masking
   multiplies). Say so in the test's docstring, so that no one adds a NaN case and reads its failure as a bug.
4. **The diff audit** ("every converted test changed only its weight and `y` arguments, never an assertion or a
   tolerance") is a good gate item. Make it mechanical where possible: list each converted test's changed lines from
   `git diff -U0`, classified by script, and read the rest.

## What I will check on each part

- **Part 1:** the reference's full-form reading and the dense pins against my own brute-force sums; its
  independence (no `jets` import); the battery recorded on the untouched tree.
- **Parts 2 to 5:**
  - each part's expected-red file against a run;
  - the engine's structural check with mutants (an unaligned `y` passes; the `y` mask dropped);
  - the `f_resolution` broadcast;
  - `take` against index loops, with a partly broadcast weight;
  - the packing of an edge-1 `ω` (never padded to entry 0 only);
  - the converted tests' diffs.
- **The final gate:** the battery comparison under finding 2's expectations, and the translation's 19 pins.
