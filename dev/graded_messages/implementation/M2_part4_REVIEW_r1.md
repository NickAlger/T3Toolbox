SIGN OFF

# M2 part 4 review, round 1: the backend fitting layer on the rule

Reviewer (the sub-arc's second), 2026-11-22. Reviewed: the uncommitted part 4 on `m2-wip` (over `14029e05`), together
with `M2_part4_HANDOVER.md`. My scripts are in `scratchpad/m2rev2/`:
- `check_p4.py`: the independent checks;
- `check_p4_jit.py`: `take` under jit and on a packed batch;
- `mutants_p4.sh` and `mutants_p4.out`: my mutants.

**Verdict: SIGN OFF.**
- The library change is right. It agrees with my own loops and dense sums in every case I built.
- Every test I perturbed catches the wrong answer.
- The red list equals the log's failing set, each id with its stated cause.
- Two minor notes follow. Neither blocks.

## 1. The code against the plan and the rulings

**`f_resolution`.** It is `eps · (√<y, M y> · √(2f) + |f|)`, where `<y, M y>` is computed by `_y_metric_y`:
- `y` is broadcast to `z`'s shape;
- the result is `Σ y · data_metric(y, ω, None)` over the jet axes, `W` and the edge, so it is kept per element of `C`;
- it is unscaled, then multiplied by `γ` once.

This follows plan §4, decision 13(c) and decision 16.

**Claude's ruling to check: `z`'s shape read off the call.** I checked it, and it is correct. `z`'s shape is assembled
from four pieces:
- `F = order + 1`;
- `W` from the sample, through `_W`, which reads the same axes as `n_measurements`;
- `C` as `f`'s shape (`f_step` keeps `C`, and `f` is never held);
- `N_i` from `frame_shape`.

Reading the shape off `r` would fail when `r` is held. My held-`r` case shows that this path is right
(`f_resolution with held`, 30 cases).

**`take`.** `y` and the weight are drawn by the same function, by their own form, never the sample's:
- each array of a ragged probe object from `m`, and a mode-leading probe array from `1 + m`;
- when every `W` axis is 1, one axis of 1, with no gather;
- otherwise broadcast to `W`, flattened and gathered.

I agree with the ruled check that raises when the `W` part is not `W`. It is structural and static. Without it, a
block weight on apply (`F`) would silently become a legal per-slot `F + (1,)` on a drawn batch.

**Packing.** `pack_data` is `_pack_data_space`: one rule, reached publicly. `pack_batch` packs the weight with it.

**`check_batch(batch, shape, stack_shape=())`.** It passes the stack to both checkers, and the protocol's docstring
says so (correction (e)).

**The razor.** The new helpers are private behind public `take` and `query`, and there is no standalone capability to
expose. I agree.

## 2. My independent checks (`check_p4.py`): 2,135 cases, 0 failures

**The setup.**
- A stacked point, `C = (2,)`, with sample stack `W = (2, 3)` (so `n = 6`). `z` is my own dense contraction of
  `x.to_dense()`, by einsum at order `()`; along a line at `(1,)`, coefficients 0 and 1 of `∏(w_i + s p_i)` written
  out.
- Probe, apply and entries, six trials each.
- `y` and `ω` with random size-1 axes everywhere. On alternate trials `y` is constant over `W`. Every third trial has
  a weight full on `W`'s second axis only. Every third probe trial has a mode-leading weight array with a ragged
  sample.
- Every expansion is by `np.repeat`, never by broadcasting.

| Check | Cases | Against |
| --- | --- | --- |
| `check_batch` passes with `stack_shape=C`; raises without it | 30 + 30 | |
| `f`, full batch | 30 | dense `½ Σ ω²(z − y)²` per element of `C`, 1e-12 |
| `f_resolution`, full batch; again with `held` from the first query | 30 + 30 | dense `eps(√Σ ω²y² · √(2f) + |f|)` over `z`'s shape, shape `C`, 1e-12 |
| uniform: `pack_batch` on the packed frame (`UT3Frame.from_t3frame`), `f` and `f_resolution` | 24 | the same dense values |
| uniform with finite garbage (`7e2`) in the padding of the packed sample, `y` and `ω` | 24 | the same dense values |
| `take`, every 2-subset of the 6 samples | 450 | my index loop (element `k` at `unravel(idx[k])`, read at 0 on a size-1 axis; all-1 `W` → one axis of 1), `array_equal` with shapes |
| the drawn batch's scale | 450 | `n/b = 3` |
| drawn `f` and drawn `f_resolution` | 450 + 450 | dense over the drawn samples, times `γ = 3` |
| unbiasedness: the mean over all 15 draws of `f` and of every block of `g` | 30 + 30 | the full batch |
| float32 `f_resolution` | 1 | float32 `eps`; the result float32; dense to 1e-4 |
| `pack_data` | 20 | a loop: the join across the modes, mixed edges, an edge of 1 kept or broadcast, never zero-padded |

**`check_p4_jit.py`.**
- A jitted `take` (traced leaves and indices), on a ragged probe at `(1,)` with three weight patterns, is bitwise the
  numpy `take`.
- `take` on a packed batch, then a uniform query, equals the ragged route to 2e-16 on `f` and `f_resolution`.
- A jitted `query(take(…))` agrees to 2e-16.

My predecessor's engine brute force (`m2rev/check_p2.py`) is unchanged: ragged with `C`, 56 cases; uniform against
ragged with mixed edges, 12 cases; no failures.

## 3. Mutants of my own (7 of 7 killed)

Each was run on a scratch copy against `test_oracle_data_space.py` and `test_least_squares_oracle.py` at `-n 2`.
Unmutated, the tree passes 28 tests with 535 subtests. The implementer's 14 were not rerun.

| Mutant | Result |
| --- | --- |
| `eps` always float64's | killed (`test_float32_inputs_give_float32_eps`) |
| `f_resolution` left unscaled on a drawn batch | killed (7) |
| `_y_metric_y` misses the last `W` axis | killed (153) |
| the weight gathered at the reversed indices (`y` right) | killed (52) |
| all-1 `W` axes returned as they are (not one axis of 1) | killed (40) |
| `√f` for `√(2f)` | killed (143) |
| `take` drops the batch's own scale | killed (7) |

## 4. The red list and the tally

**The red list.**
- The log `scratchpad/m2impl3/suite4_v.log` started at about 21:49. Every changed file is older: the last was
  `tests/test_dispatch.py`, at 21:48.
- Its failing set, from both the progress and the summary lines, equals the 23 listed ids exactly.
- Every exception line falls in four causes:
  - `prepare_derivative_data` refusing a block weight;
  - the weight checker refusing a block weight through `prepare_batch`;
  - the engine refusing a stacked `y` without `C` through the frontend;
  - the sweep's collected list, whose 64 entries are all the first two causes.
- So every red id is a part-5 frontend route. Part 3's `R` (the float32 rounding gate) is green.

**My own tally** (part 1's log, part 3's and part 4's):
- 110 nodes differ: 10 new nodes, all in `test_oracle_data_space.py`, and 100 changed nodes.
- Each of the 100 has exactly part 1's outcome and subtest count, so no conversion dropped a comparison.

## 5. The diff audit of the converted tests

The converted tests change only how `y` and the weight are built (`blocks_to_weight`, `shared_over`). The changes
beyond that are three, each the decided definition or the new form, and none loosens a tolerance:
1. **`test_least_squares_oracle`.** Its three `f_resolution` expectations now use a dense `<y, M y>` over `z`'s shape
   (`expand`, a sum), in place of the old block formula. That is decision 13(c), with the same tolerances.
2. **`test_a_drawn_batch`.** Its `inner` expands the full-form weight.
3. **The stacked-model tests in `test_curvature` and `test_optimizers`.** Their per-element references move from the
   stacked fit to the element's own fit (`F.at_element`, or the unstacked `element_problem`). This is right: the
   stacked fit's `y` now carries `C`.

`_fits.shared_over` and `at_element` locate `C` as the last group before the probe edge (or the last group). That is
true of every form the tests pass.

I agree with Claude's ruling that `_fits.py` keeps the block vocabulary through `blocks_to_weight`. The per-sample and
per-entry cases are covered by the new file and by my script.

## Minor

1. **`check_batch`'s new doctest relies on pytest's default `ELLIPSIS` flag.**
   - Its expected exception message ends in `...`. `pytest --doctest-modules` (CI) passes it, but
     `python -m doctest t3toolbox/backend/fitting.py` fails it.
   - Every other traceback doctest in `t3toolbox/backend/` carries `# doctest: +IGNORE_EXCEPTION_DETAIL` (27 of 28),
     the house rule of `doctest_style.md`.
   - Add the directive.
2. **`f_resolution` with a held `r` and `y=None` raises an obscure `TypeError`** (`NoneType * float`, inside
   `_y_metric_y`).
   - `f_resolution` is defined through `y`, so it cannot be computed without it.
   - Today's code also failed there, inside `data_block_norms`.
   - A one-line `ValueError` ("`f_resolution` needs the batch's `y`"), or a sentence in the class docstring, would make
     it plain. Optional.

Nothing is left running. My scratch copy is deleted, and the working tree is as I found it, apart from this file.
