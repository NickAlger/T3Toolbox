NOT YET: two small fixes. The entries index's leading `d` is not checked (a silent wrong answer on uniform), and one surviving mutant (the one-hot `'w'` returned as a tuple).

# P11 review, round 1: Engine G (the uniform layout, sharding over `W`)

Reviewer, 2026-11-20. This reviews the uncommitted P11 on top of `fcb20bcd`, against `P11_HANDOVER.md`, the plan's
P11 (with the settled design) and my `P11_DESIGN_REVIEW.md`. Everything was run single-process, one run at a time.
The mutants ran in scratch copies, which have been deleted. The working tree is untouched apart from this file.

**Summary.**
- The build matches the settled design: one entry table, held variables masked, the shape mask on `N` everywhere it
  belongs, the layout inferred in the whole steps, and the index checked.
- My independent checks pass, including a TT-slack point built by the library, with my own dense oracle for `z` and
  `f`, and all 98 holdable variables tested with garbage padding.
- The layering question has a clean answer (section 2): it is sound, and each layer is pinned by a permanent test.
- One finding is in the silent-wrong-answer class, the same class as Finding C:
  - an entries index whose leading axis is 1 is broadcast to every mode on uniform;
  - ragged raises on the same input.

  The fix is one structural check.
- One mutant survives, and one assertion kills it.

## 1. What I ran

| Check | Result |
| --- | --- |
| P11's tests (`test_sampling_uniform`, `test_sampling_graph`, the two dispatch tests, `TestTheEngineShardsOverW`), current env | 43 passed, 3,340 subtests, 55 s |
| The same on the floor (py3.9, numpy 1.22, jax 0.4.30), plus the `pack_vectors` doctest | 43 passed, 138 s; the doctest passes |
| The sharding class alone: not skipped (4 virtual devices), zero all-gathers, equal to unsharded | passed, 4 subtests |
| My `p11_check.py` A: a **TT-slack point from the library** (details below) | every name equal to ragged, exact zero padding; `z` and `f` against my dense oracle; worst 1e-12 (seeds 0 and 1, both envs) |
| My `p11_check.py` B: **every holdable variable** of probe/manifold, probe/corewise, apply/manifold and entries held with garbage (1e3) padding, varying ranks over `C`, `K` | 98 held variables; `f`, `g`, `JtJp`, `Kp`, `Jp_norm_sq`, `r_block_sumsq` equal to the clean call to 1e-12 |
| My `p11_check.py` C: the index (jax, list, wrong leading `d`) | jax and list match numpy. **A leading 1 is silent on uniform** (finding 1) |
| 25 mutants of my own (section 3) | 24 killed, 1 missed (finding 2) |

**The TT-slack point in check A.** It is built the way an optimizer meets it:
- `TuckerTensorTrain.randn((6, 5, 4), (4, 4, 3), (1, 9, 9, 1))`, then `from_t3`;
- its ragged frame, packed at the point's padded sizes (`frame[5]` a strict subset of the point's masks, asserted);
- `D_dot` and `L_dot` with values in the **point's** real slots (padding for the frame);
- garbage in every frame padding slot and in the sample's and `y`'s `N` padding;
- probe at `(2,)` and `()`, apply at `(1, 1)`.

The expected real extents come from the ragged frame's shapes, not from the engine's masks. In effect it is a second
`Kp` at TT slack with a real construction, not garbage.

## 2. The layering of the sample's mask, the start values' mask and the exit masks: sound, keep all three

**What each layer protects.**
- **Real parts.** All three are redundant for real parts, by construction. Every contraction over a padded axis meets
  at least one operand masked on that axis:
  - a store;
  - `V`, `Vc`, `D_dot` or `D` on `N`;
  - the stores' bond masks against a start.

  So the sample's padding, a start's padding and an unmasked output's padding can reach only *padded* output slots,
  never a real slot.
- **The single-layer masks.** These guard real parts on their own, and the covering design kills each of them alone:
  - the stores';
  - `V`/`H`/`Vc`;
  - `D_dot`/`L_dot`;
  - `y`;
  - the held variables'.

  The reason is the sums over `N` (`f`, `Jp_norm_sq`, `r_block_sumsq`) and probe's open mode (`z`, `Jp` are vectors
  over `N`), where one operand's padding is summed with nothing else masked.

**Each of the three is pinned alone by a permanent test.** The handover describes the controls as "removing one entry
family together with the exit mask". The stronger property is in their *first halves*, and my mutants confirm it:
- `test_without_the_directions_and_exit_masks...` first runs the check with the exit mask switched off, asserting that
  **the entry layer alone gives exact zero padding**. It is the only test that kills `sample_N_mask_off` and
  `start_mask_off`.
- `test_without_the_sample_and_exit_masks...` first runs it with the sample's mask switched off, asserting that **the
  exit layer alone gives exact zero padding**. It is the only test that kills `exit_mask_off`.

So neither layer can be removed or narrowed without a red test.

**Could one fail silently?**
- *Too strict:* a mask that zeroes a real slot is caught by the equivalence (`start_mask_wrong_end`, `rR_size_mask_left`
  and `nU_size_mask_down` are all killed).
- *Too permissive:* hidden by the other layer, and harmless, because padding is all it could leak. A wrong answer
  needs both layers wrong at the same slot.
- The one table shared by the two layers, `_SIZE_MASKS` (exit and held), was tested with every held variable dirty
  (check B), which is its single-layer use.

**Is one redundant?** Yes:
- the start mask duplicates the stores' bond masks for real parts and the exit mask for padding;
- the sample's `N` mask duplicates the operands' `N` masks and the exit mask.

I would still keep all three:
- each costs one multiply, and XLA fuses it under `jit`;
- the exit mask is what makes "exact zero padding" hold whatever a future step does (a constant start, a non-multilinear
  step);
- the entry masks make "every variable inside the engine has zero padding" true, which a future reader can rely on.

Dropping the start mask would save three lines and the `_STORE_FAMILY` coupling to `rule.reads[0]`. That coupling is
loud if it breaks (a `KeyError`), not silent, so it is not worth the churn of rewriting the directions control. **No
change requested.**

## 3. Mutants (25, mine)

**Killed:**

| Mutant | Killed by |
| --- | --- |
| sample's `N` mask off | the directions control's first half only (as above) |
| start mask off | the directions control's first half only |
| exit mask off | the sample control's first half only |
| start mask at the wrong end | 698 subtests |
| held mask off | the held test |
| index check off | the index test |
| negative index not wrapped | 307 subtests |
| masks left-aligned against the stacks (`W`/`K` before `C`) | everything |
| `y`'s mask off | 76 subtests |
| `Vc`'s mask off | 41 subtests |
| `V`'s mask off | 38 subtests |
| `H`'s mask off | 21 subtests |
| `D_dot`'s mask off | 20 subtests |
| `L_dot`'s mask off | 67 subtests |
| `D` without its `N` mask | everything |
| the uniform `N` check off | its test |
| `form='loop'` on uniform | the jaxpr program-size test |
| exit `HL` taking `H`'s masks | 90 subtests |
| `rR` size mask from the left family | 132 subtests |
| `nU` size mask from the down family | 132 subtests |
| `r_step` without the `C` axes | everything |
| `w_tilde_step` not stacked | the jit test |
| `g_step` returning tuples | the jit test |
| `pack_vectors` padding the jet axes at the front | the packing tests |

The last two of the step mutants (`w_tilde_step` not stacked, `g_step` returning tuples) are caught only by the jit
test: on numpy a tuple is silently coerced by the exit multiply, so the numbers stay right. Fine.

**Missed: `one_hot_not_packed`** (finding 2). `w_one_hot_step` returns a tuple of `d` padded one-hots instead of the
stacked array. `test_one_hot_w_has_the_padded_length` reads it with `np.shape` and `np.asarray`, which stack a tuple,
so a uniform entries call returning `'w'` in the ragged form passes.

## 4. Findings

**Finding 1 (should fix; silent wrong answer): the entries index's leading `d` is not checked.**
- *What happens:* on a uniform call, an index of shape `(1,) + W` with `d = 3` returns with no error. `_uniform_index`
  broadcasts it against the sizes `(d, 1)`, and the gather reads row 0 for every mode. I checked that the result equals
  the ragged call with the row repeated to every mode: `T[i, i, i]`, to 4e-15.
- *What ragged does:* raises an `IndexError` on the same input (by accident, at `index[1]`).
- *The other wrong sizes:*
  - a leading 2 raises a bare numpy broadcasting `ValueError` on uniform;
  - a leading 4 raises the same on uniform, and on ragged is **silently accepted** (the extra row ignored; pre-existing
    since P9b).
- *Why it matters:* this is the size-1 broadcast hazard that `_check_stacks` exists for, and the uniform branch checks
  the packed sample's `d` but skips entries.
- *Fix:* one structural check, raising a `ValueError` that names the shape and `d`, on both layouts:
  `np.shape(index)[0] == d` (and `ndim >= 1`). Since it is the analogue of a sample's length `d`, not a stack check, it
  belongs outside the removable `_check_stacks`. Add a test with leading 1, 2 and 4 on each layout.

**Finding 2 (should fix, one line): assert the layout of the one-hot `'w'`.** In
`test_one_hot_w_has_the_padded_length`, add `self.assertIsInstance(w, np.ndarray)` (or `is_ndarray(w)`), so the packed
form is asserted and not only the stacked shape. It kills the surviving mutant.

## 5. The departures

1. **Every output masked on exit: agree.** One multiply per output, and it is the layer pinned by the sample control's
   first half (section 2).
2. **Program size by the jaxpr's lines: agree.** The jaxpr is deterministic, unlike the compiled text, which depends on
   what the process compiled before. It still catches what matters: my `form='loop'` mutant is killed by it alone.
3. **The masks static under `jit`, said in the docstring: agree,** with one addition (nit a): `shape` must be static
   too. A traced `shape` fails inside `prefix_mask` with a tracer-conversion error rather than the named
   `require_concrete_masks` one. That is loud, but the docstring should say "the masks and `shape`".
4. **A negative index counts from `N_i`: agree.** It is numpy's ragged semantics; checked by the
   `negative_not_wrapped` mutant (killed).

## 6. Nits (optional)

- (a) The docstring's jit sentence, as above.
- (b) The module docstring says "on exit every output is masked". `'w'` given by the caller is returned as given, so a
  hand-packed sample with garbage padding comes back with it. That is the documented "the sample as given", but say
  "every output but a given `'w'`".
- (c) `test_an_eager_uniform_call_twice...` registers a `jax.monitoring` listener that is never removed, so it collects
  every later compile event in the process. Harmless, but unregister it in a `finally` where the jax version has the
  call, or filter by a flag the test clears.

## 7. Round 2

Findings 1 and 2, each with its test; nits at the implementer's discretion. I will re-run my `p11_check.py` (with the
index cases turned into assertions), the two mutants, and the new tests on both envs.
