# P11 design review: the implementer's note on Engine G (the uniform layout, sharding over `W`)

Reviewer, 2026-11-20, on `P11_DESIGN_NOTE.md` (read only; nothing run). Against the plan's P11, sections 2.2, 2.3
(steps 2 and 8) and 7.2; `docs/contributor/testing_strategy.md` (the two tools, the two-hole note, "isolate it");
`docs/uniform_equivalence_contract.md` (the contract, the packedness mirror); `docs/uniform_masks_vs_ranks.md`; and
`backend/ufv_masking.py`, `backend/ut3_masking.py` as they stand.

**Overall: the structure is right.** Mask once on entry, steps mask-free, re-mask on exit, masks host numpy, the
covering design with garbage everywhere and expected masks derived from the ragged ranks.

Three findings change the design:
- **A. One mask must be explicit:** the shape mask on the `N` axis of every Tucker-shaped input and output. The note's
  "the sample needs no mask" rests on it.
- **B. Held variables can carry garbage into `f`.**
- **C. An index of `N_i` or more gives a silent zero on uniform** where ragged raises, which is a broken equivalence,
  and the coordinator's own ruling on `index_to_one_hot` already says how to fix it.

And one recommendation: point 3, infer the layout in the whole steps rather than adding a `layout` static.

---

## 1. The entry masks: right in principle; findings A and B

**What is right:**
- the stores with their own masks (`ufv_apply_frame_masks`, `ut3_apply_masks`);
- `V`, `H` and `Vc` with the frame's variation masks;
- `D_dot` and `L_dot` with the frame's `D` and `L` masks, never the point's. At TT slack the point's are a strict
  superset, and a direction's padded slot multiplies into real slots: the two-hole note, the one place where the
  masks are correctness, not hygiene;
- `y` masked on `N` (it reaches `f` and `r_block_sumsq`, which sum over `N`);
- the start values masked by their end bond's mask, so every variable has exactly zero padding. That is a stronger
  statement than "don't care", and easier to test.

**Finding A: the `N` axis of every Tucker-shaped input and block needs the shape mask, said explicitly.**
- *Why the sample is safe today:* "the sample needs no mask: every contraction with `w` meets a masked `D`" holds for
  `D` because `ufv_apply_frame_masks` multiplies `D` by the **shape mask** `prefix_mask(shape, N)` as well as the down
  mask (`ufv_masking.py`, `SM_k * DM_k`).
- *Where it must also hold:* `w` also meets
  - `V` (in `xi_U_dot = V w`),
  - `D_dot` (in the raw `xi_dot`),
  - `Vc` (in `z` through the centers),
  - and the readouts `V_term(·, w)` produce `N` columns.

  For each, garbage in `w`'s padded `N` meets that operand's padded `N` columns. The variation masks of
  `ufv_variation_masks` are rank masks (`down`, `up`, `left`, `right`) and say nothing about `N`. If the entry masking
  of `V`, `Vc` or `D_dot` applies only those, `(garbage V column) × (garbage w entry)` lands in a real slot of
  `xi_U_dot` or `z`.
- *Resolve, one of:*
  1. **The table states the shape mask** on the `N` axis of every Tucker-shaped input (`D`, `B`, `V`, `Vc`, `D_dot`) and
     of every `V`-type block on exit (`V`, `VD`), beside its rank mask. The garbage tests then put garbage in the `N`
     padding of each (the note's list already includes "of `N` in the sample").
  2. **Simpler, and my recommendation:** mask the sample's `N` axis on entry too (one host mask, one multiply per
     mode; the jet axes untouched, O-45). Then "every input is masked by the object it is defined at" holds without
     exception, and the safety of `w` does not depend on every other operand remembering its `N` mask.

  Do (2), and still state the shape mask for `V`, `Vc`, `D_dot` (a user's `V` with garbage `N` columns must not
  reach `xi_U_dot` even with a clean sample). The covering design's garbage then tests each.

**Finding B: an unmasked held variable can change `f`.**
- *The contract:* held variables are trusted (row 6.5: shapes checked, values trusted). But the uniform contract says
  padding is don't-care, and a held variable is an input.
- *Where the padding is inert:* in most rows it meets a masked core (a held `mu` against masked `L`; a held `xi`
  against masked `L` and `U`).
- *Where it is not:*
  - a held `z` or `r` reaches `f` and `r_block_sumsq`, which **sum over `N`, padding included**;
  - a held `w_tilde` reaches the `V` readouts (re-masked on exit, so harmless) and `xi_tilde` (`D` masked on `N`,
    harmless).

  So garbage in a held `z` or `r`'s `N` padding gives a wrong `f`, silently.
- *Trust is about values; the padding is not a value under the contract.* "Whoever holds it vouches" (6.5) is not
  "whoever holds it must zero the padding". The engine's own outputs have zero padding, which is why a round trip is
  clean, but a user packing a ragged `r` by hand, or editing a held array, can leave garbage.
- *Resolve:* mask the held variables on entry with their own table's masks: the shape mask for the mode variables
  `z`, `r`, `w_tilde`, and the rank masks for the bond and site variables. It is the same one multiply as the other
  inputs, and the shape table already names each variable's size, so the mask follows from it as on exit. Add held
  variables with garbage padding to one case of the covering design.

## 2. The exit masks: agree, with A's shape mask

Reading each block's mask off its size in the shape table is the right single source:
- `V` and `H` from the variation masks;
- `VD` and `HL` from the frame's `D` and `L`;
- the point's own on a corewise call;
- broadcast over `[W] + K + C`.

The `V`-type blocks (`V`, `VD`, and entries' scatter output) need the **shape mask on `N`** as well as their rank mask
(finding A). The readout `V_term(·, w)` writes into every `N` column, padded ones included, whenever the sample's
padding is not zero.

## 3. The whole steps' `layout` static: disagree. Infer it from the operand.

- **Row 5.2 holds either way.** It asks for one set of step functions for both layouts, and a whole step already acts
  on whole variables, whose representation differs (a tuple of `d` arrays, or one packed array).
- **But a `layout` flag is against the library's own convention.** Ragged against uniform is *inferred at the lowest
  level* from the structure (`is_ndarray(...)`; `CLAUDE.md`'s dispatch convention, and
  `docs/contributor/uniform_polymorphism.md`: "a polymorphic op needs no flag").
- **Inferring is also simpler and enough:**
  - `r_step` reads `is_ndarray(z)`;
  - `g_step` returns `(V_tilde, H_tilde)` as given (a packed supercore pair is already a pair of arrays; `tuple()` was
    only needed on ragged);
  - `w_tilde_step` stacks `jet_to_dual_jet`'s tuple when its input was packed;
  - `w_one_hot_step` packs when `D` is a supercore.

  No new static, no new memo-key field, nothing threaded.
- **The vectorized probe sums over the leading `d`** (the weight `(d,) + F` against `(d,) + F + ...`) are then the
  `is_ndarray` branch of the same step.

Add one sentence to the plan's P11 ("the whole steps act on the representation they are given, inferred"), not a new
argument.

## 4. `_check_stacks`'s uniform branch: agree

Checking the packed sample's leading `d` and trailing `N` against `D`'s is right. An `N` of 1 in a hand-packed sample
would broadcast through the `'o'` label silently, the same hazard as the size-1 stacks of P7 and P10. Keep it in the
removable function, and in its test class, family by family (a wrong `d`, a wrong `N`, a size-1 `C` on one
supercore). Each must be isolated: the other checks pass for that input (`testing_strategy.md`, "isolate it").

## 5. An index of `N_i` or more on uniform: disagree with leaving it. Finding C, a silent wrong answer.

- **On ragged** a numpy index `>= N_i` raises an `IndexError` (numpy indexing in the gather, and `index_to_one_hot`
  since P4b's ruling).
- **On uniform** it reads a padded column of `D`, which the shape mask zeroed, so the entry comes back **0** with no
  error, unless it is `>= N`.

That breaks the equivalence contract (`to_uniform -> op -> to_ragged` should equal `op_ragged`, and here one raises
and the other returns a plausible zero), and it is the silent-wrong-answer class.

**The coordinator's ruling on `index_to_one_hot` (P4b) already settles it:** "an index outside the grid has no one-hot
vector ... one comparison on a numpy array is free. It is not a numerical precondition (safe mode), just what indexing
already does."

*Resolve:* on a uniform entries call, a concrete numpy index is checked against the real `N_i` (from `shape`). Out of
range raises `IndexError`, naming the mode and the bad values, as `index_to_one_hot` does. A jax index (maybe traced)
cannot be checked, and the docstring says it reads a zero there (jax's gather on ragged clamps, a documented
difference already).

This is not a check of values in the O-18 sense: it is the indexing the ragged layout does. It is one comparison per
mode, cheap.

## 6. The tests: agree with the covering design, with additions

**What is right:**
- every uniform call carries garbage in every masked slot;
- each is compared per `C` element with ragged on real parts;
- the exact zero padding is asserted, with masks derived from the ragged ranks (`np.arange(size) < rank`),
  non-circular;
- `Kp` at TT slack with a permanent negative control;
- the packed jet zeros as data (O-45);
- sharding in four programs.

The pairwise covering is justified the way the note says: uniform bugs live in masking (configs) or packing (jet
cases), and the ragged slices already cover the kind and call-type rows.

**Additions:**
1. **The `N` padding of every Tucker-shaped input** (`V`, `Vc`, `D_dot`) gets garbage, not only the sample's (finding
   A).
2. **Held variables with garbage padding** in at least one case: a held `r` or `z` wanting `f` (finding B).
3. **The index check** (finding C): a numpy index `= N_i` raises on uniform as on ragged.
4. **Isolated negative controls by mask family.** The `Kp` control is right, and should not be the only one. Each
   mask family the engine applies should turn a test red when removed:
   - the stores';
   - `V`/`H`/`Vc`;
   - `D_dot`/`L_dot`;
   - `y`;
   - the start values';
   - the exit masks;
   - and, if adopted, the sample's and the held variables'.

   Do this in the mutation pass at least, and keep permanent controls where a mask is load-bearing for correctness:
   `core_directions` (already), and the exit masks (exact zero padding).
5. **The maximal case's uniform twin** (forced padding, varying ranks across `C = (2, 3)`, `K = (2, 3)`, `W = (2, 2)`,
   `(1, 1)`), as the note says, with `sum_over_samples=False` on one kind. The unsummed blocks' exit mask, broadcast
   over `W + K + C`, is a new shape for the exit masks.

The cost stays small: a few more calls on top of the note's estimate of about 30 s added to the suite's wall time.

## 7. The uniform layout out on a uniform call; mirroring left to P12: agree

The engine is the backend's one entry. Returning packed outputs on a uniform call, and leaving the "packed in, packed
out; ragged in, ragged out" mirror to the twins (`docs/uniform_equivalence_contract.md`, "the vector I/O boundary"),
keeps the engine to one layout per call and the mirror in one place. The note's item for P12 (keep the masks static
under `jit`, with `require_concrete_masks` raising on a traced one) is right and belongs in the twins' docstrings.

## Summary of requested changes

1. **(A)** The shape mask on `N` for every Tucker-shaped input and `V`-type block, stated in the table. The sample's
   `N` masked on entry too (recommended).
2. **(B)** Held variables masked on entry by their table's masks.
3. **(C)** A concrete numpy index checked against the real `N_i` on uniform, `IndexError` as on ragged.
4. **(3)** The whole steps infer the layout from their operands; no `layout` static.
5. **(6)** The test additions: `N` garbage on `V`/`Vc`/`D_dot`, garbage held variables, the index, negative controls
   by mask family, the uniform maximal case with `sum_over_samples=False`.
