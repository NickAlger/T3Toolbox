SIGN OFF

# Review of `PACKING_PLAN.md`, revision 5 (the diff from revision 4)

Reviewer: Claude (critic), 2026-11-25. I read `DIFF_rev4_to_rev5.txt`, `RESPONSE_r3.md`, and revision 5 where the diff
needed context. I checked the two questions against the tests, the translation and the code (`tests/_fits.py`,
`tests/test_oracle_data_space.py`, `t3toolbox/optimizers.py`), and ran no new experiment.

## Status of round 3's findings

| # | Status |
|---|---|
| S1 | Resolved. K1's part 0 lists the consumers; the display reads the order off the batch; the protocol's change and O-29's reversal are named; the sentences on user oracles now agree. |
| S2 | Resolved. One kind and one order per fit, refused at the frontend entries, at the hand-over of a draw, and in the backend display. |
| S3 | Resolved. The mode, `F` and the edge are exact; the axes between them may be 1 or full. |
| S4 | Resolved. One wording point is left for the builders: note 1 below. |
| S5 | Resolved. Note 2 below says when the refusal starts. |
| S6 | Resolved. |
| M1 | Resolved, except two sentences still give the backend batch a `pack` method: note 3. |
| M2 to M6 | Resolved. |
| §14.1 | Kept as I suggested. |

## The two questions

**1. The rule for S4.**
- **Nothing legitimate on the ragged layer violates it.**
  - `y` is exact on `F` and `W`, so only the pattern of its `C` axes is concerned.
  - A full weight's structural axes are exact, so only its `W` and `C` patterns are concerned.
  - The translation reshapes every mode alike (`_shared_by_the_stack`).
  - `prepare_*_derivative_data` returns an exact `y` and a block weight.
  - The engine's `r`, `Jp` and `c` have one lead shape.
  - The only tests with per-mode patterns that differ, such as `tests/test_oracle_data_space.py:181-188`, draw their
    size-1 axes at random on the structural axes too, so they test the old rule; K1 rewrites them anyway.
- **The wording is not tight enough to exclude the sample** (note 1).

**2. The refusal of the backend's batch.**
- **The refusal is coherent on the ragged layer.** It is type discipline: a frontend entry takes frontend objects,
  and the frontend class checks at construction. It carries no numerical reason there (the tuples carry their
  sizes), so it buys consistency at the price of some churn.
- **It can only start in K4.** Until then `t3toolbox.fitting.LeastSquaresBatch` is the backend class itself (note 2).

## What the builders should know

1. **S4's wording** (§2.2, at the bullet "The modes of a probe object agree in everything but the edge"; §3.1, at
   "One packer serves the sample and every probe object of the type of `z`"):
   - Write "a probe object of the type of `z` (`y`, a full weight, `r`, `Jp`, `c`)". The sample is a per-mode object
     of the probe kind too, and its jet sizes may differ by mode: they are zero-filled when packed, which is why the
     packer has `len_P`. Applied to the sample, the rule would refuse legal input jets.
   - §3.1's sentence reads as if the sample's arrays also "agree in everything but the edge". Say that the sample
     agrees in `W` and differs in the jet sizes and the edge.
2. **The refusal of the backend's batch starts in K4,** with the frontend class. Say so in §4.3 or in K4. K4's design
   note should also measure the other direction of the one it lists: backend batches handed to frontend entries.
   - `tests/_fits.py:232` passes the shared harness's backend batch to `fitting.local_model`.
   - `tests/test_dispatch.py` and `tests/test_jax_absent.py` import the backend class beside frontend calls.
   - The translation imports the backend class (`t3toolbox/optimizers.py:57`). Its K4 row calls `batch.pack(...)`,
     which is the frontend class's method only, so from K4 it builds the frontend class.
3. **Two sentences still give the backend batch a `pack` method,** against M3's resolution (in the backend, packing
   stays the function `pack_batch`; `pack` belongs to the frontend class and arrives in K4):
   - §4.3's opening paragraph: "The batch knows its kind (both tiers) … `pack(N=None)`";
   - §5's row for `LeastSquaresOracle(kind, order)`: "… with `check`, `take`, `n_measurements`, `pack` | K1".

## Not verified

- The docs build with two classes of one name (now on K4's list).
- T3Polynomial's code.
- No experiment this round; the conclusions of E1 to E8 stand.
