# Review of the diagram hand fixes (2026.5.0), 2026-11-15

Reviewer for `dev/hand_fixes_2026-11-15.md` (implementer `a1aaf395631720c38`). The reference is HEAD `923178ff`. The
record says it was made "on HEAD `8ad6dea3`". The two later commits change only `dev/HANDOFF.md`, so the base is the
same. Every check below is my own run. Heavy jobs ran one at a time.

## What I checked

**1. The 28 edits, each against the code it annotates.**

- `git diff -U0` replaces exactly 28 lines one for one, in 27 hunks (`:74` and `:75` share one); item 29 later added one more. A character-level diff of each
  line pair shows only the intended token change and nothing else:
  - `ni`→`nDi` on the left and right cores;
  - `rRi`→`rR(i+1)` and `(rRi,)`→`(rR(i+1),)`;
  - `Dd`→`D(d-1)`;
  - `/outer` removed with six spaces added, so `einsum` stays at column 40;
  - `Tucker ` removed;
  - `unfolding`→`matricization`.
- The convention, read from the code itself:
  - `T3Frame.variation_shapes` zips `right_ranks[1:]`. Its doctest prints `((1, 10, 4), (2, 11, 5), (3, 12, 1))`
    against `right_ranks = (2, 4, 5, 1)`.
  - `fv_variation_shapes` reads `right[i].shape[-1]`.
  - `probing._sigma_step` contracts `H` as `'WCa,KCaib,WCi->WKCb'`. Its `b` is the next carry, which `t1` builds from
    `R_i`'s trailing axis.
  - `T3Frame.validate` raises unless `D.shape[-2] == L.shape[-2] == R.shape[-2]`. That is items 1–2.
- Items 7–11, 18–20 and 24 are `var_tt_cores`, the TT half of `T3Variations.data`. Each one flows into the kernels
  whose signatures already say `rR(i+1)`.
- Items 21–23 are `H_tildes`, the gradient with respect to `H`, so they have `H`'s shape:
  - in `_tt_variation_jets_step`, the `b` leg comes from `sigma~` or `nu`, both `rR(i+1)`;
  - in `_apply_derivatives_transpose_from_jets`, the `b` leg comes from `sigma_hat`, which is `rR(i+1)`.
- Items 12, 27 and 28 are the weights:
  - `fv_absorb_weights` keeps `H`'s shape (`'...aib,...a,...i,...b->...aib'`);
  - `right = tt_weights[1:]`;
  - `fv_weights_consistent` requires `right[i].shape == stack + (H_i.shape[-1],)`.
- Items 25–26 are `T3Variations.data` passed through untouched.
- Item 15 follows the glossary's own entry: the up core's `ni × (ri·r(i+1))` matricization.
- Item 16 is the N4 row (the retired alias). Item 17 is N15. The example's claim is right as stated.

Every BEFORE is an error under the bond-indexed convention, and every AFTER is exactly right.

**The two diagrams** (`frame_variations_format.py:89–97`, `:101–109`). I re-derived both by eye and checked them with a
column ruler and token-start columns, HEAD against the tree:

- **Bonds.** The bonds are `L0 (rL0, nD0, rL1)`, `H1 (rL1, nD1, rR2)`, `U1 (rL1, nU1, rR2)` and
  `R(d-1) (rR(d-1), nD(d-1), rRd)`, so the top labels `rL0 rL1 rR2 rR(d-1) rRd` are right. Under the edges, `nD0`,
  `nD1`/`nU1` and `nD(d-1)` meet `D0`, `D1`/`(V1)` and `D(d-1)`, with free legs `N0`, `N1`, `N(d-1)`.
- **Edges.** On every edge line the `|` are at 17, 27 and 48.
- **Labels.** `L0`/`D0` are at 17. `(H1)`'s `H`, `U1`, `D1` and `(V1)`'s `V` are at 27. `R(d-1)` and `D(d-1)` are at
  48 in both diagrams.
- **Movement.** HEAD's second-diagram `Dd` was at 49, one column right of its edge. Only line 94 and line 106 differ
  from HEAD, and every other token on those lines is at its HEAD column.

**2. The AST claim.** I compared `ast.dump` of HEAD against the tree for all eight touched `.py` files:
- `apply`, `entries`, `fv_operations`, `sampling_derivatives`, `sharing` and `ufv_conversions` are identical even with
  docstrings, so the changes are comments only.
- `frame_variations_format` and `uniform_fitting` are identical once docstrings are removed. The changed docstrings are
  `T3Frame`, `variation_shapes`, `is_orthogonal`, `T3Variations` and `uniform_least_squares_problem`.
- No code reads a docstring at runtime: there is no `__doc__`, `getdoc` or `get_docstring` in the package or the tests.

So behaviour is bitwise unchanged by construction, and no suite or battery run is needed.

**3. The inventory.**
- The seven blocks (DOC-FV-0080, FVF-0071, -0077, -0088, -0100, -0354, -0784) each have `new text` equal to the tree
  at their `new span`. No span line changed.
- I called the gate's own `k10`/`k10b` on the four flipped blocks:
  - K10 fails on exactly the length changes (`74→78`, `50→54`, `51→54`, `89→93`);
  - K10b passes on all four, so each n→y flip is justified.
- FVF-0071 and -0354 were already `y`, and K10b passes on both.
- `store_rename_diagram_check.py all`: exit 0. The self-test passes; K10/K10b 33 in place / 27 re-laid out; K11 94
  spans; K12 301/301.
- `store_rename_check.py --stage R6 --hits dev/store_rename_slices --prev HEAD`: exit 0, every gate green. I also ran it
  on a temporary HEAD worktree and diffed the two outputs. The only difference is the K10/K10b counts line (37/23 →
  33/27).

**4. Completeness.** This is a one-time closed enumeration, not a rule. I grepped `t3toolbox docs tests examples
README* CHANGELOG*` for `rRi`, `rR_i`, `rRᵢ`, `rR_{i}`, `r_R`, `(rL…, n…i, rR…i)` and `right_weights … (rRi,)`.
Every remaining site and my verdict:

| site | what it is | verdict |
|---|---|---|
| `backend/sharing.py:814` `(K+C)+(rLi, ni, rRi)` | `fv_share_tucker_variations_corewise`: the corewise TT perturbation `dG_i`, shape `(r_i, n_i, r_(i+1))`; the corewise frame is `(B, G, G, G)`, so nD = nU = n and rL = rR = r | **error** in the right bond: `rRi` → `rR(i+1)`. `ni` is **correct** here: it pairs with `(ni, Ni)` on `:813`, and the corewise directions are written with `ni` at `apply.py:344–345` and `entries.py:275–276`. → item 29 |
| sigma carries `W+K+C+(rRi,)`: `apply.py:435`, `probing.py:391, 407, 494`, `sampling_derivatives.py:722, 739, 783, 834, 1037` | the carry *entering* core `i` (`'WKCa,Caib'`, where `a` is `R_i`'s first axis) | correct: bond `i` |
| `fv_operations.py:502` `len=d+1, C+(rLi, rRi)` | the zipper, one per bond | correct: bond-indexed |
| `t3_linalg.py:156, 159` | names bound on `:157` from the array's own shape | correct |
| `(rRi, nDi, rR(i+1))` on R cores (many) | the frame's right cores | correct |
| `rL_i*rR_i`: `t3_svd.py:318`, `ranks.py:80, 83, 87, 545`, `tucker_tensor_train.py:4590–4591`, `docs/t3svd_minimal_ranks.md:16–18, 32, 35, 46, 49, 52–53`, `tests/backend/test_ranks.py:198`, `tests/test_sharing.py:706`, `CHANGELOG.md:967–968` | per-core T3 bonds, defined at `t3svd_minimal_ranks.md:21` (`rL_i = r_i`, `rR_i = r_{i+1}`); WD rows in the R1 reconciliation | correct under that local convention; not this class (see NIT 1) |
| `docs/exact_hessian_two_hole.tex:118, 126`, `docs/padded_frame_cone.tex:116` | the math notes, 1-based (`H_i ∈ r^L_{i-1} × n_i × r^R_i`) | correct under the notes' convention; the notes' rows are deferred to their rebuilds |
| `docs/autoapi/…/fv_share_tucker_variations_corewise.rst` | generated and gitignored | follows `:814` at the next build |

The uniform shapes `(rL, nD, rR)` carry no index and are out of scope. Once item 29 is in, the release ships no known
site of this class.

**5. Doctests and builds.**
- Module doctests of the eight touched modules (`pytest --doctest-modules`): 38 passed on the current env (py3.11,
  numpy 2.x, jax 0.10) and 38 passed on the floor env (`tttt`).
- `docs/frame_variations.md`: no examples; it passes on both envs.
- `docs/getting_started.rst`: passes on both envs.
- `git diff --check`: clean.
- `sphinx -W --keep-going -b html docs <fresh dir>`: exit 0, "build succeeded", no warning (17 min, run alone). It started before item 29, which changes one comment inside a signature, so no re-run.

## Findings

### MAJOR

None.

### MINOR

1. **`backend/sharing.py:814` is a 29th site of the same error class**, and the record had listed it as "still not
   changed". It is the corewise TT variation, and its right bond is written `rRi`. Fix: `(rLi, ni, rRi)` →
   `(rLi, ni, rR(i+1))`, and keep `ni` (evidence in the table). Messaged to the implementer as item 29. **Resolved**: the tree now reads `(K+C)+(rLi, ni, rR(i+1))`. I checked it myself:
   - its line is the only change in that hunk;
   - `sharing.py`'s full AST is identical to HEAD;
   - the record's item 29 has verbatim BEFORE/AFTER lines and the evidence, and its intro says 29 edits;
   - "Still not changed" now matches this sweep.
   `t3toolbox` + `docs` now differ from HEAD in 28 hunks, 29 lines.

### NIT

1. **The per-core `rL_i*rR_i` notation collides with the frame's bond-indexed `rR(i)`.** Under the frame convention,
   `rL_i*rR_i` reads as `r_i²`. It is defined at `t3svd_minimal_ranks.md:21`, and the store rename listed it as WD and
   left it. It is confusing only where it sits next to frame widths:
   - `tests/test_sharing.py:706`: `nU_i = min(n, rL_i*rR_i)`;
   - `tucker_tensor_train.py:4264`: `rL_0*r_1`, a mixed spelling.
   This is a possible post-release notation question for the maintainer. No change now: it is outside the closed list,
   and the convention is defined.
2. The record's opening line names the base `8ad6dea3`. HEAD is `923178ff`, which differs only in `dev/HANDOFF.md`.
   Harmless; it could say so.

VERDICT: APPROVE
