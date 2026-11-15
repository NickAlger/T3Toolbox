# The diagram hand fixes (2026.5.0), 2026-11-15

The last content change before 2026.5.0, on HEAD `8ad6dea3`. The seventeen fixes: the seven in-block rows the docs
pass deferred (`glossary_docs_pass_2026-11-15.md`, § Deferred) and the review's list gaps (R1-M4 in
`glossary_docs_pass_review_2026-11-15.md`). Then the eleven further sites this pass found (items 18–28, § Extension), and one the reviewer asked for
(item 29). That makes 29 edits.
Every edit was made by hand with the Edit tool. Each edited inventoried block
was re-derived whole and its alignment checked by eye with a column ruler; no script touched a block. Nothing else
changed. Nothing is committed.

Columns are 0-based throughout, as in the docs pass record. A `BEFORE: ` / `AFTER:  ` prefix is 8 characters; the
line follows verbatim.

## The shape convention, from the code

The TT variation `H_i` takes its left bond from `L_i`, its middle leg from `D_i`, and its right bond from the trailing
axis of `R_i`, which is the next bond `rR(i+1)`.

- `T3Frame.variation_shapes`, `frame_variations_format.py:245–247`:
  `(rL, nD, rR) for rL, nD, rR in zip(self.left_ranks[:-1], self.down_ranks, self.right_ranks[1:])`.
- Its backend twin, `fv_operations.fv_variation_shapes`, `fv_operations.py:109`:
  `(left[i].shape[-3], down[i].shape[-2], right[i].shape[-1])`.
- The doctest at `frame_variations_format.py:241–242`, the ground truth: with `right_ranks = (2, 4, 5, 1)` it prints
  `((1, 10, 4), (2, 11, 5), (3, 12, 1))`, so the third entries are `rR1, rR2, rR3`.
- The kernels contract `H` that way: `probing._sigma_step` (`probing.py:385`),
  `t2 = contract('WCa,KCaib,WCi->WKCb', mu, H, xi)`. Here `a` meets `mu` (`rL_i`), `i` meets `xi` (`nD_i`), and `b`
  is the next carry, which `t1` builds from `R`'s trailing axis (`rR(i+1)`). `probing.py:401` and
  `sampling_derivatives.py:733, 763, 829, 1052` already write `(rLi,nDi,rR(i+1))`.

The left and right frame cores' middle leg is the frame's `nDi`:

- `T3Frame.validate`, `frame_variations_format.py:553`: `if not (D.shape[-2] == L.shape[-2] == R.shape[-2])` raises.
- The class's own field comments, `:159–160`: `Li.shape = stack_shape+(rLi, nDi, rL(i+1))`,
  `Ri.shape = stack_shape+(rRi, nDi, rR(i+1))`.
- The class doctest, `:147–153`: the left cores' middle legs `10, 11, 12` equal `down_ranks = (10, 11, 12)`.

## The seventeen fixes

### 1. `t3toolbox/frame_variations_format.py:74` (FVF-0071, glossary row L873)

```
BEFORE:         - left_tt_cores     = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))
AFTER:          - left_tt_cores     = (L0, ..., L(d-1)), elm_shape=(rLi, nDi, rL(i+1))
```

Evidence: `validate` (`:553`), the field comment `:159`, the doctest (`:149`, `:153`). `ni` was undefined here and
differs from `nDi` at a slack frame.

### 2. `:75` (FVF-0071, L874)

```
BEFORE:         - right_tt_cores    = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))
AFTER:          - right_tt_cores    = (R0, ..., R(d-1)), elm_shape=(rRi, nDi, rR(i+1))
```

Evidence: `validate` (`:553`), the field comment `:160`.

### 3. `:79` (FVF-0077, L875)

```
BEFORE:         - tt_variations     = (H0, ..., H(d-1)), elm_shape=(rLi, nDi, rRi)
AFTER:          - tt_variations     = (H0, ..., H(d-1)), elm_shape=(rLi, nDi, rR(i+1))
```

Evidence: `variation_shapes` (`:245–247`) and its doctest (`:241–242`).

The class docstring's two listings now read, whole (`:72–79`):

```
        - down_tucker_cores = (D0, ..., D(d-1)), elm_shape=(nDi, Ni)
        - up_tt_cores       = (U0, ..., U(d-1)), elm_shape=(rLi, nUi, rR(i+1))
        - left_tt_cores     = (L0, ..., L(d-1)), elm_shape=(rLi, nDi, rL(i+1))
        - right_tt_cores    = (R0, ..., R(d-1)), elm_shape=(rRi, nDi, rR(i+1))

    The components of T3Variations are the "variation cores":
        - tucker_variations = (V0, ..., V(d-1)), elm_shape=(nUi, Ni)
        - tt_variations     = (H0, ..., H(d-1)), elm_shape=(rLi, nDi, rR(i+1))
```

### 4. `:241` (R1-M4), the doctest comment

```
BEFORE:         >>> print(tt_holes)       # one (rLi, nDi, rRi) per mode
AFTER:          >>> print(tt_holes)       # one (rLi, nDi, rR(i+1)) per mode
```

Evidence: the printed output on the next line, `((1, 10, 4), (2, 11, 5), (3, 12, 1))`. Only the comment changed, so
the example's output is unchanged.

### 5. `:786` (FVF-0784, R1-M4), the `T3Variations` docstring

```
BEFORE:         - tt_variations        = (H0, ..., H(d-1)), elm_shape=stack_shape+(rLi, nDi, rRi)
AFTER:          - tt_variations        = (H0, ..., H(d-1)), elm_shape=stack_shape+(rLi, nDi, rR(i+1))
```

Evidence: the docstring's own doctest builds the variations from `frame.variation_shapes` (`:806–810`).

### 6. `:814` (R1-M4), the `T3Variations.tt_variations` field comment

```
BEFORE:     tt_variations:     typ.Tuple[NDArray,...]  # len=d, elm_shape=stack_shape+(rLi, nDi, rRi)
AFTER:      tt_variations:     typ.Tuple[NDArray,...]  # len=d, elm_shape=stack_shape+(rLi, nDi, rR(i+1))
```

Evidence: as 5.

### 7. `t3toolbox/backend/apply.py:145` (R1-M4), `tv_apply`

```
BEFORE:             typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nDi,rRi)
AFTER:              typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nDi,rR(i+1))
```

Evidence: `var_tt_cores` goes to `_apply_from_xis` (`apply.py:172`). Its scan (`:456`) steps
`probing._sigma_step` (above). The module's compact spelling (no spaces) is kept, matching `probing.py:401`.

### 8. `apply.py:236` (R1-M4), `tv_apply_jacobian_from_sweep`

```
BEFORE:             typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nDi,rRi)
AFTER:              typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nDi,rR(i+1))
```

Evidence: as 7 (`_apply_from_xis`, `apply.py:264`).

### 9. `t3toolbox/backend/sampling_derivatives.py:900` (R1-M4), `tv_probe_jacobian_derivatives_from_sweep`

```
BEFORE:             typ.Sequence[NDArray],          # var_tt_cores     H.  len=d, elm_shape=K+C+(rLi,nDi,rRi)
AFTER:              typ.Sequence[NDArray],          # var_tt_cores     H.  len=d, elm_shape=K+C+(rLi,nDi,rR(i+1))
```

Evidence: `var_tt_cores` goes to `compute_sigma_jets`, `compute_tau_jets` and `compute_deta_jets` (`:922–924`),
whose signatures (`:733, 763, 829`) read `K+C+(rLi,nDi,rR(i+1))`.

### 10. `sampling_derivatives.py:933` (R1-M4), `tv_probe_derivatives`

The same line, the same fix. Evidence: it runs the same three sweeps.

### 11. `sampling_derivatives.py:1077` (R1-M4), `tv_apply_jacobian_derivatives_from_sweep`

The same line, the same fix. Evidence: `var_tt_cores` goes to `_apply_derivatives_from_jets`, whose signature
(`:1052`) reads `K+C+(rLi,nDi,rR(i+1))`.

### 12. `t3toolbox/backend/fv_operations.py:637` (R1-M4), `fv_absorb_weights`' return

```
BEFORE:     typ.Tuple[NDArray, ...],  # weighted tt_variations,     elm_shape=stack+(rLi, nDi, rRi)
AFTER:      typ.Tuple[NDArray, ...],  # weighted tt_variations,     elm_shape=stack+(rLi, nDi, rR(i+1))
```

Evidence: the einsum `'...aib,...a,...i,...b->...aib'` (`:654`) keeps `H`'s shape, and `H` is a TT variation.

None of the nine sibling sites has a different convention.

### 13. `frame_variations_format.py:94` (FVF-0088, L877), the first edge-rank diagram

`Dd` → `D(d-1)` at column 48, the line's last label. The block, re-derived whole (`:89–97`), BEFORE:

```
           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ----- (H1) ----- ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nD0     | nD1                | nD(d-1)
                 |         |                    |
                 D0        D1                   Dd
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
```

AFTER:

```
           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ----- (H1) ----- ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nD0     | nD1                | nD(d-1)
                 |         |                    |
                 D0        D1                   D(d-1)
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
```

Re-derived:
- The bonds are `L0 (rL0, nD0, rL1)`, `H1 (rL1, nD1, rR2)` (the `rR(i+1)` convention above) and
  `R(d-1) (rR(d-1), nD(d-1), rRd)`.
- Each TT core's middle leg `nDi` meets `Di (nDi, Ni)`, whose free leg is `Ni`.
- The modes are 0-based, so the last Tucker core is `D(d-1)`, as `nD(d-1)` and `N(d-1)` in the same column say.

Alignment, measured: every vertical edge `|` is at 17, 27 and 48 on all six edge lines. `L0`, `D0` are at 17. `(H1)`'s
`H` and `D1` are at 27. `R(d-1)` and `D(d-1)` are at 48. Nothing else moved.

### 14. `:106` (FVF-0100, L879), the second edge-rank diagram

`Dd` (col 49) → `D(d-1)` at col 48, under its edge. The line is rewritten whole: 17 spaces, `D0`, 7 spaces, `(V1)`,
18 spaces, `D(d-1)`. The block (`:101–109`), BEFORE:

```
           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ------ U1 ------ ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nD0     | nU1                | nD(d-1)
                 |         |                    |
                 D0       (V1)                   Dd
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
```

AFTER:

```
           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ------ U1 ------ ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nD0     | nU1                | nD(d-1)
                 |         |                    |
                 D0       (V1)                  D(d-1)
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |
```

Re-derived: `U1 (rL1, nU1, rR2)`, whose middle leg `nU1` meets the hole `V1 (nU1, N1)`. The rest is as 13.

Alignment, measured: the edges are at 17, 27 and 48. `D0` is at 17. `(V1)` spans 26–29, so `V` is at 27. `D(d-1)` is
at 48, under its `|` and `R(d-1)`, and in the same column as the first diagram's `D(d-1)`.

### 15. `docs/frame_variations.md:119` (DOC-FV-0080, L201)

```
BEFORE: | `Uᵢ` (up) | `nᵢ × (rᵢ·rᵢ₊₁)` unfolding, rows orthonormal | `nᵢ ≤ rᵢ·rᵢ₊₁` |
AFTER:  | `Uᵢ` (up) | `nᵢ × (rᵢ·rᵢ₊₁)` matricization, rows orthonormal | `nᵢ ≤ rᵢ·rᵢ₊₁` |
```

The reshape puts the mode leg against both bonds. That is a Tucker–TT cut, so it is a matricization (the M/U rule).
This is a markdown table, so there is nothing to align.

### 16. `frame_variations_format.py:356` (FVF-0354, L884)

```
BEFORE:             - up/outer U_i (all i):     ``einsum('...iaj,...ibj->...ab', U, U) = I``
AFTER:              - up U_i (all i):           ``einsum('...iaj,...ibj->...ab', U, U) = I``
```

Eleven spaces, so the backquoted `einsum` stays at column 40 on all four lines (measured). The review's optional
`down_tucker D_i` → `down D_i` on the line above is not a list row and was not made.

### 17. `t3toolbox/backend/uniform_fitting.py:503` (R1-M4, N15)

```
BEFORE:     A **non-minimal** frame -- here TT bond rank 3 is unrealizable for a 2x2x2 central Tucker core (its TT
AFTER:      A **non-minimal** frame -- here TT bond rank 3 is unrealizable for a 2x2x2 central core (its TT
```

The structure: the example's `x0` is `TuckerTensorTrain.randn((6, 6, 6), (2, 2, 2), (1, 3, 3, 1))`. Its Tucker ranks
are `(2, 2, 2)`, so the central core (the Tucker decomposition's core tensor, stored as a TT) is 2×2×2. Its TT bonds
are at most `min(2, 2·2) = 2`. Checked: `minimal_ranks` gives `((2, 2, 2), (1, 2, 2, 1))`, and so does
`uniform_minimal`. The claim was right; only the name was wrong ("central core", never "central Tucker core"; N15).
The next line is unchanged.

## The inventory entries re-recorded (`dev/store_rename_slices/diagrams_inventory.md`)

Seven blocks. No line was added or removed in any file, so every `new span` is unchanged. In each, the `new text`
now equals the file at its span. Each `roles` line gains a "hand fix 2026-11-15" clause naming the row and the change.

| block | span (unchanged) | new text | `re-laid out` |
|---|---|---|---|
| DOC-FV-0080 | `docs/frame_variations.md` 116–121 | the up row: "unfolding" → "matricization" | n/a (markdown table) |
| FVF-0071 | 72–75 | left/right `ni` → `nDi` | y, K10b (unchanged) |
| FVF-0077 | 78–79 | `rRi` → `rR(i+1)` | **n → y**: K10 → K10b |
| FVF-0088 | 89–97 | `Dd` → `D(d-1)` (whole block) | **n → y**: K10 → K10b |
| FVF-0100 | 101–109 | `Dd` (49) → `D(d-1)` (48) (whole block) | **n → y**: K10 → K10b |
| FVF-0354 | 355–358 | "up/outer U_i" → "up U_i" | y, K10b (unchanged) |
| FVF-0784 | 785–786 | `rRi` → `rR(i+1)` | **n → y**: K10 → K10b |

**Why four flags flipped.** K10 checks an in-place rewrite against `v2026.4.0`: the same line lengths, and only label
characters differ. The longer labels (`D(d-1)`, `rR(i+1)`) change a line's length, so K10 cannot hold for them:
`line 2: length 74 -> 78`, `line 6: 50 -> 54`, `line 6: 51 -> 54`, `line 2: 89 -> 93`. K10b is the gate for a block
whose labels change length: every old alignment column must map to one consistent new column. All four pass it. Each
flag records the reason, e.g. `y (R4a: in place, K10; hand fix 2026-11-15: ``Dd`` → ``D(d-1)`` lengthens the line, so
K10b)`.

## Gates

`dev/store_rename_diagram_check.py all`, exit 0:

```
self-test: PASS
K10/K10b: PASS (33 in place checked, 27 re-laid out checked, 0 blocks not yet rewritten)
K11: PASS (94 fenced spans located by text in .; 206 rewritten by hand, excepted and checked by K12)
K12: PASS (301 of 301 blocks filled)
```

Before the re-record, K12 failed on exactly the seven blocks and nothing else. HEAD's counts were 37 in place / 23
re-laid out; the four flipped blocks account for the difference.

`dev/store_rename_check.py --stage R6 --hits dev/store_rename_slices --prev HEAD`: exit 0, every gate green. The output
is identical to HEAD's except the K10/K10b counts line. That includes K9, K9-converse and K9-registered, K9-docs ×3,
K11, K12 301/301, K13, K14, K15-cross and K17 ×4. K15 reports 0 NAME hunks; K16 reports its 21.

## Verification

**Behaviour is bitwise unchanged by construction.** For each touched `.py` file, the AST parsed at HEAD equals the
working tree's once docstrings are removed:
- `apply.py`, `sampling_derivatives.py` and `fv_operations.py` are identical even with docstrings (comments only);
- `frame_variations_format.py` and `uniform_fitting.py` differ only in docstrings.

No code line changed, so no seeded A/B run was needed. The doctest comment at `:241` changes no output.

| check | result |
|---|---|
| module doctests of the five touched modules, `pytest --doctest-modules`, current env (py3.11, numpy 2.x, jax 0.10) / floor env (`tttt`: py3.9, numpy 1.22, jax 0.4.30) | 24 passed / 24 passed |
| `fitting_and_optimization.md` (108 examples), `frame_variations.md` (no examples), `aligned_padded_frame.md`, `getting_started.rst`, both envs | pass |
| CI's whole docs set (`docs/*.md docs/contributor/*.md` but `doctest_style.md`), both envs | pass |
| `python -m sphinx -W --keep-going -b html docs <fresh dir>` | exit 0, "build succeeded", no warning (988 s) |
| `git diff --check` | clean |
| full suite, current env, `-n 3`, run alone | 1,407 passed, 5 skipped, 49,058 subtests passed (24 min); the one warning is the oracle sweep's intended adam-on-manifold `RuntimeWarning`. The counts match the docs pass's run. |

Heavy jobs ran one at a time (sphinx, then the suite).

## Extension: eleven further sites (the coordinator's closed list, 2026-11-15)

After the first report, the coordinator extended the closed list by the eleven further sites found above: the same error class, these eleven and nothing else. Each was verified against its surrounding code, and none has a different convention. Each was edited by hand (Edit tool, with its function as context). None lies in an inventoried block: `sampling_derivatives.py`, `entries.py`, `sharing.py` and `ufv_conversions.py` have no fences, and `frame_variations_format.py:1457` is outside FVF's blocks. So nothing was re-recorded.

### 18. `backend/sampling_derivatives.py:1107`, `tv_apply_derivatives`

```
BEFORE:             typ.Sequence[NDArray],          # var_tt_cores     H.  len=d, elm_shape=K+C+(rLi,nDi,rRi)
AFTER:              typ.Sequence[NDArray],          # var_tt_cores     H.  len=d, elm_shape=K+C+(rLi,nDi,rR(i+1))
```

Evidence: It returns `tv_apply_jacobian_derivatives_from_sweep(variation, ...)` (`:1133`, fix 11), which hands `var_tt_cores` to `_apply_derivatives_from_jets` (signature `:1052`, `K+C+(rLi,nDi,rR(i+1))`).

### 19. `backend/sampling_derivatives.py:1161`, `tv_entries_jacobian_derivatives_from_sweep`

```
BEFORE:             typ.Sequence[NDArray],          # var_tt_cores     H.  len=d, elm_shape=K+C+(rLi,nDi,rRi)
AFTER:              typ.Sequence[NDArray],          # var_tt_cores     H.  len=d, elm_shape=K+C+(rLi,nDi,rR(i+1))
```

Evidence: `var_tt_cores` goes to `_apply_derivatives_from_jets` (`:1182`; signature `:1052`).

### 20. `backend/sampling_derivatives.py:1191`, `tv_entries_derivatives`

```
BEFORE:             typ.Sequence[NDArray],          # var_tt_cores     H.  len=d, elm_shape=K+C+(rLi,nDi,rRi)
AFTER:              typ.Sequence[NDArray],          # var_tt_cores     H.  len=d, elm_shape=K+C+(rLi,nDi,rR(i+1))
```

Evidence: It returns `tv_entries_jacobian_derivatives_from_sweep(variation, ...)` (`:1214`, item 19).

### 21. `backend/sampling_derivatives.py:1610`, `_assemble_tt_variation_jets_dense`, the `H_tildes` return

```
BEFORE: ) -> typ.Tuple[NDArray, ...]:                   # H_tildes. len=d, elm_shape=[W+]K+C+(rLi,nDi,rRi)
AFTER:  ) -> typ.Tuple[NDArray, ...]:                   # H_tildes. len=d, elm_shape=[W+]K+C+(rLi,nDi,rR(i+1))
```

Evidence: The per-core step `_tt_variation_jets_step` (`:1592–1593`) forms `H~` as mu (x) xi (x) sigma~ + tau~ (x) xi (x) nu (+ the deta~ group, which also ends on `nu`). Its `b` leg comes from `sigma~` or `nu`, both `(rR(i+1),)` in this signature (`:1602`, `:1607`).

### 22. `backend/sampling_derivatives.py:1742`, `assemble_tt_variation_jets`, the `H_tildes` return

```
BEFORE: ) -> NDArray:                                        # H_tildes supercore, [W+]K+C+(rLi,nDi,rRi)
AFTER:  ) -> NDArray:                                        # H_tildes supercore, [W+]K+C+(rLi,nDi,rR(i+1))
```

Evidence: The W-chunked wrapper of item 21: the same kernel, so the same shape.

### 23. `backend/sampling_derivatives.py:2228`, `_apply_derivatives_transpose_from_jets`, the `H_tildes` return

```
BEFORE:     typ.Tuple[NDArray, ...],    # H_tildes. len=d, elm_shape=[W+]K+C+(rLi,nDi,rRi)
AFTER:      typ.Tuple[NDArray, ...],    # H_tildes. len=d, elm_shape=[W+]K+C+(rLi,nDi,rR(i+1))
```

Evidence: Its docstring and body build `H = mu (x) xi (x) sigma_hat`, and `compute_sigma_hat_jets` returns `(rR(i+1),)` (`:2123`).

### 24. `backend/entries.py:197`, `tv_entries_jacobian_from_sweep`

```
BEFORE:             typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nDi,rRi)
AFTER:              typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nDi,rR(i+1))
```

Evidence: `var_tt_cores` goes to `apply._apply_from_xis` (`entries.py:216`), the kernel of fixes 7–8.

### 25. `backend/sharing.py:577`, `fv_share_tucker_variations`

```
BEFORE:             typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=(K+C)+(rLi, nDi, rRi)
AFTER:              typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=(K+C)+(rLi, nDi, rR(i+1))
```

Evidence: `T3Variations.data`'s TT variations, passed through untouched. The module itself writes `K+C+(rLi, nDi, rR(i+1))` at `:849`.

### 26. `backend/sharing.py:711`, `fv_tied_ambient_directions`

```
BEFORE:             typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=(K+C)+(rLi, nDi, rRi)
AFTER:              typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=(K+C)+(rLi, nDi, rR(i+1))
```

Evidence: As item 25.

### 27. `frame_variations_format.py:1457`, `T3FrameWeights.right_weights` field comment

```
BEFORE:     right_weights: typ.Tuple[NDArray, ...]  # len=d, elm_shape=stack_shape+(rRi,)
AFTER:      right_weights: typ.Tuple[NDArray, ...]  # len=d, elm_shape=stack_shape+(rR(i+1),)
```

Evidence: The builder `from_t3weights` → `fv_operations.t3weights_to_t3frameweights` sets `right = tuple(tt_weights[1:])` (`fv_operations.py:735`), which is TT bond `i+1`. Its docstring says "``H_i``'s right bond is bond ``i+1``". `fv_weights_consistent` (`:693`) requires `right[i].shape == stack + (H_i.shape[-1],)`. The class docstring puts `right` on `H`'s `rR` leg (`:1441`).

### 28. `backend/ufv_conversions.py:363`, `t3frameweights_to_ut3frameweights`, the ragged `right_weights` input

```
BEFORE:             typ.Sequence[NDArray],  # right_weights, len=d, elm_shape=C+(rRi,)
AFTER:              typ.Sequence[NDArray],  # right_weights, len=d, elm_shape=C+(rR(i+1),)
```

Evidence: A ragged `T3FrameWeights.data`, so as item 27.

### Verification of the eleven

- **AST against HEAD:** `sampling_derivatives.py`, `entries.py`, `sharing.py` and `ufv_conversions.py` are identical even with docstrings, so the change is comments only. `frame_variations_format.py` still differs only in docstrings (`:1457` is a comment). All eight touched `.py` files have identical code ASTs. **No suite run was needed**; the full suite above ran on the first seventeen and is unaffected.
- **`store_rename_check.py --stage R6 --hits dev/store_rename_slices --prev HEAD`:** exit 0. It is identical to HEAD's output except the K10/K10b counts (33 / 27, as before).
- **`store_rename_diagram_check.py all`:** PASS (self-test; K10/K10b; K11, 94 spans; K12 301/301).
- **Module doctests** of `frame_variations_format.py`, `sampling_derivatives.py`, `entries.py`, `sharing.py` and `ufv_conversions.py`: 33 passed (current env) / 33 passed (floor env).
- **`git diff --check`:** clean.
- **`sphinx -W --keep-going -b html docs <fresh dir>`:** exit 0, "build succeeded", no warning (989 s, run alone).

## Item 29: the reviewer's addition

### 29. `backend/sharing.py:814`, `fv_share_tucker_variations_corewise`

The reviewer asked for this one in its first pass. It extends the coordinator's closed list by one site. Only the
right bond changes.

```
BEFORE:             typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=(K+C)+(rLi, ni, rRi)
AFTER:              typ.Sequence[NDArray],  # tt_variations.     len=d, elm_shape=(K+C)+(rLi, ni, rR(i+1))
```

Evidence:
- The array is the corewise geometry's raw TT-core perturbation `dG_i`. Its frame is `t3_corewise_frame` =
  `(B, G, G, G)`, so `nD = nU = n` and `rL = rR = r`, and its shape is `(r_i, n_i, r_(i+1))`.
- `ni` stays. It matches the Tucker line above it, `:813` `(K+C)+(ni, Ni)`, which is correct. It also matches the
  library's other corewise-direction comments, `apply.py:344–345` and `entries.py:275–276` (`(ri,ni,r(i+1))`).
- The mixed form `(rLi, ni, rR(i+1))` already appears at `tt_operations.py:33` and `t3_orthogonalization.py:103`.
- The site is not in an inventoried block (`sharing.py` has no fences), so nothing was re-recorded.

Checks after item 29:
- `sharing.py` has a full AST identical to HEAD (comments only);
- `store_rename_check.py --stage R6`: exit 0, output identical to the previous run;
- `store_rename_diagram_check.py all`: PASS (K12 301/301);
- `pytest --doctest-modules t3toolbox/backend/sharing.py`: 13 passed / 13 passed (both envs);
- `git diff --check`: clean.
No sphinx re-run (a comment inside a signature), and no suite run (comments only).

## Still not changed

Nothing of this error class is left. The reviewer's completeness sweep over t3toolbox, docs, tests, examples and
README agrees. These are correct under their own conventions:
- The sigma carries `W+K+C+(rRi,)`: the carry entering step `i` is bond `i`.
- `fv_operations.py:502`, `ZZ`: bond-indexed, len `d+1`.
- `backend/t3_linalg.py:156` `KS + (S, rLi, ni, rRi)`: the names are bound on the next line by
  `rLi, ni, rRi = G_KSsaib.shape[-3:]`, a T3 core's own two bonds.
- The per-core T3 notation `rL_i*rR_i`, defined at `t3svd_minimal_ranks.md:21`.
- The math notes: 1-based, deferred to their rebuilds.

`docs/autoapi/` still shows the old text; it is generated and gitignored.

My search for the extension was `grep -rnE "rLi, *n[DUi]*i?, *rRi\)|\(rRi,\)|rL_i, *n_?[DU]?_?i, *rR_i\)"` over
`t3toolbox docs tests examples README.md`.
