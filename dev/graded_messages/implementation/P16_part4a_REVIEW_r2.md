SIGN OFF

# P16 part 4a, review round 2

Reviewer (the third of P16), 2026-11-21. Reviewed: the working tree on `p16-wip` (the 4a fixes on top of `4047ed53`;
the commits since are `dev/` only), against `P16_part4a_RESPONSE_r1.md` and Claude's rulings in `LOG.md`. Read only.

**Verdict.** Every finding of round 1 is resolved as ruled. My scripts now give the intended behaviour. The translation
still matches the pre-P16 library on every case, the one that used to raise included. The consolidation of the batch
handling keeps each path's behaviour, apart from one deliberate change (A3). Two of my mutants survive. Both are tests
that belong with 4b's planned work, not defects of 4a (A6).

## A. What I checked

### A1. The finding scripts, rerun on the fixes

| Script | Finding | Result |
| --- | --- | --- |
| `items23.py` | 2 | A float32 jax frame from `MANIFOLD.frame` (residual 6.6e-7) passes `check_given_frame` and builds the model. |
| `items23.py` | 3 | `local_model(COREWISE, ..., frame=)` raises the frontend's structural message. |
| `shared_frame.py` | 4 | An untied frame on `shared(MANIFOLD, (0, 1, 0))` raises "not tied" (it ran silently before). |
| `b_check.py` | 1 | Through the translation, `weight=[2.]` gives the same history and the same point as `weight=[2., 2., 2.]`. |

### A2. The side-by-side against the pre-P16 library

`fe_sbs.py` (`new_r2.pkl` against `old.pkl`): all 25 old-form cases and the 4 independent new-form cases match to
rounding.
- dx is at most 1.4e-14 (apply at order 0: 3e-10, a flat minimizer, as in round 1).
- Adam differs by 9e-8, the ruled `eps` effect.
- `B_probe_len1`, which raised in round 1, now matches: dx 6.7e-15.

### A3. The consolidation into `prepare_batch(oracle, batch, x, check=True)`, path by path

| Path | Before | Now | Same? |
| --- | --- | --- | --- |
| `_setup` | check (skipped for an ndarray sample), reduce `x0`, pack at the reduced `x0` | check, pack at the given `x0`, then reduce | yes (below) |
| `newton_cg`'s `val_batch` | check unless packed, pack at the reduced `x0` | `prepare_batch` at the given `x0`, before the setup | yes |
| the uniform draw wrapper | pack, no check | `check=False`: pack, no check | yes |
| `local_model` | check (skipped for an ndarray sample), pack | check, pack | yes, but for the change below |

Packing before the reduction is the same as packing after it. `pack_batch` reads only `N`, and the reduction keeps
`N`. I checked this on a non-minimal uniform start (`consol.py`):
- Tucker ranks (3, 3, 3) reduce to (2, 3, 2), with `N` 8 before and after.
- The frontend `newton_cg` with a `val_batch` has the losses of an independent backend twin, built from
  `uniform_minimal` and `pack_batch` at the reduced point.
- The first row's `val_err` equals the dense relative errors on the validation batch, to 7e-16.
- A pre-packed `val_batch` gives the same records, bitwise.

**The deliberate change.** A ragged point given an ndarray sample is now checked, and refused with the checker's own
message: "ww must be a tuple or a list of 3 arrays ...". It used to skip the check (and the engine happened to run it).
This is finding 6's resolution as ruled, and the sample's documented form (`check_sample`), so it is not a finding.

### A4. The other fixes, read in the code

- **The orthonormality check** uses `is_orthogonal(atol=rtol)`, with the consistency check's `effective_rtol`.
- **The tie check** is `has_shared_tucker_factors(sharing, rtol=rtol)` on the frame's `(D, L)` point, in safe mode, ragged
  and uniform. It reaches both `local_model(frame=)` and `newton_cg(start_frame=)` through `check_given_frame`.
- **The corewise `frame=`** raises at the top of `local_model`, before any check and in both modes. The dead branches are
  gone.
- **The shim's `_one_over_the_stack`** indexes the stack axes correctly in both forms:
  - probe, `F + W + C + (Ni,)`, with `lead` covering all but the stack axes and `Ni`;
  - apply and entries, with the stack axes last.

  It compares every element with element 0 by a broadcast.
- **`retract`** passes the backend model's `geom_aux` (`geom.precompute(frame)`).
- **The form items** are done. No trailing whitespace remains in `tests/test_optimizers_frontend.py`, and `_ob` and the
  vacuous spy are gone.

### A5. Runs

- `tests/test_optimizers_frontend.py` and `tests/test_aligned_padded_frame_frontend.py` at `-n 2`: 68 passed, 135
  subtests.
- The module doctests of `fitting.py`, `optimizers.py` and `backend/optimizers.py`: 10 passed.

### A6. My mutants against the fixed suite

Both survive `tests/test_optimizers_frontend.py` (56 passed: 49 + `TestReview4aChecks`' 7). Both concern tests that
4b's plan already contains (P16 item 4's new tests: the translation's tests, and "a bad batch raises in the frontend
before any work"). So I hand them on rather than hold 4a for them.

1. **The translation's per-mode weight replaced by its first entry** (`wm[:1, 0]` broadcast).
   - `test_per_mode_weight_recovers` fits exact data, where the weight does not change the minimizer.
   - The new length-1 test passes for any constant weight.
   - My side-by-side catches it (case A, weight `[1, 2, .5]` on noisy data).
   - For 4b: the translation test should include a per-mode probe weight on noisy data, beside item 9's order 2 or
     more and the regularized custom draw.
2. **`val_batch` packed but not checked** (`check=False`).
   - No test gives a bad validation batch.
   - The code is right: `badval.py` shows a bad `val_batch` raising "ww must have one array per mode" before `_setup`
     runs.
   - For 4b: add a bad `val_batch` case to `test_a_bad_batch_raises_before_any_work`.

## B. Findings

None blocking and none should-fix. The two notes for 4b are in A6.

## Scratch

`/tmp/claude-1000/-home-nick-repos-T3Toolbox/352ad37b-5ac8-40a1-a326-a7fc3a369cb2/scratchpad/reviewer3/`:
- `consol.py` and `badval.py`, new in this round;
- `new_r2.pkl`;
- the round 1 scripts.

The mutation copy is deleted. The working tree is as I found it, apart from this file.
