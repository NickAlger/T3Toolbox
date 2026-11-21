SIGN OFF: the three round-1 findings are fixed and tested, my garbage case is now exact, and my `eps` mutant is killed.

# P14 review, round 2

Reviewer, 2026-11-21, on `P14_RESPONSE_r1.md` and the updated tree (uncommitted, on top of `3769bad4`). Everything was
run single-process. The mutant copy has been deleted, and the working tree is untouched apart from this file.

## Finding 1, the packed probe `y`'s padding: fixed

`query` now unpacks a packed probe `y` with the frame's static shape before `data_block_norms`:
- the shape is `frame[4]` on a frame, `frame[2]` on a point;
- the unpack is traceable;
- it reads the same real part the engine's mask keeps.

**My round-1 case, re-run.** A uniform frame, a `pack_batch`-ed probe batch, and garbage (1e3) in `y`'s mode padding:
- `y_block_sumsq`: relative error 0, down from 8.7e5;
- `f_resolution`: 0, down from 14;
- `f`: 0;
- `r_block_sumsq`: 3.3e-16.

**The uniform point form**, `(B, G, shape, masks)` from `t3_to_ut3`, with garbage `y` padding: the same three outputs
are exact (0).

**The suite's test.** `test_the_uniform_layout_after_pack_batch` now puts garbage in the mode padding of every packed
form: the probe and apply samples and the probe `y`. Reverting the fix fails 3 of its subtests.

## Finding 2, `eps` in float32: fixed

`TestTheResolutionDtype` checks float32 inputs: `f` is float32, and `f_resolution` uses `finfo(float32).eps`. My
mutant fixing `eps` to float64 is now killed (`test_float32_inputs_give_float32_eps`).

## Finding 3, nothing calls `check_batch` anywhere in the backend: fixed

The P3 test now asserts, for every backend module, that no call is `check_batch` (as an attribute or a name). The
implementer's planted call in `optimizers.py` fails it.

## Runs

- `p14_check.py`, every part, current env: the unbiasedness enumeration, `held` reuse and `f_resolution` against today
  (worst 2.5e-16), and the garbage case (exact).
- `test_least_squares_oracle` and `test_jets`: 59 passed, 534 subtests.
- The coordinator's report: the full gate passes (1,852).
