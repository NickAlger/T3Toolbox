# P16: the reviewer's handoff for part 4

Reviewer (the fresh pair of P16), 2026-11-21, after parts 1 to 3 were signed off and committed on `p16-wip`. For the
reviewer of part 4: the frontend optimizers, the frontend `local_model` and the translation of the old call form.

## The risk list: what is settled, what part 4 must check

| Risk | Settled in parts 1 to 3 | Part 4 must check |
| --- | --- | --- |
| A direction routed to the wrong argument | `gn_quadratic`, `hvp`, `jacobian` pass `Pi p`; exact `hvp` is one query with `JtJp` and `Kp` | the frontend `LocalModel`'s tangents in and out; the same-frame guard |
| `held` reused at the wrong frame or batch | trials, references and alignment hold nothing; MC-SGD and Adam build each draw afresh; the display's validation and other-batch paths take no `held` | the frontend `local_model` passes no stale `held` |
| The minibatch scale and the regularizer | data scaled by `n/b`, regularizer unscaled (O-21) | **the custom-draw wrapper of the translation** (below) |
| A stop criterion that changes meaning | D9: the acceptance compares point route with point route; the history reports the carried value; the floor is `f_resolution + eps|rho|` | nothing new, unless the frontend adds a check |
| Uniform padding | `point_data` with garbage padding; `pack_batch`; the engine masks on entry | the frontend packing (`pack_batch` for a uniform `x0`) and `uniform_minimal` |
| Silent costs | no plain sweep in `hvp` given `held` (pinned) | the display's fast path (below) |
| The old four-store corewise frame `(B, G, G, G)` | `base_point` and `point_tangent` raise on the wrong form (`_require_frame_form`) | the frontend `LocalModel` (D6) translates `COREWISE`'s `T3Frame` to the backend point; a non-minimal uniform model through `from_frame` (D5) |

## Items carried to part 4

1. **Both or neither on `val_sample` and `val_data`.** When the translation builds `val_batch` from them, it must raise
   as the old frontend did if only one is given.
2. **The display gets the run's batch object.** `make_newton_display(oracle, batch, ...)` reports its errors on
   `batch`. Only when `info.lm.batch is batch` does it reuse the held `r`; otherwise it runs one forward more per row
   (correct, slower). Under `verbose=True` the frontend must pass the batch `newton_cg` actually runs on: under
   `use_jit`, the jax-converted one.
3. **The translation, against the pre-P16 library (not new against new).** The plan's own test compares two runs on the
   new backend. Compare against `old_lib` on the same seeds:
   - bare weight vectors on derivative kinds: per order, including the ambiguous `d == n + 1`;
   - per-mode probe weights, and the full `(mode, order)` matrix;
   - derivatives converted to Taylor coefficients (`prepare_derivative_data`: weight times `t!`), so the objectives are
     equal exactly;
   - `entries_derivatives` as `('apply', (n,))` at `affine_jet(index_to_one_hot(index, shape), pp)`;
   - derivative kinds at order 0 as `(0,)`, with data `(1,) + W ...`;
   - `val_sample`/`val_data` as `val_batch`;
   - `batch=` (an int) as `batch_size`, including the positional collision;
   - `chunk_size` accepted and dropped.
4. **The custom-draw wrapper.** An old `draw(rng) -> (sample, data)` must be wrapped so that each draw is converted in
   full:
   - `affine_jet` (and one-hot for entries derivatives);
   - the derivative data to Taylor coefficients;
   - packed for a uniform `x0`;
   - `minibatch_scale = n / min(batch, n)`.

   Test it **with a regularizer**: without one MC-SGD is scale-invariant and Adam nearly so, so a missing scale is
   invisible.
5. **Page doctests.** The pages still on the old form (`docs/fitting_and_optimization.md`, `sharing.md`,
   `getting_started.rst`) must pass with their pasted output **unchanged** in P16: they are a check of the translation.
   Treat any edit to their outputs as a finding. Most print booleans only, so they are weak; item 3 carries the weight.
6. **The full gate and the slow tier** (`T3TOOLBOX_SLOW_TESTS=1`, serially) at the end of part 4, and the red list
   back to the baseline's.

## Tools in `/tmp/claude-1000/-home-nick-repos-T3Toolbox/352ad37b-5ac8-40a1-a326-a7fc3a369cb2/scratchpad/reviewer2/`

- **`old_lib/t3toolbox`.** The pre-P16 library (`git archive 0acf7868`). Use it with
  `PYTHONPATH=<scratch>/old_lib`, in a separate process from the new tree.
- **`sbs_mine.py`.** My 18 extra side-by-side cases (shared geometries, uniform corewise, order-2 derivatives,
  regularized custom draws), on the harness of `dev/graded_messages/implementation/bench/p16_part2_side_by_side.py`
  (the implementer's 36). Run each, from the scratch folder:
  - `PYTHONPATH=<scratch>/old_lib python <script> old <old.pkl>`
  - `PYTHONPATH=<repo> python <script> new <new.pkl>`
  - `python <script> compare <old.pkl> <new.pkl>`

  Extend the harness with **frontend** calls in the old form (old library) against the translation (new tree):
  that is item 3.
- **`impl_old.pkl`, `impl_r2.pkl`.** The 36 cases, old against built.
- **`p3_check.py`.** Dense checks of the display's errors at `order=(1, 1)`, and a cell read back from the printed table.
- **`p16p1_centers_uniform.py`, `padded_centers_check.py`, `align_basepoint_check.py`.** The centers and base-point
  checks.

**Known benign differences:**
- the default trust-region radius tie (`u_switch_probe_manifold_reg`);
- a regularized shared corewise Gauss-Newton case that amplifies rounding to 1e-4. The old library against itself,
  with `x0` perturbed by 1e-14, does the same.

## How this implementer works

- **Thorough and honest.** Strong test plans, real dense oracles, mutation passes of 12 to 20 mutants, and it reports
  survivors.
- **Predicted lists drift.** Part 1's expected-red list was predicted rather than measured, and was wrong. Insist on a
  measured list from a full-suite run, then diff it yourself. `comm` against your own run takes seconds.
- **Reasoning, check it.** Its reasoning is usually right. Its claims of "restores both" and "equivalent mutant" were
  once each narrower than stated. Check every "equivalent" numerically.
- **Mutants that touch tests.** When it rewrites a test mechanism (offset oracles, stubs), run two or three mutants of
  your own against the rewritten test.
- **One full suite takes 31 to 33 minutes at `-n 2`.** Run it in the background, and never beside another test run.
