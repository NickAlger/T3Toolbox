# P16 design note: the switch, first part

Implementer (the fresh pair), 2026-11-21, against `p16-wip` at `fa38569a`. Read only: no code changed.

**What P16 is for.** After P16 the optimizers no longer know what they fit: a model is the geometry, an oracle, a
batch and a regularizer at one frame, and everything sample-shaped goes through `oracle.query`. The library and its
tests move; the examples and pages keep the old frontend form through a temporary translation (P17 removes it).

## 1. The four parts, in order

The suite may be red between parts. Each handover lists the tests expected red, by id and reason, so the reviewer
can tell an expected failure from a regression, and runs the part's own files.

**Part 1, geometry** (`backend/geometry.py`, `backend/optimizers.py`'s protocol text).
- Both corewise geometries: `frame(x)` returns `((B, G), None)` (uniform: `(B_sc, G_sc, shape, masks)`, the UT3
  `.data` the engine reads as a point); `base_point`, `point_tangent`, `retract` and the shared post-pass follow. The
  uniform additive retraction keeps calling `utv_corewise_retract` through `ut3_corewise_frame` (a relabelling, no
  cost); that use goes on P19's list for the P20 check-in.
- `aligned_padded_frame` returns `(frame, centers, movements[, seconds])` (decision D2): the old point's centers
  zero-padded on every rank axis, zeros at the cone start. Ragged and uniform; on uniform the old centers are
  embedded at the prefix slots of the level widths.
- Checks: none new; `require_frame_structure` stays (structural).
- Tests (`test_geometry`, the member parts of `test_padded_start`): the corewise frame is the point and `frame_form`
  reads it as corewise; additive retraction with a `K` stack; shared corewise retraction keeps the tie (aliased
  factors); `fv_centers_consistency_residual` and the uniform twin at rounding at the padded frame, with
  `max_passes=0` and with passes, from a nonzero old point and from the zero tensor; uniform with garbage in the old
  point's padding; the probe `z` through the padded centers against `dense_probe` of `x0`.
- Expected red: every corewise fit through the old `Problem` (the old kinds expect four stores) and the old
  optimizers' padded starts. Judged on the geometry files alone.

**Part 2, backend model and optimizers** (`backend/optimizers.py`, `backend/uniform_fitting.py`,
`backend/regularization.py`).
- `LocalModel` with the fields of plan 2.2; `local_model(geom, oracle, batch, x_cores, regularizer=None,
  hessian=..., curvature_damping=None, frame=None)`: the frame (and centers) from `geom.frame`, the companion, one
  query wanting `('f',)` (exact: `('f', 'g')`) with `Vc` from the centers. Methods as plan P16 item 2, each one query
  with `held` and a literal `want`: `gradient` `('g',)`; `gn_quadratic` `('Jp_norm_sq',)`; `hvp` `('JtJp',)`, exact
  `('JtJp', 'Kp')` with `variations=Πp`, `core_directions=L Πp`; `curvature_*` `('Kp',)`; `jacobian` `('Jp',)`;
  `residual` `('r',)` (free when `r` is held).
- `point_objective` (D1), the five phases as module-level functions, `flat_draw(oracle, batch, batch_size)` over
  `oracle.take`, the four optimizers with the plan's heads; MC-SGD and Adam step on `draw(rng)`'s batch with the
  regularizer unscaled (TARGET_DECISIONS, note of 2026-11-21, item 2, as corrected); policy a trials through
  `point_objective` (never with `held`). The zero reference, the padded start, the cone and the switch through
  `query`. The alignment hint becomes the batch, with the projector a memoized factory on `(geom, oracle)` (it is a
  static argument of the uniform whole-loop jit; `Problem` was a pytree there and is gone). A pass then recomputes the
  forward at its frame: for probe without centers one right sweep more per pass than today; a one-off per level.
- `pack_sample(kind, order, sample, N)`, `pack_data(kind, order, y, N)`; `pack_batch` calls them. `Problem`,
  `least_squares_problem`, `_minibatch_step_problem`, `_ScaledRegularizer`, `uniform_least_squares_problem` go.
- I would move two items of plan part 3 here (D8): the timing laps (`oracle` in place of `sweep` and `residual`) and
  the trust-region floor `f_resolution + eps |rho|`. Both are produced by part 2's build and loop; part 3 keeps
  their display.
- Checks (structural, always, before work): exact or switch mode needs `'Kp'` in `oracle.optional_outputs` and the
  four geometry members; `jacobian` needs `'Jp'`, `residual` needs `'r'`; MC-SGD and Adam without a draw on an oracle
  without `take` raise at entry; stacked point and stacked regularizer raise as today. Nothing numerical.
- Tests: `test_optimizers`, `test_zero_reference`, `test_padded_start`, `test_uniform_fitting`, the model parts of
  both curvature files, `oracle_sweeps/` (backend side), the backend rows of `test_dispatch`; section 3 lists the
  ones that catch silent errors. Expected red: display, frontend, examples, pages.

**Part 3, display and record** (`backend/optimizer_display.py`).
- `make_newton_display(oracle, batch, val_batch=None, print_fn=print, fmt='%.1e', record=True, error_table=True)`.
  Numerators: `r_block_sumsq` from `info.lm.oracle` at `info.lm.frame` on `info.lm.batch` with `info.lm.held`.
  Denominators: `y_block_sumsq` of `batch` and of `val_batch`, queried once at the first row's frame (a packed probe
  `y` needs the frame's static shape). Validation numerators: a query at the model's frame on `val_batch`, never with
  the model's `held`. An oracle without both block sums gets the header and no table.
- Records keep `train_err`, `val_err` in the weight's shape; `error_table=False` drops the printed table only;
  `m >= 2` prints one column per multi-index, row-major (needs D4).
- Tests (`test_optimizer_display`): record shapes at `()` per kind and at `(2,)`, `(1, 1)` for probe and apply, with
  and without validation; the `(1, 1)` table's columns; `error_table=False` against `True` (records equal); the
  header-only oracle; the timing labels; the validation numerator against an independent dense residual. Judged
  with part 2's backend `newton_cg`.

**Part 4, frontend** (`t3toolbox/optimizers.py`, `t3toolbox/fitting.py`, `t3toolbox/__init__.py`).
- The four optimizers take `(geometry, oracle, batch, x0, ...)`; the frontend `local_model` (D6). Both: map the
  geometry, reduce `x0` to minimal ranks (optimizers only; `local_model` keeps the user's point, whose frame the
  user's tangents live at), `check_batch` on `batch` and `val_batch` when the oracle has it, pack a
  `LeastSquaresBatch` for a uniform `x0` (another oracle's batch passes unchanged), and in safe mode check a supplied
  frame: orthonormal, represents `x0`, and `consistency_residual` against `safety.effective_rtol(frame.data)` (the
  default `1e-9` fails a float32 frame). A library-built frame is not checked.
- The translation: a private decorator on the four optimizers. It sees raw `*args, **kwargs` before any binding,
  which matters: an old `mc_sgd(..., rng, batch=20)` would collide with the new positional `batch`. A string in the
  second position (or `kind=`) marks the old form; the rules are plan P16 item 4. An old custom draw is wrapped to
  return a batch with `minibatch_scale = n / min(batch, n)`, which reproduces today's nominal factor. Ragged old
  forms only (none in the tree passes a packed sample; checked again when built).
- The factories, `_check_ragged_sample`, `least_squares_problem`, `LeastSquaresSetup`, `_resolve_chunk_size`,
  `_check_kind` and `chunk_size` go; `_canonical_weight` stays for the translation.
- Tests: the frontend files of plan P16's list, the translation (each old form equals the new), the frontend-only
  fit, the safe-mode frame checks, a bad batch raising before work, the factory doctests of the two pages and the
  five examples. Then the full gate and the slow tier, serially.

## 2. The settled notes, and the two the brief adds

- **`held` for one frame and one batch.** No guard; tests (section 3) kill each place it could leak: MC-SGD and Adam
  across draws, the trial point, the next build, the validation query, the alignment passes. The library's
  `newton_cg` has no draw; "minibatched Newton" is `newton_cg` called per drawn batch, and nothing crosses calls.
- **`eps |rho|` and the minibatch rule.** Floor = `f_resolution` (scaled by the batch, from the oracle) +
  `eps |rho|`, `eps` from the dtype of the model's `f`; without `f_resolution`, `eps (|f| + |rho|)`. It is queried
  only where read (a rejection, a tracked switch row), with `held`. The regularizer is not scaled.
- **Padded centers.** Part 1; consumed in part 2 through `local_model(..., centers=)` (D3).
- **`pack_sample`, `pack_data`.** Part 2.
- **`computed_names` and centers.** The model holds what its build computed and nothing more. For probe that is
  today's full sweep only when the build gets `Vc` (`z` read off `eta`, which needs `mu` and `nu`); without centers
  it holds `xi`, `mu`, `r`, and every Hessian product recomputes `nu` and `eta`. So every frame the library builds
  carries its centers, and a user's `start_frame` costs that on iteration 0 only. Apply and entries hold `xi`, `mu`,
  `r`, as their old lean sweep did.
- **The corewise frame as the point.** Exact mode on it passes `variations=Πp` and `core_directions=lift(Πp)`; the
  lift is the identity and returns the same object, which the oracle requires (else it raises).

## 3. Decisions needed

- **D1, `point_objective` on the uniform layout.** It needs the point as the engine reads it,
  `(B_sc, G_sc, shape, masks)`, and `backend/optimizers.py` (T3-free) cannot build it from the bare pair. Recommend a
  required Geometry member, e.g. `point(x_cores)`, returning that form (ragged: `x_cores`); P17's
  `point_value_and_gradient` needs the same.
- **D2, `padded_start`.** It takes a `Problem`, which goes; the plan gives no new form. Recommend
  `padded_start(geom, oracle, batch, x0, x_old, <options>, use_jit=False) -> (frame, centers, movements)`, and the
  member's return flat likewise.
- **D3, centers with a given frame.** Recommend `local_model(..., frame=None, centers=None)`; `newton_cg` passes
  `padded_from`'s centers internally and gets no new keyword.
- **D4, the table needs the mode axis.** A 1-D block sum is probe at `()` (modes) or apply at `(2,)` (orders).
  Recommend `format_newton_iter(..., order=None)`: the jet shape; the leading `ndim - len(order)` axes are modes.
  `make_newton_display` fills it from `oracle.order` when the oracle has one; otherwise columns are labelled by
  position.
- **D5, the uniform minimal-rank guard.** `uniform_least_squares_problem` raised on a non-minimal uniform start (the
  retraction would desync from the held masks mid-run). Recommend the same structural check in
  `UniformManifoldGeometryOps.from_point`, where the masks are fixed.
- **D6, the frontend `LocalModel`.** Recommend fields `(geometry, frame, backend)`: the frontend geometry, the
  `T3Frame`/`UT3Frame` the tangents live at (corewise still `(B, G, G, G)`: the frontend geometries do not change,
  O-4), and the backend model; today's methods keep their names, plus `residual` and read-only `oracle`, `batch`,
  `held`.
- **D7, top-level names.** The six factories leave `t3toolbox/__init__.py`. Recommend exporting `local_model`,
  `LeastSquaresOracle` and `LeastSquaresBatch` there, which the top-level optimizers now need.
- **D8, the partition.** As in part 2 above; Claude's call.

## 4. Risks, and the tests that catch silent wrong answers

Every comparison is to an independent construction, never the oracle against itself: the dense functions and
`sampling_definitions` at the dense tangent tensor; while they exist, the old kinds' `forward`/`transpose` (a
temporary bridge row, deleted in P20); optimizers by outcomes on known answers.
- **`held` leaking** (the new trap). MC-SGD's iterates equal a reference loop that builds each step from the drawn
  batch afresh; Adam likewise; the trial objective equals `point_objective` with no `held`; the validation table
  equals a dense residual on `val_batch` of the same `W` as the training batch (so a leaked `held` gives a wrong
  number, not a shape error).
- **Same-shaped arguments swapped**: `variations` and `core_directions` in exact `hvp` (test frames with `nU != nD`,
  where a swap is a shape error, and others where it is not, against the dense second derivative); `batch` and
  `val_batch`.
- **A size-1 axis**: the model at `C = (2,)` per element; `gn_quadratic` and `curvature_quadratic` at `K = (3,)`
  with exact output shapes; the scale against `f` of shape `C`.
- **Garbage in uniform padding**: in the packed batch, in `x_cores` for `point_objective`, in the old point's padding
  for the padded centers.
- **A float64 `eps` on a float32 run**: the floor's `eps` read off a float32 `f`, numpy float32 and jax.
- **Compile keys**: a model rebuilt each step is one key; a new weight and a new draw do not recompile (the weight
  is now a batch leaf); `want` literal at every `query` call in the two backend modules (AST).
- **The translation**: each old weight form (probe per mode, derivative per order, the full matrix, extra rows)
  against the exact new weight; old entries derivatives against apply at the one-hot jet.
- **"Equivalent" mutants**: claimed only with a numeric check, as P15 taught.
