# P16 design review: the implementer's note on the switch

Reviewer (the fresh pair), 2026-11-21, on `P16_DESIGN_NOTE.md`, read only, at `fa38569a`. I read it against the plan's
P16 and 2.2 to 2.5, the settled notes of 2026-11-21 (with Claude's correction of item 2: the regularizer is not
scaled), my orientation risk list, and the code it changes.

**Overall:** the design is right. It has the four parts, one query per model method with a literal `want`, centers on
every frame the library builds, and `held` per frame and per batch. Each mathematical choice I could check, I checked,
and it holds. Sections 1 to 5 below are recommendations; section 6 lists what the tests should add. Four points need
the note changed before part 2 is built:
- D5's placement (§1);
- D3's misuse case (§1);
- the alignment projector's geometry must be taken at each frame (§2);
- the comparisons against the pre-P16 library (§3).

## Two facts checked numerically (scratch, `reviewer2/`)

1. **Every frame of the alignment loop has the base point `x0`.** At every call of the projector in
   `aligned_padded_frame`, including the oversampled working widths (`(5, 6, 5, 4)` against the target
   `(3, 4, 3, 3)`), `X(D, L)` equals `x0` to 1.3e-15 when padding from an old point, and is exactly 0 at the cone.
   - So the note's choice is right: recompute the residual at each pass's frame through `query`, in place of today's
     fixed residual. It gives the same projection to rounding.
   - Holding today's `r` across the passes would also be right, but it bends the one-frame contract of `held` for no
     gain. Recomputing is the better choice.
   - (`align_basepoint_check.py`)
2. **The zero-padded old centers are consistent with the aligned padded frame.** With 0, 1, 3 and the default
   number of passes, ragged:
   - the centers' residual is about 5e-15;
   - the probe `z` through `Vc` equals `z` through `D` to 1e-15, and `x0.probe` to 6e-16.

   The zero-point frame and both cone frames have a base point of exactly 0, so a `g` query there gives `J^T(-y)`
   exactly. (`padded_centers_check.py`, `cone_basepoint_check.py`)

## 1. The eight decisions

**D1. A required Geometry member for the point as the engine reads it. Agree; the maintainer should decide its name
and that it is required.**
- **Why a member.** On the uniform layout the optimizer state is the bare pair, and the engine needs
  `(B_sc, G_sc, shape, masks)`. Only the geometry holds `shape` and the point's masks, and `backend/optimizers.py`
  must stay T3-free. A member is the only clean home.
- **It must be required.** Every trial point needs it.
- **The masks must be the geometry's held plain masks** (the point's), not the frame's. Test it with garbage in the
  trial supercores' padding.
- **The name `point(x_cores)` is a trap beside `base_point(frame)`.** That one returns the bare pair, the opposite
  form, and a reader will take the two for one concept. A name that says what it is for is clearer:
  `corewise_point(x_cores)` ("the point as an oracle reads it, a corewise frame"). On a corewise geometry it equals
  `frame(x)[0]`, which the docstring can say.
- **It changes the Geometry protocol.** That is the extension surface of custom geometries, and P17's
  `point_value_and_gradient` will lean on it, so it is the maintainer's call.

**D2. `padded_start(geom, oracle, batch, x0, x_old, <options>, use_jit=False) -> (frame, centers, movements)`. Agree;
Claude's call, logged.**
- The change is forced, since `Problem` goes, and it follows the plan's head `(geom, oracle, batch, x0)`.
- A flat return beside `frame(x)`'s pair is simpler than a nested one.
- No regularizer, rightly: the alignment hint is the misfit gradient, as today.
- The geometry member `aligned_padded_frame` changes its return too, so update its text in the `Geometry` protocol.
  It is an optional member a custom geometry may define.

**D3. `local_model(..., frame=None, centers=None)`, backend only. Agree with three conditions; Claude's call.**
- **Centers without a frame raise** (structural). Centers come only with the frame they belong to.
- **Wrong centers are a silent wrong answer.** The build's `f` and every value read off `z` would be wrong. The
  backend checks nothing, by the rule, so the docstring must say it and point to `fv_centers_consistency_residual`
  and its uniform twin.
- **The frontend `local_model` gets no `centers=`** (O-4: no frontend user supplies centers).
- **Test:** at `padded_from`'s start, the model's `f` (read through the padded centers) equals `point_objective(x0)`
  to rounding. That kills a mutant that passes `x0`'s own centers at the aligned frame.

**D4. `format_newton_iter(..., order=None)`. Agree; Claude's call.** It is the smallest form that tells modes from orders:
- the leading `ndim - len(order)` axes are modes;
- `make_newton_display` fills it from `oracle.order`;
- a position label is used otherwise.

Keep it explicit. Reading `info.lm.oracle.order` inside the pure formatter would break hand-built rows (`lm=None`).

**D5. The uniform minimal-rank guard in `UniformManifoldGeometryOps.from_point`. Agree only if the frontend
`local_model` does not reach it; otherwise move the guard.**
- **Where today's guard sits.** It is in `uniform_least_squares_problem`, on the optimizers' path, and the frontend
  optimizers reduce first anyway (`uniform_minimal`).
- **Today's frontend model allows a non-minimal point.** The uniform factories build the model at a non-minimal
  uniform point, and their `_bgeom` uses `from_frame`, not `from_point`. The note says the new frontend
  `local_model` "keeps the user's point".
- **What D5 then requires:**
  - the frontend `local_model` builds its backend geometry with `from_frame`, not `from_point`;
  - a test keeps `gradient`, `hvp` and `gn_quadratic` working at a non-minimal uniform point;
  - say in the docstring that a non-minimal geometry's `retract` is where the masks desync.
- **The check uses the shared notion of minimality when `sharing` is given**, as `uniform_minimal` does.
- **If the frontend model cannot avoid `from_point`,** then the guard belongs at the start of a retraction loop
  instead. That would be an optional, duck-typed geometry member (say `require_start(x_cores)`) called by the four
  backend optimizers. It keeps `optimizers.py` T3-free and constructors permissive.

**D6. The frontend `LocalModel` as `(geometry, frame, backend)`. Agree in substance; store the frame once.**
- **The frame is stored twice.** `frame` and `backend.frame` are the same arrays in two forms (corewise: the
  frontend `T3Frame(B, G, G, G)`, the backend `(B, G)`). As jax leaves they are traced twice. The project's rule is
  "prefer minimal dataclasses, derive rather than store".
- **Prefer fields `(geometry, backend)`, with the frontend frame derived.** A cached property can wrap the backend
  frame. Corewise: rebuild `T3Frame(B, G, G, G)`. Uniform: `UT3Frame` from the frame data.
- **If both are kept,** `local_model` must be the only place that builds them, and a test asserts they agree.
- **The pytree must keep compiling once across bases** (CLAUDE.md), with the frame a leaf. Test it.

**D7. Top-level exports `local_model`, `LeastSquaresOracle`, `LeastSquaresBatch`. Agree; the maintainer's.** The
top-level optimizers now need the oracle and the batch to be callable, the README shows a top-level fit, and the six
factories leave from the same place. But `t3t.local_model` is a generic name at the top level, and the namespace is
what users see. Raise the three names together with D1's name.

**D8. The timing-lap rename and the floor in part 2. Agree; Claude's call.** Part 2's build and loop produce both, and
part 2's tests cannot judge the loop without them.

## 2. The stated choices

**The alignment projector, a factory memoized on `(geom, oracle)`. Agree, with one condition.**
- Both arguments are value-hashed statics, so this is scan-body rule 4.
- The batch rides as the hint, a pytree argument of the uniform whole-loop jit.
- **The condition.** The loop's frames change width (oversampled, then target). Inside, the projector must keep
  calling `geom.with_frame(frame)`, as `_misfit_gradient_at` does today. The memoized `geom` holds the variation
  masks of one width; projecting a frame of another width with those masks is a uniform padding bug.
- **Test:** the uniform aligned frame against the ragged one on real parts, with oversampling on.
- A user oracle's batch must be a pytree under `use_jit`; say so.

**The uniform corewise retraction through `ut3_corewise_frame`. Acceptable.**
- It is the same arithmetic. `utv_corewise_retract` reads `down_mask` and `frame_left_mask` of the doubled masks,
  which are the point's.
- It keeps one more caller of `ut3_corewise_frame` alive into the P20 check-in. A direct additive retraction on the
  point (broadcast over `K`, the point's masks) is about ten lines and would free it. Implementer's choice; if kept,
  list it for P19 as the note says.

**The translation as a decorator on raw arguments. Agree.** The collision of an old `batch=20` with the new
positional `batch` is a real case, and it is caught only before binding.

**A custom old draw scaled `n / min(batch, n)`. Agree.** The old step objective was `misfit_B + (min(b, n)/n) rho`. The
new one is `s misfit_B + rho`, which is that times `n/min(b, n)`. The Cauchy step is invariant under scaling the
objective, so MC-SGD is reproduced; Adam is reproduced up to its `eps`. But the wrapped draw must do more than scale.
For each draw it must also:
- turn `(ww, pp)` into `affine_jet(ww, pp)` (and an entries-derivative draw into apply at
  `affine_jet(index_to_one_hot(index, shape), pp)`);
- convert the drawn derivative data to Taylor coefficients, slot `t` divided by `t!`, as `prepare_derivative_data`
  does for the full data;
- pack the result for a uniform `x0`.

The weight is converted once. Forgetting the per-draw data conversion gives an order-dependent wrong fit with no
error. Test: an old-form `mc_sgd` with a derivative kind, a custom draw and a regularizer against the pre-P16
library (§3).

**Packing only a `LeastSquaresBatch`. Agree.** Another oracle's batch is opaque, so packing it is the oracle's business.
- `check_batch` runs on the ragged form, before packing.
- A user who hands the frontend an already packed batch would fail `check_sample`. Either say "the frontend takes
  the ragged batch", or skip the check when `pack_batch` would keep the batch as it is. Small; choose one and
  document it.

## 3. What the note's test plan lacks

1. **Comparisons against the pre-P16 library (endorsed by Claude).** The old optimizers leave the tree in P16, so the
   suite can compare only new against new. Two things to do:
   - The handover should carry a side-by-side script. It runs the old library (`git archive 0acf7868 t3toolbox`, as in
     `reviewer2/old_lib`) and the new one on the same problems and seeds. It compares `history`, `stop_reason`, the
     step counts and the final point, to rounding. Cases: Gauss-Newton, exact and switch; line search and trust
     region; the zero, cone and warm references; `padded_from`; regularized; ragged and uniform. Plus `mc_sgd` with
     the flat draw and a custom draw step for step (the same rng calls: `rng.choice(n, size=min(b, n),
     replace=False)`), and the translation of every old weight form (including a bare vector on a derivative kind
     with `d == n + 1`, which the old rule reads per order).
   - Pin a few of those numbers in the translation test, with a tolerance.
   - I will run my own copy at review.
2. **The structural efficiency test (endorsed by Claude).** At a manifold frame with centers, take `computed_names`
   of the `hvp` query given `model.held`, in Gauss-Newton mode. It contains no plain sweep: no `xi`, `mu`, `nu`, `eta`,
   `z` or `r`. The same holds for `gradient` and `gn_quadratic`. The note's §2 states the contract; the tests should
   pin it, for probe, apply and entries, ragged and uniform, and on a corewise point.
3. **The model's `f` against `point_objective` at the build point**, at `frame(x)`, at a given frame, and at
   `padded_from`'s start: two routes, one value (it also covers D3).
4. **A `start_frame` with no centers:** the model is right (the `D` route). Iteration 1 then has a different set of
   held slots, so one extra compile under `use_jit`. That is acceptable; say it rather than claim "one cache key".
5. **`_gauss_newton_view`'s docstring.** It says its pytree is a Gauss-Newton model's. Now the exact build also
   holds `nu` and `eta` (it wants `g`), so its set of held slots, and so its treedef, differs from a Gauss-Newton
   model built without centers. Make the claim true or drop it.

## 4. Smaller points

- **The floor's `eps`.** Read it off the dtype of the model's `f`, and add `eps |rho|` with the same `eps`. Test
  float32 on numpy and jax, as the note plans.
- **"Reported objective values change by the factor."** That is only the per-step minibatch objective. The full-batch
  losses MC-SGD and Adam record are unscaled and unchanged; say so in the CHANGELOG note.
- **Exact mode on a manifold frame** recomputes `nu_L` and its tilde at every `Kp` query (they are not in Part 1's hold
  list), as today's two-hole re-sweeps. That is parity, not a regression; worth one line in "Carried to Part 2".
- **The display's denominators:** querying `y_block_sumsq` once at the first row's frame is right (it reads only the
  frame's static shape). A callback reused across continuation levels keeps the same shape, so that is fine too.

## Summary

| # | Decision | Recommendation | Whose |
| --- | --- | --- | --- |
| D1 | a Geometry member for the engine's point form | agree, required; name it so it is not confused with `base_point` (e.g. `corewise_point`) | maintainer (protocol, name) |
| D2 | `padded_start(geom, oracle, batch, x0, x_old, ...) -> (frame, centers, movements)` | agree; update the member's protocol text | Claude, logged |
| D3 | `local_model(..., centers=None)`, backend only | agree; centers without a frame raise; document the silent case; test against `point_objective` | Claude |
| D4 | `format_newton_iter(..., order=None)` | agree | Claude |
| D5 | the minimal guard in `from_point` | only if the frontend model uses `from_frame` and a non-minimal uniform model stays tested; else a duck-typed start member called by the optimizers | Claude, after the implementer checks the frontend path |
| D6 | the frontend model `(geometry, frame, backend)` | agree; prefer `(geometry, backend)` with the frame derived | Claude |
| D7 | top-level exports | agree | maintainer (namespace), with D1 |
| D8 | lap rename and floor in part 2 | agree | Claude |
