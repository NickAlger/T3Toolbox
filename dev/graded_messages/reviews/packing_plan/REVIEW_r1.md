NOT YET

# Review of `PACKING_PLAN.md`, draft r1

Reviewer: Claude (critic), 2026-11-24. Branch `graded-messages` at `3375d213`, with the uncommitted `dev/` files of the
working tree (`PACKING_PLAN.md`, `PACKING_PLAN_NOTES.md`, `packing_audit/K_MEASUREMENTS.md`). The library and the tests
are as at `3375d213`, so the plan's statements about the code are judged there.

## 1. What I read and ran

**Read (code).** The engine `backend/sampling_variables.py`: the module docstring, the graph tables, `Rule`, `_closure`,
`_Setting`, `_read_K`, `_read_setting`, `_check_stacks`, `_check_data_space`, `_check_form`, `_run`, `_UNIFORM_MASKS`,
`_SIZE_MASKS`, `_families`, `_masked`, `_mask_variable`, `_uniform_index`, `_closure_facts`, `frame_shape`,
`computed_names`, `frame_form`, and the whole body of `sampling_variables`. `backend/sampling_steps.py`: the plain and
tilde steps through `V_tilde_step`, and every whole step from `_modes` to `start_node`. `backend/jets.py`: the module
docstring, the private helpers, `dual_jet_to_jet`, `data_metric`, `blocks_to_weight`, `prepare_derivative_data`'s body,
`affine_jet`, `index_to_one_hot`, `_require_input_jets`. `backend/data_preparation.py` whole. `backend/fitting.py` from
"The least-squares oracle and its batch" to the end. `backend/uniform_fitting.py` (the packers). `backend/ut3_operations.py`
(`pack_vectors` to `_pack_data_space`). `backend/ut3_sampling.py` (the plain twins), `backend/utv_sampling.py`
(`utv_probe`, `utv_apply`, `utv_entries`, `__all__`, `_mask_once`'s users). `backend/optimizer_display.py` (its reads of
the weight and the block shape). `backend/optimizers.py` (`flat_draw`, the draw checks). `backend/common.py`
(`get_backend`, `jax_map`, `tree_contains_jax`, `tree_to_jax`). Frontends: `tucker_tensor_train.py` (`_as_index`,
`_sample_shape`, `entries`, `apply`, `probe`, `probe_ambient_transpose`, `probe_corewise_transpose`,
`apply_ambient_transpose`, `entries_ambient_transpose`; pytree registration), `uniform_tucker_tensor_train.py` (the class
up to `unscale_coordinates`, every sampling method, the pytree registration), `uniform_manifold.py` (the sampling
methods and transposes), `fitting.py` (`rescale_coordinates`, `_canonical_weight`, `LocalModel`'s docstring,
`_is_packed`, `prepare_batch`, `local_model`, the pytree registration), `optimizers.py` (`_setup`, `_frontend_draw`, the
whole translation `_old_fit` / `_old_draw` / `_translate` / `_translates`), `manifold.py`'s registrations.

**Read (documents).** `CLAUDE.md`; the plan, every line; `PACKING_PLAN_NOTES.md` whole; `PACKING_REFACTOR.md` whole;
`PACKING_INVESTIGATION.md` §7 outline and §9.8; `DATA_METRIC_PLAN.md` §2 to §4; `M_ARC_DECISIONS.md` whole;
`T3POLYNOMIAL_JET_USE.md` §2 to §4; `TARGET_DECISIONS.md` at the cited lines (316-327, 352-357, 495-511, 866-893);
`PART1_PLAN.md` §5.4, P17 to P20, §10; `K_MEASUREMENTS.md` M1 (a) to (f) and (a2), M3's head, M4's totals;
`docs/contributor/uniform_polymorphism.md:38-47`; `docs/uniform_equivalence_contract.md`'s vector-I/O section;
`docs/api_reference.rst:120-140`; `docs/conf.py`'s extension list; `implementation/run_gate.sh`'s header; the grep of
`ORCHESTRATION.md` and `BRIEF.md` for the side-branch and red-suite rules; `reviews/part1_plan/REVIEW_r1.md` for the
form.

**Checked against the code:** every `file:line` of the plan I could resolve (about 40, listed in §4), and the counts of
§5 and of K4.

**Ran** (scripts in `/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/critic/`,
the `t3toolbox` env, `PYTHONPATH` the repo; each a few seconds to a minute; nothing in the repo changed):

| Script | What it tests | Outcome |
| --- | --- | --- |
| `e1_sample_unmasked.py` | §3.2: the engine with the sample's entry mask removed (`_UNIFORM_MASKS['w'] = ()`), junk 1e3 in every padded slot of the packed sample against a zero-padded one; every public name, probe and apply, `order` `()` and `(2,)`, manifold frame and corewise point, `sum_over_samples` both ways, `K = ()` and `(2,)`, with and without `Vc`, both directions given | 1536 calls, **0 differences** (bitwise) |
| `e1b_sanity.py` | that E1's zero is real: the same junk with the exit masks also removed | junk reaches `g`'s padding (6.7e4) and the `'w'` echo (1e3); `z`, `xi`, `f` stay equal |
| `e2_program_size.py`, `e2b_where.py` | §6's gate, traced program: jaxpr equations at `d = 3, 5` | grows on **every** path, also `('xi',)` and apply `z` with no data-space step: 11 `slice` + 11 `squeeze` per mode, from the engine's shape reads on the supercores |
| `e2c_compiled.py` | §6's gate, optimized HLO instruction count at `d = 3, 5, 8` | probe `xi` 63/63/63, `z` 412/412/412, apply `('f','g')` 576/576/576 (constant: the per-site steps pass); probe `f` 438/452/491 and `('f','g')` 1414/1436/1487 (the `_modes` loop) |
| `e3_rescale_packed.py` | (a) the probe rescaling on packed data (what K6's parametric `probe_rescale_coordinates` does); (b) `data_metric` and a per-slot vector at `d = J+1` | (a) the padding of `y / scales` is `nan`, and `f` through the engine is `nan` (ragged: finite); (b) today's `data_metric` refuses `(J+1,)` on a ragged probe `r`; the plan's left-alignment rule reads it as a per-mode weight |
| `e4_nonfinite_padding.py` | "liberal in" with non-finite junk | `inf` in `y`'s padding: `f = nan`; a relative weight `1/abs(y)` on packed `y`: padding `inf`, `f = nan`; `inf` in a held `c`'s padding: 28 `nan` in the **real** `JtJp` Tucker blocks |
| `e5_take_wrapped.py` | the batch readers on a batch whose fields are a stand-in for `PackedVectors` (a frozen dataclass with `.array`, `.shape`) | `n_measurements` returns **5** (true 10; `np.shape(wrapper)` is the widths `(4, 5, 6)`); `take` raises `AttributeError`; `batch.data_metric` raises `TypeError` |
| `e6_twins_cross_layer.py` | whether the twins fix their layer | `probing.t3_probe` on a uniform point runs and returns the packed `(3, 3, 6)` array; `ut3_probe` on a ragged point fails only in its trailing `unpack_vectors`, which K4 removes |

## 2. Findings

### 1. Blocking. K4 cannot leave the frontend green: the uniform frontend does not pack (§7 K4, `:535-536`)

- **What is wrong.** K4 makes the uniform backend refuse ragged input and says "the frontend routes among them are K5's
  and stay green through the frontend's own packing until then (K4 switches the backend; the frontend still packs before
  it calls)". The uniform frontend's sampling methods do not pack. They check a ragged sample and pass it straight to
  the twin, which packs inside the engine (the mirror). The only frontend packing is `prepare_batch`, on the fitting
  path. So K4 turns red every uniform sampling method given a ragged sample, their module doctests (the 13 of K4's own
  list), `docs/sharing.md`, and the tests that call them; K5 is where they would be repaired.
- **Evidence.** `uniform_tucker_tensor_train.py:484-485` (`probe`: `_check_sample`, then
  `ut3_sampling.ut3_probe(self.data, ww, ...)`), and likewise `:447-448`, `:528-530`, `:557-559`, `:575-577`,
  `:593-596`, `:610-612`, `:628-631`; `uniform_manifold.py:581-582`, `:607-608`, `:677-678`, `:695-696`. The packing is
  in the twins: `ut3_sampling.py:107-109`, `:229-232`, `:250`; `utv_sampling.py:99-102`; and in the engine,
  `sampling_variables.py:1867-1873`. The doctest at `uniform_tucker_tensor_train.py:478` passes a ragged tuple.
- **What would resolve it.** (a) Swap the slices: K5 before K4. The frontend packs explicitly (`PackedVectors`, on K2's
  `packing.pack_vectors`) and hands packed arrays to a backend that still accepts both; K4 then removes the mirror and
  breaks only backend tests. This is the plan's own rule "additive before refusal". (b) Merge K4 and K5 into one slice
  in reviewed parts on a side branch. (c) A temporary frontend shim in K4 (the mirror moved up a layer), deleted in K5:
  code written to be deleted. I recommend (a).

### 2. Blocking. The uniform frontend's batch has no type, and the plan's draw workflow does not run (§4.3, §4.1)

- **What is wrong.** `fitting.pack_batch` returns a batch whose sample, probe `y` and full weight are `PackedVectors`.
  The batch class is the backend's `LeastSquaresBatch` (the frontend imports it: "one class each, both tiers"), and the
  oracle is the backend's `LeastSquaresOracle`. Their batch methods are backend code and cannot know a frontend class.
  So on such a batch `oracle.take` raises, `batch.data_metric` raises, `oracle.check_batch` can only refuse it, and
  `oracle.n_measurements` returns a silently wrong count (it reads the shape through `np.shape`, which returns the
  wrapper's `shape` field, the widths). The plan's own workflow for a custom draw, "pack once, draw with take" (§4.3), is
  this call. The plan also says "What is left per draw is the frontend's ordinary hand-over of `.array`", which is a
  per-draw wrapper by another name; it does not say what the user's draw returns or which `take` it calls.
- **Evidence.** E5: `n_measurements` 5 for a true 10; `take` `AttributeError`; `batch.data_metric` `TypeError`.
  `t3toolbox/fitting.py:78`; `backend/fitting.py:1166-1190` (`_w_axes`, `_W`, `take` read `np.ndim`, `np.shape`);
  `backend/optimizers.py:625-640` (`flat_draw`, which a user's draw imitates); `t3toolbox/optimizers.py:117-122`.
- **What would resolve it.** Decide the frontend type of a packed batch, and every operation on it, before K5's
  design. Options: (a) a frontend batch class (a backend batch of bare arrays plus the widths) with `take`,
  `n_measurements`, `data_metric` and `unwrap()`; (b) frontend functions `fitting.take(oracle, batch, indices)` and
  `fitting.n_measurements(...)` that unwrap, call the backend, and rewrap; (c) the packed frontend batch is a backend
  batch of bare arrays, and the widths are checked once in `fitting.pack_batch` and trusted after (the ruled-out
  "widths on the batch" would come back in another form). Say also what `val_batch`, `LocalModel.batch` (today the
  backend batch), and P17's `fitting.point_value_and_gradient` with a drawn batch take.

### 3. Blocking. `data_metric`'s contract cannot be what §2.2 says, and its block weight is checked by prefix only (§2.2, §2.3, §1.4)

- **What is wrong, part 1: the full weight.** §2.2 says `y` and `full_weight` have exactly `z`'s shape, except that a
  stack axis `C` may be 1. `data_metric` is kind-free and is not told the stacks, so it cannot tell a `C` axis (where 1
  is legal) from any other axis (where it is not). And the engine itself calls `data_metric` with the weight carrying
  size-1 `K` axes: `_with_K` inserts them for `Jp_norm_sq` and the adjoint seeded by `Jp`, and `_check_pair` accepts
  them only because today every axis may be 1. So either `data_metric` keeps the old rule ("each axis 1 or full",
  which is the broadcasting the redesign meant to remove, now in the one public metric), or every caller broadcasts the
  full weight to the exact shape first (the engine to `K`; a user with data shared over a stack to `C`). The plan does
  not say which. `_with_K` also stays (for the full weight), although §2.1 lists it among the seams the redesign
  removes.
- **What is wrong, part 2: the block weight.** Left alignment needs no kind to compute, but it needs the kind and the
  order to check. Without them `data_metric` can only check that the block weight's shape is a prefix of `r`'s (or, for
  a tuple `r`, that its first axis is `d` and each row a prefix of `r[i]`). A block weight of the wrong kind or rank
  then passes whenever the sizes coincide, and is applied to other axes, silently: a per-slot vector `F = (n+1,)` meant
  "the same in every mode" against a probe `r` with `d = n+1` is read as a per-mode weight. Today the same call raises
  by rank. This exact coincidence is the one T3Polynomial meets (`d = g+1`, legacy per-order vectors of length `J+1`,
  `J = g`), which its survey records as "a raise, not a misread". A prefix of extra axes (`(d,)+F+W`, a hidden sample
  weight) is accepted by `data_metric` and refused by the engine, so the two disagree on what a block weight is.
- **Evidence.** `sampling_steps.py:926-945` (`_with_K`), `:1006`, `:1023-1024`; `jets.py:559-576` (`_check_pair`),
  `:753-760`; E3(b); `T3POLYNOMIAL_JET_USE.md:135-139`. The plan's own §1.4 reason for one function ("`data_metric` is
  elementwise") was written for the full-form weight; with a block weight it is no longer elementwise.
- **What would resolve it.** Decide `data_metric`'s structural rule, in writing, for both weights. For the full weight:
  exact shape (callers broadcast; the engine with `xnp.broadcast_to`, a view on numpy), or "each axis 1 or full" with
  the exact rule enforced only by the engine and the checkers (and documented as such). For the block weight: (a) give
  `data_metric` the information to check exactly, which means the kind (a split, see challenge C1); (b) keep it
  kind-free and state the prefix rule and its blind spot with a doctest, as today's page does for the mode axis of 1;
  (c) take the block weight out of `data_metric`: it keeps only the elementwise full weight, and the batch method
  `LeastSquaresBatch.data_metric(r)` (which has no kind either) gains it through the oracle, which has. Whatever is
  chosen, the custom-loss workflow of §4.1 computes on `.array` with `data_metric`, so this is the user-facing path.

### 4. Should fix. The division helpers cannot keep zero padding on the packed form (§1.3, §5, K6)

- **What is wrong.** §1.3: a raw function "states zero padding in as its precondition and keeps zero padding (...
  the jet helpers ...)", and §5 makes every probe helper parametric over the packed form. Two of them divide by a
  per-mode quantity whose padding is zero: `probe_rescale_coordinates` (the data divided by the scales) and
  `probe_dual_jet_to_jet` (divided by `block**2 * full**2`, a packed full weight having zero padding). On the packed
  form `0 / 0 = nan` in the padding. The engine's entry mask multiplies, so `nan` survives it and `f` is `nan`. Neither
  function has the widths, so neither can guard the padding (a `where(scale == 0)` would also change the documented
  meaning of a real zero scale).
- **Evidence.** E3(a): padding `[nan nan]`, `f` `nan` uniform against finite ragged. `data_preparation.py:490-492`;
  `jets.py:529-539`; `sampling_variables.py:1918` (mask by `x * total`, `:1461`).
- **What would resolve it.** These two stay ragged-only on the probe side (refusing an array, naming
  `unpack_vectors`), for the reason §12.4 gives `index_to_one_hot`; or they take the widths; or §1.3's sentence names
  them as exceptions. (Challenge C3 asks whether any probe helper needs the packed branch.)

### 5. Should fix. "Liberal in" tolerates only finite junk, and the plan invites the non-finite kind (§1.3, §3.2, §4.1)

- **What is wrong.** Every entry mask is a multiplication. `inf` or `nan` in a padding survives it, and through a
  contraction with a zeroed core column (`0 * inf = nan`) it reaches real entries of every output. The frontend's
  advice is "Compute on `.array`; re-wrap", and the commonest computation on data, a relative weight `1/abs(y)`, puts
  `inf` in the padding of a packed `y`. With the sample no longer masked, the sample's padding is guarded by the zeroed
  `D` columns only, which also neutralize finite junk only (no regression from today, where the multiply keeps `nan`
  too, but the plan's claim is about junk in general).
- **Evidence.** E4: `inf` in `y`'s padding, `f = nan`; `1/abs(y)` packed, `f = nan`; `inf` in a held `c`'s padding, 28
  `nan` in the real `JtJp` Tucker blocks. `sampling_variables.py:1461`, `:1918-1920`.
- **What would resolve it.** (a) Mask the data-space inputs and the held variables by selection (`xnp.where(mask, x,
  0)`), which costs what the multiply costs and neutralizes any junk; or (b) make "finite" a stated precondition
  everywhere the plan says "junk", in `PackedVectors`'s docstring and §1.3, with a doctest of the `1/abs(y)` trap; and
  either way, (c) do not ship packed branches of the division helpers (finding 4).

### 6. Should fix. `PackedVectors.shape` collides with the array protocol (§4.1)

- **What is wrong.** numpy reads an object's `.shape` attribute as its array shape. `np.shape(pv)` and `pv.shape`
  return the widths `(N_0, ..., N_(d-1))`, not the array's shape. Any code (library or user) that asks a
  `PackedVectors` for its shape gets a plausible tuple of the wrong meaning: E5's silent `n_measurements` came from
  exactly this. A user allocating `np.zeros(z.shape)` "like `z`" gets an array of the widths' shape.
- **Evidence.** E5 (`np.shape(wrapper) = (4, 5, 6)`, `np.ndim = 0`).
- **What would resolve it.** Name the field `widths` (or `mode_sizes`) and compare `pv.widths == x.shape`. Then
  `np.shape(pv)` is `()`, which `form_of` refuses as neither a sequence nor an array.

### 7. Should fix. Exact shape for the full weight costs `y`'s memory for weights constant over the samples, and the plan states only one such case (§2.2)

- **What is wrong.** §2.2 acknowledges that a weight per sample costs `y`'s memory on the probe kind. Others lose
  their cheap form too: a weight per coordinate of a mode, constant over the samples (`F+(1,)*W+C+(N_i,)`), and a weight
  per sample and slot constant over the edge (`F+W+(1,)`). T3Polynomial needs both: its e₀-versus-z balance inside the
  input-mode rows is "served, but only by rule 3's size-1 axes, at no `W` cost", and open-issue 29's per-row per-slot
  weight was "served" by `(J+1, W, 1)`. With `W` in the thousands, on `uniform_jax` in production, the full weight
  doubles the data's memory, and each minibatch draw gathers it; the library's own rule is to prefer memory to compute
  on `W`-scaling paths.
- **Evidence.** `T3POLYNOMIAL_JET_USE.md:140-153`; `DATA_METRIC_PLAN.md` §2's table of examples; the plan §2.2 and §9
  (the dry port "passes its per-order weights as a block weight", nothing about the per-component weight).
- **What would resolve it.** Say what each lost form costs and where T3Polynomial's per-component weight goes in the
  dry port (P18), and decide: accept the memory, or a structured factor now (challenge C2).

### 8. Should fix. The "typed faces" do not fix the layer (§1.5, §3.3)

- **What is wrong.** §1.5: "every other public sampling function is a typed face, one engine call with `kind`, the call
  type and a literal `want` fixed". The layer is not fixed: the engine reads it from the object, so the ragged twin
  runs on a uniform point and returns the packed form, and after K4 ("each twin is one engine call and nothing else")
  the uniform twin will run on a ragged point and return tuples. Each twin then has two return types, against G3 and
  against the single-typed annotations §3.3 promises (`ut3_probe: NDArray -> NDArray`).
- **Evidence.** E6; `probing.py:112`; `ut3_sampling.py:107-109`.
- **What would resolve it.** Each twin checks its object's layer (`frame_form(x)[0]`) and raises on the other, naming
  its twin; §1.5 adds "the layer" to what a typed face fixes; K4's tests include the cross-layer refusals.

### 9. Should fix. K1 edits the functions K6 replaces (§7: the order, K1, K6)

- **What is wrong.** The plan puts K0 first "so that no doomed code is edited". K1 then changes the weight arguments of
  `jet_to_dual_jet`, `dual_jet_to_jet`, `derivative_weight_to_taylor_weight` and `prepare_derivative_data` (K1 part 3),
  their tests (`tests/test_jets.py` `TestDualJet`, `TestDerivativeData`) and the many doctests of `jets.py`'s module
  docstring, and K6 renames and splits the same functions and edits the same tests and doctests again.
- **Evidence.** The plan's K1 part (3); §5's rows 1 to 6; `jets.py:67-152` (the module doctests that pass weights).
- **What would resolve it.** Move the renames of the weight-carrying helpers into K1 (`derivative_block_weight_to_taylor`
  and `prepare_*_derivative_data` are weight functions; the dual-jet pair takes the two weights), and leave K6 the
  form split and `data_preparation`; or put K6 directly after K2 (it needs only `form_of`) and before K1's frontend
  part. Either way say which tests change once.

### 10. Should fix. K0's gate and inventory, as written, fail or mislead (§7 K0)

- **What is wrong.** (a) K0 adopts "P20's" gate. P20's own gate includes a grep over `t3toolbox/`, `tests/`, `docs/`
  and `examples/` that finds none of the removed names, nor `chunk_size` or `sum_over_probes`. At K0 the translation
  (alive until P17) still has `chunk_size` in its three old signatures, and the pages still name the removed names in
  prose (P19's work: M1(c) counts 70 lines in 16 pages). So that gate cannot pass at K0, and the plan does not say
  what replaces it. (b) The plan deletes "the other orphaned private helpers (M1 a2 ...)". M1(a2) lists `_BATCH_FIELDS`
  as "unreferenced anywhere in library", but it is used by `LeastSquaresBatch`'s pytree registration, inside the
  module-level `if jax_available:` block, which the scan does not read. Deleting it breaks the import wherever jax is
  installed. The method can miss any reference inside a module-level `if`. (c) Small: the plan says the pages are
  "nine, prose only"; M1(c) says 16 pages with 4 fenced (non-doctest) lines. The test counts quote "direct" figures
  (`test_curvature.py` 14 of 96); with the indirect ones it is 29 (`test_uniform_curvature.py` 10, `test_uniform_fitting.py`
  11, `test_uniform_probing.py` 2 and a module-level `import sampling_derivatives` that fails the whole file).
- **Evidence.** `PART1_PLAN.md:2657-2660`; `t3toolbox/optimizers.py:132-147`; `K_MEASUREMENTS.md:164-262`, `:432`,
  `:272-290`; `backend/fitting.py:809`, `:814`.
- **What would resolve it.** State K0's grep with its exemptions (the translation's `chunk_size`, the pages' prose, both
  removed later) or drop it for the zero-reference inventory alone; re-run M1(a2) with every module-level statement
  read (not only top-level defs); keep `_BATCH_FIELDS` (K1 changes its contents); quote direct+indirect test counts.

### 11. Should fix. The program-size gate needs its measure; the per-site steps pass it; the engine's shape reads break §1.6's rule as worded (§1.6, §6, §12.7)

- **What is wrong.** "The compiled program has the same size at two values of `d`" holds or fails depending on what is
  counted. On the traced program (jaxpr) it fails today on every path, including `xi` alone: the engine reads shapes
  by iterating over the packed supercores (`_read_setting`'s `F_in`, `nD`, `nU`, `N`, `_bonds`, `_check_stacks`,
  `_read_K`), and on jax each iteration traces a `slice` and a `squeeze`, 11 of each per mode. XLA removes them as dead
  code, so the optimized program is constant on those paths. Those loops are "a Python loop over the `d` slices of a
  packed array" in library code, which §1.6 says does not appear. The per-site steps do pass (the question §6 leaves
  open), so §12.7's fallback is not needed. (Also: an XLA loop of trip count 1 is simplified away, so `d = 1` or `2`
  is a poor test point.)
- **Evidence.** E2b (`slice`, `squeeze` 37 to 59 from `d = 3` to 5 for `want=('xi',)`); E2c (constant 63, 412, 576;
  growing 438/452/491 and 1414/1436/1487 for probe `f` and `f, g`); `sampling_variables.py:1070-1091`, `:1094-1169`,
  `:1020-1022`.
- **What would resolve it.** Define the gate as the instruction count of the optimized HLO
  (`jax.jit(f).lower(...).compile().as_text()`), at `d = 3` and `5` (or more), on CPU, for the engine and for the
  oracle's query (the oracle's data sums are on the §6 list but outside "the engine's program"). Either rewrite the
  shape reads on the uniform layout from the supercores' own shapes (one `np.shape` each), or exempt shape reads in
  §1.6's wording. Drop §12.7, or keep it with today's measurement.

### 12. Should fix. `fitting.pack_batch` cannot check a stacked model's data (§4.3, §3.1)

- **What is wrong.** `fitting.pack_batch(oracle, batch, N=None)` checks `y` and the full weight "against the sample's
  widths and `W`; no tensor is needed". A stacked model's data carry `C` (size 1 where shared), and `check_probe_data`
  without its `stack_shape` refuses them by rank. `local_model` supports stacked points today (it passes
  `stack_shape=x.stack_shape`); only the optimizers refuse a stack.
- **Evidence.** `t3toolbox/fitting.py:605-606`; the checkers' signatures in §3.1.
- **What would resolve it.** `fitting.pack_batch(oracle, batch, N=None, stack_shape=())`, as `check_batch` has; or it
  checks only the sample and leaves the data to `prepare_batch` (which has the point).

### 13. Should fix. The translation's obligations per slice are incomplete (§4.3, §12.6, §7)

- **What is wrong.** The plan adapts the translation in K1 (the weights) and keeps it packing for a uniform `x0` until
  P17. It also has to change in K3: the old entries sample is a `(d,)+W` array, refused on the ragged layer from K3, and
  `_old_fit`'s `entries_derivatives` path calls `index_to_one_hot(main, shape)` on that array, ragged-only after
  §3.3. In K5: the old draws (`_old_draw`) return ragged batches, which `_frontend_draw` packs today; K5 removes it, so
  the translation must pack each draw itself for a uniform `x0`. In K6: it calls `prepare_derivative_data`, renamed.
- **Evidence.** `t3toolbox/optimizers.py:212-229` (`index_to_one_hot`, `prepare_derivative_data`, `_old_draw`),
  `:117-122`.
- **What would resolve it.** A row per slice for the translation in §7 (what it gains, which old-form tests pin it).

### 14. Minor. Packing destroys a second piece of information, and the engine does not check what §1.2 says it checks (§1.2)

- **What is wrong.** "Packing destroys exactly one piece of information, the widths": it also destroys the per-mode
  jet sizes `F_in_i` of a sample (zero-filled to the largest). So `unpack(pack(ww))` does not give `ww` back, the
  uniform `xi`, `xi_dot`, `xi_U_dot` and `_eta_*_top` have the largest jet sizes, and §9's contract
  `unpack(op_uniform(pack(v))) == op_ragged(v)` fails for those and for the `'w'` echo (equal as polynomials, not as
  arrays). "Everything else (`d`, `N`, the jet axes, `W`, `C`) ... is checked on the packed form by the sampling
  function, always": the engine checks the sample's rank and `W`, not its jet sizes (its docstring says so).
- **Evidence.** `ut3_operations.py:161-176`; `sampling_variables.py:1074`, `:1150-1169`, `:1710-1711`.
- **What would resolve it.** Say both; give the contract "up to zero coefficients on the leaf jet axes"; either add a
  jet-size check (at least one slot per jet axis) on the packed form or drop the word "always" for the jets.

### 15. Minor. K2's re-pointing of the old checkers is not additive (§7 K2)

- **What is wrong.** K2 re-points `jets.check_sample` and `check_data` to the new checkers, which take the ragged form
  only. Until K3 the ragged layer still takes a `(d,)+W` index array and a mode-leading probe `y`, which the new
  checkers refuse.
- **What would resolve it.** Re-point in K3, with the change of form.

### 16. Minor. K3 needs `pack_batch`'s index branch (§7 K3, K4)

- **What is wrong.** From K3 a ragged entries index is a tuple. `pack_batch` passes the entries sample through
  unpacked (`uniform_fitting.py:393-394`), so a uniform entries fit at K3 hands the uniform engine a tuple. On numpy it
  works by accident (`np.asarray` in `_uniform_index`); a traced jax index is turned into numpy there. The plan lists
  `pack_batch` in K4.
- **What would resolve it.** `pack_batch`'s `pack_index` branch in K3 (notes 1c: F3.2 "gains the ragged-index branch").

### 17. Minor. The engine's form check omits the directions and the centers (§3.2)

- **What is wrong.** §3.2 lists the sample, `y`, the full weight and the held variables. `variations`,
  `core_directions` and `Vc` are per-site objects of the frame's layer too, and "the engine enforces no mixing for all
  of sampling". I did not test what a ragged direction does on a uniform frame today.
- **What would resolve it.** List them, and test their refusal on each layer.

### 18. Minor. Decisions the notes flagged and the plan does not state (§2.3, §12)

- **What is wrong.** Notes 5 lists "`LeastSquaresBatch`'s field order and the engine's two weight arguments" among
  Claude's decisions for the review; §12 omits both. The order matters: a positional `LeastSquaresBatch(ww, y, w, 0.5)`
  (as in the class's doctest) lands the scale in whatever field comes fourth. `computed_names` (its arguments "are
  exactly an engine call's") must change with the engine.
- **What would resolve it.** State both in §2.3 and §12.

### 19. Minor. §5 against its own criterion and counts (§5, §1.4)

- `entries_data_block_norms(y)` is `apply_data_block_norms((), y)`: the same computation, so by the criterion of §1.4 no
  third function. The jet functions have no entries version for exactly that reason; the two rows disagree.
- "The shape of every split" (one private core at offsets 0 and 1) does not fit `rescale_coordinates` (different
  mathematics by kind) or `data_block_norms` (a different reduction).
- The count text says `pack_sample` and `pack_data` "have no successor"; the table gives them `pack_vectors` and
  `pack_index`.
- The frontend's own `fitting.rescale_coordinates` (a kind-switched function with a check, not a re-export,
  `t3toolbox/fitting.py:110-190`) is not in the table; `computed_names(kind, ...)` is a public kind-taking function not
  listed with the engine's exemption.

### 20. Minor. The two uniform examples (§7 K5, §9, §12.6)

- Both uniform examples call the optimizers in the old form, which the translation packs for (§12.6). K5 says they
  "gain their packing line"; §9 gives that line to P17. One of the two is wrong; the examples step of the gate decides
  which slice must touch them.

### 21. Minor. `PackedVectors` as a pytree (§4.1)

- Validation in the constructor runs on unflatten (the library's precedent), so a `jax.vmap` `in_axes` prefix tree
  (`PackedVectors(1, widths)`) or a `tree_map` returning `None` cannot be built, and `vmap` over the default axis maps
  over the modes. Say that `W` is the stack and `vmap` is not supported (as for a stacked uniform T3), or construct
  without validation on unflatten. Require `array.ndim >= 2`.

### 22. Minor. The derivative-data recipe with two weights (§2.2, §12.1)

- The `k!` placement is right (§4 below). But `prepare_*_derivative_data` now always returns a block weight (the `k!`
  table by default), which a user naturally passes on; a user who also computes a data-relative full weight on the
  Taylor data (a relative error per entry, which is coordinate-free) then gets `k!` times it. Today a computed weight
  replaces the returned one; with two weights they multiply. The docstring and P19's page should give the recipe.

### 23. Minor. The oracle's packed sums (§2.3, §6)

- The plan masks `y`'s padding before `<y, M y>`; mask the full weight too (finite junk squared can overflow, and
  `inf * 0 = nan`), and keep M-arc 13(c) explicit in the one-product form (a `y` with `C = 1` counted over `z`'s shape).

### 24. Minor. `ut3_unscale_coordinates` (§3.3, K7)

- It changes in this sub-arc (packed scales) and holds the shape, so it is a mask-holder; it passes rank-padding junk
  through (its docstring says so). K7's meta-test does not enumerate it. Either list it in G2's surface or say why not.

## 3. Challenges to rulings

### C1. Ruling 1b (data_metric stays one function) was made before ruling 3d, which removed its premise

1b: split when the computation depends on the kind; "`data_metric` is elementwise and needs no axis". After 3d,
`data_metric` applies a block weight by left alignment: not elementwise, and its structural check depends on the kind
(finding 3). The criterion as worded keeps one function whose check is a prefix test with a silent coincidence. I would
extend the criterion to **"the computation or its structural check depends on the kind"**. Then the alternatives are:
`probe_data_metric(order, r, block_weight, full_weight, scale)` and `apply_data_metric(order, ...)` (entries is apply at
`()`), exact in both weights; or `data_metric(r, full_weight, scale)` elementwise and exact, with the block weight
applied only where the kind is known (the engine, the oracle, the batch method via its oracle). The second keeps 1b's
spirit (the public `M` is elementwise) and puts the block factor where its shape can be checked.

### C2. Ruling 3d's "exactly `z`'s shape" for the full weight: let the sample axes be 1, like `C`

The plan already allows `C` of size 1 ("data shared by a stack"). Allowing `W` of size 1 on the full weight (never the
edge, never `F`) means "constant over the samples", the symmetric statement. What the redesign removes stays removed:
the edge is always full, so packing is one rule with zero padding, masking is unconditional, and there is no mixed-edge
case. What returns is one well-defined case in `take` (every `W` axis 1: no gather), which decision 18's raise already
pins. It keeps T3Polynomial's per-component weight at no `W` cost (finding 7). The alternative is a third named factor
(a per-coordinate weight, `len=d`, `F+(N_i,)` per mode) when the port needs it; that is the plan's own "extend by a
named factor" rule, decided now rather than at P18.

### C3. Ruling 1b's "probe_X ... or calls it once at offset 1 on a packed array": is the packed branch needed in public?

The stated workflow is "prepare ragged, pack last, unpack at the end" (§4.1, notes 3c). The only consumer of a packed
probe branch is the engine's data-space steps, which are private. The packed branch costs one more form per helper to
test, and two helpers cannot keep zero padding without the widths (finding 4). The same reasoning the plan gives for
`index_to_one_hot` (§12.4: "nothing needs it") applies: public probe helpers ragged-only, refusing an array with a
message naming `unpack_vectors`; the offset-1 cores private. F7 is then settled by refusal, not by a third form.

### C4. "Liberal in" by multiplication (G2, the O6 ruling)

"Padding is junk (anything finite)" is the cores' rule, where junk is produced only by library operations. For data
the junk is produced by users, on `.array`, and divisions are routine (finding 5). Masking the data-space inputs by
selection costs the same and makes "liberal in" true without a finiteness precondition. Cores can keep the multiply.

## 4. What is sound

- **The sample needs no entry mask** (§3.2, §12.5). E1 found no path, over every public name of both kinds, both call
  types, jets, `K`, `Vc`, `sum_over_samples=False` and both directions, where junk in the sample's padding reaches a real
  entry, a held output or a scalar; E1b confirms the junk does reach `g`'s padding and the `'w'` echo, which the exit
  masks (and, after the plan, the echo's) remove. Keeping the held variables masked is right (the apply kind's
  `close_step` sums a bond that no core zeroes on a held `mu`).
- **The weights' algebra.** `M = gamma * b**2 * f**2` covers every diagonal metric; the block weight needs no `K`
  placement (K sits after `W`, behind the left-aligned axes); the `k!` belongs in the block weight only:
  `omega_t = k! omega_d = (k! b_d) f_d`, also when a full weight is given in derivative coordinates;
  `derivative_block_weight_to_taylor` is rightly kind-free (the table's trailing axes are `F` for both kinds).
- **The engine's per-site steps keep the optimized program constant in `d`** (E2c), so the §6 gate is meaningful once
  its measure is fixed.
- **Every `file:line` I checked is right:** `uniform_tucker_tensor_train.py:232`, `:53-61`; `sampling_variables.py:1556`,
  `:1643`, `:1678`, `:1868-1873`, `:1914`, `:1932-1935`; `sampling_steps.py:915-1099`, `:926`; `jets.py:579`, `:740-745`,
  `:767`, `:1178`, `:1303`, `:1424`, `:1568`, `:1657`; `backend/fitting.py:758`, `:840`, `:858`, `:870-936`, `:1137`,
  `:1166-1190`; `ut3_operations.py:128`, `:179`, `:191`, `:200`, `:211`; `uniform_fitting.py:375`, `:398`, `:431`;
  `ut3_sampling.py:71`, `:107-109`, `:229-232`, `:250`; `tucker_tensor_train.py:63`; `t3toolbox/fitting.py:582`, `:589`;
  `t3toolbox/optimizers.py:117-122`, `:196`, `:216`; `TARGET_DECISIONS.md:316-327`, `:354-357`, `:498-511`, `:869-873`,
  `:889-891`; `PART1_PLAN.md:2234-2252`; K0's code references (`apply.py:23`, `entries.py:26`, `backend/fitting.py:34`,
  `ut3_sampling.py:26`, `utv_sampling.py:35`, `apply.py:313`, `:427`, `:617`, `:663`, `entries.py:265`, `:375`).
- **The counts:** §5's 15 helpers today (10 in `jets`, 3 in `data_preparation`, 2 in `uniform_fitting`) and 18 from the
  split; K1's `data_metric` 6 library calls and `blocks_to_weight` 3; K4's 93 container reads (with the 2
  `isinstance(x, tuple)` the plan's breakdown leaves out of its sum).
- **K0 first** is safer than last, given finding 10: no old code is ported, the five non-canonical outputs go, and the
  remaining oracles (the dense reference, the tree before K1) are independent of the engine. K4's uniform-against-ragged
  comparison shares the per-site steps, but the code K4 changes (the data-space steps) differs between the layers, so it
  is a real oracle for that change.

## 5. Not verified

- The Sphinx build with a constrained `TypeVar` (K2), and anything on the jax-0.4.30 / numpy-1.26 floor.
- The tests: I ran none, and did not recount the investigation's breakage figures (135 tests, 275 units, 13 and 9
  module doctests) or K1's (206 `weight=`, 76 `LeastSquaresBatch(`); some of the 206 are old-form optimizer calls, which
  the translation keeps working.
- T3Polynomial's code: I read only `T3POLYNOMIAL_JET_USE.md`.
- What a ragged direction or `Vc` does on a uniform frame today (finding 17).
- The behaviour of the not-yet-written `form_of` on exotic containers; I checked only today's readers.
- GPU or other XLA backends for the program-size gate (E2c ran on CPU, jax 0.10).
