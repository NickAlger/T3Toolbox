NOT YET

# Review of `PACKING_PLAN.md`, revision 4

Reviewer: Claude (critic), 2026-11-25. Branch `graded-messages` at `3375d213`, with the working tree's `dev/` files
(revision 4, 905 lines; `PACKING_PLAN_NOTES.md` §7; `RESPONSE_r2.md`). The library and the tests are as in rounds 1
and 2.

**The verdict, briefly.** The four rulings are sound designs, and none needs to be undone:
- the kind and the order in the batch;
- a frontend batch of `PackedVectors`;
- selection as the engine's one mask;
- `W` of 1 on a full weight.

What is missing is their seams, which revision 4 does not yet specify:
- the consumers of what moved out of the oracle, outside `backend/fitting.py`. The display among them loses its
  table's labels silently (E8b), against §2.4's "nothing else changes";
- a validation or drawn batch whose kind or order differs from the fit's;
- the boundary between the two batch classes of one name;
- where the checks the oracle made at construction go;
- packing a full weight whose `W` pattern differs between modes;
- the public metric's rule set against the ruled principle.

Each is a paragraph of text, not a redesign. A revision that answers S1 to S6 gets my sign-off without another full
round; I will read only the diff.

## 1. What I read and ran

**Read.** Revision 4 whole; the notes' §6 and §7; `RESPONSE_r2.md`. Code for the new design:
- every read of `oracle.kind`, `oracle.order`, `oracle.take`, `oracle.n_measurements` and `check_batch` in
  `t3toolbox/`;
- `backend/optimizers.py`: the `Oracle` protocol `:181-231`, `_require_output`, `flat_draw` `:625-641`,
  `_require_draw` `:645-654`, `_alignment_projector` `:928-945`, the `LocalModel` registration `:2531-2546`;
- `backend/optimizer_display.py:612-660` (`make_newton_display`) and `_format_table`;
- `LeastSquaresOracle.__post_init__` (`backend/fitting.py:1057-1066`);
- `common.ValueHashedFields` (`common.py:609-690`);
- the frontend `optimizers.py` (`mc_sgd`'s signature, the translation `:232-265`), `fitting.py:582-609`,
  `t3toolbox/__init__.py`'s exports, `docs/conf.py:126-182` (how the docs resolve a name defined in two modules);
- the engine's masking sites (`sampling_variables.py:1368-1370`, `:1442-1480`, `:1910-1935`) and `sampling_steps.py`
  for any masking of its own;
- how the examples and tests use the frontend name (`examples/fit_hilbert_from_probes_optax.py:123-138`).

**Ran** (scratch folder `/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/critic/`):

| Script | What it tests | Outcome |
| --- | --- | --- |
| `e8_rev4_checks.py` | (a) packing a probe full weight whose `W` axis is 1 in one mode and full in another (each legal by §2.2); (b) the display's table when `order` is read off a fieldless oracle (`None`) against the `order` given | (a) `pack_vectors` refuses ("every vector must have the stack shape of the first"); (b) with `order` the table reads `rows=mode cols=order`, `ord0 ord1 ...` (and `ord(0,1) ...` for two jet variables); with `None`, `rows=axis 0 cols=position`, `[0] [1] ...`: the same numbers, the meaning of the axes lost |

E7 (round 2) covers selection on every engine input, cores included: equal values, only signed zeros differ.

## 2. Findings

### S1. Should fix. What moved out of the oracle has consumers the plan does not list, and one of them degrades silently (§2.4, §4.3, §5, §7 K1)

- **What is wrong.** The kind, the order, `take`, `n_measurements` and `check_batch` leave the oracle. They are read in
  places outside `backend/fitting.py` that K1's parts do not name:
  - **The display** reads `order = getattr(oracle, 'order', None)` (`optimizer_display.py:622`). With a fieldless
    oracle it gets `None` and prints the table with positional labels, silently (E8b). §2.4 says of the display
    "nothing else changes".
  - **The default draw** (`flat_draw`, `backend/optimizers.py:625-641`) calls `oracle.n_measurements(batch)` and
    `oracle.take(batch, indices)`. `_require_draw` (`:645-654`) requires them on the oracle.
  - **The `Oracle` protocol** (`:181-231`), the documented extension point for user-written oracles, has two
    contracts the move reverses, besides the "all static / all leaves" split that §2.4 names:
    - its optional members (`check_batch`, `n_measurements`, `take`: O-29) are the oracle's;
    - "its data are its batch, which the optimizers pass through unread". The default draw now reads the batch.
  - **The translation** calls `oracle.n_measurements(batch)` (`t3toolbox/optimizers.py:249`).
  - **The frontend's** `_is_packed` and `prepare_batch` read `oracle.kind` and `oracle.order` (`fitting.py:585`,
    `:608`). `mc_sgd`'s and `adam`'s parameter comments say "(with `n_measurements` / `take` for the default draw)"
    (`:331`, `:354`).
  - §4.3 contradicts itself on user oracles: "A user-written oracle receives the backend batch … and reads the kind
    and the order off it" against "An oracle that is not a `LeastSquaresOracle` has its own batch type".
- **What would resolve it.**
  - Add these sites to K1's parts.
  - The display reads `batch.order`. D4's "given, never read off the model" still holds, since the batch states it.
  - State the protocol's new form: the default draw needs `take` and `n_measurements` on the batch, and a user
    oracle with its own batch type gives its batch those two, or passes `draw=`. Record O-29's reversal by name.
  - Say which of the two sentences on user oracles holds.

### S2. Should fix. A validation batch, or a drawn batch, may now differ in kind or order from the fit's, and nothing refuses it (§4.3)

- **What is wrong.** While the oracle held the kind and the order, a validation batch was read under the fit's. Now
  each batch is read under its own, so:
  - a `val_batch` of another kind or order is queried as that kind, and the display formats both tables with one
    `order`;
  - where the block shapes coincide, the `train|val` cells set two different quantities side by side, silently
    (probe with `d = 3` at `order=()` and apply at `order=(2,)` both have the block shape `(3,)`); otherwise it
    raises late, in the formatting;
  - a user's draw that returns a batch of another kind or order fits another measurement map at that step.
- **What would resolve it.** `newton_cg` (both tiers) refuses at entry a `val_batch` whose `kind` or `order` differs
  from `batch`'s (a static comparison). The frontend's per-draw hand-over compares a drawn batch's `kind` and `order`
  with the full batch's (two tuples per draw; free).

### S3. Should fix. The public metric's full-weight rule contradicts the ruled principle, though it can obey it (§2.3, §2.2)

- **What is wrong.** The ruling is "a size of 1 is allowed exactly on stack axes, never on structural ones". §2.3
  keeps "each axis 1 or `r`'s size, except the mode axis", which allows 1 on the jet slots and on the edge. The reason
  given ("the function is not told the stacks") covers only the split between `W`, `K` and `C`. The metric does know
  the structural axes:
  - probe: the mode first, `F` from `order`, the edge last;
  - apply: `F` leading.
- **What would resolve it.** In `probe_data_metric`: the mode, `F` and the edge exact; every axis between them (`W`,
  `K`, `C`) 1 or `r`'s size. In `apply_data_metric`: `F` exact, the rest 1 or full. The engine's size-1 `K` placement
  stays legal, and the metric and the batch then agree except for `y`'s exception on `W`. If the metric is meant to
  stay looser, list that as a departure from the ruling in §13.

### S4. Should fix. A full weight whose `W` or `C` pattern differs between modes is legal and cannot be packed (§2.2, §3.1)

- **What is wrong.** §2.2 allows each `W` and `C` axis of a full weight to be 1 or full, mode by mode. The one packer
  requires every mode's lead shape to agree (E8a refuses). §3.1's premise "with exact shapes (§2.2) their arrays
  agree across the modes except in the edge" no longer holds, for `W` now and for `C` already in revision 3. A user
  who passes a per-mode mix gets the packer's refusal, not a rule. (The engine is not at risk: a ragged mix is legal
  and broadcasts correctly there.)
- **What would resolve it.** Require one size-1 pattern on the stack axes for every mode of a probe full weight (and
  of `y`'s `C`), checked by the probe data checker, by `pack` and by the frontend batch at construction, with a
  message naming `np.broadcast_to`. The alternative, the packer broadcasting the stack axes, is a data-packing rule
  again and is not recommended. Fix §3.1's sentence either way.

### S5. Should fix. The seam between the two batch classes of one name (§4.3, §14.1, §14.3)

- **What a frontend entry does with a backend batch.** Today's `prepare_batch` passes through anything that is not an
  instance of `LeastSquaresBatch` (`fitting.py:607`). Under two classes of one name, a backend packed batch built by
  hand would reach the backend unchecked, which is G1's hole again. Refusing it with "expected `LeastSquaresBatch`,
  got `LeastSquaresBatch`" helps nobody. Decide: refuse, naming both modules (and how to build the frontend one), or
  accept the ragged form only.
- **The frontend batch as a pytree.** Not stated. A custom loss jitted with the batch as an argument needs it
  registered: the members as children, `kind` and `order` as aux. Its validation runs on rebuild, which is fine for
  real arrays and tracers, as for the library's other classes.
- **The frontend `LocalModel`'s `batch`.** Today the class holds only `(geometry, backend)` and derives everything.
  Say whether its frontend batch is derived from `backend.batch`, the frame's shape and the batch's aux (no change to
  its pytree; recommended), or stored (its registration changes, and the data appear twice among its leaves).
- **The per-draw hand-over** reads `.data` (§14.3). Say what it does when a user's draw returns a backend batch.
- **The frontend `data_metric(r)`** should compare `r`'s mode sizes with the batch's (G1). Say what `pack()` does on
  an already packed batch.

### S6. Should fix. The checks the oracle made at construction must move with the fields (§2.4, §4.3)

- **What is wrong.** `LeastSquaresOracle.__post_init__` (`backend/fitting.py:1057-1066`) checks O-44 and O-46 today:
  the kind is one of three, the order a tuple of nonnegative ints, the entries kind's order `()`. The plan moves the
  fields and says nothing of the checks. The engine still raises on a bad order or on entries with jets, but only at
  the first query, not "before any work". An unknown kind, or an old-style positional `LeastSquaresBatch(ww, y)`
  (now `kind=ww`, `order=y`), fails late and confusingly.
- **What would resolve it.** The backend batch's `__post_init__` makes the same checks; they read only the static
  fields, so they are safe when jax rebuilds the object. It normalizes `order` to Python ints. The frontend batch
  inherits them.

### M1. Minor. Text still from revision 3 (internal consistency)

- §2.4 `:276`: "The structural check of `y` and the full weight is exact shape with `C` axes 1 or full". That is
  true now of `y` only.
- §3.1 `:340`: "with exact shapes (§2.2) their arrays agree across the modes except in the edge" (S4).
- §12.3 `:867-868` and §14.4 `:903`: "`check_batch`", now the batch's `check`.
- §4.4's table: its rows run K1, K3, K1, K4.
- §4.3's example omits `mc_sgd`'s required `rng` and `batch_size`. It uses `batch.n_measurements` as an attribute,
  while §4.3's list reads like methods: say which it is.

### M2. Minor. K1's size, and its parts (§7)

- K1 now holds the two weights, the exact `y`, the metric split, the helpers' renames, the kind and the order with
  five methods into the batch, the oracle emptied, and S1's consumers. It also rewrites about 104
  `LeastSquaresOracle(` and 76 `LeastSquaresBatch(` constructions (tests, examples, pages, doctests).
- The move of the kind and the order is a refactor and can have a bitwise battery; the weights cannot. Make the move
  the first part of K1 (before the dense reference), so that the numerical parts are reviewed apart from it.
- Nothing between K1 and K4 needs the frontend batch: K1 to K3 reach the backend class through the frontend's
  re-export. But the tests that use the frontend name with backend calls (for example `tests/test_least_squares_oracle.py`,
  `tests/_fits.py`, `tests/oracle_sweeps/*`) work by duck typing on the ragged layer and need `.data` on the packed
  one from K4. Measure them in K4's design note.

### M3. Minor. `pack`: two names, and an import direction (§3.1, §4.3)

- The backend batch gets a `pack(N)` method, and `uniform_fitting.pack_batch(batch, N)` stays: two names for one
  operation. A method in `backend/fitting.py` that calls `uniform_fitting` reverses today's import direction
  (`uniform_fitting` imports `backend/fitting`). `backend/packing.py` is the natural home for its body. Keep one.

### M4. Minor. "What still costs `y`'s memory" (§2.2)

- T3Polynomial *wants* the weight per sample and slot (open-issue 29; today reached by scaling the directions), it
  does not have one.
- The one use the maintainer envisions that depends on `W` ("reduce the impact of outlier samples") is exactly a
  weight that varies along `W` and is constant along the slots and the edge. On the probe kind that costs `y`'s
  memory. Say so beside "Not yet: `sample_weight`", since it is the case that would earn that factor.

### M5. Minor. The docs build with a second `LeastSquaresBatch` (K4)

- `docs/conf.py:144-178` gives references within a module the module they appear in, so a name defined in two modules
  (today `LocalModel`) resolves. A short reference to `LeastSquaresBatch` from a third module (the frontend
  optimizers import it) may be ambiguous under `-W`. Not verified; K4's design note should grep the short references.

### M6. Minor. "One mechanism" (§3.2)

- The start values are still multiplied by their bond mask (`sampling_variables.py:1368-1370`). This is harmless:
  they are ones the engine makes, not an input it reads. But "one helper, one behaviour" should make the exception.

## 3. Challenges to rulings, and to §14

- **§14.1 (the frontend batch keeps the backend's name), not a ruling.** The `LocalModel` precedent is weaker here.
  Users rarely build a backend `LocalModel`, but they build batches in both tiers (the backend tier being where a
  user oracle and the dry port work). The frontend name is exported at the top level. One name for two types makes
  `isinstance`, error messages and the docs' cross-references ambiguous (S5, M5). Either give the frontend batch
  its own name, or keep the name and make every message and `repr` name the module. I would keep the name and do the
  second, since the maintainer chose design F as code with this name in it.
- No ruling challenged. Design F, the kind and the order in the batch, selection everywhere, and `W` of 1 are each
  the better design. The findings above are their seams.

## 4. What is sound (checked this round)

- **The static `kind` and `order`.**
  - They are hashable aux; constant across draws, so no recompilation per draw.
  - `_prepare_jit_inputs` maps the leaves with `tree_map` and keeps the aux.
  - The backend `LocalModel` passes the batch whole through `partition_static`, so jax flattens it with its own
    registration.
  - `_alignment_projector` is cached on `(geom, oracle)` and the batch rides as a pytree hint.
  - A fieldless `ValueHashedFields` oracle hashes as `(type, ())`: one key for every least-squares fit, the batch's
    treedef carrying what the oracle's fields carried.
  - A batch with `y=None` and a drawn batch (`drawn_indices` set) have their own treedefs, as today.
  - `optional_outputs` stays on the oracle, as the protocol needs.
- **Selection.**
  - With the cores contracted to be finite, selection inside the engine and a product elsewhere agree in value (E7).
  - After K0 no kept twin pre-masks the cores (`_masked_point_and_directions` and `_mask_once` serve only the
    deleted `*_from_sweep` and derivative functions), and the oracle passes the frame unmasked, so the two
    mechanisms never meet on one array.
  - The exit path (`_mask_variable`) goes through the same helper, and `sampling_steps` masks nothing on its own.
  - The dtype care (the zero in the array's dtype) is right for numpy 2 and for jax's weak types.
- **`W` of 1 on a full weight.**
  - `take` handles every combination: all `W` axes 1 is kept as one axis of 1; partly 1 is broadcast then gathered,
    at the drawn size only; full is gathered.
  - Selection masking and `_with_K` placement are unaffected by a size-1 `W`.
  - The `<y, M y>` product counts over `z`'s shape, since `y` is exact on `W`.
  - "Stack against structural" is a sound line. `K` never occurs in data. A sample's jet axes are zero-filled
    coefficients, not broadcasts, so they are outside the rule. Several `W` axes with only some of size 1 are
    handled as above.
  - The exception for `y` is coherent.
- **The slice order holds at every boundary**, K1 to K7 (with M2's note). The translation's rows are complete apart
  from S1's `:249`.
- **Unchanged since round 2:** §5's count (28), K0, the E1 to E7 conclusions, the `k!` placement.

## 5. Not verified

- The docs build with two classes of one name (M5).
- Which tests and pages use the frontend name with backend calls (M2).
- T3Polynomial's code (which tier it builds its batches in).
- Selection under jit (E7 ran on numpy; the notes' §7 records a measurement on both environments, which I did not
  repeat).
