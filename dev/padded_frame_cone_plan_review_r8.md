# Padded frame / cone reference plan — critic review, round 8 (plan round 11: the frontend/backend split)

**Scope:** the round-11 delta only (CHANGES :99-105; §5.8; §12's split; §3.2's kept guards; T14, T15; slices;
I-22, I-41, I-46, I-47; O-6).

**Yardstick:** the gold standard's split, which I checked in the code.
- `TuckerTensorTrain.resize`, `zeros`, `__add__`, `inner` and `entries` carry 36–101-line docstrings, 8–51 doctest
  lines and the explicit raises.
- `tucker_change_core_shapes`, `t3_zeros`, `t3_add`, `t3_inner_product` and `t3_entries` have one- or two-line
  docstrings and no doctests. Their only raises are terse guards: `t3_add`'s stack mismatch, `t3_inner_product`'s one.
- `T3Frame.from_t3`, `random_orthogonal` and `orthogonalize` are static or instance methods on the frame class that
  validate, call and rewrap. `random_orthogonal` takes the integer structure plus `stack_shape` / `use_jax`.

## Overall assessment

**The split is applied faithfully.**
- `T3Frame.padded` is the right twin in the right place: static, beside `random_orthogonal`, frontend types,
  validate → `fv_padded_frame` → `T3Frame(*…)`.
- `TuckerTensorTrain.oversampled_ranks` sits beside `continuation_ranks`.
- Reusing `T3Tangent.to_t3` / `UT3Tangent.to_ut3` is right.
- The frontend `padded_start` and the optimizer keywords already had their twins.
- The three kept backend guards match what the gold-standard backend keeps.
- The doctest moves respect "stay if no twin".
- S2′ is still one theme.

**Three things need attention.**
- A natural twin is missing: `UniformTuckerTensorTrain.oversampled_ranks`.
- The shared geometry twin's tying order makes its frame not "at `x0`" for untied inputs.
- The largest new surface — the three geometry twins and `frame=` on six factories — is, in my view, more than this
  feature needs. That is Nick's decision; my recommendation is below.

## The twins, against the gold standard

| twin | class / module | frontend types | body | verdict |
|---|---|---|---|---|
| `T3Frame.padded` | ✓ static on `T3Frame`, beside `random_orthogonal`, which also takes `shape, tucker_ranks, tt_ranks, stack_shape, use_jax` | ✓ `T3Frame`, `TuckerTensorTrain`, `sharing` labels → `canonical_groups` | ✓ validate (structural, explicit; ORTH and TIED in safe mode) → one call → `T3Frame(*…)` | right |
| `TuckerTensorTrain.oversampled_ranks` | ✓ beside `continuation_ranks`; like `get_minimal_ranks`, it returns integers | ✓ | ✓ | right; a uniform twin is missing (MINOR 1) |
| `T3Tangent.to_t3` / `UT3Tangent.to_ut3` | existing | ✓ | the same backend calls as `tangent_to_t3` | right |
| geometry `padded_frame(x0, x_old)` ×3 | `ManifoldGeometry` / `UniformManifoldGeometry` / `SharedGeometry`: the analog of `frame(x0)` | ✓ | ragged and uniform: validate → `T3Frame.padded` / the backend member → rewrap. Shared: see MINOR 2 | correct in form; necessity is J1 |
| `frame=` on six factories | `fitting` | ✓ | ✓ | scope (J1) |

**The three kept guards** (§3.2 :673-675): a negative count, an old-frame / hint stack mismatch, and an array-free
argument passed with an array.

- **They fit the gold criterion**: keep a terse guard only where a violation would otherwise be silent or obscure.
  - The stack mismatch is `t3_add`'s guard.
  - An array-free argument passed with an array would otherwise be silently ignored.
  - A negative count gives obscure shape failures.
- **The length checks may move to the twin.** `ranks.compute_orthogonal_representation_ranks`, which the backend
  calls, already raises on wrong lengths; and the gold `t3_zeros` leaves lengths to the frontend `zeros`.
- **Optional:** a terse group-consistency guard, as the analog of `t3svd`'s kept `validate_sharing` (N3).

**Doctest moves** (T14). Each move goes to an existing twin:
- `fv_padded_frame` → `T3Frame.padded`;
- `ranks.oversampled_ranks` → `TuckerTensorTrain.oversampled_ranks`;
- the backend `padded_start` → the frontend one.

The linalg pair and the members without twins keep theirs. ✓ One proviso: if Nick drops a twin (J1), its doctests
and description stay in the backend function. In particular, the uniform member's description and the raw-`.data`
recipe would stay with `UniformManifoldGeometryOps.padded_frame`.

**S2′** (:2354) still has one theme: the construction layer in final form, backend plus its frontend twins, with the
`git diff efe5e3d -- fv_operations.py` gate. One process point: N2.

## Findings

### MINOR 1 — A natural twin is missing: `UniformTuckerTensorTrain.oversampled_ranks`

- **Why it is natural.** The uniform class already mirrors the ragged rank helpers: `minimal_ranks` and
  `has_minimal_ranks` are per-element on a stack. The frontend `padded_start` takes `working_ranks` on both layers.
- **The gap.** Without the twin, a uniform user must call the backend `ranks.oversampled_ranks` with mask sums; the
  plan's own r4-n4 example has to say exactly that.
- **The reason given does not fit.** "The rank rule's twin is `TuckerTensorTrain.oversampled_ranks`" covers only
  the ragged class.
- **Change.** Add `UniformTuckerTensorTrain.oversampled_ranks(p, sharing=None)`, mirroring `minimal_ranks`.
  `ranks.oversampled_ranks` should then accept per-element int arrays as `compute_minimal_ranks` does. The
  elementwise maximum already works on arrays. Otherwise, state that the uniform twin is unstacked-only.

### MINOR 2 — The shared twin ties `x_old` but not `x0`, and disagrees with `newton_cg` (only if the geometry twins stay)

**What.**
- `SharedGeometry.padded_frame` ties an untied `x_old` silently (§5.8; T15(d)). Its frame's base point is then the
  tied `x_old`. If `x0` is untied, that base is not `x0`'s tensor, which contradicts "the padded frame at `x0`" and
  the twin's own doctest.
- The safe-mode same-tensor check is placed before or after the tying, unspecified.
- Meanwhile the frontend `newton_cg(padded_from=)` **raises** TIED on an untied `x_old` in safe mode (§3.6), while
  `SharedGeometry.frame` documents "an untied initial guess a non-event for the frontend optimizers".

**Change.**
- In the twin, tie **both** `x0` and `x_old`, as `frame(x0)` would, then run the same-tensor check, then build.
- Choose one behaviour for untied old points across the shared frontend: tie silently, as `frame` does, or raise.
  Apply it in both places.

### J1 — The largest new surface (Nick's call; my recommendation)

**`frame=` on the six `fitting` factories: defer it.**
- Nothing in this feature needs it. `newton_cg(start_frame=, padded_from=)` reaches `Problem.local_model(frame=)`
  internally.
- It adds six public signatures, three checks (structure, ORTH, same tensor), T15 rows and docs, for a capability —
  a hand-built local model at an arbitrary frame — that no one has asked for.
- Keep `Problem.local_model(frame=)` as the optimizer's seam with **no twin**: it is reached from the frontend
  through `newton_cg(start_frame=)`, and it keeps its full docstring (Nick's correction). Add the factories'
  `frame=` to the ledger as Deferred.

**The three geometry twins: drop them for this release.**
- Frontend padded frames already come from two places: `padded_start` on both layers (the data-informed frame) and
  `T3Frame.padded` on the ragged layer (any hint, explicit old frame).
- The member `padded_frame` is the driver's per-pass step. It is plumbing, like the four members already listed
  without a twin. It keeps its description, the raw-`.data` recipe and a doctest in the backend.
- What is lost: a one-call uniform padded frame with a custom hint. That is a research niche with a documented
  backend route.
- If Nick keeps them, MINOR 2 applies.

Either way, the rule "most functions get a twin" still holds: every function a frontend user calls for this feature
has one.

### NITs

- **N1.** `UT3Frame` has no `padded`. The real reason is that a uniform frame needs padded sizes, which only a point
  or a geometry supplies; a standalone `UT3Frame.padded` would need a template. That reason is stronger than "the
  backend has no uniform constructor". Say so in §5.8 and I-47.
- **N2.** S2′ edits S4's **uncommitted** `geometry.py` ("brought to the integer form; S4 then lands as planned").
  That crosses the slice boundary, and S2′'s gate would run against working-tree code it does not commit. Stash
  S4's working-tree files during S2′'s gate, and let S4 adapt its member when it lands.
- **N3.** Optional: a terse guard in `fv_padded_frame` that the members of each group have equal `(rank, size)`.
  It is the analog of `t3svd`'s kept `validate_sharing`, because a violation fails obscurely inside the group
  concatenation.
- **N4.** The raw-`.data` recipe (I-41) is for backend users. If the uniform geometry twin stays, also point to the
  recipe from the backend member's one-liner (or from the internals page), so a raw-data user finds it.

## Checked and found correct (this round)

- §12's split text matches the code. Twins carry the descriptions, doctests and explicit validation; the backend
  keeps one-liners, shape comments and terse guards, as in the gold standard.
- T15(a): each twin equals its backend call bitwise, a sound same-code-path claim. T15(b)/(c) cover the moved checks
  in both modes, and the safe-mode checks skip under `unsafe()`.
- The no-twin reasons for the linalg pair and the four driver members are sound: plumbing, and there is no frontend
  linear-algebra layer. The members keep their doctests.
- No `dtype` anywhere; the `typ.Union` annotations; O-6 superseded by `T3Frame.padded`; I-22, I-41, I-46 and I-47
  consistent with §5.8.

VERDICT: REVISE
