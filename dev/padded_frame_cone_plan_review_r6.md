# Padded frame / cone reference plan — critic review, round 6 (plan round 9: the backend style)

**Scope.** This round reviews **only the round-9 delta** (CHANGES :85-89). The yardstick is the gold-standard rule
(§12; my notes `dev/padded_frame_slices/style_notes_critic.md`): only `tucker_tensor_train.py`,
`uniform_tucker_tensor_train.py` and the backend functions they call count as precedent.

**Probe** (read-only, session scratchpad): `probe_imports_r9.py`, the backend's AST import graph with each candidate
home for the store recurrence.

## Overall assessment

**The core of the delta meets the rule.**

- `fv_padded_frame` takes the structure as three integer sequences. It derives the stores inside and reads the old
  frame's stores, the hint's ranks and the stack inline from the arrays it uses. It infers numpy or jax from those
  arrays, checks integers inline, and returns the frame tuple. There is no template and no dict.
- `padded_structures` returns integers only.
- `padded_frame` builds the old frame from the old point.
- `with_ranks` is necessary and correct.

**What remains:**

- **Normative text still names a template** in six places (MINOR 1). The implementer would rebuild the pattern from it.
- **I-32 needs Nick's pick.** I add a fourth option, D: move the pure-integer recurrence down. It has no cycle and no
  new module, and it is my recommendation.
- **Three judgment calls:** the `dtype` argument; whether the rule covers the cone path in the driver; and a guard for
  `with_ranks`.

## The checklist, applied to the delta

| check | `fv_padded_frame` (§3.2, §5.2) | the members (§3.4, §5.4) |
|---|---|---|
| Inputs and outputs are basic immutable values | ✓ — three `Sequence[int]` with `len=` comments; frame and hint as tuples of arrays; `groups`; `seed`; output the frame tuple. One query: `dtype` (J1) | ✓ — `padded_structures` returns `(shape, target, working)` integers; `padded_frame` takes ints and the old point. These are geometry methods, not pure math, so returning a geometry from `with_ranks` is fine |
| Structure travels as integers, never as an object read only for its shapes | ✓ — no template frame; the stores come from `compute_orthogonal_representation_ranks(shape, tucker, tt)` | ✓ in the members. The driver's cone path is still a question (J2) |
| Structure read off the arrays actually used | ✓ — old stores `[U.shape[-2] …]`, hint ranks and mode sizes, stack from `old_frame` / `hint` | ✓ — ragged: the target ranks from `x_cores`' core shapes; uniform: from the held masks |
| numpy/jax inferred from arrays | ✓ — `tree_contains_jax((old_frame, hint))`, dtype by `result_type`. The `stack_shape` / `dtype` / `use_jax` trio applies only when no array is passed, and passing it alongside arrays is a structural error. That is the right reading of the gold standard's split: `t3_zeros` takes flags because it has no arrays, while array functions read everything from the arrays | — |
| Thin wrappers, no dict helpers | ✓ — integer checks inline; no `_frame_stores` | ✓ |

**On `stack_shape`.** The plan already reads the stack from the arrays when there are any, and takes it only in the
array-free case. The array-free call happens only for the cone's first construction (`x_old=None`, no hint). There
it is unavoidable, and it mirrors `t3_zeros(…, stack_shape, use_jax)`. Keep it.

**On `with_ranks`.** It is necessary. `from_point` and `from_frame` both need arrays, either a point or a frame. Using
them for a working structure would mean building a zero point or frame only to read its masks, which is exactly the
ruled-out pattern. It also mirrors `with_sharing`. See J3 for the one safeguard it needs.

**On the old frame's structure.** It is read inline from the old frame's cores (§3.2). ✓

## Findings

### MINOR 1 — The template survives in normative text

Each of these would steer the implementer back to the pattern:

- **§4.2 :1102.** The step reads "`old_frame is None` → `_empty_old_frame(frame)`", which takes a frame argument. §3.2
  says the empty block is built from `shape` and the boundary ranks. Make the call `_empty_old_frame(shape,
  stack_shape, dtype, xnp)`, or whatever §3.2 settles.
- **§5.2 :1386, the raw-uniform recipe in the docstring.** It says "unpack the template and the old frame". Replace
  with: "read the element's target ranks from the masks (integers); build the old point's frame natively and unpack
  it; unpack the hint; call `fv_padded_frame(shape, tucker_e, tt_e, …)`; repack and build the masks from the integer
  stores".
- **T4 :1653.**
  - (a) still says "the member with `old_frame=None, hint=None`"; the member now takes `x_old`.
  - (b) still says "`out[5] is template[5]`"; there is no template. Compare `out[5]` with the geometry's frame masks.
- **§6 preamble :1624 and :1646.** "masks (and their identity, `out[5] is template[5]`)" and "Valid frame … the
  template's shapes". Use "the geometry's frame masks" and "the shapes of the target stores".
- **§3.6 :930.** "a uniform frame's masks … equal the template's". Use "the problem geometry's frame masks".
- **Pitfalls :2279, :2281, :2293.** "repacks at the template's padded sizes", "the result follows the template's
  dtype", "the masks equal the template's". Use the geometry's frame padded sizes and masks, and the array inputs'
  dtype or the array-free arguments.

The reading rule at :1047 ("where §2 and §4 say 'the template', read the target structure") covers prose. It does
not cover a code call or a test.

### I-32 reopened: options A, B, C, plus D

`probe_imports_r9.py` confirms the cycle with the recurrence in `ranks`: `ranks → sharing → fv_operations → ranks`.
It also shows the recurrence can live in `t3_operations`, `fv_conversions` or `fv_operations` with **no cycle**, as long
as `ranks` keeps its public name as an alias.

| option | assessment |
|---|---|
| **A** — new `backend/padded_frames.py` | It works: imports go downward and the recurrence stays in one place. But it creates a topic module outside the family × kind module map of the gold standard, and it imports two private helpers (`_haar_rows`, `_identity_prefix`) across modules. Its cited justification, `utv_operations.py:769-776` and "as `sharing` hosts four", is precedent outside the gold standard, which the plan's own §12 now disallows. If chosen, justify it by the import graph alone. |
| **B** — function-level import | Reject. It breaks G-acyclic and the rule agreed in round 3. |
| **C** — the recurrence inline | Acceptable only with a test that asserts equality with `ranks.compute_orthogonal_representation_ranks` over a grid. It would be the third copy; `fv_conversions.py:206-215` already re-derives it inline. The recurrence has a documented order-sensitivity bug history (`ranks.py` docstring, review S1). A drift would at least fail loudly in `require_frame_structure`. |
| **D (recommended)** — move the pure-integer recurrence down; `ranks` keeps an alias | Host it in **`fv_conversions`**, beside `t3_orthogonal_representations`, the sweep it must mirror. In the gold standard's map that is the fv row, conversions column: a T3 structure converted to frame stores. `ranks.compute_orthogonal_representation_ranks` becomes a re-export, so nothing breaks. Then: `fv_padded_frame` stays in `fv_operations` beside `fv_zero_point_frame` and its private helpers; there is no new module and no duplicate; and `fv_conversions`' own inline copy can later call the function (ledger). The new edge `fv_operations → fv_conversions` is acyclic, since `fv_conversions`' closure reaches none of `ranks`, `sharing` or `fv_operations`. `t3_operations`, beside the gold `t3_core_shapes`, is an equally clean second choice. |

**Recommendation:** D, if Nick accepts moving one pure function behind an alias. Otherwise A, justified by the import
graph only.

### Judgment calls (for Nick; none holds the verdict)

- **J1 — `dtype` on the array-free call.** The gold standard's array-free constructors (`t3_zeros`, `ut3_zeros`,
  `t3_corewise_randn`) take `stack_shape` and `use_jax` but **no `dtype`**: they produce the backend's default float.
  Either drop `dtype` and let the driver cast the cone's start frame to `x0`'s dtype (one line, needed only for
  float32 numpy or x64 jax with float32 data), or keep it with that reason stated in §3.2.
- **J2 — the cone path in the driver.** `padded_start(problem, x_cores, x_old=None)` reads `x_cores` only for its
  structure, stack and dtype, and builds `corewise_zeros_like(x_cores)` from its shapes. That is the template pattern,
  one layer up, at the optimizer level. I would accept it there: `x0` is the optimizer's genuine input, and the driver
  is not a pure mathematical function. But §12 should say explicitly that the rule binds the backend's mathematical
  functions and the geometry members' outputs, and that the optimizer layer passes points. Otherwise the next review
  will ask again.
- **J3 — `with_ranks` builds masks by hand.**
  - `from_point`'s docstring warns "do not build them by hand" (`geometry.py:603-605`), and `with_ranks` reproduces
    the sweep's padded-size rules: `nU = min(n, N)`, `rL = rR = r`, `nD = min(nU, r²)`.
  - That is safe here, for two reasons. The conversion's frames are prefix by construction. And at the target, the
    integer-built masks are checked against the geometry's (§3.3 step 3).
  - The plan should say why, and add a cheap structural test: `geom.with_ranks(*geom's own ranks)` reproduces
    `geom.masks` and `geom.var_masks` exactly. That catches any drift from the sweep.

### NITs

- **The data-flow diagram, §3.1.** The cone's hint-free start `geom_w.padded_frame(shape, *working, None, None,
  seed)` needs the array-free `stack_shape` / `use_jax` (and `dtype`, if kept) from `x0`. Show them, since they are
  the only place the trio is used.
- **§3.2 option (A).** If A is chosen, drop "the house's placement answer (`utv_operations.py:769-776`)" from the
  justification (see the table).

VERDICT: REVISE
