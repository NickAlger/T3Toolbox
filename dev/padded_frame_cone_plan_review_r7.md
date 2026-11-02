# Padded frame / cone reference plan — critic review, round 7 (plan round 10)

**Scope:** the round-10 fixes (CHANGES :91-97), plus two questions from the coordinator: is the `C` return right, and
is the float64 note right, against the gold standard? Probe (read-only): `probe_imports_r10.py`.

## The r6 findings

| r6 | status | where verified |
|---|---|---|
| **MINOR 1** template remnants | **fixed** | See the breakdown below. |
| **I-32** | **settled: A** (Nick) | `backend/padded_frames.py`; imports at :696-701, justified by the import graph alone (the `utv_operations` citation survives only in round 3's CHANGES row). My probe of the pre-arc graph plus these edges — `fv_operations` back at `efe5e3d`, `geometry → padded_frames` — finds **no cycle**. The `ranks → sharing` smell is in the ledger. The S2′ row moves the S2/S3 code and restores `fv_operations`. D is recorded as not taken. Accepted. |
| **J1** `dtype` | **dropped** | §3.2, §4.2 :1152-1155, §5.2, §5.4, I-43. An array-free call gives the backend's default float, as `t3_zeros`. |
| **J2** scope | **stated** | §12 :2344 and I-44: the rule binds every backend function of the feature; in the driver and geometry layer it is the goal. |
| **J3** `with_ranks` | **done** | §3.4 explains why its hand-built masks are safe. T5(i) tests `with_ranks` of a geometry's own ranks against `from_point` over unstacked and stacked cases, varying ranks, a non-minimal point, padding above the ranks, and shared groups. Also in §1, S5, I-16, I-45. |
| NIT §3.1 | **done** | The cone path is drawn, with the array-free call's `stack_shape` / `use_jax` (:608-619). |

**MINOR 1, site by site:**

- §4.2 :1156 builds the empty old frame from integers: `_empty_old_frame(shape, C, xnp, dtype)`.
- The raw-uniform recipe takes integer ranks and builds the old frame natively.
- T4(a) passes `x_old=None`.
- T4(b) compares `out[5]` with the native sweep's frame masks and with the geometry's `var_masks`.
- "Valid frame" (:1705-1707) uses the target stores.
- §3.6 :982 and the pitfalls rows name the geometry's masks and padded sizes.
- What remains of "template" in the plan is rulings and history: :627, :814, :1958, I-26, I-42, §12, and the pitfalls style row.

## The coordinator's two questions

**The float64 note is right** (§3.5 :905-909, I-43).

- **Gold standard.** It matches `t3_zeros`, whose array-free output is `xnp.zeros(...)` in the backend's default
  float, with no `dtype` parameter.
- **Under numpy with float32 data**, the array-free first frame is float64. The first gradient then mixes a float32
  residual with a float64 frame and promotes. Every later hint and frame follows `result_type`, so all passes run in
  float64.
- **Under jax**: without x64 the default float is float32, so nothing promotes; with x64 enabled the note's case
  applies.
- **"The cone value is unaffected"** holds: it is a scalar reference, computed more precisely if anything.
- **The docstring advice** to cast such a frame before reusing it as a float32 `start_frame` is the right guard.

**The `C` return is correct but redundant** (§3.4 :850, §3.1 :572/:594/:614).

- **Correct.** The stack is a static int tuple read inline from `x_cores`' core shapes. That is a basic value derived
  the gold-standard way, as `tucker_change_core_shapes` does with `cores[0].shape[:-2]`.
- **Redundant.** The geometry protocol already has `stack_shape(x_cores)` on every geometry (`geometry.py:263-269`,
  `:418`, `:652`). Its docstring says: "which axes are the stack is a layout question, so it belongs to the geometry".
  The T3-free driver can call `geom.stack_shape(x0)` and leave `padded_structures` as `(shape, target, working)`.
- **Verdict on it: NIT.** Either is acceptable. Using the existing member avoids two sources for the same integers.

VERDICT: APPROVE
