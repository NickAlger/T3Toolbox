# Decisions taken while going through the two reviews

The running log of 2026-11-14, in the order the decisions were taken. Each entry names the findings it answers
and the evidence. The same content organized by topic is in `DESIGN_AFTER_REVIEW.md`; read that first.

- The reports are `reviews/review_1_design.md` (findings D1 to D23) and `reviews/review_2_library_fit.md` (F1 to F22).
- The scripts named here are in `scripts/review/`.
- **Words changed during the evening.** Early entries use the words of the time. "Level" became "scan"
  (`SampleScan`), "block" became "chunk", "component" became "sampling variable", `evaluate` became
  `sampling_variables`, `Plan` became `SamplePlan`, "keep" became "hold", and the grade suffixes `_t`, `_z` became
  `_dot`, `_tilde`. Entry 8 has the names.
- **On 2026-11-15 (23:23, entry 57) the names were fixed again:** `gamma` is `nu_L`, `zeta` (entry 54) is `eta_D`,
  the frame's `mu_dot`, `xi_dot`, `nu_dot`, `eta_dot` (this log's `_dot` of entry 8) are `mu_R_dot`, `xi_U_dot`,
  `nu_L_dot`, `eta_D_dot`, and the bare `mu_dot`, `xi_dot` now mean the base point's. Entries before 57 use the
  words of their time; `SAMPLE_VARIABLES.md` section 10 has the table, and `NOTATION_HISTORY.md` the earlier names
  and ideas.
- **The log ends with a pause** called by the maintainer (last entry). The proposal documents and the mockup had
  not been rewritten to these decisions when it was called.

## 1. The sample loop (D11, D17, F3; also F6, D9, part of F10)

Decided by the maintainer:

- **There is no executor concept.** The loop over blocks of samples is a scan over an axis of the sample stack
  `W`, inside the compiled graph. (The DAG traversal is a different loop: it runs at trace time, has about twenty
  steps whatever the problem size, and unrolls harmlessly.)
- **The user pads and reshapes; the operation never does.** The user pads the samples to a multiple of the block
  size and reshapes `W` to add an axis. The library provides a helper that does this, and the desired operation
  on unpadded input is the composition: unblock after the scanned operation after block. Policies ("auto mode")
  use the same helper.
- **The plan names the axes to scan as a tuple of indices into the sample stack.** Not a count of leading axes.
  The maintainer's reasons: a user of this feature can handle the generality, and this kind of generality tends
  to pay off later, as general `W` batching is paying off now.

Consequences accepted in the discussion (Claude's proposals, not objected to):

- Block sizes leave the plan. They are array shapes. The plan holds only which axes are scanned, which is
  structural, so it is a safe compile key.
- A second level of blocking is a second reshape: the operation scans one axis and the assemblies scan another.
- Padded samples must contribute nothing. Apply and probe are multilinear in the sample vectors, so zero vectors
  suffice (checked against the library's kernels: objective, gradient and Gauss-Newton product identical to
  1e-16). Entries are not (an index cannot be zero): the helper returns a 0/1 mask and every seed (the residual,
  `J p`) is multiplied by it in the fitting layer before a transpose (checked: exact with the mask, wrong by 40%
  without).
- The helper belongs to the sampling kind, beside `take`, because the kind knows where the sample axis sits in
  each operand.
- If the data is padded once to a multiple of the largest block, a smaller block that divides it is another
  reshape of the same data, so different operations can use different blockings.
- Sharding is the user placing the block axis across devices; padding must then be to a multiple of the block
  size times the number of shards.
- A host loop remains only for numpy and for streamed data.

Still to settle under this heading:

- Tuple semantics: order of the tuple as nesting order; which field names the axes the assemblies scan;
  validation at construction.
- How the plan reaches `hvp` inside the compiled CG solve (`_cg_solve` jits the whole loop and `hvp(p)` takes one
  argument).
- Whether `chunk_size` stays a field of the sampling kind during a transition.
- Not tested: jax, jets, the uniform layout, sharding; whether XLA copies on reshape.

Evidence: `pad_reshape_check.py`.

## 2. Blocking levels (extends 1; answers D17 and part of D2, F8)

LOCKED IN by the maintainer on 2026-11-14, after the experiments below.

One reservation recorded: the mask for padded entries "gives me pause"; accepted for now. Revisit if a cleaner
way to make padded entries inert turns up (for example an index value that gathers a zero column).

- **A plan is an ordered tuple of levels, outermost first.** Each level names the components to compute once for
  the whole block it receives, and the axes of the sample stack to scan (a tuple of indices into the original
  stack). Axes that no level scans are vectorized. Inside the innermost level the ordinary evaluator runs.
- **Outer blocking, inner blocking and keeping are one mechanism.** They differ only in what is computed before
  a scan. The library's chunked assembly today is one level that holds every per-sample component and scans one
  axis. The caller's `have` is the same thing one step further out.
- **Ordering.** Order of levels is nesting order, and it decides how large each held component is. Order within
  one level's tuple changes only the order of summation. Axis index order is free; leading axes outermost is
  recommended (contiguous blocks, natural unit to shard).
- **Naming.** The per-level set holds components for one block within a call. It is not the `keep` field the
  maintainer removed from the plan (keeping across calls stays the caller's `have`). It needs its own name.

Experiments (`levels_sketch.py` in numpy with the mockup's evaluator; `levels_jax_check.py` with the library's own
kernels under jax 0.12 on 4 virtual CPU devices). All differences are against an unblocked or unsharded reference.

| Case | Result |
| --- | --- |
| numpy, mockup evaluator, 7 level configurations on a (4, 8, 16) stack | f, g, Hp, r to 1e-15 |
| jit: vectorized; scan axis 0; hold the sweep and scan; nested; scan a non-leading axis | 3e-16 to 4e-16 |
| Traced program size for 4 blocks and for 64 blocks | 962 and 963 lines: the scan is not unrolled |
| XLA scratch for the product: 4096 samples vectorized; blocks of 512; blocks of 64 | 7.25, 1.05, 0.13 MB |
| `shard_map`, block axis sharded, each device scans its own blocks | 5e-16, 0 all-gathers |
| Plain jit, scanning the sharded axis | correct, but 12 all-gathers |
| Plain jit, a separate sharded axis left vectorized, scanning the next axis | 5e-16, 0 all-gathers |
| A per-sample result under `shard_map` | 4e-16, 0 all-gathers, result stays sharded |
| Six CG iterations as one compiled `while_loop`, scanned product inside, under `shard_map` | 1e-15, 0 all-gathers |
| Entries, 1000 samples padded to 1024 with a mask, one device and `shard_map` | 7e-16, 0 all-gathers (3e-02 without the mask) |
| Jets, order 2 (seed has the order axis before the sample axis), one device and `shard_map` | 3e-16, 0 all-gathers |
| Uniform layout (site scans inside the block scan), one device and `shard_map` | 3e-16, 0 all-gathers |

What the experiments add to the design:

- **A sharded axis must not be scanned under plain jit.** Two arrangements are clean: `shard_map`, where each
  device scans its local blocks (the library's documented recipe), or a separate sharded stack axis that no level
  scans. The second needs no sharding concept in the plan at all.
- **The scan body in the experiment closes over the data.** That is fine under jit. For eager jax the library's
  scan-body principles require a stable body with the operands on the carry, as `_wchunked_sum_step` does.

Not covered: real GPUs (the devices are virtual, on a 2-core CPU, so this tests the compiled program and the
partitioning, not device memory); eager jax; whether XLA copies on reshape; compile time of nested scans; the
exact Hessian; apply; holding components per level on the uniform layout (its sweep carries static data beside
the components).

## 3. Where the plan lives (the rest of F3)

Decided by the maintainer on 2026-11-14: **`LocalModel` carries the plan, with a default**, so that a user can
build a `LocalModel` without learning the `Plan` abstraction.

- This amends the earlier decision "the plan is passed to every call and is not bound into a model object". The
  evaluator still takes the plan explicitly on every call; `LocalModel` holds one and passes it down, so `hvp(p)`
  keeps its one argument inside the compiled CG solve.
- The maintainer's reservation: conceptually the plan is a solver parameter, not part of the local model (he had
  the same reservation about `chunk_size`, which reaches the model through the sampling kind). Threading it as an
  argument was costed at about 25 functions in `backend/optimizers.py` plus the frontend twins, with three
  non-mechanical spots (the three per-sample properties, the compiled CG, the overridable surface), and judged
  too much. A solver-owned pair of model and plan was offered as a smaller variant and not taken.
- The default plan has no levels: everything vectorized over the sample stack, which is today's behaviour on
  unblocked data.
- It rides as static, value-hashed data beside the geometry and the sampling kind, so it is part of the compile
  key, as the kind is today.

Still open: whether `chunk_size` stays a field of the sampling kind (see F8).

## 4. Sites and orders (the rest of D2, F8)

Decided by the maintainer on 2026-11-14: **a per-component option for sites**, defaulting to what each rule does
today. Association order stays out of the plan (F5).

- The accounting behind it: the per-site step ties on memory, step count and width either way; the hungriest
  step sets the chunk, so scanning that step over sites divides its scratch by `d` and allows a chunk `d` times
  larger, which makes the (always site-sequential) sweeps `d` times wider. A step that is not memory-binding
  should be vectorized over sites. Decidable from scratch-per-sample figures; the hardware constants (scan
  overhead per iteration, batched small products against one large one, saturation width) are not measured.
- This is also the lever the library's own record names for the scanned jet kernels' small-problem latency
  (`docs/contributor/deferred_and_rejected.md`, "Small-problem latency of the scanned jet kernels"): one order
  scan whose bodies are vectorized over the cores, instead of a map over cores of order scans.

Decided by the maintainer on 2026-11-14: **no option for orders; drop `Plan.orders`.** The plan is levels plus the
per-component site option.

- For a product of two full jets, "vectorized over orders" is the dense `trs[t,r,s] = C(t,r)[r+s=t]` contraction
  under another name: the scan carries `sh^r b`, and holding every `r` at once is `b` contracted with `trs`. The
  library removed that form from production in 2026.3.0 and keeps it only as test twins (the recurrence-only
  rule, I-33). `Plan.orders = 'vectorized'` would reintroduce it. I had not read that record when I put the
  field in (F15, F21).
- For a product with an affine factor there is nothing to choose: the banded two-term recurrence is one
  contraction with the order axis riding as a batch axis, neither a scan nor `trs`.
- One kernel has an order scan that is for memory only and involves no `trs` (the two-row `dxi~` correlation).
  The library measured it and chose the scan; a sample level reaches the same memory by a different route.

## 5. Sequencing, and the end of `chunk_size` (F20; closes the open point of 3 and F8)

Decided by the maintainer on 2026-11-14: **`chunk_size` is removed as part of this proposal**; when is not
important to him. On sequencing he is ambivalent and accepted the two-stage approach:

- **Stage 1, independent of the kernel refactor.** Block and unblock helpers beside `take`; levels that scan
  sample-stack axes around today's kernels; each level's held set limited to nothing or the frame sweep
  (`precompute` output); `LocalModel` carries a plan with a default. `chunk_size` stays on the sampling kind and
  acts as the innermost level.
- **Stage 2, with the refactor.** Arbitrary held sets per level, `want`/`have` on individual components, the
  per-component site option. `chunk_size` leaves the sampling kind and becomes an ordinary level.
- This replaces the sentence "the plan needs the drivers" in `EXECUTION_PLANS.md`.

Evidence: `levels_jax_check.py` ran the levels on the library's existing kernels (jit, nested scans, `shard_map`,
compiled CG). Not done: sizing the outer block. The library's estimators (`estimate_chunk_size`,
`max_chunk_size_within`) size the inner chunk only; extending the slope measurement on the compiled program to
the whole block computation is expected to work and is untried. This is also where D9 and D10 (the mockup's
`fill`) should be resolved.

## 6. No graded containers; explicit arguments (F2; also closes most of F1)

Decided by the maintainer on 2026-11-14, in three steps.

- **`evaluate` is a public backend function.** Its return is a tuple aligned with `want`, a static tuple of names:
  `f, g, xi = evaluate(..., want=('f', 'g', 'xi'), y=y)`.
- **No graded container anywhere in the architecture.** Fixed-slot tuples `(plain, t, z, tz)` with `None` for an
  absent grade were prototyped (`slot_tuple_examples.py`) and declined: the maintainer's objection is that a
  `None` slot does the job of a dict key, so what is present still has to be inspected at run time. The rule is
  for the reader of the code, not the compiler.
- **Every component is its own array, named by letter and grade; every function takes exactly the arrays it
  reads, as separate arguments, and returns one array.** The seven site rules are plain functions holding one
  literal subscript string each. A component is one function whose argument list is its reads and whose body is
  its terms, e.g. `mu_t_step(mu_t, mu, xi, xi_t, R, U, H)`. There is no pointed-core object: each term names the
  frame core it uses. The corewise frame calls the same functions with `G, G, G`.
- **`have` is supplied as keyword arguments, one per component**: `evaluate(..., want=('Hp',), p=p, xi=xi)`.
  A level's held set is still a static tuple of names in the plan, with the values in a tuple of the same length;
  that pairing lives inside `LocalModel`.
- **The generic graded program becomes the derivation and the test twin**, never imported by the package, in the
  way the dense `trs` forms are kept today. `component_graphs.py` derives the reads; the twin checks the terms.

Evidence: `unbundled_examples.py`. The probe case is 22 component functions with 59 terms on 7 literal strings;
`J p`, `J^T r` and the exact two-hole term match the library to 4e-16. Not covered: jets, the `K` and `C` stacks,
the uniform layout, apply and entries.

Costs accepted: the two-hole functions take 10 to 12 arrays; a positional argument in the wrong place is an
error only if shapes disagree; a new case means writing its terms out.

Consequence for F1: the subscript strings are literals again. The library test suite's scan pattern
(`SUBS_RE` plus an uppercase letter) finds all seven in `unbundled_examples.py` and none of the lowercase rule
strings in the earlier form. What remains of F1 is to keep the literals inside `contract(...)` calls so that the
call-site check sees them, and to pin them in `HISTORICAL`; both are implementation-time items.

Proposal text this replaces: `PROPOSAL.md` "Components are keyed by grade" and "Layer 1: operands know their
stacks"; the `PointedCore` class and `gmul` move to the test-twin role; `EVALUATOR.md` "One entry point".

## 7. The value is computed through the point's own cores (D1, F7, F21)

Decided by the maintainer on 2026-11-14, all four points.

1. **One route to the value: through the point's own cores.** Under decision 6 this is not a second method: the
   plain component functions are called with `(B, G, G, G)`. No frame is built for a trial point, which is what
   the library does today (`Problem.objective`). `evaluate` loses its `centers` argument.
2. **The value through the orthogonal frame is parked.** The identity `z[i] = OUT(eta[i], Vc[i])` holds (1.7e-15,
   `reuse_check.py`), but the route needs the Tucker center cores, which are unguarded (D1: scaling `Vc` by 1.001
   changes `f` by 2e-3 with no error), have no channel in the geometry protocol, and were examined and left
   alone in the library's caching audit (`docs/contributor/precompute_and_caching.md`, F21). The proposal must
   stop presenting it as a saving that needs no memory, and the contradiction between PROPOSAL / README /
   WORKED_EXAMPLE ("a manifold frame has no plain probe output") and EVALUATOR is resolved in favour of the
   former.
3. **Carrying `r` from an accepted trial** is described as a use of `want` and `have`: the trial returns `r` on
   request and the next model takes it as `have`. It is valid for the same point and the same samples only.
4. **The frame consistency check is a diagnostic, not a precondition**, and `(D, L)` is recorded as the
   authoritative base point, as the backend already reads it. This softens the guard designed in phase 1.
   `T3Frame.to_t3`, which reads `(D, R)`, is noted for the maintainer; no change is proposed here.

Counts per outer iteration with `t` trials, read from `backend/optimizers.py` (not timed):

| Route | Sample sweeps | Frames built | Needs |
| --- | --- | --- | --- |
| Today | `t + 2` | 1 | nothing |
| Carry the accepted trial's `r` | `t + 1` | 1 | holding one residual |
| Value through the frame at the accepted point | `t + 1` | 1 | center cores, a guard, a protocol channel |
| Value through the frame at every trial | `t` | `t` | the same, and a frame per rejected trial |

The last row is the retract-and-frame fusion the audit recorded as "build only if first-order manifold fitting
becomes a hot path"; that verdict stands.

## 8. Names (F4)

The maintainer chose to settle the names now, one at a time (2026-11-14).

- **`evaluate` becomes `sampling_variables`.** One function for every sampling operation, so no kind in the name.
  Rejected on the way: `evaluate` (too generic; the verb is `LocalModel.evaluate`'s), `evaluate_variable`,
  `compute_messages` / `message` ("messages" is the padded-frame `Psi` in the glossary), `tv_compute_probe_variables`
  (too long, and wrongly per kind), `compute_variables` (does not say which variables).
- **The noun for what it returns is "sampling variables"**, replacing "components" throughout the proposal. The
  glossary's "edge variables" (`xi`, `mu`, `nu`, `eta` at every grade) are the per-edge subset; `z`, `r`, `f`, `g`,
  `Hp` are the rest.
- Open, noted for after the naming: how the sampling kind (which sets the readout) is passed to
  `sampling_variables`. The mockup has the probe readout only.
- **Grade suffixes: `_dot` (tangent) and `_tilde` (residual), composing to `_tilde_dot`.** `mu_t` -> `mu_dot`,
  `mu_z` -> `mu_tilde`, `mu_tz` -> `mu_tilde_dot`, and likewise for `xi`, `nu`, `eta`.
  - `_dot` is already the library's spelling with this meaning (`xi_dot`, `mu_dot`, `mus_dot`, ...: 149 uses).
    The two-hole variables are already "tilde then dot" (`tau_tildes_dot`, ...: 114 uses). No `mu_tilde`,
    `nu_tilde`, `xi_tilde` or `eta_tilde` exists today.
  - **The letter under the tilde moves, and the documents must spell this out** (the maintainer's condition).
    Today a tilde names the variable it is the adjoint of. In the proposal the letter names the variable it is a
    variation of, and it is the adjoint of that variable's PARTNER on the same edge:
    `xi_tilde` = today's `deta_tilde` (multiplier of the `eta` constraint); `eta_tilde` = `dxi_tilde` / `dxi_hat`
    (of `xi`); `mu_tilde` = `tau_tilde` (of `nu`); `nu_tilde` = `sigma_tilde` / `sigma_hat` (of `mu`).
    The near miss to warn about: new `xi_tilde` against today's `dxi_tilde`.
  - Both readings hold (`tilde_adjoint_check.py`, corewise frame): with `F = sum_i <s_i, z_i>` and the site
    contractions as constraints, `dF/d mu = nu_tilde`, `dF/d nu = mu_tilde`, `dF/d xi = eta_tilde`,
    `dF/d eta = xi_tilde` (1e-15 to 3e-15); and `x_tilde = dx/dw` along the seed for all four (1e-10, central
    differences). So the two marks are parallel: `_dot` is the variable varied in the cores along `p`, `_tilde`
    is the variable varied in the samples along the seed. Not checked variable by variable on a manifold frame.
  - Declined: `_res` (new word, breaks the match with the docs' tilde notation, wrong for the Gauss-Newton
    product whose seed is `J p`); `_bar` (same hazard as the tilde, without the familiarity); `_t`, `_z`
    (index-letter registry).
- **The seven site rules are named after the variable they produce, with no direction word:**
  `up` -> `xi_term`, `right` -> `mu_term`, `left` -> `nu_term`, `down` -> `eta_term`, `out` -> `z_term`,
  `grad_G` -> `H_term`, `grad_B` -> `V_term`. Every per-site function takes the suffix `_step`, as the library's
  `_mu_step`, `_xi_step`, `_eta_step` do (my `_site` suffix goes). The assembled gradient blocks are `H_tilde`
  and `V_tilde` (the library's `H_tildes`, `V_tildes`), the two-hole blocks `H_tilde_dot` and `V_tilde_dot`.
  The loop drivers are `left_to_right` and `right_to_left` (as in `tt_zipper_left_to_right`).
  - Why no direction words: the library calls `mu` the "leftward edge variables" (the left environment, built
    from the left cores) while it moves from left to right, so "leftward"/"rightward" can be read either way.
    The maintainer raised this, we compared readings, and he let the proposal stand. In prose: "`mu`, the left
    edge variables, swept left to right".
  - "The seven site rules" stays as the phrase in prose; `_term` is the function suffix because each call is one
    term of a sum.
- **Output names.** Primary names: `f` (data misfit), `r` (residual), `z` (probe output), `g` (gradient, `Jt r`),
  `Jp` (Jacobian-vector product), `JtJp` (Gauss-Newton product), `Kp` (curvature part of the exact product),
  `Hp` (exact Hessian-vector product, `JtJp + Kp`). `Hp` never means Gauss-Newton: the maintainer's concern is
  confusion between the two, not the regularizer (regularization is usually handled by rank truncation).
  `JtJp` was preferred to `HGNp` because it has no `H` and composes with `Jp`.
  - **Aliases, as documented identities and not second spellings in code:** `misfit` = `f`; `z_dot` = `Jp`;
    `w_tilde` = the seed on the sample legs (today's `ztildes`; `w` and `z` are partners on the ambient leg);
    `g` = `(V_tilde, H_tilde)` seeded by `r`; `JtJp` = `(V_tilde, H_tilde)` seeded by `Jp`;
    `Kp` = `(V_tilde_dot, H_tilde_dot)` seeded by `r`. `want` and the keyword arguments accept one spelling per
    variable.
  - **The seed part of D13:** the tilde edge variables that can be wanted or held (`xi_tilde`, `mu_tilde`,
    `nu_tilde`, `eta_tilde`) are always the ones seeded by `r`; they depend on the point and the data only. The
    copies seeded by `Jp` change with every product, stay internal to `JtJp`, and have no public name.
  - **Layering:** at the sampling layer `g`, `JtJp`, `Kp`, `Hp` are the blocks before the gauge projection. The
    projection stays in the geometry and `LocalModel` applies it. The mockup projected inside `evaluate`; that
    goes.
- **Docstring table of `have` for each `want`** (the maintainer's idea, 2026-11-14): `sampling_variables` documents,
  for every variable, what it reads and which of those are worth holding. To be generated from the declared
  graph and checked by a test, so it cannot drift. Blocked for `Kp` and `Hp` on the gauge part of D13 (below).
- **The gauge part of D13: `gamma` (now `nu_L`).** The right edge variable swept through `L` (the base point's own
  cores) is not `nu`. The maintainer's view: it is an auxiliary variable, and since the curvature term reads it too
  it gets its own letter, `gamma`. `nu` means the frame's right edge variable, swept through `R`, and nothing else.
  - `gamma` is read by `z` (the value, decision 7) and by `Kp`. The library's notation for it is `nu_c`, with
    `nu_c = T nu` at a manifold frame (`docs/contributor/curvature_internals.md`); the proposal's `S` for that
    matrix becomes `T`.
  - Is the name free: yes in code (no identifier contains `gamma`). In the docs, lowercase gamma appears only as
    a scalar in one example of `padded_frame_cone.tex`; capital `Gamma_i` is the mode Gram matrix in the sharing
    documents, a different case and a different layer.
  - At a trial point the call uses the point's own cores as the frame, so `gamma` and `nu` coincide there.
  - Confirmed by the maintainer: `gamma` is a public sampling variable (it can be wanted and held; it
    depends on the point and the samples only), and the second helper, the residual-grade variable swept through
    `L` that `Kp` reads, is `gamma_tilde`. The `eta`-like helper that only `z` reads stays internal.
- **`Hp` is not a sampling variable.** `LocalModel._curvature_block` lifts `p` to a raw core direction
  (`geom.lift`), applies the kind's two-hole term to it, subtracts the geometry's tangential curvature, and
  applies the transpose of the lift. So at the sampling layer `Kp` takes the lifted direction (its own argument,
  separate from the `p` of `Jp` and `JtJp`) and returns blocks in raw core coordinates, and `Hp` is assembled by
  `LocalModel` from `JtJp` and `Kp`. Agreed by the maintainer.
- **How the sampling kind is passed** (2026-11-14): a plain string as the first argument,
  `sampling_variables('probe' | 'apply' | 'entries', frame, sample, plan, want=..., ...)`, named `kind`, with an
  integer `order` for the derivative kinds. A string keeps the sampling layer from importing the `SamplingKind`
  classes of the fitting layer. Accepted with the known wrinkle that the fitting layer also calls its
  `SamplingKind` objects `kind`.
  - **One output name for every kind: `z`** (a tuple of `d` arrays for probe, one array per sample for apply and
    entries). `r`, `f`, `g`, `Jp`, `JtJp`, `Kp` are uniform across kinds. `phi` is gone.
  - **The `SamplingKind` protocol is unchanged.** Built-in kinds implement their operations by calling
    `sampling_variables` with their own string; user-written kinds keep working through the existing operations
    and get the stage-1 levels.
  - Evidence (`kind_readout_check.py`): apply and entries reuse the probe case's per-site functions. The only
    function not shared is the gather that makes `xi` for entries; the output is the last `mu`; the seed is one
    scalar per sample entering at the right boundary, and the tilde variables are the plain steps started from
    it. `z`, `Jp` and the transpose match the library to 8e-16 for both kinds. Not checked: `Kp`, jets.
- **Plan vocabulary** (2026-11-14). Three of my words were already taken in the glossary: "block" (the `C` / `W` /
  `K` batch blocks), "level" (a rank level in rank continuation), "store" (a rank store). Replacements and names:
  - `SamplePlan`: the plan (named tuple; the argument is `plan` everywhere; the default has no scans, which is
    today's behaviour). `ExecutionPlan`, `SamplingPlan` and `Plan` were considered.
  - `SampleScan`: one nesting level, with fields `axes` (a tuple of indices into the sample stack) and `hold`
    (variables computed once for the chunk the scan receives, before it scans). Not `SampleSweep`: "sweep" is the
    glossary's noun for the precomputed frame sweep. Decision 2 now reads "a plan is an ordered tuple of scans,
    outermost first".
  - `scan_sites`, `vectorize_sites`: the per-variable site option, tuples of variable names overriding each
    variable's default.
  - `chunk`, `unchunk`: the pad-and-reshape helpers on the sampling kind, beside `take`.
  - In prose: "chunk" for a piece of the sample stack, never "block"; "held" for variables computed once before a
    scan, never "stored" or "kept".
  - Policies keep the library's names, `estimate_chunk_size` and `max_chunk_size_within`, extended to the outer
    chunk. The mockup's `fill` and `balance` go.
- **Fixes from the F4 table that needed no choice:** the gauge matrix `S` becomes `T`; the highest order is spelled
  `order`, not `p`; "messages" becomes "sampling variables"; "center cores" is reworded in the parked note
  (the glossary's terms are "carriers" and "centers"); the mockup's `sweep` and `sitewise` drivers become
  `left_to_right`, `right_to_left` and an each-site driver.
- **Form:** the proposal's name table becomes an enumerated token map (old name -> new name), checked by script
  against the glossary's symbol table, following the store rename's ruling of 2026-11-11.
- Still open, with the two API cases of phase 1: under the adapter case the public names (`sigma`, `tau`, `*_tilde`)
  stay canonical by the glossary's rule and the names above are a second, internal vocabulary; making them the
  only vocabulary is a breaking rename.

## 9. The remaining findings (triage, 2026-11-14)

All 32 findings not discussed one by one were sorted. Resolved by decisions 1 to 8: D18 (the declared per-kind
graph of explicit functions is the only layer; the generic program is the twin), F5 (one contraction per term;
no association option), F6 (the plan holds names and axes, not sizes, so keys are structural). Fixes needing no
decision, for the revision pass: D3, D4, D5, D6, D8, D14, D16, D19, D20, D21, D23, F10 (listed as an open
implementation item: where the `W` axes sit in each held variable on each layout), F12, F13, F14, F15, F16, F18
(structural slices gated by the bitwise battery, numerical changes each in their own slice), F19, F22.

Adopted as the library does it, agreed by the maintainer:

- **Tails (D7, D21, F12).** The network's boundary vectors are ones, sized by the tails of whichever cores are
  passed. Default frames have squashed tails of size 1; a point's own cores may not. Tail ranks above 1 already
  work in the current kernels, so the requirement is "must not regress"; the prototype's tails check goes into
  the package.
- **Start values of the sweeps** (the maintainer's clarification). The boundary vector is ones, but a sweep
  starts from its own variable's value on the boundary bond, and these differ: plain sweeps (`mu`, `nu`, `gamma`)
  start from ones; dot sweeps from zeros; tilde sweeps from zeros for probe, and from the seed times the ones
  vector at the right boundary for apply and entries; tilde-dot sweeps from zeros. With jets the plain start is
  ones at order 0 and zero above (the library's `_init_jet`). In the explicit-argument form every start is a
  visible argument of the driver. Starts take the dtype of the sweep's operands (the float32 backlog item, F17).
- **Carriers (D15).** The cores at the ends of the frame are read against a zero carry and must be finite.
- **Unread inputs (D12).** An input that nothing wanted reads is an error, as in `Problem.objective`. Names in
  `want` are validated against the kind's graph before any scan; misspelt keyword arguments fail by themselves.
- **Chunk-size estimators (D9, D10).** An estimator is told which calls it must serve, sizes for the worst of
  them, and confirms at the chosen size from the compiled program's memory analysis.

Decided: **weighted least squares is built into `sampling_variables`** (against D22's suggestion). Inputs `y` and
an optional `weight`; `f = 1/2 |omega * r|^2`; `omega^2 r` seeds `g` and `Kp`; `omega^2 Jp` seeds `JtJp`; `r` stays
the unweighted residual. Reason: fusing `f` and `g` in one pass needs the residual formed inside the chunk scan.
The deferred data-shaped weight (F17) becomes one more per-sample input, chunked like `y`.

Decided on F9 (who reads the per-sample residual), 2026-11-14:

- **`SamplePlan` gets a top-level `hold`**: the names of the variables the model keeps between calls, for the life
  of the model. A scan's `hold` lasts for one chunk. **Default (the lean one):** the kind's frame sweep, plus `r`
  only when the Hessian is exact, because those products read it every time. `lm.residual` stays available and
  is computed on request when `r` is not held. Today's library holds the frame sweep and `r` always.
- **The verbose display gets a summed variable** for the per-mode, per-order sums of squares of the unweighted
  residual, so it needs no per-sample `r`. The maintainer agreed and asked what else the display and the
  diagnostics reduce over the samples; see the list below.
- The trust-region floor reads only the residual's dtype (`_resolution_eps`), so it needs no array.
- **Reductions over the samples** (agreed). Every one in the optimizers and the display, and where it comes from:
  the weighted sum of squares of `r` is `f`; the per-mode, per-order sums of squares of `r` are the new summed
  variable **`r_block_sumsq`** (also used on validation samples, through the point's own cores); the same sums
  for the data, which are the display's normalization, stay `kind.block_sumsq(data)` (they do not involve the
  tensor and are computed once per problem); the weighted sum of squares of the data, for the trust-region
  floor, stays `kind.sumsq(data)`; the weighted sum of squares of `Jp`, which is the Gauss-Newton quadratic read
  after each solve and in the model value, is the new summed variable **`Jp_sumsq`** (today the library forms
  `Jp` for every sample and reduces it).
- **The entries mask moves to the readout** (agreed; amends decision 1). For entries a padded index reads a real
  entry, so `z` and `Jp` are multiplied by the 0/1 mask at the readout. Then `r` is zero at padded samples and
  every sum and seed is right with no further masking. For the gradient this is the arithmetic tested in
  `levels_jax_check.py` (mask on the seed, 7e-16); it has not been rerun in this form.

## 10. The end state must serve a consumer; jets in the explicit form (2026-11-14)

**The maintainer's requirement.** Immediately after the refactor he will profile the downstream consumer
T3Polynomial and tune the new options. So the end state must let a consumer use every addition without rewriting
the optimizers and without changing T3Toolbox. Consequences, accepted:

- Nothing about the wiring is left to the extension-surface question. `LocalModel` is written once, in stage 1,
  against a names-based interface (`f`, `g`, `Jp`, `JtJp`, `Kp`, `r`, `r_block_sumsq`, `Jp_sumsq` under a
  `SamplePlan`). In stage 1 an adapter serves it for every kind through the six existing operations; in stage 2
  the built-in kinds are served by `sampling_variables`. How a kind is marked as served by the engine (at most one
  optional member, default "not served") is decided and built inside the refactor.
- A plan needs a name for the kind's own frame sweep as a unit (`hold=('sweep',)`), meaning `xi`, `mu`, `nu`, `eta`
  for probe and `xi`, `mu` for apply and entries once the engine serves the kind.
- The proposal gets an acceptance section: what a consumer must be able to do with public arguments only.

**What T3Polynomial does** (read at `dbda0de`, not edited). `t3polynomial/fitting.py::_newton_fit` calls the
frontend `t3toolbox.optimizers.newton_cg(geometry, 'probe_derivatives', (ww, pp), data, x0, order=jet_order,
weight=weight, **kwargs)` on the ragged, uniform and uniform-jax (`use_jit=True`) backends, with a sharing
geometry, `padded_from`, validation samples, and `hessian='exact'` on later continuation levels. Every `newton_cg`
option passes through its `newton_kwargs` (a dict, or a callable per level). Today it passes
`{'chunk_size': None}` because `chunk_size='auto'` compiles a program to read its memory analysis, about 4
compiles per Newton solve and again at every level. Its training sample is `ww = [xh] * degree + [omegas]`,
`pp = [vh] * degree + [zeros]`: the same sample vectors at every input mode.

So for acceptance: a `plan=` keyword on the frontend `newton_cg` (reaching `Problem` and `LocalModel`) is enough
for T3Polynomial to tune through `newton_kwargs` with no change to its fit code; chunking of the sample, the data
and the validation pair has to be reachable from that call; it must work on all three backends, with jit, a
sharing geometry, `padded_from`, the exact Hessian and the verbose display. Two things to weigh: automatic sizing
that confirms by compiling has the compile cost T3Polynomial already avoids; and the parked shared-sample-vector
idea applies to this consumer's main fit.

**Jets in the explicit form** (`jet_steps_check.py`). Probe derivatives with residual weights, checked against
the library's own `ProbeDerivativesKind` at orders 0, 2 and 4: `z`, `f`, `r_block_sumsq`, `g`, `Jp`, `Jp_sumsq`,
`JtJp`, `Kp` all agree to 4e-16.

- Every variable gains a leading order axis. The 22 step functions (60 terms) are the order-0 ones under
  tonight's names. Jets change only the seven term functions, where a contraction of jets becomes a truncated
  product over the order axis.
- There is no second set of functions for the tilde grade, because residual jets are stored reversed: the Taylor
  convention of phase 1, used internally. The library's conventions are kept at the boundary: `(w, p)`, `y`, the
  weight `omega[mode, order]`, and `z`, `r`, `Jp` are derivative-normalized as the kind takes and returns them.
  Only held or wanted edge variables with jets are in the internal convention (rules F and R of the
  correspondence). So no user-visible convention has to change. With the library's normalization inside instead,
  forward terms would be convolutions and tilde terms correlations, and the seven terms would not serve every
  grade.
- `Kp` uses `gamma` and `gamma_tilde` as decided, shares `xi`, `mu`, `xi_tilde`, `mu_tilde` with the frame, and
  has its own dot variables for the raw core direction.
- Not covered: the uniform layout, the `K` and `C` stacks, jit, apply and entries with jets, and the order scans.
  The truncated products are plain loops here; in production each term would be a banded two-term recurrence
  when one operand is an input jet and an order scan otherwise, as the library does today, without binomial
  weights. Phase 1's float32 experiment found about 1e-6 in both normalizations up to order 20.

## 11. Pause (2026-11-14, 22:19)

The maintainer stopped the review here: "I feel like we are cutting corners here for expediency. Let's slow
things down and plan everything right." He asked for the proposal document and the handoff files to be updated
with everything discussed so far.

- Not discussed: F17.
- Not done: the rewrite of the proposal documents and of the mockup to these decisions.
- `DESIGN_AFTER_REVIEW.md` section 10.1 lists the decisions that were taken fastest and most need a second look.

## 12. The maintainer's diagnosis of the pause (2026-11-15, 02:32)

The maintainer, after some hours of thought, on why the review felt rushed:

- **"We are relying on preserving old behavior as a crutch, instead of doing the hard work of deciding what stays
  and what goes from first principles reasoning. This is a major and breaking change with a wide blast radius,
  there is no way around that, we must embrace it."**
- **His example: the jets.** Decision 10 kept the library's derivative normalization at the boundary and used
  Taylor coefficients only inside. That splits the difference. "Taylor coefficient is better. It should replace
  the derivatives, and the associated edge variables should be first class citizens like the unjetted edge
  variables. If we designed the library from first principles knowing what we knew now, this is what we would
  have done."
- **"We are focusing too much on the sequencing of phases of the refactor (how we will get it done), before we
  have fully decided on the target (what we want to do)."**

Consequences for everything above this entry: compatibility with today's behaviour is not a reason for a design
choice, and no decision whose stated reason is "as the library does today", "no user-visible change" or a
staging convenience should be treated as settled. The two stages of decision 5 and the adapter of decision 10
are sequencing, and are set aside until the target is decided.

## 13. First target decisions on jets (2026-11-15, 02:51)

- **Derivative mode is one adapter at the outer level of fitting.** The changed objective is a real concern, and
  it is answered there: "The user can ask for the old derivative mode if they like, and we make it happen by
  scaling and weighting. Taylor coefficients becomes the default and is used for everything under the hood."
- **Order 0 as the one-coefficient case: to be assessed, with conditions.** "If we can make this work without
  sacrificing performance for order-0, then we should do it." The maintainer tried this when he first implemented
  jets and could not make it work: there were extra contractions. That was with the dense `trs` tensors; it may
  differ with the recurrences, or fall out of the new framework. The extra order axis must not be created and
  then squeezed; the preferred form never creates it when it is absent, perhaps by the contraction engine
  treating it as a missing group.
  - Fact checked: `contract` already defines an uppercase letter as a group of zero or more axes, and an empty
    group vanishes (`backend/contractions.py`). So an absent order group vanishes in a term that is linear in one
    jet. The products of two jets are where the assessment has to look.
- The maintainer asked for an ordered list of target decisions, without proposed answers: `TARGET_DECISIONS.md`.

## 14. The order-0 assessment (2026-11-15)

Asked for by the maintainer for target decision 2.7. Script: `scripts/review/order0_assessment.py` (needs jax).
The decision itself is still his.

**Result: in the explicit Taylor form, order 0 is exactly the plain program.** Probe kind, ragged cores, one
sample axis, the operations `z`, `Jp` and `g`.

- The same functions, called with order 0, make the same 124 contractions on the same operand and result shapes
  in the same order as a second implementation that knows nothing about jets. The largest rank of any array is 3
  in both, so no order axis exists. Results are bitwise equal on numpy.
- Under jax the two traced programs are identical character for character (287 primitives each).
- At orders 2 and 4 the same functions match the library's `ProbeDerivativesKind` to 4e-16, converting between
  Taylor coefficients and derivatives at the comparison only.

**How.** Three things, none of them in the step functions' terms:

- The order is a group (`T` in the prototype; the letter is a placeholder) in each term's one literal string.
  `contract` treats an uppercase letter as a group of zero or more axes, so at order 0 it vanishes. At the two
  terms where the operand ranks cannot pin the split (`xi_term`, `z_term`), `len_T` is passed, as the engine
  requires.
- Each product term carries the order group on one operand and takes one coefficient of the other:
  `'TWa,aib,Wi->TWb'`. A driver loops over the other operand's coefficients and shifts.
- Four small helpers (`convolve`, `top2`, `top3`, `start`) branch on the static `order`: at order 0 they make one
  direct call and build starts without an order axis.

**Costs.** `order` is a static argument of every step function (or bound once). The branch is on a static
integer, not on which data is present. The input jet is `w` alone at order 0 and a stacked `(w, p)` above, so the
form of the sample depends on the order.

**Today's library, for comparison** (the plain probe kind against `ProbeDerivativesKind(order=0)`, traced on jax):

| Operation | `dot_general`, plain | `dot_general`, order-0 jets | All primitives, plain | All primitives, order-0 jets |
| --- | --- | --- | --- | --- |
| `z` | 32 | 36 | 38 | 122 |
| `Jp` | 112 | 116 | 160 | 396 |
| `g` | 112 | 112 | 160 | 588 |

The order-0 jet program carries an order axis of length 1. The extra contractions number one per mode (4 with
`d = 4`, 6 with `d = 6`), and `p` is not read. The rest is concatenations, multiplications and additions from
the recurrences. XLA scratch on this 5-sample problem was 14 to 26% higher; that is a tiny problem and is not
evidence about scale. This agrees with the maintainer's recollection that order 0 through the jet code cost
extra contractions, and shows it is still so with the recurrences.

**Not assessed.** Orders above 0 were checked for correctness only: the drivers are unrolled Python loops with
functional updates (304 and 452 contractions at orders 2 and 4 against 124 at order 0). The banded two-term
form and the order scan of target decision 2.9 were not built, so nothing here bears on performance above order
0. Not covered: the uniform layout, the `K` and `C` stacks, several sample axes, apply and entries, the tilde-dot
grade, and whether one contraction per term is itself the best plain program.

## 15. Scope of the target (2026-11-15, 03:19)

The maintainer's answers to target decisions 1.1 to 1.3. "They are not hard answers but more principles to help
guide judgement."

- **1.1, what may be redesigned.** "Redesign may be considered for all areas downstream of sampling, including the
  optimization and fitting layer, and frontend and backend *sampling* functions. Things that are unrelated to
  sampling, such as t3svd or orthogonalization or adding two T3s, are off limits. Modification of data structures
  for the flagship basic types (T3s, frames, tangent vectors, and their uniform twins) is off limits."
- **1.2, what is fixed.** The ragged and uniform layouts, the numpy and jax backends, the `W`, `K` and `C` stacks,
  the frame's four core families, and Tucker-factor sharing are fixed inputs, "with one minor quasi-exception,
  which is the way sharing is handled. Sharing as a feature must be preserved in full, and I can't think of a
  better way to handle it. However, if there is another way to handle sharing, with no loss of functionality or
  performance, then we can consider it. The other items are fixed completely."
- **1.3, ranking of criteria.** "When criteria conflict, there is no hard rule that allows us to avoid thoughtful
  consideration on a case-by-case basis."

Consequences recorded in `TARGET_DECISIONS.md`: the frame's data structure and four families are fixed, so a frame
cannot be given the center cores and no other frame convention is on the table (3.2, 4.1); both layouts and all
three stacks must be served (5.2, 5.4); none of the existing public sampling names has to remain (6.3); the
fitting layer and the optimizers are in scope (section 8). Asked back: whether the geometry protocol (frame
construction, projection, lift, retraction) is in scope.

## 16. The geometry boundary, and merging the plain and jetted frontend functions (2026-11-15, 03:38)

**Geometry.** "Basic frame construction, projection, lift, and retraction are off limits. The geometry singleton
pattern for organizing these functions for use in optimization and fitting is on the table in principle, though
I'm not sure if we can improve upon it. The decision to split things that have to do with a specific tangent
vector into the tangent vector class, and things that have to do with the geometry in general into the geometry,
is a split I would like to preserve, even if geometry is changed." The singleton pattern is not his style; he
tried simpler ways and "always ran into chains of mutual contradiction. Some approaches forced recompiling things
too often. Others didn't respect the frontend/backend split. Others couldn't support both numpy and jax
seamlessly."

- Corrected by the maintainer at 03:49: "off limits" means changing the signature or behaviour of an existing
  function. "We can fuse retraction with frame construction as a new additive feature." So every route to the
  value stays open. Nothing is proposed for the projection or the lift; the question (target decision 4.5) is
  only which layer calls them, and the suggestion to confirm is that they stay in the geometry and are called by
  the fitting layer. Any change to the organization has to avoid the three failures he lists.

**Merged frontend functions (new target decision 6.6).** "One goal I had previously was to merge the sample and
jetted sample frontend functions so that they would dispatch to either jetted or unjetted versions based on the
input types. E.g., just have TuckerTensorTrain.probe, but if the user supplies jetted data as input, it does the
jetted version of the function ... We should consider this again if it can be done cleanly. If it requires
complicated hackery or is awkward then we should not force it. One issue was related to the batching and stacking
axes. How do we tell whether an axis is part of the batch, or a derivative order axis."

- The library records `probe(..., order=)` as a locked rejection (`deferred_and_rejected.md`): return-rank
  polymorphism, the shape-comment-as-contract style, the transposes' residual input rank, and the
  self-documenting `*_derivatives` names.
- Claude's first thought, offered to the maintainer and not decided: treat the order as a fourth batch block,
  outermost, with zero axes at order 0 and one axis otherwise. It is declared by `order` on the functions that
  create it and recovered by pairing with the sample on the functions that consume it, as `K` is recovered from
  the pairing of variations with a frame. Input jets are two arrays `(w, p)` and never carry it. This is the
  frontend counterpart of the empty order group of the order-0 assessment (entry 14). Not prototyped.

## 17. Input jets are affine; where the order axis is (2026-11-15, 03:49)

- **Target decision 2.6, decided.** "We are interested in Taylor series in s of functions of the form f(w+sp). We
  are not supporting things like f(w + sp + tq) like in my other work." The other kind Claude had in mind was a
  curved path in the same single variable, `w + s p + s^2 q + ...`; there is no use for it. So an input jet is a
  point and one direction, an input never carries an order axis, and the two-term form of a product with an
  input jet is part of the structure.
- **The order axis does not move.** The maintainer: "The location of each axis was carefully considered." Checked
  in the library: the derivative methods return `(order+1,) + W + K + C + ...`, order outermost
  (`docs/entries_apply_probe.md`, `tucker_tensor_train.py`), and the uniform packed layout is
  `(d,) + (order+1,) + W + C + (N,)`, order at axis 1 after the site axis (`backend/uniform_fitting.py`). The idea
  under discussion for 6.6 keeps it exactly there when the order is above 0 and omits it at order 0, as the plain
  methods do today.

## 18. The order axis as an optional axis (2026-11-15, 03:55)

- **Target decision 2.7, decided in structure.** "I support the idea of allowing it to be a grouped axis which can
  be either empty or have one entry, so from a structural point of view it acts like the batching and stacking
  axes." The cases the order-0 assessment did not cover still have to be shown to behave the same way.
- **Terminology.** "I would like to be careful about the terminology, since batching axes have a different
  meaning (a bunch of the same thing bundled together), and also have less restrictions." Claude had called it a
  "fourth batch block"; that phrase is withdrawn. The maintainer's suggestion: **"optional axis"**. The phrase is
  unused in the library. It names the restriction exactly: zero axes or one, where a batch block may have any
  number.
- **The letter: `T`** (decided at 03:59). `contract` needs an uppercase letter for anything that can be absent.
  Today's strings use only `C`, `W` and `K` in uppercase, and lowercase `t` for the order axis. `T` is the
  optional order axis; lowercase `t` stays for the order axis where it is always present.

## 19. One frontend function for the plain and the jetted computation (2026-11-15, 04:04)

- **The request.** "Can you prototype the function signature for a merged plain and jetted frontend function and
  show me how it looks?"
- **The prototype** (`scripts/review/merged_frontend_prototype.py`). `probe(ww, pp=None, order=0)` returning
  `len=d, elm_shape=T+W+C+(Ni,)`, with `T = ()` at order 0 and `(order+1,)` above; and
  `probe_corewise_transpose(ztildes, ww, pp=None, order=0, sum_over_probes=False)` with residual
  `T+W+C+(Ni,)`. The bodies wrap the library's existing methods and scale by `1/t!`. Checked: order 0 is bitwise
  `x.probe(ww)` for `W = ()`, `(5,)`, `(2, 5)`; order 3 times `t!` equals `x.probe_derivatives`; the transpose is
  the adjoint of `probe` at orders 0 and 3 (1e-10 by central differences, with the tensor seeded so the run repeats); the three wrong calls each raise a
  `ValueError` that says what is wrong.
- **What Claude flagged.** `pp` and `order` carry one fact between them (`pp` given exactly when `order >= 1`)
  and neither can be removed; the return comment needs `T`, which is the rank that varies with an argument; the
  name `probe_derivatives` disappears, so the old rejection's point about discoverability is not answered.
- **Target decision 6.6, decided.** The maintainer: "Perfect. This is good for a frontend method of the flagship
  classes, and a backend function which uses the engine under the hood."
- **Not decided by this.** 2.8 (three fitting kinds with an `order`, in place of six) and 6.3 (which backend
  functions exist) are still open. Claude takes the pattern to cover `apply`, `entries`, the transposes, the
  `T3Tangent` methods and the uniform twins; the maintainer has not said so in those words.

## 20. Three kinds with an order (2026-11-15, 04:11)

- **Target decision 2.8, decided.** Claude, after 6.6: the frontend now has one function with an `order`, but
  whether the fitting layer has three kinds with an `order`, in place of six, was not decided. The maintainer:
  "Yes, let's make the fitting layer consistent."
- **What follows.** The kinds are `probe`, `apply` and `entries`; the `*Derivatives` kinds and their strings go;
  the order is a separate integer, 0 for the plain computation. T3Polynomial's `'probe_derivatives'` becomes
  `'probe'` with the `order` it already passes.
- **Not decided by this.** What a kind is (3.4), and how the fitting functions take the direction `pp`: today the
  sample is `ww` for a plain kind and `(ww, pp)` for a derivative kind (6.2, 8.1).

## 21. How the direction is passed; the comment on `order` (2026-11-15, 04:18 to 04:22)

- **The comment.** The maintainer: "in the prototype function signature comment, we should relate the input
  argument order to the output shape T", and `T=(order+1)` "does not work in the edge case order=0, because then
  T=() rather than (1). Is there an easy way to say it?" Claude's form, now in the prototype:
  `order: int = 0,  # highest Taylor order.  T=(order+1,) if order else ()`, with the return line reading
  `elm_shape=T+W+C+(Ni,)`.
- **The sample.** Claude had asked whether the fitting functions take `pp` as its own argument or keep one
  bundled sample. The maintainer: "Can it be of type Union[ww_type, Tuple[ww_type, pp_type]] and we dispatch
  accordingly? Maybe there is a subtlety I am missing." Two subtleties: a pair does not give the order, so `order`
  stays; and the type alone is ambiguous (a ragged `ww` with `d = 2` and a uniform `(ww, pp)` are both 2-tuples of
  arrays). Checked in the library: the fitting functions already document
  `sample: typ.Any,  # ww / index, or (ww, pp) / (index, pp) for derivatives`, and the kind declares which.
- **Decided.** "Ok, agree with this recommendation." The Union is the type of the sample; `order` decides, never
  the type. The fitting functions take the bundled sample. The frontend sampling methods take `pp` as its own
  argument.
- **Claude's reading, not the maintainer's words.** `sampling_variables` takes the bundled sample; the backend
  function behind each frontend method takes `ww` and `pp` separately.

## 22. Approvals; what a tilde jet is (2026-11-15, 04:26)

- **Approved.** "Yes, all of these decisions are approved, including the two ones which I did not say." The two:
  `sampling_variables` takes the bundled sample while the backend function behind each frontend method takes `ww`
  and `pp` separately; and the comment `T=(order+1,) if order else ()` on the `order` line. The earlier reading of
  the reach of 6.6 (entry 19) was not in that message and stays marked as Claude's reading.
- **Target decision 2.4, the maintainer.** "For the tilde jet. I have not figured that out yet. Maybe some insight
  can be gained from the theory of convolutions of formal polynomials."
- **Claude's assessment** (`scripts/review/tilde_jet_pairing_check.py`; the full note is under 2.4 in
  `TARGET_DECISIONS.md`). Jets of order `n` are the ring `R[s] / (s^(n+1))`. Its pairing
  `beta(a, b) = [s^n](a b)` is nondegenerate and invariant, `beta(a b, c) = beta(a, b c)`, so under it
  multiplication by a jet is its own transpose. Candidate definition: a tilde jet is a jet, acting on a plain jet
  by the top coefficient of their product; slot `t` is the sensitivity to the coefficient `t` orders below the
  top. A natural-order residual is a polynomial in `1/s` (the dual of the ring), on which jets act by correlation;
  reversal is multiplication by `s^n`, the isomorphism of the ring with its dual. The pairing shows only at the
  seed (one reversal) and at the readouts (a top coefficient). In derivative coordinates the plain flip is not the
  transpose; it needs the weights `t!(n-t)!`.
- **Status.** 2.4 is marked Assessed, not decided.

## 23. The tilde jet is held (2026-11-15, 04:44)

- **The maintainer.** "Let's hold off on deciding this for now. It hinges on whether we can come up with a more
  natural meaning where the reversed order follows naturally. My lean is to keep the residual in forward order at
  the API seam." 2.4 stays Assessed.
- **A second reading, offered by Claude and not decided.** A tilde jet is the Taylor jet in `s` of the plain tilde
  variable, with the plain transpose run at `w + s p` and seeded by the polynomial `c(s) = sum_t c[t] s^(n-t)`;
  the gradient is the coefficient of `s^n`. It follows from `z[t] = [s^n] (s^(n-t) z(s))`. The residual stays in
  forward order at the seam. Checked in part 6 of `tilde_jet_pairing_check.py`.

## 24. A tilde variable is a functional on jets (2026-11-15, 04:51)

- **The maintainer.** "Is it a jet or some kind of dual functional on jets? Maybe that is where the reversal is
  coming from."
- **Claude's answer, not a decision.** A functional, and that is the source of the reversal. Written in the dual
  basis it is in forward order (slot `t` is `d f / d z[t]`) and the ring acts on it by correlation. Written by its
  representing jet under the top-coefficient form it is reversed (`delta_t` is represented by `s^(n-t)`) and the
  ring acts by the product. The dot product on coefficients identifies the dual with the ring too, but is not
  compatible with multiplication; the compatible identification is the flip. Order 0 hides all of this. The
  "second reading" of entry 23 is this representation, with the reversal moved into `c(s)`. The full note is under
  2.4 in `TARGET_DECISIONS.md`. 2.4 stays Assessed and held.

## 25. The type catalog (2026-11-15, 04:57)

- **The maintainer.** "Let's catalog the types for all edge variables (jet vs dual jet), where are inner products
  vs dual pairings, and where any Riesz maps are occurring."
- **The catalog** is `TYPE_CATALOG.md`; its check is `scripts/review/type_catalog_check.py`. Claude's analysis,
  not a decision. The summary is under 2.4 in `TARGET_DECISIONS.md`.
- **The check.** Arbitrary invertible changes of basis of all five network spaces: all 30 variables transform as
  catalogued (1e-15). The seed is the one place that needs an inner product (wrong by 7e-01 under a
  non-orthogonal change of the mode space, right to 1e-15 under an orthogonal one). Under `s -> c s`, jets scale
  by `c^t`, forward-order functionals by `c^-t`, stored tilde variables by `c^-n c^t`, and the blocks not at all.

## 26. Arrows on the edges (2026-11-15, 05:20)

- **The maintainer's picture.** Arrows on each edge; an arrow is contracted into a core only if it points towards
  it; on internal edges an object is neither plain nor dual; the distinction is definite only at the exterior.
  `w` points in; `z` and `r` point out; `r` must be converted by a Riesz map: the weights, and for jets also the
  reversal. "Does this make any sense?" His words are quoted in full in section 8 of `TYPE_CATALOG.md`.
- **Claude's answer: yes.** It corrects the catalog (vector and covector on an internal edge are a convention).
  The order index is one more leg, with the ring product as its core. The weights come from the misfit and the
  reversal from the network's rule for combining orders; together they represent the misfit's functional in the
  network's own pairing. A bond never needs an arrow turned inside because its core has three term functions, one
  per orientation; the order leg has one, so either the arrow is turned at the seed or the order leg gets its
  second orientation. 2.4 stays held.

## 27. Arrows on the cores; a type checker (2026-11-15, 05:26)

- **The maintainer.** "Maybe this can make more sense if the frame cores are also given orientations. For example,
  L cores Laib have a: left-to-right arrow, i: up arrow, b: left-to-right arrow, and R cores Raib have
  right-to-left instead for a,b. Then every contraction can be type checked."
- **Done** in `scripts/review/arrow_type_check.py`; written up in section 9 of `TYPE_CATALOG.md`. Each leg carries
  its edge and a sign; a contraction needs the same edge and opposite signs. The seven term functions were
  replaced by typed ones and the 22 step functions run unchanged: 424 contractions and 176 sums, no error, and
  the numbers are those of the untyped program.
- **Found.** Plain and tilde frame variables flow with the arrows; `gamma`, `z` and the dot variables against.
  Each block fits its hole. The arrows are the orthogonality conditions of the frame cores. Seven wrong calls are
  rejected, among them the residual put back as it is (wrong on the mode edge) and the weighted residual with
  the order not reversed (wrong on the order leg).
- **Two remarks made to the maintainer.** His rule "arrows can only be contracted into a core if they point
  towards it" describes the flow; with fixed arrows on the cores the typing rule is the two ends of one arrow
  (`gamma` enters `L` where `L`'s arrow leaves). And the library's names say where a store sits, while its arrow
  points the other way; the CHANGELOG records that "up" and "down" once named the direction of the edges.

## 28. The convention for the arrows; forms on edges (2026-11-15, 05:39)

- **Agreed.** "I agree with the first rule change, this is right." The typing rule is the two ends of one arrow.
- **The maintainer's lean.** "inward pointing arrows towards a core should mean dual indices (contracting puts a
  vector 'in'), and outward facing arrows from a core should mean plain (contracting pairs the mode with an
  external functional). What do you think? Is this conventional?"
- **Claude's answer.** Agree. It is the map convention (the input index is dual), it is what the checker's signs
  already mean, and it gives the gauge conditions one form, `M M^T = I` along the arrows, checked on the library's
  four orthogonalization methods (`scripts/review/gauge_arrow_check.py`, all below 1e-15). Conventional in index
  notation and in TensorKit.jl's reading of tensors as maps; not universal, since TeNPy's documentation labels
  an ingoing leg as the vector space and an outgoing one as the dual. Suggested: keep the four direction words
  for the names and describe arrows relative to the hole. Not yet recorded as decided.
- **Parked for another arc.** The maintainer's remark on diagonal weight matrices between cores as
  (pseudo-)inner products, hidden identities on a plain T3, and orthogonality with respect to weighted inner
  products.

## 29. The arrow convention decided; direction words stay (2026-11-15, 05:47)

- **Target decision 9.5, decided.** "All good except the name suggestion." An arrow shows the way a vector
  travels: a plain index where it leaves an array, a dual index where it enters. A contraction joins the two ends
  of one arrow. Every arrow of a frame points away from its store, towards the hole.
- **The name suggestion, rejected.** Claude had suggested keeping the direction words for the names of the stores
  and describing arrows relative to the hole. The maintainer: "Describing directionality with respect to diagrams
  (ascii in the documentation here and tikz in papers) is immensely valuable to humans, even if there is the risk
  of misinterpreting or switching it (as I have already done). It's a risk that is worth it in my opinion."
- **The wording** (05:49). "For directionality, we can use 'left-to-right' and 'right-to-left' rather than
  rightward/leftward to avoid confusion with stores." For the vertical direction Claude proposed
  "bottom-to-top" and "top-to-bottom"; the maintainer: "confirmed." (05:50)

## 30. Pause (2026-11-15, 05:53)

- **The maintainer.** "Let's update the handoff files and pause here. We will pick back up tomorrow. Thanks for
  talking through this slowly; this has really clarified things for me. It's nice to see everything come together
  in a coherent mathematical framework."
- **State.** The section "Where this stands" at the top of `TARGET_DECISIONS.md`. Nothing in T3Toolbox or
  T3Polynomial was changed in this session.

## 31. Resumed; what remains, in groups (2026-11-15, 15:34)

- **The maintainer.** "Let's continue. Can you group the remaining open decisions and show me an overview of what
  remains?"
- **Done.** 35 of the 54 rows remain: 14 open, 9 reopened, 11 to confirm, 1 held. Seven groups: A jets and the
  misfit (7), B the frame and the value (6), C what a kind is and the public surface (8), D one program for every
  layout and stack (5), E plans, memory and scale (5), F the fitting layer above (3), G what defines correct (1).
  The section "What remains, in groups" of `TARGET_DECISIONS.md`. The grouping is Claude's and decides nothing.

## 32. Group A begun: the tilde jet; arrows of the base point (2026-11-15, 15:49)

- **The maintainer.** "Let's start with A and go item by item." On 2.4: "I think we have a clear picture now. It is
  an adjoint variable with respect to a Lagrangian, and the reversal of order within the jet comes in because it
  is a dual object which points the opposite way on its edge. Are there any more subtleties here I'm not seeing?"
- **Claude's six points** (section 11 of `TYPE_CATALOG.md`): the opposite way is on the order leg only; the
  Lagrangian on the manifold frame is that of the linear program for `Jp`, so a tilde variable is the multiplier
  of the dot of its partner; the seam functional and the turned seed are two arrays under one name; a tilde jet
  is tied to the order; the weights are indexed in forward order; the tilde of an affine jet is not affine.
- **New check.** `scripts/review/tilde_multiplier_check.py`, order 3: each dot variable perturbed, `<phi, Jp>`
  changes by the top-coefficient pairing with its tilde variable (5e-15 to 1e-11); slot by slot is wrong.
- **Arrows of the base point.** The maintainer's lean: a gauge convention, bottom-to-top and left-to-right, with
  the last `L` and first `R` core as the known exception to gauge correctness, "inward pointing like H". Claude
  agrees with the convention; the two cores are typed like the others and fail only `M M^T = I` (off by 2e+03,
  now in `gauge_arrow_check.py`); "inward pointing" is exact when the tail has dimension 1.
- **Status.** 2.4 and the base-point convention await his confirmation.

## 33. The residual is of the type of `z`; the misfit dualizes (2026-11-15, 16:04)

- **The maintainer.** "My thinking is that the residual is the same type of object as the output z ... the
  residual is r = z_true - f(w). We can expect the user to supply z_true in the same order as if it had been
  generated by an f-like function. The dualization comes from the misfit function ||r||^2 = <r,r> (norm which is
  weighted on base vector and has the appropriate inner product on jet involving the reversal). Does this comport
  with your suggestions, and if not, what are the differences?"
- **Claude's answer.** It comports. Three precisions: the inner product of the misfit is slot by slot and has no
  reversal (the reversed form is not positive); the reversal is in the map from `r` to the seed. The library's
  sign is `r = f(w) - z_true` (`backend/optimizers.py`: "r = S(x) - data"). There are two seams: fitting, where
  only data and weight cross; and the frontend transposes, whose argument is of the type of `z` and is dualized
  inside by the plain inner product and the turn. 2.4 still awaits his confirmation.

## 34. Target decision 2.4 decided (2026-11-15, 16:10)

- **The maintainer**, on the three precisions: "1. Agree. 2. No preference of z-f(w) vs f(w)-z. Keep the libraries
  sign convention, whatever it is. If the library is inconsistent, you can use the convention from your
  prototype. 3. Ok."
- **Decided.** The data and the residual are of the type of `z`, in forward order. The misfit is the weighted sum
  of squares slot by slot. The dualization is one step at the seed, the weights then the turn, and tilde variables
  are stored turned, held ones included. `w_tilde` names the seed. The sign is `r = S(x) - data`; the library is
  consistent about it, and the prototypes agree.
- **Follows.** The first half of 2.5 (what the weight multiplies) is settled. `ztildes` is a misnomer for the
  transposes' argument, left to the naming questions. 34 rows remain.
- **The base-point arrows.** Recorded under 9.5 as the maintainer's convention ("I think so"), with Claude's
  wording of the exception marked as such.

## 35. The misfit: guiding ideas, and the consumer's record (2026-11-15, 16:19)

- **2.5 opened.** Claude recommended ones on the Taylor coefficients as the default weight and a weight shaped
  `(d,) + T`. The maintainer did not answer; he asked for context first.
- **His three guiding ideas.** The aim is to approximate a function with a polynomial. "The different terms in the
  polynomial depend on the length-scale in the problem; there is no a-priori way to say which is more important
  unless a length scale is given, and a natural choice of length scale is 1.0." And "the empirical observation
  that normalizing to try to equalize the influence of each term in the objective seems to help the fit a lot."
  The current approach in T3Polynomial "should not be taken as an edict, but it can be a piece of evidence".
- **The survey.** `T3POLYNOMIAL_SCALING_SURVEY.md`, from T3Polynomial at `dbda0de` with its full history (172
  commits; the clone was deepened, nothing changed) and T3Toolbox's `dev/data_shaped_residual_weight_reviews.md`.
  How it is done now, the history in order, what the history separates (the loss, the conditioning, the score),
  and six observations for 2.5.
- **Status.** 2.5 is open. No decision.

## 36. Hooks for preparation; the default (2026-11-15, 16:33)

- **The maintainer's lean.** Standalone functions that prepare the data and create weights, which the user can
  call before the fitting functions; and "nicety kwarg options which do it for the user within the fitting, by
  calling the preparation functions from within." On the default: "I am torn between do nothing, and do
  everything." Nothing: the fit sits in a rank continuation loop and data should be prepared once. Everything:
  the fit "just works" for a non-expert whose raw data would otherwise "fail badly".
- **Claude's analysis** (under 2.5 in `TARGET_DECISIONS.md`). Three classes of preparation: loss-only, those that
  change the unknowns, and those that belong to the problem. Only the first may be done inside a fit. For the
  misfit "everything" reduces to the equalizing weight. Suggested: a layered default, the backend doing nothing
  and the frontend equalizing when no weight is given, with an explicit weight always winning. The hazard is a
  zero or nearly zero block. The adapter of 2.3 has the same shape.
- **Status.** Awaiting his choice.

## 37. The default weight is ones (2026-11-15, 16:49)

- **Decided.** "Let's go with do-nothing (ones) by default, with an asterisk that the default is subject to change
  in a later release if and when we figure out a more robust 'just works' method." At both layers. Claude's
  layered default (the frontend equalizing) was not taken.
- **Still open in 2.5.** The shape of the weight under one kind.
- **Not decided, carried forward.** The architecture of standalone preparation functions with opt-in keywords
  (the maintainer's lean); the rule that a keyword inside a fit does only loss-only preparation; the rule for a
  zero block in the equalizing function.

## 38. 2.5 closed; the preparation architecture decided; the equalizing weight (2026-11-15, 17:00)

- **The maintainer.** "Agree with architecture and keyword-inside-fit rule, and shape weight proposal."
- **Decided.** 2.5: the weight is `(d,) + T` for probe and `T` for apply and entries, with no rule for a bare
  vector (the default of ones was decided at 16:49). New row 6.7: standalone functions prepare data and create
  weights; fitting functions have opt-in keywords that call them; a keyword inside a fit does only loss-only
  preparation. 33 rows remain.
- **Unanswered.** Whether a mode dimension of 1 is accepted and broadcast. The exact shape is taken as the
  reading.
- **The zero block.** The maintainer: "an open research question"; for a default, "a filtered inverse like
  1/max(x, floor)" with a user-specified floor defaulting to about 1e-6; his concern that with a bad length scale
  the norms legitimately vary by more than that; "perhaps the guidance is that one should adjust the length
  scales first, before equalizing". Claude's answer (under 6.7 in `TARGET_DECISIONS.md`): agree, with the floor
  made relative to the largest block; no special case for an exact zero; the guidance holds because rescaling the
  direction `p` first is what makes the floor meaningful. Awaiting his reply.

## 39. The equalizing weight decided; a required example (2026-11-15, 17:04)

- **The maintainer.** "Yes, agree with broadcasting decision (exact shape only), relative floor, and two-function
  split. Also, let's make sure to provide an example with badly scaled data, which shows the bad fit as given,
  and then a good fit after we process the data with these functions."
- **Decided.** The weight's shape is exact. The equalizing weight is two standalone functions: the block norms,
  and their filtered inverse `1 / max(x, floor * max(x))` with a user-set `floor`, 1e-6 by default. No special
  case for an exactly zero block; masking is done by zeroing a weight.
- **Required.** An example with badly scaled data: the bad fit as given, the good fit after processing.
- **A correction by Claude.** For probe, rescaling a mode's input coordinates also rescales that mode's data, so
  it is not loss-neutral. Recorded under 6.7.

## 40. T3Toolbox 2026.5.0; the example prototyped (2026-11-15, 17:10)

- **The maintainer.** "Let's do the prototype now and see how it goes. By the way, in the meantime the other work
  on T3Toolbox completed and the new version is up (2026.5.0). It may be worth updating to that version before we
  move on."
- **2026.5.0.** Fetched into the read-only clone and checked out beside the pinned one (head `309e637`, 17 commits
  after `3dc15d9`). Both check suites reproduce their recorded output against it, apart from the known last-digit
  difference. Nothing in the repository was changed. Its `dev/HANDOFF.md` names this refactor as the next arc and
  says the handoff directory stays untracked while it is being worked on.
- **The example** (`scripts/review/bad_scaling_example.py`). Badly scaled coordinates and directions 100 times too
  short. As given: bad. Equalizing weight only: bad. Coordinates only: bad with 8 samples, good with 200.
  Coordinates with the direction or with the equalizing weight: good. Four seeds at 8 samples gave the same pattern.
- **Finding.** The equalizing weight alone does not rescue the fit; the rescaling of the coordinates decides it,
  and the weight (or the direction) is needed in addition when the higher orders are needed. The example uses
  functions beyond the two decided: a rescaling of the direction, a coordinate rescaling with its inverse, and two estimates.

## 41. "Step" withdrawn; development moves into T3Toolbox (2026-11-15, 17:47)

- **The maintainer.** "What is the step rescaling? Is this the newton step length or something else?"
- **Answer.** Not the Newton step. Claude had used "step" for the direction `p` of the input jet `w + s p`. The
  word is replaced by "direction" in the documents and the prototype (`rescale_direction`); entries 38 to 40 above
  were reworded accordingly.
- **The maintainer.** "Also, since T3Toolbox is now free, let's move our development there. We can put the current
  handoff content in a folder in dev/, including prototyping. When we change the library code itself, we can do
  it on a new branch, then merge it in when it is done."

## 42. Into the repository; a branch from here on (2026-11-15, 18:04)

- **The maintainer**, on what may be written here about the private consumer: "It is ok to include information
  about T3Polynomial in .md documents in T3Toolbox particularly in dev/. Eventually T3Polynomial will be made
  public, probably just before I submit a paper about it." So `T3POLYNOMIAL_SCALING_SURVEY.md` is in this folder.
- **The maintainer**, on where the work lives: "Please update the handoff and push this commit to main. Then future
  commits will be on a branch. The reason for using a branch is the fact that this refactor will be major and
  severely breaking."
- **Done.** This folder was committed to `main` as `dev/graded_messages/`, with the library's `dev/HANDOFF.md`
  updated to point at it. The work continues on the branch `graded-messages`.

## 43. Confirmations; two more preparation functions; the adapter example (2026-11-15, 18:32)

- **The maintainer.** "Agree plan after target is settled. Agree local dev copy on my laptop is old and superseded
  by our work here. Coordinate rescaling and direction rescaling become library functions. For the form of the
  derivative mode adapter: can you show me a minimal example of it being used here in chat?"
- **Recorded.** The plan for after the target is settled, as written in `dev/HANDOFF.md`, is confirmed. The laptop
  copy of the old handoff directory is superseded. The rescaling of the direction and the rescaling of a mode's
  coordinates (with its inverse) are library functions, beside the block norms and the filtered inverse (6.7).
- **2.3.** A minimal example of the adapter was shown in the conversation: the standalone conversions used by hand,
  and the keyword doing the same inside a fit. Not decided.

## 44. The derivative-mode adapter decided (2026-11-15, 18:37)

- **The maintainer.** "Yes, this is good. Regarding the keyword dependent meaning: yes, but let's make sure to
  document clearly. We can have a doctest example illustrating this behavior, showing what happens to the data in
  each case".
- **Target decision 2.3, decided.** Standalone conversions and one opt-in keyword on the fitting functions; with
  the keyword and no weight, the weight is `t!` per order, which reproduces the plain sum of squares of the
  derivative residuals. Required: documentation with a doctest of the three cases.
- **A draft of that doctest** runs in `scripts/review/adapter_doctest_draft.py` (14 examples). 32 rows remain;
  group A has four, all marked to confirm: 2.9, 3.6, 3.7 and 3.3.
- **A slip in the record.** The commit `be509da` carried only the draft and the runner; an edit script had failed
  before it wrote the documents, and the commit message said more than the commit held. The documents followed
  in the next commit.

## 45. How a term is carried out over the orders: a scan (2026-11-15, 19:08)

- **The maintainer.** "I confirm 2.9 as stated. However, I want to implement with a scan rather than a loop.
  Although the 4th order polynomial is my target, there can be other users with different problems who want
  higher order polynomials; for example to get higher accuracy on smaller problems."
- **Target decision 2.9, decided.** As stated, with the scan: the coefficients of a full jet operand are taken by a
  scan. Claude had proposed leaving loop or scan to group D.
- **Checked** in `scripts/review/order_scan_check.py`: the scan needs a shift and a mask in place of slicing; it
  matches the explicit sum at orders 1 to 16 under jit; its traced program has one size at every order, where
  the unrolled loop grows about ninefold from order 1 to 16; it does up to twice the contractions of the
  triangle (1.67 times at order 4). 31 rows remain; group A has three, all marked to confirm.

## 46. The scan reconsidered (2026-11-15, 19:14)

- **The maintainer.** "Oh hold on. We need to think more about the scan. Performance is important. Can you show me
  pseudocode of representative contractions, loop vs scan, showing how the scan doubles the contractions?"
- **Status.** The scan part of 2.9 is open again; the row is marked to confirm. The rest of 2.9 stands.
- **Shown.** Pseudocode of the loop and of the scan with a shift and a mask, and why the second contracts the
  square where the first contracts the triangle. A third form was found and checked: a scan over a static table
  of the needed pairs of orders, `P` at a time. It has a traced program of one size at every order and contracts
  the triangle plus fewer than `P` pairs of padding; it needs two gathers and a scatter-add in each iteration.
  The comparison is in `scripts/review/order_scan_check.py`. Nothing was timed.

## 47. How the library carries out a product of jets today; a correction (2026-11-15, 19:22)

- **The maintainer.** "How does this compare with how the code currently implements the analogous operation for
  convolution of jets?"
- **Read in the library** (`backend/sampling_derivatives.py`, 2026.5.0). An affine operand: one contraction against
  a stacked pair of shifted copies. Two full jets: a scan over the order of one factor; each step contracts one
  order slice with the core and then with all slots of the other jet, which is carried and shifted one slot per
  step. A real scan only on the uniform layout. The stated reason is peak memory, with a measured 14 to 28 times
  smaller peak; "an unrolled loop does not" give it.
- **Correction.** Claude's table of entry 46 counted pairs of orders as if all contractions cost alike. Separating
  the expensive contraction (core with one coefficient) from the cheap one (pairing with the other jet), the
  library's scheme costs 6 percent more than the loop at order 4 with realistic ranks, and the pair-table scan
  Claude recommended costs 2 to 7 times the loop. The recommendation is withdrawn. The library's scheme in Taylor
  form is now a form of its own in `scripts/review/order_scan_check.py`: correct under jit, 34 lines of traced
  program at every order.

## 48. Two forms over the orders, chosen through the plan (2026-11-15, 19:32)

- **The maintainer.** "Let's have two options, controllable by the user through the Plan abstraction. The analog of
  the libraries current scheme, and a loop with slices. The scan is the default. In the plentiful memory regime,
  I can imagine jax's jit compiler may find optimizations which increase speed,if it can see the whole loop."
- **Decided (2.9).** A product of two full jets has two forms, and the plan says which. The default is the scan
  over one operand's coefficients with the other jet carried and shifted one slot per step. The other is the
  loop with slices. "No user option" leaves the row; "no dense order-by-order form" stays. The plan (7.1) gains
  one setting.
- **Claude's readings, recorded as such in the note to 2.9.** One setting for the whole call; a real scan on every
  layout under the default; agreement of the two settings to rounding only; products with an affine operand,
  order 0 and the block readouts do the same contractions under both.
- **Not decided.** The name of the setting.
- **Checked.** `scripts/review/order_scan_check.py`, part 3: the two settings agree under jit at orders 1 to 16 in
  float64 and float32.
- **Counts.** 20 rows decided, 31 remain. Group A has three: 3.6, 3.7, 3.3, all marked to confirm.

## 49. How the scan is written; rules for dispatch (2026-11-15, 19:50)

- **The maintainer.** "The implementation is through the xscan abstraction, which dispatches to a jax scan if jax is used and the
  layer is uniform, and a loop if pure numpy or ragged. In general, things should respect the numpy vs jax dispatch
  rule, and scan type operations should be polymorphic in ragged vs uniform. This may have reasonable exceptions
  for performance optimizations and issues related to masking; though for uniform sampling the pattern is to mask
  first, then compute edge variables with the assumption that masking was already done. For map-type operations,
  uniform and ragged may have different behavior for performance and memory considerations; this is one of the
  things we are helping the user control with the Plan abstraction in this refactor arc. Dispatch at the lowest
  level within reason, use jax if any input is jax and numpy otherwise, write code generically with xscan and
  similar, which are obtained from the get_backend function and may also use the tree_contains_jax function.
  Uniform vs nonuniform is deduced from the input types. Look at other functions in the library for examples, for
  example things in the existing backend/probing.py."
- **A reading of Claude's corrected.** Entry 48 recorded Claude's reading that the default would be a real scan
  on every layout. It is not: the scan is written with `xscan`, a real scan with jax on the uniform layout and
  a Python loop on numpy or on the ragged layout.
- **Recorded.** For 2.9, in its notes. As general rules for how the functions are written, in the notes to
  section 5 of `TARGET_DECISIONS.md`: numpy or jax inferred from the inputs at the lowest level; the layout
  deduced from the input types; scan-type operations generic through `xscan`; mask first on the uniform layout;
  map-type operations may differ between the layouts, under the plan's control; `backend/probing.py` as the
  example. No row changes status; they bear on 5.2, 5.3 and 5.6.
- **Prototype.** `scripts/review/order_scan_check.py` rewritten in the library's pattern and checked on the four
  backends in float64 and float32. It now takes the path of the T3Toolbox checkout.
- **Claude's observation, for 7.2.** Where `xscan` is a Python loop the `'scan'` setting keeps the square of the
  cheap contractions without the scan's two properties, so `'loop'` does less work there and gives up nothing.

## 50. The library read again, not the prototypes (2026-11-15, 20:00)

- **The maintainer.** "I get the impression that after the conversation compacted you forgot some things, and are
  now going to the prototype to get context. However, the prototype is just that; the way the code is written
  there may be what we are going for, or it may be something that was done for expediency and not intended for
  the actual refactor. To address this, please read through backend/probing.py and backend/sampling_derivatives.py
  to get a feel for how the library currently works, and also review the proposal documents for our new execution
  plan feature."
- **Read.** `backend/probing.py` and `backend/sampling_derivatives.py` in full; the scans and maps of
  `backend/common.py`; the head of `backend/ut3_sampling.py`; `docs/contributor/scan_body_principles.md`,
  `docs/contributor/uniform_polymorphism.md`, `docs/chunking.md` (first half), the scanned-kernel item of
  `docs/contributor/deferred_and_rejected.md`; and this folder's `EXECUTION_PLANS.md`, `DESIGN_AFTER_REVIEW.md` and
  entries 1 to 8 of this log.
- **Recorded** in the notes to section 5 of `TARGET_DECISIONS.md`: a table of what each kind of loop is on each
  layout in the library today; how dispatch, masks, bodies and dtypes are handled; and that the two contributor
  notes, not the prototypes, are the reference for the form of the code.
- **What it changes in the records.** The rules of entry 49 were restated from the maintainer's message alone;
  they are now tied to the library's own notes. Decision 4 (no option for orders) is marked superseded by 2.9 in
  `TARGET_DECISIONS.md` and at the top of `DESIGN_AFTER_REVIEW.md`.
- **Found, not decided.** The site option is a choice only on the uniform layout with jax. The settings for
  orders and for sites combine there. Whether a scan over sample chunks is real on ragged cores with jax is a
  question for group E. The explicit prototypes form every tilde variable at every order, where the library
  forms only the two rows of `dxi_tilde` that are read; that bears on 3.3 and 6.4.
- **Prototype.** In `order_scan_check.py` the scan body was defined inside a factory function. It is now a
  module-level function, as the scan-body principles require, and the script says which of its features are the
  library's pattern and which are the prototype's only. Its output is unchanged.


## 51. What "public" means for a variable (2026-11-15, 21:35). Under discussion

- **The maintainer's take.** Edge variables are computed through the engine, by asking for what is wanted and
  supplying what is held, so there is no need for an individual function for each of them. He asked how Claude
  sees it.
- **Claude's answer.** Agreed. Public then means that the engine accepts the name, in `want` and as a held
  keyword; the public functions are the engine and the operations of 6.6. A test for a name being public was
  proposed (a route-independent meaning, one shape, a use outside), with its consequences for `eta_tilde`, for the
  docstring table as the shape contract, and for 3.4 and 6.5. The engine itself has not been prototyped.
- **Recorded** in the notes to section 6 of `TARGET_DECISIONS.md`. No row changes status.

## 52. Public: provide every variable; the shape with jets (2026-11-15, 21:44). Under discussion

- **The maintainer.** Edge variables are mathematical quantities a user may want; he sees no clear place to draw a
  line, "which suggests that we provide all of them". And the shape contract has nuance with jets, perhaps
  covered by the optional group `T`.
- **Claude's answer.** The third part of the proposed test ("a use outside") is withdrawn; every variable of the
  graph at every grade can be wanted and held. A rule is proposed with it: wanting a variable never changes a
  value, and not wanting it lets the engine skip work (the two slots of `eta_tilde`). With `r` holdable, one
  name per tilde variable covers every seed. On shapes: `T` carries it; the affine variables have a two-slot
  group of their own, the tilde grades have a turned order leg, and only plain jets are nested in the order.
- **Recorded** in the notes to section 6 of `TARGET_DECISIONS.md`. No row changes status.

## 53. Public: confirmed. The variables without names (2026-11-15, 21:57 and 21:59)

(Its naming is superseded by entry 57.)

- **Confirmed by the maintainer.** Every variable of the graph can be wanted and held, and wanting never changes
  a value. The affine variables carry a two-slot group in place of `T`.
- **Asked.** A list of the variables that still need names, with suggestions; perhaps recycling freed names such
  as `sigma` and `tau`. And: consistency with the T4S paper's notation is not to be attempted.
- **The list** is in the notes to section 6 of `TARGET_DECISIONS.md`: six variables, all of them variables of the
  corewise frame at the base point seen from a call on a manifold frame. Four are the grades of one missing
  letter (the partner of `gamma`); two are the dots of `xi` and `mu` along the raw core direction, which collide
  with the frame's dots. The raw core direction has no name as an argument either.
- **Found while listing.** The prototype's `eta_tilde_dot` is the tilde-dot of the missing letter, not of `eta`.

## 54. `zeta` decided; the collision in the terms of the type catalog (2026-11-15, 22:11)

(Superseded by entry 57: `zeta` is `eta_D`, and neither option was taken.)

- **Decided.** `zeta` is the down variable through the base point's cores: "zeta is great!"
- **Asked.** How the two options for the dots of `xi` and `mu` along the raw core direction sit within the type
  catalog's naming concepts.
- **Answered** in section 13 of `TYPE_CATALOG.md`, from the checked catalog grouped by edge and direction. The two
  rules of the catalog do not see the collision; it is a difference of edge. Two slots have no letter, the
  vector on an `R` bond and the vector on a `U` edge, and only the frame's tangent variables live there. Option B
  follows the precedent of `nu` and `gamma`; option A marks something that is not a coordinate of the catalog.
  Not decided.

## 55. Naming from first principles: a variable is a contracted subnetwork in a basis (2026-11-15, 22:24, 22:27)

- **The maintainer.** Freedom to explore broader naming concepts, with no precedent to worry about. What is the
  minimal information that identifies a variable? Could it be the core and leg it comes out of (`mu` as `L2`)?
  And: an edge variable as the fully contracted subnetwork on one side of the edge.
- **Analysis** (section 14 of `TYPE_CATALOG.md`). Three things identify a variable: the side contracted (the left
  or the right of a bond, the leaf or the trunk at a Tucker edge), the basis of the cut edge (the base point's or
  the frame's other one), and what was differentiated. 22 of the 32 combinations occur.
- **Found and checked** (`scripts/review/subtree_gauge_check.py`). Along the lifted direction, the frame's dots
  and the base point's dots are the same tangent vector: equal arrays where the basis is the same, and joined by
  the lift's factors where it differs. The same factors join `nu` with `gamma` and `eta` with `zeta`. With slack
  ranks a twin relation holds one way only. With damping the statement is not exact; not examined.
- **The core-and-leg naming** is exact for the seven plain variables and not unique for a dotted one.
- **Two schemes offered.** Eight letters (with `sigma` and `tau` for the two variables that exist only as dots),
  or four letters for the four sides with the store of the basis as a suffix where it is not the usual one.
  Not decided.
- **22:41.** The maintainer asked for the second scheme written out as a table against every current variable
  and the unnamed ones. It is at the end of section 14 of `TYPE_CATALOG.md`, with the library's present names.

## 56. A table of diagrams, one row per variable: prototype (2026-11-15, 23:02)

- **The maintainer**, on the four-letters-and-a-basis scheme: "I think this is the best so far. However, there is
  still great potential for confusion." His proposal: a battery of tensor network diagrams showing the definition
  of each variable, as a subnetwork, for one representative edge in the middle of the network, formatted as a
  table that also shows the shape, such as `mu | mu2: diagram of L0(D0(w0))L1(D1(w1)) | mu's shape`. He asked for a
  new tab with a few of these to look at.
- **Done.** The tab "Variable diagrams" of the Claude doc, with seven rows: `mu`, `nu`, `nu_L`, `xi`, `eta`, and two
  that show one term of a sum, `mu_R_dot` (a direction at one site) and `mu_tilde` (a seed at one input). Each row
  is a drawing with three columns: the name with its side and basis, the subnetwork on five sites with the open
  edge as an arrow, and the shape. The tab also has a key to the drawing and to the letters of the shapes.
- **How it is made.** `scripts/variable_diagrams_gen.py` generates the drawings. Each row is its own drawing at the
  full width of the page, because a drawing inside a table cell would be scaled down below a readable size.
- **Not decided.** The scheme, and the form of the rows; the other fifteen variables are not drawn.
- **23:10, the maintainer:** "This is excellent. There is one modification I would suggest, which is to use greek
  letters and latex-style annotations for tildes dots etc in the diagrams (keep the spelling in the left column).
  In things like nu_L, the L can become a subscript". Done in all seven rows: inside a drawing the open edge is
  labelled with the Greek letter, a tilde or a dot above it, and the basis and the site as subscripts; the cores
  and the inputs have their site as a subscript; the seed is `w` with a tilde. The left column and the sentence
  above each drawing keep the code spelling. Claude's choice, to be corrected if wrong: the site is a subscript
  after the basis, as in `nu` with the subscript `L,3`.

## 57. The notation is decided: four letters and a basis (2026-11-15, 23:23)

- **The maintainer**, on the seven drawings in formula style: "This is excellent and further confirms that this
  notation is the right one. I want to lock in this new notation. We will have to make a lot of changes to this
  proposal and the dev/graded_messages directory."
- **Decided** (target decision 9.6, and with it 3.1). `mu`, `nu`, `xi`, `eta` name the four sides of a cut edge;
  the store of the basis follows the letter where the basis is not the usual one; then the grade. `gamma` is
  `nu_L`, `zeta` is `eta_D`, the frame's `mu_dot` and `xi_dot` are `mu_R_dot` and `xi_U_dot`, and the frame's
  `nu_dot` and `eta_dot` are `nu_L_dot` and `eta_D_dot`.
- **His four steps for bringing the folder and the doc into line.** (1) A page that catalogs the complete
  taxonomy of the sampling variables and their definitions, with a table from the old names to the new. (2) An
  agent reads that page and draws the remaining diagrams, and Claude checks them: "did the opus agent understand
  the notation correctly from what is written in the sample variables page?" (3) An agent surveys the doc and
  this folder and flags where updates are needed. (4) Two agents, one editing and one reviewing, iterate until
  the reviewer is satisfied; old notation ideas and concepts are kept, marked as deprecated, and moved to the
  bottom of the documents or to an archive document.
- **Step 1, done.** `SAMPLE_VARIABLES.md`, and the tab "Sample variables" of the Claude doc. Every definition is
  given as a recurrence and as a subnetwork, and `scripts/review/sample_variables_check.py` checks the
  recurrences against the library (`z`, `g`, `Jp`, `JtJp`, `Kp`) and each subnetwork against its recurrence
  (14 variables, as sums over where the direction and the seed are put).
- **Step 2, done** (2026-11-16, 00:30). An Opus agent, given the page as its only source, drew the other 21 rows
  (fifteen variables, `z`, `Jp` and the four blocks). Claude checked each against the page: all 21 are right.
  The agent listed 17 places where the page was unclear; the page was clarified at each
  (`reviews/notation_2026-11-15/DIAGRAM_TEST.md`). The tab "Variable diagrams" now has 28 rows, grouped by the
  side contracted.
- **Step 3, done** (2026-11-16, 00:45). An Opus agent surveyed the folder and the doc:
  `reviews/notation_2026-11-15/SURVEY.md`, 103 findings.


## 58. The maintainer's answers to "Not decided"; the tilde variables are the library's adjoints (2026-11-16, 00:03)

The maintainer answered four of the six open points of `SAMPLE_VARIABLES.md` section 11, and then stopped:
"Stopping my response here, we need to resolve the tildes issue."

- **The raw core direction as an argument.** "I don't understand what is being asked here. Argument of what
  function? Please explain in more detail." The point was badly put. It is the argument that carries the
  direction of `Kp`: the library's `core_directions`, the first argument of `corewise_hessian`. Section 11 now
  says so.
- **Step functions.** "I would prefer to have separate step functions. It is ok to have lots of small
  functions." Recorded under 5.1.
- **The corewise frame.** His principle, on which he asked for the pros and cons: "The engine is designed for
  the manifold setting. When corewise operations are required, this is done by having the caller of the engine
  put in the B and G cores, and it is their responsibility to track the altered meaning of the variables. The
  engine remains manifold-focused." Not decided.
- **The tilde variables.** On "whether each seed gives the tilde variables a name of their own": "This is making
  me worried that we somehow changed the probing algorithm in this proposal process (rather than just changing
  naming and code structure), and the change may be bad. As I understand it, the tilde variables are adjoint
  variables which arise from pairing the all of the outputs with residual-like vectors. ... Is this the idea that
  we create d different sets of tilde variables, one for each network? If so this is a problem and something has
  gone wrong. There should be only one tilde variable of a type, and it is a variable which combines information
  from all the networks. Please look at how the corresponding adjoint variables are done in the existing
  codebase."
- **Claude's answer: nothing was changed, and his description is exactly what the page's variables are.** There
  is one tilde variable of each kind, one array per bond or site, combining the residuals of all the modes on
  its side. Looked at in the library (`backend/probing.py`): `_tau_tilde_step` carries
  `tau_tilde . L . xi + mu . L . deta_tilde`, which is the page's recurrence for `mu_tilde` term for term; the
  same holds for `compute_deta_tilde`, `compute_sigma_tilde` and `compute_dxi_tilde`. Checked on the arrays
  (part D of `scripts/review/sample_variables_check.py`): `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde` equal
  the library's `deta_tildes`, `tau_tildes`, `sigma_tildes`, `dxi_tildes` to rounding, and each is one array per
  site or bond with no axis over the modes.
- **Where the confusion came from.** "Seed" on the page means the whole residual-like input of the transpose,
  all `d` vectors. "Each seed" meant the two different inputs the transpose is given in a Newton step: the
  weighted residual (for `g`) and the weighted `Jp` (for `JtJp`). The library does the same with two calls of
  `transpose`. Target decision 3.3 asks only which of the two a tilde name refers to when one call computes
  both. The page's section 5.3 and section 11 were rewritten to say this.
- **Step 4 is held** until the maintainer has seen this, since the page is what the editor would carry through
  the folder.

## 59. Step functions, the corewise principle, and the raw core direction (2026-11-16, 00:12 to 00:30)

- **00:12, on the tilde variables** (entry 58): "Ok, understood. The terminology and looking at the diagram
  confused me about the tildes. This is right. You can continue while I weight the other points". Step 4 was
  started. The six plain tilde rows of the "Variable diagrams" tab were reworded: each title now says the
  variable is one array, the sum over its inputs, and the notes say which modes it combines.
- **00:23, step functions.** "Oh, I see. There is no need for extra step functions if they literally do the
  exact same things with different arguments." A step function is named by letter and grade and serves both
  bases (target decision 5.1; `SAMPLE_VARIABLES.md` section 5). This replaces his answer of 00:03.
- **00:23, the corewise principle** of entry 58, after the pros and cons: "The against is more convincing,
  particularly the extra sweep cost. This will be a real performance hit and is unacceptable." The principle is
  set aside. What stands: a call on a corewise frame must not sweep twice. How the engine is told that two
  stores are one, and which spellings such a call accepts, are open (`SAMPLE_VARIABLES.md` section 11).
- **00:23 and 00:30, the raw core direction.** Once the question was explained (the argument that carries the
  direction of `Kp`): "Ahh understood. Any suggestions? When I write it on paper, I often use q=(V',H'). The
  primes probably not a good idea in the code." Claude suggested `core_direction = (D_dot, L_dot)`: the base
  point is `(D, L)`, the argument is the direction those cores move in, and `_dot` already means "varied along
  the direction", so the recurrences read as the product rule. The maintainer: "Ok with D_dot and L_dot." The
  components are decided. The argument's name `core_direction` was part of the suggestion and not spoken of;
  to be corrected if wrong. On the corewise frame `(B, G, G, G)` it is `(B_dot, G_dot)`.
- **00:30:** "While they are working let's slowly go through outstanding open questions".

## 60. The weight inside the engine: 3.6 decided (2026-11-16, 00:56 to 01:04)

- **The question** put by Claude at 00:35: is the loss inside the sampling layer, or supplied from above? The
  proposal: weighted least squares inside, with the diagonal weight of 2.5; everything else about the objective
  above.
- **00:56, the maintainer:** "I'm leaning towards this, but want some more information first. How is it proposed
  for the weight to be supplied? Is it an extra argument the user gives to the engine when they want certain
  variables that require it? Also, the only time this is needed is for f and the fused JtJp computation, correct?
  All other ones, the user can apply the Riesz map themselves outside the engine is this right?" And his
  structure: a public helper that converts between a jet and a dual jet with the weights and the reversal, and
  its inverse; the weights passed to the engine; the engine calling the helper.
- **Claude's answer.** The weight is a keyword of the engine. It is needed inside wherever a seed is made from
  something computed in the same pass: `g` together with `z` or `f`, `JtJp`, and `Kp` unless the seed is held.
  `f` alone would not need it, given the block sums of squares. The inverse helper does not exist at a zero
  weight.
- **01:04, the maintainer decided:** the weight is on `f` too, optional on all of `f`, `g`, `JtJp`, `Kp`, ones by
  default, so that with one weight they are derivatives of each other; the transpose takes no weight; `JtJp` is
  really `Jt W J p` and keeps its name. His words are in the note to 3.6 of `TARGET_DECISIONS.md`.
- **Counts.** 23 rows decided, 29 remain. Group A has two: 3.7 and 3.3. Claude put 3.7 to him at 01:06: `f`
  (weighted), `r_block_sumsq` (unweighted, in the shape of the weight) and `Jp_sumsq` (weighted) as the three
  reductions over the samples that are sampling variables.

## 61. The sample stack on the outputs; the transposes sum by default (2026-11-16, 01:26)

- **The maintainer**, asked 3.7 (which reductions over the samples are sampling variables): "I am getting
  confused and am not sure exactly what is being asked." He then set out his picture, which is recorded word for
  word in the notes to section 3 of `TARGET_DECISIONS.md`:
  - the residual is `z - y`, with the data `y` on the sample stack `W` and broadcast over the base-point stack
    `C` of `z`;
  - **decided:** a transpose that returns cores sums the sample stack by default; the option not to sum stays,
    and then `W` goes into the stacks of the result, never in front of the site index. Today's default is the
    other way round;
  - **decided:** the sampling layer must support the `C` and `K` stacks; the optimization and fitting layers
    may assume there are none.
- **A letter clash he pointed out.** Claude had written the Riesz operator of the weight as `W` in the summary
  of 3.6. `W` is the sample stack. The note to 3.6 now writes `omega^2`.
- **Read in the library at his suggestion:** `docs/batching_and_stacking.md` (the three stacks; section 11 on
  transposes and `sum_over_probes`) and the transposes of `backend/probing.py`.
- **3.7 put again in his terms** (Claude, 01:35): by what happens to `W`, the engine's outputs are of three
  kinds. Kept: `z`, `r`, `Jp` and every edge variable. Summed by default, kept on request: the outputs with the
  shape of cores, `g`, `JtJp`, `Kp`. Always summed, because they are numbers and not arrays over the samples:
  `f`, `r_block_sumsq`, `Jp_sumsq`. 3.7 asks only whether that last list is right.

## 62. 3.7 decided: what happens to the sample stack on each output (2026-11-16, 01:32)

- **The maintainer**, on the table of entry 61: "I understand now. Confirmed. The table with columns "What
  happens to W" and "Outputs" in your message is what I want."
- **Decided.** `W` is kept on `z`, `r`, `Jp` and every edge variable; summed by default and kept on request on
  `g`, `JtJp`, `Kp`; always summed on `f`, `r_block_sumsq`, `Jp_sumsq`. The table and the details are in the
  note to 3.7 of `TARGET_DECISIONS.md`.
- **Counts.** 24 rows decided, 28 remain. Group A has one left: 3.3.

## 63. The notation carried through the folder and the doc: steps 2 to 4 done (2026-11-16, 00:12 to 01:40)

The maintainer's four steps of entry 57. The records are in `reviews/notation_2026-11-15/`.

- **Step 2** (`DIAGRAM_TEST.md`). An agent given only the page drew the other 21 rows; all are right; the page
  was clarified at the 17 places it found unclear.
- **Step 3** (`SURVEY.md`). 103 findings in the folder and the doc.
- **Step 4, the folder.** An editor agent and a reviewer agent, both Opus, in two rounds
  (`EDIT_REPORT_r1.md`, `REVIEW_r1.md`, `EDIT_REPORT_r2.md`, `REVIEW_r2.md`). Round 1: the reviewer found
  every rename in the right basis, every moved passage word for word and no quotation altered, and was not
  satisfied on three statements that were wrong (two in the type catalog's section on the arrows, one pointer
  in the design document) and sixteen smaller points. Round 2: all fixed, `(D_dot, L_dot)` applied, and the
  reviewer satisfied. Claude ran both check runners afterwards: they reproduce their records.
- **What was done to the folder.** The live documents (`TARGET_DECISIONS.md`, `TYPE_CATALOG.md`,
  `DESIGN_AFTER_REVIEW.md`, `README.md`) and the live prototype scripts use the names of
  `SAMPLE_VARIABLES.md`. The naming ideas that the decision overtook are in the new `NOTATION_HISTORY.md`,
  marked deprecated, word for word, with a pointer where each stood. This log, the reviewer reports and the
  documents from before the review keep their words; each has a notice that maps its names.
- **Step 4, the doc** (`DOC_EDIT_REPORT.md`, `DOC_REVIEW_r1.md`). The editor brought every tab that mirrors a
  file in step and added the tab "Notation history"; the reviewer compared each tab with its file word for
  word and was satisfied at the first round. No edit of the maintainer's was overwritten.
- **Known and left:** the notes of the "Target decisions" tab are shorter than the file's in places, from
  before this work.

## 64. 3.3 decided: the tilde names mean the residual's adjoint sweep (2026-11-16, 01:36)

- **The maintainer:** "Proposal confirmed."
- **Decided.** In a call that runs both adjoint sweeps (`JtJp` and `Kp` together), the tilde names refer to the
  sweep seeded by the residual. The sweep seeded by `Jp` has no names there; its variables are reached by a call
  with `Jp` given as the residual. This rests on 6.5, which is open. The details are in the note to 3.3 of
  `TARGET_DECISIONS.md` and in section 5.3 of `SAMPLE_VARIABLES.md`.
- **Counts.** 25 rows decided, 27 remain. Group A (jets and the misfit) is finished. Next by the suggested
  order: group B, the frame and the value (4.1 to 4.5, 3.2).

## 65. The frame and the value: both routes, with optional Tucker centers (2026-11-16, 01:49 to 01:52)

- **The question** put by Claude at 01:40, the root of group B: at an accepted point, is the value read off the
  frame, with the frame carrying the Tucker center cores? Claude's lean was yes, with the centers as a fifth
  member of the frame.
- **The maintainer** asked instead for both routes: the centers used if supplied, the route through `D` if
  not; the library's own optimizers supply them; the retract-and-orthogonalize function returns them. He asked
  whether the orthogonalization already creates the TT and the Tucker centers.
- **Checked:** it does, and the geometry discards them today (note to 4.1 of `TARGET_DECISIONS.md`).
- **Decided at 01:52:** 4.1, the first half of 4.3, and that wrong centers are an error in safe mode. The frame
  stays four stores; the centers are supplied beside it.
- **Counts.** 26 rows decided, 26 remain.
- **Open with it:** the name of the centers; the second half of 4.3; the rest of 4.2.
- **01:54, the name.** The Tucker centers are `Vc` and the TT centers `Hc` (the maintainer: "Agree with Vc,
  Hc."). `SAMPLE_VARIABLES.md` sections 1 and 6 now have them, and its check compares the three routes to the
  value on the library's own frame.

## 66. 4.3 decided: three policies for a trial point, a frame at every trial by default (2026-11-16, 01:59)

- **The maintainer:** "Add all three as options, default to (b)".
- **Decided.** The optimizers offer three policies for a trial point of a line search or a trust region: (a) the
  value through the trial's raw cores, the frame built on acceptance only, as today; (b) retract-and-orthogonalize
  at every trial and the value read off its frame with `Vc`, the trial's plain variables held so that an accepted
  trial needs only the adjoint pass; (c) the same with the gradient computed in the trial's pass, for a plan that
  holds nothing. (b) is the default. The table is in the note to 4.3 of `TARGET_DECISIONS.md`.
- **Counts.** 27 rows decided, 25 remain. Group B has four left: the rest of 4.2, 4.4, 3.2, 4.5.

## 67. 4.2 decided: frame consistency is a precondition in safe mode, checked only where needed, never through a dense tensor (2026-11-16, 02:07)

- **The maintainer's words** are in the note to 4.2 of `TARGET_DECISIONS.md`. In short: a precondition checked
  in safe mode at the moment a function that requires it is called; no runtime safety check through a dense
  tensor; the norm of a difference through the existing non-dense machinery; `is_consistent` rewritten to avoid
  the dense tensor; no check where it is not needed.
- **Found in the library on the way:** `T3Frame.is_consistent` forms two dense tensors and does not look at
  `U`.
- **Counts.** 28 rows decided, 24 remain. Group B has three left: 4.4, 3.2, 4.5.

## 68. 4.4 decided: frames support arbitrary tails, unchecked (2026-11-16, 02:12)

- **The maintainer:** "Frames must support arbitrary tails. Allow and do not check. Functions which require
  squashed tails should be looked at with suspicion."
- **Decided,** with the boundary vector of ones sized by the cores passed, and the start values by grade (note
  to 4.4 of `TARGET_DECISIONS.md`). An audit of the functions that squash or assume squashed tails is noted for
  the branch.
- **Counts.** 29 rows decided, 23 remain. Group B has two left, both to confirm: 3.2 and 4.5.

## 69. Group B finished: 3.2 and 4.5 decided; the tails discipline limited to what the refactor touches (2026-11-16, 02:16)

- **The maintainer:** "This refactor arc is already too big to add a broader squash tails drift audit on top.
  Keep the squash tails discipline to the functions we touch here and we can deal with other ones at a later
  time." And: "3.2: Agree with proposal." "4.5: Agree with proposal."
- **3.2:** `nu_L` stays as an ordinary variable; only `Kp`, and the value through `D`, read it.
- **4.5:** the projection, the lift and the assembly of the exact Hessian product stay in the geometry; the
  engine returns blocks before projection and takes the lifted direction `(D_dot, L_dot)`.
- **Counts.** 31 rows decided, 21 remain. Groups A and B are finished. Next by the suggested order: group C,
  what a kind is and the public surface (3.4, 8.2, 6.2, 6.3, 6.4, 6.5, 9.2).

## 70. 3.4 decided: a kind is closed in the sampling layer; the open extension point is an objective protocol in the fitting layer (2026-11-16, 02:22)

- **The maintainer:** "Agree with split. I have been wanting to do something like this for some time."
- **Decided** as in the note to 3.4 of `TARGET_DECISIONS.md`. Read for it: the library's
  `dev/OPEN_QUESTION_extension_surface.md`, whose four wanted extensions each have a place under the split.
- **Counts.** 32 rows decided, 20 remain. Group C has six left: 8.2, 6.2, 6.3, 6.4, 6.5, 9.2.
- **Raised next by Claude:** forward-only probing, which that file says T3Polynomial needs, as the probe kind
  with only some modes left open: a parameter of the built-in kind.
- **02:28, forward-only probing.** The maintainer: "Leave it off this arc. Nice to have not need to have." It
  is listed among the things outside the list in `TARGET_DECISIONS.md`.

## 71. 6.5 decided: held variables are trusted, shapes are checked (2026-11-16, 02:31)

- **The maintainer:** "Agree with proposal. Check shapes, trust values. Also, I want to leave it open to the user
  to apply the algorithm to whatever they want. There may be situations where applying the sweeping algorithms to
  modified variables could be useful."
- **Decided** as in the note to 6.5 of `TARGET_DECISIONS.md`. Decisions 6.4 (every variable can be held) and 3.3
  (the adjoint variables of `Jp` are reached by giving `Jp` as the residual) rested on this.
- **Counts.** 33 rows decided, 19 remain. Group C has five left: 8.2, 6.2, 6.3, 6.4, 9.2.

## 72. 6.2 decided up to three names: the arguments of `sampling_variables` (2026-11-16, 02:34)

- **The maintainer**, on the signature assembled from the earlier decisions: "Perfect. sketch looks good".
- **Decided.** The sketch is in the note to 6.2 of `TARGET_DECISIONS.md`. Open inside it: three keyword names,
  the kind strings (9.2), and whether the plan is an argument here (7.3).
- **A slip corrected.** The edit that recorded 6.5 (entry 71) first replaced the tail of row 3.3 and left row
  6.5 open; found on the next read and fixed in the following commit.
- **Counts.** 34 rows decided, 18 remain. Group C has four left: 8.2, 6.3, 6.4, 9.2.

## 73. 6.4 decided: the table of variables is written out and tested against the declared graph (2026-11-16, 02:37)

- **The maintainer:** "Let's go with the second. Table written out and tested against the declared graph."
- **Decided** as in the note to 6.4 of `TARGET_DECISIONS.md`.
- **Counts.** 35 rows decided, 17 remain. Group C has three left: 8.2, 6.3, 9.2.

## 74. 6.3 decided: the public functions of the sampling layer (2026-11-16, 02:40)

- **The maintainer:** "Confirmed. Exactly what I intend."
- **Decided** as in the table of the note to 6.3 of `TARGET_DECISIONS.md`: the engine, the jet-to-dual-jet
  helper pair, the preparation functions and one function behind each frontend method stay or are new; every
  per-variable kernel, every `*_derivatives` function and the sweep functions go; steps and terms are private.
- **Counts.** 36 rows decided, 16 remain. Group C has two left: 8.2 and 9.2.

## 75. 8.2 decided, the name apart: the fitting layer's extension point is one operation in the engine's vocabulary (2026-11-16, 02:45)

- **The maintainer:** "I agree with this data structure. However, I would like to find another word than
  objective. When the user hears "objective", they may think just "f", and not all of the other things that come
  bundled with it. Any other ideas for the name?"
- **Decided:** the data structure, as in the note to 8.2 of `TARGET_DECISIONS.md`. **Open:** its name.
- **Counts.** 37 rows decided, 15 remain. Group C has one left: 9.2.
- **02:52, the name.** `Oracle`. The maintainer: "Oracle is ok, let's go with it. I see your point about jargon,
  but I have not thought of anything better".

## 76. Group C finished: 9.2 decided (2026-11-16, 02:56)

- **The maintainer:** "Confirmed. We will need to discuss 3 more later in the broader context of performance
  optimization hooks and defaults".
- **Decided:** the three kind strings are the kind everywhere; `'sweep'` as a unit name goes and a `hold` lists
  variable names. The sizing function names move to 7.6.
- **Counts.** 38 rows decided, 14 remain: group D (5.2, 5.3, 5.4, 5.5, 5.6), group E (7.2, 7.3, 7.5, 7.6, 7.7),
  group F (8.1, 8.3, 8.4) and group G (9.3). Groups A, B and C are finished.

## 77. 5.2 decided: one set of step functions for both layouts (2026-11-16, 03:11)

- **The maintainer** confirmed the four points and restated them in his own words, which are in the note to 5.2
  of `TARGET_DECISIONS.md`. He added how the operations done at all sites at once behave: a loop on ragged
  cores; on uniform cores all sites at once, chunked over the sample axes through the plan, with chunk sizes set
  by the user's reshaping.
- **Counts.** 39 rows decided, 13 remain.

## 78. 5.6 decided (the absorbing dtype rule); 5.5 reversed (the contraction order goes into the plan); 5.4 as the library does it (2026-11-16, 03:32)

- **5.6.** The maintainer accepted that the engine creates no array with a default dtype, and added the rule
  for mixed inputs: "jax takes precedence over numpy and low precision takes precedence over high prececision
  ... Lower precision taking priority is because the output will not be more precise than its inputs."
- **5.5.** Claude proposed confirming "no option for the contraction order". The maintainer reversed it: "This
  should be in the plan. The user must be able to specify the contraction ordering and tune it on their problem.
  We should also provide helper functions to choose good contraction orderings." He asked how far the
  contraction engine supports einsum paths and how they interact with grouped indices. Read and run for the
  answer: `contract` has no path argument today; numpy and jax einsum both accept one; a path names operands by
  position, so groups do not interact with it (note to 5.5 of `TARGET_DECISIONS.md`).
- **5.4.** He asked how the library handles the stacks in a term today. It writes one literal contraction string
  per term at the place it is used, with `K` on the operand that carries it (note to 5.4).
- **Counts.** 40 rows decided, 12 remain.

## 79. 5.5: one contraction order per term function; the helpers are a design phase; how an order reads with jets (2026-11-16, 03:59)

- **Granularity.** The maintainer: "per term." Read by Claude as per term function (one order each for
  `mu_term`, `nu_term`, `eta_term`, `H_term`); the reading is to be confirmed.
- **Helpers.** Not decided: "Needs a design phase. We can do several." He does not trust flop counts ("flops
  often don't reflect real world performance"), counts a bound on the memory of the intermediates as a good
  criterion, and holds measuring test contractions to be the ideal, tied to the `measure` policy for automatic
  tuning. The helpers go with 7.6 and group E.
- **Jets.** He asked what the question was. Restated, with a proposal, in the note to 5.5 of
  `TARGET_DECISIONS.md`: an order ranks the pairwise products, and a pairwise product of two operands that both
  carry the order axis is their truncated product, carried out as 2.9 says. To be confirmed.
- **Found while checking.** numpy and jax spell an explicit einsum path differently (`'einsum_path'` first, or
  the bare list of pairs), so `contract` has to spell the plan's path for its backend. New script
  `scripts/review/contraction_order_check.py`, added to the runner (22 scripts).
- **Counts.** Unchanged: 40 rows decided, 12 remain. 5.5 stays open.

## 80. 5.5: the granularity of the contraction order moves to the performance discussion (2026-11-16, 04:08)

- **The maintainer**, asked whether "per term" meant per term function or per place a term is used: "I'm a bit
  unclear on granularity actually. Let's move that decision to the performance phase". The reading recorded in
  entry 79 is withdrawn; the granularity is undecided and goes with group E, beside the helpers and the form of
  the plan's field.
- **Still to be confirmed:** how an order reads where the operands are jets (the proposal in the note to 5.5).
- **Counts.** Unchanged: 40 rows decided, 12 remain.

## 81. 8.1 under discussion: what `LocalModel` keeps; the weighted quadratic (2026-11-16, 04:08 to 04:41)

- **Claude proposed** that `LocalModel` be the geometry, the oracle and the regularizer at one frame, with
  `kind`, `sample`, `sweep` and `residual` no longer fields and `jacobian(p)` gone.
- **The maintainer objected** (04:24): the model "serves as a holder for local information at a point, which may
  be used by the user for their own optimizers or their own other purposes. Having access to the residual can be
  useful for many purposes"; and MC-SGD needs Jacobian-vector products for the Cauchy step.
- **Revised:** `residual` and `jacobian(p)` stay on the model as optional outputs of an oracle; the model's own
  `gn_quadratic` and `hvp` use the fused quantities; nothing is stored that the plan's `hold` does not keep. He
  agreed that `residual` and the sweep are not always-stored fields (04:41). The whole is still to be confirmed.
- **The fused quadratic** (04:41): he asked whether it carries the weights (yes: `<Jp, omega^2 Jp>`, now checked
  against `p` paired with `JtJp` with a random weight per mode and order), objected to the name `Jp_sumsq`
  (open; `Jp_norm_sq` suggested), and asked how jets meet the inner product (diagonal, one weight per mode and
  order by 2.5). Full record in the note to 8.1 of `TARGET_DECISIONS.md`.
- **Counts.** Unchanged: 40 rows decided, 12 remain.

## 82. 8.1 decided: what `LocalModel` is; `Jp_norm_sq`; the session paused (2026-11-16, 04:46)

- **The maintainer:** "1. Go with Jp_norm_sq. 2. confirmed. After recording this, please update the handoff and
  let's pause. We will pick up tomorrow"
- **8.1.** `LocalModel` is the geometry, the oracle and the regularizer put together at one frame, and the holder
  of the local information at that point. Its own `gn_quadratic` and `hvp` use the fused `Jp_norm_sq` and `JtJp`;
  `residual` and `jacobian(p)` stay on it, as optional outputs of an oracle; it stores only what the plan's `hold`
  keeps. 8.2 is amended: `f`, `g`, `JtJp`, `Jp_norm_sq` required of an oracle; `Kp`, `Jp`, `r` optional.
- **The name.** `Jp_sumsq` is now `Jp_norm_sq`, in the live documents and in `jet_steps_check.py`; earlier entries
  of this log keep the old name.
- **Counts.** 41 rows decided, 11 remain (7 open, 3 reopened, 1 to confirm).
- **The pause.** The state is in "Where this stands" at the top of `TARGET_DECISIONS.md`, in the status list of
  the folder's `README.md`, and in `dev/HANDOFF.md` on the branch `graded-messages`.

## 83. 8.3, part 1: the optimizers' contract; `oracle.evaluate` with `want` and `held` (2026-11-16, 13:03 to 13:51)

- **Resumed** at 13:03 with 8.3, split by Claude into three parts: the contract and how `held` passes; how a
  minibatch is given; what is compiled as one program.
- **Proposed** (13:03): an optimizer touches a `Geometry`, an `Oracle`, a `Regularizer` and the `LocalModel` built
  from them; the oracle's one operation takes `want` and `held` and returns the values and `held`; a model is
  built once and never changes; trial points as in 4.3.
- **The maintainer questioned it twice.** At 13:22: whether the oracle is more than a thin wrapper, and whether
  "magic words" are still worth it here, against named methods. At 13:41: "we are trying very hard to avoid state
  (a worthy goal) for an operation that is fundamentally about state"; where state should be held, and what
  goes wrong when the optimization is sharded over many GPUs.
- **Claude's answers:** the alternatives A (`want`), B (named methods with `precompute`), C (B with fused
  methods), with a lean to B; then the placement of state by lifetime, from the library's
  `docs/contributor/precompute_and_caching.md`, and six points at scale. A correction on the way: `JtJp` and `Kp`
  share only `nu_L_dot` and `eta_D_dot`, not "the dotted sweep".
- **Decided** (13:51): "Having thought through all of this, I have a greater appreciation for your original
  design with oracle.evaluate. Let’s go with that". Alternative A, as first proposed.
- **Counts.** Unchanged: 41 rows decided, 11 remain. 8.3 stays open for parts 2 and 3.

## 84. 8.3, part 2: a minibatch is an oracle (2026-11-16, 13:55 to 14:07)

- **Decided:** `draw(rng)` returns an oracle; the optimizer never sees a sample or data. The default draw is the
  least-squares oracle's, and `n_measurements` and `take` are its methods, outside the protocol.
- **The regularizer's fraction:** the minibatch oracle is scaled by `n/batch` so that its `f` is an unbiased
  estimate of the full `f`, and the regularizer is not scaled. The scale is its own field of the least-squares
  oracle, not part of the weight. The optimizer keeps no batch logic.
- **Asked by the maintainer** (14:05): whether the minibatch data stays accessible, for future methods that build
  regularizers, preconditioners or step lengths from it. It does, through the concrete oracle's fields and
  through the model; not through the protocol.
- **Added at his word** (14:07): the default draw keeps the drawn indices on the minibatch oracle.
- **Counts.** Unchanged: 41 rows decided, 11 remain. 8.3 stays open for part 3.

## 85. 8.3, part 3: what is compiled as one program; 8.3 decided (2026-11-16, 14:10 to 14:30)

- **Decided:** five phases are the units (build, gradient, CG solve, trial, stochastic step), each a module-level
  function of pytrees, with the outer loop and its decisions on the host. Which phases are compiled by default
  is left to the performance discussion (7.2).
- **A correction by the maintainer:** Claude argued that eager scans recompile on every call. He asked whether
  the scan bodies are closures; they are not (`docs/contributor/scan_body_principles.md`), so an eager scan
  compiles once per shape and compiling the build, gradient and trial is a choice, not a necessity.
- **Decided:** in safe mode a compiled program's preconditions are checked on the host, on its concrete
  arguments, before it is called.
- **MC-SGD over several steps in one program:** wanted eventually, not in this arc. Checked at his request:
  T3Toolbox compiles one step per program, and T3Polynomial has no compiled MC-SGD loop. Decided now so that it
  stays an addition: `take(indices)` on the least-squares oracle is traceable with a static batch size.
- **The maintainer** (14:30): "Confirmed".
- **Counts.** 42 rows decided, 10 remain (6 open, 3 reopened, 1 to confirm).

## 86. 8.4 decided; derivative mode moved out of the fitting layer (2026-11-16, 14:32 to 15:12)

- **8.4.** The display and the diagnostics read from oracles: `r_block_sumsq` and `y_block_sumsq` as optional
  outputs in the shape of the weight, a validation set as an oracle, the trust-region floor as the optional
  output `f_resolution`, and no table for an oracle without block sums. At the maintainer's word (14:43) nothing
  is reduced over modes or orders, in the display or in the stored record.
- **Derivative mode.** Claude's fourth point (convert the unweighted objective back for the display) went against
  2.3 and was withdrawn. The maintainer saw Taylor and derivative mode "becoming first class citizens" and
  considered moving derivative mode out entirely. Decided (15:12): "Ok, let's move it out and add the standalone
  conversion function like you suggest." 2.2 and 2.3 are amended: no keyword anywhere; four standalone
  functions, the fourth preparing data, validation data and weight together and returning the weight `t!` when
  none is given. The doctest draft is rewritten to match.
- **Not included:** the equalizing-weight keyword of 6.7, which he sees as a tunable of the Taylor problem and
  wants thought through separately.
- **Counts.** 43 rows decided, 9 remain (6 open, 3 reopened). Groups A, B, C and F are finished.

## 87. The oracle is a description and the data are an argument; no keyword for the equalizing weight (2026-11-16, 15:12 to 15:44)

- **The maintainer's concern** (15:32), on seeing a prototype signature: "I am concerned about how the data is
  bundled with other things at such a high level and so early in the program." At 15:38: "The API:
  newton_cg(geom, oracle, batch, x0, regularizer=None, ...) is ideal if we can make it work without substantially
  complicating the rest of the code or forcing unwanted performance tradeoffs."
- **Decided** (15:44, "Confirmed both"):
  1. The oracle is a hashable description with no arrays (for least squares: the kind and the order). The sample,
     `y` and the weight are a batch of arrays, passed as an argument through the optimizers to the oracle's
     operation. A minibatch and a validation set are batches given to the same oracle. There is no `Problem` at
     the seam. Rows 8.1 to 8.4 are reworded; the trace found no performance lost.
  2. The equalizing weight is an explicit call when the batch is made, with no keyword in the fitting functions
     (6.7 amended). It is done once and no layer below can tell.
- **Counts.** Unchanged: 43 rows decided, 9 remain.

## 88. 9.3 decided: correct is defined by the definitions (2026-11-16, 15:50 to 15:58)

- **Decided:** the reference for every output is its definition, checked on small problems by dense contractions
  that use none of the sweeps; the identities are tests in their own right; the generic twin is a second check;
  numerical comparisons are to a tolerance and structural ones exact; the old library is a bridge used once.
- **The maintainer:** "I agree wholeheartedly with the proposal about how the definitions are checked, and that
  the identities listed get tests." He added that these checks can serve as documentation, and opened the
  question of how the variables are documented as a whole.
- **Counts.** 44 rows decided, 8 remain (5 open, 3 reopened), all in groups D and E. Groups A, B, C, F and G are
  finished.

## 89. 9.7 added and decided: how the sampling variables are documented (2026-11-16, 15:58 to 16:03)

- **Decided:** four layers, each kind of fact with one source. The engine's docstring is the index; one page in
  `docs/`, whose code CI runs, is the home, with an entry per variable; one example file is the tour; a `.tex`
  note is the derivation.
- **Decided:** the definitions are shipped as a public reference module, one slow dense function per variable.
- **Added by the maintainer:** the dependency structure is documented, as a table and as diagrams.
- **Counts.** A row was added: 45 of 57 rows decided, 8 remain (5 open, 3 reopened), all in groups D and E.

## 90. 5.4: the prototype of the terms over all the stacks (2026-11-16, 16:05 to 16:50)

- **Asked for by the maintainer** (16:05): "Go ahead with the 5.4 prototype".
- **`scripts/review/term_stacks_check.py`** (added to the runner, 23 scripts). The seven terms written once. One
  contraction string per term serves the site group, the order axis, any number of sample axes, the stack of
  directions on whichever operand carries it, and the stack of base points; the direction's place is solved from
  the operands' ranks, so a call site is the formula alone. Checked on 3912 calls against a reference that uses
  neither `contract` nor the helpers, to 9.5e-16; under jit in float32 as well.
- **Found:** an input and a seed have the same rank when `C` is empty and there is no `K`; the reference caught
  the wrong `V_term` this gave.
- **Not decided:** one string per term, or the library's literal string per placement; and whether `xi_term` and
  `V_term` are told whether they are given an input or a seed. 5.4 stays open.

## 91. 5.4: a literal contraction string for each place the direction sits (2026-11-16, 16:24)

- **The maintainer**, shown both options on the dotted left step: "Option A. The reason is actually so the code is
  more self-documenting. The reader of B has a harder time figuring out what contractions are being done, because
  they need to infer Y."
- **Decided:** literal strings per placement, as the library writes them today, with the order group added. The
  prototype's one-string form is not taken.
- **Corrected by Claude on the way:** the literal form is strict against a wrong rank only when every call passes
  both `len_T` and `len_C`.
- **Open in 5.4:** which lengths every call passes; whether the strings stand at the call sites or in a table
  inside a term function. The row stays open.

## 92. 5.4 decided: literal strings at the call sites, through `jet_contract` (2026-11-16, 16:26; prototype rewritten)

- **The maintainer** ("Yes, this is good"): the strings stand at the call sites in the step functions, through a
  `jet_contract` that is `contract` with the order groups multiplying as truncated polynomials; and every call
  passes both `len_T` and `len_C`.
- **The prototype is rewritten in that form** and checked the same way: 41 literal strings in seven families,
  2760 calls against the independent reference, to 4.1e-16.
- **Measured:** with the two lengths a stray axis on `mu` raises; on a seed that carries `K`, where `W` and `K`
  stand together, it raises only when `len_W` and `len_K` are passed as well.
- **Counts.** 46 of 57 rows decided, 7 remain (4 open, 3 reopened): 5.3, 5.5, 7.2, 7.3, 7.5, 7.6, 7.7.

## 93. 5.4: the lengths go to every call as one record (2026-11-16, 16:35)

- **The maintainer:** "I'm ambivalent regarding lens. Let's go with it." Every `jet_contract` call passes one
  record of the axis counts of `S`, `T`, `W`, `K` and `C`, in place of the keywords `len_T` and `len_C`. This
  closes the one case the two keywords left open (a stray axis where `W` and `K` stand together).
- **Then** he asked for a clean split of the refactor into two parts: one that can go forward without much more
  planning, and one that comes later.

## 94. The refactor is split into two parts (2026-11-16, 16:45)

- **The maintainer** asked for a clean split, his mind "running out of metaphorical context", and agreed to
  this one: "Ok, let's do it."
- **Part 1, what is computed:** the variables and names, the conventions, how the layers talk; carried out the
  way the library does it today. **Part 2, how it is carried out:** the plan and every performance option; it
  changes no number. Every decided row is in Part 1 (with the performance options of 2.9, 4.3, 7.1, 7.4 and 8.3
  waiting for Part 2); the seven open rows are Part 2.
- **The cost:** Part 1 has no chunking of samples. T3Polynomial runs with chunking off today.
- **Next:** the small planning items Part 1 needs from him, then an implementation plan for Part 1 written by one
  Opus agent and reviewed by another until they agree.
- The split is recorded in `TARGET_DECISIONS.md`, section "The two parts".

## 95. The small items of Part 1 settled; the plan for Part 1 commissioned (2026-11-16, 16:55)

- **Corewise calls** pass the point's cores `(B, G)` in place of a frame. **Frame consistency** is checked in safe
  mode only, when a `LocalModel` is built and in the engine when centers are given. **The old kernels** are
  deleted as the last step of Part 1, behind the tests and a benchmark the maintainer runs on CPU and on a GPU
  pod. **No release of Part 1 alone:** the work stays on the branch until both parts are done. **A dry port** of
  the consumer's main fit is part of the plan; the maintainer says Claude may edit T3Polynomial through the
  GitHub connector. **Names** are proposed by the plan's writer and settled at the review.
- **The plan:** `PART1_PLAN.md`, written by one Opus agent and reviewed by another until they agree, then read
  with the maintainer. To be kept in mind, in his words things that "have been tricky in the past": safe mode vs
  unsafe mode, ragged vs uniform layer, numpy vs jax dispatch, shared Tucker factors, backend vs frontend razor,
  sharding, jit compiling.
