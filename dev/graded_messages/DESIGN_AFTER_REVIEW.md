# The design after the review of 2026-11-14

**Update, 2026-11-15.** The maintainer found the cause of the pause: the review leaned on preserving old behaviour
and on the sequencing of the refactor, before the target was decided. Compatibility is no longer a reason for any
choice. `TARGET_DECISIONS.md` lists every decision of the target in dependency order and marks which of the
decisions below are reopened. Two things in this document are already overtaken: Taylor coefficients replace
derivatives everywhere, with derivative mode as one adapter at the outer level of fitting (section 4.7 kept
derivatives at the boundary); and the two stages of section 6 are sequencing, set aside until the target is
settled. A third, since 2026-11-15, 19:32: section 4.7's "No option for orders" is overtaken. The plan of section
5 gains a setting for how a product of two full jets is taken over the orders, a scan (the default) or a loop
with slices (target decision 2.9); the dense form stays out.

This document states the design as it stands after two reviewer reports were walked through with the maintainer
on 2026-11-14. It is organized by topic. The same decisions in the order they were taken, with the evidence for
each, are in `REVIEW_DECISIONS.md`. The two reports are in `reviews/`.

**Read this before `EXECUTION_PLANS.md` and `EVALUATOR.md`.** Those two documents, the mockup
`scripts/plan_mockup.py`, and parts of `PROPOSAL.md`, `API_IMPACT.md` and `WORKED_EXAMPLE.md` describe the design
as it was before the review. They have a status note at the top and have not been rewritten.

## 1. Status

- **The maintainer paused the review here.** His words: "I feel like we are cutting corners here for expediency.
  Let's slow things down and plan everything right." The decisions below stand as recorded, but they were taken
  in one fast evening. Section 10 lists the ones that most need a second, slower look.
- **One finding was not discussed:** F17 (backlog items in T3Toolbox that change the proposal's premises).
- **The proposal documents and the mockup are not yet rewritten** to the design below. That rewrite is the first
  job on resuming, after the open points are settled.
- **Nothing in T3Toolbox or T3Polynomial was modified.** Both were read only. The refactor is still not started.
- **What is verified is correctness, on small problems.** Nothing here rests on a timing. See section 9.

## 2. What changed

| Topic | Before the review | After |
| --- | --- | --- |
| Graded variables | A dict keyed by grade; a `PointedCore` object; one generic graded multiplication | No container. Every variable is its own array. Every function takes exactly the arrays it reads. The generic program is kept only as the derivation and the test twin |
| The evaluator | `evaluate(..., want, have=dict)` returning a dict | `sampling_variables(kind, ..., want=(names), name=array, ...)` returning a tuple aligned with `want` |
| The plan | `Plan(samples, assembly, path, orders, sites, executor)` | `SamplePlan(hold, scans, scan_sites, vectorize_sites)` |
| The loop over samples | An executor concept, with a host loop | A scan over axes of the sample stack inside the compiled program. The user pads and reshapes the samples |
| Inner and outer blocking | Two fields | One mechanism: an ordered tuple of scans |
| Jet orders, association order | Plan fields | No option for either |
| Where the plan lives | Passed to every call, bound to nothing | `LocalModel` carries it, with a default. `sampling_variables` takes it explicitly |
| `chunk_size` | A field of the sampling kind | Removed as part of this proposal (in stage 2) |
| The objective value | Read off the orthogonal frame through the Tucker center cores | Computed through the base point's own cores. The center-core route is parked |
| The loss | Hard-coded in the mockup without weights | Weighted least squares, built into `sampling_variables` |
| Names | `_t`, `_z`, `evaluate`, `phi`, rules `UP`/`RIGHT`/... | Section 3 |
| Reaching the optimizers | Not specified | Required: a consumer must be able to use every addition without changing the optimizers or T3Toolbox |

## 3. Names

The maintainer chose to settle the names. The proposal's name table is to become an enumerated token map (old
name to new name), checked by script against the glossary's symbol table, as the store rename did.

### 3.1 The function and its noun

- **`sampling_variables`** replaces `evaluate`. One function serves every sampling operation.
- **"Sampling variables"** replaces "components". The glossary's "edge variables" (`xi`, `mu`, `nu`, `eta` at every
  grade, and `gamma`) are the per-edge subset. The outputs (`z`, `r`, `f`, `g`, ...) are the rest.

### 3.2 Grades

| Grade | Suffix | Meaning |
| --- | --- | --- |
| Plain | none | |
| Tangent | `_dot` | The variable varied in the cores, along the direction `p` |
| Residual | `_tilde` | The variable varied in the samples, along the seed |
| Both | `_tilde_dot` | |

`_dot` is already the library's spelling with this meaning (`xi_dot`, `mu_dot`, `mus_dot`: 149 uses). The library's
two-hole variables are already "tilde then dot" (`tau_tildes_dot`: 114 uses).

**The letter under the tilde moves. This must be spelled out for the reader wherever the names are introduced**
(the maintainer's condition for choosing `_tilde`).

- Today a tilde names the variable it is the adjoint of: `tau_tilde` is the adjoint of `tau`.
- In the proposal the letter names the variable it is a variation of. It is the adjoint (the Lagrange multiplier)
  of that variable's **partner on the same edge**, the one travelling the other way.

| Proposed | Today | Is the multiplier of the constraint defining |
| --- | --- | --- |
| `xi_tilde` | `deta_tilde` | `eta` |
| `eta_tilde` | `dxi_tilde`, `dxi_hat` | `xi` |
| `mu_tilde` | `tau_tilde` | `nu` |
| `nu_tilde` | `sigma_tilde`, `sigma_hat` | `mu` |
| `w_tilde` (the seed on the sample legs) | `ztildes` | `z` |
| `xi_dot`, `mu_dot`, `nu_dot`, `eta_dot` | `dxi` / `xi_dot`, `sigma` / `mu_dot`, `tau` / `nu_dot`, `deta` / `eta_dot` | |

The near miss to warn about: the new `xi_tilde` against today's `dxi_tilde`. They differ by one letter and are
different variables.

Both readings of the tilde were checked on a corewise frame (`scripts/review/tilde_adjoint_check.py`). With
`F = sum_i <s_i, z_i>` and the contraction at each site as a constraint, `dF/d mu = nu_tilde`, `dF/d nu = mu_tilde`,
`dF/d xi = eta_tilde`, `dF/d eta = xi_tilde`, to 3e-15. And each tilde variable is the derivative of the variable it
is named after with respect to the sample vectors along the seed, to 1e-10 by central differences. Not checked
variable by variable on a manifold frame.

### 3.3 Terms, steps, drivers

The seven site rules are named after the variable they produce, with no direction word.

| Was | Now | Produces |
| --- | --- | --- |
| `UP` | `xi_term` | the `xi` family |
| `RIGHT` | `mu_term` | the `mu` family |
| `LEFT` | `nu_term` | the `nu` family, and `gamma` |
| `DOWN` | `eta_term` | the `eta` family |
| `OUT` | `z_term` | the probe output |
| `GRAD_G` | `H_term` | the TT-core gradient block |
| `GRAD_B` | `V_term` | the Tucker-core gradient block |

- Every per-site function has the suffix `_step`, as the library's `_mu_step`, `_xi_step`, `_eta_step` do.
- The assembled gradient blocks are `V_tilde` and `H_tilde` (the library's `V_tildes`, `H_tildes`). The two-hole
  blocks are `V_tilde_dot` and `H_tilde_dot`.
- The loop drivers are `left_to_right` and `right_to_left` (as in `tt_zipper_left_to_right`).
- **Why no direction words.** The library calls `mu` the "leftward edge variables": the left environment, built
  from the left cores. It moves from left to right. So "leftward" can be read as the side or as the direction of
  travel. In prose write "`mu`, the left edge variables, swept left to right". The library's own docs pass (item
  E22) makes the same replacement.

### 3.4 Outputs

| Name | Meaning | Identity in the grade notation |
| --- | --- | --- |
| `z` | the kind's output, for every kind | |
| `r` | the unweighted residual `z - y` | |
| `f` | the data misfit, half the weighted sum of squares of `r` | alias `misfit` |
| `g` | the gradient `Jt r` | `(V_tilde, H_tilde)` seeded by `omega^2 r` |
| `Jp` | Jacobian-vector product | alias `z_dot` |
| `JtJp` | the Gauss-Newton product | `(V_tilde, H_tilde)` seeded by `omega^2 Jp` |
| `Kp` | the curvature part of the exact product | `(V_tilde_dot, H_tilde_dot)` seeded by `omega^2 r` |
| `Hp` | the exact Hessian-vector product | assembled by `LocalModel` from `JtJp` and `Kp`; not a sampling variable |
| `r_block_sumsq` | per-mode, per-order sums of squares of `r` | |
| `Jp_sumsq` | weighted sum of squares of `Jp`, the Gauss-Newton quadratic | |

- `Hp` never means Gauss-Newton. The maintainer's concern is confusion between the two. `JtJp` was preferred to
  `HGNp` because it has no `H` and composes with `Jp`.
- The aliases are documented identities, not second spellings in code. `want` and the keyword arguments accept
  one spelling per variable.
- `phi` (the apply value) is gone: the output is `z` for every kind.

### 3.5 `gamma`

`nu` means the frame's right edge variable, swept through `R`, and nothing else. The right edge variable swept
through `L`, the base point's own cores, is a different variable: **`gamma`**. The maintainer's view is that it is
an auxiliary variable, not a `nu`. Its residual-grade companion is `gamma_tilde`.

- `gamma` is read by `z` and by `Kp`. `gamma_tilde` is read by `Kp`. Both are public: they can be wanted and held.
- The library writes `gamma` as `nu_c`, with `nu_c = T nu` at a manifold frame (`curvature_internals.md`). The
  proposal's matrix `S` in that identity becomes `T`.
- At a trial point the call uses the point's own cores as the frame, so `gamma` and `nu` coincide there.
- The name is free in code. In the docs, lowercase gamma is a scalar in one example of `padded_frame_cone.tex`, and
  capital `Gamma_i` is the mode Gram matrix in the sharing documents.

### 3.6 The plan

| Thing | Name |
| --- | --- |
| The plan | `SamplePlan` |
| One nesting level | `SampleScan`, with fields `axes` and `hold` |
| The per-variable site option | `scan_sites`, `vectorize_sites` |
| The pad-and-reshape helpers on a sampling kind | `chunk`, `unchunk` |
| The sizing functions | the library's `estimate_chunk_size` and `max_chunk_size_within`, extended |

Three words used before the review are already taken in the glossary and are replaced in prose:

| Was | Already means | Now |
| --- | --- | --- |
| "block" of samples | the `C`, `W`, `K` batch blocks | "chunk" |
| "level" | a rank level in rank continuation | "scan" |
| "store" / "keep" | a rank store; and `Plan.keep` was removed | "hold", "held" |

### 3.7 Smaller fixes

The highest order is spelled `order`, not `p`. "Messages" (the glossary's padded-frame `Psi`) is not used. "Center
cores" is reworded where it survives (the glossary's terms are "carriers" and "centers"). The mockup's `sweep` and
`sitewise` drivers are renamed, because `sweep` is the glossary's noun for the held frame sweep.

### 3.8 Still open on names

- **Two vocabularies.** If the refactor keeps the public kernels behind adapters, the glossary's rule makes
  `sigma`, `tau` and the `*_tilde` names canonical and the names above a second, internal vocabulary. Making them
  the only vocabulary is a breaking rename. This belongs with the two API cases of `API_IMPACT.md`, still open.
- **The kind string.** `sampling_variables` takes `'probe'`, `'apply'` or `'entries'` plus an integer `order`. The
  frontend's strings are six, with `'probe_derivatives'` and its relatives. The two spellings need reconciling.

## 4. The sampling layer

### 4.1 Explicit functions

No graded container crosses a function boundary. The maintainer rejected dicts under the backend's plain-data
rule, and then rejected fixed-slot tuples `(plain, t, z, tz)` with `None` for an absent grade, because a `None`
slot does the job of a dict key: what is present still has to be inspected at run time. The rule is for the
reader of the code, not the compiler.

```python
# a term: one contraction, one literal subscript string
def mu_term(mu, G, xi):   return contract('Wa,aib,Wi->Wb', mu, G, xi)

# a step: the argument list is what it reads, the body is its terms
def mu_dot_step(mu_dot, mu, xi, xi_dot, R, U, H):
    return mu_term(mu_dot, R, xi) + mu_term(mu, H, xi) + mu_term(mu, U, xi_dot)

def mu_tilde_step(mu_tilde, mu, xi, xi_tilde, L):
    return mu_term(mu_tilde, L, xi) + mu_term(mu, L, xi_tilde)

# a call site names everything it passes, including the start value
mu_dot = left_to_right(mu_dot_step, zeros, d, mu, xi, xi_dot, R, U, H)
```

- There is no pointed-core object. Each term names the frame core it uses, so the gauge choice is visible.
- The corewise frame calls the same functions with `G, G, G`.
- For the probe kind this is 22 step functions with 60 terms on 7 term functions (`scripts/review/jet_steps_check.py`).
- The subscript strings are literals again, so the library's test suites that scan the source for them can see
  them. This closes most of F1; keeping the literals inside `contract(...)` calls and pinning them in the
  suites' `HISTORICAL` table are implementation-time items.
- Costs accepted: the two-hole steps take 10 to 12 arrays; a positional argument in the wrong place is an error
  only if shapes disagree; a new case means writing its terms out.

### 4.2 `sampling_variables`

A public backend function. The sketch below is not a final signature: the argument for the lifted direction has
no name yet.

```python
f, g, xi = sampling_variables('probe', frame, sample, plan, want=('f', 'g', 'xi'), y=y, weight=omega)
JtJp,    = sampling_variables('probe', frame, sample, plan, want=('JtJp',), p=p, xi=xi)
```

- **`kind`** is a plain string, first. Derivative kinds add an integer `order`. A string keeps the sampling layer
  from importing the `SamplingKind` classes, which live above it. The fitting layer also calls its `SamplingKind`
  objects `kind`; that overlap was accepted.
- **`want`** is a static tuple of names. The return is a tuple aligned with it.
- **Held variables are keyword arguments, one per variable** (`xi=xi`). Nothing can be misaligned.
- **Inputs:** the sample, `y`, an optional `weight`, the direction `p` for `Jp` and `JtJp`, and the lifted raw-core
  direction for `Kp`.
- **Errors.** Names in `want` are validated against the kind's graph before any scan. An input that nothing
  wanted reads is an error, as in `Problem.objective`. A misspelt keyword fails by itself.
- **Each kind is a small declared graph over the same step functions.** The graph is data: name, reads, function.
  The reads are the function's argument list. `scripts/component_graphs.py` derives the reads from the twin.
- **The docstring carries a table**, for every variable, of what it reads and which of those are worth holding
  (the maintainer's idea). It is to be generated from the declared graph and checked by a test.

### 4.3 What each output reads

For the probe kind, among public variables. "Worth holding" depends on what a variable depends on.

| Depends on | Variables | Valid for |
| --- | --- | --- |
| The point and the samples | `xi`, `mu`, `nu`, `eta`, `gamma`, `z` | every call at this point |
| Also the data | `r`, `f`, `r_block_sumsq`, `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`, `gamma_tilde`, `g` | every call at this point with this data |
| Also the direction | the dot variables, `Jp`, `Jp_sumsq`, `JtJp`, `Kp` | one product |

| Want | Reads directly | Ideal to have |
| --- | --- | --- |
| `z` | `xi`, `mu`, `gamma` | all three |
| `f` | `r` | `r`, if it was carried from the accepted trial |
| `g` | `xi`, `mu`, `nu`, `eta`, `r`, and the four frame tilde variables | `r`; plus `xi`, `mu`, `nu`, `eta` if products follow |
| `Jp` | `eta`, `eta_dot` | `xi`, `mu`, `nu`, `eta` |
| `JtJp` | `xi`, `mu`, `nu`, `eta`, `Jp` | `xi`, `mu`, `nu`, `eta` |
| `Kp` | `xi`, `mu`, `gamma`, `r`, `xi_tilde`, `mu_tilde`, `gamma_tilde` | all seven |

These rows were derived by hand from the argument lists and are a draft. Held variables are valid for the same
point, samples, data and order.

Variables with no public name:

- The tilde variables seeded by `Jp`. They change with every product and stay internal to `JtJp`. The tilde
  variables that can be wanted or held are always the ones seeded by the residual.
- The `eta`-like variable through `L` that only `z` reads.
- The dot variables of `Kp`, which belong to the lifted direction.

### 4.4 The value

- **One route:** through the base point's own cores, `(D, L)` for a frame and `(B, G)` for a trial point. For a
  frame it reads `xi`, `mu` and `gamma`. No frame is built for a trial point, as in the library today.
- **The route through the orthogonal frame is parked.** The identity `z[i] = OUT(eta[i], Vc[i])` holds to 1.7e-15,
  but it needs the Tucker center cores. They are unguarded (scaling them by 1.001 changes `f` by 2e-3 with no
  error), have no channel in the geometry protocol, and were examined and left alone in the library's caching
  audit. `sampling_variables` takes no `centers` argument. "A manifold frame has no plain probe output" stands.
- **Carrying `r` from an accepted trial** saves the same sweep, using `want` and keyword arguments only. The
  counts are in the table below.
- **Frame consistency is a diagnostic, not a precondition,** and `(D, L)` is recorded as the authoritative base
  point, as the backend reads it. This softens the guard designed in phase 1. `T3Frame.to_t3` reads `(D, R)`; that
  is noted for the maintainer and no change is proposed.

Sample sweeps and frames built per outer iteration with `t` trials. These are counts read from
`backend/optimizers.py`, not timings.

| Route | Sample sweeps | Frames built |
| --- | --- | --- |
| Today | `t + 2` | 1 |
| Carry the accepted trial's `r` | `t + 1` | 1 |
| Value through the frame at the accepted point (parked) | `t + 1` | 1 |
| Value through the frame at every trial (the fusion the audit declined) | `t` | `t` |

### 4.5 Start values

The network's boundary vector is ones, sized by the tails of whichever cores are passed. A sweep starts from its
own variable's value on the boundary bond, and these differ (read from the library's kernels).

| Sweep | Starts from | In the library |
| --- | --- | --- |
| Plain: `mu`, `nu`, `gamma` | ones | `compute_mu` |
| Dot | zeros | `compute_sigma` |
| Tilde, probe | zeros | `compute_tau_tilde` |
| Tilde, apply and entries | the seed times the ones vector, at the right boundary | the `sigma_hat` sweep |
| Tilde-dot | zeros | `compute_tau_tilde_tangent`, `compute_sigma_hat_tangent_jets` |
| Plain, with jets | ones at order 0, zero above | `_init_jet` |

Every start is a visible argument of the driver, and takes the dtype of the sweep's operands.

- **Tails.** Default frames have squashed tails of size 1; a point's own cores may not. Tail ranks above 1 already
  work in the current kernels, so the requirement is "must not regress".
- **Carriers.** The cores at the ends of the frame are read against a zero carry and must be finite, as today.

### 4.6 Kinds

Apply and entries use the probe kind's step functions (`scripts/review/kind_readout_check.py`, to 8e-16).

- **The leaf, for entries only:** `xi` is made by a gather. That is the one function not shared.
- **The readout:** the output is the last `mu`.
- **The seed:** one scalar per sample, entering at the right boundary. The tilde variables are the plain steps
  started from it.
- **Padding.** For apply and probe a padded sample contributes zero to everything. For entries a padded index
  reads a real entry, so `z` and `Jp` are multiplied by a 0/1 mask at the readout. Then `r` is zero at padded
  samples and every sum and seed is right. The maintainer said the mask "gives me pause" and accepted it for now.

The `SamplingKind` protocol is unchanged (section 6).

### 4.7 Jets

T3Polynomial fits probe derivatives, so this case matters most. In the explicit form
(`scripts/review/jet_steps_check.py`):

- Every variable gains a leading order axis. Jets change only the seven term functions, where a contraction of
  jets becomes a truncated product over the order axis. The 22 steps and 60 terms are unchanged.
- There is no second set of functions for the tilde grade, because residual jets are stored reversed. That is
  the Taylor convention of phase 1, used internally.
- The library's conventions are kept at the boundary: `(w, p)`, `y`, the weight `omega[mode, order]`, and `z`, `r`,
  `Jp` are derivative-normalized, as the kind takes and returns them. Only held or wanted edge variables with
  jets are in the internal convention (rules F and R of the correspondence in `PROPOSAL.md`). So no user-visible
  convention has to change.
- With the library's normalization inside instead, forward terms would be convolutions and tilde terms
  correlations, and the seven terms would not serve every grade.
- **No option for orders.** For a product of two full jets, "vectorized over orders" is the dense `trs` contraction
  under another name, which the library removed from production in 2026.3.0 (the recurrence-only rule). A real
  implementation of each term would be a banded two-term recurrence when one operand is an input jet and an
  order scan otherwise, as the library does. That was not built here: the products in the check are plain loops.

### 4.8 Weighted least squares, and every reduction over the samples

Weighted least squares is built in, against the reviewer's suggestion to keep the loss outside: fusing `f` and `g`
in one pass needs the residual formed inside the chunk scan.

| What is reduced | Used for | Comes from |
| --- | --- | --- |
| Weighted sum of squares of `r` | the misfit | `f` |
| Per-mode, per-order sums of squares of `r` | numerators of the display's relative errors; also on validation samples | `r_block_sumsq` |
| The same sums for the data | the display's normalization | `kind.block_sumsq(data)`, unchanged |
| Weighted sum of squares of the data | the trust-region floor | `kind.sumsq(data)`, unchanged |
| Weighted sum of squares of `Jp` | the Gauss-Newton quadratic, read after each solve and in the model value | `Jp_sumsq` |

The trust-region floor also reads the residual, but only its dtype.

### 4.9 Layering

- The blocks `g`, `JtJp`, `Kp` are the ones before the gauge projection. The projection stays in the geometry and
  `LocalModel` applies it. The mockup projected inside; that goes.
- `Kp` takes the lifted raw-core direction, not `p`, and returns blocks in raw core coordinates. `LocalModel`
  lifts, subtracts the geometry's tangential curvature, and applies the transpose of the lift, as
  `_curvature_block` does today.
- Tucker-factor sharing stays in the geometry wrapper. The evaluator is unchanged at a shared point
  (`scripts/sharing_check.py`).

## 5. The plan

```python
plan = SamplePlan(
    hold=(),                                               # what the model holds between calls: here, nothing
    scans=(SampleScan(axes=(0,), hold=('xi', 'mu')),       # outermost first: hold xi, mu for each chunk of axis 0
           SampleScan(axes=(1,))),                         # then scan axis 1 inside it
    scan_sites=('eta_tilde',),
)
```

`SamplePlan` and `SampleScan` are named tuples: immutable, made of basic types, hashable for the compile key.

### 5.1 Scans

- A plan is an ordered tuple of scans, outermost first.
- A scan names the axes of the sample stack it loops over, as a tuple of indices, and the variables to compute
  once for the chunk it receives before it loops (`hold`).
- Axes that no scan names are vectorized.
- The default plan has no scans: everything is vectorized over the sample stack, which is today's behaviour on
  unchunked data.
- Inner and outer chunking are the same mechanism. The old inner assembly chunk is a scan whose `hold` is the
  frame sweep.

### 5.2 The top-level `hold`

What the model holds between calls, for the life of the model. A scan's `hold` lasts for one chunk.

- **Default (the lean one, chosen by the maintainer):** the kind's frame sweep, plus `r` only when the Hessian is
  exact, because those products read it every time. Today the library holds the frame sweep and `r` always.
- `lm.residual` stays available and is computed on request when `r` is not held.
- `'sweep'` names the kind's own frame sweep as a unit: `xi`, `mu`, `nu`, `eta` for probe, and `xi`, `mu` for apply
  and entries. User-written kinds have only this unit.

### 5.3 Chunks are array shapes

- The user pads the samples to a multiple of the chunk size and reshapes, adding an axis to the sample stack.
  The plan names the axes to scan. Chunk sizes are not in the plan.
- A sampling kind provides `chunk` and `unchunk` beside `take`. The operation on unpadded input is
  `unchunk` after the scanned operation after `chunk`. Automatic sizing uses the same helper.
- There is no executor concept. The scan is inside the compiled program, so the plan holds names and axes, not
  sizes, and compile keys are structural.

### 5.4 Sharding

From `scripts/review/levels_jax_check.py`, on four virtual CPU devices:

- Scanning a sharded axis under plain jit gathers the data (12 all-gathers).
- Two arrangements are clean, with no all-gathers: `shard_map`, where each device scans its local chunks; or a
  separate sharded axis that no scan names.
- A compiled conjugate-gradient loop of 6 iterations under `shard_map` matched to 1e-15.

Real GPUs were not used.

### 5.5 Sites

A per-variable option, defaulting to what each variable's kernel does today.

- The per-site step itself ties on memory, step count and width whether sites are scanned or vectorized.
- The hungriest step sets the chunk size. Scanning that step over sites divides its scratch by `d` and allows a
  chunk `d` times larger, which makes the sweeps (always sequential over sites) `d` times wider.
- A step that is not memory-binding should be vectorized over sites.
- This is decidable from scratch-per-sample figures. The hardware constants are not measured.
- It is also the lever the library's record names for the scanned jet kernels' small-problem latency.

### 5.6 Sizing

The library's `estimate_chunk_size` and `max_chunk_size_within` keep their names and are extended to the outer
chunk. An estimator is told which calls it must serve, sizes for the worst of them, and confirms at the chosen
size from the compiled program's memory analysis. **Caveat found afterwards:** T3Polynomial already turns
`chunk_size='auto'` off because it costs about 4 compiles per Newton solve, at every continuation level. Confirming
by compiling has that cost, so it should be opt-in and reusable across levels. Not designed.

## 6. Staging and wiring

The maintainer's requirement (section 7) rules out leaving any of the wiring undecided.

- **Stage 1, independent of the kernel refactor.**
  - `LocalModel` is written once against a names-based interface: it asks for `f`, `g`, `Jp`, `JtJp`, `Kp`, `r`,
    `r_block_sumsq`, `Jp_sumsq` under a `SamplePlan`.
  - An adapter serves that interface for every kind, built-in or user-written, by calling the six existing kind
    operations inside the chunk scans. The kind protocol does not change, apart from `chunk` and `unchunk`, which
    are optional in the way `take` is.
  - Held sets are limited to nothing or the kind's frame sweep.
  - `chunk_size` stays on the sampling kind and acts as the innermost scan.
- **Stage 2, with the kernel refactor.**
  - The built-in kinds are served by `sampling_variables`. The optimizers do not change again.
  - Arbitrary held sets, the per-variable site option, and `want` on individual variables become available.
  - `chunk_size` leaves the sampling kind.
  - How a kind is marked as served by the engine is decided and built here: at most one optional member,
    defaulting to "not served".
- **User-written kinds** stay on the adapter.
- The maintainer is ambivalent on the sequencing and accepted the two stages on trust.

Two observations for whoever takes up `dev/OPEN_QUESTION_extension_surface.md`. They are not decisions.

- Forward-only probing, which that file calls "not hypothetical", would be one more declared graph over the same
  step functions.
- That file's sharpest stressor is iterate-dependent weights, which force a recompile because the weight is a
  static field of the kind. In `sampling_variables` the weight is an input array. Not tested.

## 7. The consumer requirement

The maintainer, 2026-11-14: immediately after the refactor he will profile the downstream consumer T3Polynomial
and tune the new options. **The end state must let a consumer use every addition without rewriting the optimizers
and without changing T3Toolbox.**

What T3Polynomial does, read at its commit `dbda0de`:

- `t3polynomial/fitting.py::_newton_fit` calls the frontend
  `t3toolbox.optimizers.newton_cg(geometry, 'probe_derivatives', (ww, pp), data, x0, order=, weight=, **kwargs)`.
- It runs on the ragged, uniform and uniform-jax backends, the last with `use_jit=True`.
- Every `newton_cg` option passes through its `newton_kwargs`, a dict or a callable per continuation level.
- It uses a sharing geometry, `padded_from`, validation samples, and `hessian='exact'` on later levels.
- Its training sample is the same vectors at every input mode (`ww = [xh] * degree + [omegas]`). The parked
  shared-sample-vector idea applies to its main fit.

Draft acceptance, not yet agreed in detail:

- A `plan=` keyword on the frontend `newton_cg`, reaching `Problem` and `LocalModel`. T3Polynomial can then tune
  through `newton_kwargs` with no change to its fit code.
- Chunking of the sample, the data and the validation pair reachable from that same call.
- Works on all three backends, with jit, a sharing geometry, `padded_from`, the exact Hessian and the verbose
  display.
- The public route for the plan has not been specified beyond this.

## 8. Tests the design calls for

- Tolerance tests for numerical operations; exact tests for structural things (the maintainer's rule).
- Structural slices of the port are gated by the library's bitwise battery; each numerical change gets its own
  slice and tolerance argument (F18).
- The written-out terms against the generic twin; the declared reads against `component_graphs.py`.
- The docstring table against the declared graph.
- Plan invariance, including a sample count not divisible by the chunk.
- No reference cycles after a call; peak within budget for every call a sizing function claims to serve.
- Name validation; plan validation at construction.
- Pairing at the grades and frames where it holds: all four grades in the corewise frame; grade dot in a
  manifold frame; tails only for plain and tilde (D6).

## 9. Evidence, and what it does not cover

All on small problems, in numpy unless noted, against T3Toolbox at `3dc15d9`. Recorded output:
`scripts/review/expected_review_output_3dc15d9.txt`.

| Script | Shows | Agreement |
| --- | --- | --- |
| `unbundled_examples.py` | The explicit form, order 0, probe: `Jp`, `Jt r`, two-hole | 4e-16 |
| `jet_steps_check.py` | The explicit form with jets and weights, final names, against `ProbeDerivativesKind` at orders 0, 2, 4: `z`, `f`, `r_block_sumsq`, `g`, `Jp`, `Jp_sumsq`, `JtJp`, `Kp` | 4e-16 |
| `kind_readout_check.py` | Apply and entries on the probe kind's steps: `z`, `Jp`, transpose | 8e-16 |
| `tilde_adjoint_check.py` | The two readings of the tilde | 3e-15; 1e-10 |
| `slot_tuple_examples.py`, `slot_tuple_check.py` | The declined fixed-slot form, kept as a record | 2e-16 |
| `pad_reshape_check.py` | Padding and reshaping with the library's kernels: padded samples are inert for apply and probe; entries need the mask | 1e-16; wrong by 40% for entries without the mask |
| `levels_sketch.py` | Scans on a `(4, 8, 16)` stack, 7 configurations, on the mockup | 1e-15 |
| `levels_jax_check.py` (jax) | Scans under jit, nested, uniform layout, jets, masks, `shard_map`, compiled CG, on the library's kernels | 3e-16 to 1e-15 |

Not covered:

- **The explicit form on the uniform layout, with the `K` and `C` stacks, or under jit.** The scan experiments ran
  the library's existing kernels, not the explicit form.
- **Order scans and banded recurrences in the Taylor form.** Statements about memory per order rest on the
  library's present kernels.
- **Apply and entries with jets; `Kp` for apply and entries.**
- **The stage-1 adapter.**
- **The entries mask at the readout.** The tested form put the mask on the seed (7e-16 for the gradient).
- **Holding variables per scan on the uniform layout,** whose sweep carries static data beside the variables.
- **Sizing the outer chunk.**
- **Real GPUs, eager jax, compile time of nested scans.**
- **The site option.** Which variables can be scanned over sites was not enumerated or tried.
- **Anything about speed.**

## 10. Open items

### 10.1 Decided quickly; worth a second look

These are the points where, in Claude's judgment, the pace of the evening left the most unexamined. The list is
Claude's, not the maintainer's.

1. **The stage-2 wiring.** Claude first proposed deferring it to the extension-surface question. The consumer
   requirement reversed that within two messages. The adapter, the marking of a kind, and the public route for
   the plan are sketched, not designed.
2. **The Taylor convention inside.** It makes the seven terms serve every grade and the check passes, but it was
   adopted in one step, with no build of the banded and scanned forms and no look at the uniform layout.
3. **The entries mask.** It moved from the seeds to the readout to fix the sums. Not rerun; and the maintainer
   already had reservations about a mask.
4. **The top-level `hold`, its lean default, and `'sweep'` as a unit name.** Introduced late to answer F9 and the
   adapter. Changing what `LocalModel` holds by default is a behaviour change.
5. **Weighted least squares inside the sampling layer.** Decided against a reviewer's finding (D22) in one
   exchange. It fixes the loss at this layer.
6. **Automatic sizing.** The adopted rule has the compile cost the consumer already avoids.
7. **`gamma` and `gamma_tilde` as public variables.** Sound as far as checked, but the docstring rows for `Kp` are
   hand-derived.
8. **The kind as a string,** against the frontend's six strings and the `SamplingKind` objects.
9. **The site option.** Decided on an accounting argument; its representation and scope are not worked out.
10. **Dropping `Plan.orders`.** Right for full products. One kernel (the two-row `dxi~` correlation) has an order
    scan that involves no `trs`; it was judged not to earn an option.

### 10.2 Not discussed

**F17**, backlog items in T3Toolbox that change the proposal's premises:

- float32 end to end: a ready patch touches the plain scan starts.
- The data-shaped residual weight (deferred): the weight becomes a per-sample input.
- Sharded fitting.
- The reference paper, written in the binomial form, so the jet-normalization decision should precede it.
- The codebase-wide style audit announced on 2026-11-02.
- The leftover-chunk overshoot, which bears on padding the last chunk.

### 10.3 Undecided

- The two API cases of phase 1 (adapters, or removing the redundant layer), and with them the two vocabularies.
- The name of the lifted-direction argument of `sampling_variables`.
- Where the `W` axes sit in each held variable on each layout (F10).
- Whether a `have`-validity check exists, and what identifies a point.
- The acceptance list in section 7.
- Parked: per-group `xi` for shared sample vectors (`scripts/parked/`); the value through the frame.

### 10.4 Work not done

- Rewrite `EXECUTION_PLANS.md`, `EVALUATOR.md` and the affected parts of `PROPOSAL.md`, `API_IMPACT.md` and
  `WORKED_EXAMPLE.md` to this design, and redraw the dependency diagrams.
- Rewrite `scripts/plan_mockup.py` to the decided interface: scans, the new names, keyword arguments, kinds. It
  still has a reference cycle in `run` (D14).
- The token map and its check; the docstring table generator; the tests of section 8.
- The fixes the reviewers found that need no decision (section 11).

## 11. The findings, one by one

`D` findings are from `reviews/review_1_design.md`, `F` from `reviews/review_2_library_fit.md`. "Decision n" refers
to `REVIEW_DECISIONS.md`.

| Finding | Outcome |
| --- | --- |
| D1, F7, F21 | Decision 7: the value through the base point; centers parked; consistency a diagnostic |
| D2, F8 | Decisions 2, 4, 5: scans; the site option defaults to each kernel's present choice; `chunk_size` removed |
| D3, D5 | Text to rewrite |
| D4 | The "one rule" is an upper bound; the diagrams contradict the reuse table. To redraw from the argument lists |
| D6 | Restate where pairing holds |
| D7, D21, F12 | Adopted: boundary ones sized by the tails; "must not regress"; add the tails check |
| D8, D16, F13, F14, F15 | Small corrections to make |
| D9, D10 | Adopted: the sizing contract of section 5.6, with its caveat |
| D11, D17, F3 | Decisions 1, 2, 3: scans in the compiled program; `LocalModel` carries the plan |
| D12 | Adopted: validation, and unread inputs are errors |
| D13 | Decision 8: seeds (only residual-seeded tilde variables are public) and gauge (`gamma`) |
| D14 | The reference cycle in the mockup's `run`: to fix |
| D15 | Adopted: carriers are read against a zero carry. The other two points fall away with the generic program moving to the tests |
| D18, F2 | Decision 6: explicit functions; the declared per-kind graph is the only layer |
| D19, D20, D23 | Driver and test details to apply |
| D22 | Declined: weighted least squares is built in (decision 9) |
| F1 | Mostly closed by decision 6 |
| F4 | Decision 8: names |
| F5 | Decision 4: no association option; one contraction per term |
| F6 | Decisions 1, 2: the plan holds no sizes |
| F9 | Decision 9: the top-level `hold`, `r_block_sumsq`, `Jp_sumsq` |
| F10 | Open: where the `W` axes sit in held variables |
| F11 | Decision 10: wired in stage 1 through an adapter; marking decided inside the refactor |
| F16 | No conflict: the library's docs pass uses the same wording for `mu` |
| F17 | Not discussed |
| F18, F19 | To apply to the order of work and to `API_IMPACT.md` |
| F20 | Decision 5: two stages |
| F22 | Noted |
