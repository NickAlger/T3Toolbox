# Review 1: the design on its own merits

Reviewed: `/home/claude/graded_messages_handoff` (README, PROPOSAL, WORKED_EXAMPLE, EVALUATOR, EXECUTION_PLANS, API_IMPACT skimmed, all scripts, the three diagrams), against T3Toolbox at `3dc15d9`. Nothing in the package or the repository was modified. The scratch scripts I ran are in `/home/claude/reviews/review_1_scratch/` (run from `/tmp/rev`, with `PYTHONDONTWRITEBYTECODE=1`).

The algebra holds up: every numerical claim I re-ran reproduces, `run_all.sh` output is byte-identical to `expected_output_3dc15d9.txt`, and the identities survive cases the package did not test (tails above 1, pairing at all four grades). The weak part is the layer added last. The evaluator and plan documents contradict the earlier documents in several places, the `Plan` cannot express today's choices that it promises to default to, and a plan sized by `fill` breaks its own budget on the other calls of the package's own example. Most problems are fixable by tightening definitions, not by changing direction. Three of them (D1, D2, D9) should be settled before any library code is written.

"Verified" means I checked it in the files or by running something. "Not verified" means reasoning only.

## Findings

### Inconsistencies

**D1. The documents disagree on whether a manifold frame has a plain probe output.** Severity: high. Verified.
- Where: PROPOSAL "Combinations that do not exist", "Pointed cores" and the operation-families table; README "Things learned" ("A manifold frame has no plain probe output"); WORKED_EXAMPLE step 2 and "The value needs its own sweep". Against: EVALUATOR "Frames and reuse" and "The mockup"; `plan_mockup.block_graph` node `z`; `reuse_check.py` item 4. README asserts both, in "What is established" and in "Things learned".
- Evidence: `reuse_check.py` prints 1.7e-15 for `z = OUT(eta, Vc)`. The mockup uses that as its only rule for `z`. So the Tucker center core is exactly the "variant for a hole beyond the ambient leg" that PROPOSAL says does not exist. I also checked that the centers are redundant given the frame: `Hc[i] = L[i] S[i+1]` and `Vc[i] = <U[i], Hc[i]> D[i]` match the library's centers to 2e-15 (`review_1_scratch/centers.py`). And `centers` is unguarded: scaling `Vc` by 1.001 changes `f` by 2e-3 and `g` by 1e-3 with no error.
- Suggestion: add the center variants to the pointed-core table and delete the non-existence claim (only the grade-`t` core gradient remains undefined). Either derive the centers from the frame or make them part of the frame object, so `evaluate` loses the `centers` argument. Extend "Frame consistency" to cover them: the objective now reads a third representation of the base point. Mark step 2 of the worked example as the corewise alternative.

**D2. `Plan()` is not today's behaviour, and one `Plan` cannot express today's behaviour.** Severity: high. Verified.
- Where: EXECUTION_PLANS "Design constraints" ("the default plan must reproduce today's hand-made choices"), "The plan and the drivers", "Policies" (`current` is "The default"); EVALUATOR "What this means for the plan".
- Evidence, field by field, using the document's own "Today" column and the library:
  - `assembly=None` means the outer block; today is `b = chunk_size` (100, or `'auto'` in the optimizers).
  - `path='carry_first'` is numpy's fixed path. On jax the library lets jax choose. The jet sweep hand-codes input-first: `_mu_jets_step` (`sampling_derivatives.py:379-381`) forms `Gxi` and then one GEMM.
  - `sites` and `orders` are single global strings. Today they differ per rule: sequential for combines and assemblies but vectorized for lifts; a scan for full-times-full but a fused pair for affine factors.
  - EVALUATOR says a `current` preset "must mean" `have = FRAME + r`. A policy is defined as a function returning a `Plan`, and the plan has no keep field, so `current` cannot carry that.
- Suggestion: make association and sequential-or-vectorized choices per rule (a small mapping with a default), add a value meaning "the backend's own choice", and define `current` as a preset returning a plan and a keep set. Then "`Plan()` equals `current()`" is a test.

**D3. EXECUTION_PLANS still describes the kernel as it was before `want` could name per-sample components.** Severity: medium. Verified.
- Where: EXECUTION_PLANS "The plan and the drivers" and "Two levels of sample blocking"; EVALUATOR "An operation is a per-block kernel plus the sample loop".
- Evidence: "It returns a core-sized partial result and never depends on the total number of samples" against "The kernel returns a pair: a part that is summed across blocks, and a part that is written into per-sample outputs". Also "Nothing sized by `W` then exists except the data and whatever the plan keeps": the plan keeps nothing.
- Suggestion: restate the kernel contract once, with both parts, and remove the leftover.

**D4. The "one rule" for the dependency graph is an upper bound, and the drawn graph contradicts the reuse table.** Severity: medium. Verified.
- Where: EVALUATOR "One pattern"; README "What is established"; `dependency_pattern.png`; `dependency_graph.png`; README contents row for `dependency_graph.dot`.
- Evidence: the rule is "a component reads the letters upstream of it, at every grade at or below its own". The traced output shows `xi_z <- r` and `xi_tz <- r` (not `w`), apply's `eta_z <- mu, nu_z`, and sweep letters reading their own letter at lower grades, which "upstream" does not cover. In `dependency_graph.png` the exact term reads the manifold-frame tangent group ("reads all four") and the whole plain group. The "Frames and reuse" table says the tangent grade is not reusable across frames, `eta` is not needed, and `nu` needs `S`. The `.dot` file is described as "the same graph" but has no exact term.
- Suggestion: state the generating rule exactly (the output grade is split over the operands; cores absorb only `t`; leaves carry only `z`) and call the product-of-graphs picture a bound. Redraw the exact term in its own frame.

**D5. PROPOSAL and API_IMPACT were not reconciled with the evaluator and the plan.** Severity: medium. Verified.
- Evidence:
  - PROPOSAL "Requirements on the drivers" says "Association order is part of a rule". EXECUTION_PLANS makes it the plan field `path`, honoured only by `sweep`.
  - PROPOSAL layer 3 has wanted grades per rule call. EVALUATOR has wanted components per `evaluate` call. Nothing says how the two relate (see D18).
  - PROPOSAL "Performance" says plain components are "computed once and cached", and API_IMPACT has "One graded cache passed to the whole operations". The decision is `have`.
  - API_IMPACT's headline is "no public signature has to change". "The plan is passed to every call" touches all 51 frontend methods and the `SamplingKind` operations unless a default is defined.
  - The "Order of work" has no step for the evaluator or the plan. It also puts bond indexing and boundary operands at step 4, while the layer-2 code of step 2 (WORKED_EXAMPLE) already uses both.
- Suggestion: one pass over PROPOSAL and API_IMPACT with the decisions applied, and an order of work that includes the driver options, the graph and the sample loop.

**D6. "Pairing holds with no exceptions" is false in a manifold frame.** Severity: medium. Verified.
- Where: PROPOSAL "Bond variables on every bond" and "Combinations that do not exist"; `pairing_check.py`.
- Evidence: plain `mu` lives on `rL` and plain `nu` on `rR`, as the recorded output shows ("bond 1: mu {'': 3, 't': 2} | nu {'': 2, 't': 3}"). `<mu[i], nu[i]>` at the plain and `z` grades fails on shape at every interior bond (`review_1_scratch/pairing_all.py`). The package checks grade `t` only.
- Suggestion: state the identity as: all four grades in the corewise frame; grade `t` in a manifold frame; tails only for plain and `z`. Order-of-work step 1 ("add the edge-pairing identity as a test") should say which.

**D7. "Tails are not gauged" holds only for unsquashed frames, and the default frame is squashed.** Severity: low. Verified. This touches library fit; I tripped over it while testing tails.
- Where: PROPOSAL "Each family is a canonical form of the base point" ("the two families agree on `ranks[0]` and `ranks[d]`. One boundary vector per tail suffices"); API_IMPACT "Custom boundary vectors become possible"; `plan_mockup.block_graph` (`np.ones((W, 1))`).
- Evidence: for a point with TT ranks `(2,5,6,4,3)`, `t3_orthogonal_representations` returns `L` and `R` with tails of size 1 by default (`squash_tails=True`), and with tails `(2, 3)` when `squash_tails=False` (`review_1_scratch/tails.py`, `tails_nosquash.py`). In both cases `apply` and the probe read off the frame match the point to 9e-15, with ones of the frame's own tail sizes. So by default a point and its frame need boundary vectors of different sizes. Squashing absorbs the ones closure, so a non-default boundary vector cannot be combined with it. The mockup's size-1 boundary is right only for squashed frames.
- Suggestion: attach `b_L` and `b_R` to a core family, not to the call. Say which of the two frame conventions the design assumes, and that a tail closed by a fixed vector has gauge freedom (the stabilizer of that vector), which squashing removes.

**D8. Small leftovers.** Severity: low. Verified.
- `correspondence_check.py` prints "prototype has no component there: False False False False". It prints membership, so the recorded line says the opposite of what is true.
- Notation collides: `p` is the highest order, the perturbation vector and the tangent direction (the first two in one sentence of "Taylor jets"); `t` is an order and a grade; `s`, `r`, `B`, `K` and the conversion rule "R" each have two meanings.
- EVALUATOR labels MiB as MB next to an "8 MiB budget".
- "Three savings need no memory at all": two of the three presuppose kept per-sample components.
- "in general have different ranks": only at non-minimal points. The points in `reuse_check.py` and `plan_mockup.py` are rank-deficient, which is why check 1 prints "different shapes" and never compares gauges.
- `FRAME` and `POINT` are the same tuple; `INPUTS` holds `xi`, which is not an input.

### Problems

**D9. A plan sized by `fill` is valid for one operation, but the example passes it to every call and exceeds the budget.** Severity: high. Verified.
- Where: `plan_mockup.fill` and `user_example`; EVALUATOR "What the user writes"; EXECUTION_PLANS "Policies".
- Evidence: `fill` measures only `want=('Hp',)`. Under its plan (`samples=2048`, budget 8 MiB) I measured (`review_1_scratch/mock_probe.py`): the example's own `want=('f','g')+INPUTS` call peaks at 9.24 MiB; `('f','g')` at 8.11 plus 0.75 kept; `('r',)` at 13.89.
- Why it matters: the transient per sample depends on `want` and `have`, and a wanted per-sample component is `W`-sized whatever the block. "The plan is passed to every call" is fine, but one plan for all calls is not.
- Suggestion: a policy takes the set of (`want`, names in `have`) it must serve and returns one plan sized for the worst, or one plan per operation. It must count per-sample outputs, or refuse.

**D10. `fill` extrapolates from small probes and underpredicts.** Severity: medium. Verified. This is a place where the design leans on a small-scale measurement by construction.
- Evidence: the model from blocks 256 and 512 predicts 6.29 MiB at 2048; measured 6.59. The total of 7.34 exceeds the policy's own target of 0.9 x 8 = 7.2, which EVALUATOR's "totals 7.34 MB against its 8 MiB budget" does not mention. The slope is not constant: 2855, 3121, 3280, 3306 bytes per sample between successive doublings from 256 to 4096.
- Suggestion: the policy contract should be "predict, then confirm at the chosen size" (on jax, compile at the chosen block and read the memory analysis), not a two-point line. The same caution applies to `_chunk_slope`-style estimates at target scale.

**D11. The executors are not interchangeable once per-sample components are wanted or handed in.** Severity: medium; it could limit scaling. Verified by reading `over_samples`; the consequences are not verified.
- Where: EXECUTION_PLANS "Three structural rules"; `Plan.executor`; `plan_mockup.over_samples`.
- Evidence: `over_samples` allocates `np.empty((W,) + ...)` on the host and slices `have` by position. A streaming loop cannot materialize that, and a `shard_map` must leave it sharded. `evaluate` takes in-memory arrays. `executor` is a string, so it cannot carry a mesh or a stream, cannot be supplied by a user, and cannot compose, although shard, scan and host stream nest rather than exclude each other.
- Suggestion: define the executor as an object with a two-part contract (reduce to core-sized; map into a sample-aligned store). Say that `have` and wanted per-sample components live in stores aligned with the data. See D17.

**D12. Inputs are not nodes of the graph, and names are not validated.** Severity: medium. Verified (`review_1_scratch/mock_probe3.py`).
- Evidence:
  - `want=('g',)` with `have = FRAME + r` and no `y` fails with "needs y (data) or p". That is the configuration EVALUATOR calls today's behaviour.
  - With `r` in `have` and a different `y`, `y` is silently ignored.
  - A misspelt name in `have` is silently ignored and the component recomputed.
  - An unknown `want` gets the same misleading error.
  - `have` with the wrong number of samples fails late inside einsum.
- Suggestion: make `w`, `y` and the direction leaves of the graph. Validate `want` and `have` against the graph's names before the loop, and check leading sizes.

**D13. A component's identity is not (letter, grade).** Severity: medium. Verified in the mockup.
- Evidence: `Jp.xi_z` against `xi_z`; `nu` through `L` against through `R`; with jets, the highest order (residual jets are reversed relative to it). EVALUATOR files this under "Names for copies", but it decides what `have` may legally contain.
- Suggestion: key components by letter, grade, frame or gauge, and seed. Treat `J^T` as a subgraph parametrized by its seed, which `residual_grade` already is. The same token can carry the point's identity for the open `have`-validity check.

**D14. The mockup's evaluator still contains the reference cycle it warns about.** Severity: low to medium. Verified (`review_1_scratch/mock_probe2.py`).
- Evidence: `visit` in `run` is a self-referencing closure. After one `Hp` call with 8 blocks, `gc.collect()` finds 635 unreachable objects. Memory still allocated after return grows about 13 KB per block. The peak at a fixed block of 256 grows with `W`: 1.03 MiB at `W` = 512, 1.21 at 4096. Two of the three tests listed under "What the mockup taught" (nothing held after a call; peak affine in the block) are not in the package and would currently fail or be marginal.
- Suggestion: make the walk iterative and add the tests.

**D15. Three behaviours of the prototype are not settled for the library version.** Severity: medium to low. Verified.
- Absent or zero. The prototype never touches the centers (NaN centers: all four tangent operations "clean" in `extra_core_experiment`). The layer-2 drivers multiply them by zero carries, as the library does. API_IMPACT says "may or may not". A rectangular scan body can only do the second.
- Undefined combinations are dropped silently (`gmul`, the `None` check). With `rL == rR` and `nU == nD` the prototype returns a grade-`t` core gradient with no error (`review_1_scratch/g_t.py`).
- Zero coefficients are skipped by testing values (`np.any`), which cannot be traced.
- Suggestion: declare existence per (rule, grade, frame kind) and raise. Declare order support per operand. State the zero-carry contract.

**D16. Two further arguments rest on host-numpy or order-0 evidence.** Severity: low. Not verified beyond reading.
- "Declare the graph; do not discover it" is justified by a numpy peak (13.8 to 10.9 MB). Under jit, liveness is the compiler's. The durable reason is that needs are known before computing.
- The default priority (outer block before keeping) cites "one seventh by count" at order 0 for probes. The document itself notes that jets and the exact Hessian differ. Present it as a lean, as the maintainer did.

### Simplifications

**D17. Replace `samples`, `assembly`, `executor`, `orders` and `sites` by one notion: nested loop levels with a block size each.** Severity: medium. Partly verified.
- Evidence:
  - "The decisions" says scan, vectorize and chunk are one knob, yet two fields are integers and two are two-valued strings.
  - The argument for exactly two sample levels uses "A stage can subdivide the block it receives only if it reduces over `W`... the sweeps do not". The package's own `sweep_blocks.py` blocks a sweep, and its scratch falls from 68.0 to 27.5 MB. `over_samples` does the same for mapped outputs.
  - In the mockup the inner level shows no benefit. With `samples=512`, the `Hp` peak is 1.97 MiB with no inner block and 1.93 to 1.98 with `assembly` at 256, 64 or 16; with one outer block it is 13.83 against 13.80 (`review_1_scratch/asm.py`). The evidence for it is the library's chunked jet assemblies, which the mockup does not cover.
  - The stated case for the inner level ("sweeps would run better on a larger one") is a timing claim.
- Suggestion: an ordered list of levels over `W` (stream, shard, compiled block, inner block), each a mechanism plus a size, with the inner block available to any driver call. Give orders and sites a block size where 1 is sequential and `None` is vectorized.

**D18. Make the generated graph the only "layer 3".** Severity: medium. Not verified.
- A node is one rule at one grade with its term list. Its reads are the operands' (letter, grade) pairs. The generator is the product rule.
- Per-call wanted grades and dict-of-grades operands then disappear, and "Generate the graph, do not declare it per case" stops being an open question.
- Test the generator against `component_graphs.py`. I confirmed that its traces do not depend on the value-based skip: output is identical with the skip disabled (`review_1_scratch/symbolic.py`).

**D19. Let the driver pick the bond from the leg letter.** Severity: low. Verified.
- `sitewise` takes `d = min(len(op))`, which silently truncates `mu` (length `d+1`) while `nu` is passed as `nu[1:]`.
- Slot `a` is bond `i` and slot `b` is bond `i+1` in every rule. So `(mu, U, nu)` can be written with no slices and no silent truncation.

**D20. Fewer concepts.** Severity: low. Not verified.
- The seven rules are two schemas per node type: the message out of one leg, and the outer product of all incoming messages.
- The policy memo is unnecessary. The plan is a value the caller holds, so a policy can be a pure function.
- Do not add alternative rules as OR-nodes. Resolve an alternative before the walk ("use the cheap rule if its reads are in `have`"), so `run` stays a plain walk.

### Other improvements

**D21. Close "tails above 1" now.** Severity: medium (a stated requirement, listed as unchecked). Verified. With three edits to a scratch copy of the prototype (tails `(2,...,3)`, ones of the right size, the scalar-residual seed as `c` times `b_R`), all 34 comparisons pass to 2.3e-15 (`review_1_scratch/proto_tails.py`). Add this to the package. State the seed as `c` times `b_R`. The mockup and `reuse_check.py` hard-code size 1 (`np.ones((W, 1))`, `S[d] = np.ones((1, 1))`), which is right only for squashed frames (D7). Boundary vectors need not carry `W`.

**D22. Keep the loss out of the sampling layer.** Severity: medium. Not verified (reading only). The mockup's graph hard-codes `r = z - y` and `f = 0.5 |r|^2`, and it projects `p`, `g` and `Hp` inside `evaluate`. EVALUATOR already says projection belongs outside. Weights and other losses need the residual and seed nodes to be supplied by the fitting layer. Name the summed outputs by what they are (`J^T` of a seed), not `g` and `Hp`.

**D23. Tests to add.** Severity: low. Each follows from a finding above.
- Pairing at all four grades.
- Generator against trace.
- Plan invariance, including `W` not divisible by the block (works: I ran 300 samples in blocks of 64).
- No cycles after a call.
- Peak within budget for every call a policy claims to serve.
- Name validation.
- `Plan` validation at construction. Today `orders='nonsense'` is accepted, `samples=0` fails in `range`, and a `path` typo is a `KeyError` at first use.

## What holds up

- **Recorded output.** `run_all.sh` reproduces `expected_output_3dc15d9.txt` exactly (empty diff), and the older record equals its first ten sections.
- **Taylor recurrences.** I re-derived the reversed-jet adjoint recurrence and the pairing formula by hand; both are correct, as are rules F and R and the "top two rows" statement.
- **Pairing.** It holds at plain, `t`, `z` and `tz` on all 13 edges in the corewise frame, with jets and tails above 1, to 7e-16.
- **Tails above 1.** They work in the prototype (D21).
- **Component graphs.** The traced graphs are structural, not artifacts of random data (D18).
- **Counts.** 21 core-sized and 7 ambient-sized contractions per site for a Gauss-Newton product, and one seventh saved by keeping the frame sweep: correct, and about the same in flops, since the contractions within each class cost about the same.
- **Granularity.** Graph nodes at whole-letter granularity lose nothing against the library: `_apply_transpose_adjoint` also stores `sigma_hats` for all sites, and `_sigma_step` is three separate contractions, as in layer 2.
- **Tables.** All figures in the EVALUATOR and EXECUTION_PLANS tables match the recorded output and the timing record.

## Questions for the authors

1. Are the center cores part of the frame, and which representation is authoritative for the objective?
2. Is a `Plan` one value per run or one per operation? If per run, what must a policy be told?
3. Under `shard_map` or streaming, what type is `have`, and where do wanted per-sample components go?
4. At the centers, is the contract "never read" or "read against a zero carry, must be finite"?
5. Does `current` include a keep set, and is `Plan()` meant to equal it?
6. Are non-default boundary vectors meant to work with manifold frames, given that the default frame construction (`squash_tails=True`) absorbs the ones closure?
7. Is `have` invalidated when the highest jet order changes?
8. "From 13.8 to 10.9 MB" (EVALUATOR, "What the mockup taught"): for which keep set? The earlier evaluator is not in the package, so I could not check it.
