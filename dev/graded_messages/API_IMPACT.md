# API impact

> **Status after the review of 2026-11-14.** The survey and its counts stand, at commit `89ac203f`. The design they were measured against changed in places; read [DESIGN_AFTER_REVIEW.md](DESIGN_AFTER_REVIEW.md) first. This document has not been rewritten.
>
> What changed:
>
> - **"One graded cache passed to the whole operations."** Held variables are keyword arguments, and the plan is a `SamplePlan`.
> - **Public additions now planned:** `sampling_variables`, `SamplePlan`, `SampleScan`, `chunk` and `unchunk` on sampling kinds, and a `plan` argument with a default on `LocalModel` and the frontend. `chunk_size` leaves the sampling kind in stage 2, which breaks callers that pass it to a kind.
> - **The jet normalization** need not change for users.
> - **A new acceptance requirement:** a consumer must be able to use every addition without rewriting the optimizers or changing T3Toolbox.
>
> Missing from the adapter list, from the reviewers (F19): the static split or a pytree registration for held variables; a `W`-slice for held variables; block sums; dispatch rows for each new public backend function; a run on the jax 0.4.30 / numpy 1.26 floor; and the test suites that scan the source for subscript strings (F1).
>
> Known errors, from the reviewers (F14): the uniform sweep wrapper is a 4-tuple for plain kinds and a 5-tuple for jets. The exact-equality survey omits `tests/backend/test_uniform_curvature.py` (36), `tests/test_sharing.py` (24) and `tests/test_optimizers_frontend.py` (10). The list of tests naming sweep-split functions omits `tests/backend/test_uniform_fitting.py`.

With adapters, no public signature has to change. What moves is rounding-level output, the memory profile, and a few tests that look inside the kernels. Removing the redundant layer instead deletes about 94 of the 166 public backend sampling names and changes one user-visible convention, the jet normalization.

Surveyed at commit `89ac203f`. Counts come from `__all__` lists and text searches, not from running the test suite. Companion to [PROPOSAL.md](PROPOSAL.md).

## Surface surveyed

The sampling kernels are consumed by exactly two modules outside the six sampling modules, so the blast radius inside the package is small. The public surface that names them is large.

| Layer | Surface | Count |
| --- | --- | --- |
| Frontend | Sampling methods on `TuckerTensorTrain` | 15 |
| Frontend | Sampling methods on `T3Tangent` | 12 |
| Frontend | Sampling methods on `UniformTuckerTensorTrain` | 12 |
| Frontend | Sampling methods on `UT3Tangent` | 12 |
| Frontend | Model factories, `LocalModel`, optimizers | 6, 1, 4 |
| Backend | Public names in the six sampling modules | 166 |
| Backend | of which edge-variable kernels (`compute_*`, `assemble_*`, `build_input_jets`) | 46 |
| Backend | of which sweep-split functions (`*precompute*sweep*`, `*_from_sweep`) | 48 |
| Backend | of which whole operations, validators, dense oracles, chunk estimators | 72 |
| Backend | `SamplingKind` classes, each with six operations | 6 ragged, 6 uniform |
| Tests | Files that reference kernel-level names | 9 files, 229 lines |
| Docs | Files that describe the names or conventions | about 10, including two TeX notes of 646 and 1071 lines |
| Examples | Scripts that fit derivative data | 7 |

The six sampling modules are `probing`, `apply`, `entries`, `sampling_derivatives`, `ut3_sampling` and `utv_sampling`. The two outside consumers are `backend/fitting.py` and `backend/uniform_fitting.py`, through the `SamplingKind` operations.

The API reference calls the backend "a first-class surface", so the 166 backend names are public, not internal.

Per-module breakdown of `__all__`:

| Module | Names | Edge-variable kernels | Sweep-split functions | Whole operations and other |
| --- | --- | --- | --- | --- |
| `probing` | 34 | 23 | 4 | 7 |
| `apply` | 10 | 0 | 4 | 6 |
| `entries` | 10 | 0 | 4 | 6 |
| `sampling_derivatives` | 57 | 23 | 12 | 22 |
| `ut3_sampling` | 25 | 0 | 6 | 19 |
| `utv_sampling` | 30 | 0 | 18 | 12 |

Test files that reference kernel-level names (lines matching): `tests/_jet_trs_twins.py` (21), `tests/backend/_curvature_oracles.py` (78), `tests/backend/test_curvature.py` (55), `tests/backend/test_from_sweep_reuse.py` (13), `tests/backend/test_uniform_curvature.py` (8), `tests/test_dispatch.py` (10), `tests/test_jet_recurrence.py` (37), `tests/test_uniform_probing.py` (6), `tests/test_contractions_interpreter.py` (1).

## Case 1: unchanged with no adapter

Everything a frontend user touches keeps its signature, shapes and meaning, because the whole-operation functions are re-implemented as seeds plus a readout behind the same names.

- **All 51 frontend sampling methods.** Arguments, output shapes and the `W + K + C` stacking are as now.
- **The six model factories, `LocalModel` and the four optimizers.** Raw residuals and the `omega[mode, order]` weight keep their meaning.
- **The `SamplingKind` interface.** The six operations keep their signatures, so user-written kinds keep working. The sweep stays whatever a kind's own `precompute` returns.
- **Frame data.** The `(D, U, L, R)` tuple, `d` cores per family and rank tuples of length `d+1` are exactly what the proposal wants.
- **Tails.** The class docstring already defines the tail closures as vectors of ones, which is the proposed default boundary vector.
- **Ambient transposes, dense oracles and input validators.** They are outside the sweep machinery.
- **Uniform boundary code.** Masking, packing, unpacking and re-masking in `ut3_sampling` and `utv_sampling` stay; only the functions they call change underneath.
- **`contract`.** It is used as it is.

## Case 1: the adapters

Nine adapters preserve every public backend name and format. Two of them, the kernel shims and the sweep tuples, are the bulk of the work.

| Adapter | Covers | What it converts |
| --- | --- | --- |
| 1. Jet output scaling | Forward `*_derivatives` operations and their `from_sweep` forms | Multiplies the order-`t` Taylor coefficient by `t!` |
| 2. Residual seeding | Transposes and Hessian terms | Residual jet to reversed, `(p-k)!`-scaled seed, composed with the `omega` weight; scalar residual to the `z` component of `b_R` |
| 3. Edge-variable kernel shims | The 46 public kernels | Old argument order to one rule at one grade. On output: old name, boundary or terminal entry dropped, right-swept index shifted by one, rule F or R applied |
| 4. Frame-sweep tuples | The 48 sweep-split functions | Lean `(xis, mus)` and full `(xis, mus, nus, etas)` tuples, and their jet forms, to and from the graded container. The uniform 4-tuple wrapper around them stays |
| 5. Frame view | Every operation that takes a frame | `(D, U, L, R)` to pointed cores. `t3_corewise_frame` is unchanged |
| 6. Driver options | `sum_over_probes`, `chunk_size`, `n_probe` | Mapped onto the per-site driver. `n_probe` is accepted and ignored |
| 7. Chunk-size estimators | `estimate_chunk_size`, `max_chunk_size_within`, and `chunk_size='auto'` in the optimizers | Same signatures. The in-situ slope measurement follows the new kernels; the closed-form floor models are re-derived |
| 8. Sharding-test vocabulary | `tests/test_contractions_sharding.py` | An enumerator of the generated grouped strings, replacing the scan for string literals |
| 9. Consistency check | `T3Frame.is_consistent` | Same signature, non-dense implementation |

Notes on the two large ones:

- **Kernel shims.** Special cases are few. `compute_dxi_tilde_jets` and the `dxi_hat` combine return two rows, which are the top two coefficients of `eta_z`. The assembly kernels take `sum_over_probes`, `n_probe` and `chunk_size`.
- **Sweep tuples.** The built-in kinds should hold the native container and never convert per matrix-vector product. The tuple adapter is only for callers of the public `precompute` and `from_sweep` functions. The Hessian `from_sweep` functions document accepting either a probe sweep or a lean apply sweep and read only its first two entries, so the adapter must accept both.

## Case 1: what still changes

Adapters preserve names and formats, not bits, memory or compile behaviour. These are the places a behaviour-preserving port can still fail a test or surprise a user.

| What changes | Why | Where it shows |
| --- | --- | --- |
| Results at rounding level | The operations run in a different order, and jets are scaled by `t!` at the boundary instead of inside | Exact-equality assertions: 37 in `tests/backend/test_curvature.py`, 17 in `tests/test_fitting.py`, 9 in `tests/test_uniform_probing.py`. Tolerances of 1e-13 to 1e-15: 39 in `test_curvature.py` |
| Compile and trace counts | Step bodies are generated and memoized on a specification, not hand-written | 8 test files mention compile or cache counts, including `test_dispatch.py` and `test_chunk_size_estimator.py` |
| Memory and time profile | Different intermediates, and bond arrays of length `d+1` | The default chunk size of 100 and any tuned chunk sizes |
| Pytree structure of `LocalModel.sweep` | The built-in kinds hold the native container | Code that maps over, serializes or compares a model's leaves |
| What kernel-level tests exercise | The shims call the new rules through a conversion | `tests/_jet_trs_twins.py`, `tests/backend/_curvature_oracles.py`, `tests/test_jet_recurrence.py` keep passing but now test rule plus conversion |
| Non-finite centers | The new sweeps may or may not multiply a center by a zero carry | Today NaN centers poison the four forward tangent operations and not the transposes |
| Noise floor of `is_consistent` | Difference norm instead of dense subtraction | About 1e-12 instead of 1e-15. The default `rtol` of 1e-9 is unaffected |

The maintainer's testing policy (stated 2026-11-14) settles how to treat the first row. Numerical operations are tested to tolerance and bitwise tests are for structural properties, so an exact-equality assertion on a numerical operation is drift. Those assertions need sorting into structural and numerical, and the numerical ones become tolerance tests. Rounding-level changes from a port are acceptable as long as the algorithm stays numerically stable.

The assertions were counted by text search; which of them compare against values the port would change was not checked. The first two rows are therefore the upper bound of what to review, not a list of failures.

One thing does not change unless chosen: inconsistent hand-built frames. `T3Frame.to_t3()` keeps returning the right-canonical point and the geometry keeps using the left one until an authoritative family is named.

## Case 2: what would be removed

These are the things that would not exist had the design started from graded rules. Only the first row changes what a frontend user sees; the rest remove backend names, internal machinery, tests and documentation.

| Removed | Replaced by | What is lost |
| --- | --- | --- |
| Derivative-normalized jets in public inputs and outputs | Taylor coefficients, with residual weights stated in Taylor coordinates | Outputs no longer read as derivatives, and the `*_derivatives` names become misnomers. The 7 example scripts, stored derivative data and existing weights need a one-time `t!` rescale |
| The 46 edge-variable kernels as public functions | Eight rules called with a grade | Name-level correspondence with Algorithms 5 to 8 of the paper. Per-kernel docstrings and references. Calling or testing one recursion by name |
| The tilde, hat, `d` and dot vocabulary, and the letters `sigma` and `tau` | Four letters plus a grade suffix | Glossary entries and continuity with the paper's notation |
| Lean and full frame-sweep tuples, and the 48 sweep-split functions | One graded cache passed to the whole operations, with wanted grades | Documented tuple formats a user can build or inspect. The explicit statement that apply needs `(xi, mu)` and probe needs all four |
| Plain kernels in `probing.py`, `apply.py`, `entries.py` beside the jet kernels | Order 0 of the jet kernels | A simpler order-free path to read. An independent implementation that cross-checks the jets at order 0. Any speed advantage of the plain path, which has not been measured |
| Separate kernel families per sample type, and `sigma_hat` as its own algorithm | Boundary data: the leaf type and where the residual enters | The module-per-sample-type layout and the containment narrative built on it |
| Binomial machinery: weighted shifts, inverse factorials, and the tests `_jet_trs_twins.py` and `test_jet_recurrence.py` | Truncated polynomial multiplication | Reference twins built on the binomial tensor. `symmetric_probe_derivatives.tex` would be rewritten |
| Hand-derived dotted kernels and their term-by-term oracles (`_curvature_oracles.py`, 546 lines) | The graded product at grade `tz` | Regression failures localized to one term. An independent derivation that pins the product rule today. `exact_hessian_two_hole.tex` would shrink |
| Ragged and uniform twin strings, `tt_reverse` mirroring, summed and unsummed twin bodies, `len_C`, `len_W` and `n_probe` | Drivers and operand signatures | Subscript strings that can be found by searching the source, which the sharding test relies on |
| Core-indexed right-swept arrays | Bond-indexed arrays of length `d+1` | Zipping per-site operands with no slice |
| Hard-coded ones, zeros and terminal sums | Boundary-vector operands | Nothing. Custom boundary vectors become possible |
| Centers described as padding, the dense `is_consistent`, two readers of the base point | Canonical forms with one authoritative family | The current result of `to_t3` or of the geometry on an inconsistent frame, whichever family is not chosen |
| Per-kernel chunking helpers and the per-kind `chunk_size` field | A chunking option of the per-site driver | Nothing in the API. The estimators' closed-form memory models are re-derived |
| One-hot vectors in the entries transposes | A scatter-add | The bitwise containment that `test_curvature.py` asserts today, entries equal to apply at one-hot vectors |
| Six `SamplingKind` classes per layer | One kind parametrized by leaf type, output type and order | Nothing if `APPLY`, `ENTRIES` and `PROBE` remain as aliases. This one is optional |

By count, the second and fourth rows are 94 of the 166 public backend sampling names.

Four losses are real rather than cosmetic:

- **The paper correspondence.** Today a reader can go from Algorithm 7 to `compute_sigma`. Afterwards the link is a table.
- **Independent oracles.** The hand-derived kernels and their twins are a second derivation of the same mathematics. Generated terms cannot check themselves, so the dense operation-level oracles carry more weight.
- **Derivative-valued outputs.** Users think in derivatives. A thin conversion at the frontend keeps that at the cost of one scaling per call.
- **Inspectable sweeps.** A tuple of named arrays is easier to debug than a graded container, unless the container prints itself well.

## Case 2: what stays, and what the survey missed

Most of the public surface is not redundant under the new design and would exist either way.

### Stays

- **The 51 frontend sampling methods and the six model factories.** The plain and derivative frontends overlap at order 0, but they are conveniences worth keeping.
- **The whole-operation backend functions**, `t3_*` and `tv_*` with their uniform mirrors. They become thin choices of seeds and readout.
- **The corewise transposes.** They are already one-line calls of the tangent transpose at the corewise frame.
- **Ambient transposes, dense oracles and input validators.**
- **The `SamplingKind` extension interface.**
- **The chunk-size estimators as an API**, and `chunk_size='auto'` in the optimizers.
- **The uniform masking and packing layer.**
- **`sum_over_probes` and the tangent stack `K`** on residuals.

### Not covered by this survey

- **Code outside the repository.** Anything downstream that calls backend kernels by name.
- **The padded and aligned frame modules.** `padded_frames.py` and `uniform_padded_frames.py` mention padding and the boundary cores often. How they treat the centers was not read.
- **`dev/` tooling**, such as the rename scripts and the sharded-fitting plan.
- **Test outcomes.** The suite was not run against any change. The counts are of text matches.
