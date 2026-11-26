# P13 design note: the frontend sampling methods

Implementer, 2026-11-21, against `graded-messages` at `df7422eb` (P12 complete). Read-only so far. P13 is the plan's
P13 (section 5.3) with section 2.5's tables and the corewise update (O-48, O-49).

## 1. The signatures, per class

The rule is N16 with `self` as the object: the sample, then what is applied, then `order=()` (probe and apply only),
then `sum_over_samples` where there is a sum. Each method is one backend call with `self.data` (or `frame.data`),
re-wrapped where the class returns an object. For the method names below, `k` stands for probe or apply; entries
drops `order`.

**`TuckerTensorTrain`** (15 methods; the 6 `*_derivatives*` go):
- `probe(ww, order=())`, `apply(ww, order=())` (`vecs` renamed), `entries(index)`;
- `k_corewise_tangent(ww, core_directions, order=())` (new) returns the type of `z` with `K`;
- `k_corewise_transpose(ww, residual, order=(), sum_over_samples=True)` returns the raw pair of core tuples;
- `k_corewise_hessian(ww, residual, core_directions, order=(), sum_over_samples=True)` (new) returns the pair of
  block tuples;
- static: `probe_ambient_transpose(ww, residual, sum_over_samples=False)`, apply alike,
  `entries_ambient_transpose(index, shape, residual, sum_over_samples=False)`. The default stays off.

**`UniformTuckerTensorTrain`** (12): the same, without the ambient transposes. `ww` may be ragged or packed; probe
outputs, and probe's tangent, mirror its form. The corewise transposes and Hessians return supercore pairs.

**`T3Tangent`** (6; the 6 `*_derivatives*` go):
- `probe(ww, order=())`, `apply(ww, order=())`, `entries(index)`;
- static constructors with the frame first: `probe_transpose(frame, ww, residual, order=(), sum_over_samples=True)`,
  apply alike, and `entries_transpose(frame, index, residual, sum_over_samples=True)`. Each returns a `T3Tangent` at
  `frame`, with the tangent stack `[W] + [K]`.

**`UT3Tangent`** (6): the same, with mirroring on `probe`.

`core_directions` has one form (O-49): the pair `(tucker tuple, tt tuple)`, or two supercores on the uniform class.

## 2. What the frontend checks, and what it leaves to the backend

| Check | Where | Mode |
| --- | --- | --- |
| The sample's form against `order` and `self.shape` | `jets.check_sample(kind, order, sample, shape)`, in every method that takes a sample | always (structural); under jit too (static shapes) |
| A packed sample on the uniform classes | not `check_sample`, which takes the ragged form only; the engine's packed check (`d`, `N`) | always |
| `order` a tuple of ints (`x.probe(ww, 2)` raises a `TypeError`) | `check_sample` | always |
| A `TuckerTensorTrain` passed as `core_directions` | the frontend: a `TypeError` pointing to `y.data` (proposed; the documented trap) | always |
| The residual's shape, the directions' `K + C` and core sizes, and the stacks | the engine (P11, P12a round 1); not repeated | always |
| Frame orthogonality, gauge, same frame | none: the bare `J`, `J^T` and the corewise trio have no numerical precondition (`numerical_contract_catalog.md`) | — |
| The objects themselves | `validate()` in `__post_init__`, unchanged | always |

So safe mode adds nothing in P13. `check_data` is not used on residuals: a residual may carry `K`, which `check_data`
rejects.

**One behaviour changes.** `entries` today takes a nested list (the engine converts it), and `check_sample` refuses
a list. **I propose** that the frontend convert a non-array `index` with `np.asarray` before the check, keeping
today's behaviour. A jax array is left as it is.

## 3. What moves (search of `t3toolbox/`, `tests/`, `docs/`, `examples/`; CI runs all of it)

**Signature moves.** Every frontend transpose and corewise transpose call is rewritten:
- 51 calls of the tangent transposes;
- 47 of the corewise transposes;
- 20 of the ambient transposes;
- the files, beyond the defining modules' doctests: `test_uniform_probing` (24), `test_dispatch` (18),
  `test_tucker_tensor_train` (13), `o1_uniform` (10), `o1_ragged` (6), `test_manifold` (6), `test_fitting` (2).

**The 24 removed methods**, called in:
- 14 test files, the plan's list, which I confirmed;
- 4 frontend modules' docstrings, plus `optimizers.py` and `fitting.py` (2 each);
- 2 pages: `fitting_and_optimization.md` (5 calls, among them lines 562 and 665) and `numerical_contracts.md` (2);
- `chunking.md`;
- 6 examples.

The pages and examples still fit with the old optimizers' derivative data (until P16 and P17). There the data are
made as `taylor_to_derivatives('probe', (n,), x.probe(affine_jet(ww, pp), (n,)))`, and
`taylor_to_derivatives('apply', ...)` likewise. The entries derivatives are made at the one-hot jet, as the plan
says.

**The examples' dense ground truth.** In 5 examples, the per-sample loops over `pd.dense_*_derivatives` become one
`dense_apply` or `dense_probe` call at `affine_jet(ww, pp)` and `(order,)`, then `taylor_to_derivatives`:
- the 3 apply-derivative examples, at lines 124, 118 and 79, inside their local helpers named
  `dense_apply_derivatives`;
- the 2 probe ones, at lines 92 and 99.

The helpers' names change too, so that P20's grep for removed names finds nothing. `fit_probe_display.py` has the
method call only.

**`docs/chunking.md` goes**, with its `design_notes.rst` entry. The links to it must go too, or `sphinx -W` fails:
- 3 markdown links in `fitting_and_optimization.md` (lines 609, 970 and 1245);
- 2 in `glossary.md` (398 and 406);
- the `:doc:` links in `optimizers.py` (lines 167 and 235, and 767), `manifold.py` (1105) and
  `sampling_derivatives.py` (2086, 2815, 2936 and 3048);
- plain `docs/chunking.md` mentions in comments: they don't break sphinx, so they wait for P19, as the plan says.

**`tests/test_probe_derivatives.py`** (nine calls of `dense_*_derivatives`) is folded into the frontend sampling
tests on the dense functions.

## 4. Tests

- **Doctests on the merged methods:**
  - `order=()` gives today's shapes;
  - `(2,)` at `affine_jet(ww, pp)` adds one outermost jet axis;
  - `(1, 1)` adds two;
  - on `probe` and `apply`, the forgotten-`order` trap, wrong against right (O-44).
- **Each method against its backend function**, on all four classes: transposes, corewise transposes and the new
  six. Numpy only.
- **`check_sample` errors** through the methods:
  - the wrong number of modes;
  - a mismatched `W`;
  - too few axes for `order`;
  - an integer `order`;
  - an index that is not an integer;
  - `entries` takes no `order`.
- **The new corewise methods:**
  - each equals the backend function;
  - one doctest each for probe's tangent and Hessian;
  - the Hessian's symmetry `<y, H x> = <x, H y>`;
  - the Hessian is the derivative of the corewise transpose by a polynomial-exact quotient: `J^T c` has degree at
    most `2d - 1` in `x`, so `2d` points give an exact quotient;
  - the uniform ones mirror, and garbage in the padding changes nothing;
  - the `TypeError` for a `TuckerTensorTrain` passed as directions.
- **The sum flag:** each transpose's `False` keeps `W`, and it sums to the default.
- **`test_dispatch`:** the frontend rows by the new signatures, and the six new methods per class, jitted.
- **The removed methods' tests** are rewritten onto the merged methods. Where a test checked a derivative against
  dense ground truth, it now checks `taylor_to_derivatives` of the merged method against `dense_*` at the jet, which
  keeps its meaning.

## 5. What I would change: a split in two commits

**P13a, the signatures (additive and reordering):**
- the four classes get the merged forms and the six new corewise methods per tensor-train class;
- every transpose call site moves;
- `check_sample` is wired in;
- the `*_derivatives*` methods **stay**, untouched.

The examples and pages run unchanged, except for the doctests of the reordered transposes. Review is then about one
thing: the call order.

**P13b, the removals:**
- the 24 methods go;
- every caller moves to the merged method plus `taylor_to_derivatives`: the 14 test files, the module and page
  doctests, and the 6 examples with their dense ground truth;
- `chunking.md` and its links go.

This is the "CI runs everything" commit, and it is mostly mechanical. Each commit leaves the gate green.

Two smaller points:
- the `TypeError` for a `TuckerTensorTrain` as `core_directions` (section 2): cheap, and it names the trap;
- the `entries` list conversion (section 2): keeps today's behaviour rather than breaking it silently in a minor
  version.

Neither is in the plan; both are the frontend's own (inseparable from the classes, by the razor's exception).
