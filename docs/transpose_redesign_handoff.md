# Transpose redesign — work plan / handoff

Resume note for the **ragged** apply/entries transpose redesign. Self-contained: if the session drops,
read this plus [`docs/transposes.md`](transposes.md) (the user-facing rationale) and you can continue.

**Branch:** `main` (ragged work goes here; this is the prerequisite that was found while planning the
uniform layer's slice 6 — fix ragged first, then mirror into uniform).

**Progress:** **Slice 1 DONE** (ambient transposes return CP factors, keep `sum_over_probes`, eye-wart
fixed). **Slice 2 DONE** (corewise transpose `apply_corewise_transpose`/`entries_corewise_transpose`
via the §6.3 `(U,G,G,G)` substitution into `apply_tangent_transpose`/`entries_tangent_transpose`;
returns raw `(tucker_grads, tt_grads)`; backend wrappers in `probing.py`, instance methods on
`TuckerTensorTrain`; tests in `test_tucker_tensor_train` (exact adjoint identity vs the multilinear
forward Jacobian, both sum modes × stacks) + `test_dispatch`; full suite 137 OK, doctests clean).
Convention noted: corewise `c` must be an **array** (shares the tangent backend, which doesn't coerce —
matches `T3Tangent.apply_transpose`'s `np.asarray(c)` convention); the ambient version still accepts a
scalar. **Slice 3 DONE** (probe ambient + corewise transposes — `TuckerTensorTrain.probe_ambient_transpose`
returns rank-`d` CP factors `Σ_i (w0⊗…⊗ž_i⊗…⊗w_{d-1})`, base-free static; `probe_corewise_transpose`
is the §6.3 substitution into `probe_tangent_transpose`, instance, raw `(tucker_grads, tt_grads)`;
backend in `probing.py`; tests + dispatch; full suite 139 OK, doctests clean). The probe residual is
`d` vectors (not a scalar `c`); the ambient back-projection is rank-`d` (not rank-1). **All three ragged
slices complete — the full 3×3 grid (entries/apply/probe × ambient/corewise/tangent) exists in ragged.
Next is the uniform mirror** (uniform port slice 6).

Side note (resolved, separate commit): `test_manifold`'s `test_project_dense_onto_tangent` had a
**pre-existing** flake — a fragile `A @ pinv(A)` reference oracle (default rcond near rank-deficient
`A`'s null singular values), NOT a code/contract bug. Fixed with explicit `rcond=1e-8`; swept the
verified suites (multi-seed) for similar oracles, none found. See the `pinv-oracle-test-fragility` memory.

---

## Why we're doing this (one paragraph)

While planning the uniform layer's transpose slice, we realized the existing
`TuckerTensorTrain.apply_transpose` / `entries_transpose` build the **ambient** back-projection
`c·⊗w` — the literal adjoint of `apply` — but were (a) wrapping it in a `TuckerTensorTrain` when its
natural type is a **canonical (CP) decomposition**, and (b) offering a `sum_over_probes=True` path that,
*as a dense T3*, builds a rank-`|W|` object with `O(|W|³)` superdiagonal copy-tensor cores. Returning
**CP factors instead of a T3** fixes both at once: summing is then free (rank-`|W|` CP is `O(|W|·N)` —
the shared index stays implicit), and the `|W|²` dense-T3 cost moves to `from_canonical` as a user
opt-in. Separately, the operation optimizers actually want is the **corewise (non-manifold) Jacobian
transpose** of paper §6.3 — the gradient w.r.t. the cores, shaped like the cores, no `|W|` blow-up.
§6.3 shows it's the tangent transpose (§6.2.3 / Algorithm 8) under the substitution `Pᵢ,Qᵢ,Oᵢ → Gᵢ`
with `Uᵢ` no longer required orthogonal. We **verified this holds in our code** (see below). Full
reasoning and the three-transpose taxonomy are in `docs/transposes.md`.

## Decisions locked

1. **Rename** the existing free-tensor transposes to **`apply_ambient_transpose` / `entries_ambient_transpose`**
   (explicit, because users wrongly expect the corewise version as the default — Nick included).
2. **Ambient returns a raw canonical (CP) factor tuple, NOT a `TuckerTensorTrain`.** The natural type
   of the adjoint of a multilinear form is CP (`apply`: factors→scalar; ambient transpose:
   scalar→factors). Implementation = **return the `factors` the backend already builds, drop the
   `t3_from_canonical` wrap** (which removes the copy-tensor code path entirely). Factors follow the
   `from_canonical` convention (`len=d`, `factor[i].shape = stack_shape+(R,Nᵢ)`), so
   `from_canonical(...)` round-trips to a T3 if the user wants one. Rationale for raw tuple (not a
   `Canonical` class): this library is **laser-focused on T3, not a general CP library**; the raw tuple
   is the clean interop boundary with dedicated CP libraries.
3. **KEEP `sum_over_probes` on the ambient transpose** (this reverses an earlier "drop it" call — the
   reversal is *because* of the CP decision). In CP both modes are cheap: `False` → `W`-stack of rank-1
   (`R=1`); `True` → one rank-`|W|` CP, `O(d·|W|·N)`. **No `|W|³`/`|W|²` blow-up** — CP keeps the shared
   rank index implicit. The `|W|²` cost of a *dense T3* now lives in `from_canonical` (CP→T3), incurred
   only as the user's explicit opt-in. So: don't form a T3 in the transpose at all; never build the
   copy tensor.
4. **Fix the eye wart**: `entries_ambient_transpose` must build the one-hot CP factors by direct
   scatter, not `eye(N)[index]` (which is `O(N²)`).
5. **Add `apply_corewise_transpose` / `entries_corewise_transpose`** as **instance** methods on
   `TuckerTensorTrain` (`self` is the base), via the §6.3 substitution. Keep `sum_over_probes` here
   (`True` = Adam/L-BFGS gradient `Jᵀr`; `False` = per-probe stack, for `JᵀJ`). `entries_corewise_transpose`
   needs **no `shape` arg** (it comes from `self.shape`), unlike the ambient version.
6. **Return type of corewise = raw tuple `(tucker_grads, tt_grads)`** matching `.data` layout (it's a
   gradient, not a tensor; a `TuckerTensorTrain` return would make `X - lr*grad` silently wrong).
7. **`T3Tangent.apply_transpose` / `entries_transpose` are unchanged** (the tangent flavor; class name
   already implies "tangent"). They keep `sum_over_probes`.
8. **Backend placement:** corewise wrappers go in `backend/probing.py` next to `apply_tangent_transpose`
   (same §6 family); ambient stays in `backend/apply.py` / `backend/entries.py`.

## Central assumption — VERIFIED

The §6.3 substitution computes the corewise gradient in our code. Check (central finite differences vs
the corewise adjoint identity): **rel err 1.2e-11**, output shapes match the cores exactly, summed
(`sum_over_probes=True`, 7 probes) correct. Script at `/tmp/verify_corewise.py` — reproduce / promote
to a test:

```python
import numpy as np
import t3toolbox.tucker_tensor_train as t3mod
import t3toolbox.backend.probing as probing
np.random.seed(0)
X = t3mod.TuckerTensorTrain.randn((4,5,6), (2,3,2), (1,2,2,1))
tucker_cores, tt_cores = X.data
M = 7
ww = [np.random.randn(M, N) for N in (4,5,6)]
c  = np.random.randn(M)
base = (tucker_cores, tt_cores, tt_cores, tt_cores)        # §6.3: up=U, down=left=right=G
dU, dG = probing.apply_tangent_transpose(c, ww, base, sum_over_probes=True)
# adjoint identity:  sum_cores <grad, dir> == sum_m c_m * d/de apply_m(X + e*dir)
dU_dir = [np.random.randn(*u.shape) for u in tucker_cores]
dG_dir = [np.random.randn(*g.shape) for g in tt_cores]
eps = 1e-6
mk = lambda s: t3mod.TuckerTensorTrain(tuple(u+s*eps*du for u,du in zip(tucker_cores,dU_dir)),
                                       tuple(g+s*eps*dg for g,dg in zip(tt_cores,dG_dir)))
deriv = (np.asarray(mk(+1).apply(ww)) - np.asarray(mk(-1).apply(ww)))/(2*eps)
rhs = float(np.sum(c*deriv))
lhs = sum(float(np.sum(a*da)) for a,da in zip(dU,dU_dir)) + sum(float(np.sum(a*da)) for a,da in zip(dG,dG_dir))
assert abs(lhs-rhs)/abs(rhs) < 1e-8
```

---

## Slice 1 — ambient: rename + return CP factors (keep `sum_over_probes`) + eye fix

Self-contained refactor; do first.

- **Backend** `t3toolbox/backend/apply.py`: rename `tucker_tensor_train_apply_transpose →
  tucker_tensor_train_apply_ambient_transpose`. **Return `factors` directly** instead of
  `t3_operations.t3_from_canonical(factors)` — the function already builds the CP `factors` in both
  branches (`sum_over_probes=False`: `stack=W+C, R=1`; `True`: `stack=C, R=|W|`); just drop the wrap.
  Keep `sum_over_probes` (both branches stay). Update the return type / shape comments to the CP factor
  convention (`len=d`, `elm_shape=stack_shape+(R,Nᵢ)`) and `__all__`. The `t3_from_canonical` import may
  become unused here.
- **Backend** `t3toolbox/backend/entries.py`: rename `…entries_transpose → …entries_ambient_transpose`;
  it still delegates to the ambient apply backend (so it now returns CP factors too). Replace
  `ww = eye(N)[index]` with a direct scatter (`zeros((|W|,N)).at[arange,index].set(1)` for jax /
  `arange==index` broadcast for numpy — infer backend as usual). Update `__all__`.
- **Frontend** `t3toolbox/tucker_tensor_train.py`: rename the two `@staticmethod`s to
  `apply_ambient_transpose` / `entries_ambient_transpose`; **return the raw factor tuple** (drop the
  `TuckerTensorTrain(*…)` wrap). Keep `sum_over_probes`. Rewrite docstrings: returns CP factors (not a
  T3), `from_canonical(...)` to realize a T3; update the adjoint-identity doctest to wrap with
  `from_canonical` (or compare via the dense tensor built from the factors). `entries_ambient_transpose`
  keeps its `shape` arg.
- **Blast radius:** grep `apply_transpose`, `entries_transpose` across the repo — frontend + backend +
  `tests/` + doctests + any examples. (The `T3Tangent.*_transpose` names are NOT renamed; be precise so
  you don't touch them.) The return type changed (T3 → factor tuple), so update every caller/test that
  expected a `TuckerTensorTrain`. Watch `docs/batching_and_stacking.md` §11,
  `docs/apply_entries_handoff.md`, `docs/entries_apply_probe.md` for prose referencing the old
  names/return type.
- **Tests:** update `tests/test_tucker_tensor_train.py` transpose tests to the new names and CP-factor
  return; keep the adjoint identity (via `from_canonical`) for both `sum_over_probes` modes; assert the
  summed factors are `O(|W|N)` (rank axis = `|W|`, no copy tensor). Add/keep jit dispatch coverage.
- **Run:** `python -m unittest tests.test_tucker_tensor_train tests.test_manifold tests.test_dispatch 2>&1 | grep -vE "^(RAGGED|NUMPY)"`.
- **Commit** Slice 1.

## Slice 2 — corewise transpose (new)

- **Backend** `t3toolbox/backend/probing.py`: add
  - `apply_corewise_transpose(c, ww, core_pair, sum_over_probes=False)` where `core_pair =
    (tucker_cores, tt_cores)`; body = `apply_tangent_transpose(c, ww, (tucker_cores, tt_cores,
    tt_cores, tt_cores), sum_over_probes)`. (Razor: backend users get corewise without knowing the
    substitution.)
  - `entries_corewise_transpose(c, index, core_pair, sum_over_probes=False)` analogously via
    `entries_tangent_transpose` (note: it derives `ww`/`xis` from the base internally — confirm the
    substitution path works the same; it slices `Uᵢ = tucker_cores` fibers).
  - Add both to `__all__`. Follow the signature-shape-comment style (`docs/signature_style.md`).
- **Frontend** `t3toolbox/tucker_tensor_train.py`: add instance methods
  - `apply_corewise_transpose(self, c, ww, sum_over_probes=False) -> ((tucker_grads), (tt_grads))`
  - `entries_corewise_transpose(self, c, index, sum_over_probes=False) -> ((tucker_grads), (tt_grads))`
    — delegate to the backend with `self.data`. Return the **raw tuple** (document: gradients w.r.t.
    cores, NOT a tensor). Add reproducible doctests (adjoint identity; shapes match cores).
- **Tests:** new `tests/test_tucker_tensor_train.py` cases — the adjoint identity (promote the verify
  script), shapes match cores, `sum_over_probes` True/False, across structures × stack_shapes, numpy
  ground truth. Add jit dispatch coverage in `tests/test_dispatch.py` if the corewise path isn't
  already exercised (it routes through `apply_tangent_transpose`, which likely is — confirm).
- **Verify** dense oracle too (not just the adjoint identity): build the modified-core tensors and
  compare, per CLAUDE.md ("dense ground truth").
- **Run** the full relevant suite (see Slice 1) + `backend/test_contractions`. **Commit** Slice 2.

## After ragged is solid

Return to the uniform layer (the original task). The uniform port's **slice 6** (`docs/uniform_port_plan.md`,
`docs/uniform_slice_handoff.md`) now mirrors these into uniform: a `ut3` ambient transpose returning CP
factors (keep `sum_over_probes`) and a uniform corewise transpose (substitution into the uniform tangent
backend, once that exists). Keep the same names/semantics. Note the uniform ambient transpose returns
*uniform* CP factors (`(d,)+stack+(R,N)` supercore + mask) — the CP-factor convention carries over.

## Conventions (don't re-derive)

- Dispatch by input inference (no `use_jax` threading) except pure constructors. einsum with leading
  `'...'`; numpy passes `optimize=path`, jax omits.
- Signature-shape-comment style: `docs/signature_style.md`. Doctest style: `docs/doctest_style.md`
  (run the example, paste real output).
- Tests numpy-only for numerical correctness; jax invocation via `test_dispatch`.
- Commit per slice after the suite is green; stage only relevant files (leave stray
  `.npz`/`.idea`/`conf_OLD.py`/`t3*_file*.npz` untracked); end messages with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Scripts from `/tmp` need `PYTHONPATH=/home/nick/repos/T3Toolbox`.
