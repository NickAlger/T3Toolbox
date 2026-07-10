# Doctest style — reproducible examples that teach

A house convention for doctests in T3Toolbox. Read this once; apply the **principles**, the mechanics
**within reason**. Reference exemplar: `t3toolbox/manifold.py` (and the reworked `TuckerTensorTrain.__mul__`
/ `inner`). Companion to `docs/signature_style.md`.

## The principle (this is the part that matters)

**Doctests are examples first — their job is to help a user learn the API — held to a
reproducible-and-correct bar so they don't mislead.** They are *not* a test suite: coverage of
branches and edge cases belongs in `tests/`. The payoff of making them reproducible is that they can
be wired into CI, where the doctest run **guards the docs against rot** (it checks that the examples
still produce what they claim) — it does *not* test the code (that's `tests/`' job). So:

- optimize each example for **teaching one thing clearly**, not for coverage;
- but every shown output must **actually reproduce** — a captured `1.04e-13` a reader can't reproduce
  is fiction, and even a *deterministic* captured value rots (a real case we hit: a doc claimed
  `xy.tt_ranks == (1, 6, 6, 1)` when the true value is `(3, 6, 6, 2)`).

A nuance on "not coverage": *illustrating a property* (e.g. that `t3svd`'s singular values are the dense
unfoldings' spectra) is legitimate **pedagogy**, not coverage — provided it is reproducible and limited
to one **representative** case (one unfolding + "the rest are analogous"). Exhaustive verification of a
property across every mode/branch stays in `tests/` or a verification doc.

## Reproducibility (a target, not a hard rule)

Every shown output should reproduce. Achieve it by **seeding** (`np.random.seed(0)`) or using **fixed
inputs**. Escape hatch for the rare case where a non-reproducible value is genuinely the most
informative thing to show: mark the line `# doctest: +SKIP` so it stays illustrative without breaking
a doctest run.

## Mechanics

```python
>>> import numpy as np                                                  # self-contained: imports,
>>> import t3toolbox.tucker_tensor_train as t3                          #   seed, and setup IN the block
>>> np.random.seed(0)
>>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), stack_shape=(2, 3))
>>> y = t3.TuckerTensorTrain.randn((14, 15, 16), (2, 3, 4), (3, 2, 3, 2), stack_shape=(2, 3))
>>> xy = x * y                                    # elementwise product of two T3s
>>> print(np.allclose(x.to_dense() * y.to_dense(), xy.to_dense()))      # value-match -> np.allclose
True
>>> print(xy.tucker_ranks)                        # Tucker ranks MULTIPLY: 4*2, 5*3, 6*4
(8, 15, 24)
>>> print(xy.tt_ranks)                            # and the TT bonds: 1*3, 3*2, 2*3, 1*2
(3, 6, 6, 2)
```

- **Self-contained blocks are the norm.** Each block carries its own imports, `seed`, and setup so a
  user can copy-paste *that block* and run it. Re-using setup from an earlier block is a *case-by-case
  exception*, only when self-containment would make the block excessively long.
- **Value-match → `np.allclose(result, reference)`** → `True`. It is relative+absolute and
  magnitude-robust (matters: e.g. an inner product of magnitude `~3e5`, where an absolute `< 1e-6`
  would read as tight but is loose). Use a plain `bool(norm < bound)` only for a genuine
  **tolerance-bound** claim ("the truncation error is below this bound").
- **Print structure, not raw values.** Shapes, ranks, dims, lengths reproduce *and* teach the return
  contract (especially for stacked returns) — prefer them over printing values. Avoid printing raw
  arrays (NumPy reprs drift across versions); prefer `.shape`, `.tolist()`, a scalar, or a bool.
- **Non-unique or fragile outputs → check the *relationship*, not the values.** Singular vectors /
  cores / orthogonal frames are defined only up to sign/gauge, and any float array from random input is
  digit-fragile across BLAS/platforms — never print them raw. Verify the defining relationship
  (`np.allclose(ss_tt[i], dense_unfolding_svals[:k])` → `True`) and show counts (`len(...)`, a rank).
- **Exact algebraic `0.0`** is fine to show when it's guaranteed (e.g. `(2v - v) - v`); otherwise use
  `np.allclose`.
- **Magnitude in a comment** when it's informative: `True   # residual ~1e-13 (machine precision)`.
- **Small dimensions** (fast, readable) unless a realistic size is itself the point.

## Lossy / approximate operations (truncation)

When the operation is a *deliberate approximation* (truncation/rounding), the `np.allclose(result,
input)` value-match **inverts** — the result is *supposed* to differ. Instead:

- show the **controlled observable** — the reduced/capped ranks (`print(x.tt_ranks, '->', xt.tt_ranks)`);
- assert the **documented error bound**, not equality. For T3-SVD truncation there are *two* (see
  `docs/t3svd_verification.md`): **accuracy** — `||x - xt|| <= sqrt(dropped singular-value energy)` at the
  *chosen* ranks (generalized Oseledets); and **parsimony** — each chosen rank `<= #{ original singular
  values >= tau }`, `tau = max(rtol*||xt||, atol)`;
- to make a **tolerance** (`rtol`/`atol`) example truncate at all, feed a **graded-spectrum** input (a
  smooth sampled function like `1/(i+j+k)`) — a sharp random spectrum truncates nothing or everything;
  don't fake it by forcing a small `max_rank` (that dodges the tolerance behavior).

## Gotchas and failure modes — show them

A short example of a function *failing* on bad input teaches the contract better than prose (and matches
the house "warn, don't enforce, but document the failure mode" philosophy). Two kinds:

- **Structural violation → it raises** → a traceback block, matching the *type* and ignoring the
  message/path:
  ```python
  >>> x.inner(y)   # shapes must match   # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
      ...
  ValueError
  ```
- **Numerical violation → it warns (returns a wrong value)** → show the **wrong value next to the
  right one**, with the cautionary line clearly labelled (`# WRONG: frame not orthogonal, so this is
  not the HS inner product`).

A gotcha is usually its **own short block**.

## Which options/configurations to demonstrate

The tension: an example using option X is gold for a user who wants X, but examples for every
combination explode combinatorially. Resolution (it follows from "examples, not coverage"):

**Demonstrate each distinct *observable behavior* once — varying one option at a time. Never the
cross-product.** That makes coverage roughly linear in the options that actually change something, and
most options change nothing worth showing.

An option **earns an example** when it is (in priority order):
1. the **default / common path** (always — one clean example);
2. an option that **changes the output contract** (shape, type, or semantics) — e.g. `include_shift`,
   `sum_over_probes`;
3. a **gotcha / precondition** (the failure-mode block);
4. **non-obvious** behavior a reader couldn't predict from the signature.

An option **does not** earn one when it has **no observable effect on the result** (`use_jax` — same
numbers), is a **choice among equivalent outputs** (`t3m(method=...)` — all methods return the same
product; one default example + a prose line on *when to pick which*), or is **obvious from the
signature**.

Two levers:
- **One option at a time, against defaults** — show A's effect with B at default, B's with A at default.
- **Combinations only when emergent** — only if the options genuinely *couple* (the result isn't
  predictable from each alone, e.g. `oversample` only matters *for* `method='swap'`). Orthogonal
  options → say "these compose independently" in prose and write no combination example.
- For a **binary behavior-changing** option, a *single* block showing both values and how they differ
  teaches it more efficiently than two blocks.

The long tail lives in the `Parameters`/prose section — **not-doctested ≠ undocumented**. A soft cap of
**~2–5 example blocks per function** forces the prioritization. (Concretely, `t3m`'s big option grid
collapses to ~3 blocks: exact product / truncation-reduces-ranks / the `rtol`+stacked gotcha.)

## Brevity

One idea per block; keep blocks short. Two focused blocks beat one combined block.

## Verifying (and authoring)

- **Never hand-write outputs — run the example and paste the real result.** (Hand-written values are
  exactly how the stale ones crept in.)
- Run a module's doctests: `python -m doctest t3toolbox/<module>.py` (inline directives like
  `+IGNORE_EXCEPTION_DETAIL` / `+SKIP` are honored automatically). To run one function's doctests, use
  `doctest.DocTestFinder().find(obj)` + `doctest.DocTestRunner().run(...)`.
- Filter the leftover debug prints when running (`... 2>&1 | grep -vE "^(RAGGED|NUMPY)"`).

## Within reason

This is a convention, not a contract. Apply it where it earns its keep:
- a non-reproducible value that is genuinely the clearest illustration → `# doctest: +SKIP`;
- a trivial property whose behavior is obvious may need no example at all;
- realism (a genuinely large shape) can override "small dimensions" when the size *is* the lesson;
- a **foundational, math-rich function** (e.g. `t3svd`) earns a richer doctest — the ~2–5-block cap
  bends when the properties themselves (a key correspondence, the error/rank bounds) are the value.

The two accepted costs: doctests are **not yet wired into CI** (so a stale one can still slip in until
that lands — run them when you touch a docstring), and reproducibility occasionally trades a little
informativeness (mitigated by the magnitude-in-comment habit). Both are paid on purpose.
