# Scan-body sweep — the catalogue

*Working note for the in-flight conversion to closure-free scan/map bodies
(`docs/contributor/scan_body_principles.md`). Ephemeral: delete or archive when the sweep is done.
Line numbers are against the tree as of 2026-08-21, AFTER `_mu_jets_step` landed.*

## Status

1 of 54 sites converted (`sampling_derivatives.py:394`, the exemplar). The catalogue below came from
a six-way read-only fan-out; free-variable lists are from compiled code objects, and reachability
was confirmed by instrumenting the `get_backend` seam and running the ops, not by grep alone.

## How this work proceeds (agreed with Nick, 2026-08-21)

Deliberately staged, not a batch edit:

1. **Workshop a few sites one at a time**, in back-and-forth discussion, the way `compute_mu_jets`
   was done — analysis, then a proposed diff in chat, then agreement. One or two of each category.
2. **Then survey the remainder alone** and surface the most uncertain / borderline ones; workshop
   one or two of those.
3. **Then write a plan document** using what the workshop established.
4. **Only then edit on my own.**

## Reproducing the measurements

No script survives in the repo; the technique is short enough to rebuild. Count compiles by
attaching a logging handler and enabling `jax_log_compiles`; count leaked mappings from
`/proc/self/maps`:

```python
import logging, jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_log_compiles', True)
N = [0]
class C(logging.Handler):
    def emit(self, r):
        if 'Compiling' in r.getMessage(): N[0] += 1
logging.getLogger('jax').addHandler(C()); logging.getLogger('jax').setLevel(logging.DEBUG)
def maps(): return sum(1 for _ in open('/proc/self/maps'))
```

To attribute calls to sites, wrap T3Toolbox's own seam rather than jax internals (patching
`jax.lax.*` distorts attribution): replace `common.get_backend` with a version returning wrapped
`xscan`/`xmap` that record `sys._getframe(1)` and the body's `__code__.co_filename:co_firstlineno`,
and rebind it in every already-imported `t3toolbox.*` module. Track `id(body)` while holding a
reference to each body, or ids get recycled and every site looks stable.

For exact captures, compile the module and walk code objects — never read them off by eye:

```python
def walk(co, out):
    out[(co.co_name, co.co_firstlineno)] = co.co_freevars
    for c in co.co_consts:
        if hasattr(c, 'co_code'): walk(c, out)
```

## Gates for every change

- `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q` — baseline is 726 passed / 41,976 subtests, ~6 min.
- `python -m doctest $(ls docs/*.md docs/contributor/*.md | grep -v doctest_style.md)`
- `python -m sphinx -W --keep-going -b html docs <out>` — slow (~10 min), autoapi over the package.
- Numerics: compare against the pre-change implementation on **both** backends and on any degenerate
  branch (e.g. order-0). `_mu_jets_step` was verified bit-identical (0.0 max abs error) that way.

## Two mechanism findings that shape the plan

**1. `jax.lax.map` never caches — hoisting an `xmap` body alone does nothing.** From jax's source:

```python
g = lambda _, x: ((), f(x))     # a FRESH lambda every call
_, ys = scan(g, (), xs)
```

Measured, 5 calls each: `lax.scan` with a stable module-level body → **0** compiles; `lax.map` with
a stable module-level body → **5**. So `xmap` sites need the seam fixed as well: replacing
`common.py`'s `jax_map = jax.lax.map` with a scan-based map holding a **weak-keyed** wrapper cache
(measured: stable body → 0 compiles, fresh body → 5, numerics identical to `lax.map`). Weak keys
matter — a strong cache would pin every throwaway body, the array-pinning failure the principles doc
warns about, which is why jax itself uses `weakref_lru_cache`.

**The seam fix and the body hoist only pay together.** Applying the seam fix alone to the real
library changed nothing (4 compiles before and after) because the bodies are still fresh; hoisting
an `xmap` body alone changes nothing because `lax.map` discards the identity. Sequence them as one
unit, or the intermediate state measures as a no-op and looks like a failed fix.

**2. For `scan`, a stable outer body subsumes a fresh inner one.** Measured: `lax.scan` with a
stable outer body that creates a fresh inner closure → **0** compiles, because a cache hit means the
outer body's Python never runs and the inner closure is never created. This is what makes several
category-D inner scans second-order — *but only where the outer is a `scan`*. Where the outer is a
`map` it does not apply until finding 1 is addressed.

## The catalogue

`jax?` = reaches a real `lax.*` primitive. Sites marked `ragged` sit in an `else:` branch of
`if is_uniform:` (or a hardcoded `is_uniform=False`), so their `xmap` is always `ragged_map`, a
Python loop — the defect cannot occur there and conversion is consistency-only.

| file | call | body | captures | cat | jax? | note |
|---|---|---|---|---|---|---|
| tt_orthogonalization.py | 50 | 42 `_left_func` | — | A | **yes** | hottest measured (26 calls); uniform frame build + every uniform retraction |
| probing.py | 228 | 219 `_func` | — | A | **yes** | reached by ~every op in the file (16 calls) |
| probing.py | 395 | 387 `_func` | — | A | **yes** | wraps module-level `_sigma_step` |
| probing.py | 791 | 787 `_step` | — | A | **yes** | apply + entries transpose |
| probing.py | 866 | 853 `_func` | — | A | **yes** | probe transpose |
| apply.py | 66 | 57 `_func` | — | A | **yes** | `t3_apply` |
| apply.py | 341 | 334 `_func` | — | A | **yes** | apply + entries forward J |
| entries.py | 72 | 57 `_func` | xnp, n_idx | B | **yes** | `n_idx = ind.ndim` |
| tt_operations.py | 175 | 168 `_func` | xnp | B | **yes** | 2 compiles per uniform tangent projection |
| ut3_svd.py | 277 | 245 `_step` | n, r, stack_shape, xnp | B | **yes** | inside `utv_retract` → Armijo backtracking |
| ut3_svd.py | 371 | 353 `_tt_step` | n, r, stack_shape, xnp | B | **yes** | same, `sharing=` path |
| ut3_linalg.py | 140 | 134 `_push` | xnp | B | **yes** | `inner`/`norm` |
| utv_operations.py | 502 | 495 `_tt_step` | xnp | B | **yes** | `project_oblique` |
| sampling_derivatives.py | 934 | 926 `_func` | order, s_size, tvec, xnp | B | **yes** | closest analogue to the exemplar |
| sampling_derivatives.py | 1096 | 1089 `_func` | s_size, trs_push | B | **yes** | apply/entries derivatives |
| sampling_derivatives.py | 1144 | 1137 `_func` | trs_push | B | **yes** | apply/entries J |
| sampling_derivatives.py | 1962 | 1958 `_step` | s_size, trs_xi | B | **yes** | apply/entries Jᵀ |
| sampling_derivatives.py | 1439 | 1432 `_step` | order, s_size, svec, trs_r, xnp, xscan | B\* | **yes** | outer scan; subsumes 1415 |
| sampling_derivatives.py | 1415 | 1408 `_src_step` | P, deta_t, nf | D | **yes** | inner scan of 1439 — second-order |
| sampling_derivatives.py | 514 | 498 `_func` | order, trs_r, xnp, xscan | C | **yes** | **xmap** — needs the seam fix |
| sampling_derivatives.py | 511 | 504 `_accumulate` | G, nu_jet | D | **yes** | inner scan of 514 |
| sampling_derivatives.py | 868 | 843 `_func` | order, trs_r, xnp, xscan | C | **yes** | **xmap** — needs the seam fix |
| sampling_derivatives.py | 865 | 853 `_step` | P, Q, dG, nf, nu_jet, tau_jet | D | **yes** | inner scan of 868 |
| sampling_derivatives.py | 1640 | 1637 `_step` | assemble_one, cs, jax, ops, w_axes | D | **yes** | direct `lax.scan`, bypasses the seam; gated on W > chunk_size |
| sampling_derivatives.py | 317 | 309 `_func` | s_size, trs_push | B | public | `*_trs` reference form, no in-library caller |
| sampling_derivatives.py | 745 | 736 `_func` | trs_push | B | public | ditto |
| sampling_derivatives.py | 1354 | 1343 `_step` | s_size, trs, trs_xi | B | public | ditto |
| optimizers.py | 568 | 551 `cond` / 555 `body` | maxiter, tol2 / hvp, inner, xnp | D + E | only `use_jit=True` | see below |
| probing.py | 197, 277, 305, 485, 535, 829, 929 | — | — | A | ragged | consistency only |
| probing.py | 987, 1054 | 965, 1028 | sum_over_probes (+ n_probe) | C | ragged | consistency only |
| t3_orthogonalization.py | 132, 160 | 117, 149 | stack_shape, xnp / xnp | B | ragged | consistency only |
| tv_operations.py | 372, 386 | 366, 377 | xnp | B | ragged | consistency only |
| sampling_derivatives.py | 458, 540, 809, 987, 1330, 1486, 1992 | — | trs / — | A,B | ragged | consistency only |
| sampling_derivatives.py | 1524, 1606, 2003 | 1519, 1599, 1997 | einsum strings + n_probe | C | ragged | consistency only |

\* 1439 is B if the inner scan's `is_uniform` may legitimately be the constant `True`; C if it must
track the caller's raggedness. **Maintainer call** — see open questions.

## Shape of the work

- **~28 sites reach a real `lax.*` primitive.** The rest (~25) are ragged-path `xmap` bodies where
  the defect cannot occur.
- **Of the jax-reachable ones, all but two are `xscan`** and can be hoisted today. The two `xmap`
  ones (514, 868) are blocked on the seam fix.
- **7 of the jax-reachable sites are category A** — verbatim hoists, no signature change.
- **Category D is only 5 sites**, three of which are inner scans that a stable outer subsumes
  (1415 under 1439; 511 and 865 under 514/868 once the seam is fixed). The genuinely irreducible D
  work is **1640** and the optimizer `cond`.

## Beyond the `xscan`/`xmap`/`xwhile` grep

A whole-package sweep (AST scan for nested `def`s, greps for every `jax.*` attribute, plus a live
run patching `jax.lax.*`/`jit`/`vmap`/`grad`/`eval_shape`/`checkpoint`) found three things a
call-site grep misses:

- **`optimizers.py:395` — `jax.jit(fn)` on a freshly created nested `step`** (`mc_sgd` `:462`,
  `adam` `:509`). `jit` keys on `id(fn)`, so this is one recompile **per optimizer invocation** —
  not per iteration, since the compiled object is reused across all `max_iter` steps. That is
  exactly the rank-continuation pattern (one optimizer call per rank level). Measured: 3 calls →
  3 distinct `fn`. `step` closes over `step_problem` (unhashable), so a memoized factory does not
  apply; the options are to thread `step_problem` in, or to accept one compile per call.
- **`ut3_conversions.py:110`** — a CI-enforced doctest that reads
  `jax.jit(lambda a, b: ...)(tk, tt)  # RIGHT`. The `# RIGHT` is about closing host masks over
  rather than tracing them, which is correct; but the line jits an anonymous lambda inline, which
  is now exactly what `docs/contributor/scan_body_principles.md` warns against. Harmless where it
  runs; the risk is a reader copying it into a loop. Binding the lambda to a name once would keep
  the example consistent. Documentation fix, not a library defect.
- **Nothing else.** No `vmap`/`eval_shape`/`jacfwd`/`custom_jvp`/`checkpoint`/`pmap`/
  `tree_util.Partial` in library code at all; `jax.grad` and `jax.linear_transpose` appear only in
  comments and doctests. Every callable-taking helper (`corewise_map`, `apply_func_to_leaf_subtrees`,
  the `*_map_real_weights` pair, the display callback, the `GeometryOps` factory closures) was traced
  to its end and reaches plain Python, never a jax primitive. The `xmap`/`xscan` bindings in
  `t3_svd.py`, `t3_linalg.py`, `t3_operations.py` and `t3_orthogonalization.py:212/254` are dead
  locals — bound and never called.

## The house already solves this — in a different form

Worth knowing before choosing a remedy, because it is precedent rather than invention:

- **`backend/fitting.py:272-315` `SamplingKind`** — a frozen dataclass carrying an `identity` tuple
  with custom `__eq__`/`__hash__`. Its docstring documents this exact defect: its fields *are*
  lambdas, so the default `__eq__` made every rebuilt kind a fresh jit cache key — "one recompile
  per outer step" — and value-comparing `identity` fixes it. The `*_kind` factories each build ~10
  fresh lambdas per call and are safe **only** because of this.
- **`t3toolbox/fitting.py:762-782`** — pytree registrations deliberately keep closures out of
  `aux_data`, rebuilding the packed kind lazily in a `cached_property`, with the reason written down.
- **`shared_geometry.py:146/150`** — explicit `__eq__`/`__hash__` so a rebuilt wrapper is the same key.

So the library's established answer to "fresh object, stable value" is **value-based identity on a
frozen dataclass**, not a memoized factory. Note the negative result: the principle-4 memoized
factory (an `lru_cache` returning a *body*) is **not used anywhere in `t3toolbox/` today** — the
three existing `lru_cache` sites all return strings/tuples/floats. Introducing it would be a new
pattern, so it is worth checking each candidate against the `SamplingKind` approach first.

## Independent confirmation

A live trace across repeated eager calls found `_mu_jets_step` to be **the only loop body in the
package with a stable identity** (`n_distinct=1`); every other site rebuilt its body. The same trace
put `tt_orthogonalization.py:50` at the highest traffic in the library (n=11 across three `mc_sgd`
runs), independently of the call-count measurement above.

## One more open question

`_adj_sweep_scanned` (`sampling_derivatives.py:1420`, the enclosing function of sites 1439/1415) is
documented at `:1421` as an "EXPERIMENTAL memory-lean mirror of `_adj_sweep` (module-private)", yet
`compute_tau_tilde_jets`/`compute_sigma_tilde_jets` call it unconditionally and it is live in the
default `probe_derivatives` gradient. If the non-scanned `_adj_sweep` was meant to be the default,
the priority of those two sites changes.

## Open questions for the maintainer

1. **`optimizers.py:568`.** `cond` captures `tol2`, which changes every Newton iteration — a float
   that varies with the data, so it must go into the loop state, not a cache key. `body` captures
   `hvp` (a bound method of a fresh per-iteration `LocalModel`, i.e. array data behind a callable)
   and `inner`. An `lru_cache` keyed on those would never hit *and* would pin every `LocalModel`'s
   arrays for the process lifetime — worse than the status quo; record it as rejected. The real
   options are to make `LocalModel` a pytree and thread it through the while-state, or to jit
   `_cg_solve` whole. Only bites with `use_jit=True`, but when it does it is likely the single most
   expensive recompile in the library.
2. **Rebuilding `binomial_combine_tensor(order)` inside bodies** is the derivation for eight sites.
   Free on the jax path (folded as a constant), but an `O(order²)` Python loop per iteration on the
   ragged path. It is not `lru_cache`d today. Cache it, accept the cost, or pass `trs` as an operand?
3. **1439's `xscan` capture.** Dispatching it in-body as `is_uniform=True` is correct for the inner
   operands but would turn today's ragged+jax Python inner loop into a real `lax.scan` — a behaviour
   change on a path that never enters XLA today.
4. **Do the ragged-only sites get converted at all?** ~25 sites, zero defect risk, pure consistency.
