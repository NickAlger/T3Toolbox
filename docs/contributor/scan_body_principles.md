# Scan and map bodies: the design principles

> Why an `xscan` / `xmap` / `xwhile` body should be a module-level function that closes over
> nothing, what goes wrong when it isn't, and the escape hatches for a body that genuinely needs a
> static. Written after the eager-jax recompilation problem found 2026-08-20 (measured below); the
> principles are general, and meant to be applied within reason rather than as hard rules.

## The problem

`get_backend` dispatches `xscan` / `xmap` to `jax.lax.scan` / `jax.lax.map` on the uniform+jax path,
and `xwhile` to `jax.lax.while_loop`. Called **eagerly** — with jax arrays but outside a top-level
`jit` — these primitives key their trace/compile cache on the **identity of the body function** (jax
keeps a weak reference to it as the cache key). A body defined inside its caller is a new object on
every call, so the cache can never hit:

- the body is re-traced and re-compiled on every call, with byte-identical shape signatures;
- XLA's CPU backend compiles each one through LLVM, and the resulting executables' memory mappings
  are retained.

Measured on the fitting path (`probe_derivatives` under Newton-CG, uniform+jax, fixed ranks), before
any of this was addressed:

| | per Newton iteration |
|---|---|
| XLA compilations | 19 |
| new process memory mappings | ~877 |
| wall time | 3.56 s, against 0.09 s for the same solve on ragged-numpy |

Both halves matter. Linux caps mappings per process at `vm.max_map_count` (65530 by default), so a
long run **aborts** with `LLVM ERROR: Unable to allocate section memory!` after roughly seventy
accumulated iterations — and the timing says the jax path was ~40x slower than numpy, essentially
all of it recompilation rather than kernels. A jit path that exists to be faster was slower, for
this reason alone.

This is documented jax behaviour rather than something to wait out.
[Discussion #16216](https://github.com/jax-ml/jax/discussions/16216) is exactly this scenario, and
the [jit-compilation page](https://docs.jax.dev/en/latest/jit-compilation.html) carries a
benchmarked section — "Avoid calling `jax.jit()` on temporary functions defined inside loops or
other Python scopes" — measuring 583 ms against 5.24 ms for the same work. Two remedies are offered
there: define the body outside its caller, or wrap the caller in `jit`. Worth knowing:
`functools.partial` and `lambda` are named as *causes*, not fixes — each produces a fresh object
with a fresh identity.

## The principles

### 1. Prefer a module-level body

A body passed to `xscan` / `xmap` / `xwhile` should be a module-level function, defined once, so its
identity is stable across calls. This is the whole fix; everything below is about making it
possible.

It costs nothing on the numpy and ragged paths (where `xscan` is a Python loop and identity is
irrelevant), and it helps the jitted path too — one fewer thing to re-trace.

### 2. Let the body derive what it needs from its own arguments

The reason bodies get written inline is that they need something from the enclosing scope. Usually
they don't have to:

- **The backend** comes from the standard dispatch rule applied to the body's own operands (below),
  so `xnp` never needs to be captured.
- **Statics are often recoverable from shapes.** A carry stacked over derivative orders knows its
  own order: `order = mu_jet.shape[0] - 1`. Under jax a tracer's `.shape` is static at trace time,
  so this is free.
- **Small constants are better rebuilt than captured.** `xnp.arange(order + 1)` inside the body
  becomes a folded constant in the jaxpr, created once per trace on the right device.

Applied together these often leave nothing to close over at all.

### 3. Dispatch inside the body, on all of its arguments

Use the ordinary house pattern — `tree_contains_jax` over the operands, then `get_backend` — just
from inside the function rather than around it:

```python
xnp, _, _ = get_backend(True, tree_contains_jax((mu_jet, data)))   # only xnp; it ignores the flag
```

This is safe inside a loop, and the reason is worth stating because it is a property of the
library's dispatch rule rather than a coincidence. **Jax is absorbing**: any jax leaf promotes the
whole operation to jax, and the promotion is one-way. A scan body's output carry inherits the
promoted type, so the decision reaches a fixed point after at most one iteration and cannot flip
mid-scan. Concretely, a numpy carry with jax `xs` picks jax on the first iteration and stays there.

Two consequences:

- **Inspect every argument, not a chosen one.** Dispatching on the carry alone loses the fixed-point
  property (it would pick numpy on iteration 0 and jax on iteration 1), and dispatching on a subset
  risks choosing numpy while a jax leaf is present — which silently pulls data off device, since
  `np.concatenate([jax_array, numpy_array])` returns a host `ndarray` with no error.
- **The body can never disagree with its caller.** A body's leaves are a subset of the caller's,
  plus the carry, which the caller built with its own `xnp`. So if the caller chose numpy, the body
  sees only numpy. This matters because `numpy_scan` finishes with `np.stack`, which would pull a
  jax-valued body output to the host.

**One caveat, for `map` bodies specifically.** The "cannot disagree with its caller" argument above
leans on the carry, which the caller built with its own `xnp`. An `xmap` body has **no carry** -- it
sees only its own element -- so its dispatch is per-element rather than per-operation. For a
*mixed* numpy/jax sequence the caller's old whole-tree rule sent every element through `jax.numpy`,
while a hoisted body sends the numpy elements through `numpy`. Nothing in the library produces such
a sequence and the frontends do not construct one, so this is unreachable in practice; all-numpy and
all-jax inputs are bit-identical either way (measured). It is recorded because it makes the rule
"jax is absorbing **per operation**" for scans and "**per element**" for maps -- a slightly weaker
statement than the scan case, and the reason to prefer a `scan` formulation when there is a choice.

The rule also survives every jax transform: `DynamicJaxprTracer`, `BatchTracer`, `LinearizeTracer`
and `JVPTracer` are all instances of `jnp.ndarray`, so `tree_contains_jax` sees them as jax and the
body never silently falls back to numpy under `jit` / `vmap` / `grad` / `jacfwd`.

### 4. If a static truly can't be derived, memoize the specialization

Some bodies will genuinely need a static that no argument carries. Then build the body in a
module-level factory memoized on that static, so each distinct value gets one stable object:

```python
@functools.lru_cache(maxsize=None)
def _some_step(order):
    def _func(carry, data):
        ...
    return _func
```

This is the same trick jax uses internally (`weakref_lru_cache` keyed on a callable). It comes with
one rule that is easy to state and easy to check:

> **A cached factory may close over hashable scalars and modules only — never arrays.**

An array built at factory time freezes ambient state that is not part of the cache key. Its dtype is
fixed by whatever `jax_enable_x64` was set to on the first call, and it is committed to whatever
device was default then — a constant built during a CPU call would be reused on GPU work. With
`maxsize=None` it is pinned for the process lifetime. Building the array inside the body instead
costs nothing (it is traced once and folded into the jaxpr) and removes the whole class of problem.

A second, quieter hazard: a factory parameter whose value varies with the data is not *incorrect* —
`lru_cache` will key on it happily — but it grows the cache and the compile count without bound, one
specialization per distinct value. Keys should be structural.

### 5. A value that CHANGES between calls belongs in the state, not in a closure

This one is a correctness rule, not a performance preference, and it is the reason principle 4's
"never arrays" is not the whole story.

A body that is stable *and* reads a value that changed since its last call gets the **cached jaxpr
with the old value, silently** -- no error, no warning. Measured on a `lax.while_loop` whose
module-level `cond` compares against a Python float:

| threshold at call time | 10 | 20 | 3 |
|---|---|---|---|
| closed over (stable body) | 10 | **10** | **10** |
| carried in the loop state | 10 | 20 | 3 |

So the two properties have to travel together: making a body stable is only safe once everything it
reads that can change per call has moved into the state. Freshness is what protects a body that
still closes over such a value -- which means at those sites the rebuild-every-call cost is
accidentally load-bearing, and removing it without moving the value is a wrong-answer bug rather
than a no-op.

The practical test: for each captured value, ask *can this differ between two calls with the same
operand shapes?* Structural statics (a rank, a mode count, an order) cannot, and are safe to
specialize on. Tolerances, step sizes, iteration budgets and anything derived from the current
iterate can, and must be carried.

### 6. Move values into the scan operands only when they are genuine runtime data

Restructuring a body's carry or `xs` to carry a captured value is the heaviest option and should be
the last one. Reserve it for values that really are per-call data rather than structure. Note that
jax's own advice for *statics* runs the other way — passing a static through a scan makes it
non-static, and closure is
[the recommended way](https://github.com/jax-ml/jax/discussions/16667) to keep it static — so this
is not a general-purpose substitute for principles 2 and 4.

## Why not simply jit the callers instead

Wrapping the caller in `jit` is the other remedy jax offers, and it does work: inside a `jit` the
body is traced once as part of the enclosing computation, whatever its identity. It is worth doing
on its own merits, and is tracked separately as the full-jit question.

It is not a substitute for these principles, though. It only helps callers that happen to be jitted;
anyone calling `apply`, `probe`, `probe_derivatives` or `t3svd` eagerly on jax arrays — a fully
supported configuration — still pays the full recompilation cost. And a `jit` boundary has its own
requirements: it must be a stable module-level object, with static shapes, and no Python control
flow branching on traced values. The two remedies compose; the body-level one is cheaper and helps
everywhere.

## Exemplar

`_mu_jets_step` and its caller `compute_mu_jets` in `backend/sampling_derivatives.py`. The step
closes over nothing: it dispatches on its own operands, recovers `order` from the carry's leading
axis, and rebuilds the `t` multipliers inline.

```python
def _mu_jets_step(
        mu_jet: NDArray,                       # carry: (order+1,)+W+C+(rLi,); axis 0 is the order axis
        data:   typ.Tuple[NDArray, NDArray],   # (G, xi_jet) for one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (mu_jet,))
    xnp, _, _ = get_backend(True, tree_contains_jax((mu_jet, data)))   # only xnp; it ignores the flag
    order = mu_jet.shape[0] - 1                            # the carry is stacked over derivative orders
    s_size = min(2, order + 1)                             # affine input jet: orders {0, 1}
    ...
```

In a downstream consumer of the library, converting this one site alone removed 6 of the 19
recompilations per Newton iteration and ~170 of the ~877 mappings, with bit-identical output on both
backends and on the order-0 branch.

## Applying this within reason

- Many existing bodies already close over nothing and need only to be moved out — a mechanical
  change with no signature consequences.
- The `order`-from-the-carry recovery in the exemplar is a fact about *that* carry's layout, not a
  general rule. Each body needs its own look.
- A body used only on the ragged or numpy paths is not affected by any of this. Hoisting it is still
  reasonable for consistency, but it is not urgent, and it is not worth contorting a signature for.
- Where a body is genuinely clearer inline and is never reached with jax arrays, leaving it inline is
  a defensible call — write down why.
