# Parameters are fields, not closures

*Decision record for the optimization-layer restructuring (2026-08-21). Why the geometry, the sampling
kind, and the local model went from records-of-lambdas to frozen dataclasses, and the backend rule that
replaced "no classes in the backend".*

## The problem

Every jax cache — `jit`, `scan`, `while_loop` — is keyed on **identity**. Every axis object in the
optimization layer was a bag of Python closures built by a factory that partially applied its
parameters. So a rebuilt-but-identical object was always a *new* cache key, and the parameters that
would prove otherwise were sealed in closure cells where nothing could read them.

That is not a small inefficiency. A fitting loop rebuilds its local model at every outer step by
construction, so "rebuilt-but-identical" is the normal case, not an edge case.

Measured before the change:

| object | rebuilt twice, equal? |
|---|---|
| `MANIFOLD_OPS` (a module singleton) | yes |
| `uniform_geometry_ops('manifold', same data)` | **no** |
| `shared_geometry_ops(MANIFOLD_OPS, same groups)` | **no** |
| `probe_derivatives_kind(2)` | yes — but only via a hand-maintained `identity` tuple |

## The tell: the same workaround, invented three times

- **`SamplingKind.identity`** — a tuple restating the parameters as values, with custom `__eq__`/`__hash__`
  comparing it. Correct only while someone remembered to extend it, and `None` (so, object identity)
  for any kind a user built.
- **`UniformGaussNewtonModel`** — four shadow fields (`kind_name`, `x0_masks`, `order`, `weight`) whose
  only job was to reconstruct the packed kind, plus a `cached_property` that did the reconstructing,
  because the kind itself could not go in the jit aux.
- **A memoized factory** in the chunked `𝒥ᵀ` assembly, to give one scan body a stable identity.

Three independent workarounds for one cause is the signal to fix the cause.

## It was hiding a silent miscompile

`dataclasses.replace` copies the `identity` tuple unchanged, so a derived kind claimed to *be* its
parent. On the ragged apply model:

```
derived = dc.replace(APPLY, forward=lambda ...: 0.5 * APPLY.forward(...))
derived == APPLY                           ->  True         (but different math)
jit(model_with_derived).gn_quadratic(p)    ->  115.302888   <- APPLY's compiled program
eager(model_with_derived).gn_quadratic(p)  ->   28.825722   <- the truth
```

jit returned the parent's answer, silently. The same mechanism made the ragged `APPLY` and the uniform
apply kind compare equal and hash equal — the uniform kinds were built by `dc.replace` off the ragged
singletons — though that pair never miscompiled, because the surrounding pytree structure and avals
differ.

Both are now unrepresentable: `dc.replace` on a kind raises (methods are not fields), and a variant is
a **subclass**, which the type check in `__eq__` rejects.

## The decision

**Parameters are dataclass fields; behaviour is methods.** Value identity then comes from the fields
themselves (`common.ValueHashedFields`), so there is nothing to keep in sync, and a user-defined
geometry or kind gets correct cache behaviour without knowing the rule exists.

This is also the more faithful encoding of the mathematics. A uniform manifold *at a given rank* is a
different manifold from one at another rank, so the rank belongs in the object's defining data. The
closure encoding erased exactly the data that defines the object.

Note that a record-of-functions **is** a class — the dictionary-passing encoding of an interface. The
choice was never class-versus-no-class; it was which encoding, and only one of them keeps the
parameters readable.

## The backend rule this replaces

"No classes in the backend" was a proxy for something truer, and the geometry and optimization layers
had already forced it to be relaxed. The value it was protecting is that **the math stays reachable**:
a mathematically sophisticated user should be able to find a function, call it on plain data, and even
copy-paste it, without learning any architecture. The sharpened form:

> **Backend functions implement the math on plain data. Backend classes bind parameters and name
> roles — every line of math in a method is also reachable as a standalone function.**

Checkable, and strictly better for that user than what preceded it. The closure encoding was actively
eroding the value: the `v_X` direct construction had a name on the ragged path
(`_manifold_point_tangent`) and was an unreachable inner closure on the uniform path — the same math,
its accessibility decided by an accident of how each factory happened to be written. The factory shape
gives you nowhere to put a name, so math accretes inside lambdas.

Applying the rule surfaced four duplications, and each convention now has one home:

| convention | before | after |
|---|---|---|
| the variation-mask gauge shift `(up, down, left[:-1], right[1:])` | 5 copies | `ufv_variation_masks` |
| the `(U,G,G,G)` corewise frame | ~10 copies + one private function this refactor briefly deleted | `t3_corewise_frame` / `ut3_corewise_frame` |
| frame → variation shapes | frontend property only | `fv_variation_shapes` / `ufv_variation_shapes` |
| the sharing-partition normalization | open-coded | `sharing.canonical_groups` |

## Where to put a jit boundary

Making a `scan`/`while_loop` body closure-free is the body-level remedy
([`scan_body_principles.md`](scan_body_principles.md)). It is not always the cheapest one. `_cg_solve`
closes over the Hessian-apply, the inner product, the CG tolerance and the iteration cap, and two of
those change every Newton iteration — so the body-level rule applies, but the cache it would hit is
still keyed on identity.

The resolution generalizes:

> **When the values a body reads are awkward to defunctionalize, put the jit boundary where a
> value-based cache already exists, rather than hand-defunctionalizing to make an identity-based cache
> behave like a value-based one.**

`_cg_solve` became a plain function of `(local_model, rhs, tol, maxiter)`, jitted whole. The cache key
is jax's own — the model's pytree treedef, whose aux is value-hashed — and `tol` / `maxiter` are traced
arguments, so they cannot go stale. That only works because the objects crossing the boundary hash by
value, which is the whole point of the decision above.

One wrinkle worth knowing: raw backend data tuples mix arrays with static structure (a uniform frame is
`(supercores…, shape, masks)`), and a bare tuple is a pytree whose every element is a leaf — so
flattening one naively traces the rank masks, which `require_concrete_masks` rejects outright.
`common.partition_static` splits them, keeping host-numpy bool masks and Python ints in the aux, exactly
as the frontend `UT3Frame` has always kept its masks. The rule it encodes is the uniform layer's own
documented contract (masks are always host numpy and never traced), so an integer numpy *array* — the
`entries` index sample — correctly stays traced.

## What it bought

| | before | after |
|---|---|---|
| compiles per Newton iteration (uniform `probe_derivatives`) | 1 | **0** |
| a user-defined kind, rebuilt 5× as jit aux | 5 compiles | **1** (the cold one) |
| `mc_sgd` / `adam` per-step kernel | one compile per optimizer *call* | once per shape signature, process-wide |
| CG tolerance staleness | freshness was load-bearing for correctness | unrepresentable |

Plus the deletions: `SamplingKind.identity` and its `__eq__`/`__hash__`, `_kind_key`,
`UniformGaussNewtonModel` entirely (four shadow fields, a lazy rebuild, a pytree registration), one of
two same-frame guards, and a duplicated geometry mapper.

## Rejected

- **Adding an `identity` field to `GeometryOps` too.** The minimal fix for the compile-count symptom,
  and it would have propagated the bolt-on to a second class rather than removing the cause. It also
  leaves user-built objects silently wrong, which is the failure mode that produced the miscompile.
- **Structured backend data (design B).** Give backend frames and tangents their own types carrying
  shape and masks, and the geometry could be stateless — one hierarchy for backend and frontend, no
  mapping layer. Rejected because the backend's data stops being plain tuples of arrays, which is the
  backend razor's core promise. Keeping bare data and putting the rank on the geometry is also what
  [`uniform_backend_jit_recipe.md`](../uniform_backend_jit_recipe.md) already prescribed.
- **Sharing as a wrapper class in the backend.** It is a `groups` field on the geometry instead, so
  shared and unshared are one code path and value identity falls out. The public frontend
  `SharedGeometry` stays as the user-facing constructor.
- **Aliasing `UniformGaussNewtonModel = GaussNewtonModel`** for compatibility. That would make
  `isinstance(ragged_model, UniformGaussNewtonModel)` true, which is worse than an `ImportError`.

## A verification note

Two measurement mistakes cost real time here; both are the same shape, and worth remembering.

**Breadth of cases is not coverage when the cases share a degeneracy.** A shape derivation was checked
across 21 structures including stack shapes and passed every one. It was wrong. All 21 happened to have
`nD == nU`; the first shared geometry where a group's rank differed from the down rank broke it
immediately.

**Compare invariants, not representations.** An optimizer comparison reported a 1.75 absolute
difference and looked like a real regression. It was comparing raw cores, which carry a gauge freedom
(`U → UQ`, `G → Qᵀ G` leaves the tensor unchanged). Comparing the dense tensor instead: agreement to
1e-12 relative, with Newton trajectories matching to ten digits.
