# P12 design note: the backend sampling functions on the engine

Implementer, 2026-11-20, read only, against `graded-messages` at `0710c3c6` (P11 committed; the plan tidied for the
second pause's decisions). Sources:
- the plan: P12, sections 2.4 (the function table, the sketches, the trio at a point), 7.2, 11.5;
- the engine as P5 to P11 built it;
- today's `probing`, `apply`, `entries`, `ut3_sampling`, `utv_sampling`.

## 1. Each backend function is one engine call

The 30 changed names and the 6 new ones (`t3_<kind>_corewise_tangent` and its twins) each become one call of
`sampling_variables`. The function chooses the call (the frame, the names wanted, the arguments) and, for the uniform
twins, the form of the output. Nothing else.

| Function | The engine call |
| --- | --- |
| `t3_probe(x, ww, order)` | `('z',)` on the corewise point `x` |
| `tv_probe(frame, variations, ww, order)` | `('Jp',)` with `variations` |
| `t3_probe_corewise_tangent(x, ww, core_directions, order)` | `('Jp',)` on the corewise point with `core_directions` (the raw route, P8) |
| `tv_probe_transpose(frame, ww, residual, order, sum_over_samples)` | the residual's adjoint: `g` with the residual held as `r`, no weight, so the seed is the residual turned (O-22) |
| `t3_probe_corewise_transpose(x, ww, residual, ...)` | the same on the corewise point |
| `t3_probe_corewise_hessian(x, ww, residual, core_directions, order, sum_over_samples)` | `('Kp',)` with `r` held and `core_directions` |
| apply, entries | alike (entries without `order`); `t3_apply` and `t3_entries` keep their positions |

**The one gap: a residual with a stack `K`.** The plan's transposes take a residual of the type of `z` *with* an
optional `K`, `F + W + [K] + C`, and return `[W] + [K] + C` blocks. Today's transposes do, and `T3Tangent`'s static
transposes rely on it. The engine's `r` row does not carry `K` (the residual's adjoint has `K = 0`), so a held `r`
with `K` fails the held-shape check. The engine *has* the computation: the adjoint seeded by `Jp`, the twin path,
carries `K` and gives `JtJp`. But `JtJp` needs `variations` (P8 gave it that need, so its message names `JtJp`), and
`K` is read only off a direction. Three ways:
- **(a), recommended.** A small engine addition in P12, its first user:
  - a second route row for `JtJp`, reachable from a held `Jp` with no direction (a needs fact for a held variable,
    e.g. `'held:Jp'`, added to `present` for each held name);
  - `_read_K` falling back to a held `Jp`'s `K` when no direction is given.

  The transposes then always call `want=('JtJp',)`, `held=HeldVariables(Jp=residual)`. That is one route for
  `K = ()` and for `K` alike, the plain sweeps run once, and a residual with `K` costs only the twin path.
- **(b)** Broadcast the frame over `K` (views) and treat `K + C` as the call's `C`. No engine change, but every plain
  sweep runs `K` times.
- **(c)** A residual without `K` only, the backend looping over `K`. This drops a capability the plan keeps.

The corewise Hessian's residual has no `K` (plan sketch), so it holds `r`; its `K` comes from `core_directions`.

**What the functions check themselves: nothing beyond the engine.**
- The razor: a backend user who calls the engine directly must get the same checks, and the engine already checks what
  it needs to run (structure, the stacks, the entries index's `d`, the held shapes).
- The jets checkers (`check_sample`, `check_data`, `check_weight`) stay the caller's (O-18), as the plan says; the
  frontend methods call them in P13.
- One thing is structural and belongs to the function: the frame's form. A `tv_*` function given a corewise point
  becomes a corewise call, where `variations` raises (P8); that message is the engine's and is clear.

**The sum flag.** `sum_over_samples` reaches the engine as itself (P10's `keep_W`) on the transposes and, new, on the
corewise Hessians (the blocks `[W] + K + C`). The tangent has no flag: no sum.

## 2. The uniform twins: mask in the engine, mirror the sample's form

- **The call.** A twin passes the uniform `.data` straight to the engine:
  - `ut3_*`: the point's `(B, G, shape, masks)`;
  - `utv_*`: the frame's six elements and `variations_data[:2]`. The variations' own masks are not used: the engine
    masks the variations with the frame's variation masks (P11), and the variations' masks are derived from the frame
    anyway. The docstring says so.
- **The input.** The engine packs a ragged sample (`len_J=len(order)`); the twins pack a ragged residual themselves,
  `pack_if_ragged(residual, N)`. A residual has full jet sizes in every mode, so it needs no `len_J`.
- **The mirror** (`docs/uniform_equivalence_contract.md`, "the vector I/O boundary"). A vector output (`z`, the
  tangent, and `Jp` through `tv_probe`) is unpacked to a `len=d` tuple when the sample came ragged (`is_packed(ww)` is
  false), by `unpack_vectors(z, shape)`, and returned packed when it came packed. The engine always returns packed.
- **What is not mirrored.** Apply and entries outputs are one array per sample (no mode axis), and the blocks are
  core-shaped supercores, always.
- **Masks on exit.** The engine re-masks, and the twins add nothing (O-3). `ut3_full_sum` stays as it is.

## 3. The call sites that move, and how the suite stays green

The plan's P12 lists them by function (its table of calls: tests, doctests, by reference) and by file. Three rules
cover the edits:
1. **The object first.** Every positional call of a function whose order changes is rewritten in N16's order. No
   caller passes a renamed argument by keyword (P12's search).
2. **The sum flag.** Every `sum_over_probes=` becomes `sum_over_samples=` (inverted). A call that relied on the old
   default of a tangent transpose (no sum) now passes `sum_over_samples=False` explicitly, so no value changes.
3. **The frontend keeps its own form until P13.** Its call sites move their arguments and pass
   `sum_over_samples=not sum_over_probes`, the methods keeping `sum_over_probes=False` as their own default.

**What else moves:**
- The uniform twins' bodies are rewritten (above).
- The two `point_forward` calls of probe move their arguments.
- The three ambient transposes take their new order, and their four callers follow.
- **The old kinds keep calling the old sweep, `*_from_sweep` and `*_derivatives` functions,** which do not change, so
  the fitting suite is untouched.
- The six corewise Hessians held "by reference" in `test_uniform_curvature.py`, `test_dispatch.py` and
  `test_curvature.py` are rewritten with their tables.
- **Order of work** (one commit, four reviewable parts):
  1. the ragged functions with their own tests and doctests;
  2. the uniform twins;
  3. the call sites in the library, the frontend first;
  4. the tests and the "by reference" tables.

  Each part ends with the affected files green, so a broken call shows up where it was edited.

**The bridge loses its order-0 comparisons' independence.** `tests/bridge/` compares the engine with the old kernels
through `probing.t3_probe`, `tv_probe_transpose`, `t3_*_corewise_hessian` and others at order 0. After P12 those
names *are* the engine, so the comparison would be the engine with itself. **I propose** that its order-0 rows call
the old code still in the tree instead: the old kinds' `transpose` and `corewise_hessian` (`ApplyKind`, `EntriesKind`,
and the order-0 derivative kinds), and the `*_from_sweep` functions, all of which stay until P20. The bridge keeps
its meaning until P20 deletes it.

## 4. The tests

**Existing tests, rewritten:** the calls above; dense ground truth stays dense ground truth (`dense_probe`,
`dense_apply`, `dense_entries`).

**Added, by function:**
- **every changed name, against the reference module by definition** at `order = ()`, `(2,)`, `(1, 1)` (entries at
  `()`), through its public signature, on the three problems of the engine's tests;
- **the adjoint identities:** `<tv_probe(v), c> = <v, tv_probe_transpose(c)>`, and the corewise pair
  `<t3_*_corewise_tangent(x_dot), c> = <x_dot, t3_*_corewise_transpose(c)>`, per element of a stack `K`;
- **the default sums**, and `sum_over_samples=False` summed over `W` equals it, on the transposes and the corewise
  Hessians;
- **the ambient transposes' default does not sum;** the entries functions have no `order`;
- **the corewise Hessian:**
  - against automatic differentiation in jax (O-22): the second derivative of `phi(x) = <c, t3_probe(x, ww, order)>`
    along `core_directions`, at `()`, `(2,)`, `(1, 1)`, skipped where jax is absent;
  - its symmetry `<y, H x> = <x, H y>`, numpy;
- **the corewise tangent**, as the plan has it:
  - **the numpy dense oracle first:** `dense_probe` of the sum of the `2d` one-core-replaced dense tensors, exact at
    every order and with mixed input jet sizes;
  - `jax.jvp` second;
  - the substitution `tv_probe` at `t3_corewise_frame(x)` while it stands;
  - a structural step count (each sweep once);
- **the uniform twins:** each equals its ragged function on real parts (`to_uniform -> op -> to_ragged`), with
  garbage in every masked slot (the residual's and `core_directions`' padding included), the exact output masks
  derived from the ragged ranks, and the mirror (ragged in ragged out, packed in packed out) at `(2,)` and `(1, 1)`;
- **a residual with `K`** (whichever of 1(a) to (c) is ruled) gives, per element, the transpose of that element;
- `tests/test_dispatch.py`: a row per function by the new signatures (jit, jax in and out), the six new ones included.

**Cost.** The by-definition tests reuse the engine tests' problems and the reference module; at `-n 2` about 20 to 40
s, mostly the reference. The jax AD test adds a few compilations.

## 5. Plan changes I'd propose

1. **The residual's `K` through the engine's `JtJp` twin** (section 1, option (a)): a route row from a held `Jp` and
   `K` read off it. It is a small engine change, first needed here. Otherwise (b), at the cost stated.
2. **The bridge's order-0 rows rewired to the old kinds and `*_from_sweep` functions** (section 3), so the bridge
   stays independent until P20.
3. **The `utv_*` twins ignore the variations' own masks** (the frame's are used); the docstring says so.
4. **A split, if the coordinator wants smaller reviews:**
   - P12a: the ragged functions, the call sites and their tests;
   - P12b: the uniform twins, the tangent trio member, the Hessian's flag, the AD tests.

   The plan calls P12 "large", and its four parts (section 3) are each reviewable.
