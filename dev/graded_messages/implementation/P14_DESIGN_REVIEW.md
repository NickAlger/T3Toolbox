# P14 design review: the implementer's note on the oracle and the batch

Reviewer, 2026-11-21, on `P14_DESIGN_NOTE.md`, read only. I read it against the plan's 2.2 (the sketches and the text
after them), P8's ruling on the corewise direction (plan, around line 1680), O-7, O-8, O-18, O-21, O-29, O-30, and
`optimizers._objective_resolution`.

**Overall:** the design is right: one engine call, the hold list as the plan fixes it, the scale on exactly O-21's
outputs, and an additive slice. My recommendations on the four proposals are below. One gap: the note does not say
what `query` does on a corewise frame when it is given **both** `variations` and `core_directions`.

## 1. `keep=` against `computed_names(...)`: recommend `keep=`

**The razor's user.** A backend user writing their own oracle wants exactly what the least-squares oracle wants: "run
this, and also hand me what you computed that I can hold". `keep=` gives that in the one call that already has every
fact (the call type, `present`, `held:Jp`, the closure).

**Why not `computed_names(...)`.**
- It makes the user pass the same arguments twice and trust that the two readings agree. The engine's facts are just
  what is easy to get wrong (`held:Jp` was added in P12a).
- Being public, it also exposes the closure as an API: a name list whose content changes whenever the graph changes.

**Conditions:**
1. `keep` names must be `HeldVariables` slots; an unknown name raises, as an unknown `want` does.
2. The returned record holds:
   - the `held` passed in;
   - plus every `keep` name the closure **computed**;
   - and nothing computed only for it (8.3).

   A `keep` name the call did not compute is simply absent. Say so in the docstring.
3. The extra return element appears only when `keep` is given; this follows `np.unique(return_counts=True)`. Under
   `jit` the set of filled slots is static (it is the closure's), so the record is a valid pytree output. The test
   should assert both.

## 2. A public `structure(frame)`: needed, but give it another name

**Whether to read the structure** is settled: the plan has the oracle pass the direction as `core_directions` on a
corewise frame, "telling the call type from the frame's structure, not from the geometry".

**Whether it should be public.** The reading is not trivial: **a count of 4 means a ragged manifold frame or a
uniform corewise point**, decided only by `is_ndarray(frame[0])`. That is the razor's "easy to get wrong", and every
custom oracle needs it. So a public reader, used by the engine and the oracle alike, is right, and better than a
private helper the user re-derives.

**The name.** `structure` already means something else in the library: `TuckerTensorTrain.structure` and
`UniformTuckerTensorTrain.structure` return shapes and ranks. A function `structure(frame)` returning
`('ragged', 'corewise')` would be a second meaning of one word. Propose a name for what it returns, for example
`frame_form(frame) -> (layout, call)` or `call_type(frame)`. This is the maintainer's choice, under
`naming_conventions.md`'s "names exist to help the user".

**The gap: both directions given on a corewise frame.** The plan names the case: the exact Hessian on a corewise
geometry, whose lift is the identity, passes `variations = p` for `JtJp` and `core_directions = theta_hat = p` for
`Kp`. The engine takes one form there. The oracle must therefore:
- pass `core_directions` when it is given (it is the lifted `p`, the same object);
- otherwise rename `variations`;
- and when both are given and are not the same object (`is`), raise structurally. Two different directions on a
  call that has one form are a caller's error, and silently dropping one would be a wrong answer.

Test all three cases.

## 3. `take` sets `minibatch_scale = n / len(indices)` (times any scale already there): agree

- **Unbiasedness.** For a uniform draw without replacement of `b` of `n`, `E[(n/b) sum_{i in S} f_i] = sum_i f_i`.
  So the scale makes a drawn batch an estimator of the full one by construction, and the draw (P16) only chooses
  indices.
- **Nested takes.** Multiplying the scales is the right chain: `n` is the batch being taken from.
- **Conditions:**
  - Say in the docstring that `take` *is a draw* in this sense, and that `drawn_indices` are relative to the batch
    taken from.
  - A caller wanting a plain subset with no scale (a fixed validation slice, say) uses
    `dc.replace(sub, minibatch_scale=None)`. Show this once, since a validation `f` scaled by `n / b` would otherwise
    surprise.
- The note's subset-enumeration test (the mean over all subsets equals the full `f` and `g`) is exactly the right
  test.

## 4. `pack_sample` / `pack_data` unchanged until P16; `pack_batch` on `pack_if_ragged`: agree

P14 is additive, and the old uniform kinds call the two with the old kind names. `pack_batch` packs the jet axes with
`len_J=len(order)` (zero coefficients, O-45) and probe's `y` on `N`, which is right. Record in P16's list that the two
get the plan's `(kind, order, ...)` forms there.

## 5. The other checks

**`f_resolution`.**
- *Today:* `eps (|w y| sqrt(2 misfit) + |f|)`, where `f = misfit + rho`. With `rho >= 0`, `|f| = misfit + rho`.
- *The oracle:* computes `eps (|w y| sqrt(2 f_o) + f_o)` with `f_o` its misfit only.
- So the optimizer must add `eps * |rho|` itself in P16 to reproduce today's floor on a regularized fit. Put that in
  P16's list.
- *Order of operations with a scale.* Compute `f_resolution` from the **unscaled** batch values (`|w y_b|`,
  `sqrt(2 f_b)`, `f_b`), then multiply by `minibatch_scale` once. That is O-21's inner product, under which both
  norms scale by `sqrt(s)`. Scaling `f` first and then taking `sqrt(2 f)` would scale by `s^(3/2)`. Test it with a
  drawn batch.

**The minibatch scale's reach.** `f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp` and `f_resolution` are scaled; `r`, `Jp` and the
block sums stay plain. That is O-21 exactly. The regularizer's minibatch rule is P16's, and P17's
`point_value_and_gradient` must use the same rule (N47).

**`held` growing across calls at one frame.** It is correct, and is valid for one frame **and one batch**: `r` depends
on `y`, and the sweeps on the sample. The record carries no batch identity (it cannot under `jit`), so the guard is
the contract plus P16's tests. Put in P16's list a mutant that carries `held` across draws in MC-SGD, which some test
must kill.

## 6. The tests

Sufficient, with these additions:
- the three cases of the corewise direction (§2);
- `keep`'s conditions, including the record's static structure under `jit` (§1);
- `f_resolution` on a drawn batch (§5);
- `take`'s docstring example of an unscaled subset (§3).

## Summary

| # | Proposal | Verdict |
| --- | --- | --- |
| 1 | `keep=` or `computed_names` | `keep=`, with the three conditions; the maintainer decides the surface |
| 2 | `structure(frame)` | Public, yes (the count-4 ambiguity); a name not already taken; plus the both-directions rule |
| 3 | `take` sets the scale | Agree, documented as a draw, with the unscaled-subset idiom shown |
| 4 | Packing unchanged until P16 | Agree |
| — | `f_resolution`, the scale, `held` | Unscaled first then scaled; P16 adds `eps \|rho\|`; P16 tests `held` per draw |
