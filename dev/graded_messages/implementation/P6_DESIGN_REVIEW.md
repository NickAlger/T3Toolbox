# P6 design review: the implementer's note on Engine B

Reviewer, 2026-11-20, on `P6_DESIGN_NOTE.md` (read only; nothing run). Against the plan's section 2.3 (step 3), P6,
`SAMPLE_VARIABLES.md` sections 5.3 and 6, the decision notes of rows 6.4 and 6.5 in `TARGET_DECISIONS.md`, row 0.8, and
my P5 design review.

**Overall: sound, with one change I recommend strongly** (point 1: one route for `V_tilde`, no want-fact). It removes
point 2's restatement and point 5's open question at the same time. One forward-looking finding for P9: the start of a
sweep chosen "by grade" will not carry the apply kind's seed.

---

## 1. The short form of `eta_tilde`

**The mathematics is right, in any number of variables.** `V_tilde[i]`'s first term is the corner `c = order` of
`eta_tilde[i] * w[i]`, that is `Σ_k eta_tilde[c - k] ⊗ w[k]` over the slots `k` of the input clipped to the box. Those
`k` form the box `[0, a)` with `a = min(J_in_i, F)` per axis, so the slots of `eta_tilde` read are exactly the box
`[c - a + 1, c]` on every axis. That is `prod(a)` slots, and never below 0, since `a ≤ F = c + 1`.

The two other steps are also right:
- **Each slot as two readouts.** A slot `s` of the truncated product `mu_tilde ⊗ U ⊗ nu + mu ⊗ U ⊗ nu_tilde` is the
  coefficient at `s` of each product: a `jet_contract` readout with `order = s`, a lowered corner. The fold meets at
  `s`, and nothing beyond the box is read.
- **`V_tilde` from the slice.** With `S[j] = eta_tilde[c - a + 1 + j]`, `Σ_k eta_tilde[c - k] w[k] = Σ_k
  S[a - 1 - k] w[k]` is the coefficient `a - 1` of `S * w`, axis by axis. So it is the same `V` string, read at the
  corner `a - 1`. `jet_contract` reads `w`'s slots only up to `a - 1`, so an input larger than the box is clipped
  there too.

**Is it worth it?** Yes. It is what keeps P6 at today's cost, which Part 1 promises for the cases the library uses.
The library forms only those slots today. For a line (`a = 2`, `m = 1`), per site and per term of `eta_tilde`:

| Form | `jet_contract` calls | Slot-pair contractions |
| --- | --- | --- |
| full `eta_tilde`, `'loop'` | `n + 1` | `(n + 1)(n + 2)/2` |
| full `eta_tilde`, `'scan'` (the uniform layout) | `n + 1` | `(n + 1)²` |
| short form (top 2 slots) | 2 readouts | `(n + 1) + n = 2n + 1` |

Two terms per site in every row. Concretely:

| `n` | full, `'loop'` | full, `'scan'` | short |
| --- | --- | --- | --- |
| 3 | 10 | 16 | 7 |
| 6 | 28 | 49 | 13 |

`eta_tilde`'s two full × full products are a large part of the probe adjoint per site: next to `mu_tilde`'s and
`nu_tilde`'s one full × full term each, and `H_tilde`'s three-full readout. So the short form removes something like a
third of it at moderate orders, more on the uniform layout. At m = 2 with a plane the saving is smaller (4 slots × 2
readouts against two products of 6 partner multi-indices each), but still a saving.

**The mechanism: disagree with the want-fact. Make `V_tilde` read the short node always, and give `eta_tilde` its
own full row when wanted.**

The want-fact has three costs:
- **It changes values, so point 2 is needed.** `g` then depends, at rounding, on whether `eta_tilde` was also wanted.
  That contradicts the letter of the decided rule (point 2).
- **It is the first fact in `present` that is about `want` rather than the call's arguments.** The closure already
  takes `want`, so the fact duplicates it. It also makes one variable's route depend on whether *another* variable is
  wanted, which is exactly the kind of coupling the graph-as-data was meant to keep visible.
- **It needs point 5's open question at all.**

*Concrete alternative:*
- `_eta_tilde_top` is a private node, as proposed. `V_tilde` has **one** row, reading `_eta_tilde_top` (and `eta`,
  `w`, `w_tilde`).
- `eta_tilde` has its own row (the full two products), run only when it is wanted or held-and-read. Nothing but the
  user reads it in P6.

The results:
- **Values.** `V_tilde` and `g` are bit-identical whether or not `eta_tilde` is wanted.
- **Cost.** When both are wanted, the top slots are computed twice: `2 prod(a)` extra readouts beside a full
  `eta_tilde`. That is negligible, and only in the call that asked for `eta_tilde` itself.
- **The plan's test.** "The short form equals a slice of the full one" stays, as a numerical identity test between
  two nodes (tolerance, row 0.8).
- **Mechanism.** No want-fact, no second route, no change to `present`.

**Declared sizes of the private node:** `_eta_tilde_top[i]` has the jet sizes `min(J_in_i, F)`, per site like a
leaf. Put it in the shape table as such (its "Jet sizes" column `min(J_in_i, F)`), even though it is not a slot, so
P11's packed sizes follow the same rule as the leaves.

**Strings.** The lowered-corner readout keeps `W`, `'JWCa,Caib,JWCb->WCi'`. Its `K` forms for P7's `Jp`-seeded rows
are `'JWKCa,Caib,JWCb->WKCi'` (the tilde on `mu`) and `'JWCa,Caib,JWKCb->WKCi'` (on `nu`). With `K = 0` both reduce
to the first. Add the two `K`-written literals to `STRINGS`, as the note says, and nothing else.

## 2. "Wanting a variable never changes a value", to rounding

**Where it is stated:**
- the plan's section 2.3, step 3: "wanting it never changes a value (6.4)";
- `TARGET_DECISIONS.md`, the decision notes under row 6.4: "wanting a variable never changes any value; not wanting
  it lets the engine skip work. `eta_tilde` is the case in point ... the short one has to equal a slice of the full
  one, by test", and "a meaning that does not depend on the route".

**My reading.** The rule is about *values*. The comparison of the short form with a slice of the full one is
numerical, so by row 0.8 it is a tolerance test, and "equal ... by test" there means to tolerance. Saying so is a
clarification, not a departure.

But "never changes **any** value" also covers the *other* outputs of a call. Under the want-fact design, `g`'s bits
change with `want`. That goes beyond what row 0.8 licenses: 0.8 is about how equality is tested, not a permission
for an output to vary with an unrelated choice of the caller.

**With the alternative of point 1, nothing needs restating.** `g` never depends on `want`, and the one comparison
that differs at rounding (short against full) is a test between two nodes, tolerance by 0.8. I recommend that, and
no amendment to the plan's sentence.

## 3. `carries_K=False` on every tilde row, P7 adding `Jp`-seeded private rows: agree

This is the `K = 0` record ruling working as intended:
- **The steps' strings** carry `K` on the tilde operands: `'JWKCa,Caib,JWCi->JWKCb'`, `'JWCa,Caib,JWKCi->JWKCb'`, the
  `V` and `H` readouts with `K` on the tilde operand. All are among the 29.
- **The residual's rows** run them with `K = 0`.
- **P7's private rows** (seed from `Jp`) run the same steps with the call's `K`.

Two requests:
- **Name the `Jp`-seeded private rows by a fixed rule** (for example `_jp:` plus the public name). A test can then
  check that every public tilde row has its private twin, with the same step and `carries_K=True`.
- **The seed row `w_tilde`** reads `('r', 'weight')`. The `Jp`-seeded seed reads `('Jp', 'weight')`, through the same
  `jets.jet_to_dual_jet`. That is the plan's "`JtJp` seeds from `Jp` the same way".

## 4. The bridge and the generic twin: agree, with one condition on the bridge

- **The bridge scope is right.** `z` and `g` are required; `mu`, `nu` and `eta` against `compute_*_jets` by
  `old = t!·new` are cheap and worth having. The note uses P4c's conversion for `g` (`ω_t² r_t / t!`, forward).
- **The old tilde intermediates are optional, but not silently dropped.** The plan's P6 names "the old `compute_*`
  kernels". The bridge file should list the old tilde kernels it does *not* compare, each with one line of why ("no
  conversion confirmed: ..."). Then their absence is a recorded choice. If a script does confirm a conversion, I
  expect, from the P4c derivation, the form `old^(r) = r! · new[n - r]` under the forward-seed conversion. Include
  them then.
- **The generic twin.** Numpy and itertools only, its own truncated product and flip, an AST test that it imports
  nothing of `t3toolbox`: that is the independence the plan's 9.3 asks for. Two small points:
  - It compares best on the whole jet matrix, *including* the mixed-modes case at `(1, 2)`. That case is the
    per-axis check of the short form's slot box.
  - Keep its `V_tilde` computed from the full `eta_tilde`, so it checks the short form against the long way round.

## 5. A held, unwanted `eta_tilde` does not switch the route: agree (and moot under point 1)

Under point 1's alternative there is no route to switch: `V_tilde` reads the short node, whatever is held. A held
`eta_tilde` is used only where `eta_tilde` itself is read, which in P6 is nowhere but the return. That is consistent
with "a held variable never changes the route" from P5.

## Other findings

**A. For P9: the start of a sweep cannot be chosen by grade alone.**
- The note says `start_zeros` arrives, "chosen from the name's grade by one helper". For the probe kind the tilde
  sweeps start at zero.
- For apply and entries, the seed enters at the right end: `nu_tilde[d]` *is* the seed times the ones of bond `d`
  (`SAMPLE_VARIABLES.md` section 5.3 and the note to row 4.4), and `mu_tilde` and `xi_tilde` are zero.
- So in P9 the start of `nu_tilde` is a *value read from another node* (`w_tilde`). A helper keyed on the grade would
  have to special-case the kind.
- *Recommend:* a row may declare its start: `start='ones'`, `'zeros'`, or the name of a node it reads. Make the
  helper take the row, not the grade. P6 can build that with the two values it uses.

**B. The slots.**
- `w_tilde`, `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`, `V_tilde` and `H_tilde` are the computed variables
  another row reads (or, for `eta_tilde`, the user can want). The rule "private nodes are never slots" is right, and
  pinned by a test.
- With point 1's alternative, `eta_tilde` is read by no row in P6. The P5 rule "the slots are exactly the computed
  variables another row reads" then excludes it, until P8 (`eta_D_tilde` on the corewise call) or P9 reads it.
  Decide this explicitly. My suggestion: amend the rule to "every public computed variable" (so `eta_tilde` is a
  slot now). Row 6.4 says "every variable ... can be wanted and held", which already implies that.

**C. The dtype.** `weight` is now read by `w_tilde`, so it counts when `g` is wanted. That is right under O-11, and
P5's machinery handles it. `jets.jet_to_dual_jet` inside the engine receives arrays already cast, and its own dtype
handling then does nothing. Fine.

**D. Tests I will run on the built P6.**
- Every new name against the P4c reference with a nonuniform weight, over the jet matrix.
- `g` against the finite-difference derivative of `f` in the frame's variations, which is P4c's `g` route but written
  by me.
- The short node against the slice of the full `eta_tilde`, per axis on the mixed-modes case.
- `g` bit-identical whether or not `eta_tilde` is wanted (under point 1).
- The bridge's conversions checked by my own derivation.
- Mutants: the slot box off by one at the low end, the lowered corner off by one, the flip missing from the seed, a
  wrong start of `nu_tilde`.
