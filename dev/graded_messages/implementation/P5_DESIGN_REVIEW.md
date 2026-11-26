# P5 design review: the implementer's sketch of Engine A

Reviewer, 2026-11-20, on `P5_DESIGN_SKETCH.md` (read only; nothing run). Against the plan's sections 2.2 to 2.4, 4
and the slices P5 to P11; `SAMPLE_VARIABLES.md` sections 1 to 9; the decided rows 3.1 to 3.7, 4.1, 4.4, 5.1, 5.2,
5.4, 5.6 and 6.1 to 6.5; and the two contributor notes.

**Overall: a sound design, and I would let it be built.** The step functions are pure, one site's math. The strings
are the final literals with `C` written. The statics reach the steps through memoized factories. The graph is data,
and the closure is a pure function. P10 and P11 attach at two named seams. The open points are mostly decided the
right way.

Below: my answer on each of the seven points; one finding against the **plan** (P10 needs strings beyond the 29),
which guard (ii) of point 6 would trip over; and four smaller requests.

---

## The seven points

### 1. Routes in the graph. Agree with several records per name chosen by `needs`, with two conditions. Corewise: agree with renaming, as a derived graph, not as a name map before the closure.

**Records chosen by `needs`.** A route chosen by which optional arguments are present is a static fact of the call.
Several records per name, with `needs`, say that in the graph itself. The docstring table can list both rows of `z`.
And "a wanted variable whose argument is missing raises" falls out of the closure, with no extra code. That is
simpler than my pre-read's "graph keyed by (kind, call type)" for routes that depend on arguments.

The two conditions:
- **The precedence must be declared and tested, not implicit in tuple order.** "The first record whose `needs` are
  present" works today because the `Vc` row comes first, and row 4.1 says the value is read off the frame when the
  centers are given. Write that rule in the graph's docstring: among applicable records, the earlier is preferred,
  and the table is ordered from the most specific to the least. Add a structural test: for every name with two or
  more records, a call that satisfies both takes the first. Otherwise reordering two tuple entries silently changes
  which cores `z` reads.
- **A held variable does not change the route.** With `Vc` given and `eta_D` held, the closure takes the `Vc` route
  and computes `eta` from scratch. That is consistent ("held entries are leaves, the route is chosen by the
  arguments") and keeps the closure simple, so state it. Do not add route selection by what is held. That is a
  cost optimization, Part 2's.

**The corewise call (8.7).** A name map applied to `want` before the closure is not enough. The corewise *routes*
differ, not only the names:
- `g` on a corewise call is `V_term(eta_D_tilde, w) + V_term(eta_D, w_tilde)` and the three `H_term`s with `nu_L`
  (`SAMPLE_VARIABLES.md` section 6), where on a manifold call it reads `eta_tilde`, `eta`, `nu`, `nu_tilde`;
- the frame forms `xi_U_dot` and `mu_R_dot` do not exist on a corewise call at all.

So renaming the wanted names onto `nu` and `eta` would run the manifold rows on `(B, G, G, G)`. Mathematically that
is the same arrays, but it does not reproduce "each sweep once": `nu` (through the `R` slot) and `nu_L` (through the
`L` slot) are two sweeps of the same `G`, unless the rows themselves are renamed.

*Concrete alternative:* make the call type a pseudo-argument of `present_args` (for example `'_corewise'` or
`'_manifold'`), so routes by call type use the same `needs` mechanism as routes by argument. Then derive the
corewise graph mechanically from the manifold graph by one declared table:
- each basis-named variable mapped onto its basis-free name (`nu_L -> nu`, `eta_D -> eta`, `nu_L_dot -> nu_dot`,
  ...), in **both the names and the `reads`** of every record;
- the frame-form rows dropped;
- the corewise `g` row selected by `needs={'_corewise'}`.

That gives one mechanism and one source of truth, and a test checks the derived graph against `SAMPLE_VARIABLES.md`
section 9's name table. It also answers my pre-read's point: the graph is effectively keyed by call type, but the
second graph is derived, not hand-written. P5 need not build it. P5 should only make `present_args` the one carrier
of static call facts, so P8 adds a row and a table.

### 2. The `K = 0` record. Agree. It is sound and strict, and better than a flag selecting two literals.

- **Strict:**
  - Every call still passes a whole `GroupLengths`, so every group of the string is pinned.
  - An operand with a `K` axis given to a node whose record says `K = 0` has one rank too many, and `jet_contract`
    raises.
  - An operand without `K` given to a node whose record says `K = len(K)` has too few, and raises.

  The property the plan's "one record per computation" protected (a stray or a missing axis cannot be read as a
  longer stack) holds node by node.
- **Simpler:** a flag selecting two literals would double the tilde and readout strings and add a branch to every
  step that runs both ways.

**Two conditions:**
- **Which record a node gets is declared in its graph row**, a static field (`carries_K: bool`), never inferred from
  the operands' shapes. Inferring it would bring back the silent reading that `GroupLengths` exists to prevent.
- **The plan's sentence is amended** to "one record per node of a computation: the same `J`, `W`, `C`, and `K`
  either the direction's or zero, as the node's row declares".

The memo keys of the bodies include the record, so a `K = 0` body and a `K > 0` body are two bodies, as they must
be.

### 3. The drivers and the bonds. Agree, with two specifics to write into the design.

- **The start value at an end is sized by the store the record reads, not by a fixed store.** For `nu` the bond `d`
  start is `R[d-1].shape[-1]`; for `nu_L` it is `L[d-1].shape[-1]`. With arbitrary tails (row 4.4) the two can
  differ. The sketch says "sized by the end bond of the cores passed", which is right if "the cores passed" means
  the record's own store, read from its `reads`. Pin it with a test on a frame whose `L` and `R` end bonds differ.
- **Prepend/append is the one layout-aware line of the drivers.** On ragged the `xscan` result is a tuple and the
  start is joined as a tuple; on uniform it is a stacked `(d,) + ...` array and the start is concatenated on axis 0
  (`xnp.concatenate`, or `jax.lax.scan`'s stacked outputs plus one `concatenate`).
  - Slicing `[:d]` and `[1:]` is indeed one syntax on both.
  - The `'right'` sweep by `[::-1]` on the per-site sequences works on supercores as well (a reversed view on numpy,
    a `rev` on jax). It needs no `tt_reverse`, since `nu_step` has its own string.
  - On uniform the start ones sized by the padded end bond meet masked-zero cores at the padded slots, so they are
    inert (mask once on entry).

  Keep that line in one small helper (`_with_end(bonds, start, at_start)`) so P11 changes one place.

The rule "a `mu`-family variable is read at bond `i`, a `nu`-family one at bond `i + 1`" is fixed by the letter. That
is the definition in `SAMPLE_VARIABLES.md`: the left of bond `i`, the right of bond `i + 1`. Write it once as a
table in the module (letter to offset), not as string tests scattered through the drivers, and test it.

### 4. The memo keys. Agree: `(step_name, lens, order, form)`, with the leaf's order computed inside the step, is enough.

- The per-site difference on ragged is only in the leaves' jet sizes. The leaf step computes its output order from
  its operand's static shape at each call, so no per-site static is closed over. The sweep steps pass `order`, the
  full box, and `jet_contract` reads every partner's sizes from its shape.
- On ragged the "scan" is a Python loop, so a different shape at a site is just another call. On uniform the packed
  leaves have one shape.
- What must not happen: a factory that captures a per-site leaf order (for example to pre-slice) keyed without it.
  The sketch's form avoids that. Keep the rule in the module's comment: "a body closes over the record, the box and
  the form, nothing per site".

### 5. `w` not a slot of `HeldVariables`. Agree; amend the plan's sentence.

- `w` is an argument. Holding it would mean "use this sample instead of the sample", which is the `sample` argument
  itself.
- The plan's rule should read: "the slots are exactly the **computed** variables (the names some record makes) that
  some other record reads". The structural test is then written against that, so it does not fail on `w` and does
  not let a computed variable slip out.
- `w_tilde` (P6) is computed from `r` and the weight, so it *is* a slot, as the plan wants.
- Wanting `'w'` returns the sample as given (on uniform, the packed array). That matches the reference module (P4c:
  `w` as given).

### 6. `HISTORICAL`, option (b). Agree, with guard (ii) loosened, and a gap in the plan that it exposes (finding A).

Excluding the first argument of a `jet_contract` call from `scan_vocabulary` is right:
- those strings never reach `contract` without every length;
- `TestJetContractDerivedStrings` already sweeps every library `jet_contract` literal against the loop oracle and
  checks every derived string;
- the P1 handover's option (b) said so.

Guard (i) (every library literal with `J` is such a first argument) is right.

Guard (ii) (every library `jet_contract` literal is one of the 29 `STRINGS`) is right in intent, keeping the case
file authoritative, but its wording will break in P10 (finding A). Write it as: "every library `jet_contract`
literal is in `_jet_contract_cases.STRINGS`", and let P10 add to `STRINGS`.

### 7. Anything else

- **The `step` field as a string resolved by `getattr`.** Acceptable for the test hook, and the graph holding no
  function objects keeps it plain data. But the code is less statically readable. Add a structural test that every
  `step_name` in every graph is an attribute of `sampling_steps`, so a typo fails at test time, not at the first
  call that reaches that row.
- **The dtype in P5 (6, [open]).** Agree: P5 computes the result dtype by the rule over the arrays the closure
  reads, because a start value cannot be built in a default dtype (row 5.6). P10 then adds the stacks and the
  dtype matrix of tests, not the rule.
- **`Vc` (8.5, [open]).** Agree with the reading: "with `Vc` read off the frame" in P5's entry means the value is
  read off the frame through the centers (row 4.1). The engine does not derive `Vc`; the tests compute it by
  `t3_orthogonal_representations`.
- **The short form of `eta_tilde` (8.6, P6).** Fine to settle in P6. Make the short form a separate record with
  `needs` or a declared reader condition, not a `short=True` flag decided in the engine body: the same reasoning as
  point 1, the graph says what runs.
- **One namespace for `reads`** (graph names, stores, arguments) works as long as no name is both. Today they are
  disjoint (`D` against `D_dot`, `V` against `V_tilde`, `Vc`). A structural test that the three sets are disjoint
  costs one line and protects P7 and P8.
- **No hidden state.** The sketch has none: memoized factories on hashables, a pure closure, `HeldVariables` frozen
  with `with_values` returning a new record. Keep `_closure`'s cache, if any, keyed on hashables only (`want`, the
  frozenset of held names, the frozenset of present arguments), never on a `HeldVariables` instance.
- **What the engine checks** is exactly section 2.3's step 1 list. The held-shape check is the one place where
  shapes from the frame and the sample enter. It should use the same shape table the docstring prints and the
  structural test reads (one function `(name, index, setting) -> shape`), so the three cannot drift.

---

## Finding A (against the plan; for Claude): `sum_over_samples=False` needs readout strings beyond the 29

The blocks `g`, `JtJp` and `Kp` with `sum_over_samples=False` keep `W` ("W joins the stack, W+K+C"; row 3.7). Their
readout strings then have `W` in the output: the `V` family `'JWCi,JWo->WCio'` and its `K` variants, the `H` family
`'JWCa,JWCi,JWCb->WCaib'` and its variants.
- None of these is among the 29 of section 2.4. The library today has the same split, as two bodies
  (`_tucker_variation_jets_step_summed` and `..._unsummed`, `backend/sampling_derivatives.py`, lines 1464 to 1470).
- So P10 (which builds `sum_over_samples=False`) must add literals, about nine (5 `V` and 4 `H`, with `W` in the
  output), and choose between them by the static flag. `jet_contract` already supports a readout that keeps `W`
  (the fold along `J` leaves `W` alone).
- The plan's P10 text and section 2.4's "The strings in Part 1" say neither. Guard (ii) of point 6 as written
  would fail at the first such literal.
- *Resolve:*
  - one sentence in the plan's P10: "the readout strings with `W` kept are added to `STRINGS`, chosen by the static
    `sum_over_samples`";
  - guard (ii) worded as in point 6.

  Not a P5 change.

## Four smaller requests for the build of P5

1. The closure's precedence rule written and tested (point 1).
2. The `carries_K` (or equivalent) field present in the row type from P5 on, even if every P5 row is `False`, so P7
   adds values, not a field.
3. The end-start sized from the record's own store, tested on a frame with differing end bonds (point 3).
4. Structural tests: every `step_name` resolves, and the three namespaces of `reads` are disjoint (point 7).

**For my review of the built P5** (from my pre-read): every name against the P4c reference over the jet matrix with
a nonuniform weight; `z`, `r`, `f` and `r_block_sumsq` against my own dense oracle; a held variable deliberately
modified, to show it is used; step counts derived from the graph by hand; mutants (a wrong output order, a leaf not
clipped, `R` against `L` in `nu` and `nu_L`, the route swapped, the held-shape check removed); jit and float32; and
no dependence on the safety mode.
