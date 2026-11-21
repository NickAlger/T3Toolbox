# P5 design sketch (Engine A), by the implementer, 2026-11-19

Written before any code of P5, from the plan's sections 2.2 to 2.4, 4 and P5 to P11, `SAMPLE_VARIABLES.md` 1 to 7,
`docs/contributor/scan_body_principles.md`, `uniform_polymorphism.md`, the prototype, `jet_steps_check.py`,
`_jet_contract_cases.py`, `common.py`'s `xscan`/`xmap`, `ORCHESTRATION.md` section 6 and `P1_HANDOVER.md` section 6.
Saved by Claude from the implementer's report, for the reviewer's critique and as the record of the design. Points
marked **[open]** are the implementer's.

## 1. Modules

**`backend/sampling_steps.py`** (private, `__all__ = []`): the step functions and the start values, nothing else.
P5: `xi_step`, `mu_step`, `nu_step`, `eta_step`, `z_step` (the `z_term`), and `_start_ones`, `_start_zeros`. A step
is pure math on ONE site's arrays: no loop over sites, no masks, no knowledge of layout or graph. Each term is one
`contractions.jet_contract('<literal>', ...)` call, through the module attribute (never `from ... import
jet_contract`), so a test can patch `contractions.jet_contract` and record every call with its `order`.

**`backend/sampling_variables.py`**: the engine `sampling_variables`, `HeldVariables`, the declared graphs
(`_PROBE_GRAPH` in P5), the shape table, the closure, the drivers (sweep, map, whole) with their memoized bodies,
layout detection and the held-shape check. Imports `contractions`, `common`, `jets` (for the seed in P6) and
`sampling_steps`; never `safety`.

## 2. The declared graph

A tuple of records `Rule(name, step, reads, needs)`, a frozen NamedTuple:

- **`name`**: the variable it makes. **Several records may share a name**: alternatives, tried in order (`needs`).
- **`step`**: `(driver, step_name)`. `driver` is one of `'site'` (a map over the sites `i = 0..d-1`), `'left'` (a
  sweep producing the bonds `0..d`, its start value at bond 0), `'right'` (the mirror, its start at bond `d`) and
  `'whole'` (not per site: `r`, `f`, `r_block_sumsq`, the seed in P6). `step_name` is a **string** naming a function
  of `sampling_steps`, resolved by `getattr` at run time, so the step-counting test hook sees a patched attribute and
  the graph holds no function objects.
- **`reads`**: the step's positional array arguments, in order, each a name of one namespace: graph variables
  (`xi`, `mu`, ...), frame stores (`D`, `U`, `L`, `R`), call arguments (`w`, `y`, `weight`, `Vc`; later `V`, `H`,
  `D_dot`, `L_dot`). Which bond a site op reads is fixed by the letter: at site `i` a `mu`-family variable is read at
  bond `i` (sliced `[:d]`) and a `nu`-family one at bond `i+1` (`[1:]`), in `eta`, `z`, the blocks and the tilde steps
  alike. A sweep's own carry is its first argument and is not listed. Slicing is the same syntax on a tuple and on a
  stacked array.
- **`needs`**: a frozenset of the call arguments the record requires to apply: `z` through `Vc` needs `{'Vc'}`, `z`
  through `D` nothing, `r` needs `{'y'}`; P7's dots `{'variations'}`, P8's `{'core_directions'}`.

P5's probe graph:

| Variable | Driver, step | Reads | Needs |
| --- | --- | --- | --- |
| `xi` | site, `xi_step` | `D`, `w` | |
| `mu` | left, `mu_step` | `L`, `xi` | |
| `nu` | right, `nu_step` | `R`, `xi` | |
| `nu_L` | right, `nu_step` | `L`, `xi` | |
| `eta` | site, `eta_step` | `mu`, `U`, `nu` | |
| `eta_D` | site, `eta_step` | `mu`, `L`, `nu_L` | |
| `z` (first record) | site, `z_step` | `eta`, `Vc` | `Vc` |
| `z` (second record) | site, `z_step` | `eta_D`, `D` | |
| `r` | whole | `z`, `y` | `y` |
| `f` | whole | `r`, `weight` | |
| `r_block_sumsq` | whole | `r` | |

**The closure** is a pure function `_closure(graph, want, held_names, present_args)` returning the records to run in
dependency order, and each node's readers (for P6's short form). A depth-first search from `want`: a name in
`held_names` is a leaf and stops the search; otherwise the first record of that name whose `needs` are all in
`present_args` is taken, its variable `reads` visited, then the record appended (postorder). No applicable record
raises, naming the missing argument and the chain ("f needs y, through r"). The least-squares oracle (P14) can call
`_closure` to learn which names of its hold list a call computes anyway. `HeldVariables`' slots are the union over
every kind's graph of the variable names in `reads`, and a test checks the hand-written class against it.

## 3. Step signatures, layouts and stacks

```python
def mu_step(
        mu_JWCa:  NDArray,               # the carry, bond i: J+W+[K]+C+(ra,)
        G_Caib:   NDArray,               # C+(ra, n, rb)
        xi_JWCi:  NDArray,               # J+W+[K]+C+(n,)
        *,
        lens:     GroupLengths,
        order:    typ.Tuple[int, ...],
        form:     str,                   # 'loop' | 'scan'
) -> NDArray:                            # bond i+1: J+W+[K]+C+(rb,)
    return contractions.jet_contract('JWCa,Caib,JWCi->JWCb', mu_JWCa, G_Caib, xi_JWCi,
                                     lens=lens, order=order, over_orders=form)
```

- **The strings are the final ones** (the 29 of `_jet_contract_cases.STRINGS`, `C` written). P5 passes
  `GroupLengths(J=m, W=1, K=0, C=0)`; P10 changes the record, not a string or a step.
- **Both layouts with no change**: the uniform per-site slice has the ragged per-site layout (the core index `d`
  leads), so one step serves both.
- **The drivers**: `xscan(body, start, per_site)` for `'left'`; the same on the per-site sequences reversed with
  `[::-1]` for `'right'`, then reversed back; the start prepended or appended to give `d + 1` bonds; `xmap(body,
  per_site)` for `'site'`.
- **The bodies** come from module-level factories memoized on hashables only: `@lru_cache _sweep_body(step_name,
  lens, order, form)` and `_site_body(...)`, each closing over those scalars and calling `getattr(sampling_steps,
  step_name)` (scan-body rule 4, so P11's uniform + jax path compiles once).
- **`K` in P7 and P10, [open]: a second record with `K = 0`.** A step's terms carry `K` on exactly the operands that
  carry it when the step is used with a direction, and the step is used both ways (the residual's tilde application
  has no `K`; the `Jp`-seeded one has `K` on its tilde operands). The engine passes such a record a `GroupLengths`
  with `K = 0` (a group of zero axes, which `contract` supports: `SHAPE_MATRIX` tests `K = ()`); the `K`-carrying
  strings then reduce to the plain ones, and one tilde step serves `g` and `JtJp` with no string variants. The choice
  is static per record: `K` for the dot, tilde-dot, `Jp` and `Jp`-seeded nodes, `0` otherwise. This puts two records
  in one computation where the plan's 2.4 says "one record per computation". The alternative: a static flag on the
  step selecting between two literals. Recommended: the `K = 0` record.

## 4. The jet shape and the declared jet sizes

The engine reads `order` once (`contractions._read_order`), applies the entries rule (reusing `jets._kind_and_order`),
sets `m = len(order)`, `W = ww[0].shape[m:-1]`, builds `lens`, and sets `form` (`'loop'` ragged, `'scan'` uniform).
`(lens, order, form)` reach every step as keyword statics through the body factories. Each step implements its row of
the table:

- **Full outputs** (`mu`, `nu`, `eta`, `z`; later every tilde and dot) pass `order`.
- **Leaf steps** (`xi`; later `xi_dot`, `xi_U_dot`) pass `tuple(min(a, n + 1) - 1 for a, n in zip(w.shape[:m],
  order))`, read off the operand's static shape (scan principle 2). The ride slices the input to that box, so slots
  beyond `F` are dropped there, and per-site differences in `J_in_i` cost nothing on ragged.
- **Start values** are built with the full `F`, slot `(0, ..., 0)` one (plain grade) or all zero (other grades),
  with `C`, sized by the end bond of the cores passed: a scan carry needs one shape, and `mu[0]` is returned with
  its declared sizes.

## 5. `held`, and what the engine checks

`held` (a `HeldVariables` or `None`) comes in; its filled names enter the closure as leaves and their values are used
as given. The engine returns only the tuple aligned with `want`; the oracle builds its own `held` from the names it
asked for (plan 8.3).

The checks (section 2.3 step 1, nothing more): an unknown kind; an unknown name in `want`; an entries call with jets;
`order` not a tuple (`_read_order`); a frame structure that matches neither layout nor count; no applicable record
for a name the closure needs; every filled held slot has the table's shape (a sequence of `d` (sites) or `d + 1`
(bonds) arrays, each exactly `J + W + [K] + C + (size,)` with its declared jet sizes; `size`, `C` and the `J_in_i`
from the frame and the sample; `z` and `r` as `F + W + C + (N_i,)`). Ranks are checked in `jet_contract`. No
`safety`, nothing numerical.

**`HeldVariables`**: a frozen dataclass, one `Optional` field per slot with its shape comment; reads like a
dictionary (`__getitem__`, `__contains__`, `keys` over the filled slots, `__iter__`, `__len__`); an unknown name
raises a `ValueError` at construction and at lookup; `with_values(**kw)` is `dataclasses.replace` after the name check;
registered as a pytree inside `try: import jax` (leaves: the filled values; aux: the tuple of filled names). P5's
slots: `xi`, `mu`, `nu`, `nu_L`, `eta`, `eta_D`, `z`, `r`. **[open]** Whether `w` is a slot: it is read by `xi` but
it is an input, not computed. Proposed: `w` is an argument name, not a graph node; wanting `'w'` returns the sample
as given, and `w` has no slot.

## 6. Where P10 and P11 attach

The engine body, in this order:

```
_read_inputs -> _closure -> _prepare_inputs (dtype; masking) -> run the records -> _finish_outputs (re-mask) -> the tuple
```

- **P10, the dtype rule:** in `_prepare_inputs`, over exactly the arrays the closure's records read, cast once; start
  values built in it. **[open]** P5 must build start values in some dtype; proposed: P5 already computes the result
  dtype by the rule over the arrays read (a few lines), and P10 brings the stacks and the tests.
- **P11, masking:** `_prepare_inputs` on the uniform layout masks each input with the frame's own masks;
  `_finish_outputs` re-masks the core-shaped outputs. On ragged both are the identity.

## 7. The `HISTORICAL` table: option (b)

Literals that are the first positional argument of a `jet_contract` call are kept out of `scan_vocabulary` (an AST
rule skipping the constant of such a call): `jet_contract` hands `contract` its strings only with every length
(pinned by `TestJetContractDerivedStrings`), so their identifiability is moot, and `TestJetContractDerivedStrings`
already sweeps every library `jet_contract` literal against the loop oracle. Two guards: every library string literal
containing `J` must be such a first argument; every library `jet_contract` literal must be one of the 29 `STRINGS`
(the case file stays authoritative and `_library_shapes` right). The sharding sweep then skips them; P11's engine
sharding class covers them. Call sites write the string inline.

## 8. Changes proposed to the plan

1. Option (b) above.
2. The `K = 0` record (section 3), for P7 and P10; it amends "one record per computation".
3. The `step` field `(driver, step_name)` with a string name, for the test hook.
4. Several records per name, `needs` choosing between them (`z` through `Vc` or `D`): "a wanted variable whose
   argument is missing raises" falls out of the closure.
5. **[open]** The P5 entry "`z` (through `D`, and with `Vc` read off the frame)" read as: the second form when `Vc` is
   given, with the tests computing `Vc` by `t3_orthogonal_representations`; the engine does not derive `Vc` from the
   frame.
6. **[open], for P6**: the short form of `eta_tilde`: the closure returns each node's readers, and a step receives a
   static `short=True` when the node is not wanted and is read only by `V_tilde`. To settle in P6.
7. **[open], for P8**: on a corewise call the store tuple is `(B, G, G, G)`; to keep each sweep once, the engine maps
   the basis names onto the basis-free ones before the closure (`nu_L` to `nu`, `eta_D` to `eta`) and refuses the
   basis names in `want`: an alias table per call type, not a second graph.
