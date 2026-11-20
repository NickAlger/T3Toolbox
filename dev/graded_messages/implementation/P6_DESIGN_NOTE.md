# P6 (Engine B): design note

Implementer, 2026-11-20. This note covers only what is new on top of P5's engine. Nothing has been run. **[open]**
marks what I am unsure of.

## 1. The seed

```
Rule('w_tilde', ('whole', 'w_tilde_step'), ('r', 'weight'), _NONE, False)
```

- `w_tilde_step` is `jets.jet_to_dual_jet('probe', order, r, weight)`: the weight squared, then every jet axis
  flipped. It reuses that function and is not written again.
- `y` comes in through `r`, so a `g` with neither `y` nor a held `r` raises through the closure's chain.
- `w_tilde` is a slot (`xi_tilde` and `V_tilde` read it), a mode variable `F + W + C + (N_i,)`, stored turned.

## 2. The tilde rows and the readouts

| Variable | Driver, step | Reads (the carry not listed) |
| --- | --- | --- |
| `xi_tilde` | site, `xi_tilde_step` | `D`, `w_tilde` |
| `mu_tilde` | left (start zeros), `mu_tilde_step` | `L`, `mu`, `xi`, `xi_tilde` |
| `nu_tilde` | right (start zeros), `nu_tilde_step` | `R`, `nu`, `xi`, `xi_tilde` |
| `eta_tilde` | site, `eta_tilde_step` | `mu_tilde`, `mu`, `U`, `nu_tilde`, `nu` |
| `V_tilde` | site (two routes, section 3) | `eta_tilde`, `eta`, `w`, `w_tilde` |
| `H_tilde` | site, `H_tilde_step` | `mu_tilde`, `mu`, `xi_tilde`, `xi`, `nu_tilde`, `nu` |
| `g` | whole, `g_step` | `V_tilde`, `H_tilde` |

- **Strings and starts.** The terms are those of `SAMPLE_VARIABLES.md` section 5.3, each a literal of `STRINGS`.
  - The bonds are read through `_BOND_OFFSET`.
  - `start_zeros` arrives, chosen from the name's grade by one helper, which is tested.
- **`carries_K = False`** on every row now.
  - The steps' strings are written with `K` on the tilde operands where P7's `Jp`-seeded rows put it
    (`'JWKCa,Caib,JWCi->JWKCb'`, ...).
  - With the record's `K = 0` they are the plain strings, so P7 adds private rows, not steps.
- **Readouts.** A block term is one `jet_contract` call with no `J` in its output (the corner, `order` passed).
  - `V_tilde`: two readouts, `'JWCi,JWo->Cio'` and `'JWCi,JWCo->Cio'`.
  - `H_tilde`: three `'JWCa,JWCi,JWCb->Caib'`, the tilde on `mu`, `xi` or `nu`.
  - The step sums them (section 6). The blocks have no `J` and no `W`.
  - `g` pairs the two tuples. `sum_over_samples=False` is P10's.
- **Slots.** `HeldVariables` gains `w_tilde`, `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`, `V_tilde` and
  `H_tilde`. Private nodes (a leading underscore) are never slots, which a test pins.

## 3. The short form of `eta_tilde`, as graph rows

`V_tilde` reads only the top `a = min(J_in_i, F)` slots of `eta_tilde`, two rows for a line. The library today forms
only those.

- **A private node:** `_eta_tilde_top[i] = eta_tilde[i][c - a + 1 : c + 1]`, a slice on each jet axis, `c` the
  corner.
  - Each slot is two readouts (the two terms) at the lowered corner `c - a + 1 + j`, string
    `'JWCa,Caib,JWCb->WCi'` (`W` kept).
  - The readouts are stacked over the box of `a`, read off `w[i]`'s static shape.
  - A line costs 2 readouts, against `order + 1` contractions.
- **The routes of `V_tilde`:** the first reads `eta_tilde` and needs `{'want:eta_tilde'}`; the second reads
  `_eta_tilde_top`. Since `sum_k eta_tilde[c - k] w[k]` is the coefficient `a - 1` of `S * w`, `S` the slice, the
  short step is the same `V` string at the corner `a - 1`.
- **The choice** is a fact: the engine adds `'want:' + name` to `present` for each wanted name. `present` stays the
  one carrier of the static facts. "Read only by `V_tilde`" is the graph's structure, pinned by a test.
- **Wanting never changes a value, to rounding.** The tests: the private node equals the slice of the full
  `eta_tilde`, and `V_tilde` agrees across the two routes to `1e-13`.
- **[open] A held `eta_tilde` that is not wanted** does not switch the route (held never changes the route). I
  recommend keeping that rule: a user who wants the held one read can also want it.

## 4. The bridge (`tests/bridge/test_old_kernels_bridge.py`, deleted in P20)

The engine runs at `affine_jet(ww, pp)` with `(n,)`, `n` from 0 to 3, a nonuniform weight, and a frame from
`t3_orthogonal_representations`. It is compared with:
- `t3_probe_derivatives` for `z`: old = `t!` × new;
- `ProbeDerivativesKind.transpose` and `tv_probe_derivatives_transpose` for `g`, given the forward-order
  `c_t = weight_t^2 r_t / t!`, not turned (P4c's conversion);
- `compute_mu_jets`, `compute_nu_jets` and `compute_eta_jets`, old = `t!` × new.

**[open]** The old tilde kernels (`compute_deta_tilde_jets` and the rest; section 10's names) are not turned and are
derivative-weighted. I will derive their conversion by script and include them only where it is confirmed. `z` and
`g` are the required pair.

## 5. The generic twin (`tests/_generic_twin.py`)

The prototype's `gmul`, at the grades `''` and `'z'`, brought to the decided forms:
- Taylor input jets, and the seed weighted then flipped;
- its truncated product over multi-indices, for any jet shape;
- `D`, `U`, `L`, `R` as `PointedCore`.

**Scope:** the probe kind; `z`, `w_tilde`, the tilde variables, `V_tilde`, `H_tilde` and `g`.

**Independence:** only `numpy` and `itertools`, one `np.einsum` per leg rule, and its own product and flip. An AST
test pins that it imports nothing of `t3toolbox`. It is compared with the engine on the jet matrix, a second witness
beside the reference.

## 6. Changes I would make to the plan

1. The short form as the private node plus a second `V_tilde` route chosen by a `'want:'` fact (section 3). Two new
   readout literals join `STRINGS`.
2. "Wanting never changes a value" should say "to rounding", for `V_tilde` across the routes.
3. The bridge's tilde intermediates are optional, kept where a script confirms the conversion.
