# P7 design review: the implementer's note on Engine C (the dot grade, frame form)

Reviewer, 2026-11-20, on `P7_DESIGN_NOTE.md` (read only; nothing run). Against `SAMPLE_VARIABLES.md` sections 5.2, 6
and 8, the plan's sections 2.3 (the paragraph on `nu_L_dot` and `eta_D_dot`, O-12) and P7 to P11, and the rulings so
far. Weighed by simplicity and performance, with values equal to rounding taken as the same.

**Overall: sound.** The recurrences are right, and the twins are a clean mechanical derivation.

Two changes I recommend:
- **Point 4:** a cleaner way for the two names with both forms, which uses the existing route mechanism and moves no
  public name between slices.
- **Point 3:** a one-line structural check, because a size-1 `K` on `H` is not a confusing error but a silent one.

---

## 1. The dot rows: agree. The recurrences and the bases are right.

Checked term by term against `SAMPLE_VARIABLES.md` section 5.2:

| Variable | Basis, size | Terms of the recurrence | The bases, leg by leg |
| --- | --- | --- | --- |
| `xi_U_dot[i]` | `U`, `nU_i` | `xi_term(V[i], w[i])`, the leaf's jet shape | `V[i]` is `(nU_i, N_i)` |
| `mu_R_dot[i+1]` | `R`, `rR_(i+1)` | the carry through `R[i]`; `mu[i]` through `H[i]` over `xi[i]`; `mu[i]` through `U[i]` over `xi_U_dot[i]` | `mu[i]` is in `L` (`rL_i`); `H[i]` and `U[i]` both take an `L` bond to an `R` bond; `R[i]` takes `R` to `R` |
| `nu_L_dot[i]` | `L`, `rL_i` | the mirror: the carry through `L[i]`; `H[i]` over `xi[i]` with `nu[i+1]`; `U[i]` over `xi_U_dot[i]` with `nu[i+1]` | `nu[i+1]` is in `R`, and `H` and `U` take it to `L` |
| `eta_D_dot[i]` | `D`, `nD_i` | `mu_R_dot[i]` through `R[i]` with `nu[i+1]`; `mu[i]` through `H[i]` with `nu[i+1]`; `mu[i]` through `L[i]` with `nu_L_dot[i+1]` | the open Tucker leg is `R[i]`'s, `H[i]`'s or `L[i]`'s, all in the `D` basis |

Further:
- **The starts.** A sweep that sums over the place of the direction is a carried sum. The site's terms put the
  direction at site `i`, and the carry holds the earlier sites, so the starts are zero. They carry `K`, sized by the
  swept store (`R` for `mu_R_dot`, `L` for `nu_L_dot`).
- **The bond offsets by letter** hold: `mu_R_dot` at bond `i`, `nu_L_dot` at `i + 1`.
- **The strings** put `K` exactly on the operand that carries the direction (the carry, `H`, `V` or `xi_U_dot`). All
  are among the 29.
- **Cost.** `Jp` reads no `nu_L` and no `eta_D`: the frame form needs `mu`, `nu`, `eta` and the dots only, so it does
  not sweep `L` from the right. That is the cheapest closure, and it is what the graph gives.
- **`Jp`** is `z_term(eta_D_dot, D) + z_term(eta, V)`, section 6. **`Jp_norm_sq`** is `<Jp, ω² Jp>` with no half,
  keeping `K + C` (step 7). Both right.

## 2. The private twins: agree. The derivation is sound for the short node and the seed.

The rule is: the same step, `carries_K=True`, the same start, every *tilde* read renamed by the rule, and the seed's
`r` replaced by `Jp`. Going through the path:
- **The seed.** `_Jp.w_tilde` is `jet_to_dual_jet('probe', order, Jp, weight)`. The weight squared, reshaped to
  `F + (1,)...`, broadcasts over `W + K + C`, and the flip is on the jet axes only. That is "`JtJp` seeds from `Jp`
  the same way" (step 6).
- **The tilde rows.** `_Jp.xi_tilde`, `_Jp.mu_tilde`, `_Jp.nu_tilde`:
  - their tilde operands carry `K` in the strings already (`'JWKCa,..'`, `'..,JWKCi->..'`);
  - their plain reads (`mu`, `xi`, `nu`) have none, so the record's `K` pins exactly the right operands;
  - the starts are zero with `K`.
- **The short node.** `_Jp.eta_tilde_top` reads `_Jp.mu_tilde`, `mu`, `U`, `_Jp.nu_tilde`, `nu`, `w`. Its box `a`
  comes from `w`, which is unchanged, and its readouts have `K` on the tilde operand (`'..->WKCi'`). So the
  derivation carries over with no special case.
- **The blocks.** `_Jp.V_tilde` and `_Jp.H_tilde` have `K` in front, and `JtJp = g_step` over them.
- **The narrowing is right.** There is no twin of the full `eta_tilde` (nothing reads it, and a private name can
  never be wanted), and none of `g` (`JtJp` is its twin). Word the twin test as "every row on the path that `JtJp`
  reads". That is what "every public tilde row" meant.

Two details to make sure of:
- **The prefix in every lookup.** `_SHAPES` (for `_at_sites`, the bond test) and the start sizing must find a
  twin's entries through the name after the prefix, by one helper (`_base(name)`), not by `split('_')[0]` on the
  prefixed name. `'_Jp.mu_tilde'.split('_')` gives `''` first.
- **Twins are never slots.** A user cannot hold the `Jp`-seeded adjoint, and does not need to: row 3.3 reaches it by
  a call with `Jp` held as `r`.

## 3. `K` read from `V[0]` with no check of its own: disagree. Add one static structural check.

- *A wrong count of `K` axes on `H` or another `V[i]`* raises in `jet_contract`'s rank check. The message names the
  derived string and the operand's `ndim`, which is confusing but an error.
- *A size-1 axis where `V[0]` has a larger one* is **silent**. Group axes follow einsum's extent rules, and a size-1
  axis broadcasts (the `contract` docstring says so). So `H` with `K = (1,)` beside `V` with `K = (2,)` gives each
  direction `k` the pair `(V[k], H[0])`: a wrong `Jp` with no error.

That is a wrong shape accepted, which the house rule makes a hard error always ("structural problems always error").
It is also cheap, static and jit-safe.

There is no backend checker of a `(frame, variations)` structure that a raw-`.data` user could call instead. The
frontend's `T3Variations` validates, but the backend engine is the one entry for both tiers.

*Recommend:* at the reading of `K`, check that every `V[i]` and `H[i]` has the prefix `K + C` of `V[0]`. That is a
loop over `d` comparing tuples, with one message ("variations: H[2] has the direction stack (1,), V[0] has (2,)").
It is "what the engine needs to run" in the sense that matters: the directions are one stack.

*The same hole for `W` and `C`* (one more reason to settle it once), for P10:
- `W` is read from `ww[0]` and `C` from `D[0]`;
- an input or a core with a size-1 axis where another has more would broadcast silently.

P10 should decide one entry check of every group across the operands that carry it: static, cheap, the same message
form. Or decide explicitly that the backend trusts the structure and say where the checker is. I lean to the check.

## 4. The two names with both forms (O-12): disagree with the move in P8. Use the routes from P7 on.

The plan's sentence (section 2.3): "`nu_L_dot` and `eta_D_dot` are one name each for both dot forms ... In a call that
has both `variations` and `core_directions` ... the engine forms the frame form for `Jp`/`JtJp` internally, and the
public name is the raw form."

So the public name has two **routes**: the raw form when `core_directions` is given, else the frame form. And `Jp`
must always read the frame form. A frame-form public row that `Jp` reads in P7, which P8 renames to private and
re-points `Jp` at, works. But it moves names and reads between slices. It also makes the P7 tests and slots of
`nu_L_dot` and `eta_D_dot` change meaning in P8.

There is a P8 hazard too. If `Jp` read the *public* name and a user held `nu_L_dot` meaning the raw form, `Jp` would
silently take the raw form. The P8 rename exists to prevent exactly that, so it is better not to create the hazard at
all.

*Concrete alternative, from P7 on:*
- **Private frame nodes, read by `Jp`.** `_frame.nu_L_dot` and `_frame.eta_D_dot` are the frame-form rows. `Jp` and
  the `eta_D_dot` frame recurrence read them. They never change.
- **Public names as routes.** The public `nu_L_dot` and `eta_D_dot` are each a route row that reads the private node
  through a trivial identity step (`'whole'` or `'site'`, returning its input), `needs={'variations'}`. Wanting them
  costs nothing extra: the closure computes the private node once.
- **P8 adds a route in front:** the raw row, `needs={'core_directions'}`. The precedence rule makes it the public
  meaning when both arguments are given, which is O-12, and `Jp` is unaffected because it reads the private node.

The results:
- **Nothing moves between slices.** P7's tests of the public names (by definition, with `variations` only, where the
  two forms are one name) stay valid in P8. P8 only adds rows and tests.
- **The held-variable hazard cannot arise.** A held `nu_L_dot` is a leaf for the public name, never for `Jp`'s input.
- **Slots.** `nu_L_dot` and `eta_D_dot` are public slots from P7 on, with one shape in both forms.

The cost is two identity rows. A trivial step is ugly but honest, and it can be one shared step `_identity_step`.
Alternatively the row type gains an "alias" driver; I prefer the trivial step, since it adds no concept.

Same for `xi_U_dot` and `mu_R_dot`? No. They exist only in the frame form, and the raw form has its own names
(`xi_dot`, `mu_dot`). Only the two shared names need routes.

## 5. The `JtJp` identity per `K` element, and the bridge: agree

- **`JtJp[k] = g` with `Jp[..., k, ...]` held as `r`** (the `K` axis sliced off, after `W`), with the same weight and
  no `y`. A held `r` has no `K`, so per element is the only form, and it follows from 3.3: the tilde sweeps are
  linear in the seed, and the `Jp`-seeded seed is `jet_to_dual_jet(Jp)`. Compare to rounding.
- **The bridge.** `ProbeDerivativesKind.forward` is `t!` times the new `Jp`, as for `z`. `.transpose` given
  `c_t = ω_t² Jp_t / t!`, forward, is the new `JtJp`, by the same derivation I gave in P6 for `g` (the old map is
  `t!` times the new, transposed). Run at `K = ()` and `(2,)`, as the note says.

## 6. What P8 to P11 would otherwise have to undo, and other points

1. **Point 4's move**, avoided by the alternative.
2. **The `_base(name)` helper** for twin lookups (point 2), so P8's `Kp` twins and P11's masks find the base row's
   table entries one way.
3. **P10's `sum_over_samples=False`** applies to `JtJp` as to `g`. The `W`-keeping readouts (my P5 finding A) need
   their `K` forms (`'JWKCi,JWo->WKCio'`, `'JWCi,JWKCo->WKCio'`, and the `H` family with `K` on the tilde operand).
   Put them in the P10 sentence that the plan gained, so the count of literals is known there.
4. **The dtype.** `V` and `H` count when read, under O-11; nothing new.
5. **The step counts.** `('JtJp',)` alone runs the plain sweeps `xi`, `mu`, `nu`, `eta`, the frame dots, `Jp`, and
   each tilde step once with `K`. `('g', 'JtJp')` runs each tilde step twice, as the note says. Derive the counts by
   hand in the test, as in P5 and P6.
6. **The check of `K`** (point 3) should also run on a held `Jp`: the note already says a held `Jp` must have the
   call's `K`.

For my review of the built P7:
- every new name against the reference, applied per `(V[k], H[k])`;
- `Jp` against my own dense tangent tensor (the coordinate construction of P6, with the direction itself);
- `JtJp` against my transposed dense Jacobian seeded by `ω² Jp`;
- the per-`K` identities to rounding;
- mutants: a dot term dropped, `H` taken without `K`, the frame node replaced by the public route in `Jp`, a twin
  reading the residual's tilde, and `K` from `H`;
- a size-1 `K` on `H` must raise, if point 3 is adopted.
