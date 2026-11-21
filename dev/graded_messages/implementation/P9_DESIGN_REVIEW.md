# P9 design review: the implementer's note on Engine E (apply and entries)

Reviewer, 2026-11-20, on `P9_DESIGN_NOTE.md` (read only; nothing run). Against `SAMPLE_VARIABLES.md` sections 5.3,
5.4, 6 and 8, the plan's section 2.3 and P9 to P11, row 4.4, and the rulings so far. Weighed by simplicity,
performance and memory, with values equal to rounding taken as the same.

**Overall: sound, and economical.** The mathematics in section 0 is right, with tails. Reusing the existing steps with
different arguments is exactly the maintainer's rule. I agree with most choices.

On the zero tilde names, I lean to the implementer's recommendation, with two additions, and agree it should go to
the maintainer. I recommend the split into P9a and P9b. One thing for P10 to know now: entries' scatter must also
keep `W`.

---

## 1. The fact it rests on: right, with tails

For apply and entries the value is `z = <mu[d], 1>`, bond `d` closed by the ones of the end bond. The objective's
seed `s` (one number per sample and slot) multiplies that closing vector. So in the pairing `<s, z>` the seed sits at
bond `d`:
- **Every side that reaches bond `d`** (a right side, a trunk) has the factor `s`. Its recurrence is the plain one,
  started from `s ⊗ 1`.
- **A left side or a leaf** never reaches bond `d`, so has no place for the seed: `xi_tilde` and `mu_tilde` are
  zero, and so are their tilde-dots.

That is `SAMPLE_VARIABLES.md` 5.3 ("A side that reaches bond `d` ... has the seed as a factor ... A side that does
not ... its tilde variable is zero"), and section 8's note that the rows of `mu_tilde` and `xi_tilde` do not hold for
apply and entries.

The details check out:
- **As jets.** The plain recurrence started from `w_tilde ⊗ 1` is the truncated product `w_tilde · nu`. The
  recurrence is linear in the carry, and the box product is associative, so the seed can be multiplied in at the end
  or at the start. The implementer's planned identity test (`nu_tilde = w_tilde · nu` as a jet product) states this.
- **Tails** (row 4.4: the boundary is ones sized by the end bond of the cores passed).
  - `nu_tilde` starts at `s ⊗ 1(rR_d)` (the `R` store's end bond), `nu_L_tilde` at `s ⊗ 1(rL_d)`.
  - The value closes `mu[d]` with `1(rL_d)`, and `Jp` closes `mu_R_dot[d]` with `1(rR_d)` (the direction's sweep is
    in `R` after the direction). That is consistent with the tangent's definition, whose right boundary is the `R`
    store's.
  - So `close_step` must be a sum over the last axis, not a squeeze, as the note says.
- **The tilde-dots.** `nu_L_tilde_dot` is `nu_dot_step` with `nu_L_tilde` in `nu`'s place, from zero: the
  derivative of `s · nu_L` along the raw direction, with `s` held. `eta_D_tilde_dot` is `eta_dot_step` with the tilde
  forms. Right.
- **`H_tilde_dot`.**
  - Of the nine `H_term`s, only those with the tilde on `nu_L` survive: `(mu_dot, xi, nu_L_tilde)`,
    `(mu, xi_dot, nu_L_tilde)`, `(mu, xi, nu_L_tilde_dot)`.
  - `H_tilde_step(mu_dot, mu, xi_dot, xi, nu_L_tilde_dot, nu_L_tilde)` gives exactly these three.
  - Its strings put `K` on the first argument of each term (`'JWKCa,...'`, `'..,JWKCi,..'`, `'..,JWKCb->..'`). Those
    are the dotted operands here, and `nu_L_tilde` has no `K`. So the `K` placement is right too.
- **`V_tilde`** is only `V_term(eta_tilde, w)`, and `V_tilde_dot` only `V_term(eta_D_tilde_dot, w)`: no seed at a
  leaf. Right.

## 2. The graphs: agree, with one structural request

- **Shared blocks.** `_PLAIN_ROWS` and `_DOT_ROWS` shared by every kind, plus each kind's own rows, is the simplest
  correct form. Apply's adjoint is a different recurrence, not a renaming, so building it is right, and deriving it
  would be forced.
- **Entries derived from apply by a declared step table** (the gather for the leaves, the scatter for `V`, `w` read
  as `index`, no short nodes at `order = ()`), with an import-time check: right. It is the same pattern as the
  corewise derivation, one mechanism.
- **Six graphs, `(kind, call type)`:** right.
- **The request: one place that lists the paths.** `_JP_PATH` becomes per kind and per call type. Today there are
  `_JP_PATH` and `_JP_PATH_COREWISE`. Make it one table keyed like `_GRAPHS`, and derive each graph's twins from its
  entry, so P10 and P11 never look for a fourth spelling. A test checks that every path's names exist in its graph.
- **The import-time invariants** (entries reads no `w` and no short node; the corewise ones per kind) should
  `raise`, as P8's do now, not `assert`.

## 3. The zero tilde names: leave them off, with two additions; and tell the maintainer

The three considerations:
- **Consistency with the definitions.** `SAMPLE_VARIABLES.md` defines them for apply and entries, as zero, and the
  reference module returns zeros. Row 6.4 says "every variable of a kind's graph ... can be wanted". Whether a
  structural zero is "a variable of the kind's graph" is exactly what the maintainer can settle in one line.
- **Simplicity.** A zeros row needs its shape, and the steps make shapes only from their operands. `mu_tilde`'s zeros
  could take `mu`'s shape, but `xi_tilde` is full (`F`), so it would read `eta_D` or a store only for a shape. That is
  a false dependency the closure would then compute. Or it needs a new `'zero'` driver that reads the shape table: a
  new concept for four names. Leaving them off costs one message.
- **Generic code.** The oracle and the optimizers ask for `f`, `g`, `JtJp`, `Jp_norm_sq` and `Kp`, never a
  left-side tilde. Code that iterates over the names (tests, a user's diagnostics) needs to *discover* a kind's
  names, which neither form gives today without the docstring.

**Recommendation:** off the graph, as the implementer proposes, and:
1. **The message says it is zero and why**, as proposed ("it is zero there, since the seed enters at bond `d`").
2. **A small public reader of the names** a call accepts, `sampling_variables_names(kind, call='manifold')` (or the
   same as an attribute), returning the public names of the graph. Generic code then discovers the set, and the test
   ORCHESTRATION section 6 asks for compares it with the reference module's names per kind, the difference being
   exactly the declared zero names. If the maintainer prefers zero rows, the reader is still the right companion.

This is worth his one line: it decides what "the names of a kind" means in the documentation (P19). Put to him as:
"on apply and entries, are `xi_tilde`, `mu_tilde`, `xi_tilde_dot` and `mu_tilde_dot` names that return zeros, or
unknown names whose message says they are zero?"

## 4. Entries' `V` block: agree with the scatter-add, and make it keep `W` for P10

- **Memory.** `W × n` instead of the one-hot product's `W × N` intermediate is the memory-over-compute preference.
  Bitwise agreement with the old kernels has low weight, and the bridge compares to tolerance.
- **The forms.** jax `.at[..., index].add` and numpy `np.add.at` both handle duplicate indices correctly. That is
  the one trap: `+=` drops duplicates, and the planned mutant catches it.
- **On numpy speed.** `np.add.at` is slow, but correct. Part 2 can swap in a faster numpy form behind the step
  (`np.bincount` per column, or a sort and `add.reduceat`), so do not optimize now.
- **For P10** (`sum_over_samples=False`): the block keeps `W`, `W + K + C + (n, N)`. The scatter then writes each
  sample's row into its own slice, `out[w, ..., :, index[w]] += eta[w, ...]`, which is a different index expression
  from the summing form. Write `V_scatter_step` so that keeping `W` is a static argument from the start, with the
  summing form the default. Then P10 adds a test, not a second step. The probe and apply readouts need their `W`-kept
  literals in P10 too (my P5 finding A); entries needs no literal, only this flag.

## 5. `_deps(rule)`, `nu_step`/`eta_step` with `K` on the carry, whole steps with `kind`: agree

- **`_deps`.** One helper, reads plus a start node, used by the closure, the last-read drop, the dtype count and both
  renames. Right: a start node is a dependency that is not a per-site read, and every graph walk must see it. The
  tests the note lists are the right ones. Add one for the corewise rename carrying a start node: apply's corewise
  `nu_tilde` (the renamed `nu_L_tilde`) starts from `w_tilde`.
- **`K` on the carry of `nu_step` and `eta_step`.** Right. `_Jp.nu_tilde` carries `K` through `nu_step`, and the
  strings are already in `STRINGS` (the tilde steps' terms). With `K = 0` they are the old contraction.
  - The literal `'Caib,JWCi,JWCb->JWCa'` leaving the source is fine: `STRINGS` lists the case file, guard (ii) only
    asks that library literals are in it, and the row-5.5 expected failure reads `STRINGS`, not the source.
- **Whole steps with `kind` as a static.** Right. The math differs only in the mode axis, `jets.jet_to_dual_jet`
  already takes the kind, and the rows are then shared. The kind is a Python string, never traced.

## 6. A `Vc` given to apply or entries: agree, accept it silently

Today an argument nothing reads is accepted silently (`y` without `r` wanted, `weight` without `f`). `Vc` on apply
is the same case, and the oracle passes the geometry's centers without knowing the kind. Raising would make the
oracle kind-aware for no benefit. State it in the docstring: "`Vc` is read only by the probe kind's `z`".

## 7. Size: yes, split it

- **P9a, apply (manifold and corewise):**
  - the row blocks;
  - `_deps` and start nodes;
  - per-kind paths;
  - `close_step`;
  - apply's adjoint and `Kp`;
  - the bridge to the apply derivatives kind.
- **P9b, entries:**
  - the derived graph by the step table;
  - the gather and the scatter;
  - `index` among the arguments;
  - the `t3_entries` and entries-derivatives bridges.

Each is then reviewable with an independent oracle: apply by my dense oracle and the dense Jacobian or Hessian
constructions of P6 to P8; entries against apply at `index_to_one_hot`. And P9b's step table depends on P9a's rows
being final. The split costs one extra gate.

## 8. Tests, mutants, and what P10 and P11 must not have to undo

- **The tests listed are right.** Add:
  - the corewise rename of a start node (point 5);
  - the name reader against the reference per kind (point 3);
  - apply's `Jp` with tails whose `rL_d ≠ rR_d`. That case is where closing `mu_R_dot` with `rR_d` ones, not `rL_d`,
    shows.
- **The mutants listed are right.** Add:
  - `close_step` of `Jp` using the `L` end bond;
  - entries' gather reading `index` of the wrong mode;
  - apply's `nu_tilde` started from the turned seed's wrong slot. The seed `w_tilde` is the dual jet; the start is
    `w_tilde ⊗ 1`, not `r`.
- **For P10:** the scatter with `W` kept (point 4); `index` excluded from the dtype count (the note's section 3 says
  so: write it into `_ARGUMENTS` with the note).
- **For P11:** the gather on a uniform layout reads `D`'s real columns (`index < N_i`), and the padding of `N` is
  masked. A garbage-padded `D` must not reach the result through the gather. That is the same contract as today's
  `_entry_xis`, and the uniform tests will check it.

**For my review of P9a:**
- every apply name against the reference (zeros for the zero names);
- `z`, `g` and `Kp` against my dense oracle, the transposed dense Jacobian and the dense second derivative;
- `nu_tilde` against the jet product `w_tilde · nu`;
- each sweep once on apply `('z', 'g')` (no plain `nu`/`eta` sweep);
- tails with `rL_d ≠ rR_d`;
- the floor env.
