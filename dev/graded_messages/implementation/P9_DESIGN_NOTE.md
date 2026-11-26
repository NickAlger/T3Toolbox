# P9 design note: Engine E, apply and entries

Implementer, 2026-11-20, read only, against `graded-messages` with P8 built (not committed). Sources: the plan's P9
and section 2.3; `SAMPLE_VARIABLES.md` sections 3 to 8; the reference module's docstring (the seed "multiplies the
vector of ones that closes bond `d`"); the README's "Things learned".

## 0. The fact the design rests on

For apply and entries the seed `s` (one number per sample, `F + W + [K] + C`) sits on the ones that close bond `d`.
So every right side and trunk is **the plain recurrence started from `s ⊗ ones` instead of `1 ⊗ ones`**, and every
left side and leaf has no tilde (zero):

- `nu_tilde[i] = nu_term(R[i], xi[i], nu_tilde[i+1])` from `nu_tilde[d] = s ⊗ ones(rR_d)`: this is `nu_step` itself.
  `nu_L_tilde` is the same through `L`.
- `eta_tilde = eta_term(mu, U, nu_tilde)`: `eta_step`. `eta_D_tilde` likewise.
- `nu_L_tilde_dot` = `nu_dot_step(L, xi, carry, L_dot, xi_dot, nu_L_tilde)`, from zeros. `eta_D_tilde_dot` =
  `eta_dot_step` with `nu_L_tilde` and `nu_L_tilde_dot` in the places of `nu_L` and `nu_L_dot`.
- `H_tilde = corner H_term(mu, xi, nu_tilde)`. `H_tilde_dot` is `H_tilde_step` with different arguments:
  `(mu_dot, mu, xi_dot, xi, nu_L_tilde_dot, nu_L_tilde)` gives the three terms with one mark on each operand.

The steps don't know that a tilde operand is stored turned. A truncated product with a turned operand, read at the
corner, is the pairing; P5 to P8 already rely on that. So these steps can be reused as they are. This is the
maintainer's rule: no extra step function when it does the same thing with different arguments.

## 1. The graphs

**Recommendation: build the graphs from shared blocks of rows, not by a transform of probe's.**

The corewise derivation is drop-and-rename. Apply's adjoint is not a renaming of probe's: different recurrences, a
different seed type, a different start. So:

- **Split `_PROBE_ROWS` into blocks:**
  - `_PLAIN_ROWS`: `xi`, `mu`, `nu`, `nu_L`, `eta`, `eta_D`;
  - `_DOT_ROWS`: frame and raw dots, the O-12 routes, `Jp_norm_sq`;
  - the probe-only rows: `z`, `Jp`, the seed, tilde, tilde-dot, blocks, `g`, `Kp`.
  - The plain and dot grades depend only on the frame and the inputs, so they are the same for every kind. (For
    entries, see section 3 for which steps change.)
- **Apply's own rows:**
  - `z` reads `mu`; `Jp` reads `_frame.mu_R_dot`, or `mu_dot` on the raw route. Both use one new whole step,
    `close_step`: bond `d` contracted with the closing ones, which is a sum over the last axis. With tails it is a
    sum, not a squeeze.
  - The seed `w_tilde`.
  - `nu_tilde` and `nu_L_tilde`: `nu_step`, starting at the seed.
  - `eta_tilde`, `eta_D_tilde`: `eta_step`.
  - The short node `_eta_tilde_top` and the corewise `_eta_D_tilde_top`: a new `eta_top_step`, the short form of
    `eta_step` (one readout per slot).
  - `V_tilde`: a new `V_top_step`, probe's first readout alone. `H_tilde`: a new `H_readout_step`, probe's third readout
    alone.
  - The tilde-dots as in section 0. `_eta_D_tilde_dot_top` uses a new `eta_dot_top_step` (three readouts).
    `V_tilde_dot` uses `V_top_step`. `Kp` uses `g_step`.
  - The corewise routes of `V_tilde` and `H_tilde`, as in probe.
- **Entries = apply's rows under a declared step table:**
  - `xi_step` becomes `xi_gather_step`;
  - `xi_dot_step` becomes the same gather of `D_dot`, or of `V` for `xi_U_dot`;
  - `V_top_step` becomes `V_scatter_step`, reading `eta_tilde` (or `eta_D_tilde_dot`) directly. At `order = ()` the
    short node is the whole trunk, so the entries graph has no short nodes;
  - `w` becomes `index`.
  - This one is a real derivation, as corewise's is, with an import-time check that no row of the entries graph
    reads `w` or a short node.
- **What it needs:**
  - **K on the carry of `nu_step` and `eta_step`.** The twin `_Jp.nu_tilde` carries `K`. The strings become
    `'Caib,JWCi,JWKCb->JWKCa'` and `'JWCa,Caib,JWKCb->JWKCi'`. They are already in `STRINGS` (the tilde steps' terms)
    and equal the old ones at `K = 0`. The plain literals leave the source; the contract test counts follow.
  - **Whole steps take `kind` as a static** (`r`, `f`, `r_block_sumsq`, `w_tilde`, `Jp_norm_sq`). The math is the same
    up to the mode axis, and `jets.jet_to_dual_jet` already takes the kind. The rows are then shared.
  - **No new literals,** as far as I can see.

**Each sweep once.** Apply's `('z', 'g')` runs `xi` d, `mu` d, the `nu_step` (tilde) d, `eta_top` d, `V_top` d, `H` d,
and the whole steps once each. There is no plain `nu`/`eta` sweep, which is the README's point. A step-count test
pins it.

## 2. The seed and the starts

- **A start may name a node** (the P6 ruling). The value is the node times the closing ones vector of the sweep's end
  bond: `start_seed(node, size)`, with `F + W + [K] + C + (size,)`. With tails `rR_d` and `rL_d` differ, as for
  `start_ones`.
- **The start node cannot be a read**, because reads are sliced per site. So **one helper, `_deps(rule) = reads +
  (start,)` when the start is a node**, serves:
  - the closure, the last-read drop and the dtype count;
  - the twin rename (`'w_tilde'` becomes `'_Jp.w_tilde'`; P8 review, P9 note);
  - the corewise rename.
  - A test should check each of these.
- **The seed step:** `w_tilde_step` with `kind`, giving `F + W + C` for apply and `W + C` for entries.
- **The zero tilde variables** (`xi_tilde`, `mu_tilde`, `xi_tilde_dot`, `mu_tilde_dot`). **Recommendation: they are
  not on apply's and entries' graphs.**
  - Asking for one raises: "unknown variable 'mu_tilde' for the apply kind: it is zero there, since the seed enters
    at bond d".
  - A zeros row would cost nothing unless wanted, and would match the reference module, which returns zeros. But it
    would be the only row no output reads, and a structural zero is free for a caller to build.
  - The plan's sentence ("every name of a kind's graph can be wanted") holds either way. This is easy to switch (a
    site row with a zeros step), so I'd put it to the maintainer.

## 3. Entries

- `index` is `(d,) + W`, integer, and is not counted for the dtype or cast. The rule already speaks of floating point
  arrays; I would add `index` to `_ARGUMENTS` with that note.
- `xi[i] = D[i][..., index[i]]`, moved to `W + C + (n,)`, as the old `_entry_xis` does.
- An `order` other than `()` raises before the graph is read (section 2.3, step 1).
- **`V_scatter_step` is the decision here.** The old transpose contracts with one-hot vectors (`W × N` per mode).
  I recommend a scatter-add instead: `.at[..., index].add` on jax, `np.add.at` on numpy. Its memory is
  `W × n`, which is the memory-over-compute preference. The cost is a slower numpy path; jit and sharding over `W`
  are a reduction either way. The fallback is the one-hot product through `V_top_step`, which also matches the old
  kernels bitwise.

## 4. Dots, tilde-dots, `Kp`

- The dots come free (shared block). `Jp` needs only the left dot sweep, `close(mu_R_dot)` or `close(mu_dot)`; no
  `nu_L_dot` or `eta_D_dot`.
- `Jp_norm_sq` comes with the kind static. `JtJp` is the twin of `g`, along apply's path (`w_tilde`, `nu_tilde`,
  `_eta_tilde_top`, `V_tilde`, `H_tilde`). So **`_JP_PATH` becomes per kind, and per call type.**
- `Kp` is as in section 0. There is no `w_tilde` term in `V_tilde_dot`, since the seed never sits at a leaf.

## 5. Corewise

Yes, the same derivation: `_corewise(rows, kind)` runs on each kind's rows, the renames carry starts, and the two
assertions and the reachability loop run per kind. `_GRAPHS` gets six entries.

## 6. Tests

- **By definition:** apply on the jet matrix and entries at `()`, against `sdef`. The zero names are checked as
  unknown, and against the reference's zeros in the message's sense.
- **Apply's own identity:** `nu_tilde` equals the jet product `w_tilde · nu`.
- **The identities of P5 to P8 for both kinds,** without the `mu_tilde`/`xi_tilde` multiplier rows (section 8).
- **Entries against apply** at `index_to_one_hot`, to tolerance.
- **Step counts:** apply `('z','g')` and `('Kp',)`; entries `('g',)`.
- **The `_deps` cases** of section 2.
- **The bridge:**
  - the apply derivatives kind (forward, transpose, `corewise_hessian`) at `affine_jet(ww, pp)`;
  - `t3_entries` and the entries kind at order 0;
  - `t3_entries_derivatives` against apply at `affine_jet(index_to_one_hot(index, shape), pp)`, with the `t!`
    conversions of P7 and P8.
- **Mutants:** the seed start replaced by ones; `close_step` squeezing instead of summing (with tails); the scatter
  dropping duplicate indices (`np.add.at` against `+=`); `_deps` ignoring the start in the twin rename.

## 7. Plan changes I'd propose

1. Say in the plan that apply's scalar output is **bond `d` closed by the ones**, which is a sum with tails.
2. Whole steps take the kind. `nu_step`/`eta_step` carry `K` on the carry.
3. `_deps`, and the start node in every rename.
4. Settle the zero-tilde names (section 2) and the scatter (section 3).
5. A `Vc` given to apply or entries is read by nothing (`z` needs no `eta` there): accept it silently, as an unread
   argument is today, or raise. I'd accept.
6. **Size.** This is "medium" in the plan, but with the row split, `_deps`, the per-kind paths and corewise it is
   closer to P8. If the coordinator wants two slices: P9a apply (manifold and corewise), P9b entries.
