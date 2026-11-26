# P10 design review: the implementer's note on Engine F (stacks, unsummed blocks, dtype)

Reviewer, 2026-11-20, on `P10_DESIGN_NOTE.md` (read only; nothing run). Against the plan's P10 and section 2.3 (steps
1, 4 and 7), `docs/contributor/testing_strategy.md` ("The stack matrix"), my P5 finding A and P7 ruling, and the engine
as I read it in P5 to P9b. Weighed by simplicity and performance, with values equal to rounding taken as the same.

**Overall: agree with the design.** "Six literals" is right. The stack check has the right scope. I agree with testing
against the unstacked engine, with one real gap to close: as written, the oracle shares the multi-axis `W` with the
code it checks. Two small additions to the covering design close it at little cost.

---

## 1. "Most of P10 already works, untested": agree, by my reading

From the code I reviewed in P5 to P9b:
- **`W`** is read as `shape(w[0])[m:-1]` (`shape(index)[1:]` for entries) and **`C`** as `shape(D[0])[:-2]`. The steps
  get only `GroupLengths`.
- **`r_step`** inserts `lens.C` unit axes into `y` before the mode edge (P5). **`f_step`** sums the jet axes, `W` and
  the mode edge and keeps `C`. **`Jp_norm_sq_step`** keeps `K + C`. **`r_block_sumsq`** sums all but the jet axes,
  giving the weight's shape. The plan says so for all four.
- **The starts** take `KC` (P7). **`start_node`** broadcasts a node already `F + W + [K] + C` (P9a).
- **`jet_to_dual_jet`** reshapes the weight `F + (1,)*rest`, which broadcasts over any trailing stacks, the leading jet
  axes being first.
- **The gather and the scatter** move `W` over `n_W` axes and build `W` grids of any length (P9b).
- **No reshape anywhere**, so a group of any length rides through `jet_contract`.

So the claim holds as a reading. The matrix is what turns it into evidence (point 4).

## 2. `_check_stacks`: agree, the right scope

- **The operands in scope** are exactly those that *define or carry* a stack the engine reads from one place:
  - `C` from `D[0]`, so every core of every store, and `Vc`;
  - `W` from `w[0]`, so every `w[i]` after its `m` jet axes;
  - `K` from `V[0]` or `D_dot[0]`, already done by `_read_K`.

  These are where a size-1 axis broadcasts silently into a wrong answer. It is the P7 ruling, extended to the same
  hazard.
- **`y`, the weight and the jet sizes stay with the `jets` checkers** (section 2.3 step 1). That is right:
  - `y` enters only `r`, the weight only the sums and the seed;
  - section 2.3 decides that the engine checks neither;
  - a wrong `y` is a data error the frontend's `check_data` catches.
- **Held variables** are already checked against the table, stacks included.
- **The comparison must be exact equality of the stack tuples**, not broadcast compatibility. A consistent `C = (1,)`
  everywhere is legitimate; a `(1,)` beside a `(2,)` is not. The note's wording ("must be the prefix") says so.
- **Written as removable,** one function, one call, one test class: right while O-18 is open. Under `jit` it costs
  only trace time.

## 3. `sum_over_samples=False`: six is right; the flag design is right

**Deriving the six.** The blocks are read at the corner by the `V` and `H` readouts. Collecting every readout string
the block rows use, across kinds, grades and the `Jp` twins:

*`V` readouts:*

| String | Used by |
| --- | --- |
| `'JWKCi,JWo->KCio'` | `V_term(trunk tilde, input)`: probe's and apply's `V_tilde` (and its twin), `V_tilde_dot`, apply's `V_top` |
| `'JWCi,JWKCo->KCio'` | `V_term(eta, seed)`: probe's `V_tilde`; its twin carries `K` on the seed |
| `'JWKCi,JWCo->KCio'` | `V_term(eta_D_dot, seed)`: probe's `V_tilde_dot` |

*`H` readouts:* `K` on the first, second or third operand, three strings. They serve:
- probe's `H_tilde` (the tilde on each operand);
- its twin;
- `Kp`'s nine terms (`K` on the dotted operand);
- apply's `H_readout_step` (`K` on `nu`'s place);
- `H_tilde_dot` on apply.

There are no other `V` or `H` readouts:
- the lowered-corner readouts of the short nodes (`'..->WKCi'`) already keep `W`, and are not blocks;
- the corewise routes use the same strings with other operands;
- entries' `V` is the scatter.

So six unsummed literals (`'->WKCio'`, `'->WKCaib'`), and `K` being a group is why the `K` forms are not more strings.
"About nine" in my P5 finding A counted the `K` placements as if they were separate. Six is the right count.

**The flag:**
- a static `keep_W` choosing between two literals in place, both visible in the source;
- passed only to rows whose variable is a block, read off the shape table (jets `'none'`) rather than a new declared
  set;
- joining the memoized body key.

Right on all three. The output order `W + [K] + C + core` matches section 2.3 ("W joins the stack, W+K+C").
**`f`, `r_block_sumsq` and `Jp_norm_sq` ignore the flag:** right, the plan names it for `g`, `JtJp` and `Kp` only.
Say it in the docstring and the plan, as proposed.

## 4. The test strategy: agree with the unstacked oracle, with two additions

**Is the unstacked engine a sound oracle for stacking?** Mostly yes. A stacked call with `C = (2, 3)` expands `C` into
two einsum letters on every operand. The unstacked call on frame element `c` has `C = ()`, a genuinely different
contraction: different strings handed to `contract`, different ranks, no group axes. Likewise for `K`. An error in how
a group of several axes is laid out, broadcast or summed would make element `(k, c)` differ from the unstacked call.
And the unstacked engine is itself pinned against the reference (P5 to P9b) and against my dense oracles. So for `C`
and `K` the oracle is independent where it matters.

**The gap: `W` is not sliced.** The note compares "with the same `W` (the engine takes any `W`)". Then a bug in the
handling of a **multi-axis `W`** is present in both the stacked call and its oracle, and cancels. Examples:
- an axis count `m + n_W` off by one in a sum;
- `_modes` or `f_step` summing `range(m + lens.W)` against the wrong offset;
- the scatter's `moveaxis` over `n_W` axes;
- the `keep_W` grids.

The reference anchor sees `W = (2, 2)` only for probe at `(2,)`.

*Close it, either way:*
- **(a)** For the cases with a multi-axis `W`, compare against the unstacked engine on `W` **flattened** to one axis
  (`(2, 2)` to `(4,)`): the same samples, a different group length. Outputs that keep `W` (`z`, `r`, the variables,
  the unsummed blocks) compare after a reshape; the sums compare directly.
- **(b)** Compare against calls at `W = ()`, one per sample, with the blocks summed.

(a) is cheaper and enough: it pins the multi-axis bookkeeping against a single-axis path.

**What else the covering design misses that the full product would catch: interactions of three or more factors.**
- Covering every (config, `W`) pair once per kind, with the orders cyclic, catches every bug that needs two factors
  together. It can miss one that needs three, for example a multi-axis `K` *and* a multi-axis `W` *and* two jet axes,
  where an offset is computed as `m + n_W + n_K`.
- *Add one maximal case per kind:* `K = (2, 3)`, `C = (2, 3)`, `W = (2, 2)`, and the order `(1, 1)` for probe and
  apply (entries at `()`). Every group multi-axis at once. That is three more calls, and it covers the
  highest-interaction corner, where an additive offset error shows first.

**Two smaller additions:**
- **A held stacked variable** in at least one matrix case (a held `mu` with `C` and `W`), compared with the recomputed
  one. The held-shape table with stacks is otherwise only covered at `C = ()`.
- **The unsummed tier also on a multi-axis `K`.** The `'->WKC...'` strings put `W` before a two-axis `K`, the case
  most likely to transpose.

**The anchor:** I would take the thin anchor (the two configs with a multi-axis group) plus the maximal case of point
4 at probe `(2,)`, against the reference. With the flattened-`W` oracle and the maximal cases, the full diagonal adds
little.

## 5. The dtype matrix: agree, with two cases added

The cases listed are right.
- **A held variable of another precision that is read** (float64 held `mu` in a float32 call gives float32 outputs).
  This was P5's ruling, and the place to test it across backends.
- **Entries' `'w'`** (the one-hot vectors, P9b round 1) in the result dtype, float32 with a float32 frame, as the
  ruling requires.

Jax float64 needs x64. The suite enables it in a context (P4c's `jax_in_double_precision`), so use that and do not
depend on a global setting.

## 6. What P11 needs

- **The readers.** `_read_setting`, `_read_K` and `_check_stacks` gain a uniform branch:
  - `C = shape(D_sc)[1:-2]`, after the leading `d`;
  - `W = shape(packed sample)[1 + m:-1]`;
  - `K` off the stacked `V`.

  Nothing below changes, since steps see only `GroupLengths`. Keep `_check_stacks`'s uniform branch as removable as
  the ragged one.
- **The unsummed blocks keep `W` unreshaped,** so they shard over `W`. Include them in P11's sharding test.
- **The gather and the scatter stay inside the real columns of a padded `D`.** The index is below the real `N_i`, and
  the masked padding columns are zero. The uniform tests should put garbage in the padding of `D` and check the real
  parts.
- **`keep_W` on uniform** goes through `xmap` like any site row. Its memo key already includes the flag.

## Proposed plan changes: agree with all four

1. Six unsummed literals.
2. `f`, `Jp_norm_sq` and `r_block_sumsq` ignore `sum_over_samples`.
3. The removable stack check, pending O-18.
4. The matrix against the unstacked engine with a reference anchor.

Add to item 4: the flattened-`W` comparison and the maximal case per kind (point 4).
