# P10 design note: Engine F, the stacks, the unsummed blocks and the dtype matrix

Implementer, 2026-11-20, read only, against `graded-messages` with P9b built (not committed). Sources: the plan's P10
and section 2.3 (steps 1, 4, 7), `docs/contributor/testing_strategy.md` "The stack matrix", P5's review (Finding A),
P7's `K` check and the engine as it stands.

## 0. Where the engine already is

Much of P10 is in the code already, untested:
- `W` is read as `shape(w[0])[m:-1]`, or `shape(index)[1:]` for entries, so it already has any number of axes.
- `C` is read as `shape(D[0])[:-2]`.
- `K` is read off `V[0]` or `D_dot[0]`, and P7/P8 check it.
- The record is `GroupLengths(J=m, W=len(W), K=len(K), C=len(C))`, with `K = 0` for rows that do not carry it.
- Every string is grouped (`'JWKCa,...'`), so a group of any length rides through `jet_contract`. No reshape
  happens anywhere, including P9b's gather and scatter.
- `f` keeps `C` and `Jp_norm_sq` keeps `K + C` (P5, P7).
- The dtype rule with held variables (P5's amendment) and the integer `index` (P9b) are in.

So P10 has three parts:
- the unsummed blocks;
- the structural check over `W` and `C`;
- the matrix that proves the rest.

## 1. Reading and recording `W`, `K`, `C`; the order `W + K + C`

- **One reader each.** `_read_setting` reads `W` and `C` and `_read_K` reads `K`. Each is read once, from one operand.
  Nothing downstream re-reads a stack from an array; the steps get only the lengths.
- **The order is `J + W + [K] + C + (edge,)` on every variable**, and `[K] + C + core` on a block (`W + [K] + C +
  core` when unsummed). The places that build arrays rather than contract them each follow it:
  - **Start values.** `start_ones` and `start_zeros` take `box`, `W` and `KC` (`K + C` for a row carrying `K`, else
    `C`) and append the bond. `start_node` broadcasts a node that is already `F + W + [K] + C`.
  - **The seed.** `jet_to_dual_jet` reshapes the weight to `F + (1,)*(rest)`. That broadcasts over `W + [K] + C (+ N)`
    for any lengths, because the jet axes lead. `r_step` inserts `len(C)` unit axes into `y` (`F + W (+ N)`) before
    the mode edge. `y` never carries `K`, since the residual's adjoint has `K = 0`.
  - **The gather and the scatter** move `W` to the front (`moveaxis` over `n_W` axes). `keep_W` builds `W` grids of
    any length.
  - **The sums** keep what they promise. `f` sums `J + W (+ edge)` and keeps `C`. `Jp_norm_sq` sums the same and
    keeps `K + C`. `r_block_sumsq` sums everything after `J`, `C` included, giving the weight's shape.
- **The direction stack.** `K` is one block shared by `V`, `H`, `D_dot` and `L_dot` (P8). A multi-axis `K = (2, 3)` is
  the same code with `len(K) = 2`; the matrix tests it.

## 2. The structural check over `W` and `C` (separate, removable)

**The hazard.** The engine reads `W` from `w[0]` and `C` from `D[0]`. The interpreter checks ranks only, and an
explicit label of size 1 broadcasts in an einsum (P7 met this for `K`). So a stray size-1 axis gives a wrong answer
with no error:
- a `w[i]` with `W = (1,)` where `w[0]` has `(3,)`;
- a `U[i]` with `C = (1,)` where `D[0]` has `(2,)`.

**Proposal:** one function, `_check_stacks(setting, stores, sample, Vc)`, called by one line in step 1, with one test
class. It raises a `ValueError` naming the operand, its shape and the expected prefix:
- **`C`** must be the prefix of every core of every store (`D`, `U`, `L`, `R`; `D`, `L` on a corewise point) and of
  every `Vc[i]`. The directions are already checked by `_read_K` (`K + C`).
- **`W`** must be the shape after the jets and before the mode edge of every input `w[i]`. Entries' `index` is one
  array, so there is nothing to compare.
- **Out of it:** `y`, the weight, and the jet sizes `J_in_i`. Section 2.3 step 1 gives those to the `jets` checkers,
  and a wrong `y` broadcasts only into `r`, which `check_data` catches. Held variables are already checked against
  the table, stacks included.

**Cost:** about `5d` comparisons of short integer tuples per call, in Python, on static shapes. Under `jit` that is
trace time only: nothing reaches the compiled program, and the cache key is unchanged. Eager, it is microseconds
against contractions of milliseconds. To remove it: delete the function, its call and its test class; nothing else
depends on it. I would write it so, because O-18 is open (your pause after P11).

## 3. `sum_over_samples=False`

**The literals.** There are six readout strings. Each gains a twin with `W` leading its output:
- `V` family:
  - `'JWKCi,JWo->KCio'` (the top-slot readout of `V_tilde`, `V_tilde_dot` and `V_top`);
  - `'JWCi,JWKCo->KCio'` (`eta` against the seed);
  - `'JWKCi,JWCo->KCio'` (`eta_dot` against the seed).
- `H` family: `'JWKCa,JWCi,JWCb->KCaib'`, `'JWCa,JWKCi,JWCb->KCaib'`, `'JWCa,JWCi,JWKCb->KCaib'`.

The unsummed twins replace `->KC...` with `->WKC...`. **Six, not "about nine":**
- `K` is a group, so the `K` forms for `JtJp` and `Kp` are these same strings with `K` nonzero.
- `H_tilde_dot`'s nine terms use the three `H` strings.

They go into `STRINGS`, with their library shapes; the contract and sharding counts follow.

**The flag.** A readout step takes a static `keep_W: bool = False` and picks one of the two literals in place:
`'JWKCi,JWo->WKCio' if keep_W else 'JWKCi,JWo->KCio'`. Both literals then stay in the source for the sharding scan.

**Which rows get it:** a row whose variable is a block, read from the shape table (jets `'none'`). That covers
`V_tilde`, `H_tilde`, `V_tilde_dot`, `H_tilde_dot` and their twins. So no new declared set is needed. `_run` passes
`keep_W = not sum_over_samples` to those rows only, and the memoized body key gains it, so one compile serves each
value.

**Entries:** `V_scatter_step(keep_W=True)` is built (P9b) and is now wired the same way.

**The shapes.** The setting gains the flag. `_shape` puts `W` before `[K] + C + core` for a block. `g`, `JtJp` and `Kp`
are pairs of such blocks, giving `W + K + C` in that order, as the plan says. A held block is checked against the same
table.

**Not affected:** `f`, `r_block_sumsq` and `Jp_norm_sq` stay summed over `W`. The plan names the flag for `g`, `JtJp`
and `Kp`, and these are objectives. The docstring will say so.

## 4. The dtype rule

It is already as ruled. P10 adds the matrix:
- **Counted:** the arrays the closure reads (`_deps`), plus held variables that are read or wanted. Never the integer
  `index`, and never an argument nothing reads.
- **`use_jax`** reads every array input, held ones included.
- **Cast once,** start values built in the result dtype, constants Python scalars.
- **`'w'`:** after your ruling, on entries `'w'` is the one-hot vectors, built only when wanted, in the result dtype
  (P9b round 1).

**The four backends:** numpy float64, numpy float32, jax float64 (x64 on, as the suite runs) and jax float32. The cases:
- float32 in, float32 out, for every output;
- float32 mixed with float64 gives float32;
- jax mixed with numpy gives jax;
- a float64 `y` on a call wanting only `z` in float32 gives float32;
- an int32 or int64 `index` does not count;
- the one-hot vectors of `index_to_one_hot` count like any float array;
- a held variable neither read nor wanted leaves the dtype alone, but a jax one makes the call jax.

That is one call per (case, backend), and every output of that call is checked: about 30 calls.

## 5. The tests, and their size

**The full product is too big:** 6 stack configs × 3 `W` × (3 + 3 + 1) kind-orders = 126 calls.
- Each call has about 35 names, checked per element of `K` and `C` (up to 6).
- Against the dense reference that is about 26,000 reference evaluations of 5–50 ms each: 2 to 20 minutes on one
  core. Not affordable.

**Proposal: two tiers.**
1. **The matrix, against the unstacked engine, element by element.**
   - The unstacked engine is the reference-verified one (P5 to P9b), and it costs milliseconds. So element `(k, c)`
     of a stacked call is compared with the engine on frame element `c` and direction `k`. With `W` it is compared
     with the same `W` (the engine takes any `W`).
   - **A covering design, not the product.** Every (config, `W`) pair appears once: 18 per kind. The orders are
     assigned cyclically, so each config meets each order of its kind. `(1, 1)` with `W = (2, 2)` appears for probe
     and for apply: two jet axes before two sample axes, the bookkeeping most likely to slip.
   - That is 18 × 3 kinds = 54 stacked calls plus at most 6 unstacked calls each: about 380 engine calls, roughly
     20 ms each, so 8 s single-process, about 5 s at `-n 2`.
   - The 6 configs come from the stack-matrix doc minus forced padding, which is a uniform concern (P11): no stack,
     `C = (2,)`, `K = (3,)`, `K + C`, `C = (2, 3)`, `K = (2, 3)`.
2. **An anchor against the reference:** one diagonal of 6 calls, each config once, probe at `(2,)` with `W = (2, 2)`,
   every name, element by element. That is about 35 × 6 × 6 ≈ 1,300 reference calls, about 30–60 s single-process.
   **This is the expensive part.** If that is too much, take the anchor at the configs with a multi-axis group only
   (2 of 6), about 15 s.
3. **Unsummed** (`sum_over_samples=False`), on 4 of the matrix's cases (a `C`, a `K`, a multi-axis `W`, entries):
   - the blocks summed over `W` equal the summed call;
   - the block of sample `w` equals the summed call on the sample `w` alone;
   - the shape is `W + K + C + core`.
4. **The structural check:** a size-1 `W` in `w[2]`, a size-1 `C` in `U[1]`, a missing `C` axis in `Vc[0]`, each
   with its message; and a matching stack passes.
5. **The dtype matrix** of section 4.

**Estimated cost at `-n 2`:** about 1 minute added to the suite with the full anchor, about 20 s with the thin one.

## 6. For P11, and changes to the plan

**For P11:**
- On the uniform layout the stacks are read off the supercores: `C = shape(D)[1:-2]`, after the leading `d`; `W` off
  the packed sample. `_read_setting` and `_check_stacks` gain a uniform branch, and nothing below them changes,
  because the steps see only `GroupLengths`.
- The unsummed blocks keep `W` unreshaped, so they shard over `W`. P11's sharding test should include them.
- The gather on a padded `D` stays in real columns (the reviewer's P9b note), and so does the scatter's write.

**Plan changes I'd propose:**
1. "About nine" unsummed readout strings becomes **six** (section 3).
2. Say that `f`, `Jp_norm_sq` and `r_block_sumsq` ignore `sum_over_samples`.
3. Add the structural stack check as an explicit, removable item pending O-18.
4. Have the stack matrix run against the unstacked engine, with a reference anchor on a diagonal, rather than the
   product against the reference (section 5).
