# M1 design review: the implementer's note on `data_metric`, `blocks_to_weight`, the converters, the laws and the grammar

Reviewer, 2026-11-22, on `M1_DESIGN_NOTE.md` at `cb3c0fdc`, read only. I read it against:
- the plan's §2, §3, §4 (Q7), 9.0 to 9.3, §8 and the M1 row;
- the maintainer's notes of 2026-11-22;
- the M1 points of plan reviews r3 (point 6), r4 (minor points 1 to 3) and r7 (points 2 and 4);
- the code it touches: `jets.py` (`_lowest_precision_dtype`, `_squared_weights`, the converters),
  `backend/fitting.py` (`LeastSquaresBatch`), `fitting.py`, `data_preparation.py` (the rescalings) and
  `signature_style.md`.

I ran one check, on decision 5.

**Verdict: the design is sound, and all five decisions are agreeable. Build it with findings 1 to 4 folded in.**
Each is small; none needs another design round.

## What holds

- **Homes and exports match plan §3.** Everything is defined in `jets.py` (which imports only `contractions` and
  `common`, so the engine and the oracle both reach it with no cycle) and re-exported by `t3toolbox.fitting`. The
  method sits on the class in `backend/fitting.py`. Both factors are required, with no defaults. There is no `kind`
  or `order`.
- **The dtype follows today's rule.** `_omega_squared` casts `ω` to the lower precision of the pair and then squares
  (`ω*ω`), exactly as `_squared_weights` does (`jets.py:289-290`). So `data_metric(r, blocks_to_weight(…), None)`
  equals today's `_squared_weights(…) · r` bitwise: same cast, same square, and a single multiply, which is
  commutative in IEEE. The test the note proposes is the right bridge for M2.
- **`γ` is handled correctly.** Casting it to the result dtype, so that a float64 0-d `γ` cannot upcast a float32
  `r` under numpy 2, is right. So is skipping it for `None`.
- **`blocks_to_weight`.** It places the blocks on the leading jet axes, with size 1 on `W`, `C` and the edge; reads
  `len W + len C` from `y`'s rank (ragged, packed, apply, entries); and keeps the blocks' dtype and backend. That is
  r7's verified placement.
- **The law helper.**
  - `JetLaw` refuses the vacuous settings (`ℓ_j = 1`, repeated values, all orders zero: r3).
  - `InputLaw` carries the full table, with the entries law and the ambient output compared as a CP tensor (r4
    minor point 1).
  - The converters are checked under both readings (r4 minor point 2).
  - The rescalings are checked as the library's own instances of the laws. I checked the conventions against
    `data_preparation.py:279-300, 391-420, 495-515`:
    - `rescale_direction` multiplies slot `k` by `ℓ^k`, so `P`;
    - `rescale_coordinates` multiplies the inputs by `Δ` and divides the probe data, so `V_*`;
    - `t3_unscale_coordinates` multiplies the Tucker columns by `Δ`, so it inverts `tucker`.

    The note's laws 3 and 4 read them correctly.
- **M1 changes no existing number.** The engine, the oracle, the checkers, `prepare_derivative_data` and the
  converters' code are untouched. The only removals are the frontend converter and O-20's check, which 9.2
  supersedes.

## Findings

### 1. SHOULD FIX: the mode count of a ragged `r` is exact; only the packed case has the size-1 limit

The note's rule: "an array whose first axis is not 1 or `len(r)` raises". So a ragged `r` (a tuple) accepts an array
weight with a mode axis of 1, broadcast over the modes.

Plan §2 says the opposite: "The number is exactly `d`; a mode axis of 1 is not broadcast (6.7's explicit point,
kept)". The plan's edge case (b), "a packed weight whose mode axis has size 1 broadcasts unchecked", is a documented
limit of the **packed array-with-array** pairing only. There, it is ordinary elementwise broadcasting and `pack_batch`
builds packed weights from checked ragged ones.

**Resolve:** for a tuple or list `r`, the weight must be a tuple or list of length `len(r)`, or an array with
`shape[0] == len(r)`. A first axis of 1 raises. Test it.

### 2. SHOULD FIX: `data_metric` must not change `r`'s shape; check that each axis of `ω` is 1 or `r`'s size

Equal rank alone lets a weight **grow** the result. Take an `r` with a size-1 axis, for example `W = (1,)` on a
one-sample batch, or a `C` of 1, and a full-size weight from another batch. The product broadcasts to the weight's
size and returns an array larger than `r`, with no error.

That is the §2 rule ("each axis 1 or `z`'s size") broken at the one public `M`. It is also close to the M2 failure
the plan names: a full-form weight that missed `take`.

The check is static and cheap, so it runs under `jit`: per pair, `ω.shape[a] in (1, r.shape[a])` for every axis.
Raise a `ValueError` naming the axis. Test it with an `r` whose `W` axis is 1 and a weight whose `W` axis is not.

### 3. SHOULD FIX: say what the docstrings carry, for `data_metric` and for the retyped backend converters

The note lists the converters' code as untouched, but plan M1 and 9.2 also **retype** them. Two pieces of text are
due in M1:
- **`data_metric`'s docstring** is one of the three places plan §4 names for Q7. It should say:
  - data are in Taylor coordinates everywhere;
  - the recipe for a data-relative weight: convert the data, then compute the weight;
  - the contragredient rule for a `Y` element held in derivative coordinates (D4, §1: convert by
    `taylor_to_derivatives`);
  - and how the result feeds a transpose: `J^T(data_metric(r, ω, γ))` is the weighted gradient. The transposes'
    argument is still named `residual` until M4, so phrase it by role.
- **`jet_to_dual_jet` / `dual_jet_to_jet` in the backend**, per 9.2 and r4 minor point 2:
  - the type is `Y* -> Y` in turned coordinates (`flip ∘ M` at `γ = 1`), with the `weight=None` reading
    `Y -> Y` turned;
  - "for `sampling_variables`' `w_tilde` and held tilde variables; never for a transpose (D3)";
  - and the division by zero documented as the only behaviour, now that O-20 is gone.

### 4. SHOULD FIX: give the input law a negative control too

All three negative controls in the note are for the jet law. The input law is the one that 9.3 says "catches treating
`c` as `V_i*`" and "wrong co- or contravariance". Show it does. For example, a variant of `data_metric` checked with
`move_out = V_star` instead of `V` must fail; or a fake probe transpose that treats `c` as `V_i*` (`Δ^-1`) must fail.
One or two lines, like the jet-law controls.

### 5. MINOR

1. **`γ` is a scalar.** Raise if `np.ndim(minibatch_scale) != 0`. Otherwise a per-`C` or per-sample array would
   broadcast as a hidden weight. Structural, and static under `jit`.
2. **Two more cheap law tests:**
   - `blocks_to_weight`: blocks moved by `P*` (with `lead=1` for probe) give an output moved by `P*`;
   - the batch method: `batch.data_metric` under the jet law, with the batch's weight moved by its own law.
3. **The grammar: where the role name goes.** Many returns name the role before the shape (`# z. len=d, …`, `# J
   x_dot. shape=…`). The rule should say where a role name goes relative to the type, for example
   `# z: P⊗V_i*, len=d, elm_shape=…`, and how a `HOST bool, static` tag sits beside a type. Otherwise M5 will
   improvise it on 100 signatures. One sentence and one example in `signature_style.md`.
4. **The bitwise claim is for `γ = None`.** With a scale, `(γ ω²) r` and today's oracle (`γ` applied after the sums)
   agree only to rounding. Say so in the test's docstring, so that M2 and M3 compare to rounding where `γ ≠ 1`.
5. **A packed `r` with garbage in its padding** comes out with `ω² ×` garbage. That is correct, since the masks are
   the engine's (M2). Say so in one line of the docstring, so that no one masks twice or expects zeros.

## The five decisions

1. **`_with_K` in M2: agree.** It is engine-private with no caller in M1, and M2's reference tests over `K` are its
   real oracle. Built now, it would be dead code tested only against itself. Record it as a ruling, since it moves an
   item between plan rows.
2. **Edge case (a) raises, naming `pack_batch`: agree.** Reading a ragged weight mode by mode against a packed `r`
   needs `N` and the masks, which belong to the packing. Raising keeps `data_metric` free of layout.
3. **The batch method raises by rank on block-weight batches until M2: agree.** I checked the cases. No block form
   can be silently misread:
   - an apply `F` against `F+W+C` differs in rank unless `W = C = ()`;
   - a probe `(d,)+F` row against `F+W+C+(Ni,)` always differs, because the edge is always there;
   - entries `()` against `W+C` coincides only at `W = C = ()`, where the two forms mean the same thing.

   Converting blocks inside the method would bring back the reading rule the maintainer removed. Add a test that a
   block-weighted batch raises, so the M1 behaviour is pinned until M2 removes block weights.
4. **The frontend converters removed from `fitting`'s namespace entirely: agree.** "Frontend-private" should mean
   not reachable. None of it is in `v2026.5.0` (the frontend `fitting.py` there has no `dual_jet`; `jets.py` did not
   exist), so it needs no CHANGELOG break note. No page or example uses them. The only uses are `test_jets` (20) and
   `test_jax_absent` (3), which move to the backend names.
5. **M0's `-v` log as M1's reference: agree, verified.** I unpacked `git archive 64465d8b t3toolbox tests` and
   compared it with my snapshot of the tree that log ran on (`diff -rq`). They are identical, apart from an untracked,
   ignored `t3toolbox/backend/.claude/settings.local.json`, which exists in the checkout and is not package code. The
   post-M0 commit's other changes are docs and the gzipped baseline. One caution for the tally: `test_jets`,
   `test_dispatch` and `test_jax_absent` nodes whose frontend calls move must keep their subtest counts exactly, or
   drop exactly the frontend ones. The note says so; hold the tally to it.

## What I will check on the build

- `data_metric` against my own explicit loop, including:
  - findings 1 and 2;
  - packed with a size-1 mode axis (the documented limit);
  - the float32 `r` with float64 `ω` and float64 0-d `γ`, under jit.
- The bitwise bridge to `_squared_weights` on all three kinds, ragged and packed, `C` empty and not.
- The laws by my own small script; mutants of `data_metric` (no square, `γ` dropped, `ω` applied as `1/ω`) and of
  the helper (a law that is the identity).
- The removals reach nothing else.
- The grammar text, and the node-by-node tally against the M0 log.
