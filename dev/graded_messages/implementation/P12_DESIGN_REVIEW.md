# P12 design review: the implementer's note on the backend sampling functions on the engine

Reviewer, 2026-11-20, on `P12_DESIGN_NOTE.md`, read only. I read it against the plan's P12, sections 2.4, 3.3 and 6.5,
today's `backend/fitting.py` and `backend/uniform_fitting.py`, and the uncommitted tidy of `PART1_PLAN.md`.

**Overall:** the design is right. Each function is one engine call, with no checks beyond the engine's, and the twins
pass `.data` through and mirror the sample's form.

There is one blocker, and it is in the plan tidy, not the note. It deletes sections 11.3 and 11.4.

## 0. The plan tidy deletes sections 11.3 and 11.4 (must fix before P12 starts)

The uncommitted diff of `PART1_PLAN.md` (35+, 77−) removes two whole sections:
- **11.3**, "Names added after the table was settled": `over_orders`, `dense_apply`/`dense_entries`/`dense_probe`, the
  checkers, `ut3_unscale_coordinates`;
- **11.4**, "Names of the generalized jet group": the rows **N31 to N42**.

Section 11.5 now follows 11.2 directly. Everything else in the tidy is right:
- O-48 to O-50 move to 11.1 as answered;
- 11.2 reads "None";
- 5.2 is reset;
- `_tangent` is written throughout, with the `naming_conventions.md` note for P19;
- the round-2 amendment on the shared tie (the mean, `k` times) has landed in P17.

But 5.2 and 11.2 still cite "sections 3, 11.3, 11.4, 11.5", and N41 (`len_J`) and N42 (`error_table`) are in the code.

*Restore both sections verbatim from `HEAD`.* I checked the name rows of `HEAD` against the tree, N1 to N47: only
N31 to N42 are missing.

## 1. A residual with a stack `K`: agree with (a), with three conditions

**Why not (b) or (c).**
- (c) drops a capability the plan keeps; `T3Tangent`'s static transposes rely on it.
- (b) is correct but runs every plain sweep `K` times, in memory and in time, for the sweeps that do not depend on the
  residual. That is the wrong cost for a stack of tangent vectors at one frame.

**Why (a) is the simplest *correct* form.** The alternative inside the engine would be to let the residual adjoint's
tilde rows take the residual's `K`. That breaks the per-node `K` the plan declares (P5's amendment, 2.3): in the
corewise Hessian the tilde variables have `K = 0` while the directions carry `K`, so a tilde node's `K` cannot follow
the residual.

The `Jp`-seeded twin path *is* "the residual adjoint with `K`". So reaching it from a held `Jp` reuses the one path
that already has the right per-node `K`. It is linear in what is held, so `JtJp` with `Jp` held equals `J^T` applied to
any vector of `Jp`'s type, turned, with no weight when none is given.

**Decided rows.** It changes no decided row:
- 3.3's twin variables stay unnamed, and only `JtJp`, already public, is read;
- `Jp` is already a slot of `HeldVariables`.

**Conditions:**
1. **Say it in 6.5 and in `HeldVariables`' docstring.** A held variable is described as "computed earlier at this
   frame and sample". A held `Jp` used as an arbitrary vector of its type is a new, legitimate use: `JtJp` is then
   `J^T` of that vector. Write it as a one-line amendment to the plan's P12, so the twin path's linearity becomes a
   stated contract, not an accident.
2. **Keep the routing honest.**
   - The `'held:Jp'` fact is added to `present` only for a held `Jp`.
   - P8's error ("`JtJp` needs `variations`") must still be what a call with neither `variations` nor a held `Jp`
     gets.
   - `_read_K` from a held `Jp`: the direction wins when both are given, and the two must agree (a structural error
     otherwise).
   - Check that the corewise graph has the `JtJp` twin path (the note's corewise transpose with `K` depends on it).
3. **Tests at the engine level, besides the transposes':**
   - `JtJp` with `Jp` held at `K = ()` equals `g` with `r` held (two routes, one number, to rounding);
   - at `K`, it equals the per-element `g`;
   - the rows run are the plain sweeps once plus the twin path. The structural step count, as for the tangent.

## 2. The bridge's order-0 rows: agree, with one guard

Rewiring the bridge to code that stays until P20 is necessary; otherwise it compares the engine with itself. Not all
of the old kinds stays independent, though:
- **What stays independent:** the old plain kinds' `forward`, `transpose` and `corewise_hessian`. They call
  `*_jacobian_from_sweep`, `*_transpose_from_sweep` and `*_corewise_hessian_from_sweep` (`fitting.py` 532 to 612).
- **What becomes the engine at P12:** their `point_forward` (`fitting.py` 538, 565, 604; `uniform_fitting.py` 143,
  172, 207), which calls `t3_apply`, `t3_entries` and `t3_probe`.

So the bridge's `z` rows must not go through a plain kind's `point_forward`. Use the derivative kinds at order 0
(`pd.t3_*_derivatives`), or `dense_*`.

*Guard:* while the bridge computes its old side, patch `sampling_variables.sampling_variables` to raise. Any path that
silently reaches the engine then fails loudly, which settles the question once for every row instead of by reading.

## 3. The `utv_*` twins with the frame's masks: agree

The variations' masks are derived from the frame's (`ufv_variation_masks`), and P11's engine already masks `V` and `H`
with them, broadcast over `K`. Variations from a different frame are a same-frame matter for the frontend, not this
function. The docstring sentence is enough.

Two smaller points on the twins:
- The residual is packed by `pack_if_ragged(residual, N)` (full jet sizes, so no `len_J`: right).
- The held residual (`r` or `Jp`) is masked on entry by the engine (P11, Finding B), so a hand-packed residual with
  garbage padding is safe. Include it in the twins' garbage tests, as the note does.

## 4. The split: agree, with the engine change in P12a

- **P12a:** the ragged functions, their call sites and tests, **and the engine addition of 1(a) with its own tests**.
  The ragged transposes need it.
- **P12b:** the uniform twins and their call sites (`uniform_fitting`, the frontend's uniform methods), the tangent,
  the Hessian's flag, and the AD tests.

Each part must leave the suite green. Changing a uniform twin's signature in P12b together with its callers keeps that
true.

## 5. The razor: two things to check, so no backend user is worse off

1. **The checks today's functions make.** "Nothing beyond the engine" is right only if nothing today's functions check
   is lost. List what each of the 30 checks today (an `N_i` of `ww[i]` against the point's shape, a residual's shape)
   and confirm each survives in the engine or in `check_sample`.

   The case to settle is a ragged `ww[i]` whose mode axis has size 1 against `N_i > 1`. numpy's `einsum` broadcasts a
   size-1 labelled axis. P11 checks `N` for the packed sample, but I see no ragged `N` check in the engine. If today's
   `t3_probe` raises there (by `einsum` shape or by a check) and the engine does not, that is a new silent wrong answer.
   One test closes it.
2. **Reusing sweeps.** Today a backend user who calls many transposes at one frame (inside CG) can reuse the sweep
   through `*_from_sweep`, which P20 deletes. The new one-call functions recompute the plain sweeps each time. Their
   reuse is the engine with `held=`.

   That is the plan's design, but by the razor's corollary the docstrings of the transposes and of the tangent should
   name it ("for repeated calls at one frame, call `sampling_variables` with `held=`"), so the capability is not lost
   in plain sight. No new argument.

## 6. The tests: sufficient with 1's engine tests and 5's size-1 case

What the note lists is right:
- by definition, at `()`, `(2,)` and `(1, 1)`;
- the two adjoint identities per element of `K`;
- the sums;
- the Hessian against `jax` and its symmetry;
- the tangent's dense oracle first;
- the twins garbage-padded, with masks from the ragged ranks and the mirror;
- dispatch rows.

## Summary

| # | Request |
| --- | --- |
| 0 | Restore plan sections 11.3 and 11.4 (blocker) |
| 1 | (a) agreed: the held-`Jp` contract stated, the routing honest, the engine-level tests |
| 2 | The bridge off `point_forward`, with the "engine raises" guard |
| 3 | Agreed |
| 4 | Split agreed, the engine change in P12a |
| 5 | List and keep today's checks (the ragged size-1 `N`); docstrings name `held=` for reuse |
