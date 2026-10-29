# `dev/curvature_impl_review/` — the review record behind `../riemannian_newton_curvature_impl_plan.md`

The implementation plan for the exact Riemannian Hessian (`hessian='exact'`) was produced by a writer/critic loop
of two Opus 5.5 agents over six rounds (2026-10-25/26), relayed by Claude and adjudicated by Nick twice (after v4
and after v6). This folder is the record: it exists so that every `[V]` and `[E]` claim in the plan can be re-run,
and so that the reasoning behind each settled decision is not lost. It is ephemeral in the `dev/` sense — sweep it
into `dev/archive/` together with the implementation plan, which stays live after S13 (it sequences R / S14 / S15b)
and is archived at the release (S13 archived only the design plan, the G0 plan and `curvature_oracles/`).

| file(s) | what |
|---|---|
| `plan_v1.md` … `plan_v5.md` | the earlier plan versions; each from v2 on carries a CHANGES table for its round. `plan_v6.md` is the adjudicated `../riemannian_newton_curvature_impl_plan.md` |
| `critique_v1.md` … `critique_v6.md` | the critic's reviews (BLOCKER / MAJOR / MINOR / NIT findings with citations and fixes); v4 and v6 are the two AGREE verdicts |
| `critic_checklist.md` | the convention checklist the critic held the plan to, one line per rule with its doc/section |
| `critic_prechecks_v5.md` | the critic's pre-checks for the round-5 restructure (the zero-reference findings P1a–P1c, P3, P5) |
| `checks/c*.py` | the writer's read-only verification scripts: the `[V]` facts F1–F38 in the plan (`c22` is the probe_derivatives two-hole recipe, `c35d` the GN-form cost re-basing, `c41` the zero-frame recipe) |
| `critic_checks/check_*.py` | the critic's experiments: the `[E]` facts E1–E12 (`check_restart_tr*.py` is E2, the zero-padded-restart test that fixed the damping default) |

## Running the scripts

They are plain numpy scripts (none needs jax) that import the library from the working directory,
so run them **from the repo root**:

```
cd <repo root>
python -I -B dev/curvature_impl_review/checks/c41_zero_frame_recipe.py
python -I -B dev/curvature_impl_review/critic_checks/check_d1_normal.py
```

(`-I` isolates the interpreter — each script puts the cwd on `sys.path` itself; `-B` keeps `__pycache__` out of
the tree. Scripts that need the repo's tests or oracle sweeps reference `tests/` relative to the root.) Several
cross-import each other by file name within their folder; the critic's scripts reach the writer's through
`../checks`. Expected output and tolerances are quoted next to each fact in the plan's "Verified facts" table.

They were run against `main` @ `920d776`. As slices land the scripts are kept runnable: S2 moved the private
`tv._project_t3_*` kernels to `t3_linalg._inner_product_*` and the scripts were updated to the new names. From S3b on
they are kept as the plan-time record instead: a script on a removed API (the dense `trs` jet kernels, S3b;
`stats['newton']`, S0d) or whose warm-up now resolves a different default reference (S0d) carries a one-line
banner at its top saying so (S13), and is not rewritten.
