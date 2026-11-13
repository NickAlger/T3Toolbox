# Critic review r2 — `dev/newton_timing_plan_2026-11-10.md` (draft 2)

Read in full against r1. Every MINOR finding is resolved:

| r1 | resolution in draft 2 | status |
|---|---|---|
| M1 §2.7 vs the switch | §2.7 rewritten against switch draft 2 (five `_SWITCH_FIELDS`, excluded from `_NEWTON_SCALAR_FIELDS`, switch-mode rows only). `PhaseSeconds.tracking` added (`None` until the switch lands; the switch adds the lap, as the switch plan's W1 now says). `track` cell. §4.13 kept and `### Added` created; the switch takes §4.14 | resolved |
| M2 W-3 probe | §2.6 / W-3: `timing=True` only when the signature names `timing`; T7 adds a `**kwargs`-forwarding member that must run equal to `timing=False` | resolved |
| M3 W-6 guards | `ValueError` on `clock` with `use_jit` (T7 asserts it); the partial-binding note names the clock slot; start frame synchronized before pass 1 on both layers, each pass [synced start, synced end] (T7 mocks the sync order) | resolved |
| M4 `local_model` override | §2.5: stock split only when `type(problem).local_model is Problem.local_model`; an override is called, one lap in `frame`, the other split phases `None`; T3 (on == off, override honoured) and T4 | resolved |

All four NITs are applied as proposed:
- the async caveat in §2.4;
- T2's identities only, with the 10 % bound in the slow tier;
- the host-side accumulator note;
- `phases` excludes the one-offs and the pass tuples.

The `total` identity is now exact by tiling (both ends read at the same instant), which is better than r1 asked.

There is no new MAJOR or MINOR finding. The added text is consistent with the switch plan's draft 2:
- `tracking` runs after the trials, matching the switch's countable-only product after step control;
- `curvature` is the per-point tracking build, per Q-1(a);
- the firing compile is in the first exact row's `compile_seconds.cg`.

## NIT (non-blocking)

1. **Field order.** §2.1's table says `tracking` "runs after the trials" and the display line puts `track` after
   `trials`, but `PhaseSeconds` declares it between `forms` and `trial_retraction`. Declaring it after
   `trial_objective` makes the field order the execution order, as the other phases are. This is cosmetic, since the
   records are keyed by name, and needs no re-review.

VERDICT: APPROVE
