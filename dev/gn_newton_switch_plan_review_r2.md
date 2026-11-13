# Critic review r2 — `dev/gn_newton_switch_plan_2026-11-13.md` (draft 2)

Read in full against r1. Every MINOR is resolved:

| r1 | resolution in draft 2 | status |
|---|---|---|
| M1 product on countable rows | §1.1 step 7, §1.2 table ("reset, with no product"), §1.5, §3.1 field comments, T4 (`switch_pKp is switch_e_n is None` on each reset row), T5, display, G-7 | resolved |
| M2 timing coordination | §3.3: `curvature` = the per-point build via timing's Q-1(a); one request, `PhaseSeconds.tracking`; G-5 removed | resolved (raised on the timing plan in its r1, M1) |
| M3 docs numbering | §7: §4.14 after timing's §4.13, nothing renumbered; CHANGELOG appends to timing's `### Added` | resolved |
| M4 G-4's basis | §1.5 two currencies; the jitted-product middle option (T2 stays bitwise: exact rows keep the eager `pKp`); G-4 rule over both currencies and both starts; ship question on W3's agenda | resolved |
| M5 padded-start stress | §6.1 starred arms with `START=aligned` in W3; W5 repeats at 20×1000 | resolved |
| M6 frontend order | §2.1 / §2.3: `_check_switch_options` before `least_squares_problem`; T7 spy | resolved |
| M7 `info.lm` | §1.1 step 1, §3.1: the GN view, same object across a rejection; T8 | resolved |
| M8 edges | §3.2 `switched_at` = first exact row's index, `len(history)` if fired on the last row; §3.1 converged and exact-phase fields; T8 | resolved |
| M9 compile gate | T9: `switch_tau=0` after pure GN-TR grows `_cache_size()` by 0; firing ≤ 1 more; second firing 0 | resolved |

NITs:
- **N1 (not adopted).** Accepted. Answer 4's wording is binding, and §1.3 states that the counting equals the
  source's gate.
- **N9 (a new subtest instead of editing the existing one).** Accepted, for the stated reason: the existing test's
  `hessian='exact'` options would now raise early with switch keywords set.
- **The other ten** are taken as proposed.

No new MAJOR or MINOR. The rules added since r1 are consistent:
- `switch_k` with `switch_tau=0` raises, and so does `switch_tau` with `switch_k=0`. Together they reject `(0, 0)`
  either way.
- T1 sets only `switch_tau=0` and T2 only `switch_k=0` (plus the forcing power, which `switch_k=0` reads), so both stay
  legal.
- The converged row's `switch_count` never reaches the display, because the converged line returns before the
  segments.

## NIT

1. **W1, §3.3: who adds `PhaseSeconds.tracking`.** In the timing review I proposed that the timing plan adds the field
   and this plan only laps it. W1 lists "`PhaseSeconds.tracking` and the tracking build lapped as `curvature`".
   Reword W1 to "lap `'tracking'` (the field ships with timing, its §2.7)". If timing's draft 2 does not add the
   field, W1 adds it. This is a one-line edit with no re-review needed.

VERDICT: APPROVE
