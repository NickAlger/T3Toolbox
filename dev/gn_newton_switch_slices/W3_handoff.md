# W3 handoff: the Darcy benchmark of `newton_cg(hessian='switch')`, laptop and A100 (2026.5.0)

The session's record (plan §8: W3 is run by the session in T3Polynomial). Base: T3Toolbox `c97c754f` (W1 `ca13b37c`,
W2 `c39b30bc`; the plan `dev/gn_newton_switch_plan_2026-11-13.md` at draft 5). **A report, not a stop**: G-4 and G-7
are resolved (§9), so no decision hangs on these numbers. Scope: §6.1's table at both starts, and the GPU runs that
went with it: all five arms at 10 × 100 on an A100, W5's 20 × 1000 repeat as far as it ran, and the GPU figures for
the timing feature (§4.13). The maintainer's reading rule: economics only off full ladders, never off early levels;
laptop wall-clock is information, GPU wall-clock is the real economics.

**Changed in T3Toolbox:** nothing. **New:** this file. The data are committed in T3Polynomial (`90dbd17`, `dbda0de`;
§7). §6 drafts the docs text as text only: `docs/fitting_and_optimization.md` is held by W4's verification.

## 1. What was run

| run | machine | Darcy | starts × arms | status |
|---|---|---|---|---|
| laptop | the maintainer's laptop, CPU | 10 × 100 | library × {switch, exact}: the decisive pair (the maintainer trimmed the other arms to the GPU) | complete, 2026-11-14 12:20–20:43 local; 16,962 s (exact) + 13,233 s (switch) |
| GPU 10 × 100 | A100 | 10 × 100 | {library, aligned} × all five arms | complete: 10 pairs, 6,507 s, every exit 0 |
| GPU 20 × 1000, part a | A100 | 20 × 1000 | {library, aligned} × {switch, exact} | complete: 4 pairs, 19,054 s, every exit 0 |
| GPU 20 × 1000, part b | A100 | 20 × 1000 | {library, aligned} × {gn_tr, tracking} | **partial**: `library_gn_tr` levels 0–4 of 11 (`complete: false`; level 5 was running at the last collection, 2026-11-15 02:47 UTC); `library_tracking`, `aligned_gn_tr`, `aligned_tracking` never started. The pod was handed back (~02:50 UTC); they may never finish |
| GPU timing | A100 | §4.13's cases A and B, and one Darcy 10 × 100 level | `timing` off against on | complete: 13 processes, 28 min |

**Arms** (`newton_cg` options on every level; every arm adds `chunk_size=None, timing=True`):

| arm | options |
|---|---|
| `gn` | `hessian='gauss_newton'`, `cg_forcing_power=0.5`, `step_control='line_search'` (today's default) |
| `gn_tr` | `'gauss_newton'`, 0.5, `'trust_region'` (separates the step control from the switch) |
| `exact` | `'exact'`, 1.0, `'trust_region'` |
| `switch` | `'switch'`, 0.5, `switch_cg_forcing_power=1.0`, `'trust_region'`; τ = 0.25, k = 2, `switch_track='countable'` (the defaults) |
| `tracking` | `'switch'`, 0.5, `switch_tau=0.0`, `'trust_region'`: GN-TR's trajectory plus the tracking (T1) |

**Common settings.** T3Polynomial `scripts/w03_switch_benchmark.py` (how to run: `scripts/w03_README.md`): each
(start, arm) pair in its own process with fresh jax caches, one job at a time; every arm on d10's data split, seed and
adaptive ladder, growing ranks from its own iterate. `BACKEND=uniform_jax` (`use_jit`: CG is one compiled call; the
build, the tracking product and the trials run eagerly in the host loop), x64, `MAX_NEWTON=300` per level, the stop
`‖g‖ ≤ 10⁻³·‖g0‖` with `g0_reference='zero'` on every arm (equal tolerance), the trust region's defaults otherwise.
**Starts:** `library` is the ladder's default (`padded_start=False`); `aligned` passes the previous level's point as
`padded_from` at every level after the first (§6.1's stress case, answer 9).

| | 10 × 100 (d10's o01 configuration) | 20 × 1000 |
|---|---|---|
| data | `data/darcy_o01`: 10 points × 100 directions, n = 22 reduced inputs, jet order J = 4, 800 / 200 train / val rows, κ = 3.605 | `data/darcy_s20d1000`: 20 × 1000, n = 22, J = 4, 16,000 / 4,000 rows, κ = 3.800 |
| ladder | N_CHUNK 5, MAX_RANK 24, MAX_LEVELS 10: 6 levels, Tucker / inner TT 1, 6, 11, 16, 21, then 23–24; dof 112 → 38,040 | N_CHUNK 5, MAX_RANK 50, MAX_LEVELS 12 (the 2026-11-10 A100 runs' ladder): 11 levels, Tucker to 23–24, inner TT to 50; dof 112 → 106,550 |

Both: tolerance 0.05, basis `as`, degree 4, seed 0, val fraction 0.2, centred, ladder τ 10 / τ_data 2, selection
`forward_summed_reverse` on val_fsr; reduction floor p50 0.0374. Every arm reached the same ranks at every level.

| | laptop | GPU |
|---|---|---|
| hardware | AMD Ryzen 7 3700U (4 cores / 8 threads, 13 GB), CPU only | Runpod pod `x4lh6oimycsqcp`: NVIDIA A100-SXM4-80GB (driver 595.91.07, CUDA 13.2), 27.2-core cgroup quota, Ubuntu 24.04.5 |
| software | Python 3.11.15, numpy 2.4.6, jax 0.10.2 (`cpu:0`) | Python 3.11.18, numpy 2.4.6, jax 0.10.2 with `jax-cuda12-plugin` 0.10.2 (`cuda:0`); `gpu_env.txt` has the `pip freeze` |
| T3Toolbox | `exact`: `c97c754f`. `switch`: recorded `6b01889c+dirty`, the same code: `6b01889c` changes only `dev/HANDOFF.md`, and the dirt was W4's uncommitted edits, which in `t3toolbox/` are two docstring sentences | `c97c754f`, a fresh GitHub clone (editable install) |
| T3Polynomial | `e1078cb` plus the then-untracked `scripts/w03_*` (committed afterwards in `90dbd17`) | the same, rsynced; the tree clean, so the version line reads `e1078cb` |

Both T3Toolbox commits report `2026.4.0` (the version moves at the release commit).

**Where the settings depart from §6.1's text** (README, "Where the settings depart"):
- `MAX_NEWTON=300`, not 60: the exact Hessian took 66 and 71 steps at levels 4–5 on the 2026-11-10 A100 runs, and 300
  (those runs' cap) never bound there. The Gauss-Newton arms still hit it (CAP below).
- The knobs live in a separate script, not in d10 (`HESSIAN`, `CG_POWER_AFTER`); d10 is untouched.
- The laptop ran only the decisive pair; every arm at both starts ran on the GPU instead, so §6.1's full table is the
  GPU's, with the laptop pair beside it (§5.4).
- W5's 20 × 1000 repeat ran without waiting for §6.2's verdict, on the same pod, with W3's zero reference (not the
  2026-11-10 runs' `'cone'`) so that both scales share one reference and tolerance; its arm list has no `gn`.

## 2. The §6.1 tables

Cells: `Newton / CG (stop) · apply-equivalents · net wall s`. **Newton** = `stats['newton_steps']` (rejected
trust-region trials included); **CG** on the switch arm is `GN+exact`; **stop** `gtol` or `CAP` (the 300 cap);
**sw@j** = `switched_at`, the first exact row. **AE** = `CG_GN + m_N·CG_exact + m_t·tracking products` (§6.2), in
Gauss-Newton applies, with `m_N`, `m_t` measured per rank structure from the same directory's timed rows (§4.1): AE
compares arms within one table, not across machines. **Net** = `stats['timing']['total']` less compile less the
display's callback. Copied from the summaries (§7), nothing recomputed: the laptop cells combine its summary's two
per-level tables, and the GPU switch cells take the GN+exact split from the per-directory summaries. The per-level
error tables and the switch rows are in the summaries.

### 2.1 Laptop, 10 × 100, START=library (the decisive pair)

| level | tucker / inner TT | **switch** | **exact** |
|---|---|---|---|
| 0 | 1 / 1 | 28 / 7+122 (gtol) sw@5 · 260 · 7 | 24 / 98 (gtol) · 164 · 5 |
| 1 | 6 / 6 | 20 / 31+129 (gtol) sw@10 · 401 · 75 | 43 / 291 (gtol) · 783 · 150 |
| 2 | 11 / 11 | 43 / 28+798 (gtol) sw@7 · 2219 · 613 | 47 / 680 (gtol) · 1854 · 513 |
| 3 | 16 / 16 | 29 / 42+1163 (gtol) sw@7 · 3321 · 1204 | 38 / 961 (gtol) · 2698 · 991 |
| 4 | 21 / 21 | 52 / 45+3226 (gtol) sw@6 · 9863 · 6470 | 65 / 3468 (gtol) · 10543 · 7324 |
| 5 | 23,23,23,23,24 / 23,24,24,24 | 47 / 67+2321 (gtol) sw@6 · 6646 · 4698 | 60 / 3641 (gtol) · 10302 · 7794 |

The totals keep the summary's column order (exact, then switch):

|  | **exact** | **switch** |
|---|---|---|
| Newton steps (accepted) / rows | 277 (234) / 283 | 219 (183) / 225 |
| CG iterations: GN / exact | 0 / 9139 | 220 / 7759 |
| tracking products / GN-phase builds | — | 41 / 41 |
| levels at CAP | 0 | 0 |
| levels switched | — | 6 of 6 |
| wall s: total / compile / callback / **net** | 16917 / 126 / 14 / **16777** | 13190 / 111 / 11 / **13068** |
| curvature + tracking, net (% of the GN phase) | — | 31.1 s (18.5 %) |
| **apply-equivalents** (§6.2) | **26344** | **22711** |
| tracking overhead, apply-equiv. (§1.5) | — | 170 |
| selected level (val_fsr / test p50) | 4 (0.3740 / 0.0674) | 5 (0.3750 / 0.0614) |
| last level (val_fsr / test p50) | 5 (0.3765 / 0.0623) | 5 (0.3750 / 0.0614) |
| ladder wall s | 16958 | 13228 |

### 2.2 GPU, 10 × 100, START=library

| level | tucker / inner TT | **switch** | **exact** | **gn_tr** | **gn** | **tracking** |
|---|---|---|---|---|---|---|
| 0 | 1 / 1 | 28 / 7+122 (gtol) sw@5 · 456 · 8 | 24 / 98 (gtol) · 232 · 7 | 300 / 582 (CAP) · 582 · 44 | 282 / 622 (gtol) · 622 · 43 | 300 / 582 (CAP) · 10181 · 80 |
| 1 | 6 / 6 | 30 / 24+166 (gtol) sw@8 · 633 · 11 | 26 / 145 (gtol) · 396 · 9 | 173 / 749 (gtol) · 749 · 30 | 143 / 668 (gtol) · 668 · 26 | 139 / 611 (gtol) · 3300 · 41 |
| 2 | 11 / 11 | 41 / 33+609 (gtol) sw@8 · 1963 · 20 | 48 / 830 (gtol) · 2421 · 24 | 245 / 2799 (gtol) · 2799 · 51 | 300 / 3236 (CAP) · 3236 · 65 | 300 / 3017 (CAP) · 8764 · 98 |
| 3 | 16 / 16 | 34 / 43+825 (gtol) sw@7 · 2647 · 21 | 41 / 1626 (gtol) · 4891 · 33 | 300 / 5262 (CAP) · 5262 · 77 | 300 / 5368 (CAP) · 5368 · 76 | 300 / 5550 (CAP) · 10785 · 112 |
| 4 | 21 / 21 | 44 / 47+2286 (gtol) sw@6 · 7439 · 59 | 45 / 2260 (gtol) · 7232 · 59 | 300 / 7525 (CAP) · 7525 · 106 | 300 / 6712 (CAP) · 6712 · 103 | 171 / 4315 (gtol) · 6510 · 80 |
| 5 | 23,23,23,23,24 / 23,24,24,24 | 46 / 52+2089 (gtol) sw@5 · 6656 · 57 | 83 / 5792 (gtol) · 18136 · 143 | 300 / 10106 (CAP) · 10106 · 128 | 300 / 8076 (CAP) · 8076 · 115 | 300 / 10446 (CAP) · 14228 · 163 |

| total | **switch** | **exact** | **gn_tr** | **gn** | **tracking** |
|---|---|---|---|---|---|
| Newton steps (accepted) | 223 (191) | 267 (227) | 1618 (1618) | 1625 (1625) | 1510 (1510) |
| CG iterations GN + exact | 206 + 6097 | 0 + 10751 | 27023 + 0 | 24682 + 0 | 24521 + 0 |
| tracking products | 39 | 0 | 0 | 0 | 1510 |
| levels at CAP | 0 | 0 | 4 | 4 | 4 |
| levels switched | 6 of 6 | — | — | — | 0 of 6 |
| **apply-equivalents** | **19794** | **33309** | **27023** | **24682** | **53769** |
| wall s: total / compile / net | 340 / 151 / **177** | 431 / 144 / **275** | 559 / 94 / **436** | 550 / 94 / **427** | 742 / 140 / **575** |
| ladder wall s (job) | 386 | 478 | 604 | 596 | 790 |
| selected level: val_fsr / test p50 | 4: 0.3697 / 0.0675 | 4: 0.3722 / 0.0682 | 4: 0.3720 / 0.0664 | 4: 0.3736 / 0.0689 | 4: 0.3744 / 0.0660 |
| best test p50 (level) | 0.0632 (5) | 0.0631 (5) | 0.0601 (5) | 0.0614 (5) | 0.0630 (5) |

### 2.3 GPU, 10 × 100, START=aligned

| level | tucker / inner TT | **switch** | **exact** | **gn_tr** | **gn** | **tracking** |
|---|---|---|---|---|---|---|
| 0 | 1 / 1 | 28 / 7+122 (gtol) sw@5 · 456 · 8 | 24 / 98 (gtol) · 232 · 7 | 300 / 582 (CAP) · 582 · 43 | 282 / 622 (gtol) · 622 · 40 | 300 / 582 (CAP) · 10181 · 77 |
| 1 | 6 / 6 | 44 / 25+277 (gtol) sw@9 · 957 · 16 | 26 / 142 (gtol) · 388 · 9 | 254 / 1037 (gtol) · 1037 · 42 | 210 / 893 (gtol) · 893 · 35 | 254 / 1037 (gtol) · 5951 · 71 |
| 2 | 11 / 11 | 42 / 30+685 (gtol) sw@8 · 2181 · 21 | 31 / 963 (gtol) · 2809 · 21 | 269 / 3064 (gtol) · 3064 · 58 | 260 / 3050 (gtol) · 3050 · 58 | 300 / 2489 (CAP) · 8236 · 95 |
| 3 | 16 / 16 | 47 / 32+1307 (gtol) sw@7 · 4086 · 32 | 53 / 2381 (gtol) · 7163 · 48 | 300 / 4179 (CAP) · 4179 · 73 | 300 / 4514 (CAP) · 4514 · 76 | 300 / 5049 (CAP) · 10284 · 107 |
| 4 | 21 / 21 | 85 / 30+4743 (gtol) sw@6 · 15284 · 120 | 51 / 2602 (gtol) · 8326 · 67 | 300 / 8534 (CAP) · 8534 · 113 | 300 / 6396 (CAP) · 6396 · 101 | 300 / 7637 (CAP) · 11488 · 143 |
| 5 | 23,23,23,23,24 / 23,24,24,24 | 57 / 49+3861 (gtol) sw@8 · 12240 · 97 | 46 / 3051 (gtol) · 9553 · 77 | 300 / 9960 (CAP) · 9960 · 125 | 300 / 8994 (CAP) · 8994 · 121 | 212 / 6992 (gtol) · 9664 · 116 |

| total | **switch** | **exact** | **gn_tr** | **gn** | **tracking** |
|---|---|---|---|---|---|
| Newton steps (accepted) | 303 (253) | 231 (204) | 1723 (1723) | 1652 (1652) | 1666 (1666) |
| CG iterations GN + exact | 173 + 10995 | 0 + 9237 | 27356 + 0 | 24469 + 0 | 23786 + 0 |
| tracking products | 43 | 0 | 0 | 0 | 1666 |
| levels at CAP | 0 | 0 | 4 | 3 | 4 |
| levels switched | 6 of 6 | — | — | — | 0 of 6 |
| **apply-equivalents** | **35203** | **28471** | **27356** | **24469** | **55806** |
| wall s: total / compile / net | 544 / 237 / **294** | 473 / 230 / **230** | 662 / 178 / **454** | 641 / 180 / **432** | 863 / 226 / **609** |
| ladder wall s (job) | 626 | 555 | 742 | 722 | 945 |
| selected level: val_fsr / test p50 | 4: 0.3722 / 0.0684 | 5: 0.3688 / 0.0620 | 4: 0.3743 / 0.0678 | 4: 0.3746 / 0.0671 | 5: 0.3748 / 0.0621 |
| best test p50 (level) | 0.0610 (5) | 0.0620 (5) | 0.0624 (5) | 0.0625 (5) | 0.0621 (5) |

### 2.4 GPU, 20 × 1000: switch and exact at both starts; GN-TR partial

START=library:

| level | tucker / inner TT | **switch** | **exact** |
|---|---|---|---|
| 0 | 1 / 1 | 18 / 9+58 (gtol) sw@7 · 247 · 6 | 15 / 71 (gtol) · 119 · 5 |
| 1 | 6 / 6 | 31 / 29+199 (gtol) sw@9 · 634 · 16 | 31 / 307 (gtol) · 842 · 20 |
| 2 | 11 / 11 | 36 / 20+524 (gtol) sw@6 · 1626 · 53 | 28 / 504 (gtol) · 1519 · 48 |
| 3 | 16 / 16 | 29 / 25+719 (gtol) sw@6 · 2401 · 98 | 30 / 730 (gtol) · 2389 · 98 |
| 4 | 21 / 21 | 24 / 25+850 (gtol) sw@5 · 2843 · 192 | 36 / 1147 (gtol) · 3779 · 258 |
| 5 | 23,23,23,23,24 / 23,26,26,24 | 24 / 37+889 (gtol) sw@6 · 2893 · 231 | 23 / 1201 (gtol) · 3832 · 303 |
| 6 | 23,23,23,23,24 / 23,31,31,24 | 23 / 34+1326 (gtol) sw@6 · 4268 · 440 | 21 / 665 (gtol) · 2114 · 225 |
| 7 | 23,23,23,23,24 / 23,36,36,24 | 29 / 40+1128 (gtol) sw@6 · 3671 · 454 | 21 / 1379 (gtol) · 4418 · 536 |
| 8 | 23,23,23,23,24 / 23,41,41,24 | 26 / 51+1459 (gtol) sw@6 · 4701 · 742 | 25 / 1231 (gtol) · 3910 · 619 |
| 9 | 23,23,23,23,24 / 23,46,46,24 | 27 / 40+1614 (gtol) sw@5 · 5247 · 884 | 33 / 1947 (gtol) · 6266 · 1065 |
| 10 | 23,23,23,23,24 / 23,50,50,24 | 20 / 68+1162 (gtol) sw@6 · 3778 · 736 | 28 / 1329 (gtol) · 4226 · 833 |

| total | **switch** | **exact** |
|---|---|---|
| Newton steps (accepted) | 287 (250) | 291 (248) |
| CG iterations GN + exact | 378 + 9928 | 0 + 10511 |
| tracking products | 68 | 0 |
| levels at CAP | 0 | 0 |
| levels switched | 11 of 11 | — |
| **apply-equivalents** | **32309** | **33413** |
| wall s: total / compile / net | 4167 / 294 / **3851** | 4311 / 278 / **4011** |
| ladder wall s (job) | 4367 | 4485 |
| selected level: val_fsr / test p50 | 7: 0.3842 / 0.0461 | 8: 0.3841 / 0.0452 |
| best test p50 (level) | 0.0449 (9) | 0.0450 (9) |

START=aligned:

| level | tucker / inner TT | **switch** | **exact** |
|---|---|---|---|
| 0 | 1 / 1 | 18 / 9+58 (gtol) sw@7 · 247 · 6 | 15 / 71 (gtol) · 119 · 5 |
| 1 | 6 / 6 | 21 / 31+134 (gtol) sw@9 · 458 · 12 | 20 / 129 (gtol) · 354 · 11 |
| 2 | 11 / 11 | 39 / 14+610 (gtol) sw@5 · 1875 · 61 | 42 / 765 (gtol) · 2306 · 73 |
| 3 | 16 / 16 | 28 / 24+715 (gtol) sw@6 · 2387 · 98 | 27 / 653 (gtol) · 2137 · 89 |
| 4 | 21 / 21 | 50 / 45+1288 (gtol) sw@8 · 4317 · 299 | 33 / 661 (gtol) · 2178 · 156 |
| 5 | 23,23,23,23,24 / 23,26,26,24 | 20 / 15+846 (gtol) sw@4 · 2727 · 219 | 34 / 1385 (gtol) · 4419 · 354 |
| 6 | 23,23,23,23,24 / 23,31,31,24 | 23 / 27+757 (gtol) sw@5 · 2449 · 260 | 19 / 926 (gtol) · 2944 · 309 |
| 7 | 23,23,23,23,24 / 23,36,36,24 | 25 / 17+1423 (gtol) sw@4 · 4587 · 562 | 26 / 1506 (gtol) · 4824 · 591 |
| 8 | 23,23,23,23,24 / 23,41,41,24 | 20 / 37+1043 (gtol) sw@5 · 3363 · 535 | 32 / 2198 (gtol) · 6981 · 1095 |
| 9 | 23,23,23,23,24 / 23,46,46,24 | 40 / 22+2820 (gtol) sw@4 · 9108 · 1543 | 22 / 1386 (gtol) · 4460 · 759 |
| 10 | 23,23,23,23,24 / 23,50,50,24 | 24 / 46+1400 (gtol) sw@5 · 4510 · 890 | 18 / 1326 (gtol) · 4216 · 822 |

| total | **switch** | **exact** |
|---|---|---|
| Newton steps (accepted) | 308 (269) | 288 (246) |
| CG iterations GN + exact | 287 + 11094 | 0 + 11006 |
| tracking products | 62 | 0 |
| levels at CAP | 0 | 0 |
| levels switched | 11 of 11 | — |
| **apply-equivalents** | **36028** | **34938** |
| wall s: total / compile / net | 4962 / 454 / **4486** | 4725 / 440 / **4262** |
| ladder wall s (job) | 5204 | 4966 |
| selected level: val_fsr / test p50 | 8: 0.3841 / 0.0452 | 8: 0.3842 / 0.0451 |
| best test p50 (level) | 0.0450 (9) | 0.0449 (9) |

**GN-TR, START=library, levels 0–4 of 11 only** (`w03_library_gn_tr.json`, `complete: false`). Recorded, not read for
economics (a partial ladder). Newton / CG (stop), with the switch and exact cells of the same levels:

| level | 0 | 1 | 2 | 3 | 4 | levels 0–4 |
|---|---|---|---|---|---|---|
| **gn_tr** | 221 / 404 (gtol) | 148 / 572 (gtol) | 300 / 1692 (CAP) | 240 / 2542 (gtol) | 248 / 4519 (gtol) | 1,157 / 9,729 |
| **switch** | 18 / 67 (gtol) | 31 / 228 (gtol) | 36 / 544 (gtol) | 29 / 744 (gtol) | 24 / 875 (gtol) | 138 / 2,458 |
| **exact** | 15 / 71 (gtol) | 31 / 307 (gtol) | 28 / 504 (gtol) | 30 / 730 (gtol) | 36 / 1147 (gtol) | 140 / 2,759 |

At level 4 the three agree: val_fsr 0.3912, test p50 0.0578 / 0.0578 / 0.0577 (gn_tr / switch / exact).

### 2.5 `switched_at` per level

| run | start | `switched_at`, level 0 on | levels switched |
|---|---|---|---|
| laptop 10 × 100 | library | 5, 10, 7, 7, 6, 6 | 6 of 6 |
| GPU 10 × 100 | library | 5, 8, 8, 7, 6, 5 | 6 of 6 |
| GPU 10 × 100 | aligned | 5, 9, 8, 7, 6, 8 | 6 of 6 |
| GPU 20 × 1000 | library | 7, 9, 6, 6, 5, 6, 6, 6, 6, 5, 6 | 11 of 11 |
| GPU 20 × 1000 | aligned | 7, 9, 5, 6, 8, 4, 5, 4, 5, 4, 5 | 11 of 11 |

On every level of every run the Gauss-Newton-phase rows, the tracked rows and `switched_at` are one number (the
summaries' "switch rows" tables): every Gauss-Newton-phase row was countable, and every rejected trial of a switch run
fell in its exact phase. The two hits that fired had `e_N / e_GN` between 0.00078 and 0.23 (τ = 0.25); the misses before
them, as far as the summaries show them (each level's last four tracked rows), reach 15 at 10 × 100 and 150 at
20 × 1000 (aligned level 10). No level is at CAP in any switch or exact run.

## 3. §6.2's bar

"Promising": the switch arm's AE below GN's (`gn`, today's default) at no loss in val / test error; a **lower bound** on
the switch's advantage (AE leaves out builds, trial retractions and objectives).

| config, start | switch AE | GN AE | margin | errors at the selected level, switch vs GN | verdict |
|---|---|---|---|---|---|
| GPU 10 × 100, library | 19,794 | 24,682 | −4,888, **−19.8 %** (4,888 / 24,682) | level 4 both: val_fsr 0.3697 vs 0.3736, test p50 0.0675 vs 0.0689: both better | **met** |
| GPU 10 × 100, aligned | 35,203 | 24,469 | +10,734, **+43.9 %** (10,734 / 24,469) | level 4 both: val_fsr 0.3722 vs 0.3746 (better), test p50 0.0684 vs 0.0671 (worse by 0.0013); best test p50 0.0610 vs 0.0625 (better) | **not met** |
| GPU 20 × 1000, both | 32,309 / 36,028 | — | — | no `gn` arm; GN-TR partial (§2.4) | open |
| laptop 10 × 100, library | 22,711 | — | — | no GN arm on the laptop | not judged |

- **The library margin** holds against GN-TR too (27,023) and against exact (33,309). At the last level (5) the
  switch's test p50 is 0.0632 against GN's 0.0614 (val_fsr 0.3740 against 0.3759, better); GN's level 5 stopped at the
  cap, not at the tolerance. The ladder delivers the selected level, where the switch is better on both errors.
- **The GN cap caveat.** GN stopped at the 300-step cap on 4 of 6 levels at the library start (levels 2–5) and 3 of 6
  at the aligned start (3–5), without reaching the tolerance the switch reached on every level; GN-TR is at CAP on 4 of
  6 at both. Their AE is therefore a lower bound on what Gauss-Newton would need. "Met" holds a fortiori; "not met" at
  the aligned start is measured against a censored GN and does not show that GN to tolerance would be cheaper. The
  uncensored comparison is switch against exact (§5.1).
- **Net wall, the GPU's real economics, sides with the switch at both starts:** 177 s against GN's 427 s at the
  library start (−58.5 %: 250 / 427) and 294 s against 432 s at the aligned start (−31.9 %: 138 / 432), where AE puts
  the switch above GN. AE leaves out the per-step costs, and GN took 1,625–1,652 steps against the switch's 223–303.

## 4. The tracking overhead (§1.5), as information

G-4 is decided (the product on every countable row; no thinning option, no jitted product, no further options now).
These numbers inform; they decide nothing.

### 4.1 Per product (the multipliers)

`c` = net seconds per operation (compile excluded), `m = c / c_GN`; ranges over the rank structures (rank 1, where a
Gauss-Newton apply is tiny, is set apart where it is the outlier), pooled in brackets. `c_t` is one tracking product,
`c_b` the switch's per-point build extra.

| machine, config | `c_GN` ms | `c_t` ms | `m_t` | `c_b` ms | `m_b` | `m_N` |
|---|---|---|---|---|---|---|
| laptop, 10 × 100 | 6.73 (rank 1) → 717.49 (23/24) | 65.61 → 1,433.28, rising with rank (703.41) | 9.76 at rank 1; 1.94–2.28 from rank 6 (1.48) | 41.54–72.53 (53.96) | 7.93 at rank 1; 0.10–0.27 (0.11) | 1.68 at rank 1; 2.69–3.04 (3.59) |
| GPU, 10 × 100 | 2.50 → 6.44 (5.56) | **78.91–81.13 at every structure** (79.77) | 12.61–32.00 (14.36) | 33.98–36.02 (34.84) | 5.60–13.58 (6.27) | 2.37–3.20 (3.23) |
| GPU, 20 × 1000 | 4.39 → 190.27 (105.33) | 88.35 (rank 1), 87.68 (rank 6) → 485.48 (inner TT 50) (242.40) | 20.11 at rank 1; 2.55–6.58 (2.30) | 47.82–103.84 (73.59) | 12.22 at rank 1; 0.52–3.59 (0.70) | 1.68 at rank 1; 2.74–3.29 (3.52) |

- **Laptop: arithmetic.** The product costs about 2 Gauss-Newton applies from rank 6 on, as §1.5 predicted (2–3), and
  grows with the rank.
- **GPU at 10 × 100: dispatch-bound and rank-independent, ~80 ms** from rank 1 to 23/24, while a jitted CG iteration
  costs 2.5–6.4 ms: hence `m_t` 12.6–32. The build's extra is flat at ~35 ms the same way. Under `use_jit` both run
  eagerly in the host loop (§1.5), and at this size their dispatch is the cost.
- **GPU at 20 × 1000:** the ~88 ms floor shows at ranks 1 and 6, then the arithmetic takes over (485 ms at inner TT
  50, `m_t` 2.55 there), close to the laptop's ~2.
- **A jitted tracking product would remove the dispatch cost.** Recorded as a future option only (G-4's deferred items
  in `docs/contributor/deferred_and_rejected.md`, W4), not a proposal.

### 4.2 The `tracking` arm (`switch_tau=0`) against GN-TR (GPU, 10 × 100)

| start | per Newton step (net wall / steps over the ladder) | ladder net wall | curvature + tracking inside `tracking` | in apply-equivalents |
|---|---|---|---|---|
| library | 0.381 vs 0.270 s: **+41.1 %** | 575 vs 436 s: +138 s (+31.7 %) | 174 s, 30.2 % of its net wall | 1,510 products + 1,512 builds = 41,969 = 155.3 % of GN-TR's 27,023 CG iterations |
| aligned | 0.366 vs 0.264 s: **+38.7 %** | 609 vs 454 s: +155 s (+34.1 %) | 188 s, 30.8 % | 1,666 + 1,668 = 45,959 = 168.0 % of 27,356 |

- The two arms follow the same trajectory only at level 0 (300 identical rows, both at CAP; §5.3). There the excess is
  +36.0 s on 44.2 s (+81.5 %, library) and +34.3 s on 42.5 s (+80.8 %, aligned), at rank 1, where a CG iteration
  costs 2.5 ms. From level 1 on the processes part (§5.3) and the per-level excess swings (−25.5 s at library level 4,
  where `tracking` reached the tolerance in 171 steps and GN-TR capped at 300), so per Newton step is the fair reading.
- Per step this is near the source's 30–40 % estimate (§1.5), but for the GPU's reason (eager dispatch at ~80 ms a
  product), not by the source's arithmetic. No laptop `tracking` arm ran (trimmed).

### 4.3 Inside the switch arm

| run | curvature + tracking, net | of the GN phase | of the net wall | tracking overhead in AE | of the arm's AE |
|---|---|---|---|---|---|
| laptop 10 × 100, library | 31.1 s | 18.5 % | 0.24 % (31.1 / 13,068) | **170** | 0.75 % (170 / 22,711) |
| GPU 10 × 100, library | 5.8 s | 33.2 % | 3.3 % (5.8 / 177) | 1,049 | 5.3 % (1,049 / 19,794) |
| GPU 10 × 100, aligned | 6.3 s | 35.2 % | 2.1 % (6.3 / 294) | 1,131 | 3.2 % (1,131 / 35,203) |
| GPU 20 × 1000, library | 21.5 s | 23.9 % | 0.56 % (21.5 / 3,851) | 539 | 1.7 % (539 / 32,309) |
| GPU 20 × 1000, aligned | 19.6 s | 26.8 % | 0.44 % (19.6 / 4,486) | 516 | 1.4 % (516 / 36,028) |

Small because the switch fires after 4–10 Gauss-Newton rows a level: 39–68 tracking products a ladder. (The overhead
in AE is `m_t·products + m_b·builds`; the arm's AE contains the first term, not the second, so its share is
approximate.)

## 5. Findings

### 5.1 The aligned-start inversion at 10 × 100

At the library start the switch beat exact (GPU: 223 against 267 Newton steps, 19,794 against 33,309 AE, 177 against
275 s). At the aligned start it lost: **exact 231 against switch 303 Newton steps**, 9,237 against 11,168 CG
(173 + 10,995), 28,471 against 35,203 AE (+23.6 %: 6,732 / 28,471), 230 against 294 s net (+27.8 %: 64 / 230). Most
of it is level 4: 85 Newton / 4,773 CG with 16 rejected trials, against exact's 51 / 2,602 with 8; levels 1 and 5 add
44 against 26 and 57 against 46. The switch fired on every padded level (sw@6–9), so this is not §1.6's hold-off; but
each level restarts in Gauss-Newton (answer 8) and pays its 5–9 Gauss-Newton rows before an exact step. Hypothesis,
untested: from an aligned start the exact model already pays from row 0, and the forced Gauss-Newton rows move the
iterate and the carried-over trust radius off the exact arm's path onto a longer exact phase. Per-level swings of this
size occur in both directions at 20 × 1000 (aligned level 8: switch 20 against exact 32; level 9: 40 against 22), so a
single ladder does not show whether level 4 here is systematic.

### 5.2 20 × 1000: a wash

| start | switch | exact | AE | net wall |
|---|---|---|---|---|
| library | 287 / 378 + 9,928 · 32,309 · 3,851 s | 291 / 10,511 · 33,413 · 4,011 s | −3.3 % (1,104 / 33,413) | −4.0 % (160 / 4,011) |
| aligned | 308 / 287 + 11,094 · 36,028 · 4,486 s | 288 / 11,006 · 34,938 · 4,262 s | +3.1 % (1,090 / 34,938) | +5.3 % (224 / 4,262) |

The errors are the same: level by level, switch and exact agree to ±0.0004 in val_fsr and in test p50 at both starts,
except library level 1 (test p50 0.1855 against 0.1722); best test p50 0.0449–0.0450 in all four runs. (The selected
levels differ at the library start, 7 against 8, so their test p50 reads 0.0461 against 0.0452.) The 2026-11-10 A100
runs showed a few Newton steps either way as GPU noise, and the per-level cells here swing both ways. GN-TR, partial
(levels 0–4): 1,157 Newton / 9,729 CG against the switch's 138 / 2,458 and exact's 140 / 2,759 over the same levels,
level 2 at the cap: a count, not economics. §6.2's bar at this scale is open: no `gn` arm, and the GN-TR and
`tracking` ladders did not finish.

### 5.3 Cross-process GPU rounding; T1 holds

The w03 tracking table's "same trajectory" reads yes at level 0 only. Each pair runs in its own process, and the GPU
rounds differently from process to process: the GN-TR and `tracking` level-0 endpoints agree to 1.5e-15, cores
included, but row 0's `‖g‖` at the next, zero-padded level start depends on the frame completion at a rank-deficient
point, which amplifies the difference. The checks (`gpu/t1_check_*.log`, `w05_timing/t1_check.py`):

- **One process, one start** (GN-TR's level-0 endpoint padded to level 1, two Newton steps): GN-TR and `switch_tau=0`
  are **bitwise equal** on the GPU (`uniform_jax`: row-0 `‖g‖` 1.469476207474e+01, `pHp` 2.315492724627 in both) and on
  the CPU (ragged, uniform, `uniform_jax`). **T1 holds.**
- **Two starts that agree to 1.5e-15** (the two arms' level-0 endpoints, CPU uniform): row-0 `‖g‖` 14.136 against
  14.906.
- **One start, three backends**: 15.34 (ragged, CPU), 14.14 (uniform, CPU), 14.69 (uniform, GPU).

So the per-level overhead (§4.2) compares slightly different trajectories, and a second GPU process would not
reproduce these counts row for row; the same amplification fits the laptop and the GPU agreeing at level 0 and parting
from level 1 (§5.4).

### 5.4 Laptop and GPU agree on the library pair

| | laptop (CPU), switch / exact | GPU, switch / exact |
|---|---|---|
| Newton steps | 219 / 277: −20.9 % (58 / 277) | 223 / 267: −16.5 % (44 / 267) |
| CG iterations | 220 + 7,759 / 9,139: −12.7 % (1,160 / 9,139) | 206 + 6,097 / 10,751: −41.4 % (4,448 / 10,751) |
| AE (each machine's multipliers) | 22,711 / 26,344: −13.8 % (3,633 / 26,344) | 19,794 / 33,309: −40.6 % (13,515 / 33,309) |
| net wall s | 13,068 / 16,777: −22.1 % (3,709 / 16,777) | 177 / 275: −35.6 % (98 / 275) |
| `switched_at` | 5, 10, 7, 7, 6, 6 | 5, 8, 8, 7, 6, 5 |
| selected level: val_fsr / test p50 | 5: 0.3750 / 0.0614; 4: 0.3740 / 0.0674 | 4: 0.3697 / 0.0675; 4: 0.3722 / 0.0682 |

The same direction in every currency, and the switch fired on every level on both. Level 0 is identical on the two
machines (exact 24 / 98 with 1 rejection; switch 28 / 7+122 with 2, sw@5); from level 1 the trajectories part (§5.3).
The GPU's wider margin is mostly the exact arm's level 5: 83 / 5,792 there against 60 / 3,641 on the laptop (level
5's AE on the GPU, exact 18,136 against the switch's 6,656, is 11,480 of the 13,515 margin). The A100 ran the ladders
61× (exact: 16,777 / 275) and 74× (switch: 13,068 / 177) faster in net wall.

### 5.5 The two Gauss-Newton arms; the switch everywhere

- GN with the line search and GN with the trust region cost about the same at 10 × 100 (library: 24,682 against 27,023
  AE, 427 against 436 s; aligned: 24,469 against 27,356, 432 against 454 s; CAP on 3–4 of 6 levels each), so the
  switch's gain is not its step control's.
- §1.6's "the switch may hold off at a zero-padded start" did not occur on these ladders: it fired on every level of
  all five switch runs, the zero-padded levels included, after 4–10 Gauss-Newton rows (§2.5).

## 6. What the docs should say (text only; for the session to paste)

### 6.1 `fitting_and_optimization.md` §4.14: the paragraph for `<!-- W3 BENCHMARK TABLE: ... -->`

Twelve lines with the table (blank separators aside); "downstream" as §4.12 says it. Numbers from §2.2 and §5.2.

```
**Measured downstream.** A rank continuation of a Darcy-flow surrogate (directional derivatives up to order 4 at 10
points × 100 directions, six levels to rank 24, `‖g‖ ≤ 10⁻³·‖g0‖` on the zero reference, at most 300 steps a level;
jax x64, `use_jit`, an A100 GPU). Ladder totals, apply-equivalents in Gauss-Newton applies at measured cost:

| `newton_cg` | Newton steps | CG iterations | apply-equivalents | wall s, less compile and display | levels at the cap |
|---|---|---|---|---|---|
| Gauss-Newton, line search (the default) | 1,625 | 24,682 | 24,682 | 427 | 4 of 6 |
| Gauss-Newton, trust region | 1,618 | 27,023 | 27,023 | 436 | 4 of 6 |
| `hessian='exact'` | 267 | 10,751 | 33,309 | 275 | 0 |
| `hessian='switch'` | 223 | 6,303 | 19,794 | 177 | 0 |

The switch fired on every level after 5–8 Gauss-Newton steps, its tracking took 3 % of its time, and its errors were
no worse than Gauss-Newton's. Restarted from the aligned frame (`padded_from`) it fired everywhere too, but took 303
Newton steps to the exact Hessian's 231; on a 20 × 1000 ladder of 11 levels the two were even, at both starts.
```

Optional, W4's call 12: the "It may never fire" bullet names the zero-padded start among the hold-off cases. These
ladders did not show it (§5.5), so a clause such as "(on the downstream ladders above it fired on every zero-padded
level)" could follow "until the new singular values have grown". The session's call.

### 6.2 `fitting_and_optimization.md` §4.13: replacing the GPU caveat

Replaces "A GPU overlaps far more of the host's work with the device's than a laptop CPU does, so it loses more to the
waits: measure on the GPU before reading its split." (numbers from `gpu/w05_timing/timing_table.md`):

```
On a GPU (an A100-SXM4-80GB, jax 0.10.2 in x64, the same protocol) the two cases above and one level of a downstream
fit (dof 12,352) measured −1.4 % to +3.1 %, jit and eager, each within its run-to-run spread (up to 10 %): the waits
cost nothing measurable there either. The small cases are dispatch-bound on the GPU and run slower than on the laptop
(the example, uniform `use_jit`: 1.62 s against 0.82 s), with CG at 1–4 ms per iteration and the eager phases around
it dominating the split, while on the downstream level CG takes 66 % of the rows' time. Cold runs compile a little
longer than on the laptop (the example under `use_jit`: 43 s ragged and uniform, against 31 and 36 s).
```

The CHANGELOG's timing entry ends with the same caveat ("A GPU overlaps more of the host's work with the device's and
can lose more."); a matching replacement: "On an A100 GPU the same cases and a downstream level differed by −1.4 % to
+3.1 %, also within the noise." (`CHANGELOG.md` is W4's too.)

## 7. Files and provenance

**T3Polynomial, committed** (`data/` is gitignored; added with `-f`; json and md only):

- `90dbd17` (2026-11-14): `scripts/w03_switch_benchmark.py`, `scripts/w03_README.md`;
  `data/w03_switch_2026_11/{w03_summary.md, w03_library_exact.json, w03_library_switch.json, w03_queue.log}` (the
  laptop).
- `dbda0de` (2026-11-14): `data/w03_switch_2026_11/gpu/{summary.md, gpu_env.txt}`;
  `gpu/w03_switch_2026_11_gpu_10x100/` (the ten `w03_<start>_<arm>.json`, `w03_summary.md`, `w05_headline.md`);
  `gpu/w03_switch_2026_11_gpu_20x1000/` (`w03_{library,aligned}_{exact,switch}.json`, the partial
  `w03_library_gn_tr.json`, `w03_summary.md`); `gpu/w05_timing/timing_table.md`.

**On disk only** (same directories): each pair's `.npz` (every level's whole `newton_cg` stats, history and timing rows:
`json.loads(str(np.load(f)['L0_stats']))`), `*_run.log`, `levels/`, the queue and job logs, `gpu/t1_check_{gpu,cpu,
endpoints_cpu}.log`, `gpu/chain.log`, `gpu/w05_timing/{measure,darcy}.jsonl` with the per-process `.err` and
`run_timing.log`.

**The pod notes**: `nicks_research_experiments/tensor/runpod/` (committed `ed9e55b`): `README.md` "The third pod"
(provisioning, what ran, the 20 × 1000 timing, the status at hand-back), `w05_w03_job.sh`, `w05_chain.sh`,
`w05_finalize.sh`, `w05_headline.py`, `w05_watch.sh`, and `w05_timing/` (`configs.py`, `darcy_level.py`,
`measure.py`, `run_timing.sh`, `t1_check.py`, `timing_table.py`).

**Should part b ever land**: `gpu/summary.md` §2 has the one-command rsync and says what to replace; the README says to
keep the pod up until `/workspace/results/DONE_finalize`.

**Anomalies, none affecting a number:** the 20 × 1000 `w03_summary.md` is titled "Darcy 10 × 100" (hard-coded; its
configuration line is right), and it has no "§6.2 bar" line (no `gn` arm); `w05_timing/darcy.jsonl` starts with three
lines of d10's `prepare()` output, which the table script skips; the laptop switch run's version line reads
`6b01889c+dirty` (§1). Timestamps: the laptop's logs are local (UTC−6), the pod's UTC.
