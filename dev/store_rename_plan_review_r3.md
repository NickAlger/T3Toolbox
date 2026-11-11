# The store rename plan — critic review, round 3

Reviewed: draft 3 (677 lines, 00:43), the changed parts: CHANGES, §1.2-§1.4, §3.2-§3.4, §4.8, §5.1 B7, §5.3, §7, D-1.

**Resolved from r2.** N1: `\ell` is unused in `exact_hessian_two_hole.tex` (checked; its other uses, in
`pad_safe_svd.tex` and `padded_frame_cone.tex` for leaves, are other notes), `\Sh` is undefined everywhere, and K7
checks the old Lagrangian forms. N2: §3.4's R1 inventory with its classification, §1.4's locate/replace/layer
procedure (locating by the R2 image is sound: a stale or doubly matched row fails), tier E before tier D on a shared
line, and K4/K8 extended. Minors 1-6 and the nits are fixed: sentinels carry no old text and resolve per-key counts;
the catch-all is function-scoped and the `th` rename is gone; the dense oracle maps `Uo/Uc/E/Eo/Ec`; K8b uses `[OPQ]`;
K7 has the bare forms; B5 flips the T3 test's `'U'` with its `dU`.

## MINOR

1. **K9's clause (b) accepts a wrong flip.** "A word whose aligned old-tree token was the opposite store word" also
   passes a keep word flipped by mistake: an ORTH adjective (`up-orthogonal` → `down-orthogonal` on a TT core) or a MSG
   direction ("contract up"). K6 covers only six phrasings. Add the converse: every old-tree token on a keep row
   (ORTH, MSG, UNREL) survives unchanged at its aligned position, unless the line is a tier-E proposed text (§3.3's
   rewordings of "frame down jets", "up-index", "down `eta`" are such rows).

## NIT

- §1.4 says tier D's "patterns include the hybrids", but §1.2's tier-D list has no `(D, U, P, Q)` → `(D, U, L, R)`
  entry. List it, or drop the claim (L1 and K4 cover it).
- The dense-oracle row leaves the dense-`D` names `startD`, `Da`, `Db` (`_padded_frame_oracles.py:123-178`) beside the
  renamed `old_dense`/`cur_dense`, and `test_padded_start.py:478-481` (`startD`, the dense result `D`, in
  `shared_one_pass`) lies outside its 566-580 range. No `U` shares those functions, so this is naming only.
- `gYU` (`_curvature_oracles.py:72-75`, the gradient's Tucker part) is a `U`-family name missing from §3.4's list, so
  the R1 scan pattern may not see it (→ `gYD`).

VERDICT: REVISE (0 MAJOR, 1 MINOR)
