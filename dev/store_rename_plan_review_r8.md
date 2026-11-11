# The store rename plan — critic review, round 8

Reviewed: draft 8, the r7 changes.

- **r7 MINOR 1, resolved.** The lookbehind `(?<![\\\w])` replaces `\b` in the A3 family, the `nO` rule and K2. I ran
  the new pattern over `t3toolbox/`, `tests/`, `examples/` and `docs/`. Every hit is a width token (`nU`, `nUi`,
  `nOi`, `nD_i`, `nU0`, `nU(d-1)`, the `def nU(self)`/`def nD(self)` properties). The `\nOn` escape, `\nDone`,
  `NDArray`, `nUnits` and LaTeX `n_{D,i}` are not hit.
- **r7 MINOR 2, resolved.** K13 checks only store-named entries and V/H, and reports generic lines. §4 item 12 gives
  the ten generic kernel shape lines a tier-E disposition each, by call site (`:473` keeps the up store's `nU`).
- **NIT, resolved.** The default-width calls are in B1/B5.

## NIT

- §4 item 12 says a generic kernel parameter "takes the generic `n`/`r`". D-3 leaves `rL`/`rR` untouched, so only the
  `n` leg changes; say `n`, or say whether `rLi`/`rR(i+1)` become `ri`/`r(i+1)` on those ten lines.

VERDICT: AGREE
