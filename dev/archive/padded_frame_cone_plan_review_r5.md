# Padded frame / cone reference plan — critic review, round 5 (plan round 7)

**Plan:** `dev/padded_frame_cone_impl_plan_2026-11-02.md` (round 7, 2273 lines; CHANGES :79-82). This round re-checks
only what changed in response to review r4.

## Overall assessment

All three MINORs of r4 are fixed correctly and consistently across the plan, and all seven NITs are taken. The search
for leftovers turned up three small residues of the n2/n3 edits, listed below. All three are wording NITs that can be
fixed during the S5 and S10 slices; none needs another round.

## r4 findings — status

| r4 | status | where verified |
|---|---|---|
| **A** keyword placement | **fixed** | The six `newton_cg` keywords are appended **keyword-only** behind a `*`: in the backend after `callback` (§5.5 :1439-1452), in the frontend after `tr_radius0` (§5.7 :1499-1502, §1 :224). `gradient_descent`'s two keywords follow its last parameter, `g0_reference` (:1456). The plan states this in §3.6 :836-837, the CHANGELOG :1961-1962 and I-40 :2070. `inspect.signature(...).parameters` includes keyword-only parameters, so the mirror test (names and defaults) passes them. No existing positional call can change meaning. |
| **B** `1e-4` wording | **fixed** | The user docs now carry no comparison: fitting §5 :1620, the release notes :1929-1931, the CHANGELOG :1971. §3.7 :918-922 and O-9 :2104 keep the comparison "for the plan's record only", attributed to **T3Polynomial's** reference and explicitly not to the library's pre-2026.3.0 warm-start stop. §0 :157-159 is fixed. |
| **C** exception list and "replaces" | **fixed** | §3.3 :697-709 now holds the canonical list E1–E4: near-tie, degenerate hint, old frames differing by more than signs (tied or numerically deficient `x_old`), and an in-block coordinate seed in the old span. R10 :2090 references the list. R5 :2085 states E4's rarity. §4.3 :1056-1061 names E4. The contract-doc paragraph quotes all four (:1697-1705), and the internals page quotes them (:1738). Fitting §4.11 (:1600-1604) and `rank_continuation.md` (:1648-1651) point to the list. "Replaces" became "damped by the hinted passes but, at the default four, not removed" in §3.3 and §4.3. A new §3.8 caveat row covers E4 (:943). |
| n1 "natively" | taken | Three shipped sentences now say "each layer builds its **old** frame natively". The contract doc adds "the padded construction itself runs on unpacked real blocks in both cases" (:1693-1695). |
| n2 backend `N` check | taken; one residue | §3.4 :744-749 drops the requirement and says why. T5(g), I-37 and the §3.8 backend row agree. The `allclose` docstring mismatch is in the ledger (:1792-1794). Residue: see n8. |
| n3 G-nonbreaking scope | taken; one residue | The gate (:1572-1578) now runs main's `tests/test_*.py` **and** `tests/backend/test_*.py`, with the reason given. Residue: see n10. |
| n4 uniform example | taken | §5.7 :1487 shows the ragged and uniform forms; the backend signature comment is at :1342. |
| n5 non-minimal T4 row | taken | T4(a) now has a non-minimal `x_old` / template row, with the contract's reason. |
| n6 forwarding comment | taken | §5.7 :1502. |
| n7 §3.8 caveat row | taken | :943. |

## New NITs (leftovers; fix in the slices, no further round needed)

- **n8.** S5's content (§8 :2004) still lists "padded `N`" among the backend `padded_structures` guards, which n2
  removed. Drop it there.
- **n9.** §3.8's frontend structural row (:938) does not list the uniform padded-`N` check that the frontend now owns.
  §3.6 :881 and the pitfalls table :2163 have it. Add it to the row.
- **n10.** Three places still say "the pre-existing **frontend** tests pass unmodified": §1 :212, the release notes
  :1922 and the pitfalls table :2174. After n3, G-nonbreaking also covers the backend tests. The text is true but
  understates what the gate checks. Say "the pre-existing tests (frontend and backend) pass unmodified".

## Checked and found correct (this round)

- The keyword-only block is placed after the last positional-or-keyword parameter in all three signatures. The
  frontend `gradient_descent` still reaches the backend through `**kwargs` (docstring only), and its new keywords
  are keyword-only on the backend side.
- No sentence of shipped text compares the `1e-4` recommendation with a pre-2026.3.0 regime. The only remaining
  comparisons are labelled plan-record text attributed to T3Polynomial.
- The E1–E4 list matches the r4 analysis: covariance under signs, E4's rarity per the r4 probe, and E3 covering both
  tied and numerically deficient old points. Every user-facing paragraph either quotes the list or points to it.
- The rest of the plan is unchanged since r4, and r4's "checked" list stands.

VERDICT: APPROVE
