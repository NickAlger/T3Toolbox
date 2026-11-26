# K1 design note: response to review round 2

Implementer (the second), 2026-11-25, to `K1_DESIGN_REVIEW_r2.md` (SIGN OFF, five minor findings). All five accepted
and folded into the note and the prose; the reviewer checks them with part 0's build.

- **11. Accepted.** Note §7.4 (b): one level of helper is resolved (a batch whose pair is its scope's parameters is
  compared, at each call site in the file, with the call site's old oracle pairs); listed by hand: what still cannot be
  paired, every removed oracle whose scope keeps no paired batch, every scope with two distinct old pairs (one, in
  `LeastSquaresOracle`'s doctest), and every test-subclass construction's old and new argument lists side by side.
- **12. Accepted.** Note §2.2 part 1: part 1 rewrites every test that hands the reference a now-refused form,
  `TestAgainstTheReference` (`tests/test_engine_data_space.py:139-158`) among them; the part's run finds the list.
- **13. Accepted.** Note §2.2 part 2: hand-built probe or apply full-form weights with a structural axis of 1 join
  the predicted red; the expected red is recorded per subtest from the `-v` log; part 4's green run is named as the
  first check of the engine on the cases hidden behind a red node.
- **14. Accepted.** `prose_edits.md` §1.9: the backend `mc_sgd` and `adam` say a draw's batch is not checked there.
- **15. Accepted.** `prose_edits.md` §1.2 and §1.4: part 0's interim texts keep today's sentences on `y` and the weight;
  the `y` sentence arrives with part 2 (the engine) and part 3 (`check_data`). The example's exception line is pasted
  from the run on Python 3.11 (`LeastSquaresBatch.__init__() missing ...`), `+IGNORE_EXCEPTION_DETAIL` covering the
  floor.
