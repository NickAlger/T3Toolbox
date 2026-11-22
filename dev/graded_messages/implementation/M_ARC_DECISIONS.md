# Decisions Claude made during the M slices, for discussion with the maintainer after the arc

The maintainer (2026-11-22): "Go ahead with M2 and continue through the rest of the M slices, unless a major issue
arises. Please track decisions you make and we will discuss them after the M slice arc is complete." Each entry: the
decision, the reason, where it is recorded. Larger rulings are also in `LOG.md`.

## M0 (the `J -> P` rename; committed `64465d8b`)
1. `F_in,i` is spelled `F_in_i` in code, comments and docstrings (an identifier cannot hold the comma; same width as
   `J_in_i`, so aligned columns do not move); prose writes `F_in,i`.
2. The shape rule: a concrete shape spells the sizes (`F`, `min(F_in_i, F)`); a family of shapes spells the group
   (`P + W + [K] + C`); the packed input is `(d,)+F_in+W+(N,)`.
3. The test-local jet ring `J = Jets(order)` in `test_sampling_definitions.py` renamed `ring`.
4. A static guard that no library contraction literal contains `J`, and a floor in the literal scan (a missed `'J'`
   filter would otherwise empty a test's loop silently).
5. The baseline for per-test counts is one local `-v` run of a `git archive` copy (not CI), kept gzipped in
   `implementation/m0/`.

## M1 (`data_metric`, `blocks_to_weight`, the law helper; committed `c16aa640`)
6. `_with_K` (the engine's `K`-aware weight alignment) moved from M1 to M2, where the engine first calls it.
7. `data_metric` with a packed `r` and a ragged tuple weight raises (naming `pack_batch`) rather than reading the tuple.
8. `LeastSquaresBatch.data_metric(r)` raises by rank on today's block-weight batches until M2 switches batches to the
   full form (pinned by a test), rather than converting blocks inside the method.
9. The frontend dual-jet converters leave `t3toolbox.fitting`'s namespace entirely (they are not in 2026.5.0).
10. A ragged `r` needs the exact mode count; every weight axis must be 1 or `r`'s size (so `data_metric` never grows
    `r`); the scale must be 0-d.
11. Type comments write the algebraic types in Unicode (`⊗`, `⊕`, `ℝ`, `*`), identifiers stay ASCII (`omega`,
    `gamma`); recorded in `docs/contributor/signature_style.md`.
12. The law helper is `tests/_typing_laws.py` (the plan's `_typing` shadows a CPython module).

## M2 onward
