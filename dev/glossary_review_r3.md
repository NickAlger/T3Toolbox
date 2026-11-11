# Glossary review, round 3 (critic)

I re-reviewed the writer's changes after round 2 (`dev/glossary_review_r2.md`). Only
`dev/glossary_naming_decisions.md` changed; `docs/glossary.md` has not changed since the round-2 `sphinx -W` build.

## Round-2 findings: status

- **MINOR 1, the `UT3FrameMasks` E4 row: resolved.** `uniform_frame_variations_format.py:52` now has its own text:
  "canonically a prefix, gappy in a padded frame's block layout". That matches `UT3Frame.padded` / `ufv_padded_frame`.
- **NIT 1: done.** There are new N4 rows for `frame_variations_format.py:101` (`D1` → `O1`) and `:1271` (`D2` → `O2`),
  and the `:355` row now rewrites the einsum operands to `O, O`.
- **NIT 2: done.** The E6 row at `fitting.py:642` drops the whole parenthesis. A new row at `:641` adds the period that
  closes the sentence; I checked it against the source.
- **NIT 3: done.** C1 now suggests `'cone': 'cone (aligned zero tensor)'`, with the wording left to Nick.

**`sphinx -W`:** a clean rebuild of the round-2 tree (`build succeeded`, exit 0, no warnings, `glossary.html` written).
It includes every glossary change, since the glossary is unchanged since then.

## Open for Nick (decisions, not defects)

- **N17:** the orthogonality adjective for the `O`/TT mode-leg property. Its 20 rows are HELD.
- **N12:** "padding" vs "junk slot". It has no rows.
- **The E6 policy for contributor pages and math notes:** add a key line or drop the codes.
- **C1:** the display label for `'cone'`.

No MAJOR or MINOR remains.

Verdict: AGREE
