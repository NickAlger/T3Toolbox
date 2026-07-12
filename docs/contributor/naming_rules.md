# Naming rules for contributors

The user-facing catalog — the grammar, module map, semantic markers, and deliberate exceptions —
is [`../naming_conventions.md`](../naming_conventions.md). **Read it before naming anything new.**
This note carries the policy side: the rules that govern *changing* names, which are contributor
constraints rather than user information.

- **Governing principle (Nick):** naming exists to help the user — orient, find, understand
  inputs/outputs. When convention and clarity conflict, clarity wins.
- **The semantic markers (`corewise`, `numerically_`) are rename-invariant**: no rename may add or
  drop them. They separate same-looking, different-math near-twins (`sum_stack` vs
  `sum_stack_corewise`; `has_minimal_ranks` vs `has_numerically_minimal_ranks`) — dropping one
  silently changes what a name claims.
- **Do not "unify" the converter names** (`from_t3` vs `from_ut3` vs `from_t3frame`, …): the names
  carry the operand types; a unified name would erase exactly the information they exist to carry.
- **`utt_` is reserved and deliberately empty** — the ragged `tt_` name is the polymorphic name.
  Do not add `utt_` members; a new chain op that serves both layers goes in `tt_operations`
  unadorned.
- **The cataloged exceptions and intended asymmetries are load-bearing** — they were weighed and
  chosen (decisions log: `dev/archive/naming_review.md`; execution:
  `dev/archive/naming_pass_plan.md`). Do not "normalize" them to the grammar; the catalog in the
  user doc is the registry of record.
- **Parameter names encode representation** (`x` ragged vs `data` uniform; `xx` vs `uxx`) — keep
  them; they are documentation, not style drift.
- Renames have a wide blast radius — follow `refactoring_methodology.md` *(lands with the S4 slice
  of the docs split; until then see the methodology lessons in `dev/HANDOFF.md` /
  `dev/archive/naming_pass_plan.md` §D)* for the mechanics (inventory both import forms, `getattr`
  string refs, substring module names, frontend-method collisions, gate discipline).
