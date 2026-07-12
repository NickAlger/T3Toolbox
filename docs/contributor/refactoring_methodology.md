# Refactoring and rename methodology

Hard-won mechanics for renames, module moves, and file re-shelving in this codebase — distilled
from the frame rename, the 2026-07 naming pass (`dev/archive/naming_pass_plan.md`), and the
doctest sweep (`dev/archive/doctest_handoff.md`). Every item below cost a real incident or a real
near-miss. `contributor/naming_rules.md` says *what* may be renamed; this note says *how*.

## Before you start

- [ ] **Exhaustive inventory first.** Grep every inbound reference to the tokens/files you will
  touch — code, tests, docstrings, docs prose, `CLAUDE.md`, `dev/` — before changing anything. For
  file moves this is non-negotiable: the reference sweep is planned from the inventory, not
  discovered from build failures.
- [ ] **Inventory BOTH import forms**: `import x.y as z` *and* `from x import y as z`. A missed
  `from`-alias once silently broke the uniform seam.
- [ ] **Plan longest-token-first substitution order**, so substring collisions cannot fire; renames
  of bare names that other layers share must be **word-bounded and module-scoped**.

## During the rename

- [ ] **Frontend-method collisions need scoped renames.** Many backend names double as frontend
  method names (`apply_corewise_transpose`, `retract`, `rank_adjustment_sweep`); those need
  *qualified* (alias-scoped) renames, with bare renames only inside the defining module — a blind
  bare rename mangles the frontend.
- [ ] **Substring module names need lookbehinds** (`orthogonalization.py` ⊂
  `t3_orthogonalization.py`).
- [ ] **`getattr(module, 'name')` string references are invisible to token renames** — the suite
  catches them, loudly; expect and fix them rather than being surprised.
- [ ] **Slash-glob prose** (`compute_xis/mus/...`) and **brace-glob docstrings**
  (`probing.{a,b,c}_*`) never match a token sweep — find and fix them by hand.

## Gates

- [ ] **Never pipe the gate.** `pytest ... | tail` reports the *pipe's* exit code, not pytest's —
  use `set -o pipefail`, or capture the status directly.
- [ ] **Never edit tracked files while a gate runs.** One mixed-tree incident forced a reset and a
  deterministic replay; the gated tree must be exactly the committed tree.
- [ ] **The docs build is a gate too**: `sphinx -W` (zero warnings) after any change that touches
  docstrings, doc pages, or file locations — it is what catches missed cross-references.

## Delegating mechanical sweeps to agents

- [ ] **One agent per file.** Two agents editing the *same* file race and lose each other's edits;
  different files in parallel is fine.
- [ ] **Always re-verify by running** — the delegate's green report is not the gate; re-run the
  module's doctests/tests yourself after it finishes.
