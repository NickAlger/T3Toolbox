# The store rename plan — critic review, round 18 (draft 18)

Reviewed: the draft-18 diff against `d63bc017`, which applies Nick's two rulings: D-6 corrected so the corewise frame
is `(B, G, G, G)`, and the ttm/HT note's letters.

**Sound.**
- The corewise rule is rewritten from its meaning. The manifold operation runs at an orthogonal `(D, U, L, R)`; the
  corewise one substitutes `B → D`, `G → U, L, R` and works for any `B`, `G`.
- The template sentence is given. "No longer required orthogonal" and "the down slot holds" are dropped.
- Tier D, L3, §3.1, §3.3, D-6, the utv retract row and the convention paragraph agree. K4 bans the old forms.
- The ttm/HT changes (`B_i`, `T_t`, `\X`/`\Y`/`\Z`, each with an HT-literature line) are applied as Nick accepted.

## MINOR

1. **The corewise site list misses four sites, and one of them escapes every gate.** I grepped the tree for the
   substitution's equalities and the dropped phrase:
   - `tests/backend/_curvature_oracles.py:312` "combine … at ``O = G``" and `:496` "dense combine ``mu O sigma^`` at
     ``O = G``": these become `U = G`. K4 bans only `Q = O = P = G` and `Q = O = G`. K8 has no `O = G` context, and its
     `mu O` pattern flags `:496`'s other half only. `:312` escapes entirely.
   - `t3toolbox/backend/sampling_derivatives.py:3252` (`# Q = O = G`, beside `compute_sigma_jets(tt_dirs, tt_cores,
     tt_cores, …)`) → `R = U = G`. K4 catches it, but the list should name it.
   - `docs/symmetric_probe_derivatives.tex:559` "$\mathbf P,\mathbf Q,\mathbf O\to\mathbf G_i$, $\mathbf U$ no longer
     required orthogonal": a corewise substitution sentence in a note. It needs the template, as
     `\mathbf B\to\mathbf D`, `\mathbf G\to\mathbf U,\mathbf L,\mathbf R`. K4's phrase ban catches it, but neither §1.2a
     nor §4.8 lists it.

   Fix: add the four sites to §1.2a's corewise list, and add the single-equality forms (`\b[OPQ] = G\b`) to K4.

VERDICT: REVISE (0 MAJOR, 1 MINOR)
