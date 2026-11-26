# Step 2: the remaining diagrams, drawn from the page alone

**What was done** (2026-11-16, 00:00 to 00:30). An Opus agent was given `SAMPLE_VARIABLES.md` as its only source
for what a variable is, and the diagram generator with its seven finished rows. It was told not to read any other
document or script of this folder, and to report every place where the page was unclear instead of resolving it
elsewhere. It added 21 rows to `scripts/variable_diagrams_gen.py`: the other fifteen variables, the outputs `z`
and `Jp`, and the blocks `V_tilde`, `H_tilde`, `V_tilde_dot`, `H_tilde_dot`.

**The check** (Claude). Each of the 21 rows was compared with the definitions of the page: which cores stand at
which sites, which core is the direction, which input is the seed, which edge is open, in which basis, which way
the arrow points, and the shape. All 21 are right. The seven earlier rows are unchanged byte for byte. Every new
row was looked at as rendered locally, and six as rendered by the doc.

**The answer to the maintainer's question** ("did the opus agent understand the notation correctly from what is
written in the sample variables page?"): yes, for every row. In particular it got right the cases that the names
exist to separate: `mu_dot` through `L` cores with the raw direction against `mu_R_dot`; `R[2]` at the open site
of `eta_D_dot` when the direction is to its left; `nu_L_dot` beginning in the `L` basis; `xi_U_dot` as `V` in
place of `D`; and that the direction can be `B_dot[i]` but not `G_dot[i]` in `H_tilde_dot[i]`.

**Where the page was unclear, by the agent's report, and what was done.**

| # | The agent's question | Done |
| --- | --- | --- |
| 1 | Do blocks carry `C`? | Page section 7 now says so, with the two shapes |
| 2 | Do the blocks of `JtJp` have `K`? | Section 7 now says when a block has `K` |
| 3 | How are outputs and blocks written in a formula? | Section 2 now says |
| 4 | How do a tilde and a dot stack? | Section 2: the tilde next to the letter, the dot above it |
| 5, 6 | `V_tilde_dot` and `H_tilde_dot` stand for a `D` and an `L` core but are named `V` and `H`; what is taken out? | Section 6 now says which core is taken out, and why the letters stay |
| 7 | "Top coefficient" and "summed over the samples" are not explained on the page | Left: target decision 2.4 is cited. Section 7 now says why a block has no `T` and no `W` |
| 8 | The nine terms of `H_tilde_dot` | Section 6 now spells them out, and says the direction at site `i` can only be `B_dot[i]` |
| 9 | Which TT core is at site `i` in `eta_D_dot` when the direction is elsewhere? | Section 5.2 now says: `R[i]` if the direction is to the left, `L[i]` if to the right |
| 10 | In "U over V", which part is the direction? | Section 5.2: the direction's own core is `H`, or `V` |
| 11 | How to draw `B_dot` and `G_dot` (the argument's name is not decided) | Drawn as `B` and `G` with a dot. The name stays open (section 11) |
| 12 | The "one dot" caveat was dropped from the notes for space | The two notes now say "for the lifted direction, with no damping" |
| 13 | Section 6 spells corewise variables with a basis; section 9 says a corewise call uses none | Section 6 now says which point of view its names are from |
| 14 | A seed under the hole of `V_tilde` | Section 6: "which may be input `i`" |
| 15 | Why `xi_tilde` has `T` and not `A` | Section 7: the seed is a full dual jet |
| 16 | The basis of a mode edge | Section 2: a mode edge has one basis |
| 17 | How to show the place of an input on an open mode edge | A dashed circle; said in the tab's key |

**The agent's own devices**, accepted: a dot over `G` or `B` for the raw core direction; a dashed coloured box
for a core taken out, with the legs of the block as arrows into it and the block's name inside; a dashed circle
for the place of the input on an open mode edge.

**Seen in the doc's rendering, not changed:** the tilde over the letter xi touches the letter in the doc's font.
