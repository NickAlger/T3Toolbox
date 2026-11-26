NOT YET

# Review of `K0_DESIGN_NOTE.md`, round 2

Reviewer: Claude, 2026-11-25. I read the revised note, `K0_DESIGN_RESPONSE_r1.md`, `k0/prose_edits.md`, and
`k0/experiments/e1c.py` and `mutate_step.py`, and judged them against the code at `b5125f76`. Where a statement
concerns the code after K0, I checked it against my own simulated K0 library (`mysim.py`, in my scratch folder;
it is regenerated when needed and nothing is left there).

**Verdict.** Every finding of round 1 is resolved in the note itself, not only in the response. The rulings are
applied as given. Two should-fix points remain, both in the new prose. Each is a few lines of `prose_edits.md`, and
a third round need only check that file.

## 1. The round-1 findings, against the revised note

| # | Status | Checked how |
|---|---|---|
| S1 | **Resolved.** §6 (i) and (iv) now name library steps per kind, each with its expected outcome; the r1 mutation is withdrawn | Reran (iv) with my own jvp script. `H_tilde_step` ×(1+1e-6) fails probe (7e-7 to 8.6e-7), apply (7.2e-7, 7.7e-7) and uniform probe; `H_tilde_dot_step` fails probe and uniform probe and leaves apply alone; `V_top_step` fails nothing (≤ 3.4e-16). This matches the note and the response, including the stated survivor `V_top_step`, an identity test's blind spot that `test_sampling_apply.TestByDefinition` covers |
| S2 | **Resolved in form** (`prose_edits.md`). Two points in the content remain: §2 below | read in full against the simulated K0 tree |
| S3 | **Resolved.** One assertion, the module's absence, in `test_import_graph.py` (§3.8); no export lists, no patterns; the count stays 1,882 | note §3.8, §6 |
| M1 | **Resolved.** Factorials inline (§3.7) | note §3.7 |
| M2 | **Resolved as ruled.** Both bullets go whole and `:130` loses the module | my own scan of the page's 93 `:py:` roles (import and `getattr` per target): the same nine missing targets the implementer found, no other. The page reads correctly without the two bullets. "These are the one exception to the rule above" (`Regularizer`) becomes accurate; it was not while the `least_squares_problem` bullet stood. No other stale name in its text |
| M3 | **Resolved.** All three left as history | §3.7 |
| M4 | **Resolved,** with a third patch (§3 below) | reran E1c: padding 5.4e32 (probe), 8.3e31 (apply), 8.6e31 (entries), on both faces |
| §6 | **Resolved.** The recorded step before the deletion, with the slow tier of `test_uniform_curvature.py` (§4 below) | note §6 |

## 2. Findings

**S1 (r2). Should fix. A false statement in `probing`'s new module docstring** (`prose_edits.md` §1, `probing.py:5-13`).
The after-text says: "the probe of a point and of a tangent, their transposes (ambient, corewise, tangent), the
corewise forward derivative and the corewise Hessian ..., **each one call of the engine**".
`t3_probe_ambient_transpose` does not call the engine: it returns CP factors, frame-free. An `ast` scan of the kept
functions on the simulated tree finds the engine call in each of the six others and not in this one. The
docstrings of `apply` and `entries` say it correctly: "each but the ambient transpose one call of the engine". Resolve:
the same wording for `probing`.

**S2 (r2). Should fix. The mechanical rule "a comment block before a deleted statement goes" removes the section
headers of code that stays** (note §1.3; `prose_edits.md` does not cover these). I listed, for each range the note
deletes, the first kept statement after it. Three ranges head a section whose kept functions follow a deleted helper:

- **`probing.py:672-674`**, the banner "Transpose of tangent to probes map". It heads `tv_probe_transpose`,
  `t3_probe_ambient_transpose`, `t3_probe_corewise_tangent` and `_transpose`; the deleted `_deta_tilde_step` merely
  comes first. Without it these four sit under the banner "Probing a tangent vector". The ambient and corewise
  functions do not belong there, and the new `__all__` grouping (`prose_edits.md` §2) groups them otherwise.
- **`ut3_sampling.py:124-133`**, the header of the kept corewise transposes (`ut3_*_corewise_tangent`, `_transpose`).
  The deleted `_remasked_core_gradient` comes first. The block is mostly pre-engine ("Mask-once + pack at the boundary",
  "via the now-polymorphic `probing.*_corewise_transpose`"). It also holds statements that are still true and useful:
  the boundary-bond squash broadcasts into the padded slots, the exit mask zeroes them, and "the masks, not the
  values, govern rank".
- **`ut3_sampling.py:469-473`**, the header of the kept plain Hessians (`ut3_*_corewise_hessian`). The deleted
  `_masked_point_and_directions` comes first. Its text describes the pre-engine wrapper ("the polymorphic kernel run
  on the supercores"), but its title line heads code that stays.

Applied as written, the rule leaves `ut3_sampling`'s kept corewise functions with no section header at all. Resolve:
move these three from "deleted" to "cut to what stays", with their after-text in `prose_edits.md` (a banner or a
one-line header each; for `:124-133`, keep the true sentences). Also check the other deleted ranges for the same case
by the same test (the first kept statement after the range, and whether the block heads it). By my listing the
others head deleted code only, except `fitting.py:154-160` and `:745-749`, which the note already rewrites.

**M1 (r2). Minor, and optional.**
- The header `utv_sampling.py:444`, "the transpose 𝒥ᵀ (probe)", already heads all three kinds' transposes today (not
  K0's doing). If K0 touches it, "(probe, apply, entries)".
- `fitting`'s new module docstring drops the old factoring `J = 𝒥∘Π`, `Jᵀ = Π∘𝒥ᵀ`. That factoring is still true, and
  the new text's "the geometry ... is the other half of a fit" leaves the reader to find it. One clause would keep it:
  "(the gauge projection `Π`: `J = 𝒥∘Π`, `Jᵀr = Π∘𝒥ᵀr`)".

Everything else in `prose_edits.md` I found true of the code after K0:
- `fitting`: the oracle's two static fields, the engine call, `check_batch`/`take`/`n_measurements`, the batch's five
  fields, and `optimizers` composing the two.
- `apply` and `entries`: the engine calls, `apply`'s use of `probing._dense_jet`, `entries`' use of `apply`'s ambient
  transpose.
- `ut3_sampling`: "packs a ragged sample" and "mirrors the sample's form" are true at K0 (`ut3_probe` unpacks when
  given ragged); `ut3_full_sum` exists.
- `utv_sampling`: the transposes re-mask through `_remasked_transpose_output`.
- `uniform_fitting`: `pack_batch(kind, order, batch, N)` is today's signature.
- `fitting.py:154-160`: `_flat_gather` is used by `take` and, through `_take_data_space` → `_take_W`, for `y` and the
  weight.
- `entries.py:113-116`, `common.py:524-526`, `utv_sampling.py:449-453`.
- Nothing anticipates K1 to K7.

The list "reviewed and left as they are" is right as far as I sampled it:
- `ut3_sampling.py:68-69`;
- `uniform_fitting.py:366-374`;
- `fitting.py:751-755`;
- the docstrings of every kept function of the seven modules, scanned for "polymorphic", "kernel", "wrapper",
  "mask-once", "from_sweep", "old", "P1x/P20", "precompute", "derivatives", "chunk": nothing stale but the
  `ut3_entries_corewise_hessian` "kernel" phrase, which the note reviewed and kept, rightly.

No signature column or diagram of a kept function is touched by any listed edit.

**Watch during the build** (not findings on the note):
- `test_import_graph.py`'s module docstring lists the facts it pins. The new test adds one, so the docstring gains a
  line. `prose_edits.md` §4 does not list it.
- The mutations of §6 were measured "on prototypes of the re-bases". Rerun them on the built tests, in scratch, and
  record the outcomes in the handover.

## 3. The negative control with two patch points

The control removes every masking layer between a junk-padded input and the output:
- the engine's entry masks (`_UNIFORM_MASKS` emptied);
- its exit mask (`_mask_variable` the identity);
- for the tangent faces, their own re-mask (`ufv_masking.ufv_apply_variations_masks` returning the supercores).

It then asserts that clause (c) fails. That shows what it claims: zero padding is produced by the masking, not by the
arithmetic, so (c) is not vacuous. Since the input is junk-padded, (d) is load-bearing too. Patching two places is
right, not a weakness: there are two masking layers on the tangent faces, and a control that removed only one would
not fail (E1c measured 0.0 without the third patch). What it does not show is which layer is necessary. That is a
property of the design (layered masks, either suffices), and `TestKpAtTTSlack` pins the layers individually for the
engine.

Robustness to K5: the control patches private names. If K5 renames or removes one, `mock.patch.object` raises
`AttributeError`. If K5 keeps the name but routes masking elsewhere (selection in a new helper), the patch stops
biting, the padding stays zero, and the expected `AssertionError` does not come. Either way it fails loudly, never
silently. **What K5 must do with it, in a line:** point the patches at K5's masking points (the selection on entry,
the exit mask, and the twins' re-mask if K5 keeps it, else drop that patch), and keep the assertion that (c) fails.

## 4. The recorded step before the deletion

Running the edited test files on the untouched library shows three things:
- each re-base passes without any deleted code being reachable, since a re-based test that still reached deleted code
  would also pass there;
- combined with the gate's `--check` on K0's tree, which proves no reference remains, it shows a re-based test passes
  on kept code only;
- combined with the mutations, it shows each re-based oracle can fail for a real bug.

That is "passes for the right reason". The slow tier of the edited `test_uniform_curvature.py` before the deletion is
the right addition: its two `_SLOW` tests are otherwise checked only at the end. Enough.

## 5. What I ran

- `mysim.py`, the K0 simulation (library only, my own deletion set).
- An `ast` scan of the kept functions for engine calls.
- A scan of every `:py:` role of `docs/api_reference.rst`.
- A listing, for each comment range the note deletes, of what follows it.
- A grep of the kept docstrings and comments for stale phrases.
- E1c.
- The jvp re-base under three step mutants.

All in scratch copies with `python -B`. No test run in the repository, nothing left running.
