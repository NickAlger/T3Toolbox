# K1 design note: response to review round 1

Implementer (the second), 2026-11-25, to `K1_DESIGN_REVIEW_r1.md` (NOT YET; no error in the design) and the
orchestrator's rulings of the same round. The note is revised in place (`K1_DESIGN_NOTE.md`, revision 2, its status
line lists the changes); the prose in `k1/prose_edits.md` (revision 2). I read `k1/REVIEWER_PRENOTES.md` after the
review; its points are answered by the review's §4 table, and nothing in it is left open by this revision. Every
finding is accepted; none is declined.

## The orchestrator's rulings, carried into the note

- **O1, `flat_draw(batch, batch_size)`, and the guards with the move.** Part 0 now holds them (§2.2 part 0, §4.11):
  the frontend `newton_cg` refuses at entry a `val_batch` whose `(kind, order)` differs from `batch`'s;
  `_frontend_draw(draw, batch, x0)` wraps every user draw on both layers (today only on a uniform `x0`) and compares
  each drawn batch's `(kind, order)` with the full batch's; `flat_draw`'s and the protocol's texts say a user draw
  returns a batch of the same kind and jet shape, and that the backend does not check it. New tests (§6.3 part 0) and
  mutations (§6.4) for both refusals. §9 records that K4 keeps the comparison.
- **O2** is K2's; the batch's `check` uses the private helper, and §9 says where K2 finds its rule table.
- **O3** accepted; `check_data`'s sentence is in `prose_edits.md` §7 (part 3).
- **O4** accepted. **D4** confirmed (recorded in §12's ruling paragraph).
- **§10's rule:** a part's build starts only when its public docstrings are written in full and reviewed. Parts 0
  and 1 are in full now (`prose_edits.md` §1, §6); parts 2 to 5 are completed before their builds.
- **§9's additions:** the public full-weight checker (K2, pending the maintainer); `inf` in a packed `y`'s padding
  (`y_block_sumsq` finite from K1, `f` and `f_resolution` from K5) and the statement that no K1 docstring promises
  more; the frontend comparisons of a drawn and a validation batch.

## The findings

**1. Part 0's review cannot see a wrong `(kind, order)` (should fix). Accepted.** §7.4 is now "The audits", written
before part 0's build. Its part (b) binds every old construction of `LeastSquaresOracle` or of a subclass of it in the
same file to the base's `(kind, order)`, every new `LeastSquaresBatch(` to its `(kind, order)`, and requires, scope by
scope (the function; then the enclosing class's `setUp` or body; then the module), that each new pair equals an old
pair as `ast.unparse` text, and that the counts of batch constructions agree. The cases it cannot pair mechanically (an
oracle received as an argument; a pair built inside a helper such as `tests/_fits.fit`) are listed with their call
chain for the reviewer to read by hand, their count in the handover. Part 0's review line (§2.2) names it.

**2. Part 4's review cannot see a block weight converted into a full weight (should fix). Accepted.** §7.4 (c): every
removed `blocks_to_weight(k, o, b, y)` is paired with a `block_weight=b` in the same scope (directly or through the one
assignment the old code made); every old `weight=` (any callee, `dc.replace` included, and the batch's positional
weight) is paired with the new call of the same callee and classified; **every new `full_weight=` site needs a reason
line** in `k1/full_weight_reasons.tsv` (the audit fails on a site without one; apply and entries sites are marked,
since only the reason tells a block weight passed as a full weight from a real one); and every new `block_weight=`
whose old value was not a `blocks_to_weight` output is listed with a reason. It runs for parts 1 to 5. Part 4's review
line (§2.2) names it.

**3. The test subclasses of the oracle (minor). Accepted.** §5.1 lists the nine classes and their about 15
constructions, with `_OffsetOracle`'s field shift; part 0's construction count names them; the audit (b) reads them as
oracle constructions. `test_dispatch.py:2178-2193` keeps its subject: with both oracles fieldless it pins the type in
`ValueHashedFields.__hash__` / `__eq__` (`common.py:678`, "type included: field-free siblings must not collide").

**4. The predicted red of parts 2 and 3; §0 1(b) (minor). Accepted.** §0 1(b) now says which weights the bridge
refuses (probe's `blocks_to_weight` output, and so **every probe derivative fit, weighted or not**, whose default
`k!` weight is placed by `blocks_to_weight` with an edge of 1) and which pass (apply's and entries', bitwise). §2.2
parts 2 and 3 and §5.4 list them, with what is not expected red; the list itself is recorded by a run.

**5. `pack_batch`'s own refusal (minor). Accepted.** §4.10: `pack_batch` checks the modes' pattern of size-1 axes
itself, before packing, with the engine's message (the object, the mode, the shapes, `np.broadcast_to`); a test
asserts the message (§6.3 part 4) and a mutation leaves the check to `pack_vectors` (§6.4).

**6. `LeastSquaresBatch(ww, y)` fails on "missing 'sample'" (minor). Accepted as ruled: no sentinel** (D16). §4.6 says
the plan's "fails on its kind" holds for three or more positional arguments and that two fail on the missing argument;
the class docstring's example (`prose_edits.md` §1.2) shows that message and says so.

**7. The default draw and user draws (minor). Accepted.** (a) `_require_draw` requires an integer `n_measurements` and
says, for a callable one, that it is a property now (§4.11; test and mutation in §6.3, §6.4). (b) `flat_draw`'s text:
"a batch of the same kind and jet shape as the full batch: the backend does not check it (the frontend optimizers
do)"; the frontend check is the orchestrator's ruling (above). (c) The docstrings of `mc_sgd` (`:1243`) and `adam`
(`:1295`) are in part 0's prose (`prose_edits.md` §1.9), with the frontend `mc_sgd`'s and `newton_cg`'s sentences and
`_frontend_draw`'s docstring (§1.11).

**8. The prose (minor). Accepted.** The engine's `full_weight` entry now says what the engine enforces: the shape rule,
and the forms it accepts (a sequence of `d` arrays or one mode-leading array; one packed array on the uniform layout),
not "in `y`'s form" (`prose_edits.md` §3). Part 1's reference texts are written in full (`prose_edits.md` §6: the
module's two paragraphs, `w_tilde`, the six tilde examples, `JtJp`, `Jp_norm_sq`, `f`, `r_block_sumsq`, the private
helpers; I checked that the rewritten examples draw the same random numbers as today's, so the streams after them are
unchanged). The `ω` lines of `backend/optimizers.py:1849-1851`, `:2096-2098` change in part 4 to the norm of the data
metric (`prose_edits.md` §7). `probe_data_metric`'s packed example stays a placeholder until part 3's prose round (the
rule of §10).

**9. The flip, and asymmetric tables (minor). Accepted.** §6.4 adds "the block weight applied after the flip" in
`w_tilde_step` (part 2) and in the probe converters (part 3); §6.3 requires block tables drawn at random, asymmetric on
every jet axis, at `order = (2,)` and `(2, 1)`, in the tests of parts 2 and 3.

**10. D4 (minor, for the orchestrator).** Confirmed by the orchestrator and logged; no change.

## What I ran

A check that `rng.random((3, 3))` and `rng.random(3)` draw the same numbers as today's `rng.random((3, 3, 1, 1))` and
`rng.random((3, 1, 1))` (they do). Nothing else; nothing is left running.
