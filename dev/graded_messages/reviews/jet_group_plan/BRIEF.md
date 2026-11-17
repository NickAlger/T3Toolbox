# Brief for the plan of the generalized jet group

Written by Claude on 2026-11-17 for two agents: a writer, who writes the new slices into
`dev/graded_messages/PART1_PLAN.md` and an account of the change, and a critic, who reviews them. They iterate
until the critic agrees. The maintainer then reads the result and signs off before anything is implemented.

## What was decided

Read first, in `dev/graded_messages/TARGET_DECISIONS.md`, the note **"One jet group, of any number of axes"**
(just before "Carried to Part 2 from the implementation of Part 1"). It holds the maintainer's words and the
analysis he accepted. In short:

- Today the jet contraction `jet_contract` (built in slice P1, `t3toolbox/backend/contractions.py`) knows two
  order groups: `T`, the order axis of a full jet (`order + 1` slots), and `A`, that of an affine variable (two
  slots), each of zero or one axis.
- They become **one jet group of any number `m` of axes**. A jet in `m` variables with maximum order `n_j` in
  direction `j` is an element of the polynomial ring in `s_1 ... s_m` cut off at `s_j^(n_j + 1)`. Each operand
  carries its own sizes along the jet axes; an affine variable is simply a jet with two slots per axis. The
  contraction is the truncated product of polynomials.
- **The truncation is a box** (a maximum order per direction), not total degree: the box keeps the property the
  dual jets rest on (pairing with a dual jet is the top, corner, coefficient of the product; the turn is a flip
  along every jet axis; row 2.4).
- **An operand's shape means: a polynomial whose coefficients beyond it are zero.** A full jet is a truncation and
  is valid only to its own order; the output's shape must not exceed the shape of any operand that is a
  truncation.
- **The output's jet shape is an explicit argument of the jet contraction, with no default.**
- Efficient algorithms for several variables belong to Part 2, behind the seam of the jet contraction. Part 1
  needs a simple correct algorithm for the general case and the present efficiency in the cases the library
  uses (one variable; a full jet with a full or an affine partner; the readouts).

## What the maintainer asked for

"I would like to do this as a new slice (or multiple slices) at the pause, inbetween P4 and P5. We would do a
plan phase with writer and critic Opus agents to create the new slices and update the plan document before
implementing."

## Where things stand

- The repository is `/home/claude/t3toolbox`, branch `graded-messages`. `CLAUDE.md` at its root is the project
  guide and binds the plan. The design folder is `dev/graded_messages/`: `README.md` (status), `PART1_PLAN.md`
  (the approved plan, 21 slices P0 to P20), `TARGET_DECISIONS.md` (the decided rows and notes; the latest note
  wins), `SAMPLE_VARIABLES.md` (every variable's name, definition, recurrence and shape),
  `implementation/BRIEF.md` (how slices are implemented and reviewed), `implementation/LOG.md` (what each
  finished slice did and decided), and each slice's handover and reviews in `implementation/`.
- Slices P0 to P4 are implemented and committed (P4 is `5991725`; check `git log` and `implementation/LOG.md`).
  P5 to P20 are not started, and nothing else is being implemented while the plan is written. So far built: P1 `jet_contract` and `GroupLengths(T, W, K, C)`; P2 the reference module
  `backend/sampling_definitions.py` and the dense functions; P3 `backend/jets.py` and
  `backend/data_preparation.py`; P4 the checkers of a frame.
- Decisions made during the implementation that bear on this: the site group `S` was removed and a rule set
  ("new groups must be weighed carefully and only introduced with good reason and after deliberation"); the
  group `A` was kept after deliberation, with a definition (now superseded by this generalization, which keeps
  its meaning as a smaller jet shape); the contraction order inside one step is deferred to Part 2 and marked by
  an expected-failure test (`TARGET_DECISIONS.md`, "Carried to Part 2", item 1); jets in several variables had
  been noted as a future feature.

## What the writer delivers

1. **The new slices, written into `PART1_PLAN.md`** in the plan's own form (what changes, how, the tests, the
   gate, what breaks and where it is fixed), placed between P4 and P5 and numbered so that P5 to P20 keep their
   numbers (for example P4a, P4b). Each small enough to review. They rebuild what P1 to P3 built wherever the
   generalization reaches it, and they change the plan for P5 to P20 wherever those slices assumed `T` and `A`.
2. **The plan brought in line everywhere else**: sections 1 (scope and the seams for Part 2), 2 (the end state:
   the data, the engine's signature and its table of names and shapes, the public surface, the signature
   sketches), 3 and 11.3 (names), 4, 5.2, the slices P5 to P20 that mention `T`, `A`, the 41 strings, `order`,
   `GroupLengths` or the order axis, 6 (tests), 7 (the tricky areas: dispatch, uniform, sharding, jit), 8
   (documentation), and the record in section 11.
3. **An account**, `reviews/jet_group_plan/ACCOUNT.md`: what changed in the plan, section by section; the design
   in one page; every choice you had to make, each marked as following from a decision or as open.
4. **Open points for the maintainer**, in the plan's section 11.2, each stated so he can answer it. You do not
   settle design questions; a plan that silently decides one is wrong. New public names are proposals for him.

## What has to be worked out (the writer proposes, the critic tests, the maintainer decides)

These are the questions the plan must answer or list as open. Reason from first principles, as the library's
guide asks: design the correct thing, then say what it costs.

1. **The interface of the jet contraction.** One reserved letter for the jet group, `J` (decided; see item 3).
   How each operand's jet sizes are known (from its array). How the output's jet shape is
   passed (an argument of each call; or a field of the record every call already passes; say which and why).
   What the record `GroupLengths` holds. What `over_orders` means now. Whether the three behaviours of today
   (ride, product, top coefficient) are one behaviour. What a readout is in the general setting (the corner
   coefficient; or the pairing of a jet with a dual jet). What exactly is checked (the rule that an operand of
   the wrong rank raises, row 5.4).
2. **The semantics, stated exactly**, with the cases that are easy to get wrong: an operand with fewer slots than
   the output; with more; a size-1 jet axis; operands with no jet group; `m = 0` (must be exactly `contract`);
   the truncation rule and who is responsible for it.
3. **The scope in Part 1: the public interface is generalized from the start** (the maintainer, 2026-11-17,
   13:58: "I would like to generalize it from the start. I am concerned that if we do not do it right in the
   first place, downstream code and future features may start to assume the current structure, and we will have
   to break them later if we generalize. This release is already mega-breaking, so this is the time to do it.").
   So not only the jet contraction: the sampling functions and methods, the engine, the oracle and the batch,
   the data and the weight, the jet helpers, the preparation functions, the checkers, the reference module, the
   dense functions and the display all take a jet shape and jets of any number of variables. What Part 1 owes
   for several variables is correctness (tested against the definitions) and the right interface; speed for
   several variables is Part 2. Claude's proposal for the structure, put to the maintainer on 2026-11-17 and
   recorded in `TARGET_DECISIONS.md` (the note "One jet group, of any number of axes", and what follows it), is
   the starting point: work it out fully, test it against the code, and list as open whatever it leaves open or
   gets wrong. Its main points: one concept, the jet shape (a tuple of maximum orders, one per jet variable; the
   empty tuple for no jets), in place of the integer `order`; **the inputs are jets** (the sample's vectors carry
   the jet axes, of sizes of their own, and `pp` leaves every signature; a helper builds the common affine
   input); every array of the type of `z` (data, residual, weight, block sums) has the jet axes where the one
   order axis is today; the dual jet turns every jet axis; each variable of the engine has a declared jet shape,
   the input's or the full one. To be worked out: the display of a table with several order axes; what the
   consumer's calls become.
   **Decided by the maintainer since (2026-11-17, 14:20 to 14:44; log entries 118 to 120), and binding on the
   plan:**
   - One `ww` plus `order`, with a helper that builds the input jet from inputs of the `(ww, pp)` type.
   - An input may have any number of slots along a jet axis: two for a line, more for a curve whose higher
     coefficients are free. `order` stays an argument and is never inferred from the input.
   - **The entries kind is of order 0 only.** It has no jets: its functions take an index and no `order`, and an
     oracle of the entries kind has no jet shape. A helper turns an index into one-hot vectors, so that what
     `entries_derivatives` computed is written as apply with the input jet `(one_hot(index), p)`. The entries
     parts of P2 and P3 lose their `order`, and the engine slices carry jets for two kinds, not three.
   - **The letter of the jet group is `J`.** `GroupLengths.T` becomes `GroupLengths.J`; `T` and `A` leave the
     contraction strings.
4. **What is rebuilt of P1 to P3.** `jet_contract` and its tests; the 41 strings (which merge once `A` is the
   jet group; give the new table of strings, derived, not guessed); the expected-failure marker for Part 2;
   `GroupLengths`; the reference module's and `jets.py`'s wording and shapes (`A = (2,)`); anything else. Use
   the inventory script and your own searches; give counts.
5. **What changes in P5 to P20**: the engine's table of variables (each variable's jet shape: the full one, or
   the input's), the start values of the sweeps, the step functions' strings, the dual jet and the turn, the
   readouts, the uniform slice, the tests against the reference module, the documentation (the page on grouped
   contractions gets a table of the letters; `docs/contributor/deferred_and_rejected.md` was to get an entry for
   jets in several variables).
6. **The algorithm for the general case in Part 1**, only as far as the plan needs: a simple correct one (the
   product applied along each jet axis in turn, since the ring is a tensor product), what it costs, and how the
   cases the library uses keep today's cost (the scan with one jet carried and shifted; two contractions for an
   affine partner; the readout). The tests that pin both. No design of fast algorithms for several variables:
   that is Part 2, and the plan says so in its seams.
7. **The tests**: the definition is polynomial multiplication, so the reference is independent arithmetic on
   coefficient arrays (for example `numpy.polynomial` in one variable, explicit multi-index sums or an FFT in
   several); exact structural tests; strictness; dtype; jit (one program size at every order where the scan is
   used); sharding of the sample axis; the expected-failure marker kept or restated.
8. **What it does to the decisions on record**: row 5.4 (the groups, the record), row 2.9 (the two forms over
   the orders), rows 2.1 to 2.4 (the order axis, the dual jet), the definition of `A` recorded on 2026-11-17,
   `SAMPLE_VARIABLES.md` section 7 (shapes). List the amendments needed; do not edit those files (Claude does).

## Rules for both agents

- Check every statement about the code against the code; name files and functions; recount what you rely on.
- Write plainly: what will be done and how, without the history of how it was arrived at. Short sentences.
- The writer changes `PART1_PLAN.md` and writes `reviews/jet_group_plan/ACCOUNT.md`; the critic writes
  `reviews/jet_group_plan/REVIEW_r<n>.md`. Change no other file. Do not commit. Do not touch
  `/home/claude/t3polynomial` (read only; do not run its code). Do not put anyone's email address in any file.
- Do not run the whole test suite or the gate (the machine has 2 cores and 7 GB, and the writer and the critic
  never run tests at the same time); small scripts of your own in `/tmp` to check a mathematical statement or a
  statement about the code are fine and expected.
- The log entries 116 to 120 of `REVIEW_DECISIONS.md` are the maintainer's decisions on this, in order; where
  this brief and they differ, they win. Read them.
