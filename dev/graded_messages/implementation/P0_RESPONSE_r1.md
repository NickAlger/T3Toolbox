# P0: response to the review, round 1

Implementer, 2026-11-16. Every finding is accepted. Nothing outside `dev/graded_messages/implementation/` was
changed; the full suite was not rerun (the baseline stands, the reviewer reproduced it).

## The findings

**1. `stop` removes the lock whatever run holds it. Accepted, fixed.** `stop LOGDIR` now removes the lock only
when the lock names the same run as `LOGDIR/PID` (same pid, start time and boot id). Tested: with a detached run
going in one directory, `stop` on an empty directory and on a directory of a refused run both print "no run is
going ... nothing was signalled" and the lock is still there; `stop` on the run's own directory ends it and
removes the lock.

**2. A run is identified by a bare pid. Accepted, fixed.** `LOGDIR/PID` and the lock now hold
`<pid> <start time from /proc/<pid>/stat> <boot id> <LOGDIR>`. One function, `run_state`, decides for `wait`,
`stop`, the lock check and `--detach`: a run is `alive` only if the boot id is this boot's and the process with
that pid has that start time; anything else is `dead` (or `orphans`, see 7). `stop` signals only in the states
`alive` and `orphans`. Tested with a live unrelated process as the victim (a `sleep` leading its own group), in
four forms of a stale file, each put into both `LOGDIR/PID` and the lock: a bare pid (the old format); the live
pid with a wrong start time; the live pid and its true start time with another boot id; and pid 1. In every
form: `wait` exits 2 ("not running and did not finish"); `stop` prints "nothing was signalled" and the victim is
alive afterwards; a new run is not refused, passes, and leaves no lock.

**3. Star patterns in backticks are not listed. Accepted, fixed.** The lookbehind no longer excludes a backtick.
Markdown emphasis (`*word*`, `**word**`) is skipped so that it is not taken for a pattern. The table now has
`*_from_sweep` 9 times (the reviewer's count), `*_model` 4 times (the reviewer's 6 less the 2 in
`docs/release_notes.md`, which is no longer searched, see 4), `*_derivatives` 8 times (6 in backticks and 2
bare), `ut3_*` 10 times. `P0_INVENTORY.md` is regenerated.

**4. The root `README.md` is not searched. Accepted, as decided.** `README.md` is in the searched files (12
occurrences: 5 in fenced code, 7 in prose). `CHANGELOG.md` and `docs/release_notes.md` are left out on purpose,
and `docs/release_notes.md`, which was searched before because it lies under `docs/`, is now taken out to match
P20's final search. The script's docstring and the note say so and why, and the summary ends with a table of
their counts (`CHANGELOG.md` 159 occurrences of 63 names on 91 lines; `docs/release_notes.md` 9 of 6 on 8). The
plan gap is recorded in the handover, section 5, item 2b. The file count is still 189 (one in, one out); the
total is 7,390 occurrences. Every count checked against the plan is unchanged (the same query script gives
byte-identical output before and after).

**5. An identifier after a non-ASCII character loses its kind. Accepted, fixed.** The syntax tree's columns
(UTF-8 bytes) are converted to character columns on any line that is not ASCII. Tested on the reviewer's two
lines: `x.probe(ww)` is now `call` with binding `obj`, `compute_xi(1)` is `call`. One remainder is stated in the
code: the extent of a string literal inside a doctest example is still measured in bytes, so on such a line it
can be off by a few columns; it affects only the kind `str` in doctests.

**6. `doctest-pages` passes when a glob matches nothing. Left as CI's behaviour, as instructed.** The step
already printed the number of files; `P0_BASELINE.md` now states the number to expect (48 at P0, 47 after P13
removes `docs/chunking.md`) under "Known limits", and the error text of `ls` goes to
`LOGDIR/doctest-pages.ls-errors` and no longer to the terminal.

**7. Other behaviours of the gate script.**
- *`-n two` runs serially with a PASS.* Fixed: `-n` must be a whole number, 1 or more, else exit 2 before
  anything runs. Tested with `two` and `0`.
- *`kill -9` of the gate leaves its step running.* Fixed in the way suggested. Every run, in the foreground too,
  is now started as the leader of its own session, and `timed.py` no longer puts the step in a session of its
  own, so the gate, `timed.py`, the step and its workers are one process group. When the gate process is gone
  and the group is not empty, `run_state` answers `orphans`: the lock refuses a new run and says which directory
  to stop, `wait` says so and exits 2, and `stop` ends the group and frees the lock. Tested: after `kill -9`,
  5 processes left; `wait` exit 2 with the hint; a new run refused; after `stop`, 0 processes and no lock.
  Nothing ends the step unasked; that is stated in the header and in `P0_BASELINE.md`.
- *`--detach` onto a live run's directory deletes its `PID` file.* Fixed: refused before anything is touched.
  Tested: exit 2, the `PID` file still there, the run undisturbed.
- *No time limit.* Added, off by default: `-t SECONDS`, implemented in `timed.py` (it ends the step's whole
  process tree, SIGTERM then SIGKILL after ten seconds). Tested with `-t 5` on `tests/test_dispatch.py` with two
  workers: FAIL, "stopped at the time limit of 5s.", no process left.
- *The default invocation was never run as one; `slow` unexercised.* Not changed, both stated in
  `P0_BASELINE.md` ("Known limits" and "The slow tier").
- A consequence of the new session: a foreground run no longer receives the terminal's Ctrl-C directly, so the
  outer process passes SIGINT and SIGTERM on to the run's group. Tested with SIGTERM on a foreground run with two
  workers: no process and no lock left.

**8. Statements in the notes. Accepted, corrected.**
- 616, not 622: I recounted with the syntax tree (strings outside docstrings that are exactly one of the six
  kind names) and get 616, the reviewer's number. My 622 came from a text match that also took quoted kind names
  inside longer lines. Corrected in the handover and in the inventory note.
- The star imports: the docstring and the handover now say that 47 library modules have
  `from t3toolbox.backend.common import *` (recounted: 47) and that it has no effect, since `common` defines none
  of the names.
- `from t3toolbox.backend import apply as bapply` is now kind `import-module`.

## What was tested after the changes

All in scratch directories under `/tmp/p0r/`, one run at a time.

| What | Result |
| --- | --- |
| `quick tests/test_import_graph.py`, foreground | PASS, 5 passed, 2 subtests; exit 0; no lock left |
| `doctest-modules doctest-quickstart doctest-pages` in one run | PASS: 241 passed; 1 file; 48 files; exit 0 |
| `-n two`, `-n 0` | exit 2 with a message |
| a detached run; a second run; `--detach` onto its directory; `stop` on two other directories; `wait` 3 s; `stop` on it; `wait` | refused (exit 2); refused (exit 2), `PID` kept; lock kept both times; exit 3; stopped, lock gone, no process; exit 2 |
| the four stale forms of finding 2, in `PID` and in the lock | `wait` exit 2; `stop` signals nothing, victim alive; a new run passes |
| `kill -9` of the gate process | detected by `wait` and the lock; `stop` cleans up |
| `--detach examples`, then `wait` (the per-example path through the changed `timed.py`) | PASS, 14 examples, 0 failed, 211 s |
| `-t 5` on a long test file | FAIL at the limit, nothing left |
| SIGTERM to a foreground run | nothing left, no lock |
| `inventory.py`: the reviewer's two non-ASCII lines; a module import with an alias; a star pattern in backticks beside `**bold**` and `*emphasis*` | `call`, `call`; `import-module`; the pattern listed, the emphasis not |
| `inventory.py` output against the tables in `P0_INVENTORY.md` | identical |

## Files changed in this round

`run_gate.sh`, `timed.py`, `inventory.py`, `P0_INVENTORY.md` (regenerated, header updated), `P0_BASELINE.md` (the
restart section, a new section "Known limits of the gate script"), `P0_HANDOVER.md` (the script's description,
the tests, section 5), and this file.
