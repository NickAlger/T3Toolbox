# The store rename plan — critic review, round 13 (draft 14)

Reviewed: draft 14, the changed parts. These are D-1 final (§0 item 5, §3.2, the A1 shifts row, the σ̂ row), §1.5
items 1, 8, 9 and 13, K7, K8, the `-ward` travel words (§3.3), and the seed-list split.

**r12 Q1, resolved.**
- Math writes `sh`/`shᵀ`. Code writes `shift`/`shift_t`.
- The four shift functions become `_shift`, `_shift_t`, `_shift_pair` and `_shift_t_pair`.
- The σ̂ locals become `sig_hat`/`sig_hat_d`, function-scoped, at the counts I measured (19/11/3). The shape locals
  `sh` stay.
- All six new names are free in `t3toolbox/` and `tests/` (checked).
- K7's `\bsh(?:ᵀ|\^)` cannot hit a shape local, and it bans the old function names.

**r12 Q2, resolved.**
- K8 now matches `O` only in frame-core contexts.
- §1.5 item 13 lists the other meanings: the oracle alias, the order count, `-O`, I/O, author initials and big-O.
- I checked the new patterns against those sites:
  - the tuple pattern needs capital members, so `(O, nW, -1, Nm)` does not match;
  - `O.shape` does not match `O.S(`/`O.misfit`.

**Also sound.**
- The `-ward` travel words are enumerated by site, and R1's grep rerun is authoritative.
- The theta sites are on §3.3's list.
- The seed list has one block per diagram.

No MAJOR or MINOR finding remains.

VERDICT: AGREE
