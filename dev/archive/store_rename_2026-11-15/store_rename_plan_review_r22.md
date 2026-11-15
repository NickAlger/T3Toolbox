# The store rename plan — critic review, round 22 (draft 22, confirmation)

Reviewed: the draft-22 diff against `c8e80bd6`, which records Nick's rule "Tucker variations are V and TT variations
are H. The D U L R are for the frames."

**Sound.**
- The rule is stated once in §3 and once at the head of §1.2a. Its derivative clause matches the rows already agreed:
  a gradient with respect to a frame core takes that core's letter (`dD_tildes`, `Ddot`). The notes' `δD_i` is the
  coordinate `V_i` carries, which is T4S's δU on the Tucker frame core.
- K17 is mechanical and binding-aware. It flags any name bound from a variation (`tucker_variations`, `tt_variations`,
  `V`, `H`, `tkv`, `ttv`, a `.variations` field, or the matching element of a `zip`) that is mapped to a bare frame
  letter. It runs at R1 (report) and R2, and it leaves the derivative forms alone.
- The re-check changes no row, and its seven zip matches bind the store, not the variation. I checked
  `test_t3_orthogonalization.py:123`'s `V, Oc` last round, and `tv_operations.py:136`'s
  `for dB, U in zip(tucker_variations, up_tucker_cores)` binds `U` to the store.
- The existing variation locals spelled with a point's letters are held for Nick, with proposals, and are not
  renamed. That fits §0's "anything not on the list is asked, not done". These are `tv_operations.py`'s `dB`/`dG`,
  `t3_linalg.py:683–694`'s and `utv_operations.py:608–615`'s `dB`, and `probing.py:535–545`'s `B, dB`.

No MAJOR or MINOR finding.

VERDICT: AGREE
