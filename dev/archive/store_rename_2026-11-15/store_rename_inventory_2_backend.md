# Store rename inventory 2 of 3: the backend (`t3toolbox/backend/*.py`)

2026-11-10, T3Toolbox HEAD `b272e5de`. Inventory for the BREAKING store rename decided in `dev/HANDOFF.md` (top block,
"DECISION (Nick 2026-11-10): THE STORE RENAME"); first measurement `dev/up_down_naming_blast_radius.md`. Partition:
all 43 modules of `t3toolbox/backend/` (28,607 lines). Tests, frontend and docs are inventories 1 and 3; the few test
rows the brief singled out (`update=` callers, the contraction HISTORICAL table) are cross-referenced in §10.

**The decision, as applied here.** A store is named for the side of the edge its cores occupy: the UP store = the
TT-shaped cores (old `down`, old letter `O`; up-orthogonal), the DOWN store = the Tucker-factor cores (old `up`, old
letter `U`; down-orthogonal). Words flip; the tuple keeps its physical order, now read `(down, up, left, right)`. No
letters in the API: `update='ULRO'` becomes words; the widths `nU, nD, rL, rR` become `n_down, n_up, r_left, r_right`
(old `nU` = Tucker-store width = `n_down`; old `nD`/`nO` = TT-store mode width = `n_up`). Letters stay in diagrams and
equations as U = up (TT-shaped), D = down (Tucker) unless a severe conflict exists -- conflicts are REPORTED in §5, not
resolved. Orthogonality adjectives in method names stay.

**How this was produced.** Every backend file was tokenized; every identifier, comment, docstring line and string
literal with a store word, a width symbol, a store letter or an `update` letter was collected (2421 lines). A generator
applied the rules of §1 as a simultaneous token map (never sequential sed: same-spelling names swap), every proposed
line was read, and 62 lines got hand overrides (§4). The generator and its overrides live in the session scratchpad,
not in the repo; §9 is their complete output. Proposals are text, not run: the implementer re-derives them with a
tokenizer-level rename and the suite/doctests/`-W` docs build check the code half.

## 0. Summary

| category | lines (rows in §9) | token occurrences changed | meaning |
|---|---|---|---|
| STORE-WORD | 805 | 1097 | up/down as a store or edge name: identifiers, keywords, string literals, prose -- FLIPS |
| WIDTH | 944 | 1876 | nU/nD/nO/rL/rR (and kU/kD/kL/kR, k1*, mU.. masks) in frame contexts -> n_down/n_up/r_left/r_right |
| WIDTH-GENERIC | 86 | 163 | rL/rR/nU naming a plain T3/TT core bond or Tucker rank, not a frame store -- optional (decision) |
| CORE-LETTER | 673 | 1103 | U/O/D (and letter-suffixed locals) naming frame cores in code or equations -> new letters |
| LETTER-API | 37 | 37 | the update letters 'U','L','R','O' / 'UL','RO','ULRO','UOLR' -> store words |
| ADJECTIVE | 110 | keep: 58 | up/down-orthogonal, up/down SVD/unfolding/matricization -- STAYS (backwards ones listed in §7) |
| UNRELATED | 164 | keep: 157 | English ('up to', 'up front'), jet shifts (_shift_up, U^r), T4S message directions (upward, up-index) -- STAY |
| CONTRACTION | 0 | 0 | helper names / subscripts carrying store words or letters -- none (§6) |

Rows: 2421 lines in 38 files; 2197 change, 224 are conscious keeps. A row may carry several categories. Token counts
are position-wise token replacements (hand-edited lines count approximately). Flags in §9: `HAND` 69 (mechanical swap wrong or
incomplete -- use the full text given), `C1` 35 / `C2` 51 / `C3` 7 / `C4` 11 (letter conflicts, §5), `SILENT` 5 (same spelling,
other meaning, nothing fails), `API` 16 (public keyword), `HAZARD` 2 (bitwise-stability traps), `AMBIG` 7 (reading unclear; keep proposed).

Biggest files: `sampling_derivatives.py` 537, `probing.py` 283, `padded_frames.py` 225, `uniform_padded_frames.py` 192, `utv_operations.py` 138, `fv_operations.py` 120, `ufv_conversions.py` 119, `tv_operations.py` 107.

## 1. Rules behind every mechanical proposal

- **M1 store identifiers** (exact token, simultaneous, code AND prose mentions): `up_tucker_cores`→`down_tucker_cores`,
  `down_tt_cores`→`up_tt_cores` (+ `_dot`, `_supercore`, `masked_*`), and the SWAPS `up_ranks`↔`down_ranks`, `up_mask`↔`down_mask`,
  `variations_up_mask`↔`variations_down_mask`, `var_up_mask`↔`var_down_mask`, `up_sc`↔`down_sc`, `up_cores`↔`down_cores`,
  `up_supercore`↔`down_supercore`, `up_weights`↔`down_weights`, `up_weight_supercore`↔`down_weight_supercore`, `up_inds`↔`down_inds`,
  `up_r`↔`down_r`, `up_out`↔`down_out`, `up0`↔`down0`, `vup`↔`vdown`, `mup`↔`mdown`, bare locals `up`↔`down`, `_down`→`_up`;
  private functions `_up_message`↔`_down_message`, `_up_edge`↔`_down_edge`, `_u_up_messages`↔`_u_down_messages` (named for the
  edge/store they serve: leaf→core holds the Tucker = DOWN store, core→leaf the TT-shaped = UP store). Full list: §8.
- **M2 prose store words**: bare `up`/`down` (and `Up`/`Down`, `up-core`, `up-Tucker`, `(up, left)`) flip simultaneously,
  EXCEPT the protected spans of M7. Listings `(up, down, left, right)` become `(down, up, left, right)` -- the physical order
  read in the new words, as the decision says.
- **M3 widths**: `nU`→`n_down`, `nD`/`nO`→`n_up`, `rL`→`r_left`, `rR`→`r_right`, `kU`→`k_down`, `kD`→`k_up`, `kL`→`k_left`,
  `kR`→`k_right`, `k1L`→`k1_left` …; digits stay attached (`nU0`→`n_down0`, `rL1`→`r_left1`), an index letter gets an
  underscore (`nUi`→`n_down_i`, `kDj`→`k_up_j`, `rR_d`→`r_right_d`, `rL1i`→`r_left1_i`), a parenthesised index stays as is
  (`rR(i+1)`→`r_right(i+1)`); `nU_pad`→`n_down_pad`, `sum_nD`→`sum_n_up`. Masks/slots/selection locals in
  `uniform_padded_frames.py` use words too (`mU0`→`m_down0`, `pU`→`p_down`, `iU`→`i_down`, `WU`→`W_down`). Spelling is a plan
  choice; the meaning swap (old nU → n_down) is not. Generic T3 bonds (`rL, n, rR = G.shape`) are WIDTH-GENERIC (§5, C5).
- **M4 letters (frame stores only)**: old `U` (Tucker factor) → `D`, old `O` (TT-shaped) → `U`, simultaneously; frame tuples
  `(U, O, P, Q)`→`(D, U, P, Q)`, `(U, O, L, R)`→`(D, U, L, R)`, frame-letter pairs `(U, P)`/`(U, L)`→`(D, P)`/`(D, L)`; old `D` in
  `fv_operations`/`fv_conversions` (= the old DOWN core) → `U`. KEPT as T3 letters (C1): `(U, G)`, `(U, G, G, G)`, `Udot`, `dU*`,
  `U_g`. Applied to code locals with per-file/range maps (listed in §8). The padded-frame coordinate products `D, D_L, D_R,
  DL, DR, DU, DUs` are renamed FIRST to `E, E_L, E_R, EL, ER, ED, EDs` (C2) -- otherwise the swap collides.
- **M5 update words**: `update: str = 'ULRO'` → `update: typ.Tuple[str, ...] = ('down', 'up', 'left', 'right')`; colours
  `('UL', 'RO')` → `(('down', 'left'), ('right', 'up'))`; `'U' in update` → `'down' in update`, `'O'` → `'up'`, `'L'` → `'left'`,
  `'R'` → `'right'`; the `'UOLR'` letter column of the transport check disappears (`store not in update`); a bare `str` must be
  rejected (`set('down')` is a set of characters). Tuples of str stay hashable, so `_STEP_STATICS` / jit statics work unchanged.
- **M6 string literals**: `_STORE_NAMES`, `_SEED_TAGS` keys, `_seed_rows`/`_store_seeds` store arguments and every error
  message naming a store flip like prose. `_SEED_TAGS` KEEPS ITS INTEGERS PER PHYSICAL STORE: `{'down': 0, 'up': 1, 'left': 2,
  'right': 3}` (Tucker factor stays 0) -- otherwise every `padded_seed`/`cone_seed` stream changes (HAZARD).
- **M7 keeps**: adjectives and the unfolding family (`down-orthogonal`, `up_svd`, `up unfolding`, `up-matricization`, `TT-up SVD`,
  `Tucker down-orth`); T4S message directions (`upward/downward edge variables`, `var-upward`, `adjoint-up lift`, `up-index`,
  `down eta`); English (`up to`, `up front`, `written down`, `drill down`, `pair up`, `looked up`, `blow-up`, `(d-1) down to 0`);
  jet shifts (`_shift_up*`, `_shift_down*`, `U^r`, `U x`, `D^r`, `D x`). Index letters inside contraction subscripts stay (§6).

## 2. (a) The compatibility surface: backend keywords and values that change spelling or meaning

Public = in the module's `__all__` (an `ast` walk of every function signature). Positional order never changes.

| function (module) | keyword / value today | new | kind |
|---|---|---|---|
| `probing.compute_xi`, `compute_deta_tilde` | `up_tucker_cores` | `down_tucker_cores` | rename (loud) |
| `probing.compute_deta_tilde_tangent` | `up_tucker_cores`, `up_tucker_cores_dot` | `down_tucker_cores`, `down_tucker_cores_dot` | rename (loud) |
| `probing.compute_eta`, `compute_sigma`, `compute_tau`, `compute_dxi_tilde` | `down_tt_cores` | `up_tt_cores` | rename (loud) |
| `probing.compute_dxi_tilde_tangent` | `down_tt_cores`, `down_tt_cores_dot` | `up_tt_cores`, `up_tt_cores_dot` | rename (loud) |
| `sampling_derivatives.compute_sigma_jets`, `compute_tau_jets`, `compute_dxi_tilde_jets` | `down_tt_cores` | `up_tt_cores` | rename (loud) |
| `sampling_derivatives.compute_deta_tilde_jets` | `up_tucker_cores` | `down_tucker_cores` | rename (loud) |
| `sampling_derivatives.compute_deta_tilde_tangent_jets` | `up_tucker_cores`, `up_tucker_cores_dot` | `down_tucker_cores`, `down_tucker_cores_dot` | rename (loud) |
| `sampling_derivatives.compute_dxi_tilde_tangent_jets` | `down_tt_cores`, `down_tt_cores_dot` | `up_tt_cores`, `up_tt_cores_dot` | rename (loud) |
| `ranks.frame_has_minimal_ranks(shape, up_ranks, down_ranks, left_ranks, right_ranks, sharing)` | params 2-3 `up_ranks`, `down_ranks` | `down_ranks`, `up_ranks` (same positions) | **SILENT swap** for keyword callers |
| `ufv_masking.ufv_make_frame_masks(up_ranks, down_ranks, left_ranks, right_ranks, nU, nD, rL, rR)` | params 1-2 `up_ranks`, `down_ranks` | `down_ranks`, `up_ranks` | **SILENT swap** for keyword callers |
| `ufv_masking.ufv_make_frame_masks` (same) | params 5-8 `nU`, `nD`, `rL`, `rR` | `n_down`, `n_up`, `r_left`, `r_right` | rename (loud) |
| `ufv_conversions.t3frame_to_ut3frame`, `t3variations_to_ut3variations`, `t3frameweights_to_ut3frameweights` | `nU=`, `nD=`, `rL=`, `rR=` | `n_down=`, `n_up=`, `r_left=`, `r_right=` | rename (loud); frontend `from_t3frame(nD=)` etc. forward to these |
| `padded_frames.tv_align_padded_frame` | `update: str = 'ULRO'` | `update: Tuple[str, ...] = ('down', 'up', 'left', 'right')` | type + values (old strings rejected loudly once a bare str is refused) |
| `uniform_padded_frames.utv_align_padded_frame` | `update: str = 'ULRO'` (static) | as above | type + values |
| `uniform_padded_frames.ufv_compact_frame` | `level_widths` documented `(WU, WD, WL, WR)` | name kept; documented `(W_down, W_up, W_left, W_right)` | doc only (physical order) |
| `ufv_masking.ufv_frame_widths` (returns) | documented `(nU, nD, rL, rR)` | `(n_down, n_up, r_left, r_right)` | doc only |
| `ranks.compute_orthogonal_representation_ranks` (returns) | labels `up_tucker_ranks`, `down_tucker_ranks`; symbols `nU`, `nD` | `down_tucker_ranks` (n_down), `up_tucker_ranks` (n_up) | doc only; order unchanged |
| every frame 4-tuple / uniform 6-tuple in backend signatures | `(up, down, left, right[, shape, masks])` | same data, read `(down, up, left, right, ...)` | doc only |
| error messages (`padded_frames`, `uniform_padded_frames`, `geometry`, `fv_operations`) | store names in text, e.g. `'old_frame down[1] has shape'`, `'start frame: expected (up, down, left, right)'` | flipped words | tests assert on these strings |
| `padded_frames._SEED_TAGS` (private, but defines the public seed streams) | `{'up': 0, 'down': 1, ...}` | `{'down': 0, 'up': 1, ...}` | integers MUST stay with the physical store |
| `optimizers.padded_start` doctest (frontend attribute) | `F.up_ranks` | `F.down_ranks` | **SILENT**: prints the same `(3, 4, 3)` either way at this structure |

No `padded_*` / `cone_*` optimizer keyword carries a store word or letter (`optimizers.py` 886-891, 1338+: checked); their
docs need no change. Private functions with store-named parameters (renamed, positional callers only): `apply._apply_from_xis`,
`apply._apply_transpose_adjoint`, `entries._onehot_vectors`, `sampling_derivatives._apply_derivatives_from_jets`,
`_apply_derivatives_transpose_from_jets`, `_entries_ambient_jets`, `padded_frames._frame_core_shapes(up_ranks, down_ranks)`
(→ `(down_ranks, up_ranks)`, positional), `padded_frames._aligning_pass(update)`, `uniform_padded_frames._utv_align_kernel(update)`.

## 3. (b) Docstring/comment sentences that state the convention (quoted; each must read right afterwards)

| where | today | becomes |
|---|---|---|
| padded_frames.py:7-9 | "an old frame in the leading slots of every store (the four families of frame cores, up / down / left / right; :doc:`/frame_variations`, "Four rank stores")" | M2: "down / up / left / right" (physical order) |
| padded_frames.py:285-288 | "Every directed edge of the T3 tree (leaf -> core, core -> core both ways, core -> leaf) gets a new completion" | arrows unchanged; leaf→core is now the DOWN edge, core→leaf the UP edge (names of `_up_edge`/`_down_edge`, `_up_message`/`_down_message` swap) |
| padded_frames.py:376, 436, 457, 530 | "The message of the edge core -> leaf" / "leaf -> core" / "The edges leaf -> core of one sharing group" / "The edge core -> leaf" | unchanged text; the functions holding them swap names |
| padded_frames.py:554 | "stores updated: 'U' up, 'L' left, 'R' right, 'O' down" | M5 (§4) |
| ranks.py:644-647 | "1. Tucker down-orthogonalization: ``up_i = min(n_i, N_i)``; 2. … ``left_{i+1} = min(left_i * up_i, r_{i+1})``; 3. … ``right_i = min(left_i, up_i * right_{i+1})``; 4. the down cores: ``down_i = min(up_i, left_i * right_{i+1})``" | the store symbols swap (`down_i = min(n_i, N_i)` … `4. the up cores: up_i = min(down_i, …)`); "down-orthogonalization" stays |
| ranks.py:740-742 | "Requires the left/right and up/down rank stores to agree, and the up/left ranks to equal the minimal ranks" | "down/up … down/left" |
| fv_operations.py:97-98 | "``(nD_i, N_i)`` per Tucker variation and ``(rL_i, nU_i, rR_i)`` per TT variation" | "(n_up_i, N_i) … (r_left_i, n_down_i, r_right_i)": the Tucker VARIATION rides the UP width |
| fv_operations.py:201 | "up_tucker U_i (all i), outer/down D_i (all i)" | HAND: "down_tucker D_i (all i), outer/up U_i (all i)" (old D = old down core) |
| fv_operations.py:313-318 | "**Tucker factors** (``up``) … set ``U_g = Q.T`` … **TT cores** (``down`` / ``left`` / ``right``): identity prefixes" | "(``down``) … ``D_g = Q.T`` … (``up`` / ``left`` / ``right``)" |
| fv_operations.py:640-641; ufv_operations.py:587-588 | "``down`` -> V's ``nD`` leg; ``up``/``left``/``right`` -> H's ``nU``/``rL``/``rR`` legs" | "``up`` -> V's ``n_up`` leg; ``down``/``left``/``right`` -> H's ``n_down``/``r_left``/``r_right`` legs" |
| fv_operations.py:666 | "Ranks: up<->H.nU (axis -2), down<->V.nD (axis -2), left<->H.rL (axis -3), right<->H.rR (axis -1)." | "down<->H.n_down …, up<->V.n_up …" |
| fv_operations.py:726-730; ufv_operations.py:665-671 | "``up = down = tucker_weights`` … (where the down/complement rank ``nD`` equals the Tucker rank ``nU`` …) A non-minimal tangent has ``nD < nU`` and would mismatch the down family" | "``down = up = …`` … the up/complement rank ``n_up`` equals the Tucker rank ``n_down`` … ``n_up < n_down`` … mismatch the up family" |
| ufv_operations.py:276-277, 293-294 | "over the four senses (up ``U``, down/outer ``O``, left ``L``, right ``R``)"; "# up: rows orthonormal over mode", "# down/outer: mode orthonormal over bonds" | "(down ``D``, up/outer ``U``, …)"; "# down: rows …", "# up/outer: mode …" |
| probing.py:640-651 (= 1215-1226) | "up_tucker_cores. len=d. U_xo U_yo = I_xy … down_tt_cores. len=d. O_ixj O_iyj = I_xy … up_tucker_supercore … up orthogonal elements … down_tt_supercore … down orthogonal elements … frame order = T3Frame.data = (up, down, left, right) = (U, O, P, Q)" | the backwards adjectives become right by the swap (HAND, §7); letters → `D_xo D_yo`, `U_ixj U_iyj`, `(D, U, P, Q)` |
| tv_operations.py:392-393 | "the canonical conditions (U row-orthonormal, L/R left/right-canonical, O outer-orthonormal)" | "(D row-orthonormal, …, U outer-orthonormal)" |
| tv_operations.py:400-401 | "``dG_i = <U_i, core_env_i>`` (the TT variation) and ``dU_i = <O_i, core_env_i>`` (the Tucker variation; ``O`` = the outer/down cores)" | "<D_i, …> … dU_i = <U_i, …> (…; ``U`` = the outer/up cores)" -- C1: `dU` keeps the T3 letter |
| tv_operations.py:770; utv_operations.py:448, 538 | "each tucker variation orthogonal to its up-core" | "… its down-core" |
| ufv_masking.py:33; ufv_conversions.py:96 | "The up/down masks pass through" / "the variation masks reuse the up/down masks" | "down/up" |
| ufv_masking.py:58 | "the Tucker width capped by N, the down width by r*r" | "the up width by r*r" |
| utv_operations.py:213-215 | "``Udot`` has the UP width, so the variation block is rebuilt at height ``nU`` … group block masks are the frame's UP mask" | "DOWN width … ``n_down`` … DOWN mask" (C1: Udot kept) |
| utv_operations.py:315 | "Tucker axis [U \| dU] = [up \| down]; the TT bond is [Q \| P] = [R \| L]" | "[D \| dU] = [down \| up]" (C1) |
| geometry.py:143-145 | "the Tucker variation has the ``up`` supercore's shape") holds only when ``nD == nU`` … differs from the down rank" | "``down`` supercore's … ``n_up == n_down`` … the up rank" |
| uniform_padded_frames.py:47-48 | "up (d,)+C+(nU, N), down (d,)+C+(rL, nD, rR) … block layout: nU = nD, rL = rR" | "down (…n_down, N), up (…r_left, n_up, r_right) … n_down = n_up, r_left = r_right" |
| adjective statements (stay; verified) | t3_orthogonalization.py:38 "Tucker cores down-orthogonal"; ut3_orthogonalization.py:158 "Orthogonalize the Tucker cores (rows orthonormal over the mode index), pushing the remainder up", 190-191 "TT-up SVD … the remainder pushed down into the Tucker core", 215 "Up-orthogonalize the TT cores … pushing the remainder down"; linalg.py:330-334, 463 "up unfolding … up-orthonormal" | unchanged; consistent with the new store words |

## 4. (c) Hand edits: where a mechanical swap is wrong, incomplete or ambiguous

Every `HAND`/`AMBIG` row of §9, with the full proposed text. Classes: backwards adjective (the store convention leaked into an
adjective; the rename makes the word right only if flipped by hand, because the generic rule protects adjectives); both senses
in one sentence; pre-existing typos that the token map would carry into a wrong name; LETTER-API rewrites; MSG words a word
swap would wrongly flip.

| file:line | today | proposed | why |
|---|---|---|---|
| apply.py:215 | ```**Lean ``(xis, mus)`` only** (not the right ``nu`` / down ``eta`` sweeps): the all-modes apply``` | keep | 'down eta' / 'frame down jets' = T4S's DOWNWARD edge variables eta (travel direction, MSG): keep 'down' here although eta contracts the (new) UP store; generated text wrongly flips it -- apply only the width/identifier changes on this line |
| fitting.py:544 | `# the lean sweep's xi / mu are base_point's own (xi through U, mu through L at the manifold frame` | `# the lean sweep's xi / mu are base_point's own (xi through D, mu through L at the manifold frame` |  |
| fitting.py:656 | `# the lean sweep's xi / mu jets are base_point's own (xi through U, mu through L at the manifold frame` | `# the lean sweep's xi / mu jets are base_point's own (xi through D, mu through L at the manifold frame` |  |
| fitting.py:718 | `# the sweep's xi / mu are base_point's own: the manifold frame (U, O, L, R) sweeps xi through U and mu` | `# the sweep's xi / mu are base_point's own: the manifold frame (D, U, L, R) sweeps xi through D and mu` |  |
| fv_conversions.py:58 | `NDArray,  # down_tucker_supercore` | `NDArray,  # up_tt_supercore` | pre-existing typo `down_tucker_supercore` for the TT-shaped supercore; the mechanical map would leave a wrong name |
| fv_conversions.py:60 | `NDArray,  # right_tucker_supercore` | `NDArray,  # right_tt_supercore` | pre-existing typo `right_tucker_supercore` (not a store-word issue; fix in passing) |
| fv_conversions.py:157 | `NDArray,  # down_tucker_supercore` | `NDArray,  # up_tt_supercore` | same pre-existing typo |
| fv_conversions.py:159 | `NDArray,  # right_tucker_supercore` | `NDArray,  # right_tt_supercore` | same pre-existing typo |
| fv_conversions.py:219 | `# Orthogonalize Tucker cores upward to get up_tt_cores U` | `# Down-orthogonalize the Tucker cores to get the down store D (down_tucker_cores)` | was 'Orthogonalize Tucker cores upward to get up_tt_cores U': backwards direction word + wrong identifier; a word swap cannot fix it |
| fv_conversions.py:251 | `# Orthogonalize TT cores downward to get outer_tt_cores O and tucker_variations V` | `# Up-orthogonalize the TT cores to get the up store U (up_tt_cores) and tucker_variations V` | was 'Orthogonalize TT cores downward to get outer_tt_cores O ...': backwards direction word |
| fv_operations.py:201 | `- up_tucker U_i (all i), outer/down D_i (all i),` | `- down_tucker D_i (all i), outer/up U_i (all i),` | was 'up_tucker U_i ..., outer/down D_i': old D = the old down (TT-shaped) core, now U; both letters swap |
| fv_operations.py:315 | ```sign-fixed reduced QR of ``G.T``, and set ``U_g = Q.T``: Haar-distributed orthonormal rows, the same``` | ```sign-fixed reduced QR of ``G.T``, and set ``D_g = Q.T``: Haar-distributed orthonormal rows, the same``` | U_g (the Haar Tucker factor of the zero-point frame = the down store) -> D_g; generated text kept U_g because U_g is protected as a T3 letter elsewhere |
| geometry.py:348 | ```(Tucker up-orth + one left TT sweep, no ``W`` factor -- negligible beside the misfit evaluation it``` | ```(Tucker down-orth + one left TT sweep, no ``W`` factor -- negligible beside the misfit evaluation it``` | backwards adjective: the Tucker cores are DOWN-orthogonalized |
| geometry.py:602 | `frame_data:  typ.Tuple,   # uniform frame .data = (U, O, P, Q, shape, (up, down, left, right) masks)` | `frame_data:  typ.Tuple,   # uniform frame .data = (D, U, P, Q, shape, (down, up, left, right) masks)` |  |
| geometry.py:707 | `frame_data:   typ.Tuple,                          # uniform frame .data = (U, O, P, Q, shape, masks)` | `frame_data:   typ.Tuple,                          # uniform frame .data = (D, U, P, Q, shape, masks)` |  |
| geometry.py:893 | `frame_data:         typ.Tuple,                   # uniform frame .data = (U, O, P, Q, shape, masks)` | `frame_data:         typ.Tuple,                   # uniform frame .data = (D, U, P, Q, shape, masks)` |  |
| optimizers.py:807 | `>>> F.up_ranks, len(movements) <= 10` | `>>> F.down_ranks, len(movements) <= 10` | frontend attribute T3Frame.up_ranks: the Tucker-store widths are now down_ranks. At this (minimal) structure up_ranks would print the SAME (3, 4, 3) -- a stale name passes the doctest silently |
| padded_frames.py:13 | ```one fixed tensor, a truncating pass to the target ranks, then two-colour passes in ``('UL', 'RO')`` pairs. The``` | ```one fixed tensor, a truncating pass to the target ranks, then two-colour passes in ``(('down', 'left'), ('right', 'up'))`` pairs. The``` | the colour pairs 'UL'/'RO' in words: old U = Tucker = new down, old O = TT-shaped = new up |
| padded_frames.py:554 | `update:            str = 'ULRO',                             # stores updated: 'U' up, 'L' left, 'R' right, 'O' down; the rest transported` | `update:            typ.Tuple[str, ...] = ('down', 'up', 'left', 'right'),   # the stores updated (store words); the rest transported` | API: str of letters -> tuple of store words (hashable: jit static). Spelling/type is a plan decision |
| padded_frames.py:560 | `if not update or len(set(update)) != len(update) or not set(update) <= set('ULRO'):` | `if isinstance(update, str) or not update or len(set(update)) != len(update) or not set(update) <= set(_STORE_NAMES):` | a bare str must be rejected: set('down') is a set of characters |
| padded_frames.py:561 | `raise ValueError("update must be a non-empty set of the stores 'U', 'L', 'R', 'O' without repeats; got %r"` | `raise ValueError("update must be a non-empty tuple of the stores 'down', 'up', 'left', 'right' without repeats; got %r"` | message text is asserted by tests ('without repeats', 'update must be') |
| padded_frames.py:581 | `for store, letter, old, new, own in zip(_STORE_NAMES, 'UOLR', (nU0, nD0, rL0, rR0), out, (nU, nD, rL, rR)):` | `for store, old, new, own in zip(_STORE_NAMES, (n_down0, n_up0, r_left0, r_right0), out, (n_down, n_up, r_left, r_right)):` | drop the 'UOLR' letter column; the store word is the update key |
| padded_frames.py:624 | `tuple(nU1[g[0]] - nU0[g[0]] for g in groups), ('U' in update,) * len(groups)))` | `tuple(n_down1[g[0]] - n_down0[g[0]] for g in groups), ('down' in update,) * len(groups)))` | ('D' in update,) -> ('down' in update,)  [generated text shows the interim letter] |
| padded_frames.py:636 | `tuple(XiL[i] for i in js), tuple(rL1[i + 1] - rL0[i + 1] for i in js), ('L' in update,) * len(js)))` | `tuple(XiL[i] for i in js), tuple(r_left1[i + 1] - r_left0[i + 1] for i in js), ('left' in update,) * len(js)))` | ('L' in update,) -> ('left' in update,) |
| padded_frames.py:646 | `tuple(XiR[i] for i in js), tuple(rR1[i] - rR0[i] for i in js), ('R' in update,) * len(js)))` | `tuple(XiR[i] for i in js), tuple(r_right1[i] - r_right0[i] for i in js), ('right' in update,) * len(js)))` | ('R' in update,) -> ('right' in update,) |
| padded_frames.py:655 | `tuple(nD1[i] - nD0[i] for i in range(d)), ('O' in update,) * d))` | `tuple(n_up1[i] - n_up0[i] for i in range(d)), ('up' in update,) * d))` | ('U' in update,) -> ('up' in update,)  [old 'O'] |
| padded_frames.py:687 | `update:     str,                                      # 'ULRO' \| 'UL' \| 'RO'` | `update:     typ.Tuple[str, ...],                      # all four stores \| ('down', 'left') \| ('right', 'up')` |  |
| padded_frames.py:750 | `frame, movement = _aligning_pass(step, frame, hint, project, old, out, sharing, 'ULRO')` | `frame, movement = _aligning_pass(step, frame, hint, project, old, out, sharing, _STORE_NAMES)` | 'ULRO' -> all four store words |
| padded_frames.py:752 | `if alternate:                                               # ('UL', 'RO') pairs while a whole pair fits; the stop` | `if alternate:                                               # (('down', 'left'), ('right', 'up')) pairs while a whole pair fits; the stop` |  |
| padded_frames.py:755 | `for update in ('UL', 'RO'):` | `for update in (('down', 'left'), ('right', 'up')):` |  |
| padded_frames.py:759 | `if not stop and len(movements) < max_passes:           # an odd remainder: one final 'UL' pass` | `if not stop and len(movements) < max_passes:           # an odd remainder: one final ('down', 'left') pass` |  |
| padded_frames.py:760 | `frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, 'UL')` | `frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, ('down', 'left'))` |  |
| padded_frames.py:762 | `else:                                                       # 'ULRO' passes, the stop tested after each (the` | `else:                                                       # all-store passes, the stop tested after each (the` |  |
| padded_frames.py:765 | `frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, 'ULRO')` | `frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, _STORE_NAMES)` |  |
| probing.py:646 | `NDArray,  # up_tucker_supercore. shape=(d, nU, N),      up orthogonal elements` | `NDArray,  # down_tucker_supercore. shape=(d, n_down, N),      down orthogonal elements` | backwards adjective (blast radius §1.4): the rename makes store word and adjective agree |
| probing.py:647 | `NDArray,  # down_tt_supercore.   shape=(d, rL, nO, rR), down orthogonal elements` | `NDArray,  # up_tt_supercore.   shape=(d, r_left, n_up, r_right), up orthogonal elements` | backwards adjective |
| probing.py:773 | `# assembly -- roughly half of tv_probe. entries is apply with the up-index xis obtained by` | keep | 'up-index' = the index of T4S's upward edge variables xi (MSG); it is also the Tucker-store width -- keep, or reword ('xi index') so it cannot be read as a store |
| probing.py:824 | `xis:            typ.Sequence[NDArray],  # frame up-index edge vars, len=d, elm_shape=W+C+(nUi,)` | keep | 'up-index' = the index of T4S's upward edge variables xi (MSG); it is also the Tucker-store width -- keep, or reword ('xi index') so it cannot be read as a store |
| probing.py:1221 | `NDArray,  # up_tucker_supercore. shape=(d, nU, N),      up orthogonal elements` | `NDArray,  # down_tucker_supercore. shape=(d, n_down, N),      down orthogonal elements` | backwards adjective |
| probing.py:1222 | `NDArray,  # down_tt_supercore.   shape=(d, rL, nO, rR), down orthogonal elements` | `NDArray,  # up_tt_supercore.   shape=(d, r_left, n_up, r_right), up orthogonal elements` | backwards adjective |
| sampling_derivatives.py:1479 | `etas:           typ.Sequence[NDArray],  # frame down jets, len=d, elm_shape=(order+1,)+W+C+(nOi,)` | `etas:           typ.Sequence[NDArray],  # frame down jets, len=d, elm_shape=(order+1,)+W+C+(n_up_i,)` | 'down eta' / 'frame down jets' = T4S's DOWNWARD edge variables eta (travel direction, MSG): keep 'down' here although eta contracts the (new) UP store; generated text wrongly flips it -- apply only the width/identifier changes on this line |
| sampling_derivatives.py:1513 | `etas:           typ.Sequence[NDArray],  # frame down jets, len=d, elm_shape=(order+1,)+W+C+(nOi,)` | `etas:           typ.Sequence[NDArray],  # frame down jets, len=d, elm_shape=(order+1,)+W+C+(n_up_i,)` | 'down eta' / 'frame down jets' = T4S's DOWNWARD edge variables eta (travel direction, MSG): keep 'down' here although eta contracts the (new) UP store; generated text wrongly flips it -- apply only the width/identifier changes on this line |
| sharing.py:619 | `>>> raw = (tuple(np.random.randn(O.shape[-2], U.shape[-1])` | `>>> raw = (tuple(np.random.randn(U.shape[-2], D.shape[-1])` | doctest: O -> U and U -> D together (the mechanical O->U alone collides) |
| sharing.py:620 | `...              for O, U in zip(frame_d[1], frame_d[0])),` | `...              for U, D in zip(frame_d[1], frame_d[0])),` | doctest: as 619 |
| t3_orthogonalization.py:7 | ``` ``t3_left/right_orthogonalize``, the up/down core sweeps, the single-core SVD steps they are ``` | keep | 'the up/down core sweeps' names t3_up_orthogonalize_tt_cores / t3_down_orthogonalize_tucker_cores (adjective convention): KEEP, do not swap |
| t3_orthogonalization.py:147 | `NDArray,  # new_Uxo.  up-orthogonal tucker core, C+(x,Ni)` | `NDArray,  # new_Uxo.  down-orthogonal tucker core, C+(x,Ni)` | backwards adjective (store convention): the Tucker core is DOWN-orthogonal |
| t3_orthogonalization.py:171 | `"""Orthogonalize Tucker cores upwards, pushing remainders onto TT cores above.` | `"""Down-orthogonalize the Tucker cores, pushing the remainders up onto the TT cores above.` | was 'Orthogonalize Tucker cores upwards, ...' (backwards; blast radius §1.4) |
| t3_svd.py:187 | `mode) and to every group core's up leg. Note the tolerance acts on the concatenation,` | keep | 'group core's up leg': the leg that becomes rows in the up-matricization (ORTH family) -- keep, or reword to 'mode leg'; a mechanical flip to 'down leg' would be wrong |
| t3_svd.py:312 | ```(a Tucker rank above its mode size, ``n_i > N_i``, is reduced by a Tucker up-SVD first -- the one case``` | ```(a Tucker rank above its mode size, ``n_i > N_i``, is reduced by a Tucker down-SVD first -- the one case``` | 'Tucker up-SVD' is backwards (it calls t3_down_svd_tucker_core): -> 'Tucker down-SVD' |
| t3_svd.py:345 | `if x[0][ii].shape[-2] > x[0][ii].shape[-1]:       # n_i > N_i: only a Tucker up-SVD can reduce it` | `if x[0][ii].shape[-2] > x[0][ii].shape[-1]:       # n_i > N_i: only a Tucker down-SVD can reduce it` | 'Tucker up-SVD' -> 'Tucker down-SVD' (calls t3_down_svd_tucker_core) |
| t3_svd.py:388 | `assigned as ONE array per group and each group core's up leg absorbs its coefficient` | keep | as 187 |
| uniform_padded_frames.py:522 | `update:    str,                       # static: the stores updated; the rest transported` | `update:    typ.Tuple[str, ...],       # static: the stores updated (store words); the rest transported` |  |
| uniform_padded_frames.py:566 | `Y = _u_content_d(g_old, g_new, PhiU[..., Na:], XiU) if 'U' in update else g_old` | `Y = _u_content_d(g_old, g_new, PhiD[..., Na:], XiD) if 'down' in update else g_old` | 'U' in update -> 'down' in update |
| uniform_padded_frames.py:582 | `L1, oL, DL, mvL = _sweep(xscan, _u_left_edge_step if 'L' in update else _u_left_edge_transport_step, zero_edge, (` | `L1, oL, EL, mvL = _sweep(xscan, _u_left_edge_step if 'left' in update else _u_left_edge_transport_step, zero_edge, (` | 'L' in update -> 'left' in update D-map renamed E* |
| uniform_padded_frames.py:586 | `R1, oR, DR, mvR = _sweep(xscan, _u_right_edge_step if 'R' in update else _u_right_edge_transport_step, zero_edge, (` | `R1, oR, ER, mvR = _sweep(xscan, _u_right_edge_step if 'right' in update else _u_right_edge_transport_step, zero_edge, (` | 'R' in update -> 'right' in update D-map renamed E* |
| uniform_padded_frames.py:601 | `Y = _u_content_d(g_old, g_new, PhiO[..., Na:], XiO) if 'O' in update else g_old` | `Y = _u_content_d(g_old, g_new, PhiU[..., Na:], XiU) if 'up' in update else g_old` | 'O' in update -> 'up' in update |
| uniform_padded_frames.py:621 | `update:           str = 'ULRO',                                      # static; as tv_align_padded_frame` | `update:           typ.Tuple[str, ...] = ('down', 'up', 'left', 'right'),   # static; as tv_align_padded_frame` |  |
| uniform_padded_frames.py:627 | `if not update or len(set(update)) != len(update) or not set(update) <= set('ULRO'):` | `if isinstance(update, str) or not update or len(set(update)) != len(update) or not set(update) <= {'down', 'up', 'left', 'right'}:` |  |
| uniform_padded_frames.py:628 | `raise ValueError("update must be a non-empty set of the stores 'U', 'L', 'R', 'O' without repeats; got %r"` | `raise ValueError("update must be a non-empty tuple of the stores 'down', 'up', 'left', 'right' without repeats; got %r"` |  |
| uniform_padded_frames.py:669 | `for store, letter, o, s1, s in zip(('up', 'down', 'left', 'right'), 'UOLR', old, out, own):` | `for store, o, s1, s in zip(('down', 'up', 'left', 'right'), old, out, own):` | drop the 'UOLR' letter column |
| uniform_padded_frames.py:776 | `statics:  typ.Tuple,   # as _aligning_pass, at the working structure, 'ULRO'` | `statics:  typ.Tuple,   # as _aligning_pass, at the working structure, all four stores` |  |
| uniform_padded_frames.py:788 | `) -> NDArray:                # 0-d bool: run another ('UL', 'RO') pair` | `) -> NDArray:                # 0-d bool: run another (('down', 'left'), ('right', 'up')) pair` |  |
| uniform_padded_frames.py:795 | `statics:  typ.Tuple[typ.Tuple, typ.Tuple],   # the 'UL' and the 'RO' passes' statics, at the target structure` | `statics:  typ.Tuple[typ.Tuple, typ.Tuple],   # the ('down', 'left') and the ('right', 'up') passes' statics, at the target structure` |  |
| uniform_padded_frames.py:798 | `"""One ('UL', 'RO') pair; the stop: both movements below tol (the max over the stack; a NaN never stops)."""` | `"""One (('down', 'left'), ('right', 'up')) pair; the stop: both movements below tol (the max over the stack; a NaN never stops)."""` |  |
| uniform_padded_frames.py:808 | `statics:  typ.Tuple,   # the 'UL' pass's statics` | `statics:  typ.Tuple,   # the ('down', 'left') pass's statics` |  |
| uniform_padded_frames.py:811 | `"""The odd remainder's final 'UL' pass."""` | `"""The odd remainder's final ('down', 'left') pass."""` |  |
| uniform_padded_frames.py:833 | `statics:  typ.Tuple,   # the 'ULRO' pass's statics, at the target structure` | `statics:  typ.Tuple,   # the all-store pass's statics, at the target structure` |  |
| uniform_padded_frames.py:894 | `statics = (project, hint, shape, masks, split, (None, None), sharing, 'ULRO')` | `statics = (project, hint, shape, masks, split, (None, None), sharing, ('down', 'up', 'left', 'right'))` |  |
| uniform_padded_frames.py:901 | `state = xwhile(_pairs_cond, ft.partial(_pairs_body, (at('UL'), at('RO'))), state, use_jit)` | `state = xwhile(_pairs_cond, ft.partial(_pairs_body, (at(('down', 'left')), at(('right', 'up')))), state, use_jit)` |  |
| uniform_padded_frames.py:902 | `state = _cond(_single_cond(state), ft.partial(_final_pass, at('UL')), _no_pass, state, use_jit)` | `state = _cond(_single_cond(state), ft.partial(_final_pass, at(('down', 'left'))), _no_pass, state, use_jit)` |  |
| uniform_padded_frames.py:905 | `state = xwhile(_single_cond, ft.partial(_single_body, at('ULRO')), state, use_jit)` | `state = xwhile(_single_cond, ft.partial(_single_body, at(('down', 'up', 'left', 'right'))), state, use_jit)` |  |
| ut3_orthogonalization.py:189 | `) -> typ.Tuple[NDArray, NDArray]:   # (new_tucker = variations V, new_tt = down-orthogonal O)` | `) -> typ.Tuple[NDArray, NDArray]:   # (new_tucker = variations V, new_tt = up-orthogonal U)` | backwards adjective ('down-orthogonal O'): the TT core after a TT-up SVD is UP-orthogonal |
| ut3_sampling.py:374 | ```frame's ``U`` and ``P`` are ``theta`` masked with the frame's ``(up, left)`` masks, so ``data`` must carry those``` | ```frame's ``D`` and ``P`` are ``theta`` masked with the frame's ``(down, left)`` masks, so ``data`` must carry those``` |  |

## 5. Conflicts for the letter decision (reported, not resolved)

**C1 -- a load-bearing old U (the T3 Tucker letter).** The T3's Tucker cores are `U` in T4S and across the library: the
base point `theta = (U, G)`, the corewise frame `(U, G, G, G)`, the raw-core directions `Udot`/`dU`/`U_g`, `xi = U^T w`, the
Section 6.3 substitution. The frame's DOWN store IS a T3's Tucker cores (orthogonal representation, corewise frame), so in
many formulas the new frame letter D and the T3 letter U name the same array side by side, and the new frame letter U (the
TT-shaped store) sits next to a T3 `U`/`Udot`. The proposals keep T3 tuples (`(U, G)`, `(U, G, G, G)`, `Udot`, `dU*`, `U_g`) and
change frame tuples (`(D, U, P, Q)`, `(D, P)`, `(D, L)`), which yields lines such as `(D, U, P, Q) = (U, G, G, G)` (fv_conversions
:32, geometry :509) and `dU_i = <U_i, core_env_i> (… U = the outer/up cores)` (tv_operations :401). 35 flagged rows:
`fitting.py` 16, 544, 545, 656, 657, 718, 719; `fv_conversions.py` 32; `fv_operations.py` 557; `geometry.py` 509; `probing.py` 1457, 1471, 1485, 1721; `sampling_derivatives.py` 672, 859, 864, 886, 2102, 2464, 2483, 2989; `sharing.py` 722; `t3_linalg.py` 717, 770; `tv_operations.py` 232, 234, 401; `utv_operations.py` 213, 315, 320, 382, 391, 538, 555.
In the T4S-algorithm modules (`probing`, `sampling_derivatives`, `apply`, `entries`) the code locals `U` (Tucker core, used for
BOTH T3 probing and frame sweeps: `compute_xi(up_tucker_cores, ww)` is called by `t3_probe` with a T3's cores) and `O` were
mapped U→D, O→U (`Ud`→`Dd`, `Od`→`Ud`, `mO`→`mU`, `On`→`Un`, `dmO`→`dmU`, `dOn`→`dUn`, `Odxi`→`Udxi`) per the decision; `dU`
(Tucker variation, 32 code tokens), `dU_tilde(s)` (40), `Udot` (15) and the public docs `theta_hat = (Udot, Gdot)` were KEPT.
Option for the plan: exempt these modules as a documented T4S correspondence (keep U, O, P, Q there), which removes C1 and C3
in one move.

**C2 -- D already means something else.**
- `padded_frames.py` / `uniform_padded_frames.py`: the coordinate products `D = F^T g_new`, `D_L`, `D_R`, `DL`, `DR`, `DU`, `DUs`
  (code, comments, the `(o_L, D_L)` carry tuples). Proposed: rename to `E*` BEFORE the letter swap (`DU`→`ED`). 51 rows.
- `fv_operations.py:201, 208-220, 422-427` and `fv_conversions.py:113-118`: `D`/`DD` = the OLD down (TT-shaped) core -- the
  same-spelling swap D(old down)→U while U(old up)→D (`UU, DD = frame`→`DD, UU = frame`).
- `sampling_derivatives.py:300-364, 434-487, 782-842`: `D` = the forward Leibniz jet shift (`(D x)^(t) = t x^(t-1)`, `D^r nu`),
  with `U` = its adjoint reverse shift; they share docstrings with the Tucker factor (`z_i^(t) = U_i eta_i^(t)` at :523 → `D_i`).
- `entries.py:361-363`: `D = tucker_cores.shape[0]` (the mode count d, uniform path) -- a code local next to Tucker cores.
- `ut3_linalg.py:408-410` doctest: `D = pack(dirs)` (directions); `linalg.py:601` "Method D" (design-record label);
  `optimizer_display.py:21` "(D2)" and `optimizers.py:1647` "D6" (review codes); `t3_linalg.py:887,889` `D^2 Phi` (second derivative).
- `ufv_conversions.py:107-118`: the locals `(uc, dc)` swap to `(dc, uc)`; `dc` is the house alias of `dataclasses` in 7 backend
  modules (not imported in this one) -- consider `c_down, c_up`.
- The mode count `d` is lowercase everywhere (no clash in code; prose like "D_i (all i), d modes" reads poorly).

**C3 -- U already means something else.**
- The SVD left factor `U` (code locals and docstrings) in `linalg.py` (43 identifier hits incl. `U_i_a_x`, `U_nk`, `U_T`, …),
  `t3_svd.py` (8), `ut3_svd.py` (28), `tt_orthogonalization.py` (2), `t3_linalg.py` (`U, s, Vt = truncated_svd`, `Uw`), `sharing.py`
  (`U_M`, `svd_U`, 19), `uniform_padded_frames.py` (`Us`), `ut3_orthogonalization.py` (`U_o_x`), `t3_orthogonalization.py`
  (`Uox`, `U_a_x_b`, `U_i_a_x`). Untouched by the proposals (KEEP); a blanket `\bU\b` replace would corrupt them.
- `sampling_derivatives.py`: `U` = the reverse Leibniz shift (`U x = [(s+1) x^(s+1)]_s`, `U^r deta~`, `[U^u sigma~]`) IN THE
  SAME EQUATIONS as the TT-shaped frame core: e.g. :1381 `dxi~^(u) += (mu^(j) O) . [U^u sigma~]^(j) + [U^u tau~]^(j) . (O nu^(j))`
  becomes `(mu^(j) U) . [U^u sigma~]^(j) …`. Severe. Rows: `sampling_derivatives.py` 1381, 1382, 1431, 2187, 2699, 3167, 3168 (and the docstrings at 1426-1433,
  2187-2199, 2699-2700, 3167-3168, which carry `mu O` / `O nu` with the shift nearby).
- T3-level Tucker `U` in `t3_linalg.py` (`Ux`, `Uy`, `UL`, `UR`, `Ua`, `Ub`, `[G, U, mode]` chains), `t3_operations.py:89`,
  `geometry.py:395-418` (`U` over a point's Tucker cores): KEPT (T3 letter, not a frame store).
- `utv_sampling.py:115, 233` "U3" (= uniform T3): unrelated.

**C4 -- paper notation.** T4S writes the frame `(U, P, Q, O)` / "outer" cores `O` and the Section 6.3 substitution
`(P, Q, O) -> G`; the proposals turn the substitution into `(P, Q, U) -> G` (and "Algorithm 11" stays). Keeping the paper's
letters there with a correspondence note is the alternative. Rows: `apply.py` 321; `fv_conversions.py` 33; `geometry.py` 491, 510; `probing.py` 1427; `sampling_derivatives.py` 2386, 2408; `t3_linalg.py` 770; `ut3_sampling.py` 111, 229; `utv_operations.py` 382.

**C5 -- WIDTH-GENERIC (a scope question, not a clash).** `rL`/`rR`/`rLi`/`rR(i+1)` also name the left/right bond of ANY TT core
(`rL, n, rR = G.shape[-3:]`; `(rLi, ni, rR(i+1))` in plain-T3 shape comments) in `t3_svd`, `t3_operations`, `tt_operations`,
`tt_orthogonalization`, `t3_conversions`, `t3_constructors`, `ut3_linalg`, `ut3_orthogonalization`, `t3_orthogonalization`,
`ut3_svd`, `linalg`, `ranks` (minimal ranks, manifold dim, the shared-ceiling docstrings), `t3_linalg` (outside 660-760).
86 rows / 163 tokens, proposed `r_left`/`r_right` for uniformity but optional: they are not store widths. Also `nU` as a T3's
Tucker rank in `sampling_derivatives` chunk estimators (:1819-1921, `(nU_1..nU_d) Tucker ranks`).

## 6. CONTRACTION

- **Helper names**: `backend/contractions.py` exports `contract` and private `_pairwise_path`, `_grouped_einsum`,
  `_parse_grouped_subscripts`, `_co_travel_runs`, `_solve_group_lengths`, `_expanded_subscripts` -- none carries a store word or
  letter. No named-contraction helper in the backend is store-named. **Nothing changes.**
- **Subscripts**: no `contract(...)`/`einsum(...)` subscript in the backend contains an uppercase `U`, `O` or `D` (grep over all
  modules). Lowercase index labels such as `u` in `'Ckn,Cun->Cku'` (the Tucker-store leg), `o` (the physical leg) and `d` (the
  leading mode axis of the d-prefixed twins) are labels, not stores: **all subscripts stay**. Relabelling `u`→`d` would collide
  with the reserved `d` prefix, and every subscript is frozen in `tests/test_contractions_interpreter.py`'s HISTORICAL table
  (a vocabulary scan fails on any new string).
- **HISTORICAL table** (`tests/test_contractions_interpreter.py`): keys untouched. Its comments name stores: lines 360-363
  ("Far-side messages: down (c1, c2), left (c3-c8), right (c9-c14), up (c15-c20); … U+ g_new (c24) … and down (c30, c31)
  edges") flip to "up (c1, c2) … down (c15-c20) … D+ g_new (c24) … up (c30, c31) edges"; lines 227 and 335 ("the dotted
  adjoint-up lift (Udot z~)") are MSG + T3 letter: keep.

## 7. ADJECTIVE verification

Every orthogonality/unfolding adjective was checked against what the code does (old U = Tucker = new down; old O = TT = new up).

Correct today, STAY (names and prose): `t3_down_orthogonalize_tucker_cores`, `_down_orthogonalize_tucker_cores_step`,
`t3_up_orthogonalize_tt_cores`, `_up_orthogonalize_tt_cores_step`, `down_orthogonalize_tucker_supercores`,
`up_orthogonalize_tt_supercores`, `ut3_down_orthogonalize_tucker_cores`, `ut3_up_orthogonalize_tt_cores`, `t3_down_svd_tucker_core`
(Tucker core left down-orthogonal), `t3_down_svd_tt_core` (keeps the remainder in the TT core, Tucker core `Vt Q0` down-orthogonal),
`t3_up_svd_tt_core`, `linalg.up_svd`/`up_svd_pair`/`down_svd_pair`, `t3_svd._up_matricization` ("up-matricization W2"), the
prose "Tucker down-orthogonal(ization)" (`t3_orthogonalization`:38, `t3_svd`:67,103, `ut3_svd`:299,331,345,485,529,
`ut3_orthogonalization`:40, `ranks`:535,580,644, `ut3_linalg`:179-180, `t3_linalg`:537, `fv_conversions`:207), "TT-up SVD" and
"Up-orthogonalize the TT cores" (`ut3_orthogonalization`:190,215), "up unfolding"/"up-orthonormal" (`linalg`:330,348,463,482),
`t3_orthogonalization`:7 "the up/down core sweeps" (names the two methods: KEEP, the word rule must not flip it).

**Backwards (used the store convention) -- fixed by hand, §4**: `probing.py`:646-647, 1221-1222 ("up_tucker_supercore … up
orthogonal elements" / "down_tt_supercore … down orthogonal elements"); `t3_orthogonalization.py`:147 ("new_Uxo. up-orthogonal
tucker core"), :171 ("Orthogonalize Tucker cores upwards"); `ut3_orthogonalization.py`:189 ("new_tt = down-orthogonal O");
`geometry.py`:348 ("Tucker up-orth"); `t3_svd.py`:312, 345 ("Tucker up-SVD" for `t3_down_svd_tucker_core`);
`fv_conversions.py`:181-185 (local aliases `up_orthogonalize_tucker_cores = t3_down_orthogonalize_tucker_cores`,
`down_orthogonalize_tt_cores = t3_up_orthogonalize_tt_cores` -- the store-word rename makes them agree), :219 ("Orthogonalize
Tucker cores upward to get up_tt_cores U"), :251 ("Orthogonalize TT cores downward …"). Ambiguous (AMBIG, keep proposed):
`t3_svd.py`:187, 388 "group core's up leg" (the up-matricization leg).

## 8. Identifier map (code tokens; every occurrence)

From the tokenizer (NAME tokens only; prose mentions are in §9). `cat` per M1/M3/M4; `→ same` = a KEEP reference. Locations
are `file:line`; repeated lines mean several tokens on one line.

| identifier | new | cat | n | locations |
|---|---|---|---|---|
| `_down` | `_up` | STORE-WORD | 4 | geometry 128; utv_operations 459,551,721 |
| `_down_edge` | `_up_edge` | STORE-WORD | 2 | padded_frames 521,650 |
| `_down_mask` | `_up_mask` | STORE-WORD | 2 | utv_operations 356,394 |
| `_down_message` | `_up_message` | STORE-WORD | 2 | padded_frames 370,602 |
| `_u_down_messages` | `_u_up_messages` | STORE-WORD | 2 | uniform_padded_frames 368,550 |
| `_u_up_messages` | `_u_down_messages` | STORE-WORD | 2 | uniform_padded_frames 380,561 |
| `_up_edge` | `_down_edge` | STORE-WORD | 2 | padded_frames 448,620 |
| `_up_message` | `_down_message` | STORE-WORD | 2 | padded_frames 428,615 |
| `down` | `up` | STORE-WORD | 34 | fv_operations 106,108,382,401,408,421,652,653,681,683,691,733,736; geometry 620,621; padded_frames 71,74,87,89,276,281,650,660; ranks 674,676,678; ufv_operations 503,504,607,609,633,644,674,677 |
| `down0` | `up0` | STORE-WORD | 3 | padded_frames 256,257,276 |
| `down_cores` | `up_cores` | STORE-WORD | 6 | ufv_conversions 169,180,184,225,232,243 |
| `down_inds` | `up_inds` | STORE-WORD | 4 | ufv_conversions 172,180,285,290 |
| `down_mask` | `up_mask` | STORE-WORD | 20 | ufv_conversions 163,164,172,191; ufv_masking 36,37,83,86,113,114,129,165,166,180; ufv_operations 451,468,470,471; utv_operations 592,614 |
| `down_orthogonalize_tt_cores` | `up_orthogonalize_tt_cores` | STORE-WORD | 3 | fv_conversions 182,185,253 |
| `down_out` | `up_out` | STORE-WORD | 3 | fv_operations 419,427,430 |
| `down_r` | `up_r` | STORE-WORD | 2 | fv_conversions 215,263 |
| `down_ranks` | `up_ranks` | STORE-WORD | 14 | padded_frames 57,71; ranks 732,750; ufv_conversions 111,115,232,238,249,335,341,350; ufv_masking 66,83 |
| `down_sc` | `up_sc` | STORE-WORD | 27 | sharing 1020; ufv_conversions 243,254; ufv_operations 90,93,228,232,282,284,339,349,452,453,461; utv_operations 216,220,225,248,254,258,268,506,509,514,589,593,607 |
| `down_slices` | `up_slices` | STORE-WORD | 4 | utv_operations 237,242,249,254 |
| `down_supercore` | `up_supercore` | STORE-WORD | 3 | ufv_conversions 162,180,189 |
| `down_tt_cores` | `up_tt_cores` | STORE-WORD | 121 | apply 165,172,227,261,264,294,296,444,456,460,483,494; entries 132,139,213,216,236,239; fv_conversions 85,113,253,260,265; fv_operations 181,184,245,499,505; probing 284,297,298,302,304,403,425,433,440,466,617,620,621,733,742,747,751,804,808,976,989,996,997,1000,1197,1202,1322,1330,1340,1565,1580,1580,1582,1582; sampling_derivatives 594,597,735,749,756,758,765,776,917,922,923,1051,1060,1067,1070,1094,1099,1178,1183,1419,1436,1437,1440,2036,2043,2223,2234,2235,2240,2269,2273,2329,2333,2742,2756,2758,2761; t3_linalg 723,724; tv_operations 70,121,171,185,241,247,247,251,253,253,266,266,283,293,305,325,406,444,814,884 |
| `down_tt_cores_dot` | `up_tt_cores_dot` | STORE-WORD | 6 | probing 1566,1580,1582; sampling_derivatives 2743,2756,2761 |
| `down_tt_supercore` | `up_tt_supercore` | STORE-WORD | 5 | ufv_masking 112,120,121,122,136 |
| `down_weights` | `up_weights` | STORE-WORD | 1 | ufv_conversions 387 |
| `masked_down_tt_supercore` | `masked_up_tt_supercore` | STORE-WORD | 2 | ufv_masking 136,141 |
| `masked_up_tucker_supercore` | `masked_down_tucker_supercore` | STORE-WORD | 2 | ufv_masking 135,141 |
| `mdown` | `mup` | STORE-WORD | 4 | ufv_operations 283,287,294,294 |
| `mup` | `mdown` | STORE-WORD | 4 | ufv_operations 283,287,293,293 |
| `up` | `down` | STORE-WORD | 52 | fv_operations 106,107,108,109,382,383,384,393,396,401,408,411,416,416,652,655,681,683,686,690,732,736; geometry 128,129,620,621; padded_frames 70,74,87,88,92,93,94,275,281,628,660; ranks 663,667,672,674,676,678; ufv_operations 503,505,607,610,633,637,644,674,677 |
| `up0` | `down0` | STORE-WORD | 5 | padded_frames 256,257,258,259,274 |
| `up_cores` | `down_cores` | STORE-WORD | 9 | ufv_conversions 169,179,184,225,226,228,230,231,242 |
| `up_inds` | `down_inds` | STORE-WORD | 6 | ufv_conversions 171,179,181,182,284,291 |
| `up_mask` | `down_mask` | STORE-WORD | 27 | ufv_conversions 163,164,171,191; ufv_masking 36,37,82,86,113,114,127,128,165,166,183; ufv_operations 451,468,470,471; utv_operations 356,357,394,410,592,614,721,722 |
| `up_mask_frame` | `down_mask_frame` | STORE-WORD | 6 | utv_operations 230,232,232,232,233,233 |
| `up_mask_v` | `down_mask_v` | STORE-WORD | 2 | utv_operations 231,243 |
| `up_masks` | `down_masks` | STORE-WORD | 5 | fv_conversions 229,231,236,248,262 |
| `up_orthogonalize_tucker_cores` | `down_orthogonalize_tucker_cores` | STORE-WORD | 3 | fv_conversions 181,184,220 |
| `up_out` | `down_out` | STORE-WORD | 2 | fv_operations 417,430 |
| `up_r` | `down_r` | STORE-WORD | 7 | fv_conversions 207,211,214,215,228,229,236 |
| `up_ranks` | `down_ranks` | STORE-WORD | 23 | padded_frames 56,70,72,73; ranks 731,750,754,756,758; tv_operations 886,890; ufv_conversions 111,115,231,237,248,334,340,349; ufv_masking 65,82; utv_operations 357,367 |
| `up_sc` | `down_sc` | STORE-WORD | 48 | sharing 1020; ufv_conversions 242,254; ufv_operations 90,91,92,92,228,231,282,284,286,339,341,349,349,452,453,454,454; utv_operations 216,220,225,227,229,257,258,258,260,268,391,397,408,459,462,470,471,506,509,512,513,551,553,555,589,593,596,605 |
| `up_supercore` | `down_supercore` | STORE-WORD | 6 | ufv_conversions 162,165,166,179,187,189 |
| `up_tucker_cores` | `down_tucker_cores` | STORE-WORD | 123 | apply 165,168,227,228; entries 132,135,188,189,236,237,375,378,379,386,388; fv_conversions 85,88,94,104,106,220,225,229,233,236,254,261,263,265; fv_operations 181,183,245,247,248,249,499,500; probing 194,206,207,211,213,617,623,733,736,759,804,805,867,879,880,888,890,1197,1199,1322,1324,1334,1480,1487,1488,1492,1493; sampling_derivatives 618,619,619,638,639,639,661,662,662,917,925,1244,1250,1251,1257,1259,2036,2040,2329,2331,2338,2344,2345,2346,2492,2499,2500,2504,2505; t3_linalg 723,734,749; t3_orthogonalization 84,86,97,99,177,178; tv_operations 70,83,121,136,171,183,241,265,265,270,273,406,407,414,423,439,443,814,827,884,886 |
| `up_tucker_cores_dot` | `down_tucker_cores_dot` | STORE-WORD | 8 | probing 1481,1487,1491,1493; sampling_derivatives 2493,2499,2503,2505 |
| `up_tucker_supercore` | `down_tucker_supercore` | STORE-WORD | 6 | ufv_masking 112,116,117,118,119,135 |
| `up_weights` | `down_weights` | STORE-WORD | 2 | ufv_conversions 387,388 |
| `up_widths` | `down_widths` | STORE-WORD | 3 | padded_frames 94,96,103 |
| `var_down_mask` | `var_up_mask` | STORE-WORD | 5 | utv_operations 218,252,252,255,320 |
| `var_up_mask` | `var_down_mask` | STORE-WORD | 2 | utv_operations 218,320 |
| `vdown` | `vup` | STORE-WORD | 6 | ufv_conversions 276,277,285,300; ufv_operations 244,249 |
| `vup` | `vdown` | STORE-WORD | 6 | ufv_conversions 276,277,284,300; ufv_operations 244,249 |
| `kD` | `k_up` | WIDTH | 6 | uniform_padded_frames 203,204,210,217,258,259 |
| `kL` | `k_left` | WIDTH | 6 | uniform_padded_frames 203,204,210,218,245,246 |
| `kR` | `k_right` | WIDTH | 6 | uniform_padded_frames 203,204,210,219,251,252 |
| `kU` | `k_down` | WIDTH | 8 | uniform_padded_frames 203,204,210,216,224,224,233,240 |
| `nD` | `n_up` | WIDTH | 51 | fv_operations 401,402,421,422,422; padded_frames 272,276,566,570,579,581; ufv_conversions 114,115,206,238,238,238,243,252,313,341,341,341,345,353,366,391; ufv_masking 70,83,121,129,170,180; ufv_operations 93,94,287,294,349,642,644; utv_operations 258,277,279,288,288,289,300,302,627,634,637 |
| `nD0` | `n_up0` | WIDTH | 13 | padded_frames 576,581,602,605,611,622,623,632,642,652,653,653,655 |
| `nD1` | `n_up1` | WIDTH | 2 | padded_frames 580,655 |
| `nD_pad` | `n_up_pad` | WIDTH | 2 | utv_operations 225,226 |
| `nO` | `n_up` | WIDTH | 6 | sampling_derivatives 1820,1823,1824,1889,1892,1893 |
| `nU` | `n_down` | WIDTH | 70 | fv_operations 401,402; padded_frames 272,274,277,278,279,280,566,570,579,581,592; sampling_derivatives 1819,1820,1822,1824,1870,1875,1875,1875,1875,1888,1889,1891,1891,1892,1892,1906,1911,1911,1912,1912; ufv_conversions 114,115,205,237,237,237,242,244,245,252,312,340,340,340,346,353,365,391; ufv_masking 69,82,118,127,128,173,183; ufv_operations 92,94,287,293,349,643,644; utv_operations 258,279,284,297,302 |
| `nU0` | `n_down0` | WIDTH | 17 | padded_frames 576,581,592,616,616,617,621,621,624,634,644,651,651,652,654,657,658 |
| `nU1` | `n_down1` | WIDTH | 7 | padded_frames 580,592,624,634,644,657,658 |
| `nU_pad` | `n_down_pad` | WIDTH | 2 | utv_operations 225,226 |
| `rL` | `r_left` | WIDTH | 64 | fv_operations 401,402,421,422,422; padded_frames 272,276,277,277,278,278,566,570,579,581; ufv_conversions 114,115,207,239,239,239,243,244,244,252,314,342,342,342,346,353,367,391; ufv_masking 71,84,120,130,131,172,182; ufv_operations 93,94,287,297,349,455,473,477,643,644; utv_operations 259,277,277,279,284,288,289,289,297,302,302,321,521,523 |
| `rL0` | `r_left0` | WIDTH | 24 | padded_frames 576,581,605,605,606,606,616,633,634,634,634,636,642,642,643,643,645,651,653,657; sampling_derivatives 1346,1348,2644,2647 |
| `rL1` | `r_left1` | WIDTH | 6 | padded_frames 580,634,636,653,657,657 |
| `rR` | `r_right` | WIDTH | 64 | fv_operations 401,402,421,422,422; padded_frames 272,276,279,279,280,280,566,570,579,581; ufv_conversions 114,115,208,240,240,240,243,245,245,252,315,343,343,343,346,353,368,391; ufv_masking 72,85,122,132,133,174,184; ufv_operations 93,94,287,298,349,455,473,477,643,644; utv_operations 259,279,279,284,288,288,288,297,300,300,302,322,521,523 |
| `rR0` | `r_right0` | WIDTH | 28 | apply 454,455; padded_frames 576,581,611,611,612,612,617,632,632,633,633,635,643,644,644,644,646,652,653,658; probing 430,431; sampling_derivatives 755,757,1066,1068 |
| `rR1` | `r_right1` | WIDTH | 6 | padded_frames 580,644,646,653,658,658 |
| `rR_d` | `r_right_d` | WIDTH | 6 | probing 843,845; sampling_derivatives 2141,2143,3153,3155 |
| `rL` | `r_left` | WIDTH-GENERIC | 31 | ranks 136,139,145,148,149,710,713,715; t3_operations 264,265; t3_orthogonalization 115,116,120; t3_svd 642,648,653,770,774,775,786,788,793; tt_orthogonalization 51,52,53; ut3_linalg 113,117,119; ut3_orthogonalization 198,199,208 |
| `rLi` | `r_left_i` | WIDTH-GENERIC | 2 | t3_linalg 157,159 |
| `rR` | `r_right` | WIDTH-GENERIC | 31 | ranks 137,139,146,148,149,149,151,711,713,715,715; t3_operations 264,265; t3_orthogonalization 115,116,120; t3_svd 650,653,791,793,797; tt_orthogonalization 51,52,53; ut3_linalg 113,117,119,122; ut3_orthogonalization 198,199,208 |
| `rRi` | `r_right_i` | WIDTH-GENERIC | 2 | t3_linalg 157,159 |
| `BU` | `BD` | CORE-LETTER | 6 | t3_linalg 673,674,690,691; utv_operations 605,606 |
| `BU1` | `BD1` | CORE-LETTER | 2 | utv_operations 596,597 |
| `D` | `U` | CORE-LETTER | 10 | fv_conversions 113,116,116,118; fv_operations 218,220,220,220,422,427 |
| `D` | `E` | CORE-LETTER | 12 | padded_frames 490,491,491,517,518,518; uniform_padded_frames 428,429,429,472,473,473 |
| `DD` | `UU` | CORE-LETTER | 2 | fv_operations 208,218 |
| `DL` | `EL` | CORE-LETTER | 8 | padded_frames 637,638,638,654; uniform_padded_frames 582,593,593,597 |
| `DR` | `ER` | CORE-LETTER | 8 | padded_frames 647,648,648,654; uniform_padded_frames 586,594,594,596 |
| `DU` | `ED` | CORE-LETTER | 14 | padded_frames 480,482,507,509,627,635,645; uniform_padded_frames 418,420,462,464,576,584,588 |
| `DUs` | `EDs` | CORE-LETTER | 2 | padded_frames 620,627 |
| `D_L` | `E_L` | CORE-LETTER | 6 | padded_frames 479,483,532,535; uniform_padded_frames 417,421 |
| `D_R` | `E_R` | CORE-LETTER | 6 | padded_frames 506,510,532,534; uniform_padded_frames 461,465 |
| `O` | `U` | CORE-LETTER | 122 | apply 440,441,493,494; geometry 868,880,885,885; padded_frames 438,443,444,564,566,605,611,617,632,642,652,653; probing 373,386,396,397,965,968,969; sampling_derivatives 695,707,728,729,1384,1386,1387,1388,1391,1400,1402,1402,1403,2189,2191,2192,2192,2201,2203,2203,2204,2701,2703,2704,2705,2707,2709,2714,2723,2725,2727,3170,3172,3173,3174,3178,3188,3190,3191; t3_linalg 687,692; tv_operations 185,191,266,266,283,286,293,295,295,297,305,308,309,310,317,325,326,332; uniform_padded_frames 260,261,384,394,399,400,531,534,546,546,554,558,561,583,587,599,600,630,641,707,724,726,733,735,735,742,742,746; utv_operations 248,250,250,250,268,277,289,300 |
| `O0` | `U0` | CORE-LETTER | 7 | uniform_padded_frames 183,185,209,212,213,213,257 |
| `O1` | `U1` | CORE-LETTER | 4 | uniform_padded_frames 603,609,708,709 |
| `O_P` | `U_P` | CORE-LETTER | 10 | padded_frames 480,482,507,509,532,537; uniform_padded_frames 418,420,462,464 |
| `Od` | `Ud` | CORE-LETTER | 11 | sampling_derivatives 2701,2706,2708,2714,2723,2727,3170,3175,3178,3188,3191 |
| `Odxi` | `Udxi` | CORE-LETTER | 3 | sampling_derivatives 707,713,717 |
| `On` | `Un` | CORE-LETTER | 4 | sampling_derivatives 1388,1390,2707,2712 |
| `P_O` | `P_U` | CORE-LETTER | 3 | uniform_padded_frames 257,259,260 |
| `PhiO` | `PhiU` | CORE-LETTER | 18 | padded_frames 378,379,379,394,395,417,418,602,607,613,654; uniform_padded_frames 376,377,377,550,554,558,601 |
| `PhiU` | `PhiD` | CORE-LETTER | 10 | padded_frames 443,444,445,615,623; uniform_padded_frames 399,400,401,561,566 |
| `U` | `D` | CORE-LETTER | 104 | fv_operations 183,183,217,219,219,219; geometry 868,880,887,888; padded_frames 92,92,94,94,259,259,377,378,379,564,566,567,568,569,597,602,621,621,733,733; probing 189,190,314,315,860,863,1475,1476; sampling_derivatives 513,514,863,866,867,1239,1240,2487,2488; t3_linalg 671,673,687,690; tv_operations 136,137,137,183,188,189,265,265,273,273,414,414,827,828,886,886; ufv_conversions 230,230,231,231; uniform_padded_frames 241,261,370,376,377,531,534,536,537,538,543,546,546,550,564,565,630,636,637,637,641,707,724,726,733,734,734,734,741,741,746; utv_operations 268,272 |
| `U0` | `D0` | CORE-LETTER | 21 | padded_frames 188,194,195,196,198,199,200; uniform_padded_frames 183,185,188,189,190,209,212,213,213,239,241,938,941,956 |
| `U1` | `D1` | CORE-LETTER | 6 | padded_frames 620,628; uniform_padded_frames 577,609,708,709 |
| `UU` | `DD` | CORE-LETTER | 3 | fv_operations 208,209,217 |
| `U_new` | `D_new` | CORE-LETTER | 3 | padded_frames 463,464,464 |
| `Ud` | `Dd` | CORE-LETTER | 4 | probing 1475,1476; sampling_derivatives 2487,2488 |
| `Ui` | `Di` | CORE-LETTER | 2 | padded_frames 568,568 |
| `WD` | `W_up` | CORE-LETTER | 5 | uniform_padded_frames 728,730,733,738,745 |
| `WL` | `W_left` | CORE-LETTER | 5 | uniform_padded_frames 728,730,733,738,745 |
| `WR` | `W_right` | CORE-LETTER | 5 | uniform_padded_frames 728,730,733,738,745 |
| `WU` | `W_down` | CORE-LETTER | 5 | uniform_padded_frames 728,730,733,738,745 |
| `XiO` | `XiU` | CORE-LETTER | 12 | padded_frames 394,397,417,420,602,607,613,654; uniform_padded_frames 550,554,558,601 |
| `XiU` | `XiD` | CORE-LETTER | 4 | padded_frames 615,623; uniform_padded_frames 561,566 |
| `XiUs` | `XiDs` | CORE-LETTER | 2 | padded_frames 459,461 |
| `bO` | `bU` | CORE-LETTER | 6 | padded_frames 394,395,397,417,418,420 |
| `dOn` | `dUn` | CORE-LETTER | 2 | sampling_derivatives 2708,2713 |
| `dc` | `uc` | CORE-LETTER | 3 | ufv_conversions 107,114,117 |
| `dm` | `um` | CORE-LETTER | 7 | ufv_conversions 115,117,118; ufv_operations 228,236,282,294 |
| `dmO` | `dmU` | CORE-LETTER | 4 | sampling_derivatives 2705,2710,3174,3176 |
| `iD` | `i_up` | CORE-LETTER | 2 | uniform_padded_frames 737,742 |
| `iL` | `i_left` | CORE-LETTER | 4 | uniform_padded_frames 737,742,743,743 |
| `iR` | `i_right` | CORE-LETTER | 4 | uniform_padded_frames 737,742,744,744 |
| `iU` | `i_down` | CORE-LETTER | 4 | uniform_padded_frames 737,741,743,744 |
| `left_UU` | `left_DD` | CORE-LETTER | 3 | fv_conversions 104,108,110 |
| `mD` | `m_up` | CORE-LETTER | 9 | uniform_padded_frames 217,261,633,643,651,652,689,703,706 |
| `mD0` | `m_up0` | CORE-LETTER | 7 | uniform_padded_frames 186,192,215,215,217,258,259 |
| `mD1` | `m_up1` | CORE-LETTER | 2 | uniform_padded_frames 689,709 |
| `mL` | `m_left` | CORE-LETTER | 11 | uniform_padded_frames 218,245,258,261,633,644,651,652,690,703,706 |
| `mL0` | `m_left0` | CORE-LETTER | 7 | uniform_padded_frames 186,192,215,215,218,245,246 |
| `mL1` | `m_left1` | CORE-LETTER | 4 | uniform_padded_frames 690,700,702,709 |
| `mO` | `mU` | CORE-LETTER | 8 | sampling_derivatives 1387,1389,2191,2192,2704,2711,3173,3177 |
| `mR` | `m_right` | CORE-LETTER | 11 | uniform_padded_frames 219,251,258,261,633,644,651,652,690,703,706 |
| `mR0` | `m_right0` | CORE-LETTER | 7 | uniform_padded_frames 186,192,215,215,219,251,252 |
| `mR1` | `m_right1` | CORE-LETTER | 4 | uniform_padded_frames 690,701,702,709 |
| `mU` | `m_down` | CORE-LETTER | 13 | uniform_padded_frames 216,245,251,261,633,643,651,652,680,680,689,703,706 |
| `mU0` | `m_down0` | CORE-LETTER | 9 | uniform_padded_frames 186,192,215,215,216,224,224,230,239 |
| `mU1` | `m_down1` | CORE-LETTER | 4 | uniform_padded_frames 689,700,701,709 |
| `moves_O` | `moves_U` | CORE-LETTER | 2 | padded_frames 650,659 |
| `moves_U` | `moves_D` | CORE-LETTER | 2 | padded_frames 620,659 |
| `mvO` | `mvU` | CORE-LETTER | 2 | uniform_padded_frames 602,604 |
| `new_O` | `new_U` | CORE-LETTER | 2 | uniform_padded_frames 259,260 |
| `new_U` | `new_D` | CORE-LETTER | 2 | uniform_padded_frames 238,241 |
| `oU` | `oD` | CORE-LETTER | 9 | padded_frames 620,626,626,635,645; uniform_padded_frames 573,574,584,588 |
| `o_U` | `o_D` | CORE-LETTER | 16 | padded_frames 480,481,483,485,507,508,510,512; uniform_padded_frames 418,419,421,423,462,463,465,467 |
| `old_U` | `old_D` | CORE-LETTER | 2 | uniform_padded_frames 230,235 |
| `pD` | `p_up` | CORE-LETTER | 3 | uniform_padded_frames 688,689,705 |
| `pL` | `p_left` | CORE-LETTER | 3 | uniform_padded_frames 688,690,704 |
| `pR` | `p_right` | CORE-LETTER | 3 | uniform_padded_frames 688,690,705 |
| `pU` | `p_down` | CORE-LETTER | 3 | uniform_padded_frames 688,689,704 |
| `pinO` | `pinU` | CORE-LETTER | 2 | uniform_padded_frames 532,602 |
| `pinU` | `pinD` | CORE-LETTER | 2 | uniform_padded_frames 532,570 |
| `real_O` | `real_U` | CORE-LETTER | 2 | uniform_padded_frames 258,259 |
| `right_UU` | `right_DD` | CORE-LETTER | 3 | fv_conversions 106,108,110 |
| `rowO` | `rowU` | CORE-LETTER | 2 | uniform_padded_frames 532,602 |
| `rowU` | `rowD` | CORE-LETTER | 2 | uniform_padded_frames 532,570 |
| `seeds_O` | `seeds_U` | CORE-LETTER | 2 | uniform_padded_frames 258,259 |
| `selO` | `selU` | CORE-LETTER | 2 | uniform_padded_frames 532,602 |
| `selU` | `selD` | CORE-LETTER | 3 | uniform_padded_frames 532,541,570 |
| `uc` | `dc` | CORE-LETTER | 3 | ufv_conversions 107,114,117 |
| `um` | `dm` | CORE-LETTER | 7 | ufv_conversions 115,117,118; ufv_operations 228,236,282,293 |
| `_down_orthogonalize_tucker_cores_step` | → same | ADJECTIVE | 2 | t3_orthogonalization 144,177 |
| `_up_matricization` | → same | ADJECTIVE | 3 | t3_svd 130,257,263 |
| `_up_orthogonalize_tt_cores_step` | → same | ADJECTIVE | 2 | t3_orthogonalization 102,140 |
| `down_orthogonalize_tucker_supercores` | → same | ADJECTIVE | 8 | fv_conversions 181,225; ut3_orthogonalization 125,173; ut3_svd 374,379,562,566 |
| `down_svd_pair` | → same | ADJECTIVE | 2 | linalg 497; t3_orthogonalization 199 |
| `t3_down_orthogonalize_tucker_cores` | → same | ADJECTIVE | 11 | fv_conversions 184; t3_linalg 576,577; t3_orthogonalization 84,97,165,381,387,408; t3_svd 105,225 |
| `t3_down_svd_tt_core` | → same | ADJECTIVE | 4 | t3_orthogonalization 299; t3_svd 115,349,354 |
| `t3_down_svd_tucker_core` | → same | ADJECTIVE | 2 | t3_orthogonalization 181; t3_svd 346 |
| `t3_up_orthogonalize_tt_cores` | → same | ADJECTIVE | 2 | fv_conversions 185; t3_orthogonalization 128 |
| `t3_up_svd_tt_core` | → same | ADJECTIVE | 1 | t3_orthogonalization 336 |
| `up_orthogonalize_tt_supercores` | → same | ADJECTIVE | 4 | fv_conversions 182,260; ut3_orthogonalization 182,226 |
| `up_svd` | → same | ADJECTIVE | 3 | linalg 319,489; t3_orthogonalization 322 |
| `up_svd_pair` | → same | ADJECTIVE | 3 | linalg 448; t3_orthogonalization 354,392 |
| `ut3_down_orthogonalize_tucker_cores` | → same | ADJECTIVE | 2 | ut3_linalg 192; ut3_orthogonalization 157 |
| `ut3_up_orthogonalize_tt_cores` | → same | ADJECTIVE | 1 | ut3_orthogonalization 214 |
| `_shift_down` | → same | UNRELATED | 5 | sampling_derivatives 315,345,444,798,798 |
| `_shift_down_pair` | → same | UNRELATED | 4 | sampling_derivatives 338,381,710,711 |
| `_shift_up` | → same | UNRELATED | 8 | sampling_derivatives 327,355,1295,1563,2527,2527,2845,2845 |
| `_shift_up_pair` | → same | UNRELATED | 16 | sampling_derivatives 348,1278,1404,1404,1592,1593,2167,2204,2554,2555,2726,2874,3192,3192,3220,3220 |

Letter maps per file (code; the same maps applied to that file's comments/docstrings unless noted): `padded_frames` --
`U→D, O→U, Ui→Di, U0→D0, U1→D1, U_new→D_new, oU→oD, o_U→o_D, PhiO→PhiU, XiO→XiU, PhiU→PhiD, XiU→XiD, XiUs→XiDs, bO→bU,
O_P→U_P, moves_U→moves_D, moves_O→moves_U, D→E, D_L→E_L, D_R→E_R, DL→EL, DR→ER, DU→ED, DUs→EDs`; `uniform_padded_frames` -- the
same plus `O0→U0, O1→U1, old_U→old_D, new_U→new_D, seeds_O→seeds_U, real_O→real_U, new_O→new_U, P_O→P_U, rowU→rowD, rowO→rowU,
pinU→pinD, pinO→pinU, selU→selD, selO→selU, mvO→mvU` and the word forms `mU/mD/mL/mR(0|1)→m_down/m_up/m_left/m_right(0|1)`,
`pU…→p_down…`, `iU…→i_down…`, `WU…→W_down…`; `fv_operations` -- `UU↔DD, U→D, D→U`; `fv_conversions` -- `D→U, left_UU→left_DD,
right_UU→right_DD`; `ufv_conversions` -- `uc↔dc, um↔dm, U→D`; `ufv_operations` -- `um↔dm`; `tv_operations` -- `U→D, O→U`;
`utv_operations` -- `U→D, O→U, BU→BD, BU1→BD1`; `t3_linalg` lines 660-760 only -- `U→D, O→U, BU→BD`; `geometry` lines 860-892 --
`U→D, O→U`; `probing`, `sampling_derivatives`, `apply` -- `U→D, O→U, Ud→Dd, Od→Ud, mO→mU, On→Un, dmO→dmU, dOn→dUn, Odxi→Udxi`
(C1/C3 apply); `entries` -- prose only. Every map is a SIMULTANEOUS substitution (U→D and D→U in one pass).

## 9. Per-file line inventory (every occurrence)

Columns: line; kind (`code`, `cmt` comment, `doc` docstring line, `str` string literal); categories; the change (token pairs
`old→new` with multiplicity, or the full proposed line for hand rows, or `keep`); flags and notes. Keep rows are the conscious
keeps of M7 that a grep for `up`/`down`/`U`/`O`/`D` would hit.

### `apply.py` (51 rows, 49 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 38 | cmt | WIDTH | `rLi→r_left_i` |  |
| 39 | cmt | WIDTH | `nUi→n_down_i ×2; rLi→r_left_i; rL→r_left` |  |
| 144 | cmt | WIDTH | `nOi→n_up_i` |  |
| 145 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 148 | cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U→D` |  |
| 149 | cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O→U` |  |
| 152 | cmt | STORE-WORD | `up→down; down→up` |  |
| 165 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 168 | code+cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U_i→D_i` |  |
| 172 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 179 | cmt | STORE-WORD | `up→down; down→up` |  |
| 200 | cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U→D` |  |
| 201 | cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O→U` |  |
| 204 | cmt | STORE-WORD | `up→down; down→up` |  |
| 207 | cmt | WIDTH | `nUi→n_down_i` |  |
| 208 | cmt | WIDTH | `rLi→r_left_i` |  |
| 215 | doc | UNRELATED | keep | HAND AMBIG: 'down eta' / 'frame down jets' = T4S's DOWNWARD edge variables eta (travel direction, MSG): keep 'down' here although eta contracts the (new) UP store; generated text wrongly flips it -- apply only the width/identifier changes on this line |
| 227 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 228 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 235 | cmt | WIDTH | `nOi→n_up_i` |  |
| 236 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 240 | cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U→D` |  |
| 241 | cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O→U` |  |
| 244 | cmt | STORE-WORD | `up→down; down→up` |  |
| 261 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 264 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 271 | cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U→D` |  |
| 272 | cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O→U` |  |
| 275 | cmt | STORE-WORD | `up→down; down→up` |  |
| 284 | doc | CORE-LETTER | `O→U` |  |
| 294 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 296 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 316 | doc | UNRELATED | keep |  |
| 321 | doc | CORE-LETTER | `O→U; U_i→D_i` | C4 |
| 354 | doc | CORE-LETTER | `U→D; O→U` |  |
| 355 | doc | CORE-LETTER | `U→D ×3` |  |
| 435 | cmt | WIDTH | `rRi→r_right_i` |  |
| 436 | cmt | CORE-LETTER | `O→U` |  |
| 440 | code | CORE-LETTER | `O→U` |  |
| 441 | code | CORE-LETTER | `O→U` |  |
| 444 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 454 | code | WIDTH | `rR0→r_right0` |  |
| 455 | code | WIDTH | `rR0→r_right0` |  |
| 456 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 460 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 465 | doc | CORE-LETTER STORE-WORD WIDTH | `O_k→U_k; down→up; nO→n_up` |  |
| 466 | doc | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 467 | doc | WIDTH | `nO→n_up` |  |
| 483 | code+str | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 493 | code+cmt+str | CORE-LETTER | `O→U ×2` |  |
| 494 | code | CORE-LETTER STORE-WORD | `O→U; down_tt_cores→up_tt_cores` |  |

### `common.py` (1 rows, 0 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 543 | doc | UNRELATED | keep |  |

### `entries.py` (34 rows, 30 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 43 | cmt | WIDTH | `rLi→r_left_i` |  |
| 44 | cmt | WIDTH | `nUi→n_down_i ×2; rLi→r_left_i; rL→r_left` |  |
| 120 | cmt | STORE-WORD | `up→down; down→up` |  |
| 124 | doc | UNRELATED | keep |  |
| 125 | doc | CORE-LETTER | `U_i→D_i` |  |
| 132 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 135 | code+cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U_i→D_i` |  |
| 139 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 146 | cmt | STORE-WORD | `up→down; down→up` |  |
| 151 | doc | UNRELATED | keep |  |
| 166 | cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U→D` |  |
| 167 | cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O→U` |  |
| 170 | cmt | STORE-WORD | `up→down; down→up` |  |
| 173 | cmt | WIDTH | `nUi→n_down_i` |  |
| 174 | cmt | WIDTH | `rLi→r_left_i` |  |
| 188 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 189 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 196 | cmt | WIDTH | `nOi→n_up_i` |  |
| 197 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 203 | cmt | CORE-LETTER STORE-WORD | `U→D; O→U ×2; down→up` |  |
| 213 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 216 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 225 | cmt | CORE-LETTER | `U→D ×2; O→U ×2` |  |
| 236 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 237 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 239 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 286 | doc | CORE-LETTER | `U→D` |  |
| 314 | doc | UNRELATED | keep |  |
| 344 | doc | UNRELATED | keep |  |
| 375 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 378 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 379 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 386 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 388 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |

### `fitting.py` (8 rows, 7 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 16 | doc | CORE-LETTER | `U→D; O→U` | C1: (D,U,P,Q) -> (U,G,G,G): frame letters vs T3 letters in one formula |
| 326 | doc | UNRELATED | keep |  |
| 544 | cmt | CORE-LETTER | -> `# the lean sweep's xi / mu are base_point's own (xi through D, mu through L at the manifold frame` | HAND C1 |
| 545 | cmt | CORE-LETTER | -> `# (D, U, L, R); through U and G at the corewise frame) -- unweighted, so the residual is r itself` | C1: 'through U and G' = T3 letters (corewise) |
| 656 | cmt | CORE-LETTER | -> `# the lean sweep's xi / mu jets are base_point's own (xi through D, mu through L at the manifold frame` | HAND C1 |
| 657 | cmt | CORE-LETTER | `U→D; O→U` | C1 |
| 718 | cmt | CORE-LETTER | -> `# the sweep's xi / mu are base_point's own: the manifold frame (D, U, L, R) sweeps xi through D and mu` | HAND C1 |
| 719 | cmt | CORE-LETTER | `U→D` | C1: base_point is (D, L) / (U, G): frame pair vs T3 pair side by side |

### `fv_conversions.py` (49 rows, 49 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 32 | cmt | CORE-LETTER | `U→D; O→U` | C1: (D, U, P, Q) = (U, G, G, G): frame letters on the left, T3 letters on the right -- the two U's differ |
| 33 | doc | CORE-LETTER | `O→U` | C4: T4S's substitution (P, Q, O) -> G cites the paper's letters; keep the paper's O with a correspondence note, or use the new U |
| 51 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 52 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 57 | cmt | STORE-WORD | `up_tucker_supercore→down_tucker_supercore` |  |
| 58 | cmt | STORE-WORD | -> `NDArray,  # up_tt_supercore` | HAND: pre-existing typo `down_tucker_supercore` for the TT-shaped supercore; the mechanical map would leave a wrong name |
| 60 | cmt | UNRELATED | -> `NDArray,  # right_tt_supercore` | HAND: pre-existing typo `right_tucker_supercore` (not a store-word issue; fix in passing) |
| 85 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 88 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 94 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 104 | code | CORE-LETTER STORE-WORD | `left_UU→left_DD; up_tucker_cores→down_tucker_cores` |  |
| 106 | code | CORE-LETTER STORE-WORD | `right_UU→right_DD; up_tucker_cores→down_tucker_cores` |  |
| 108 | code | CORE-LETTER | `left_UU→left_DD; right_UU→right_DD` |  |
| 110 | code | CORE-LETTER | `left_UU→left_DD; right_UU→right_DD` |  |
| 113 | code | CORE-LETTER STORE-WORD | `D→U; down_tt_cores→up_tt_cores` |  |
| 116 | code | CORE-LETTER | `D→U ×2` |  |
| 118 | code | CORE-LETTER | `D→U` |  |
| 144 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 145 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 156 | cmt | STORE-WORD | `up_tucker_supercore→down_tucker_supercore` |  |
| 157 | cmt | STORE-WORD | -> `NDArray,  # up_tt_supercore` | HAND: same pre-existing typo |
| 159 | cmt | UNRELATED | -> `NDArray,  # right_tt_supercore` | HAND: same pre-existing typo |
| 180 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 181 | code | ADJECTIVE STORE-WORD | `up_orthogonalize_tucker_cores→down_orthogonalize_tucker_cores` | NOTE: local alias up_orthogonalize_tucker_cores -> down_orthogonalize_tucker_cores: the alias used the store convention (backwards); the rename makes it agree with the callee |
| 182 | code | ADJECTIVE STORE-WORD | `down_orthogonalize_tt_cores→up_orthogonalize_tt_cores` | NOTE: local alias down_orthogonalize_tt_cores -> up_orthogonalize_tt_cores (backwards alias fixed by the rename) |
| 184 | code | ADJECTIVE STORE-WORD | `up_orthogonalize_tucker_cores→down_orthogonalize_tucker_cores` | NOTE: as 181 |
| 185 | code | ADJECTIVE STORE-WORD | `down_orthogonalize_tt_cores→up_orthogonalize_tt_cores` | NOTE: as 182 |
| 207 | code+cmt | ADJECTIVE STORE-WORD | `up_r→down_r` |  |
| 211 | code | STORE-WORD | `up_r→down_r` |  |
| 214 | code | STORE-WORD | `up_r→down_r` |  |
| 215 | code+cmt | STORE-WORD | `down_r→up_r; up_r→down_r; down→up` |  |
| 219 | cmt | ADJECTIVE CORE-LETTER STORE-WORD UNRELATED | -> `# Down-orthogonalize the Tucker cores to get the down store D (down_tucker_cores)` | HAND: was 'Orthogonalize Tucker cores upward to get up_tt_cores U': backwards direction word + wrong identifier; a word swap cannot fix it |
| 220 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; up_orthogonalize_tucker_cores→down_orthogonalize_tucker_cores` |  |
| 225 | code | ADJECTIVE STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 228 | code | STORE-WORD | `up_r→down_r` |  |
| 229 | code | STORE-WORD | `up_masks→down_masks; up_r→down_r; up_tucker_cores→down_tucker_cores` |  |
| 231 | code | STORE-WORD | `up_masks→down_masks` |  |
| 233 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 236 | code | STORE-WORD | `up_masks→down_masks; up_r→down_r; up_tucker_cores→down_tucker_cores` |  |
| 248 | code | STORE-WORD | `up_masks→down_masks` |  |
| 251 | cmt | ADJECTIVE CORE-LETTER STORE-WORD UNRELATED | -> `# Up-orthogonalize the TT cores to get the up store U (up_tt_cores) and tucker_variations V` | HAND: was 'Orthogonalize TT cores downward to get outer_tt_cores O ...': backwards direction word |
| 253 | code | STORE-WORD | `down_tt_cores→up_tt_cores; down_orthogonalize_tt_cores→up_orthogonalize_tt_cores` |  |
| 254 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 259 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 260 | code | ADJECTIVE STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 261 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 262 | code | STORE-WORD | `up_masks→down_masks` |  |
| 263 | code | STORE-WORD | `down_r→up_r; up_tucker_cores→down_tucker_cores` |  |
| 265 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |

### `fv_operations.py` (120 rows, 119 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 95 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 97 | doc | WIDTH | `nD_i→n_up_i` |  |
| 98 | doc | WIDTH | `rL_i→r_left_i; nU_i→n_down_i; rR_i→r_right_i` |  |
| 101 | doc | UNRELATED | keep |  |
| 106 | code | STORE-WORD | `up→down; down→up` |  |
| 107 | code | STORE-WORD | `up→down` |  |
| 108 | code | STORE-WORD | `down→up; up→down` |  |
| 109 | code | STORE-WORD | `up→down` |  |
| 163 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 164 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 169 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 170 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 174 | doc | STORE-WORD | `up→down; down→up` |  |
| 178 | doc | STORE-WORD | `up→down; down→up` |  |
| 181 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 183 | code | CORE-LETTER STORE-WORD | `U→D ×2; up_tucker_cores→down_tucker_cores` |  |
| 184 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 191 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 192 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 201 | doc | CORE-LETTER STORE-WORD | -> `- down_tucker D_i (all i), outer/up U_i (all i),` | HAND: was 'up_tucker U_i ..., outer/down D_i': old D = the old down (TT-shaped) core, now U; both letters swap |
| 208 | code | CORE-LETTER | `UU→DD; DD→UU` |  |
| 209 | code | CORE-LETTER | `UU→DD` |  |
| 217 | code | CORE-LETTER | `U→D; UU→DD` |  |
| 218 | code | CORE-LETTER | `D→U; DD→UU` | C2: old `D` = the old down (TT-shaped) core -> U; `DD`/`UU` swap. Same-spelling swap: tokenizer-level, simultaneous |
| 219 | code+str | CORE-LETTER | `U→D ×3` | C2: old `D` = the old down (TT-shaped) core -> U; `DD`/`UU` swap. Same-spelling swap: tokenizer-level, simultaneous |
| 220 | code+str | CORE-LETTER | `D→U ×3` | C2: old `D` = the old down (TT-shaped) core -> U; `DD`/`UU` swap. Same-spelling swap: tokenizer-level, simultaneous |
| 232 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 233 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 239 | doc | STORE-WORD | `up→down ×2` |  |
| 245 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 247 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 248 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 249 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 290 | cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 291 | cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 292 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 293 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 298 | cmt | STORE-WORD WIDTH | `up→down; nUi→n_down_i` |  |
| 299 | cmt | STORE-WORD WIDTH | `down→up; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 313 | doc | STORE-WORD | `up→down` |  |
| 315 | doc | CORE-LETTER | `U_g→D_g` | HAND: U_g (the Haar Tucker factor of the zero-point frame = the down store) -> D_g; generated text kept U_g because U_g is protected as a T3 letter elsewhere |
| 317 | doc | STORE-WORD | `down→up` |  |
| 358 | doc | STORE-WORD | `up→down; down→up` |  |
| 359 | doc | STORE-WORD | `up→down` |  |
| 361 | doc | STORE-WORD | `up→down; down→up` |  |
| 363 | doc | STORE-WORD | `up→down` |  |
| 368 | doc | STORE-WORD | `up_more→down_more` |  |
| 369 | doc | STORE-WORD | `up_more→down_more; up→down` |  |
| 371 | doc | STORE-WORD | `up_more→down_more; up→down` |  |
| 378 | doc | STORE-WORD | `up_shared→down_shared` |  |
| 379 | doc | STORE-WORD | `up_shared→down_shared ×2` |  |
| 382 | code | STORE-WORD | `up→down; down→up` |  |
| 383 | code | STORE-WORD | `up→down` |  |
| 384 | code | STORE-WORD | `up→down` |  |
| 393 | code | STORE-WORD | `up→down` |  |
| 396 | code | STORE-WORD | `up→down` |  |
| 401 | code | STORE-WORD WIDTH | `nU→n_down; rL→r_left; nD→n_up; rR→r_right; up→down; down→up` |  |
| 402 | code | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 407 | str | STORE-WORD | `up→down; down→up` |  |
| 408 | code | STORE-WORD | `up→down; down→up` |  |
| 411 | code | STORE-WORD | `up→down` |  |
| 416 | code | STORE-WORD | `up→down ×2` |  |
| 417 | code | STORE-WORD | `up_out→down_out` |  |
| 419 | code | STORE-WORD | `down_out→up_out` |  |
| 421 | code | STORE-WORD WIDTH | `rL→r_left; nD→n_up; rR→r_right; down→up` |  |
| 422 | code+cmt | CORE-LETTER WIDTH | `D→U; nD→n_up ×3; rL→r_left ×2; rR→r_right ×2` | C2: old `D` = the old down (TT-shaped) core -> U; `DD`/`UU` swap. Same-spelling swap: tokenizer-level, simultaneous |
| 427 | code | CORE-LETTER STORE-WORD | `down_out→up_out; D→U` | C2: old `D` = the old down (TT-shaped) core -> U; `DD`/`UU` swap. Same-spelling swap: tokenizer-level, simultaneous |
| 430 | code | STORE-WORD | `up_out→down_out; down_out→up_out` |  |
| 438 | cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 439 | cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 440 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 441 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 444 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 445 | cmt | WIDTH | `nDi→n_up_i; nUi→n_down_i` |  |
| 446 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 459 | doc | CORE-LETTER WIDTH | `O_i→U_i; nDi→n_up_i; nUi→n_down_i` |  |
| 460 | doc | CORE-LETTER | `U_i→D_i` |  |
| 463 | doc | CORE-LETTER | `O→U` |  |
| 499 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 500 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 502 | cmt | WIDTH | `rLi→r_left_i; rRi→r_right_i` |  |
| 505 | code+str | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 535 | cmt | WIDTH | `nUi→n_down_i; nDi→n_up_i` |  |
| 536 | cmt | WIDTH | `rR→r_right; rL→r_left` |  |
| 537 | cmt | WIDTH | `nDi→n_up_i` |  |
| 538 | cmt | WIDTH | `rR→r_right` |  |
| 547 | cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 548 | cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 549 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 550 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 557 | doc | CORE-LETTER | `U→D` | C1: theta = (U, L) -> (D, L): the base point's raw cores as frame stores; Udot two lines below keeps the T3 letter |
| 576 | doc | WIDTH | `nDi→n_up_i; nUi→n_down_i` |  |
| 577 | doc | WIDTH | `rR→r_right; rL→r_left` |  |
| 634 | cmt | STORE-WORD | `up→down; down→up` |  |
| 636 | cmt | WIDTH | `nDi→n_up_i` |  |
| 637 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 640 | doc | STORE-WORD WIDTH | `down→up; nD→n_up; up→down` |  |
| 641 | doc | WIDTH | `nU→n_down; rL→r_left; rR→r_right` |  |
| 652 | code | STORE-WORD | `up→down; down→up` |  |
| 653 | code+str | STORE-WORD | `down→up` |  |
| 655 | code | STORE-WORD | `up→down` |  |
| 662 | cmt | STORE-WORD | `up→down; down→up` |  |
| 666 | doc | STORE-WORD WIDTH | `up→down; nU→n_down; down→up; nD→n_up; rL→r_left; rR→r_right` |  |
| 681 | code | STORE-WORD | `up→down; down→up` |  |
| 683 | code | STORE-WORD | `up→down; down→up` |  |
| 686 | code+cmt | STORE-WORD | `up→down` |  |
| 690 | code | STORE-WORD | `up→down` |  |
| 691 | code | STORE-WORD | `down→up` |  |
| 697 | cmt | STORE-WORD | `up→down; down→up` |  |
| 704 | cmt | STORE-WORD | `up→down; down→up` |  |
| 719 | cmt | STORE-WORD | `up_weights→down_weights` |  |
| 720 | cmt | STORE-WORD | `down_weights→up_weights` |  |
| 726 | doc | STORE-WORD | `up→down; down→up` |  |
| 729 | doc | STORE-WORD WIDTH | `down→up; nD→n_up; nU→n_down` |  |
| 730 | doc | STORE-WORD WIDTH | `nD→n_up; nU→n_down; down→up` |  |
| 732 | code | STORE-WORD | `up→down` |  |
| 733 | code | STORE-WORD | `down→up` |  |
| 736 | code | STORE-WORD | `up→down; down→up` |  |
| 742 | cmt | STORE-WORD | `up→down; down→up` |  |
| 755 | cmt | STORE-WORD | `up→down; down→up` |  |

### `geometry.py` (54 rows, 53 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 104 | cmt | CORE-LETTER | `U→D` |  |
| 106 | doc | CORE-LETTER | `U→D` |  |
| 113 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 119 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 121 | doc | CORE-LETTER | `U→D` |  |
| 125 | doc | CORE-LETTER | `U→D` |  |
| 128 | code | STORE-WORD | `up→down; _down→_up` |  |
| 129 | code+cmt | STORE-WORD | `up→down` |  |
| 136 | cmt | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 143 | doc | UNRELATED | keep |  |
| 144 | doc | STORE-WORD WIDTH | `up→down; nD→n_up; nU→n_down` |  |
| 145 | doc | STORE-WORD | `down→up` |  |
| 217 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 291 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 306 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 307 | cmt | CORE-LETTER | `U→D` |  |
| 308 | doc | CORE-LETTER | `U→D` |  |
| 314 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 343 | cmt | WIDTH | `nD→n_up` |  |
| 348 | doc | ADJECTIVE STORE-WORD | -> ```(Tucker down-orth + one left TT sweep, no ``W`` factor -- negligible beside the misfit evaluation it``` | HAND: backwards adjective: the Tucker cores are DOWN-orthogonalized |
| 361 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 362 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 422 | str | STORE-WORD | `up→down; down→up` |  |
| 423 | str | STORE-WORD | `up→down; down→up` |  |
| 433 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 469 | doc | CORE-LETTER | `U→D` |  |
| 491 | doc | CORE-LETTER | `O→U` | C4: T4S substitution letters |
| 509 | cmt | CORE-LETTER | `U→D; O→U` | C1: (D, U, P, Q) = (U, G, G, G) |
| 510 | doc | CORE-LETTER | `O→U` | C4: T4S substitution letters |
| 602 | cmt | CORE-LETTER STORE-WORD | -> `frame_data:  typ.Tuple,   # uniform frame .data = (D, U, P, Q, shape, (down, up, left, right) masks)` | HAND |
| 603 | cmt | CORE-LETTER STORE-WORD | `U→D; up→down` |  |
| 604 | doc | STORE-WORD | `up→down` |  |
| 612 | cmt | STORE-WORD | `up→down; down→up` |  |
| 619 | doc | WIDTH | `nD_i→n_up_i; nU→n_down` |  |
| 620 | code | STORE-WORD | `up→down; down→up` |  |
| 621 | code+cmt | STORE-WORD | `down→up; up→down` |  |
| 686 | cmt | STORE-WORD | `up→down; down→up` |  |
| 707 | cmt | CORE-LETTER | -> `frame_data:   typ.Tuple,                          # uniform frame .data = (D, U, P, Q, shape, masks)` | HAND |
| 742 | cmt | STORE-WORD | `up→down; down→up` |  |
| 751 | doc | WIDTH | `nU→n_down` |  |
| 755 | doc | CORE-LETTER | `U→D` |  |
| 815 | doc | STORE-WORD | `up→down` |  |
| 866 | str | STORE-WORD | `up→down; down→up` |  |
| 868 | code | CORE-LETTER | `U→D; O→U` |  |
| 872 | str | STORE-WORD | `up→down; down→up` |  |
| 880 | code | CORE-LETTER | `U→D; O→U` |  |
| 885 | code | CORE-LETTER | `O→U ×2` |  |
| 886 | str | STORE-WORD | `down→up` |  |
| 887 | code | CORE-LETTER | `U→D` |  |
| 888 | code+str | CORE-LETTER | `U→D` |  |
| 893 | cmt | CORE-LETTER | -> `frame_data:         typ.Tuple,                   # uniform frame .data = (D, U, P, Q, shape, masks)` | HAND |
| 923 | doc | CORE-LETTER | `U→D` |  |
| 925 | doc | STORE-WORD | `up→down` |  |
| 1003 | doc | WIDTH | `nU→n_down` |  |

### `linalg.py` (21 rows, 0 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 7 | doc | ADJECTIVE | keep |  |
| 30 | str | ADJECTIVE | keep |  |
| 33 | str | ADJECTIVE | keep |  |
| 34 | str | ADJECTIVE | keep |  |
| 319 | code | ADJECTIVE | keep |  |
| 330 | doc | ADJECTIVE | keep |  |
| 342 | doc | ADJECTIVE | keep |  |
| 348 | doc | ADJECTIVE | keep |  |
| 448 | code | ADJECTIVE | keep |  |
| 463 | doc | ADJECTIVE | keep |  |
| 474 | doc | ADJECTIVE | keep |  |
| 482 | doc | ADJECTIVE | keep |  |
| 489 | code | ADJECTIVE | keep |  |
| 497 | code | ADJECTIVE | keep |  |
| 511 | doc | ADJECTIVE | keep |  |
| 522 | doc | ADJECTIVE | keep |  |
| 612 | doc | UNRELATED | keep |  |
| 679 | cmt | UNRELATED | keep |  |
| 715 | cmt | UNRELATED | keep |  |
| 746 | doc | UNRELATED | keep |  |
| 951 | doc | UNRELATED | keep |  |

### `optimizer_display.py` (1 rows, 0 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 212 | doc | UNRELATED | keep |  |

### `optimizers.py` (6 rows, 2 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 79 | doc | UNRELATED | keep |  |
| 88 | doc | UNRELATED | keep |  |
| 111 | doc | UNRELATED | keep |  |
| 206 | doc | UNRELATED | keep |  |
| 228 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 807 | doc | STORE-WORD | -> `>>> F.down_ranks, len(movements) <= 10` | HAND SILENT: frontend attribute T3Frame.up_ranks: the Tucker-store widths are now down_ranks. At this (minimal) structure up_ranks would print the SAME (3, 4, 3) -- a stale name passes the doctest silently |

### `padded_frames.py` (225 rows, 223 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 8 | doc | STORE-WORD | `up→down; down→up` |  |
| 13 | doc | LETTER-API | -> ```one fixed tensor, a truncating pass to the target ranks, then two-colour passes in ``(('down', 'left'), ('right', 'up'))`` pairs. The``` | HAND: the colour pairs 'UL'/'RO' in words: old U = Tucker = new down, old O = TT-shaped = new up |
| 41 | cmt | STORE-WORD WIDTH | `up→down; down→up; nUi→n_down_i ×3; rLi→r_left_i ×2; nDi→n_up_i; rR→r_right ×2; rL→r_left; rRi→r_right_i` |  |
| 43 | cmt | WIDTH | `nDi→n_up_i; rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 51 | str | STORE-WORD | `up→down; down→up` | HAZARD: _STORE_NAMES is in physical order and also the error-message labels; tests match messages such as 'old_frame down[1]' |
| 56 | code+cmt | STORE-WORD WIDTH | `up_ranks→down_ranks; nU→n_down` |  |
| 57 | code+cmt | STORE-WORD WIDTH | `down_ranks→up_ranks; nD→n_up` |  |
| 58 | cmt | WIDTH | `rL→r_left` |  |
| 59 | cmt | WIDTH | `rR→r_right` |  |
| 62 | cmt | STORE-WORD WIDTH | `up→down; nUi→n_down_i` |  |
| 63 | cmt | STORE-WORD WIDTH | `down→up; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 64 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 65 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 70 | code | STORE-WORD | `up→down; up_ranks→down_ranks` |  |
| 71 | code | STORE-WORD | `down→up; down_ranks→up_ranks` |  |
| 72 | code | STORE-WORD | `up_ranks→down_ranks` |  |
| 73 | code | STORE-WORD | `up_ranks→down_ranks` |  |
| 74 | code | STORE-WORD | `up→down; down→up` |  |
| 81 | cmt | WIDTH | `nU→n_down` |  |
| 82 | cmt | WIDTH | `nD→n_up` |  |
| 83 | cmt | WIDTH | `rL→r_left` |  |
| 84 | cmt | WIDTH | `rR→r_right` |  |
| 86 | doc | STORE-WORD | `up→down` |  |
| 87 | code | STORE-WORD | `up→down; down→up` |  |
| 88 | code | STORE-WORD | `up→down` |  |
| 89 | code | STORE-WORD | `down→up` |  |
| 92 | code | CORE-LETTER STORE-WORD | `U→D ×2; up→down` |  |
| 93 | code | STORE-WORD | `up→down` |  |
| 94 | code | CORE-LETTER STORE-WORD | `up_widths→down_widths; U→D ×2; up→down` |  |
| 96 | code | STORE-WORD | `up_widths→down_widths` |  |
| 102 | str | STORE-WORD | `up→down` |  |
| 103 | code+str | STORE-WORD | `up_widths→down_widths` |  |
| 114 | cmt | WIDTH | `nU→n_down` |  |
| 115 | cmt | WIDTH | `nD→n_up` |  |
| 116 | cmt | WIDTH | `rL→r_left` |  |
| 117 | cmt | WIDTH | `rR→r_right` |  |
| 128 | cmt | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 129 | cmt | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 139 | cmt+str | STORE-WORD | `up→down; down→up` | HAZARD: _SEED_TAGS: the INTEGERS stay with the physical store (Tucker factor = 0, TT-shaped = 1) or every padded_seed stream changes; the test oracle keeps its own copy of this table (tests/backend/test_padded_frame.py), so it must be edited in lockstep |
| 143 | cmt | STORE-WORD | `up→down; down→up` |  |
| 148 | cmt | STORE-WORD | `up→down` |  |
| 164 | cmt | STORE-WORD | `down→up` |  |
| 167 | cmt | STORE-WORD | `down→up` |  |
| 171 | doc | STORE-WORD | `down→up` |  |
| 188 | code+cmt | CORE-LETTER | `U0→D0` |  |
| 192 | cmt | CORE-LETTER | `U0→D0` |  |
| 194 | code | CORE-LETTER | `U0→D0` |  |
| 195 | code | CORE-LETTER | `U0→D0` |  |
| 196 | code | CORE-LETTER | `U0→D0` |  |
| 197 | str | STORE-WORD | `up→down` |  |
| 198 | code | CORE-LETTER | `U0→D0` |  |
| 199 | code | CORE-LETTER | `U0→D0` |  |
| 200 | code | CORE-LETTER | `U0→D0` |  |
| 208 | cmt | STORE-WORD | `up→down` |  |
| 209 | cmt | STORE-WORD | `down→up` |  |
| 220 | doc | STORE-WORD | `up→down; down→up` |  |
| 221 | doc | CORE-LETTER STORE-WORD | `U→D ×2; up→down` |  |
| 248 | cmt | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 249 | cmt | STORE-WORD WIDTH | `up→down; nUi→n_down_i` |  |
| 250 | cmt | STORE-WORD WIDTH | `down→up; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 251 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 252 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 256 | code | STORE-WORD | `up0→down0; down0→up0` |  |
| 257 | code+str | STORE-WORD | `up0→down0; down0→up0` |  |
| 258 | code | STORE-WORD | `up0→down0` |  |
| 259 | code | CORE-LETTER STORE-WORD | `U→D ×2; up0→down0` |  |
| 272 | code | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 274 | code | STORE-WORD WIDTH | `up0→down0; nU→n_down` |  |
| 275 | code+cmt | STORE-WORD | `up→down` |  |
| 276 | code+str | STORE-WORD WIDTH | `down→up ×2; down0→up0; rL→r_left; nD→n_up; rR→r_right` |  |
| 277 | code+str | WIDTH | `rL→r_left ×2; nU→n_down` |  |
| 278 | code+cmt | WIDTH | `rL→r_left ×2; nU→n_down` |  |
| 279 | code | WIDTH | `rR→r_right ×2; nU→n_down` |  |
| 280 | code+str | WIDTH | `rR→r_right ×2; nU→n_down` |  |
| 281 | code | STORE-WORD | `up→down; down→up` |  |
| 370 | code | STORE-WORD | `_down_message→_up_message` |  |
| 371 | cmt | CORE-LETTER WIDTH | `U→D; kDi→k_up_i; nUi→n_down_i` |  |
| 373 | cmt | CORE-LETTER WIDTH | `PhiO→PhiU; kD→k_up; nU→n_down` |  |
| 374 | cmt | CORE-LETTER WIDTH | `XiO→XiU; kD→k_up` |  |
| 376 | doc | STORE-WORD | keep | NOTE: function renamed _down_message -> _up_message (edge core -> leaf holds the TT-shaped = up store); the docstring's arrow notation stays |
| 377 | code | CORE-LETTER | `U→D` |  |
| 378 | code+str | CORE-LETTER | `PhiO→PhiU; U→D` |  |
| 379 | code+str | CORE-LETTER | `PhiO→PhiU ×2; U→D` |  |
| 383 | cmt | WIDTH | `kL→k_left ×2; rR→r_right` |  |
| 384 | cmt | CORE-LETTER WIDTH | `bO→bU; PhiO→PhiU; XiO→XiU; kLj→k_left_j; nUj→n_down_j; rR→r_right` |  |
| 385 | cmt | WIDTH | `kLj→k_left_j ×2; kDj→k_up_j; rR→r_right ×2; nUj→n_down_j ×2; kL→k_left; rRj→r_right_j` |  |
| 386 | cmt | WIDTH | `kDj→k_up_j ×2; nUj→n_down_j` |  |
| 394 | code | CORE-LETTER | `bO→bU; PhiO→PhiU; XiO→XiU` |  |
| 395 | code+str | CORE-LETTER | `bO→bU; PhiO→PhiU` |  |
| 397 | code+str | CORE-LETTER | `bO→bU; XiO→XiU` |  |
| 406 | cmt | WIDTH | `kRj→k_right_j ×2; rLj→r_left_j` |  |
| 407 | cmt | CORE-LETTER WIDTH | `bO→bU; PhiO→PhiU; XiO→XiU; rLj→r_left_j; nUj→n_down_j; kR→k_right` |  |
| 408 | cmt | WIDTH | `rLj→r_left_j ×2; kDj→k_up_j; kR→k_right ×2; kRj→k_right_j; nUj→n_down_j ×2; rL→r_left` |  |
| 409 | cmt | WIDTH | `kDj→k_up_j ×2; nUj→n_down_j` |  |
| 417 | code | CORE-LETTER | `bO→bU; PhiO→PhiU; XiO→XiU` |  |
| 418 | code+str | CORE-LETTER | `bO→bU; PhiO→PhiU` |  |
| 420 | code+str | CORE-LETTER | `bO→bU; XiO→XiU` |  |
| 428 | code | STORE-WORD | `_up_message→_down_message` |  |
| 429 | cmt | CORE-LETTER WIDTH | `O→U; rLi→r_left_i; kUi→k_down_i; rR→r_right` |  |
| 430 | cmt | WIDTH | `rLi→r_left_i ×2; kUi→k_down_i ×2; kL→k_left; kRi→k_right_i; rR→r_right ×2; nDi→n_up_i` |  |
| 431 | cmt | WIDTH | `kL→k_left ×2; rR→r_right; kRi→k_right_i ×2; rLi→r_left_i` |  |
| 433 | cmt | CORE-LETTER WIDTH | `PhiU→PhiD; kU→k_down; nD→n_up` |  |
| 434 | cmt | CORE-LETTER WIDTH | `XiU→XiD; kU→k_down` |  |
| 436 | doc | STORE-WORD | keep | NOTE: function renamed _up_message -> _down_message (edge leaf -> core holds the Tucker = down store) |
| 438 | code | CORE-LETTER | `O→U` |  |
| 443 | code+str | CORE-LETTER | `PhiU→PhiD; O→U` |  |
| 444 | code+str | CORE-LETTER | `PhiU→PhiD; O→U` |  |
| 445 | code | CORE-LETTER | `PhiU→PhiD` |  |
| 448 | code | STORE-WORD | `_up_edge→_down_edge` |  |
| 449 | cmt | CORE-LETTER | `XiUs→XiDs` |  |
| 450 | cmt | WIDTH | `kDi→k_up_i ×2` |  |
| 452 | cmt | CORE-LETTER | `U→D` |  |
| 453 | cmt | CORE-LETTER | `o_U→o_D` |  |
| 454 | cmt | CORE-LETTER WIDTH | `DU→ED; kD→k_up` |  |
| 459 | code | CORE-LETTER | `XiUs→XiDs` |  |
| 461 | code | CORE-LETTER | `XiUs→XiDs` |  |
| 463 | code | CORE-LETTER | `U_new→D_new` |  |
| 464 | code+str | CORE-LETTER | `U_new→D_new ×2` |  |
| 468 | cmt | CORE-LETTER WIDTH | `D_L→E_L; k1Li→k1_left_i; kLi→k_left_i; rL1i→r_left1_i; kRi→k_right_i` |  |
| 469 | cmt | CORE-LETTER | `O_P→U_P; o_U→o_D; DU→ED` |  |
| 470 | cmt | WIDTH | `rLi→r_left_i ×2; nUi→n_down_i ×2; kR→k_right ×3; kDi→k_up_i; kRi→k_right_i` |  |
| 471 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; kL→k_left; rL1i→r_left1_i; nU1i→n_down1_i; rL0→r_left0; k1Ui→k1_down_i; kUi→k_down_i` |  |
| 472 | cmt | WIDTH | `nU1i→n_down1_i; kDi→k_up_i; kL→k_left ×2; kR→k_right` |  |
| 474 | cmt | CORE-LETTER | `D_L→E_L` |  |
| 475 | cmt | CORE-LETTER | `D_L→E_L` |  |
| 479 | code | CORE-LETTER | `D_L→E_L` |  |
| 480 | code | CORE-LETTER | `O_P→U_P; o_U→o_D; DU→ED` |  |
| 481 | code | CORE-LETTER | `o_U→o_D` |  |
| 482 | code+str | CORE-LETTER | `O_P→U_P; DU→ED` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 483 | code+str | CORE-LETTER | `D_L→E_L; o_U→o_D` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 485 | code | CORE-LETTER | `o_U→o_D` |  |
| 490 | code+str | CORE-LETTER | `D→E` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 491 | code | CORE-LETTER | `D→E ×2` |  |
| 495 | cmt | CORE-LETTER WIDTH | `D_R→E_R; k1R→k1_right; kR→k_right; rR1→r_right1` |  |
| 496 | cmt | CORE-LETTER | `O_P→U_P; o_U→o_D; DU→ED` |  |
| 497 | cmt | WIDTH | `kLi→k_left_i ×3; nUi→n_down_i ×2; rR→r_right ×2; kDi→k_up_i; kL→k_left` |  |
| 498 | cmt | WIDTH | `kRi→k_right_i; nUi→n_down_i; rR→r_right; rR0i→r_right0_i; nU1i→n_down1_i; rR1→r_right1; k1Ui→k1_down_i; kUi→k_down_i` |  |
| 499 | cmt | WIDTH | `nU1i→n_down1_i; kDi→k_up_i; kRi→k_right_i ×2; kLi→k_left_i` |  |
| 501 | cmt | CORE-LETTER | `D_R→E_R` |  |
| 502 | cmt | CORE-LETTER | `D_R→E_R` |  |
| 506 | code | CORE-LETTER | `D_R→E_R` |  |
| 507 | code | CORE-LETTER | `O_P→U_P; o_U→o_D; DU→ED` |  |
| 508 | code | CORE-LETTER | `o_U→o_D` |  |
| 509 | code+str | CORE-LETTER | `O_P→U_P; DU→ED` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 510 | code+str | CORE-LETTER | `D_R→E_R; o_U→o_D` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 512 | code | CORE-LETTER | `o_U→o_D` |  |
| 517 | code+str | CORE-LETTER | `D→E` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 518 | code | CORE-LETTER | `D→E ×2` |  |
| 521 | code | STORE-WORD | `_down_edge→_up_edge` |  |
| 522 | cmt | CORE-LETTER | `O_P→U_P; D_L→E_L; D_R→E_R` |  |
| 523 | cmt | WIDTH | `rLi→r_left_i ×3; kUi→k_down_i ×3; rR→r_right ×3; kL→k_left; kRi→k_right_i; kDi→k_up_i` |  |
| 524 | cmt | WIDTH | `rL1i→r_left1_i ×2; nD0i→n_up0_i; rR1→r_right1; k1Li→k1_left_i; kLi→k_left_i; kRi→k_right_i; k1R→k1_right; kR→k_right` |  |
| 525 | cmt | WIDTH | `rR1→r_right1; kL→k_left; kDi→k_up_i ×2; kUi→k_down_i` |  |
| 527 | cmt | CORE-LETTER WIDTH | `O→U; rL1i→r_left1_i; nD1i→n_up1_i; rR1→r_right1` |  |
| 532 | code | CORE-LETTER | `O_P→U_P; D_L→E_L; D_R→E_R` |  |
| 534 | code+str | CORE-LETTER | `D_R→E_R` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 535 | code+str | CORE-LETTER | `D_L→E_L` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 537 | code | CORE-LETTER | `O_P→U_P` |  |
| 554 | cmt+str | LETTER-API | -> `update:            typ.Tuple[str, ...] = ('down', 'up', 'left', 'right'),   # the stores updated (store words); the rest transported` | HAND: API: str of letters -> tuple of store words (hashable: jit static). Spelling/type is a plan decision |
| 560 | str | LETTER-API | -> `if isinstance(update, str) or not update or len(set(update)) != len(update) or not set(update) <= set(_STORE_NAMES):` | HAND: a bare str must be rejected: set('down') is a set of characters |
| 561 | str | CORE-LETTER LETTER-API | -> `raise ValueError("update must be a non-empty tuple of the stores 'down', 'up', 'left', 'right' without repeats; got %r"` | HAND: message text is asserted by tests ('without repeats', 'update must be') |
| 564 | code | CORE-LETTER | `U→D; O→U` |  |
| 566 | code+str | CORE-LETTER WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right; U→D; O→U` |  |
| 567 | code | CORE-LETTER | `U→D` |  |
| 568 | code | CORE-LETTER | `Ui→Di ×2; U→D` |  |
| 569 | code | CORE-LETTER | `U→D` |  |
| 570 | code | WIDTH | `nD→n_up; rL→r_left; nU→n_down; rR→r_right` |  |
| 576 | code+str | WIDTH | `nU0→n_down0; nD0→n_up0; rL0→r_left0; rR0→r_right0` |  |
| 579 | code+str | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 580 | code | WIDTH | `nU1→n_down1; nD1→n_up1; rL1→r_left1; rR1→r_right1` |  |
| 581 | code+str | LETTER-API WIDTH | -> `for store, old, new, own in zip(_STORE_NAMES, (n_down0, n_up0, r_left0, r_right0), out, (n_down, n_up, r_left, r_right)):` | HAND: drop the 'UOLR' letter column; the store word is the update key |
| 592 | code+str | WIDTH | `nU0→n_down0; nU1→n_down1; nU→n_down` |  |
| 597 | code | CORE-LETTER | `U→D` |  |
| 602 | code | CORE-LETTER STORE-WORD WIDTH | `PhiO→PhiU; XiO→XiU; _down_message→_up_message; nD0→n_up0; U→D` |  |
| 605 | code | CORE-LETTER WIDTH | `rL0→r_left0 ×2; O→U; nD0→n_up0` |  |
| 606 | code | WIDTH | `rL0→r_left0 ×2` |  |
| 607 | code | CORE-LETTER | `PhiO→PhiU; XiO→XiU` |  |
| 611 | code | CORE-LETTER WIDTH | `rR0→r_right0 ×2; O→U; nD0→n_up0` |  |
| 612 | code | WIDTH | `rR0→r_right0 ×2` |  |
| 613 | code | CORE-LETTER | `PhiO→PhiU; XiO→XiU` |  |
| 615 | code | CORE-LETTER STORE-WORD | `PhiU→PhiD; XiU→XiD; _up_message→_down_message` |  |
| 616 | code | WIDTH | `nU0→n_down0 ×2; rL0→r_left0` |  |
| 617 | code | CORE-LETTER WIDTH | `rR0→r_right0; nU0→n_down0; O→U` |  |
| 619 | cmt | STORE-WORD | `up→down; down→up` |  |
| 620 | code | CORE-LETTER STORE-WORD | `U1→D1; oU→oD; DUs→EDs; moves_U→moves_D; _up_edge→_down_edge` |  |
| 621 | code | CORE-LETTER WIDTH | `U→D ×2; nU0→n_down0 ×2` |  |
| 622 | code | WIDTH | `nD0→n_up0` |  |
| 623 | code | CORE-LETTER WIDTH | `PhiU→PhiD; nD0→n_up0; XiU→XiD` |  |
| 624 | code+str | CORE-LETTER LETTER-API WIDTH | `nU1→n_down1; nU0→n_down0; U→down` | HAND: ('D' in update,) -> ('down' in update,)  [generated text shows the interim letter] |
| 626 | code | CORE-LETTER | `oU→oD ×2` |  |
| 627 | code | CORE-LETTER | `DU→ED; DUs→EDs` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 628 | code+cmt | CORE-LETTER STORE-WORD | `up→down; U1→D1` |  |
| 632 | code | CORE-LETTER WIDTH | `rR0→r_right0 ×2; O→U; nD0→n_up0` |  |
| 633 | code | WIDTH | `rR0→r_right0 ×2; rL0→r_left0` |  |
| 634 | code | WIDTH | `rL0→r_left0 ×3; nU0→n_down0; rL1→r_left1; nU1→n_down1` |  |
| 635 | code | CORE-LETTER WIDTH | `oU→oD; DU→ED; rR0→r_right0` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 636 | code+str | LETTER-API WIDTH | `rL1→r_left1; rL0→r_left0; L→left` | HAND: ('L' in update,) -> ('left' in update,) |
| 637 | code | CORE-LETTER | `DL→EL` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 638 | code+cmt | CORE-LETTER | `DL→EL ×2` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 642 | code | CORE-LETTER WIDTH | `rL0→r_left0 ×2; O→U; nD0→n_up0` |  |
| 643 | code | WIDTH | `rL0→r_left0 ×2; rR0→r_right0` |  |
| 644 | code | WIDTH | `rR0→r_right0 ×3; nU0→n_down0; nU1→n_down1; rR1→r_right1` |  |
| 645 | code | CORE-LETTER WIDTH | `oU→oD; DU→ED; rL0→r_left0` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 646 | code+str | LETTER-API WIDTH | `rR1→r_right1; rR0→r_right0; R→right` | HAND: ('R' in update,) -> ('right' in update,) |
| 647 | code | CORE-LETTER | `DR→ER` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 648 | code+cmt | CORE-LETTER | `DR→ER ×2` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 650 | code | CORE-LETTER STORE-WORD | `down→up; moves_O→moves_U; _down_edge→_up_edge` |  |
| 651 | code | WIDTH | `nU0→n_down0 ×2; rL0→r_left0` |  |
| 652 | code | CORE-LETTER WIDTH | `rR0→r_right0; nU0→n_down0; O→U; nD0→n_up0` |  |
| 653 | code | CORE-LETTER WIDTH | `O→U; rL0→r_left0; nD0→n_up0 ×2; rR0→r_right0; rL1→r_left1; rR1→r_right1` |  |
| 654 | code | CORE-LETTER WIDTH | `DL→EL; DR→ER; PhiO→PhiU; nU0→n_down0; XiO→XiU` | C2: the coordinate products D, D_L, D_R, DL, DR, DU, DUs are NOT the down store; renamed (E*) first, else the letter swap collides |
| 655 | code+str | CORE-LETTER LETTER-API WIDTH | `nD1→n_up1; nD0→n_up0; O→up` | HAND: ('U' in update,) -> ('up' in update,)  [old 'O'] |
| 657 | code+cmt | WIDTH | `rL0→r_left0; nU0→n_down0; rL1→r_left1 ×2; nU1→n_down1` |  |
| 658 | code+cmt | WIDTH | `nU0→n_down0; rR0→r_right0; rR1→r_right1 ×2; nU1→n_down1` |  |
| 659 | code | CORE-LETTER | `moves_U→moves_D; moves_O→moves_U` |  |
| 660 | code | STORE-WORD | `up→down; down→up` |  |
| 687 | cmt | LETTER-API | -> `update:     typ.Tuple[str, ...],                      # all four stores \| ('down', 'left') \| ('right', 'up')` | HAND |
| 733 | code | CORE-LETTER | `U→D ×2` |  |
| 750 | str | LETTER-API | -> `frame, movement = _aligning_pass(step, frame, hint, project, old, out, sharing, _STORE_NAMES)` | HAND: 'ULRO' -> all four store words |
| 752 | cmt | LETTER-API | -> `if alternate:                                               # (('down', 'left'), ('right', 'up')) pairs while a whole pair fits; the stop` | HAND |
| 755 | str | LETTER-API | -> `for update in (('down', 'left'), ('right', 'up')):` | HAND |
| 759 | cmt | LETTER-API | -> `if not stop and len(movements) < max_passes:           # an odd remainder: one final ('down', 'left') pass` | HAND |
| 760 | str | LETTER-API | -> `frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, ('down', 'left'))` | HAND |
| 762 | cmt | LETTER-API | -> `else:                                                       # all-store passes, the stop tested after each (the` | HAND |
| 765 | str | LETTER-API | -> `frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, _STORE_NAMES)` | HAND |

### `probing.py` (283 rows, 268 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 75 | cmt | UNRELATED | keep |  |
| 76 | cmt | UNRELATED | keep |  |
| 185 | cmt | CORE-LETTER WIDTH | `U→D; nUi→n_down_i` |  |
| 186 | cmt | WIDTH | `nUi→n_down_i` |  |
| 189 | code | CORE-LETTER | `U→D` |  |
| 190 | code+str | CORE-LETTER | `U→D` |  |
| 194 | code+cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 196 | cmt | WIDTH | `nUi→n_down_i` |  |
| 197 | doc | UNRELATED | keep |  |
| 206 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 207 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 211 | code+str | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 213 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 219 | cmt | WIDTH | `rLi→r_left_i` |  |
| 220 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i ×2; rL→r_left` |  |
| 230 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 231 | cmt | WIDTH | `nUi→n_down_i` |  |
| 232 | cmt | WIDTH | `rLi→r_left_i` |  |
| 255 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 256 | cmt | WIDTH | `nUi→n_down_i` |  |
| 257 | cmt | WIDTH | `rR→r_right` |  |
| 275 | cmt | WIDTH | `rLi→r_left_i ×2; nOi→n_up_i; rR→r_right ×2` |  |
| 276 | cmt | WIDTH | `nOi→n_up_i` |  |
| 284 | code+cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 285 | cmt | WIDTH | `rLi→r_left_i` |  |
| 286 | cmt | WIDTH | `rR→r_right` |  |
| 287 | cmt | WIDTH | `nOi→n_up_i` |  |
| 288 | doc | UNRELATED | keep |  |
| 297 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 298 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 302 | code+str | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 304 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 310 | cmt | CORE-LETTER | `U→D` |  |
| 314 | code | CORE-LETTER | `U→D` |  |
| 315 | code+str | CORE-LETTER | `U→D` |  |
| 322 | doc | UNRELATED | keep |  |
| 347 | cmt | WIDTH | `nOi→n_up_i` |  |
| 349 | cmt | WIDTH | `nOi→n_up_i` |  |
| 350 | doc | UNRELATED | keep |  |
| 373 | code | CORE-LETTER | `O→U` |  |
| 379 | doc | CORE-LETTER | `O→U` |  |
| 386 | code+str | CORE-LETTER | `O→U` |  |
| 391 | cmt | WIDTH | `rRi→r_right_i` |  |
| 392 | cmt | CORE-LETTER | `O→U` |  |
| 396 | code | CORE-LETTER | `O→U` |  |
| 397 | code | CORE-LETTER | `O→U` |  |
| 401 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 402 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 403 | code+cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 404 | cmt | WIDTH | `nUi→n_down_i` |  |
| 405 | cmt | WIDTH | `nOi→n_up_i` |  |
| 406 | cmt | WIDTH | `rLi→r_left_i` |  |
| 407 | cmt | WIDTH | `rRi→r_right_i` |  |
| 425 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 430 | code | WIDTH | `rR0→r_right0` |  |
| 431 | code | WIDTH | `rR0→r_right0` |  |
| 433 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 438 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 439 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 440 | code+cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 441 | cmt | WIDTH | `nUi→n_down_i` |  |
| 442 | cmt | WIDTH | `nOi→n_up_i` |  |
| 443 | cmt | WIDTH | `rR→r_right` |  |
| 444 | cmt | WIDTH | `rL→r_left` |  |
| 466 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 475 | cmt | WIDTH | `nUi→n_down_i` |  |
| 489 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 490 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 491 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 492 | cmt | WIDTH | `rLi→r_left_i` |  |
| 493 | cmt | WIDTH | `rR→r_right` |  |
| 494 | cmt | WIDTH | `rRi→r_right_i` |  |
| 495 | cmt | WIDTH | `rL→r_left` |  |
| 496 | cmt | WIDTH | `nUi→n_down_i` |  |
| 497 | doc | UNRELATED | keep |  |
| 521 | cmt | WIDTH | `rR→r_right ×2; nU→n_down` |  |
| 550 | cmt | WIDTH | `nUi→n_down_i` |  |
| 551 | cmt | WIDTH | `nOi→n_up_i` |  |
| 552 | cmt | WIDTH | `nOi→n_up_i` |  |
| 553 | cmt | WIDTH | `nUi→n_down_i` |  |
| 591 | cmt | WIDTH | `nOi→n_up_i` |  |
| 592 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 598 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 617 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 620 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 621 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 623 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 630 | cmt | WIDTH | `nOi→n_up_i` |  |
| 631 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 640 | cmt | CORE-LETTER STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; U_xo→D_xo; U_yo→D_yo; U→D; nU→n_down` |  |
| 641 | cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; O_ixj→U_ixj; O_iyj→U_iyj; O→U; rL→r_left; nO→n_up; rR→r_right` |  |
| 642 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 643 | cmt | CORE-LETTER WIDTH | `rL→r_left; nU→n_down; rR→r_right` | NOTE: pre-existing: Q.shape is (r_right, n_down, r_right), not (r_left, ...) |
| 646 | cmt | ADJECTIVE STORE-WORD WIDTH | -> `NDArray,  # down_tucker_supercore. shape=(d, n_down, N),      down orthogonal elements` | HAND: backwards adjective (blast radius §1.4): the rename makes store word and adjective agree |
| 647 | cmt | ADJECTIVE STORE-WORD WIDTH | -> `NDArray,  # up_tt_supercore.   shape=(d, r_left, n_up, r_right), up orthogonal elements` | HAND: backwards adjective |
| 648 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 649 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 651 | cmt | CORE-LETTER STORE-WORD | `up→down; down→up; U→D; O→U` |  |
| 672 | doc | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 674 | doc | CORE-LETTER | `U→D; O→U` |  |
| 733 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 736 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 742 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 747 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 751 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 759 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 773 | cmt | UNRELATED | keep | AMBIG: 'up-index' = the index of T4S's upward edge variables xi (MSG); it is also the Tucker-store width -- keep, or reword ('xi index') so it cannot be read as a store |
| 779 | cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U→D` |  |
| 780 | cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O→U` |  |
| 783 | cmt | STORE-WORD | `up→down; down→up` |  |
| 786 | cmt | WIDTH | `nUi→n_down_i` |  |
| 787 | cmt | WIDTH | `rLi→r_left_i` |  |
| 788 | cmt | WIDTH | `rR→r_right` |  |
| 789 | cmt | WIDTH | `nOi→n_up_i` |  |
| 804 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 805 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 808 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 813 | cmt | WIDTH | `rR→r_right` |  |
| 814 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i ×2; rR→r_right` |  |
| 823 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 824 | cmt | UNRELATED | keep | AMBIG: 'up-index' = the index of T4S's upward edge variables xi (MSG); it is also the Tucker-store width -- keep, or reword ('xi index') so it cannot be read as a store |
| 826 | cmt | WIDTH | `rR→r_right` |  |
| 841 | cmt | WIDTH | `rR_d→r_right_d` |  |
| 842 | cmt | WIDTH | `rR_d→r_right_d ×2` |  |
| 843 | code | WIDTH | `rR_d→r_right_d` |  |
| 845 | code | WIDTH | `rR_d→r_right_d` |  |
| 856 | cmt | CORE-LETTER WIDTH | `U→D; nUi→n_down_i` |  |
| 857 | cmt | WIDTH | `nUi→n_down_i` |  |
| 860 | code | CORE-LETTER | `U→D` |  |
| 861 | cmt | CORE-LETTER | `U→D` |  |
| 863 | code+str | CORE-LETTER | `U→D` |  |
| 867 | code+cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 869 | cmt | WIDTH | `nUi→n_down_i` |  |
| 870 | doc | UNRELATED | keep |  |
| 879 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 880 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 887 | cmt | CORE-LETTER | `U→D` |  |
| 888 | code+str | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 890 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 896 | cmt | WIDTH | `rLi→r_left_i` |  |
| 912 | cmt | WIDTH | `nUi→n_down_i` |  |
| 913 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 914 | cmt | WIDTH | `nUi→n_down_i` |  |
| 915 | cmt | WIDTH | `rLi→r_left_i` |  |
| 916 | cmt | WIDTH | `rLi→r_left_i` |  |
| 938 | cmt | WIDTH | `nUi→n_down_i` |  |
| 939 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 940 | cmt | WIDTH | `nUi→n_down_i` |  |
| 941 | cmt | WIDTH | `rR→r_right` |  |
| 942 | cmt | WIDTH | `rR→r_right` |  |
| 961 | cmt | CORE-LETTER | `O→U` |  |
| 962 | cmt | WIDTH | `nOi→n_up_i` |  |
| 965 | code | CORE-LETTER | `O→U` |  |
| 966 | cmt | CORE-LETTER | `O→U` |  |
| 967 | cmt | CORE-LETTER | `O→U` |  |
| 968 | code+str | CORE-LETTER | `O→U` |  |
| 969 | code+str | CORE-LETTER | `O→U` |  |
| 974 | cmt | WIDTH | `rR→r_right` |  |
| 975 | cmt | WIDTH | `rLi→r_left_i` |  |
| 976 | code+cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 977 | cmt | WIDTH | `rLi→r_left_i` |  |
| 978 | cmt | WIDTH | `rR→r_right` |  |
| 979 | cmt | WIDTH | `nOi→n_up_i` |  |
| 980 | doc | UNRELATED | keep |  |
| 989 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 995 | cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O→U` |  |
| 996 | code+str | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 997 | code+str | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1000 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1009 | cmt | WIDTH | `nOi→n_up_i ×2` |  |
| 1046 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1048 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1050 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1087 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 1128 | cmt | WIDTH | `rR→r_right` |  |
| 1129 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1130 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1131 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1132 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1133 | cmt | WIDTH | `rR→r_right` |  |
| 1136 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 1179 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 1197 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 1199 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 1202 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1215 | cmt | CORE-LETTER STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; U_xo→D_xo; U_yo→D_yo; U→D; nU→n_down` |  |
| 1216 | cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; O_ixj→U_ixj; O_iyj→U_iyj; O→U; rL→r_left; nO→n_up; rR→r_right` |  |
| 1217 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 1218 | cmt | CORE-LETTER WIDTH | `rL→r_left; nU→n_down; rR→r_right` | NOTE: pre-existing: Q.shape is (r_right, n_down, r_right) |
| 1221 | cmt | ADJECTIVE STORE-WORD WIDTH | -> `NDArray,  # down_tucker_supercore. shape=(d, n_down, N),      down orthogonal elements` | HAND: backwards adjective |
| 1222 | cmt | ADJECTIVE STORE-WORD WIDTH | -> `NDArray,  # up_tt_supercore.   shape=(d, r_left, n_up, r_right), up orthogonal elements` | HAND: backwards adjective |
| 1223 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 1224 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 1226 | cmt | CORE-LETTER STORE-WORD | `up→down; down→up; U→D; O→U` |  |
| 1230 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1231 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 1234 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1235 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 1259 | doc | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 1261 | doc | CORE-LETTER | `U→D; O→U` |  |
| 1322 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 1324 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 1330 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1334 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 1340 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1427 | doc | CORE-LETTER | `O→U; U→D` | C4 |
| 1429 | doc | UNRELATED | keep |  |
| 1457 | cmt | CORE-LETTER | `O→U` | C1 |
| 1459 | cmt | UNRELATED | keep |  |
| 1471 | cmt | CORE-LETTER WIDTH | `U→D; nUi→n_down_i ×2` | C1 |
| 1472 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1473 | doc | CORE-LETTER | `U→D` |  |
| 1475 | code | CORE-LETTER | `U→D; Ud→Dd` |  |
| 1476 | code+str | CORE-LETTER | `Ud→Dd; U→D` |  |
| 1480 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; U→D; nUi→n_down_i` |  |
| 1481 | code+cmt | STORE-WORD WIDTH | `up_tucker_cores_dot→down_tucker_cores_dot; nUi→n_down_i` |  |
| 1483 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1485 | doc | CORE-LETTER | `U→D` | C1 |
| 1487 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; up_tucker_cores_dot→down_tucker_cores_dot` |  |
| 1488 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 1491 | code+str | STORE-WORD | `up_tucker_cores_dot→down_tucker_cores_dot` |  |
| 1492 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 1493 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; up_tucker_cores_dot→down_tucker_cores_dot` |  |
| 1498 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1503 | doc | CORE-LETTER | `O→U` |  |
| 1514 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 1515 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 1516 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1517 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1518 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1519 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1520 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1521 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1522 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1523 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1544 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 1545 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 1546 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1547 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1548 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1549 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1550 | cmt | WIDTH | `rR→r_right` |  |
| 1551 | cmt | WIDTH | `rR→r_right` |  |
| 1552 | cmt | WIDTH | `rR→r_right` |  |
| 1553 | cmt | WIDTH | `rR→r_right` |  |
| 1565 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; O→U; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 1566 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores_dot→up_tt_cores_dot; O_dot→U_dot; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 1567 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1568 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1569 | cmt | WIDTH | `rR→r_right` |  |
| 1570 | cmt | WIDTH | `rR→r_right` |  |
| 1571 | cmt | WIDTH | `rR→r_right` |  |
| 1572 | cmt | WIDTH | `rR→r_right` |  |
| 1573 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1574 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1575 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1576 | doc | CORE-LETTER | `O→U ×2` |  |
| 1578 | doc | CORE-LETTER | `O_dot→U_dot ×2; O→U ×4` |  |
| 1580 | code | STORE-WORD | `down_tt_cores_dot→up_tt_cores_dot; down_tt_cores→up_tt_cores ×2` |  |
| 1582 | code | STORE-WORD | `down_tt_cores_dot→up_tt_cores_dot; down_tt_cores→up_tt_cores ×2` |  |
| 1591 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1602 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1604 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1605 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1622 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 1650 | cmt | WIDTH | `rR→r_right` |  |
| 1651 | cmt | WIDTH | `rR→r_right` |  |
| 1652 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1653 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1654 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1655 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1656 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1657 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1658 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1659 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1660 | cmt | WIDTH | `rR→r_right` |  |
| 1661 | cmt | WIDTH | `rR→r_right` |  |
| 1663 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 1702 | doc | CORE-LETTER | `U→D; O→U` |  |
| 1703 | doc | CORE-LETTER | `U→D ×2` |  |
| 1721 | cmt | CORE-LETTER | `O→U` | C1 |
| 1728 | cmt | UNRELATED | keep |  |
| 1777 | doc | UNRELATED | keep |  |

### `ranks.py` (44 rows, 41 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 80 | doc | WIDTH-GENERIC | `rL_i→r_left_i; rR_i→r_right_i` |  |
| 83 | doc | WIDTH-GENERIC | `rL_i→r_left_i; rR_i→r_right_i` |  |
| 87 | doc | WIDTH-GENERIC | `rL_i→r_left_i; rR_i→r_right_i` |  |
| 136 | code | WIDTH-GENERIC | `rL→r_left` |  |
| 137 | code | WIDTH-GENERIC | `rR→r_right` |  |
| 139 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 145 | code | WIDTH-GENERIC | `rL→r_left` |  |
| 146 | code | WIDTH-GENERIC | `rR→r_right` |  |
| 148 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 149 | code | WIDTH-GENERIC | `rR→r_right ×2; rL→r_left` |  |
| 151 | code | WIDTH-GENERIC | `rR→r_right` |  |
| 334 | doc | UNRELATED | keep |  |
| 535 | doc | ADJECTIVE | keep |  |
| 545 | doc | WIDTH-GENERIC | `rL_i→r_left_i; rR_i→r_right_i` |  |
| 580 | cmt | ADJECTIVE | keep |  |
| 624 | cmt | WIDTH | `nU0→n_down0; nU→n_down` |  |
| 626 | cmt | STORE-WORD | `up_tucker_ranks→down_tucker_ranks` | NOTE: return-annotation labels: first output = Tucker-store width (down), second = TT-store mode width (up); 'tucker' in the second label is a misnomer (it is O's mode leg) -- consider down_ranks / up_ranks |
| 628 | cmt | WIDTH | `nD0→n_up0; nD→n_up` |  |
| 630 | cmt | STORE-WORD | `down_tucker_ranks→up_tucker_ranks` | NOTE: return-annotation labels: first output = Tucker-store width (down), second = TT-store mode width (up); 'tucker' in the second label is a misnomer (it is O's mode leg) -- consider down_ranks / up_ranks |
| 632 | cmt | WIDTH | `rL0→r_left0; rLd→r_left_d` |  |
| 636 | cmt | WIDTH | `rR0→r_right0; rRd→r_right_d` |  |
| 644 | doc | ADJECTIVE STORE-WORD | `up_i→down_i` |  |
| 645 | doc | STORE-WORD | `up_i→down_i` |  |
| 646 | doc | STORE-WORD | `up_i→down_i` |  |
| 647 | doc | STORE-WORD | `down→up; down_i→up_i; up_i→down_i` |  |
| 651 | doc | STORE-WORD | `down→up` |  |
| 663 | code | STORE-WORD | `up→down` |  |
| 667 | code | STORE-WORD | `up→down` |  |
| 672 | code | STORE-WORD | `up→down` |  |
| 674 | code | STORE-WORD | `down→up; up→down` |  |
| 676 | code | STORE-WORD | `up→down; down→up` |  |
| 678 | code | STORE-WORD | `up→down; down→up` |  |
| 710 | code | WIDTH-GENERIC | `rL→r_left` |  |
| 711 | code | WIDTH-GENERIC | `rR→r_right` |  |
| 713 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 715 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right ×2` |  |
| 731 | code | STORE-WORD | `up_ranks→down_ranks` | SILENT: PUBLIC keyword up_ranks -> down_ranks (same spelling, other position): keyword callers silently pass the other store |
| 732 | code | STORE-WORD | `down_ranks→up_ranks` | SILENT: PUBLIC keyword down_ranks -> up_ranks (silent swap) |
| 740 | doc | STORE-WORD | `up→down ×2; down→up` | NOTE: states the store pairing; after the swap reads 'down/up rank stores ... down/left ranks' |
| 742 | doc | STORE-WORD | `up→down` |  |
| 750 | code | STORE-WORD | `up_ranks→down_ranks; down_ranks→up_ranks` |  |
| 754 | code+cmt | STORE-WORD | `up_ranks→down_ranks` |  |
| 756 | code | STORE-WORD | `up_ranks→down_ranks` |  |
| 758 | code | STORE-WORD | `up_ranks→down_ranks` |  |

### `regularization.py` (1 rows, 1 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 58 | doc | CORE-LETTER | `U→D; O→U` |  |

### `sampling_derivatives.py` (537 rows, 465 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 52 | cmt | UNRELATED | keep |  |
| 145 | cmt | WIDTH | `nUi→n_down_i` |  |
| 146 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 162 | doc | UNRELATED | keep |  |
| 191 | cmt | CORE-LETTER | `U_i→D_i` |  |
| 192 | cmt | CORE-LETTER | `U_i→D_i` |  |
| 244 | cmt | WIDTH | `nUi→n_down_i` |  |
| 245 | cmt | WIDTH | `nUi→n_down_i` |  |
| 246 | cmt | WIDTH | `nUi→n_down_i` |  |
| 251 | doc | UNRELATED | keep |  |
| 257 | cmt | WIDTH | `nU→n_down` |  |
| 258 | cmt | WIDTH | `nU→n_down` |  |
| 293 | cmt | UNRELATED | keep |  |
| 294 | cmt | UNRELATED | keep |  |
| 300 | cmt | UNRELATED | keep |  |
| 303 | cmt | UNRELATED | keep |  |
| 315 | code | UNRELATED | keep |  |
| 327 | code | UNRELATED | keep |  |
| 330 | cmt | UNRELATED | keep |  |
| 331 | doc | UNRELATED | keep |  |
| 332 | doc | UNRELATED | keep |  |
| 338 | code | UNRELATED | keep |  |
| 345 | code | UNRELATED | keep |  |
| 348 | code | UNRELATED | keep |  |
| 352 | cmt | UNRELATED | keep |  |
| 355 | code | UNRELATED | keep |  |
| 364 | doc | UNRELATED | keep |  |
| 369 | cmt | WIDTH | `rLi→r_left_i` |  |
| 381 | code+str | UNRELATED | keep |  |
| 388 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 389 | cmt | WIDTH | `nUi→n_down_i` |  |
| 391 | cmt | WIDTH | `rLi→r_left_i` |  |
| 416 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 417 | cmt | WIDTH | `nUi→n_down_i` |  |
| 419 | cmt | WIDTH | `rR→r_right` |  |
| 434 | cmt | WIDTH | `nOi→n_up_i ×2; rR→r_right ×2; rLi→r_left_i` |  |
| 435 | cmt | WIDTH | `rLi→r_left_i` |  |
| 444 | code | UNRELATED | keep |  |
| 450 | cmt | WIDTH | `nOi→n_up_i` |  |
| 473 | cmt | WIDTH | `rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 474 | cmt | WIDTH | `rLi→r_left_i` |  |
| 475 | cmt | WIDTH | `rR→r_right` |  |
| 476 | cmt | WIDTH | `nOi→n_up_i` |  |
| 509 | cmt | CORE-LETTER | `U→D` |  |
| 513 | code | CORE-LETTER | `U→D` |  |
| 514 | code+str | CORE-LETTER | `U→D` |  |
| 518 | cmt | WIDTH | `nUi→n_down_i` |  |
| 519 | cmt | WIDTH | `nUi→n_down_i` |  |
| 523 | doc | CORE-LETTER | `U_i→D_i` |  |
| 563 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 564 | cmt | UNRELATED WIDTH | `nUi→n_down_i` |  |
| 567 | cmt | WIDTH | `nUi→n_down_i` |  |
| 568 | cmt | WIDTH | `rLi→r_left_i` |  |
| 582 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 583 | cmt | UNRELATED WIDTH | `nUi→n_down_i` |  |
| 586 | cmt | WIDTH | `nUi→n_down_i` |  |
| 587 | cmt | WIDTH | `rLi→r_left_i` |  |
| 588 | cmt | WIDTH | `rR→r_right` |  |
| 589 | cmt | WIDTH | `nOi→n_up_i` |  |
| 594 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 597 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 605 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 610 | cmt | WIDTH | `nUi→n_down_i` |  |
| 611 | cmt | WIDTH | `rLi→r_left_i` |  |
| 613 | doc | CORE-LETTER UNRELATED | `U_i→D_i ×2` |  |
| 618 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 619 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores ×2` |  |
| 627 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 632 | cmt | WIDTH | `nUi→n_down_i` |  |
| 633 | cmt | WIDTH | `rLi→r_left_i` |  |
| 636 | doc | UNRELATED | keep |  |
| 638 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 639 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores ×2` |  |
| 647 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 652 | cmt | WIDTH | `nUi→n_down_i` |  |
| 653 | cmt | WIDTH | `rLi→r_left_i` |  |
| 654 | cmt | WIDTH | `rR→r_right` |  |
| 655 | cmt | WIDTH | `nOi→n_up_i` |  |
| 661 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 662 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores ×2` |  |
| 671 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 672 | cmt | CORE-LETTER | `U→D` | C1 |
| 695 | code | CORE-LETTER | `O→U` |  |
| 707 | code+cmt+str | CORE-LETTER | `Odxi→Udxi; O→U` |  |
| 710 | code | UNRELATED | keep |  |
| 711 | code+str | UNRELATED | keep |  |
| 713 | code+cmt+str | CORE-LETTER | `Odxi→Udxi` |  |
| 717 | code+str | CORE-LETTER | `Odxi→Udxi` |  |
| 722 | cmt | WIDTH | `rRi→r_right_i` |  |
| 723 | cmt | CORE-LETTER | `O→U` |  |
| 728 | code | CORE-LETTER | `O→U` |  |
| 729 | code | CORE-LETTER | `O→U` |  |
| 733 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 734 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 735 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; O→U; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 736 | cmt | WIDTH | `nUi→n_down_i` |  |
| 737 | cmt | WIDTH | `nOi→n_up_i` |  |
| 738 | cmt | WIDTH | `rLi→r_left_i` |  |
| 739 | cmt | WIDTH | `rRi→r_right_i` |  |
| 749 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 755 | code | WIDTH | `rR0→r_right0` |  |
| 756 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 757 | code | WIDTH | `rR0→r_right0` |  |
| 758 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 763 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 764 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 765 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; O→U; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 766 | cmt | WIDTH | `nUi→n_down_i` |  |
| 767 | cmt | WIDTH | `nOi→n_up_i` |  |
| 768 | cmt | WIDTH | `rR→r_right` |  |
| 769 | cmt | WIDTH | `rL→r_left` |  |
| 771 | doc | CORE-LETTER | `O→U` |  |
| 776 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 783 | cmt | WIDTH | `rLi→r_left_i; rRi→r_right_i` |  |
| 798 | code | UNRELATED | keep |  |
| 805 | cmt | WIDTH | `nUi→n_down_i` |  |
| 829 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 830 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 831 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 832 | cmt | WIDTH | `rLi→r_left_i` |  |
| 833 | cmt | WIDTH | `rR→r_right` |  |
| 834 | cmt | WIDTH | `rRi→r_right_i` |  |
| 835 | cmt | WIDTH | `rL→r_left` |  |
| 836 | cmt | WIDTH | `nUi→n_down_i` |  |
| 859 | cmt | CORE-LETTER | `U→D` | C1 |
| 863 | code | CORE-LETTER | `U→D` |  |
| 864 | cmt | CORE-LETTER | `U→D` | C1 |
| 865 | cmt | CORE-LETTER | `U→D` |  |
| 866 | code | CORE-LETTER | `U→D` |  |
| 867 | code+str | CORE-LETTER | `U→D` |  |
| 872 | cmt | CORE-LETTER WIDTH | `U→D; nUi→n_down_i` |  |
| 873 | cmt | WIDTH | `nOi→n_up_i` |  |
| 874 | cmt | WIDTH | `nOi→n_up_i` |  |
| 875 | cmt | WIDTH | `nUi→n_down_i` |  |
| 877 | doc | CORE-LETTER | `U_i→D_i` |  |
| 886 | cmt | CORE-LETTER | `U→D` | C1 |
| 887 | cmt | CORE-LETTER WIDTH | `U→D; nU→n_down` |  |
| 899 | cmt | WIDTH | `nOi→n_up_i` |  |
| 900 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 907 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 917 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 922 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 923 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 925 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 932 | cmt | WIDTH | `nOi→n_up_i` |  |
| 933 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 936 | cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U→D` |  |
| 937 | cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O→U` |  |
| 940 | cmt | CORE-LETTER STORE-WORD | `up→down; down→up; U→D; O→U` |  |
| 955 | doc | UNRELATED | keep |  |
| 980 | cmt | UNRELATED | keep |  |
| 984 | cmt | WIDTH | `rLi→r_left_i` |  |
| 994 | cmt | WIDTH | `nUi→n_down_i` |  |
| 995 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 1017 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1018 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 1037 | cmt | WIDTH | `rRi→r_right_i` |  |
| 1038 | cmt | CORE-LETTER | `O→U` |  |
| 1047 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1048 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1049 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1050 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 1051 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; O→U; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 1052 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 1060 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1066 | code | WIDTH | `rR0→r_right0` |  |
| 1067 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1068 | code | WIDTH | `rR0→r_right0` |  |
| 1070 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1076 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1077 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 1084 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 1094 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1099 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1106 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1107 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 1110 | cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U→D` |  |
| 1111 | cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O→U` |  |
| 1114 | cmt | CORE-LETTER STORE-WORD | `up→down; down→up; U→D; O→U` |  |
| 1140 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1141 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 1147 | doc | UNRELATED | keep |  |
| 1160 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1161 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 1168 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 1176 | doc | UNRELATED | keep |  |
| 1178 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1183 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1190 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1191 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 1194 | cmt | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores; U→D` |  |
| 1195 | cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O→U` |  |
| 1198 | cmt | CORE-LETTER STORE-WORD | `up→down; down→up; U→D; O→U` |  |
| 1204 | doc | UNRELATED | keep |  |
| 1230 | cmt | UNRELATED | keep |  |
| 1235 | cmt | CORE-LETTER | `U→D` |  |
| 1236 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1237 | doc | UNRELATED | keep |  |
| 1239 | code | CORE-LETTER | `U→D` |  |
| 1240 | code+str | CORE-LETTER | `U→D` |  |
| 1244 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; U→D; nUi→n_down_i` |  |
| 1246 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1247 | doc | CORE-LETTER UNRELATED | `U_i→D_i` |  |
| 1250 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 1251 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 1256 | cmt | CORE-LETTER | `U→D` |  |
| 1257 | code+str | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 1259 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 1264 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1265 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 1266 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1268 | cmt | WIDTH | `rL→r_left` |  |
| 1271 | doc | UNRELATED | keep |  |
| 1278 | code+str | UNRELATED | keep |  |
| 1283 | cmt | UNRELATED WIDTH | `rL→r_left ×2; nUi→n_down_i ×2; rLi→r_left_i` |  |
| 1284 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1288 | doc | UNRELATED | keep |  |
| 1295 | code | UNRELATED | keep |  |
| 1314 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1325 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1346 | code | WIDTH | `rL0→r_left0` |  |
| 1348 | code | WIDTH | `rL0→r_left0` |  |
| 1354 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 1355 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1356 | cmt | UNRELATED WIDTH | `nUi→n_down_i` |  |
| 1357 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1358 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1365 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 1366 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1367 | cmt | UNRELATED WIDTH | `nUi→n_down_i` |  |
| 1368 | cmt | WIDTH | `rR→r_right` |  |
| 1369 | cmt | WIDTH | `rR→r_right` |  |
| 1378 | cmt | CORE-LETTER WIDTH | `O→U; nOi→n_up_i ×2; rLi→r_left_i; rR→r_right` |  |
| 1379 | cmt | UNRELATED WIDTH | `rLi→r_left_i ×2; rR→r_right ×2` |  |
| 1381 | doc | CORE-LETTER UNRELATED | `O→U` | C3 |
| 1382 | doc | CORE-LETTER UNRELATED | `O→U ×3` | C3 |
| 1383 | doc | CORE-LETTER | `O→U` |  |
| 1384 | code | CORE-LETTER | `O→U` |  |
| 1386 | code+cmt | CORE-LETTER | `O→U ×2` |  |
| 1387 | code+str | CORE-LETTER | `mO→mU; O→U` |  |
| 1388 | code+str | CORE-LETTER | `On→Un; O→U` |  |
| 1389 | code+str | CORE-LETTER | `mO→mU` |  |
| 1390 | code+str | CORE-LETTER | `On→Un` |  |
| 1391 | code | CORE-LETTER | `O→U` |  |
| 1395 | cmt | CORE-LETTER | `O→U` |  |
| 1397 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1398 | doc | UNRELATED | keep |  |
| 1400 | code | CORE-LETTER | `O→U` |  |
| 1402 | code+cmt | CORE-LETTER | `O→U ×2` |  |
| 1403 | code | CORE-LETTER | `O→U` |  |
| 1404 | code | UNRELATED | keep |  |
| 1419 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; O→U; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 1420 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1421 | cmt | WIDTH | `rR→r_right` |  |
| 1422 | cmt | WIDTH | `rR→r_right` |  |
| 1423 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1424 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1425 | doc | UNRELATED | keep |  |
| 1426 | doc | CORE-LETTER | `O→U ×2` |  |
| 1431 | doc | CORE-LETTER UNRELATED | `O→U ×2` | C3 |
| 1433 | doc | UNRELATED | keep |  |
| 1436 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1437 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1440 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 1447 | cmt | WIDTH | `nOi→n_up_i ×2` |  |
| 1476 | cmt | UNRELATED WIDTH | `nOi→n_up_i` |  |
| 1479 | cmt | UNRELATED WIDTH | `nOi→n_up_i` | HAND AMBIG: 'down eta' / 'frame down jets' = T4S's DOWNWARD edge variables eta (travel direction, MSG): keep 'down' here although eta contracts the (new) UP store; generated text wrongly flips it -- apply only the width/identifier changes on this line |
| 1482 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1510 | cmt | UNRELATED WIDTH | `nOi→n_up_i` |  |
| 1513 | cmt | UNRELATED WIDTH | `nOi→n_up_i` | HAND AMBIG: 'down eta' / 'frame down jets' = T4S's DOWNWARD edge variables eta (travel direction, MSG): keep 'down' here although eta contracts the (new) UP store; generated text wrongly flips it -- apply only the width/identifier changes on this line |
| 1517 | cmt | WIDTH | `nOi→n_up_i` |  |
| 1519 | doc | WIDTH | `nO→n_up` |  |
| 1549 | doc | UNRELATED | keep |  |
| 1550 | doc | UNRELATED | keep |  |
| 1563 | code | UNRELATED | keep |  |
| 1578 | doc | UNRELATED | keep |  |
| 1579 | doc | UNRELATED | keep |  |
| 1592 | code+str | UNRELATED | keep |  |
| 1593 | code+str | UNRELATED | keep |  |
| 1594 | cmt | WIDTH | `nU→n_down` |  |
| 1602 | cmt | WIDTH | `rR→r_right` |  |
| 1603 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1604 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1605 | cmt | WIDTH | `nUi→n_down_i` |  |
| 1606 | cmt | WIDTH | `rLi→r_left_i` |  |
| 1607 | cmt | WIDTH | `rR→r_right` |  |
| 1610 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 1742 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 1784 | cmt | UNRELATED | keep |  |
| 1788 | cmt | WIDTH | `nO→n_up` |  |
| 1819 | code | WIDTH | `nU→n_down` |  |
| 1820 | code+cmt | STORE-WORD WIDTH | `nO→n_up; nU→n_down; down→up` |  |
| 1822 | code+cmt | WIDTH | `nU→n_down` |  |
| 1823 | code+cmt | WIDTH | `nO→n_up` |  |
| 1824 | code+cmt | WIDTH | `nU→n_down; nO→n_up` |  |
| 1855 | doc | UNRELATED | keep |  |
| 1870 | code | WIDTH | `nU→n_down` |  |
| 1875 | code+cmt | CORE-LETTER WIDTH | `nU→n_down ×4; U→D; O→U` |  |
| 1888 | code | WIDTH | `nU→n_down` |  |
| 1889 | code+cmt | STORE-WORD WIDTH | `nO→n_up; nU→n_down; down→up` |  |
| 1891 | code+cmt | WIDTH | `nU→n_down ×2` |  |
| 1892 | code+cmt | WIDTH | `nU→n_down ×2; nO→n_up` |  |
| 1893 | code+cmt | WIDTH | `nO→n_up` |  |
| 1906 | code | WIDTH | `nU→n_down` |  |
| 1911 | code | WIDTH | `nU→n_down ×2` |  |
| 1912 | code | WIDTH | `nU→n_down ×2` |  |
| 1921 | cmt | WIDTH-GENERIC | `nU_1→n_down_1; nU_d→n_down_d` |  |
| 1979 | cmt | WIDTH-GENERIC | `nU_1→n_down_1; nU_d→n_down_d` |  |
| 1997 | doc | UNRELATED | keep |  |
| 2021 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 2036 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 2040 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 2043 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 2060 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 2102 | cmt | CORE-LETTER | `O→U` | C1 |
| 2108 | cmt | WIDTH | `rR→r_right` |  |
| 2120 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 2121 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2123 | cmt | WIDTH | `rR→r_right` |  |
| 2127 | doc | WIDTH | `rR_d→r_right_d` |  |
| 2139 | cmt | WIDTH | `rR_d→r_right_d` |  |
| 2140 | cmt | WIDTH | `rR_d→r_right_d` |  |
| 2141 | code | WIDTH | `rR_d→r_right_d` |  |
| 2143 | code | WIDTH | `rR_d→r_right_d` |  |
| 2157 | doc | UNRELATED | keep |  |
| 2158 | doc | UNRELATED | keep |  |
| 2167 | code+str | UNRELATED | keep |  |
| 2184 | cmt | CORE-LETTER WIDTH | `O→U; nOi→n_up_i ×2; rLi→r_left_i; rR→r_right` |  |
| 2185 | cmt | UNRELATED WIDTH | `rLi→r_left_i; rR→r_right` |  |
| 2187 | doc | CORE-LETTER UNRELATED | `O→U` | C3 |
| 2188 | doc | CORE-LETTER | `O→U` |  |
| 2189 | code | CORE-LETTER | `O→U` |  |
| 2191 | code+str | CORE-LETTER | `mO→mU; O→U` |  |
| 2192 | code+str | CORE-LETTER | `mO→mU; O→U ×2` |  |
| 2196 | cmt | CORE-LETTER | `O→U` |  |
| 2198 | cmt | WIDTH | `nOi→n_up_i` |  |
| 2199 | doc | CORE-LETTER UNRELATED | `O→U` |  |
| 2201 | code | CORE-LETTER | `O→U` |  |
| 2203 | code+cmt | CORE-LETTER | `O→U ×2` |  |
| 2204 | code | CORE-LETTER UNRELATED | `O→U` |  |
| 2220 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2221 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2223 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; O→U; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 2224 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 2227 | cmt | WIDTH | `nOi→n_up_i` |  |
| 2228 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 2234 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 2235 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 2240 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 2255 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 2269 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 2273 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 2283 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 2315 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 2329 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 2331 | code+cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 2333 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 2338 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; U→D; nUi→n_down_i` |  |
| 2344 | code+cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 2345 | code+cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 2346 | code+cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 2358 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 2367 | doc | UNRELATED | keep |  |
| 2386 | cmt | CORE-LETTER | `O→U; U→D` | C4 |
| 2408 | doc | CORE-LETTER | `O→U` | C4 |
| 2464 | cmt | CORE-LETTER | `O→U` | C1 |
| 2466 | cmt | UNRELATED | keep |  |
| 2483 | cmt | CORE-LETTER | `U→D` | C1 |
| 2484 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2485 | doc | CORE-LETTER | `U→D` |  |
| 2487 | code | CORE-LETTER | `U→D; Ud→Dd` |  |
| 2488 | code+str | CORE-LETTER | `Ud→Dd; U→D` |  |
| 2492 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; U→D; nUi→n_down_i` |  |
| 2493 | code+cmt | STORE-WORD WIDTH | `up_tucker_cores_dot→down_tucker_cores_dot; nUi→n_down_i` |  |
| 2495 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2499 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; up_tucker_cores_dot→down_tucker_cores_dot` |  |
| 2500 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 2503 | code+str | STORE-WORD | `up_tucker_cores_dot→down_tucker_cores_dot` |  |
| 2504 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 2505 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; up_tucker_cores_dot→down_tucker_cores_dot` |  |
| 2510 | cmt | UNRELATED | keep |  |
| 2511 | cmt | WIDTH | `rLi→r_left_i ×2` |  |
| 2514 | doc | UNRELATED | keep |  |
| 2515 | doc | UNRELATED | keep |  |
| 2527 | code | UNRELATED | keep |  |
| 2531 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2532 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2533 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 2534 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 2535 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2536 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2539 | cmt | WIDTH | `rL→r_left` |  |
| 2554 | code+str | UNRELATED | keep |  |
| 2555 | code+str | UNRELATED | keep |  |
| 2561 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2562 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2563 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 2564 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 2565 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2566 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2567 | cmt | UNRELATED WIDTH | `nUi→n_down_i` |  |
| 2568 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2569 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2570 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2573 | cmt | WIDTH | `rL→r_left` |  |
| 2609 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2621 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2644 | code | WIDTH | `rL0→r_left0` |  |
| 2647 | code | WIDTH | `rL0→r_left0` |  |
| 2654 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 2655 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 2656 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2657 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2658 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2659 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2660 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2661 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2662 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2663 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2673 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 2674 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 2675 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2676 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2677 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2678 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2679 | cmt | WIDTH | `rR→r_right` |  |
| 2680 | cmt | WIDTH | `rR→r_right` |  |
| 2681 | cmt | WIDTH | `rR→r_right` |  |
| 2682 | cmt | WIDTH | `rR→r_right` |  |
| 2694 | cmt | CORE-LETTER WIDTH | `O→U; O_dot→U_dot; nOi→n_up_i` |  |
| 2699 | doc | CORE-LETTER UNRELATED | `O→U ×3` | C3 |
| 2700 | doc | CORE-LETTER | `O→U` |  |
| 2701 | code | CORE-LETTER | `O→U; Od→Ud` |  |
| 2703 | code+cmt | CORE-LETTER | `O→U ×2` |  |
| 2704 | code+str | CORE-LETTER | `mO→mU; O→U` |  |
| 2705 | code+cmt+str | CORE-LETTER | `dmO→dmU; O→U` |  |
| 2706 | code+cmt+str | CORE-LETTER | `Od→Ud` |  |
| 2707 | code+str | CORE-LETTER | `On→Un; O→U` |  |
| 2708 | code+cmt+str | CORE-LETTER | `dOn→dUn; Od→Ud` |  |
| 2709 | code+cmt+str | CORE-LETTER | `O→U` |  |
| 2710 | code+str | CORE-LETTER | `dmO→dmU` |  |
| 2711 | code+str | CORE-LETTER | `mO→mU` |  |
| 2712 | code+str | CORE-LETTER | `On→Un` |  |
| 2713 | code+str | CORE-LETTER | `dOn→dUn` |  |
| 2714 | code | CORE-LETTER | `O→U; Od→Ud` |  |
| 2718 | cmt | CORE-LETTER | `O→U; O_dot→U_dot` |  |
| 2720 | cmt | WIDTH | `nOi→n_up_i` |  |
| 2723 | code | CORE-LETTER | `O→U; Od→Ud` |  |
| 2725 | code+cmt | CORE-LETTER | `O→U` |  |
| 2726 | code | UNRELATED | keep |  |
| 2727 | code | CORE-LETTER | `O→U; Od→Ud` |  |
| 2742 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; O→U; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 2743 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `down_tt_cores_dot→up_tt_cores_dot; O_dot→U_dot; rLi→r_left_i; nOi→n_up_i; rR→r_right` |  |
| 2744 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2745 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2746 | cmt | WIDTH | `rR→r_right` |  |
| 2747 | cmt | WIDTH | `rR→r_right` |  |
| 2748 | cmt | WIDTH | `rR→r_right` |  |
| 2749 | cmt | WIDTH | `rR→r_right` |  |
| 2750 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2751 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2752 | cmt | WIDTH | `nOi→n_up_i` |  |
| 2754 | doc | CORE-LETTER | `O→U ×2` |  |
| 2756 | code | STORE-WORD | `down_tt_cores→up_tt_cores; down_tt_cores_dot→up_tt_cores_dot` |  |
| 2758 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 2761 | code | STORE-WORD | `down_tt_cores→up_tt_cores; down_tt_cores_dot→up_tt_cores_dot` |  |
| 2768 | cmt | WIDTH | `nOi→n_up_i` |  |
| 2801 | cmt | WIDTH | `nOi→n_up_i` |  |
| 2804 | cmt | WIDTH | `nOi→n_up_i` |  |
| 2807 | cmt | WIDTH | `nOi→n_up_i` |  |
| 2832 | doc | UNRELATED | keep |  |
| 2845 | code | UNRELATED | keep |  |
| 2861 | doc | UNRELATED | keep |  |
| 2862 | doc | UNRELATED | keep |  |
| 2874 | code | UNRELATED | keep |  |
| 2911 | cmt | WIDTH | `rR→r_right` |  |
| 2912 | cmt | WIDTH | `rR→r_right` |  |
| 2913 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2914 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2915 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2916 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2917 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2918 | cmt | WIDTH | `nUi→n_down_i` |  |
| 2919 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2920 | cmt | WIDTH | `rLi→r_left_i` |  |
| 2921 | cmt | WIDTH | `rR→r_right` |  |
| 2922 | cmt | WIDTH | `rR→r_right` |  |
| 2925 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 2969 | doc | CORE-LETTER | `U→D; O→U` |  |
| 2970 | doc | CORE-LETTER | `U→D` |  |
| 2971 | doc | CORE-LETTER | `U→D` |  |
| 2989 | cmt | CORE-LETTER | `O→U` | C1 |
| 2996 | cmt | UNRELATED | keep |  |
| 3049 | doc | UNRELATED | keep |  |
| 3101 | cmt | CORE-LETTER | `O→U` |  |
| 3122 | cmt | WIDTH | `rR→r_right` |  |
| 3134 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 3135 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 3136 | cmt | WIDTH | `nUi→n_down_i` |  |
| 3137 | cmt | WIDTH | `nUi→n_down_i` |  |
| 3138 | cmt | WIDTH | `rR→r_right` |  |
| 3139 | cmt | WIDTH | `rR→r_right` |  |
| 3153 | code+cmt | WIDTH | `rR_d→r_right_d` |  |
| 3155 | code+cmt | WIDTH | `rR_d→r_right_d` |  |
| 3163 | cmt | CORE-LETTER WIDTH | `O→U; O_dot→U_dot; nOi→n_up_i` |  |
| 3167 | doc | CORE-LETTER UNRELATED | `O→U ×3` | C3 |
| 3168 | doc | CORE-LETTER UNRELATED | `O→U ×2; O_dot→U_dot` | C3 |
| 3170 | code | CORE-LETTER | `O→U; Od→Ud` |  |
| 3172 | code+cmt | CORE-LETTER | `O→U ×2` |  |
| 3173 | code+str | CORE-LETTER | `mO→mU; O→U` |  |
| 3174 | code+cmt+str | CORE-LETTER | `dmO→dmU; O→U` |  |
| 3175 | code+cmt+str | CORE-LETTER | `Od→Ud` |  |
| 3176 | code+str | CORE-LETTER | `dmO→dmU` |  |
| 3177 | code+str | CORE-LETTER | `mO→mU` |  |
| 3178 | code | CORE-LETTER | `O→U; Od→Ud` |  |
| 3182 | cmt | CORE-LETTER | `O→U; O_dot→U_dot` |  |
| 3184 | cmt | WIDTH | `nOi→n_up_i` |  |
| 3185 | doc | CORE-LETTER UNRELATED | `O→U` |  |
| 3188 | code | CORE-LETTER | `O→U; Od→Ud` |  |
| 3190 | code+cmt | CORE-LETTER | `O→U` |  |
| 3191 | code | CORE-LETTER | `O→U; Od→Ud` |  |
| 3192 | code | UNRELATED | keep |  |
| 3208 | cmt | WIDTH | `nOi→n_up_i; rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 3212 | doc | UNRELATED | keep |  |
| 3220 | code | UNRELATED | keep |  |
| 3232 | cmt | WIDTH | `nUi→n_down_i` |  |
| 3233 | cmt | WIDTH | `nUi→n_down_i` |  |
| 3234 | cmt | WIDTH | `rLi→r_left_i` |  |
| 3236 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 3237 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 3239 | cmt | WIDTH | `nOi→n_up_i` |  |
| 3240 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 3252 | cmt | CORE-LETTER | `O→U` |  |
| 3282 | doc | CORE-LETTER | `U→D; O→U` |  |
| 3283 | doc | CORE-LETTER | `U→D ×3` |  |
| 3386 | doc | CORE-LETTER | `U→D` |  |
| 3415 | doc | UNRELATED | keep |  |

### `sharing.py` (41 rows, 39 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 479 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 480 | cmt | WIDTH | `sum_nD→sum_n_up` |  |
| 487 | cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 488 | cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 489 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 490 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 501 | doc | CORE-LETTER | `O_i→U_i` |  |
| 506 | doc | CORE-LETTER | `O_i→U_i` |  |
| 507 | doc | CORE-LETTER STORE-WORD | `down→up; O2_i→U2_i` | O2_i (up-matricization of the outer core) -> U2_i |
| 510 | doc | WIDTH | `nD_i→n_up_i` |  |
| 550 | cmt | CORE-LETTER | `O_i→U_i` |  |
| 552 | cmt | CORE-LETTER | `O→U` |  |
| 557 | cmt | WIDTH | `nDi→n_up_i; nU→n_down` |  |
| 561 | cmt | WIDTH | `nD→n_up; nU→n_down` |  |
| 576 | cmt | WIDTH | `nDi→n_up_i` |  |
| 577 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 619 | doc | CORE-LETTER | -> `>>> raw = (tuple(np.random.randn(U.shape[-2], D.shape[-1])` | HAND: doctest: O -> U and U -> D together (the mechanical O->U alone collides) |
| 620 | doc | CORE-LETTER | -> `...              for U, D in zip(frame_d[1], frame_d[0])),` | HAND: doctest: as 619 |
| 641 | cmt+str | WIDTH | `nD→n_up` |  |
| 653 | cmt | WIDTH | `nD_i→n_up_i` |  |
| 656 | doc | WIDTH | `nD_i→n_up_i` |  |
| 670 | cmt | WIDTH | `nDi→n_up_i` |  |
| 684 | cmt | WIDTH | `nD→n_up` |  |
| 695 | cmt | WIDTH | `nDi→n_up_i` |  |
| 704 | cmt+str | WIDTH | `nD→n_up` |  |
| 710 | cmt | WIDTH | `nDi→n_up_i` |  |
| 711 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rRi→r_right_i` |  |
| 722 | doc | CORE-LETTER | keep | C1: [U_g \| Udot]: T3 letters (the shared group factor and its direction); kept |
| 730 | cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 731 | cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 732 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 733 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 814 | cmt | WIDTH | `rLi→r_left_i; rRi→r_right_i` |  |
| 848 | cmt | WIDTH | `nDi→n_up_i` |  |
| 849 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 998 | cmt | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 1002 | doc | CORE-LETTER | `O_i→U_i` |  |
| 1007 | doc | UNRELATED | keep |  |
| 1018 | doc | WIDTH | `nD→n_up` |  |
| 1020 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 1071 | doc | WIDTH | `nDi→n_up_i` |  |

### `stacking.py` (2 rows, 0 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 54 | cmt | UNRELATED | keep |  |
| 242 | cmt | UNRELATED | keep |  |

### `t3_constructors.py` (2 rows, 2 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 29 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 73 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |

### `t3_conversions.py` (5 rows, 5 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 113 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 142 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 144 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 157 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 160 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |

### `t3_linalg.py` (34 rows, 29 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 156 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rRi→r_right_i` |  |
| 157 | code | WIDTH-GENERIC | `rLi→r_left_i; rRi→r_right_i` |  |
| 159 | code | WIDTH-GENERIC | `rLi→r_left_i; rRi→r_right_i` |  |
| 534 | doc | UNRELATED | keep |  |
| 537 | doc | ADJECTIVE UNRELATED | keep |  |
| 576 | code | ADJECTIVE | keep |  |
| 577 | code | ADJECTIVE | keep |  |
| 666 | cmt | CORE-LETTER WIDTH | `U→D; nUi→n_down_i` |  |
| 667 | cmt | WIDTH | `nUi→n_down_i` |  |
| 668 | doc | STORE-WORD | `up→down` |  |
| 671 | code | CORE-LETTER | `U→D` |  |
| 673 | code+str | CORE-LETTER | `BU→BD; U→D` |  |
| 674 | code+str | CORE-LETTER | `BU→BD` |  |
| 680 | cmt | CORE-LETTER | `O→U; U→D` |  |
| 682 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 683 | cmt | WIDTH | `nOi→n_up_i` |  |
| 687 | code | CORE-LETTER | `O→U; U→D` |  |
| 690 | code+str | CORE-LETTER | `BU→BD; U→D` |  |
| 691 | code+str | CORE-LETTER | `BU→BD` |  |
| 692 | code+str | CORE-LETTER | `O→U` |  |
| 699 | cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 700 | cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 701 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 702 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 709 | cmt | WIDTH | `nDi→n_up_i` |  |
| 710 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 717 | doc | CORE-LETTER | keep | C1: corewise frame (U, G, G, G) kept in T3 letters |
| 719 | doc | STORE-WORD | `up→down` |  |
| 723 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 724 | code | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 732 | cmt | STORE-WORD | `up→down` |  |
| 734 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 749 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 770 | doc | CORE-LETTER | `O→U` | C4 C1 |

### `t3_operations.py` (9 rows, 8 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 40 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 43 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 45 | doc | UNRELATED | keep |  |
| 95 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 217 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 254 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 264 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 265 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 273 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |

### `t3_orthogonalization.py` (45 rows, 18 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 7 | doc | ADJECTIVE | keep | HAND: 'the up/down core sweeps' names t3_up_orthogonalize_tt_cores / t3_down_orthogonalize_tucker_cores (adjective convention): KEEP, do not swap |
| 21 | str | ADJECTIVE | keep |  |
| 22 | str | ADJECTIVE | keep |  |
| 23 | str | ADJECTIVE | keep |  |
| 26 | str | ADJECTIVE | keep |  |
| 27 | str | ADJECTIVE | keep |  |
| 38 | doc | ADJECTIVE | keep |  |
| 79 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` | NOTE: type-comment label of a plain-T3 function's output (t3_left_orthogonalize): the Tucker cores after down-orthogonalization; flips with the store name |
| 84 | code | ADJECTIVE STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 86 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 92 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 97 | code | ADJECTIVE STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 99 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 102 | code | ADJECTIVE | keep |  |
| 103 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 106 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 108 | doc | ADJECTIVE | keep |  |
| 115 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 116 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 120 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 128 | code | ADJECTIVE | keep |  |
| 134 | doc | UNRELATED | keep |  |
| 140 | code | ADJECTIVE | keep |  |
| 144 | code | ADJECTIVE | keep |  |
| 145 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 147 | cmt | ADJECTIVE STORE-WORD | -> `NDArray,  # new_Uxo.  down-orthogonal tucker core, C+(x,Ni)` | HAND: backwards adjective (store convention): the Tucker core is DOWN-orthogonal |
| 148 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 150 | doc | ADJECTIVE | keep |  |
| 165 | code | ADJECTIVE | keep |  |
| 168 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 171 | doc | ADJECTIVE UNRELATED | -> `"""Down-orthogonalize the Tucker cores, pushing the remainders up onto the TT cores above.` | HAND: was 'Orthogonalize Tucker cores upwards, ...' (backwards; blast radius §1.4) |
| 177 | code | ADJECTIVE STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 178 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 181 | code | ADJECTIVE | keep |  |
| 192 | doc | UNRELATED | keep |  |
| 199 | code | ADJECTIVE | keep |  |
| 299 | code | ADJECTIVE | keep |  |
| 322 | code | ADJECTIVE | keep |  |
| 336 | code | ADJECTIVE | keep |  |
| 347 | doc | UNRELATED | keep |  |
| 354 | code | ADJECTIVE | keep |  |
| 381 | code | ADJECTIVE | keep |  |
| 387 | code | ADJECTIVE | keep |  |
| 392 | code | ADJECTIVE | keep |  |
| 408 | code | ADJECTIVE | keep |  |

### `t3_svd.py` (33 rows, 19 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 67 | doc | ADJECTIVE | keep |  |
| 103 | cmt | ADJECTIVE | keep |  |
| 105 | code | ADJECTIVE | keep |  |
| 115 | code+cmt | ADJECTIVE | keep |  |
| 130 | code | ADJECTIVE | keep |  |
| 131 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 132 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 133 | doc | ADJECTIVE | keep |  |
| 182 | doc | ADJECTIVE | keep |  |
| 187 | doc | ADJECTIVE | keep | AMBIG: 'group core's up leg': the leg that becomes rows in the up-matricization (ORTH family) -- keep, or reword to 'mode leg'; a mechanical flip to 'down leg' would be wrong |
| 225 | code | ADJECTIVE | keep |  |
| 257 | code | ADJECTIVE | keep |  |
| 263 | code | ADJECTIVE | keep |  |
| 312 | doc | ADJECTIVE STORE-WORD | `up→down` | HAND: 'Tucker up-SVD' is backwards (it calls t3_down_svd_tucker_core): -> 'Tucker down-SVD' |
| 318 | doc | WIDTH-GENERIC | `rL_i→r_left_i ×2; rR_i→r_right_i ×2` |  |
| 345 | cmt | ADJECTIVE STORE-WORD | `up→down` | HAND: 'Tucker up-SVD' -> 'Tucker down-SVD' (calls t3_down_svd_tucker_core) |
| 346 | code | ADJECTIVE | keep |  |
| 349 | code+cmt | ADJECTIVE WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 351 | cmt | WIDTH-GENERIC | `rR→r_right` |  |
| 354 | code+cmt | ADJECTIVE | keep |  |
| 388 | doc | ADJECTIVE | keep | AMBIG: as 187 |
| 642 | code | WIDTH-GENERIC | `rL→r_left` |  |
| 648 | code | WIDTH-GENERIC | `rL→r_left` |  |
| 650 | code | WIDTH-GENERIC | `rR→r_right` |  |
| 653 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 770 | code | WIDTH-GENERIC | `rL→r_left` |  |
| 774 | code | WIDTH-GENERIC | `rL→r_left` |  |
| 775 | code | WIDTH-GENERIC | `rL→r_left` |  |
| 786 | code+cmt | WIDTH-GENERIC | `rL→r_left ×2` |  |
| 788 | code | WIDTH-GENERIC | `rL→r_left` |  |
| 791 | code | WIDTH-GENERIC | `rR→r_right` |  |
| 793 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 797 | code+str | WIDTH-GENERIC | `rR→r_right` |  |

### `tt_operations.py` (7 rows, 7 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 8 | doc | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 33 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 37 | cmt | WIDTH-GENERIC | `rR→r_right; rLi→r_left_i` |  |
| 49 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 67 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 127 | cmt | WIDTH-GENERIC | `rLi→r_left_i; rR→r_right` |  |
| 130 | cmt | WIDTH-GENERIC | `new_rLi→new_r_left_i; new_rR→new_r_right` |  |

### `tt_orthogonalization.py` (5 rows, 5 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 26 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 37 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 51 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 52 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 53 | code+cmt | WIDTH-GENERIC | `rL→r_left ×2; rR→r_right ×3` |  |

### `tv_operations.py` (107 rows, 104 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 53 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 54 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 70 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 83 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 91 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 92 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 108 | doc | CORE-LETTER | `U_i→D_i` |  |
| 121 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 134 | cmt | STORE-WORD | `up→down` |  |
| 136 | code | CORE-LETTER STORE-WORD | `U→D; up_tucker_cores→down_tucker_cores` |  |
| 137 | code+str | CORE-LETTER | `U→D ×2` |  |
| 148 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 149 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 164 | doc | CORE-LETTER STORE-WORD | `U→D; down→up` |  |
| 171 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 181 | cmt | CORE-LETTER STORE-WORD | `U→D; down→up` |  |
| 183 | code | CORE-LETTER STORE-WORD | `U→D; up_tucker_cores→down_tucker_cores` |  |
| 185 | code | CORE-LETTER STORE-WORD | `O→U; down_tt_cores→up_tt_cores` |  |
| 188 | code+str | CORE-LETTER | `U→D` |  |
| 189 | code+str | CORE-LETTER | `U→D` |  |
| 191 | code+str | CORE-LETTER | `O→U` |  |
| 207 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 208 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 224 | doc | CORE-LETTER | `U_i→D_i` |  |
| 232 | doc | CORE-LETTER | keep | C1: [U_g; Udot]: T3 letters kept |
| 233 | doc | CORE-LETTER | `O_i→U_i` |  |
| 234 | doc | CORE-LETTER | keep | C1: [U_g; V_i]: T3 letter kept |
| 235 | doc | WIDTH | `nD_i→n_up_i` |  |
| 241 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 247 | code | STORE-WORD | `down_tt_cores→up_tt_cores ×2` |  |
| 251 | code+cmt | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores; O_i→U_i` |  |
| 253 | code | STORE-WORD | `down_tt_cores→up_tt_cores ×2` |  |
| 260 | cmt | UNRELATED | keep |  |
| 264 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 265 | code | CORE-LETTER STORE-WORD | `up_tucker_cores→down_tucker_cores ×2; U→D ×2` |  |
| 266 | code | CORE-LETTER STORE-WORD | `down_tt_cores→up_tt_cores ×2; O→U ×2` |  |
| 270 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 272 | cmt | CORE-LETTER | `U_i→D_i` |  |
| 273 | code | CORE-LETTER STORE-WORD | `U→D ×2; up_tucker_cores→down_tucker_cores` |  |
| 283 | code | CORE-LETTER STORE-WORD | `O→U; down_tt_cores→up_tt_cores` |  |
| 286 | code | CORE-LETTER | `O→U` |  |
| 293 | code | CORE-LETTER STORE-WORD | `O→U; down_tt_cores→up_tt_cores` |  |
| 295 | code | CORE-LETTER | `O→U ×2` |  |
| 297 | code | CORE-LETTER | `O→U` |  |
| 305 | code | CORE-LETTER STORE-WORD | `O→U; down_tt_cores→up_tt_cores` |  |
| 308 | code | CORE-LETTER | `O→U` |  |
| 309 | code | CORE-LETTER | `O→U` |  |
| 310 | code | CORE-LETTER | `O→U` |  |
| 317 | code | CORE-LETTER | `O→U` |  |
| 325 | code | CORE-LETTER STORE-WORD | `O→U; down_tt_cores→up_tt_cores` |  |
| 326 | code | CORE-LETTER | `O→U` |  |
| 332 | code | CORE-LETTER | `O→U` |  |
| 341 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 342 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 373 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 374 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 392 | doc | CORE-LETTER | `U→D` |  |
| 393 | doc | CORE-LETTER | `O→U` |  |
| 398 | doc | CORE-LETTER | `U→D ×2` |  |
| 400 | doc | CORE-LETTER | `U_i→D_i` |  |
| 401 | doc | CORE-LETTER STORE-WORD | `O_i→U_i; O→U; down→up` | C1: 'dU_i = <U_i, core_env_i> (the Tucker variation; U = the outer/up cores)': the T3 variation letter dU next to the new frame letter U |
| 406 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 407 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 414 | code+cmt | CORE-LETTER STORE-WORD | `U→D ×2; up_tucker_cores→down_tucker_cores` |  |
| 423 | code+cmt+str | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 439 | code+cmt+str | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 443 | code+str | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 444 | code+str | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 453 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 484 | cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 485 | cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 486 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 487 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 490 | cmt | WIDTH | `nDi→n_up_i` |  |
| 491 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 495 | cmt | WIDTH | `nUi→n_down_i` |  |
| 496 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 504 | doc | CORE-LETTER | `U→D` |  |
| 506 | doc | CORE-LETTER | `U→D` |  |
| 547 | doc | CORE-LETTER | `U→D ×2` |  |
| 549 | doc | CORE-LETTER | `U→D` |  |
| 573 | cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 574 | cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 575 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 576 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 579 | cmt | WIDTH | `nUi→n_down_i` |  |
| 580 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 584 | cmt | WIDTH | `nDi→n_up_i` |  |
| 585 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 620 | doc | CORE-LETTER | `U→D ×3` |  |
| 656 | cmt | STORE-WORD WIDTH | `up→down; nU→n_down` |  |
| 657 | cmt | WIDTH | `nD→n_up` |  |
| 756 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 757 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 770 | doc | STORE-WORD | `up→down` |  |
| 772 | doc | CORE-LETTER | `U→D` |  |
| 773 | doc | CORE-LETTER | `U→D` |  |
| 777 | doc | CORE-LETTER | `U→D` |  |
| 814 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 827 | code | CORE-LETTER STORE-WORD | `U→D; up_tucker_cores→down_tucker_cores` |  |
| 828 | code+str | CORE-LETTER | `U→D` |  |
| 855 | cmt | STORE-WORD | `up_tucker_cores→down_tucker_cores` |  |
| 856 | cmt | STORE-WORD | `down_tt_cores→up_tt_cores` |  |
| 873 | doc | STORE-WORD | `up→down` |  |
| 884 | code | STORE-WORD | `up_tucker_cores→down_tucker_cores; down_tt_cores→up_tt_cores` |  |
| 886 | code | CORE-LETTER STORE-WORD | `up_ranks→down_ranks; U→D ×2; up_tucker_cores→down_tucker_cores` |  |
| 890 | code | STORE-WORD | `up_ranks→down_ranks` |  |

### `ufv_conversions.py` (119 rows, 117 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 34 | cmt | UNRELATED | keep |  |
| 37 | doc | UNRELATED | keep |  |
| 71 | cmt | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 73 | cmt | STORE-WORD | `up→down; down→up` |  |
| 78 | cmt | STORE-WORD | `up→down; down→up` |  |
| 96 | doc | STORE-WORD | `up→down; down→up` |  |
| 107 | code | CORE-LETTER | `uc→dc; dc→uc` | C2: locals (uc, dc) swap to (dc, uc): `dc` is the house alias of `dataclasses` in 7 other backend modules (not imported here); consider c_down/c_up |
| 111 | code | STORE-WORD | `up_ranks→down_ranks; down_ranks→up_ranks` |  |
| 114 | code | CORE-LETTER WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right; uc→dc; dc→uc` |  |
| 115 | code | CORE-LETTER STORE-WORD WIDTH | `um→dm; dm→um; up_ranks→down_ranks; down_ranks→up_ranks; nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 117 | code | CORE-LETTER | `uc→dc; dc→uc; um→dm; dm→um` |  |
| 118 | code | CORE-LETTER | `um→dm; dm→um` |  |
| 143 | cmt | STORE-WORD | `up_tucker_supercore→down_tucker_supercore` |  |
| 144 | cmt | STORE-WORD | `down_tt_supercore→up_tt_supercore` |  |
| 148 | cmt | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 153 | cmt | STORE-WORD | `up_cores→down_cores; down_cores→up_cores` |  |
| 162 | code | STORE-WORD | `up_supercore→down_supercore; down_supercore→up_supercore` |  |
| 163 | code | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 164 | code+cmt | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 165 | code | STORE-WORD | `up_supercore→down_supercore` |  |
| 166 | code | STORE-WORD | `up_supercore→down_supercore` |  |
| 168 | cmt | STORE-WORD | `up→down; down→up` |  |
| 169 | code | STORE-WORD | `up_cores→down_cores; down_cores→up_cores` |  |
| 171 | code | STORE-WORD | `up_inds→down_inds; up_mask→down_mask` |  |
| 172 | code | STORE-WORD | `down_inds→up_inds; down_mask→up_mask` |  |
| 179 | code | STORE-WORD | `up_cores→down_cores; up_supercore→down_supercore; up_inds→down_inds` |  |
| 180 | code | STORE-WORD | `down_cores→up_cores; down_supercore→up_supercore; down_inds→up_inds` |  |
| 181 | code | STORE-WORD | `up_inds→down_inds` |  |
| 182 | code | STORE-WORD | `up_inds→down_inds` |  |
| 184 | code | STORE-WORD | `up_cores→down_cores; down_cores→up_cores` |  |
| 187 | code | STORE-WORD | `up_supercore→down_supercore` |  |
| 189 | code | STORE-WORD | `up_supercore→down_supercore; down_supercore→up_supercore` |  |
| 191 | code | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 199 | cmt | STORE-WORD WIDTH | `up_tucker_cores→down_tucker_cores; nUi→n_down_i` |  |
| 200 | cmt | STORE-WORD WIDTH | `down_tt_cores→up_tt_cores; rLi→r_left_i; nDi→n_up_i; rR→r_right` |  |
| 201 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rL→r_left` |  |
| 202 | cmt | WIDTH | `rRi→r_right_i; nUi→n_down_i; rR→r_right` |  |
| 205 | code+cmt | STORE-WORD WIDTH | `nU→n_down; up→down; up_ranks→down_ranks` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 206 | code+cmt | STORE-WORD WIDTH | `nD→n_up; down→up; down_ranks→up_ranks` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 207 | code+cmt | WIDTH | `rL→r_left` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 208 | code+cmt | WIDTH | `rR→r_right` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 210 | cmt | STORE-WORD WIDTH | `up_tucker_supercore→down_tucker_supercore; nU→n_down` |  |
| 211 | cmt | STORE-WORD WIDTH | `down_tt_supercore→up_tt_supercore; rL→r_left; nD→n_up; rR→r_right` |  |
| 212 | cmt | WIDTH | `rL→r_left ×2; nU→n_down` |  |
| 213 | cmt | WIDTH | `rR→r_right ×2; nU→n_down` |  |
| 215 | cmt | STORE-WORD | `up→down; down→up` |  |
| 222 | doc | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 225 | code | STORE-WORD | `up_cores→down_cores; down_cores→up_cores` |  |
| 226 | code | STORE-WORD | `up_cores→down_cores` |  |
| 228 | code | STORE-WORD | `up_cores→down_cores` |  |
| 230 | code+cmt | CORE-LETTER STORE-WORD | `U→D ×2; up_cores→down_cores` |  |
| 231 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `up_ranks→down_ranks; U→D ×2; up_cores→down_cores; nU→n_down` |  |
| 232 | code+cmt | STORE-WORD WIDTH | `down_ranks→up_ranks; down_cores→up_cores; nD→n_up` |  |
| 233 | cmt | WIDTH | `rL→r_left` |  |
| 234 | cmt | WIDTH | `rR→r_right` |  |
| 237 | code | STORE-WORD WIDTH | `nU→n_down ×3; up_ranks→down_ranks` |  |
| 238 | code | STORE-WORD WIDTH | `nD→n_up ×3; down_ranks→up_ranks` |  |
| 239 | code | WIDTH | `rL→r_left ×3` |  |
| 240 | code | WIDTH | `rR→r_right ×3` |  |
| 242 | code | STORE-WORD WIDTH | `up_sc→down_sc; up_cores→down_cores; nU→n_down` |  |
| 243 | code | STORE-WORD WIDTH | `down_sc→up_sc; down_cores→up_cores; rL→r_left; nD→n_up; rR→r_right` |  |
| 244 | code | WIDTH | `rL→r_left ×2; nU→n_down` |  |
| 245 | code | WIDTH | `rR→r_right ×2; nU→n_down` |  |
| 248 | code | STORE-WORD | `up_ranks→down_ranks` |  |
| 249 | code | STORE-WORD | `down_ranks→up_ranks` |  |
| 252 | code | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 254 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 259 | cmt | WIDTH | `nD→n_up` |  |
| 260 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 262 | cmt | STORE-WORD | `up→down; down→up` |  |
| 274 | doc | STORE-WORD WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right; up→down` |  |
| 276 | code | STORE-WORD | `vup→vdown; vdown→vup` |  |
| 277 | code+cmt | STORE-WORD | `vup→vdown; vdown→vup` |  |
| 284 | code | STORE-WORD | `up_inds→down_inds; vup→vdown` |  |
| 285 | code | STORE-WORD | `down_inds→up_inds; vdown→vup` |  |
| 290 | code | STORE-WORD | `down_inds→up_inds` |  |
| 291 | code | STORE-WORD | `up_inds→down_inds` |  |
| 300 | code | STORE-WORD | `vup→vdown; vdown→vup` |  |
| 308 | cmt | WIDTH | `nDi→n_up_i` |  |
| 309 | cmt | WIDTH | `rLi→r_left_i; nUi→n_down_i; rR→r_right` |  |
| 312 | code+cmt | STORE-WORD WIDTH | `nU→n_down; up→down; up_ranks→down_ranks` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 313 | code+cmt | STORE-WORD WIDTH | `nD→n_up; down→up; down_ranks→up_ranks` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 314 | code+cmt | WIDTH | `rL→r_left` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 315 | code+cmt | WIDTH | `rR→r_right` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 317 | cmt | WIDTH | `nD→n_up` |  |
| 318 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 320 | cmt | STORE-WORD | `up→down; down→up` |  |
| 326 | doc | WIDTH | `rLi→r_left_i; rR→r_right` |  |
| 334 | code+cmt | STORE-WORD WIDTH | `up_ranks→down_ranks; nUi→n_down_i` |  |
| 335 | code+cmt | STORE-WORD WIDTH | `down_ranks→up_ranks; nDi→n_up_i` |  |
| 336 | cmt | WIDTH | `rLi→r_left_i` |  |
| 337 | cmt | WIDTH | `rR→r_right` |  |
| 340 | code | STORE-WORD WIDTH | `nU→n_down ×3; up_ranks→down_ranks` |  |
| 341 | code | STORE-WORD WIDTH | `nD→n_up ×3; down_ranks→up_ranks` |  |
| 342 | code | WIDTH | `rL→r_left ×3` |  |
| 343 | code | WIDTH | `rR→r_right ×3` |  |
| 345 | code | WIDTH | `nD→n_up` |  |
| 346 | code | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 349 | code | STORE-WORD | `up_ranks→down_ranks` |  |
| 350 | code | STORE-WORD | `down_ranks→up_ranks` |  |
| 353 | code | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 360 | cmt | STORE-WORD WIDTH | `up_weights→down_weights; nUi→n_down_i` |  |
| 361 | cmt | STORE-WORD WIDTH | `down_weights→up_weights; nDi→n_up_i` |  |
| 362 | cmt | WIDTH | `rLi→r_left_i` |  |
| 363 | cmt | WIDTH | `rRi→r_right_i` |  |
| 365 | code+cmt | STORE-WORD WIDTH | `nU→n_down; up→down` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 366 | code+cmt | STORE-WORD WIDTH | `nD→n_up; down→up` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 367 | code+cmt | WIDTH | `rL→r_left` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 368 | code+cmt | WIDTH | `rR→r_right` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (old nU = Tucker-store width = n_down) |
| 370 | cmt | STORE-WORD WIDTH | `up_weight_supercore→down_weight_supercore; nU→n_down` |  |
| 371 | cmt | STORE-WORD WIDTH | `down_weight_supercore→up_weight_supercore; nD→n_up` |  |
| 372 | cmt | WIDTH | `rL→r_left` |  |
| 373 | cmt | WIDTH | `rR→r_right` |  |
| 383 | doc | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 387 | code | STORE-WORD | `up_weights→down_weights; down_weights→up_weights` |  |
| 388 | code+cmt | STORE-WORD | `up_weights→down_weights` |  |
| 391 | code | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 403 | cmt | STORE-WORD | `up→down; down→up` |  |
| 407 | cmt | STORE-WORD | `up→down; down→up` |  |

### `ufv_masking.py` (72 rows, 72 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 28 | cmt | STORE-WORD | `up→down; down→up` |  |
| 29 | cmt | STORE-WORD | `up→down; down→up` |  |
| 33 | doc | STORE-WORD | `up→down; down→up` |  |
| 36 | code | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 37 | code | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 50 | doc | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 58 | doc | STORE-WORD | `down→up` |  |
| 65 | code+cmt | STORE-WORD | `up_ranks→down_ranks` | SILENT: PUBLIC ufv_make_frame_masks: positional order kept (Tucker first); the keyword names swap: up_ranks= now names the 2nd positional slot |
| 66 | code+cmt | STORE-WORD | `down_ranks→up_ranks` | SILENT: as 65 |
| 69 | code | WIDTH | `nU→n_down` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (nU is the Tucker-store width = n_down) |
| 70 | code | WIDTH | `nD→n_up` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (nU is the Tucker-store width = n_down) |
| 71 | code | WIDTH | `rL→r_left` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (nU is the Tucker-store width = n_down) |
| 72 | code | WIDTH | `rR→r_right` | API: PUBLIC keyword nU/nD/rL/rR -> n_down/n_up/r_left/r_right (nU is the Tucker-store width = n_down) |
| 74 | cmt | STORE-WORD WIDTH | `up_mask→down_mask; nU→n_down` |  |
| 75 | cmt | STORE-WORD WIDTH | `down_mask→up_mask; nD→n_up` |  |
| 76 | cmt | WIDTH | `rL→r_left` |  |
| 77 | cmt | WIDTH | `rR→r_right` |  |
| 82 | code | STORE-WORD WIDTH | `up_mask→down_mask; up_ranks→down_ranks; nU→n_down` |  |
| 83 | code | STORE-WORD WIDTH | `down_mask→up_mask; down_ranks→up_ranks; nD→n_up` |  |
| 84 | code | WIDTH | `rL→r_left` |  |
| 85 | code | WIDTH | `rR→r_right` |  |
| 86 | code | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 91 | cmt | STORE-WORD WIDTH | `up_tucker_supercore→down_tucker_supercore; nU→n_down` |  |
| 92 | cmt | STORE-WORD WIDTH | `down_tt_supercore→up_tt_supercore; rL→r_left; nD→n_up; rR→r_right` |  |
| 93 | cmt | WIDTH | `rL→r_left ×2; nU→n_down` |  |
| 94 | cmt | WIDTH | `rR→r_right ×2; nU→n_down` |  |
| 97 | cmt | STORE-WORD WIDTH | `up_mask→down_mask; nU→n_down` |  |
| 98 | cmt | STORE-WORD WIDTH | `down_mask→up_mask; nD→n_up` |  |
| 99 | cmt | WIDTH | `rL→r_left` |  |
| 100 | cmt | WIDTH | `rR→r_right` |  |
| 104 | cmt | STORE-WORD | `masked_up_tucker_supercore→masked_down_tucker_supercore` |  |
| 105 | cmt | STORE-WORD | `masked_down_tt_supercore→masked_up_tt_supercore` |  |
| 112 | code | STORE-WORD | `up_tucker_supercore→down_tucker_supercore; down_tt_supercore→up_tt_supercore` |  |
| 113 | code | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 114 | code+cmt | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 116 | code | STORE-WORD | `up_tucker_supercore→down_tucker_supercore` |  |
| 117 | code | STORE-WORD | `up_tucker_supercore→down_tucker_supercore` |  |
| 118 | code | STORE-WORD WIDTH | `nU→n_down; up_tucker_supercore→down_tucker_supercore` |  |
| 119 | code | STORE-WORD | `up_tucker_supercore→down_tucker_supercore` |  |
| 120 | code | STORE-WORD WIDTH | `rL→r_left; down_tt_supercore→up_tt_supercore` |  |
| 121 | code | STORE-WORD WIDTH | `nD→n_up; down_tt_supercore→up_tt_supercore` |  |
| 122 | code | STORE-WORD WIDTH | `rR→r_right; down_tt_supercore→up_tt_supercore` |  |
| 127 | code | STORE-WORD WIDTH | `up_mask→down_mask; nU→n_down` |  |
| 128 | code | STORE-WORD WIDTH | `up_mask→down_mask; nU→n_down` |  |
| 129 | code | STORE-WORD WIDTH | `down_mask→up_mask; nD→n_up` |  |
| 130 | code | WIDTH | `rL→r_left` |  |
| 131 | code | WIDTH | `rL→r_left` |  |
| 132 | code | WIDTH | `rR→r_right` |  |
| 133 | code | WIDTH | `rR→r_right` |  |
| 135 | code | STORE-WORD | `masked_up_tucker_supercore→masked_down_tucker_supercore; up_tucker_supercore→down_tucker_supercore` |  |
| 136 | code | STORE-WORD | `masked_down_tt_supercore→masked_up_tt_supercore; down_tt_supercore→up_tt_supercore` |  |
| 141 | code | STORE-WORD | `masked_up_tucker_supercore→masked_down_tucker_supercore; masked_down_tt_supercore→masked_up_tt_supercore` |  |
| 148 | cmt | WIDTH | `nD→n_up` |  |
| 149 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 152 | cmt | STORE-WORD WIDTH | `variations_up_mask→variations_down_mask; nU→n_down` |  |
| 153 | cmt | STORE-WORD WIDTH | `variations_down_mask→variations_up_mask; nD→n_up` |  |
| 154 | cmt | WIDTH | `rL→r_left` |  |
| 155 | cmt | WIDTH | `rR→r_right` |  |
| 165 | code | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 166 | code+cmt | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 170 | code | WIDTH | `nD→n_up` |  |
| 172 | code | WIDTH | `rL→r_left` |  |
| 173 | code | WIDTH | `nU→n_down` |  |
| 174 | code | WIDTH | `rR→r_right` |  |
| 178 | cmt | STORE-WORD WIDTH | `nD→n_up ×2; variations_down→variations_up` |  |
| 180 | code | STORE-WORD WIDTH | `down_mask→up_mask; nD→n_up` |  |
| 181 | cmt | STORE-WORD WIDTH | `rL→r_left ×2; nU→n_down ×2; rR→r_right ×2; up→down` |  |
| 182 | code | WIDTH | `rL→r_left` |  |
| 183 | code | STORE-WORD WIDTH | `up_mask→down_mask; nU→n_down` |  |
| 184 | code | WIDTH | `rR→r_right` |  |
| 196 | cmt | STORE-WORD | `up→down; down→up` |  |
| 216 | doc | STORE-WORD | `up→down` |  |

### `ufv_operations.py` (102 rows, 100 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 59 | cmt | STORE-WORD | `up→down; down→up` |  |
| 71 | cmt | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 73 | cmt | WIDTH | `nD→n_up` |  |
| 74 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 83 | doc | STORE-WORD WIDTH | `nU→n_down; up→down ×2; nD→n_up; down→up; rL→r_left; rR→r_right` |  |
| 85 | doc | WIDTH | `rL→r_left` |  |
| 90 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 91 | code+cmt | STORE-WORD | `up_sc→down_sc` |  |
| 92 | code | STORE-WORD WIDTH | `nU→n_down; up_sc→down_sc ×2` |  |
| 93 | code | STORE-WORD WIDTH | `nD→n_up; rL→r_left; rR→r_right; down_sc→up_sc` |  |
| 94 | code | WIDTH | `nD→n_up; rL→r_left; nU→n_down; rR→r_right` |  |
| 205 | cmt | WIDTH | `nD→n_up` |  |
| 225 | doc | STORE-WORD | `up→down; down→up` |  |
| 226 | doc | STORE-WORD | `down→up` |  |
| 228 | code | CORE-LETTER STORE-WORD | `up_sc→down_sc; down_sc→up_sc; um→dm; dm→um` | NOTE: mask locals um/dm swap (same-spelling) |
| 231 | code | STORE-WORD | `up_sc→down_sc` |  |
| 232 | code | STORE-WORD | `down_sc→up_sc` |  |
| 236 | code+cmt | CORE-LETTER STORE-WORD | `um→dm; dm→um; up→down; down→up` |  |
| 244 | code | STORE-WORD | `vup→vdown; vdown→vup` |  |
| 249 | code+cmt | STORE-WORD | `vup→vdown; vdown→vup; up→down; down→up` |  |
| 276 | doc | CORE-LETTER STORE-WORD | `up→down; U→D` |  |
| 277 | doc | CORE-LETTER STORE-WORD | `down→up; O→U` |  |
| 282 | code | CORE-LETTER STORE-WORD | `up_sc→down_sc; down_sc→up_sc; um→dm; dm→um` | NOTE: mask locals um/dm swap (same-spelling) |
| 283 | code+cmt | STORE-WORD | `mup→mdown; mdown→mup` |  |
| 284 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 286 | code | STORE-WORD | `up_sc→down_sc` |  |
| 287 | code | STORE-WORD WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right; mup→mdown; mdown→mup` |  |
| 293 | code+cmt+str | CORE-LETTER STORE-WORD WIDTH | `mup→mdown ×2; um→dm; nU→n_down; up→down` |  |
| 294 | code+cmt+str | CORE-LETTER STORE-WORD WIDTH | `mdown→mup ×2; dm→um; nD→n_up; down→up` |  |
| 297 | code+cmt+str | WIDTH | `rL→r_left` |  |
| 298 | code+cmt+str | WIDTH | `rR→r_right` |  |
| 303 | cmt | STORE-WORD | `up→down; down→up` |  |
| 339 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 341 | code | STORE-WORD | `up_sc→down_sc` |  |
| 349 | code | STORE-WORD WIDTH | `up_sc→down_sc ×2; nU→n_down; nD→n_up; down_sc→up_sc; rL→r_left; rR→r_right` |  |
| 382 | cmt | STORE-WORD WIDTH | `up_tucker_supercore→down_tucker_supercore; nU→n_down` |  |
| 383 | cmt | STORE-WORD WIDTH | `down_tt_supercore→up_tt_supercore; rL→r_left; nD→n_up; rR→r_right` |  |
| 384 | cmt | WIDTH | `rL→r_left ×2; nU→n_down` |  |
| 385 | cmt | WIDTH | `rR→r_right ×2; nU→n_down` |  |
| 387 | cmt | STORE-WORD | `up→down; down→up` |  |
| 396 | doc | UNRELATED | keep |  |
| 398 | doc | CORE-LETTER | `O_i→U_i` |  |
| 403 | doc | WIDTH | `nU→n_down ×2; nD→n_up ×2` |  |
| 404 | doc | WIDTH | `rR→r_right; rL→r_left` |  |
| 407 | doc | WIDTH | `rL→r_left; rR→r_right` |  |
| 435 | doc | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 451 | code | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 452 | code+cmt | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 453 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 454 | code | STORE-WORD | `up_sc→down_sc ×2` |  |
| 455 | code | WIDTH | `rL→r_left; rR→r_right` |  |
| 457 | cmt | CORE-LETTER | `O_i→U_i` |  |
| 458 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 459 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 461 | code+cmt+str | STORE-WORD WIDTH | `down_sc→up_sc; nD→n_up; nU→n_down` |  |
| 468 | code | STORE-WORD | `down_mask→up_mask; up_mask→down_mask` |  |
| 470 | code+cmt | STORE-WORD WIDTH | `up_mask→down_mask; down_mask→up_mask; nU→n_down; nD→n_up` |  |
| 471 | code | STORE-WORD | `down_mask→up_mask; up_mask→down_mask` |  |
| 473 | code | WIDTH | `rR→r_right; rL→r_left` |  |
| 477 | code | WIDTH | `rL→r_left; rR→r_right` |  |
| 479 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 483 | cmt | WIDTH | `rR→r_right; rL→r_left` |  |
| 491 | cmt | STORE-WORD | `up→down; down→up` |  |
| 496 | cmt | WIDTH | `nD_i→n_up_i; nU→n_down` |  |
| 499 | doc | WIDTH | `nD_i→n_up_i` |  |
| 500 | doc | WIDTH | `nU→n_down` |  |
| 501 | doc | WIDTH | `nD_i→n_up_i; nU→n_down` |  |
| 502 | doc | WIDTH | `nD_i→n_up_i; nU→n_down` |  |
| 503 | code | STORE-WORD | `up→down; down→up` |  |
| 504 | code+cmt | STORE-WORD WIDTH | `down→up; nD_i→n_up_i` |  |
| 505 | code+cmt | STORE-WORD WIDTH | `up→down; nU→n_down` |  |
| 512 | cmt | STORE-WORD WIDTH | `up_tucker_supercore→down_tucker_supercore; nU→n_down` |  |
| 513 | cmt | STORE-WORD WIDTH | `down_tt_supercore→up_tt_supercore; rL→r_left; nD→n_up; rR→r_right` |  |
| 514 | cmt | WIDTH | `rL→r_left ×2; nU→n_down` |  |
| 515 | cmt | WIDTH | `rR→r_right ×2; nU→n_down` |  |
| 517 | cmt | STORE-WORD | `up→down; down→up` |  |
| 533 | doc | WIDTH | `nD_i→n_up_i; nU→n_down` |  |
| 535 | doc | WIDTH | `nD_i→n_up_i; nU→n_down` |  |
| 562 | doc | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 567 | doc | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 584 | cmt | STORE-WORD | `up→down; down→up` |  |
| 587 | doc | STORE-WORD WIDTH | `down→up; nD→n_up` |  |
| 588 | doc | STORE-WORD WIDTH | `up→down; nU→n_down; rL→r_left; rR→r_right` |  |
| 607 | code | STORE-WORD | `up→down; down→up` |  |
| 609 | code+str | STORE-WORD | `down→up` |  |
| 610 | code+str | STORE-WORD | `up→down` |  |
| 616 | cmt | STORE-WORD | `up→down; down→up` |  |
| 627 | doc | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 633 | code | STORE-WORD | `up→down; down→up` |  |
| 637 | code | STORE-WORD | `up→down` |  |
| 642 | code | WIDTH | `nD→n_up` |  |
| 643 | code | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 644 | code | STORE-WORD WIDTH | `up→down; nU→n_down; down→up; nD→n_up; rL→r_left; rR→r_right` |  |
| 649 | cmt | UNRELATED | keep |  |
| 661 | cmt | STORE-WORD | `up→down; down→up` |  |
| 665 | doc | STORE-WORD | `up→down; down→up` |  |
| 669 | doc | WIDTH | `nD→n_up` |  |
| 670 | doc | WIDTH | `nU→n_down ×2; nD→n_up` |  |
| 671 | doc | STORE-WORD | `down→up` |  |
| 674 | code | STORE-WORD | `up→down; down→up` |  |
| 677 | code | STORE-WORD | `up→down; down→up` |  |
| 681 | cmt | STORE-WORD | `up→down; down→up` |  |

### `uniform_fitting.py` (11 rows, 9 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 74 | cmt | STORE-WORD | `up→down` |  |
| 78 | cmt | STORE-WORD | `up→down` |  |
| 81 | doc | STORE-WORD | `up→down` |  |
| 151 | doc | STORE-WORD | `up→down` |  |
| 179 | doc | STORE-WORD | `up→down` |  |
| 215 | doc | STORE-WORD | `up→down` |  |
| 246 | doc | STORE-WORD | `up→down` |  |
| 277 | doc | STORE-WORD | `up→down` |  |
| 316 | doc | STORE-WORD | `up→down` |  |
| 488 | doc | UNRELATED | keep |  |
| 504 | doc | UNRELATED | keep |  |

### `uniform_padded_frames.py` (192 rows, 192 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 47 | cmt | STORE-WORD WIDTH | `up→down; nU→n_down ×3; down→up; rL→r_left ×3; nD→n_up; rR→r_right ×3` |  |
| 48 | cmt | STORE-WORD WIDTH | `up→down; nU→n_down ×2; down→up; nD→n_up ×2; rL→r_left ×2; rR→r_right ×2` |  |
| 50 | cmt | WIDTH | `nD→n_up; rL→r_left; nU→n_down; rR→r_right` |  |
| 89 | cmt | STORE-WORD | `down→up` |  |
| 138 | cmt | STORE-WORD | `up→down` |  |
| 139 | cmt | STORE-WORD | `down→up` |  |
| 143 | cmt | STORE-WORD | `up→down; down→up` |  |
| 150 | doc | STORE-WORD | `up→down; down→up` |  |
| 151 | doc | STORE-WORD | `up→down; down→up` |  |
| 183 | code | CORE-LETTER | `U0→D0; O0→U0` |  |
| 185 | code | CORE-LETTER | `U0→D0; O0→U0` |  |
| 186 | code | CORE-LETTER | `mU0→m_down0; mD0→m_up0; mL0→m_left0; mR0→m_right0` |  |
| 188 | code | CORE-LETTER | `U0→D0` |  |
| 189 | code | CORE-LETTER | `U0→D0` |  |
| 190 | code | CORE-LETTER | `U0→D0` |  |
| 192 | code+cmt | CORE-LETTER | `mU0→m_down0; mD0→m_up0; mL0→m_left0; mR0→m_right0` |  |
| 194 | str | STORE-WORD | `up→down; down→up` |  |
| 196 | str | STORE-WORD | `up→down` |  |
| 203 | code+cmt | WIDTH | `kU→k_down; kD→k_up; kL→k_left; kR→k_right` |  |
| 204 | code+str | STORE-WORD WIDTH | `up→down; down→up; kU→k_down; kD→k_up; kL→k_left; kR→k_right` |  |
| 209 | code | CORE-LETTER | `U0→D0; O0→U0` |  |
| 210 | code | WIDTH | `kU→k_down; kD→k_up; kL→k_left; kR→k_right` |  |
| 212 | code+cmt | CORE-LETTER | `U0→D0; O0→U0` |  |
| 213 | code | CORE-LETTER | `U0→D0 ×2; O0→U0 ×2` |  |
| 215 | code | CORE-LETTER | `mU0→m_down0 ×2; mD0→m_up0 ×2; mL0→m_left0 ×2; mR0→m_right0 ×2` |  |
| 216 | code | CORE-LETTER WIDTH | `mU→m_down; mU0→m_down0; kU→k_down` |  |
| 217 | code | CORE-LETTER WIDTH | `mD→m_up; mD0→m_up0; kD→k_up` |  |
| 218 | code | CORE-LETTER WIDTH | `mL→m_left; mL0→m_left0; kL→k_left` |  |
| 219 | code | CORE-LETTER WIDTH | `mR→m_right; mR0→m_right0; kR→k_right` |  |
| 224 | code | CORE-LETTER WIDTH | `mU0→m_down0 ×2; kU→k_down ×2` |  |
| 225 | str | STORE-WORD | `up→down` |  |
| 230 | code+cmt | CORE-LETTER | `old_U→old_D; mU0→m_down0` |  |
| 233 | code | WIDTH | `kU→k_down` |  |
| 235 | code+str | CORE-LETTER STORE-WORD | `up→down; old_U→old_D` |  |
| 238 | code | CORE-LETTER | `new_U→new_D` |  |
| 239 | code | CORE-LETTER | `U0→D0; mU0→m_down0` |  |
| 240 | code | WIDTH | `kU→k_down` |  |
| 241 | code+cmt | CORE-LETTER | `U→D; U0→D0; new_U→new_D` |  |
| 245 | code+str | CORE-LETTER WIDTH | `mL→m_left; mU→m_down; mL0→m_left0; kL→k_left` |  |
| 246 | code | CORE-LETTER WIDTH | `mL0→m_left0; kL→k_left` |  |
| 251 | code+str | CORE-LETTER WIDTH | `mU→m_down; mR→m_right; mR0→m_right0; kR→k_right` |  |
| 252 | code | CORE-LETTER WIDTH | `mR0→m_right0; kR→k_right` |  |
| 256 | cmt | STORE-WORD | `down→up ×2` |  |
| 257 | code | CORE-LETTER | `P_O→P_U; O0→U0` |  |
| 258 | code+str | CORE-LETTER STORE-WORD WIDTH | `seeds_O→seeds_U; real_O→real_U; down→up; mL→m_left; mR→m_right; mD0→m_up0; kD→k_up` |  |
| 259 | code | CORE-LETTER WIDTH | `new_O→new_U; P_O→P_U; mD0→m_up0; seeds_O→seeds_U; real_O→real_U; kD→k_up` |  |
| 260 | code | CORE-LETTER | `O→U; P_O→P_U; new_O→new_U` |  |
| 261 | code | CORE-LETTER | `U→D; O→U; mU→m_down; mD→m_up; mL→m_left; mR→m_right` |  |
| 368 | code | STORE-WORD | `_u_down_messages→_u_up_messages` |  |
| 370 | code+cmt | CORE-LETTER | `U→D` |  |
| 372 | cmt | CORE-LETTER | `PhiO→PhiU` |  |
| 373 | cmt | CORE-LETTER | `XiO→XiU` |  |
| 376 | code+str | CORE-LETTER | `PhiO→PhiU; U→D` |  |
| 377 | code+str | CORE-LETTER | `PhiO→PhiU ×2; U→D` |  |
| 380 | code | STORE-WORD | `_u_up_messages→_u_down_messages` |  |
| 384 | code+cmt | CORE-LETTER | `O→U` |  |
| 390 | cmt | CORE-LETTER | `PhiU→PhiD` |  |
| 391 | cmt | CORE-LETTER | `XiU→XiD` |  |
| 394 | code | CORE-LETTER | `O→U` |  |
| 399 | code+str | CORE-LETTER | `PhiU→PhiD; O→U` |  |
| 400 | code+str | CORE-LETTER | `PhiU→PhiD; O→U` |  |
| 401 | code | CORE-LETTER | `PhiU→PhiD` |  |
| 405 | cmt | CORE-LETTER | `D_L→E_L` | C2: D-map renamed E* |
| 406 | cmt | CORE-LETTER | `O_P→U_P; o_U→o_D; DU→ED` | C2: D-map renamed E* |
| 412 | cmt | CORE-LETTER | `D_L→E_L` | C2: D-map renamed E* |
| 413 | cmt | CORE-LETTER | `D_L→E_L` | C2: D-map renamed E* |
| 417 | code | CORE-LETTER | `D_L→E_L` | C2: D-map renamed E* |
| 418 | code | CORE-LETTER | `O_P→U_P; o_U→o_D; DU→ED` | C2: D-map renamed E* |
| 419 | code | CORE-LETTER | `o_U→o_D` |  |
| 420 | code+str | CORE-LETTER | `O_P→U_P; DU→ED` | C2: D-map renamed E* |
| 421 | code+str | CORE-LETTER | `D_L→E_L; o_U→o_D` | C2: D-map renamed E* |
| 423 | code | CORE-LETTER | `o_U→o_D` |  |
| 428 | code+str | CORE-LETTER | `D→E` | C2: D-map renamed E* |
| 429 | code | CORE-LETTER | `D→E ×2` | C2: D-map renamed E* |
| 449 | cmt | CORE-LETTER | `D_R→E_R` | C2: D-map renamed E* |
| 450 | cmt | CORE-LETTER | `O_P→U_P; o_U→o_D; DU→ED` | C2: D-map renamed E* |
| 456 | cmt | CORE-LETTER | `D_R→E_R` | C2: D-map renamed E* |
| 457 | cmt | CORE-LETTER | `D_R→E_R` | C2: D-map renamed E* |
| 461 | code | CORE-LETTER | `D_R→E_R` | C2: D-map renamed E* |
| 462 | code | CORE-LETTER | `O_P→U_P; o_U→o_D; DU→ED` | C2: D-map renamed E* |
| 463 | code | CORE-LETTER | `o_U→o_D` |  |
| 464 | code+str | CORE-LETTER | `O_P→U_P; DU→ED` | C2: D-map renamed E* |
| 465 | code+str | CORE-LETTER | `D_R→E_R; o_U→o_D` | C2: D-map renamed E* |
| 467 | code | CORE-LETTER | `o_U→o_D` |  |
| 472 | code+str | CORE-LETTER | `D→E` | C2: D-map renamed E* |
| 473 | code | CORE-LETTER | `D→E ×2` | C2: D-map renamed E* |
| 522 | cmt | LETTER-API | -> `update:    typ.Tuple[str, ...],       # static: the stores updated (store words); the rest transported` | HAND |
| 524 | cmt | CORE-LETTER | `U→D` |  |
| 525 | cmt | CORE-LETTER | `O→U` |  |
| 531 | code | CORE-LETTER | `U→D; O→U` |  |
| 532 | code | CORE-LETTER | `rowU→rowD; rowO→rowU; pinU→pinD; pinO→pinU; selU→selD; selO→selU` |  |
| 534 | code | CORE-LETTER | `U→D; O→U` |  |
| 536 | code | CORE-LETTER | `U→D` |  |
| 537 | code | CORE-LETTER | `U→D` |  |
| 538 | code | CORE-LETTER | `U→D` |  |
| 541 | code | CORE-LETTER | `selU→selD` |  |
| 543 | code | CORE-LETTER | `U→D` |  |
| 546 | code+cmt | CORE-LETTER | `U→D ×2; O→U ×2` |  |
| 550 | code | CORE-LETTER STORE-WORD | `PhiO→PhiU; XiO→XiU; _u_down_messages→_u_up_messages; U→D` |  |
| 554 | code | CORE-LETTER | `O→U; PhiO→PhiU; XiO→XiU` |  |
| 558 | code | CORE-LETTER | `O→U; PhiO→PhiU; XiO→XiU` |  |
| 561 | code | CORE-LETTER STORE-WORD | `PhiU→PhiD; XiU→XiD; _u_up_messages→_u_down_messages; O→U` |  |
| 563 | cmt | STORE-WORD | `up→down` |  |
| 564 | code+cmt | CORE-LETTER | `U→D` |  |
| 565 | code | CORE-LETTER | `U→D` |  |
| 566 | code+str | CORE-LETTER LETTER-API | `PhiU→PhiD; XiU→XiD; U→down` | HAND: 'U' in update -> 'down' in update |
| 570 | code+cmt | CORE-LETTER | `pinU→pinD; rowU→rowD; selU→selD` |  |
| 573 | code+str | CORE-LETTER | `oU→oD` |  |
| 574 | code+str | CORE-LETTER | `oU→oD` |  |
| 576 | code+cmt+str | CORE-LETTER | `DU→ED` | C2: D-map renamed E* |
| 577 | code+cmt | CORE-LETTER | `U1→D1` |  |
| 582 | code+str | CORE-LETTER LETTER-API | `DL→EL; L→left` | HAND C2: 'L' in update -> 'left' in update D-map renamed E* |
| 583 | code | CORE-LETTER | `O→U` |  |
| 584 | code | CORE-LETTER | `oU→oD; DU→ED` | C2: D-map renamed E* |
| 586 | code+str | CORE-LETTER LETTER-API | `DR→ER; R→right` | HAND C2: 'R' in update -> 'right' in update D-map renamed E* |
| 587 | code | CORE-LETTER | `O→U` |  |
| 588 | code | CORE-LETTER | `oU→oD; DU→ED` | C2: D-map renamed E* |
| 592 | cmt | STORE-WORD | `down→up` |  |
| 593 | code+cmt | CORE-LETTER | `DL→EL ×2` | C2: D-map renamed E* |
| 594 | code+cmt | CORE-LETTER | `DR→ER ×2` | C2: D-map renamed E* |
| 596 | code+str | CORE-LETTER | `DR→ER` | C2: D-map renamed E* |
| 597 | code+str | CORE-LETTER | `DL→EL` | C2: D-map renamed E* |
| 599 | code | CORE-LETTER | `O→U` |  |
| 600 | code | CORE-LETTER | `O→U` |  |
| 601 | code+str | CORE-LETTER LETTER-API | `PhiO→PhiU; XiO→XiU; O→up` | HAND: 'O' in update -> 'up' in update |
| 602 | code | CORE-LETTER | `mvO→mvU; pinO→pinU; rowO→rowU; selO→selU` |  |
| 603 | code | CORE-LETTER | `O1→U1` |  |
| 604 | code | CORE-LETTER | `mvO→mvU` |  |
| 609 | code | CORE-LETTER | `U1→D1; O1→U1` |  |
| 621 | cmt+str | LETTER-API | -> `update:           typ.Tuple[str, ...] = ('down', 'up', 'left', 'right'),   # static; as tv_align_padded_frame` | HAND |
| 627 | str | LETTER-API | -> `if isinstance(update, str) or not update or len(set(update)) != len(update) or not set(update) <= {'down', 'up', 'left', 'right'}:` | HAND |
| 628 | str | CORE-LETTER LETTER-API | -> `raise ValueError("update must be a non-empty tuple of the stores 'down', 'up', 'left', 'right' without repeats; got %r"` | HAND |
| 630 | code | CORE-LETTER | `U→D; O→U` |  |
| 633 | code | CORE-LETTER | `mU→m_down; mD→m_up; mL→m_left; mR→m_right` |  |
| 636 | code | CORE-LETTER | `U→D` |  |
| 637 | code | CORE-LETTER | `U→D ×2` |  |
| 641 | code+str | CORE-LETTER STORE-WORD | `up→down; U→D; down→up; O→U` |  |
| 643 | code+str | CORE-LETTER STORE-WORD | `up→down; mU→m_down; down→up; mD→m_up` |  |
| 644 | code+str | CORE-LETTER | `mL→m_left; mR→m_right` |  |
| 646 | str | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 651 | code+cmt | CORE-LETTER | `mU→m_down; mD→m_up; mL→m_left; mR→m_right` |  |
| 652 | code | CORE-LETTER | `mU→m_down; mD→m_up; mL→m_left; mR→m_right` |  |
| 657 | str | STORE-WORD | `up→down` |  |
| 669 | str | LETTER-API STORE-WORD | -> `for store, o, s1, s in zip(('down', 'up', 'left', 'right'), old, out, own):` | HAND: drop the 'UOLR' letter column |
| 680 | code | CORE-LETTER | `mU→m_down ×2` |  |
| 682 | str | STORE-WORD | `up→down` |  |
| 688 | code | CORE-LETTER | `pU→p_down; pD→p_up; pL→p_left; pR→p_right` |  |
| 689 | code | CORE-LETTER | `mU1→m_down1; mD1→m_up1; mU→m_down; pU→p_down; mD→m_up; pD→p_up` |  |
| 690 | code | CORE-LETTER | `mL1→m_left1; mR1→m_right1; mL→m_left; pL→p_left; mR→m_right; pR→p_right` |  |
| 700 | code | CORE-LETTER | `mL1→m_left1; mU1→m_down1` |  |
| 701 | code | CORE-LETTER | `mU1→m_down1; mR1→m_right1` |  |
| 702 | code | CORE-LETTER | `mL1→m_left1; mR1→m_right1` |  |
| 703 | code | CORE-LETTER | `mU→m_down; mL→m_left; mR→m_right; mD→m_up` |  |
| 704 | code | CORE-LETTER | `pU→p_down; pL→p_left` |  |
| 705 | code | CORE-LETTER | `pR→p_right; pD→p_up` |  |
| 706 | code | CORE-LETTER | `mU→m_down; mD→m_up; mL→m_left; mR→m_right` |  |
| 707 | code | CORE-LETTER | `U→D; O→U` |  |
| 708 | code | CORE-LETTER | `U1→D1; O1→U1` |  |
| 709 | code | CORE-LETTER | `U1→D1; O1→U1; mU1→m_down1; mD1→m_up1; mL1→m_left1; mR1→m_right1` |  |
| 721 | cmt | CORE-LETTER WIDTH | `WU→W_down; WD→W_up; WL→W_left; WR→W_right` | NOTE: public keyword level_widths keeps its name; its documented tuple (WU, WD, WL, WR) is physical order: (W_down, W_up, W_left, W_right) |
| 724 | code | CORE-LETTER | `U→D; O→U` |  |
| 726 | code | CORE-LETTER | `U→D; O→U` |  |
| 728 | code | CORE-LETTER | `WU→W_down; WD→W_up; WL→W_left; WR→W_right` |  |
| 730 | code+str | CORE-LETTER STORE-WORD | `up→down; down→up; WU→W_down; WD→W_up; WL→W_left; WR→W_right` |  |
| 733 | code | CORE-LETTER | `U→D; WU→W_down; O→U; WD→W_up; WL→W_left; WR→W_right` |  |
| 734 | code+cmt | CORE-LETTER | `U→D ×3` |  |
| 735 | code | CORE-LETTER | `O→U ×2` |  |
| 737 | code+cmt+str | CORE-LETTER | `iU→i_down; iD→i_up; iL→i_left; iR→i_right` |  |
| 738 | code | CORE-LETTER | `WU→W_down; WD→W_up; WL→W_left; WR→W_right` |  |
| 741 | code | CORE-LETTER | `U→D ×2; iU→i_down` |  |
| 742 | code | CORE-LETTER | `O→U ×2; iL→i_left; iD→i_up; iR→i_right` |  |
| 743 | code | CORE-LETTER | `iL→i_left ×2; iU→i_down` |  |
| 744 | code | CORE-LETTER | `iR→i_right ×2; iU→i_down` |  |
| 745 | code | CORE-LETTER | `WU→W_down; WD→W_up; WL→W_left; WR→W_right` |  |
| 746 | code | CORE-LETTER | `U→D; O→U` |  |
| 762 | cmt | CORE-LETTER | `U→D; O→U` |  |
| 776 | cmt | LETTER-API | -> `statics:  typ.Tuple,   # as _aligning_pass, at the working structure, all four stores` | HAND |
| 788 | cmt | LETTER-API | -> `) -> NDArray:                # 0-d bool: run another (('down', 'left'), ('right', 'up')) pair` | HAND |
| 795 | cmt | LETTER-API | -> `statics:  typ.Tuple[typ.Tuple, typ.Tuple],   # the ('down', 'left') and the ('right', 'up') passes' statics, at the target structure` | HAND |
| 798 | doc | LETTER-API | -> `"""One (('down', 'left'), ('right', 'up')) pair; the stop: both movements below tol (the max over the stack; a NaN never stops)."""` | HAND |
| 808 | cmt | LETTER-API | -> `statics:  typ.Tuple,   # the ('down', 'left') pass's statics` | HAND |
| 811 | doc | LETTER-API | -> `"""The odd remainder's final ('down', 'left') pass."""` | HAND |
| 833 | cmt | LETTER-API | -> `statics:  typ.Tuple,   # the all-store pass's statics, at the target structure` | HAND |
| 894 | str | LETTER-API | -> `statics = (project, hint, shape, masks, split, (None, None), sharing, ('down', 'up', 'left', 'right'))` | HAND |
| 901 | str | LETTER-API | -> `state = xwhile(_pairs_cond, ft.partial(_pairs_body, (at(('down', 'left')), at(('right', 'up')))), state, use_jit)` | HAND |
| 902 | str | LETTER-API | -> `state = _cond(_single_cond(state), ft.partial(_final_pass, at(('down', 'left'))), _no_pass, state, use_jit)` | HAND |
| 905 | str | LETTER-API | -> `state = xwhile(_single_cond, ft.partial(_single_body, at(('down', 'up', 'left', 'right'))), state, use_jit)` | HAND |
| 916 | cmt | CORE-LETTER | `WU→W_down; WD→W_up; WL→W_left; WR→W_right` |  |
| 938 | code | CORE-LETTER | `U0→D0` |  |
| 941 | code | CORE-LETTER | `U0→D0` |  |
| 950 | str | STORE-WORD | `up→down; down→up` |  |
| 956 | code | CORE-LETTER | `U0→D0` |  |

### `ut3_linalg.py` (9 rows, 6 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 113 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 117 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 119 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 122 | code | WIDTH-GENERIC | `rR→r_right` |  |
| 128 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 151 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 179 | cmt | ADJECTIVE | keep |  |
| 180 | doc | ADJECTIVE | keep |  |
| 192 | code | ADJECTIVE | keep |  |

### `ut3_operations.py` (1 rows, 0 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 78 | doc | UNRELATED | keep |  |

### `ut3_orthogonalization.py` (26 rows, 9 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 20 | str | ADJECTIVE | keep |  |
| 21 | str | ADJECTIVE | keep |  |
| 22 | str | ADJECTIVE | keep |  |
| 23 | str | ADJECTIVE | keep |  |
| 40 | doc | ADJECTIVE | keep |  |
| 125 | code | ADJECTIVE | keep |  |
| 134 | doc | ADJECTIVE | keep |  |
| 157 | code | ADJECTIVE | keep |  |
| 158 | doc | UNRELATED | keep |  |
| 173 | code | ADJECTIVE | keep |  |
| 182 | code | ADJECTIVE | keep |  |
| 184 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 186 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 188 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 189 | cmt | ADJECTIVE CORE-LETTER STORE-WORD | -> `) -> typ.Tuple[NDArray, NDArray]:   # (new_tucker = variations V, new_tt = up-orthogonal U)` | HAND: backwards adjective ('down-orthogonal O'): the TT core after a TT-up SVD is UP-orthogonal |
| 190 | doc | ADJECTIVE | keep |  |
| 191 | doc | UNRELATED | keep |  |
| 192 | doc | ADJECTIVE | keep |  |
| 198 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 199 | code | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 208 | code+cmt | WIDTH-GENERIC | `rL→r_left ×2; rR→r_right ×2` |  |
| 214 | code | ADJECTIVE | keep |  |
| 215 | doc | ADJECTIVE UNRELATED | keep |  |
| 216 | doc | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 224 | cmt | WIDTH-GENERIC | `rL→r_left; rR→r_right` |  |
| 226 | code | ADJECTIVE | keep |  |

### `ut3_sampling.py` (11 rows, 10 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 111 | cmt | CORE-LETTER | `O→U` | C4 |
| 220 | doc | UNRELATED | keep |  |
| 229 | cmt | CORE-LETTER | `O→U` | C4 |
| 373 | doc | CORE-LETTER | `U→D ×2; O→U` |  |
| 374 | doc | CORE-LETTER STORE-WORD | -> ```frame's ``D`` and ``P`` are ``theta`` masked with the frame's ``(down, left)`` masks, so ``data`` must carry those``` | HAND |
| 463 | doc | CORE-LETTER | `U→D; O→U` |  |
| 464 | doc | CORE-LETTER STORE-WORD | `U→D; up→down` |  |
| 497 | doc | STORE-WORD | `up→down` |  |
| 524 | doc | STORE-WORD | `up→down` |  |
| 599 | doc | STORE-WORD | `up→down` |  |
| 636 | doc | STORE-WORD | `up→down` |  |

### `ut3_svd.py` (9 rows, 0 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 299 | cmt | ADJECTIVE | keep |  |
| 331 | cmt | ADJECTIVE | keep |  |
| 345 | doc | ADJECTIVE | keep |  |
| 374 | code | ADJECTIVE | keep |  |
| 379 | code | ADJECTIVE | keep |  |
| 485 | cmt | ADJECTIVE | keep |  |
| 529 | cmt | ADJECTIVE | keep |  |
| 562 | code | ADJECTIVE | keep |  |
| 566 | code | ADJECTIVE | keep |  |

### `utv_operations.py` (138 rows, 134 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 58 | cmt | STORE-WORD | `up→down; down→up` |  |
| 63 | doc | STORE-WORD WIDTH | `up→down; nU→n_down` |  |
| 64 | doc | WIDTH | `nD→n_up` |  |
| 67 | cmt | STORE-WORD WIDTH | `up→down; nU→n_down` |  |
| 68 | cmt | WIDTH | `nD→n_up` |  |
| 188 | cmt | STORE-WORD | `up→down; down→up` |  |
| 196 | doc | CORE-LETTER | `U→D` |  |
| 198 | doc | WIDTH | `rL→r_left; rR→r_right` |  |
| 201 | doc | STORE-WORD | `up→down; down→up` |  |
| 205 | doc | UNRELATED | keep |  |
| 213 | doc | CORE-LETTER STORE-WORD | `O_i→U_i; UP→DOWN` | C1: Udot (T3 letter) 'has the DOWN width' |
| 214 | doc | WIDTH | `nU→n_down; nD→n_up` |  |
| 215 | doc | STORE-WORD | `UP→DOWN` |  |
| 216 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 218 | code | STORE-WORD | `var_up_mask→var_down_mask; var_down_mask→var_up_mask` |  |
| 220 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 225 | code | STORE-WORD WIDTH | `nU_pad→n_down_pad; nD_pad→n_up_pad; up_sc→down_sc; down_sc→up_sc` |  |
| 226 | code | WIDTH | `nU_pad→n_down_pad; nD_pad→n_up_pad` |  |
| 227 | code | STORE-WORD | `up_sc→down_sc` |  |
| 229 | code | STORE-WORD | `up_sc→down_sc` |  |
| 230 | code+cmt | STORE-WORD WIDTH | `up_mask_frame→down_mask_frame; nU→n_down` |  |
| 231 | code+cmt | STORE-WORD WIDTH | `up_mask_v→down_mask_v; nU→n_down` |  |
| 232 | code | STORE-WORD | `up_mask_frame→down_mask_frame ×3` |  |
| 233 | code | STORE-WORD | `up_mask_frame→down_mask_frame ×2` |  |
| 237 | code | STORE-WORD | `down_slices→up_slices` |  |
| 241 | cmt | WIDTH | `nU→n_down` |  |
| 242 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `down_slices→up_slices; O_i→U_i; rL→r_left; nU→n_down; rR→r_right` |  |
| 243 | code | STORE-WORD | `up_mask_v→down_mask_v` |  |
| 248 | code | CORE-LETTER STORE-WORD | `O→U; down_sc→up_sc` |  |
| 249 | code | STORE-WORD | `down_slices→up_slices` |  |
| 250 | code | CORE-LETTER | `O→U ×3` |  |
| 252 | code | STORE-WORD | `var_down_mask→var_up_mask ×2` |  |
| 253 | cmt | WIDTH | `nU→n_down` |  |
| 254 | code+cmt | STORE-WORD WIDTH | `down_sc→up_sc; down_slices→up_slices; rL→r_left; nU→n_down; rR→r_right` |  |
| 255 | code+cmt | STORE-WORD WIDTH | `var_down_mask→var_up_mask; nU→n_down` |  |
| 257 | code | STORE-WORD | `up_sc→down_sc` |  |
| 258 | code | STORE-WORD WIDTH | `nU→n_down; up_sc→down_sc ×2; nD→n_up; down_sc→up_sc` |  |
| 259 | code | WIDTH | `rL→r_left; rR→r_right` |  |
| 260 | code+cmt | STORE-WORD | `up_sc→down_sc` |  |
| 268 | code | CORE-LETTER STORE-WORD | `U→D; up_sc→down_sc; O→U; down_sc→up_sc` |  |
| 271 | cmt | CORE-LETTER | `U→D` |  |
| 272 | code+cmt | CORE-LETTER WIDTH | `U→D; nU→n_down; nD→n_up` |  |
| 274 | cmt | WIDTH | `rR→r_right ×2; rL→r_left ×2; nU→n_down; nD→n_up` |  |
| 275 | cmt | WIDTH | `rR→r_right` |  |
| 276 | cmt | WIDTH | `rL→r_left ×2; nU→n_down; rR→r_right` |  |
| 277 | code+cmt | CORE-LETTER WIDTH | `O→U; rL→r_left ×4; nD→n_up ×2; rR→r_right` |  |
| 278 | cmt | WIDTH | `rL→r_left ×2; nU→n_down; nD→n_up; rR→r_right` |  |
| 279 | code+cmt | WIDTH | `rR→r_right ×4; nU→n_down ×2; nD→n_up ×2; rL→r_left ×3` |  |
| 284 | code+cmt | WIDTH | `rR→r_right ×3; nU→n_down ×2; rL→r_left ×2` |  |
| 285 | cmt | WIDTH | `rL→r_left ×2; nU→n_down; rR→r_right` |  |
| 286 | cmt | WIDTH | `rR→r_right ×2; rL→r_left ×2; nU→n_down` |  |
| 288 | code+cmt | WIDTH | `rR→r_right ×5; nD→n_up ×3; rL→r_left ×2` |  |
| 289 | code+cmt | CORE-LETTER WIDTH | `O→U ×2; rL→r_left ×4; nD→n_up ×2; rR→r_right` |  |
| 290 | cmt | WIDTH | `rR→r_right ×2; rL→r_left ×2; nD→n_up` |  |
| 291 | cmt | WIDTH | `rR→r_right ×2; rL→r_left ×2; nU→n_down; nD→n_up` |  |
| 293 | cmt | WIDTH | `rL→r_left` |  |
| 297 | code+cmt | WIDTH | `rL→r_left ×2; nU→n_down ×2; rR→r_right ×2` |  |
| 299 | cmt | WIDTH | `rR→r_right ×2; rL→r_left; nU→n_down` |  |
| 300 | code+cmt | CORE-LETTER WIDTH | `rR→r_right ×4; nD→n_up ×2; O→U; rL→r_left` |  |
| 301 | cmt | WIDTH | `rR→r_right ×2; rL→r_left; nU→n_down; nD→n_up` |  |
| 302 | code+cmt | WIDTH | `rR→r_right ×3; rL→r_left ×4; nU→n_down ×2; nD→n_up ×2` |  |
| 305 | cmt | WIDTH | `rR→r_right; rL→r_left` |  |
| 306 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 312 | cmt | WIDTH | `rR→r_right ×2; rL→r_left ×2; nU→n_down; nD→n_up` |  |
| 315 | cmt | CORE-LETTER STORE-WORD | `U→D; up→down; down→up` | C1: [D \| dU] = [down \| up]: frame letter beside the T3 variation letter dU |
| 320 | code+cmt | CORE-LETTER STORE-WORD WIDTH | `var_up_mask→var_down_mask; var_down_mask→var_up_mask; nU→n_down; nD→n_up; U→D` | C1 |
| 321 | code+cmt | WIDTH | `rL→r_left ×2` |  |
| 322 | code+cmt | WIDTH | `rR→r_right ×2` |  |
| 323 | cmt | WIDTH | `rR→r_right; rL→r_left` |  |
| 336 | doc | STORE-WORD | `up→down` |  |
| 356 | code | STORE-WORD | `up_mask→down_mask; _down_mask→_up_mask` |  |
| 357 | code+cmt | STORE-WORD | `up_ranks→down_ranks; up_mask→down_mask` |  |
| 367 | code | STORE-WORD | `up_ranks→down_ranks` |  |
| 382 | doc | CORE-LETTER | `O→U` | C1 C4 |
| 383 | doc | STORE-WORD | `up_tucker_supercore→down_tucker_supercore` |  |
| 387 | doc | UNRELATED | keep |  |
| 389 | doc | STORE-WORD | `up_mask→down_mask` |  |
| 391 | code+cmt | STORE-WORD WIDTH | -> `down_sc = frame_data[0]       # (d,)   + C + (n_down, N)        -- the point's Tucker core U` | C1: corewise frame: the point's Tucker core keeps the T3 letter U |
| 392 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |
| 394 | code | STORE-WORD | `up_mask→down_mask; _down_mask→_up_mask` |  |
| 396 | cmt | WIDTH | `nD→n_up; rL→r_left; nU→n_down; rR→r_right` |  |
| 397 | code+cmt | STORE-WORD | `up_sc→down_sc` |  |
| 408 | code | STORE-WORD | `up_sc→down_sc` |  |
| 410 | code | STORE-WORD | `up_mask→down_mask` |  |
| 448 | doc | CORE-LETTER STORE-WORD | `up→down; U→D` |  |
| 452 | doc | UNRELATED | keep |  |
| 459 | code | STORE-WORD | `up_sc→down_sc; _down→_up` |  |
| 462 | code | STORE-WORD | `up_sc→down_sc` |  |
| 465 | cmt+str | WIDTH | `rL→r_left; rR→r_right` |  |
| 469 | cmt | CORE-LETTER | `U→D` |  |
| 470 | code+cmt+str | CORE-LETTER STORE-WORD WIDTH | `up_sc→down_sc; U→D; nD→n_up; nU→n_down` |  |
| 471 | code+str | STORE-WORD | `up_sc→down_sc` |  |
| 479 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 480 | cmt | WIDTH | `rL→r_left ×3; rR→r_right ×3` |  |
| 487 | cmt+str | WIDTH | `rL→r_left; rR→r_right` |  |
| 498 | doc | CORE-LETTER STORE-WORD | `U→D; down→up; O→U` |  |
| 504 | doc | UNRELATED | keep |  |
| 506 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 509 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 511 | cmt | CORE-LETTER STORE-WORD | `U→D; down→up; O→U` |  |
| 512 | code+cmt+str | CORE-LETTER STORE-WORD WIDTH | `up_sc→down_sc; U→D; nD→n_up; nU→n_down` |  |
| 513 | code+str | STORE-WORD | `up_sc→down_sc` |  |
| 514 | code+cmt+str | CORE-LETTER STORE-WORD | `down_sc→up_sc; O→U` |  |
| 517 | cmt | WIDTH | `rL→r_left; rR→r_right` |  |
| 521 | code | WIDTH | `rL→r_left; rR→r_right` |  |
| 523 | code | WIDTH | `rL→r_left; rR→r_right` |  |
| 538 | doc | CORE-LETTER STORE-WORD | `up→down; U→D` | C1: D^T dU: frame letter with T3 variation letter |
| 551 | code | STORE-WORD | `up_sc→down_sc; _down→_up` |  |
| 553 | code | STORE-WORD | `up_sc→down_sc` |  |
| 555 | code+cmt+str | CORE-LETTER STORE-WORD WIDTH | `up_sc→down_sc; U→D; nU→n_down; nD→n_up` | C1 |
| 557 | cmt+str | WIDTH | `rL→r_left; rR→r_right` |  |
| 585 | doc | STORE-WORD UNRELATED | `up→down` |  |
| 589 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 592 | code | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 593 | code | STORE-WORD | `up_sc→down_sc; down_sc→up_sc` |  |
| 595 | cmt | STORE-WORD | `up→down` |  |
| 596 | code+cmt+str | CORE-LETTER STORE-WORD WIDTH | `BU1→BD1; up_sc→down_sc; U→D; nU→n_down` |  |
| 597 | code+cmt+str | CORE-LETTER WIDTH | `BU1→BD1; nU→n_down` |  |
| 604 | cmt+str | WIDTH | `rL→r_left; rR→r_right` |  |
| 605 | code+cmt+str | CORE-LETTER STORE-WORD WIDTH | `BU→BD; up_sc→down_sc; U→D; nU→n_down` |  |
| 606 | code+cmt+str | CORE-LETTER WIDTH | `BU→BD; rL→r_left; nU→n_down; rR→r_right` |  |
| 607 | code+cmt+str | STORE-WORD WIDTH | `down_sc→up_sc; nD→n_up` |  |
| 608 | cmt+str | WIDTH | `nD→n_up` |  |
| 614 | code | STORE-WORD | `up_mask→down_mask; down_mask→up_mask` |  |
| 626 | doc | WIDTH | `nD→n_up` |  |
| 627 | code | WIDTH | `nD→n_up` |  |
| 634 | code | WIDTH | `nD→n_up` |  |
| 636 | str | WIDTH | `nD→n_up` |  |
| 637 | code+str | WIDTH | `nD→n_up` |  |
| 669 | cmt | STORE-WORD | `up→down` |  |
| 671 | doc | CORE-LETTER | `U→D` |  |
| 677 | doc | STORE-WORD | `up→down` |  |
| 704 | doc | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 705 | doc | WIDTH | `nU→n_down; nD→n_up; rL→r_left; rR→r_right` |  |
| 708 | doc | WIDTH | `nU→n_down ×2; rL→r_left ×2` |  |
| 721 | code | STORE-WORD | `up_mask→down_mask; _down→_up` |  |
| 722 | code | STORE-WORD | `up_mask→down_mask` |  |
| 754 | cmt | STORE-WORD | `up→down; down→up` |  |

### `utv_sampling.py` (6 rows, 5 change)

| L | kind | cat | change | flags / note |
|---|---|---|---|---|
| 69 | cmt | STORE-WORD | `up→down; down→up` |  |
| 71 | cmt | STORE-WORD | `up→down; down→up` |  |
| 72 | doc | UNRELATED | keep |  |
| 429 | doc | STORE-WORD | `up→down; down→up` |  |
| 439 | cmt | WIDTH | `nD→n_up` |  |
| 440 | cmt | WIDTH | `rL→r_left; nU→n_down; rR→r_right` |  |

## 10. Cross-references outside the partition (for inventories 1/3)

- `update=` / letter callers in tests: `tests/test_dispatch.py`:2019-2025 (the four align-dispatch cases pass `'ULRO'` / `'UL'`;
  the case name `'shared, UL'`), :2034 (`static_argnames=(..., 'update')` -- unchanged), :2058, :2345, :2395 (comments 'UL'/'RO');
  `tests/backend/test_padded_frame.py`:40-41, 513, 547, 574 (`('UL', 'RO', 'LRO', 'ULO', 'U', 'O')`), 723, 731
  (`''.join(c for c in 'ULRO' if c != missing)` -- a letter-arithmetic test that needs a rewrite), 736 (`('', 'UU', 'X', 'ULROX',
  'ulro')` invalid cases; message regex `'without repeats'`), 899-1015 (recorded schedules `('ULRO', t), ('UL', t), ('RO', t)`);
  `tests/backend/test_uniform_padded_frame.py`:360-392, 514-547, 771-802 (same patterns); `tests/backend/test_padded_start.py`:480;
  `tests/test_aligned_padded_frame_frontend.py`:125-126 (`update='UL'` on the frontend `align_padded` and the backend).
- Frontend: `uniform_frame_variations_format.py`:1611 "``('UL', 'RO')`` pairs" and the `align_padded(update=)` member forward
  the backend `update`; `from_t3frame(nD=)` etc. forward the §2 width keywords.
- `tests/backend/test_padded_frame.py`:91 keeps its own copy of `_SEED_TAGS` (blast radius §1.3): edit in lockstep, integers kept.
- Message-text assertions on store names (blast radius §1.1, e.g. `test_padded_frame.py`:405 `'old_frame down\[1\]'`) follow the
  flipped words of the §9 `str` rows of `padded_frames.py` / `uniform_padded_frames.py` / `geometry.py`.

