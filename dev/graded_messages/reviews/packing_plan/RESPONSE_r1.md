# Response to `REVIEW_r1.md` (the writer, 2026-11-24)

The plan is revised as a whole (revision 2). Every finding is answered by a change or by a stated reason. The
changes that depart from the planning session's rulings are listed in the plan's §13 for the maintainer.

| # | Verdict | Where resolved, or why not |
|---|---|---|
| 1 | Accepted, option (a) | K4 and K5 exchanged: the frontend switches while the backend still accepts both forms (§7, "Order"; K4; K5). |
| 2 | Accepted, a fourth option | The batch holds plain arrays in both tiers and carries one static field `widths`; `PackedVectors` is not put inside it (§4.3, §13.1). `oracle.take`, `n_measurements` and the oracle's `data_metric` then run on a packed batch as they are; `_frontend_draw` goes with nothing in its place; `val_batch`, `LocalModel.batch` and P17's `point_value_and_gradient` take that batch. Your (a) and (b) duplicate the oracle's batch API in the frontend; (c) loses the automatic width check in a fit, which was the point. |
| 3 | Accepted, with C1's first alternative | The metric is split by kind, with the block weight checked exactly; the full weight is checked as a factor of an elementwise product (`_check_pair`'s rule), and the exact rule of §2.2 is stated as the engine's and the checkers'; `_with_K` stays for the full weight and §2.1 no longer lists it as removed; the batch method moves to the oracle (§2.3, §13.3). |
| 4 | Accepted, with C3 | The probe helpers are ragged only, except `probe_data_metric` and `affine_jet`, which multiply and keep zero padding (§3.3, §5, §13.4). |
| 5 | Accepted, option (a) for the data | The data-space inputs and the held variables are masked by selection; the cores and the sample stay under "finite", which is stated (§1.3, §3.2, §13.5). K5 tests `inf` and `nan` in those paddings. |
| 6 | Accepted | The field is `widths` (§4.1, §13.2); `unpack_vectors`' argument too (§3.1). |
| 7 | Stated; the design question put to the maintainer | §2.2 states the three lost cheap forms and where T3Polynomial's per-component weight goes (a full weight, measured at the dry port, §9). Whether `W` may be 1 is §13.7 (C2). |
| 8 | Accepted | Each typed face fixes its layer and refuses the other, naming its twin (§1.5); in K3 for the ragged functions, K5 for the twins. |
| 9 | Accepted, the first alternative | The weight-carrying helpers are renamed in K1; K6 has the rest (§5, the "Slice" column). |
| 10 | Accepted | (a) K0's gate is the zero-code-reference inventory; P20's grep is split between K0, P17 and P19. (b) The orphan scan is re-run reading every module-level statement, `_BATCH_FIELDS` is named as its known miss, and the gate imports every module with and without jax. (c) 16 pages; the direct-plus-indirect test counts (K0). |
| 11 | Accepted | The measure is the optimized program at `d = 3` and `5`, for the engine and the oracle's query; the shape reads on uniform come under §1.6 and are rewritten in K5; the fallback is dropped (§1.6, §6). |
| 12 | Accepted, the second alternative | `pack_batch` checks only the agreement of its members' widths (the input's own form) and records them; the data's shapes against the stack are checked where the batch meets the point (§3.1, §4.3). |
| 13 | Accepted | §4.4, a row per slice. |
| 14 | Accepted | §1.1 "What packing does not keep" and the contract "up to zero coefficients"; §1.2 says what the engine checks, and adds the one-slot check. |
| 15 | Accepted | The old checkers are replaced in K3 (§3.1, K2, K3). |
| 16 | Accepted | `pack_batch`'s index branch is in K3. |
| 17 | Accepted | The directions, `Vc` and the held variables are in §1.1, §3.2 and K5's refusal tests. |
| 18 | Accepted | §2.4 (the field order, the keyword question on the 3.9 floor, the engine's argument order, `computed_names`), §12.9. |
| 19 | Accepted in three parts, not in one | "The shape of a split" is now conditional (§1.4); the count is fixed; the frontend's `rescale_coordinates` and `computed_names` are listed (§5, §1.5). **Not accepted:** dropping `entries_data_block_norms`. The entries kind has data and no jets, so its face has no `order`; a user holding entries data should not have to know it is apply at `order=()`. The jet functions have no entries version because the entries kind has no jets, which is a different reason (§5, §12.10). |
| 20 | Accepted | The uniform examples are P17's; K4 does not touch them (K4, §9). |
| 21 | Accepted | `ndim >= 2`; validation on rebuild, and no `vmap` over a `PackedVectors`, stated (§4.1). |
| 22 | Accepted | The caution is in §2.2 and on P19's list (§9). |
| 23 | Accepted | §2.4. |
| 24 | Accepted | `ut3_unscale_coordinates` is a mask-holder (§1.3, §3.3) and is in K7's enumeration. |
| C1 | Adopted | As finding 3. The criterion gains "or the structural check of its arguments" (§1.4). |
| C2 | Not adopted; put to the maintainer | §13.7, with the argument for and against. It is his ruling, and the trade (one size-1 meaning back, against a full weight's memory) is his to make. |
| C3 | Adopted, except for the metric | As finding 4. `probe_data_metric` keeps its packed branch: it runs inside uniform loops (the custom-loss walk of §4.1). |
| C4 | Adopted for the data-space inputs and the held variables | As finding 5. The cores keep the multiply, as you suggest; so does the sample's neutralization, which is by the zeroed core. |
