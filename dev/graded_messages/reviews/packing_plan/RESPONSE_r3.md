# Response to `REVIEW_r3.md` (the writer, 2026-11-25)

Revision 5 takes every finding. `DIFF_rev4_to_rev5.txt` (this folder) is the change.

| # | Verdict | Where resolved |
|---|---|---|
| S1 | Accepted | K1's part 0 lists the sites (the display, the default draw, the protocol's text, the translation, the frontend's readers, the constructions); the display reads the order off the batch (§2.4); the `Oracle` protocol's new form and O-29's reversal by name, and one statement on user oracles (§4.3). |
| S2 | Accepted | One kind and one order per fit: the frontend entries and the hand-over of a draw compare them; the backend display refuses (§4.3, §2.4). |
| S3 | Accepted | The public metric keeps the structural axes exact (the mode, `F`, the edge) and lets the axes between them be 1 or full (§2.3). |
| S4 | Accepted | "The modes of a probe object agree in everything but the edge", one rule on both layers, checked by the engine, the checker, `pack_batch` and the frontend batch (§2.2, §3.1). |
| S5 | Accepted | A frontend entry refuses the backend's batch, messages and `repr` naming the module; the frontend batch is a registered pytree; the frontend `LocalModel` derives its batch; a draw returning the backend's batch is refused; `data_metric(r)` compares mode sizes; `pack` on a packed batch raises (§4.3). |
| S6 | Accepted | The backend batch's constructor makes the oracle's checks and normalizes `order` (§2.4); the frontend batch inherits them (§4.3). |
| M1 | Accepted | The stale sentences are fixed (§2.4, §3.1, §12.3, §14.4); the table of §4.4 is in order; the example no longer calls `mc_sgd`; `n_measurements` is a property (§4.3, §14.7). |
| M2 | Accepted | Part 0 of K1 is the move, with a bitwise battery; K4's design note measures the tests that use the frontend name with backend calls (§7). |
| M3 | Accepted | In the backend packing stays the function `pack_batch` in `uniform_fitting` (no import reversed); the method `pack` is the frontend batch's (§4.3, §14.7). |
| M4 | Accepted | §2.2 says T3Polynomial wants that weight, and names the outlier weight as the case that would earn `sample_weight`. |
| M5 | Accepted | On K4's list for its design note. |
| M6 | Accepted | §3.2 states the exception for the start values. |
| §14.1 | Kept, as you would | The name stays; every message and `repr` names the module; your argument is in §14.1 for the maintainer. |
