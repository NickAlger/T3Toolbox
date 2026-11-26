# Response to `K0_DESIGN_REVIEW_r2.md`

Implementer, 2026-11-25. Both findings, M1 and the orchestrator's addition are accepted. The changes are in
`k0/prose_edits.md` (§1 and a new first part of §2) and in note §1.3 and §6.

**S1 (r2).** `probing`'s module docstring now reads "each but the ambient transpose one call of the engine". I checked
the sentence against the kept functions. Six call the engine: `t3_probe`, `tv_probe`, `tv_probe_transpose`,
`t3_probe_corewise_tangent`, `_transpose` and `_hessian`. `t3_probe_ambient_transpose` and `dense_probe` do not, and
`dense_probe` is named apart in the docstring.

**S2 (r2).** Accepted. I went through every block on the "deleted whole" list once more with your question: after K0,
is the next statement below this block a kept one that the block heads or explains? The script lists each block,
the statement after it today, and the first kept statement after it. Your three are confirmed. **Two more** answer
yes:
- `probing.py:274-276`, the banner "Probing a tangent vector". After K0 it heads `tv_probe`. It stays as it is.
- `probing.py:1236-1266`. The banner "Corewise two-hole: the probe Hessian" heads `t3_probe_corewise_hessian`, so
  the banner (`:1236-1238`) stays. The 28-line comment below it describes the deleted dotted kernels and goes.

All the other blocks head deleted code only. `fitting.py`'s blocks lead to `_KINDS` and `_flat_gather`, which are
headed by the two rewritten blocks. `uniform_fitting.py`'s lead to `pack_sample`, which keeps its packing comment, and
to `uniform_minimal`. `utv_sampling.py`'s lead to `_gauge_masks_over_Knew`, which is headed by `:444`.

The after-texts:
- `probing.py:672-674`: a new banner, "Transposes of the probe; the corewise tangent".
- `ut3_sampling.py:124-133`: rewritten, keeping your true sentences on the padding and on "the masks, not the values,
  govern rank". **One sentence is not kept: the boundary-bond squash leaving junk in the padding.** It is no longer
  true of the engine. E1 measured the corewise transposes' padding already exactly zero with the exit mask removed.
- `ut3_sampling.py:469-473`: cut to a title and one sentence on the engine's masking.

The note's §1.3 now states the test and both lists.

**M1 (r2).** `utv_sampling.py:444` reads "(probe, apply, entries)". The factoring is the orchestrator's addition,
below.

**The orchestrator's addition.** `fitting`'s new module docstring keeps the factoring, checked against the code. The
oracle's `Jp`, `g` and `JtJp` are bare: `backend/optimizers.py:192` states that the blocks are returned before the
geometry's projection. The `LocalModel` projects `p` before asking for `Jp` (`:380-382`), and projects the oracle's
`g` and `JtJp` (`:375`, `:399`). So the docstring says the oracle returns the bare `𝒥` and `𝒥ᵀ`. It says the
geometry supplies `Π`, giving `J = 𝒥∘Π` and `Jᵀr = Π∘𝒥ᵀr`.

**The two build notes** are in note §6:
- the mutations are rerun on the built tests and recorded in the handover;
- `test_import_graph.py`'s module docstring gains a line for the new test.
