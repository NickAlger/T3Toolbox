SIGN OFF: the corewise tangent, the Hessian's flag and the 18 uniform twins are right by my own dense and garbage checks, and every mutant of mine is killed.

# P12b review, round 1: the uniform twins, the corewise tangent, the Hessian's flag

Reviewer, 2026-11-21. This reviews the uncommitted P12b on top of `5905b733`, against the plan's P12 and 2.4, the
design review, and O-48. Everything was run single-process, one run at a time. The mutant copies have been deleted,
and the working tree is untouched apart from this file.

## My independent checks (`p12b_check.py`, seeds 0 and 1, both envs; worst 1.5e-15)

**A. The tangent against my own dense construction.**
- `t3_{probe,apply,entries}_corewise_tangent` along directions with `K = (2,)` at `C = (2,)`.
- Per `(k, c)`, each result equals my roots-of-unity oracle applied to `T_dot`, the sum of the `2d` dense tensors that
  each have one core replaced by its direction.
- Orders `()`, `(2,)` and `(1, 1)`, with input jets of mixed sizes per mode.
- Entries with negative indices.

**B. The Hessian's flag.**
- With `sum_over_samples=False`, each sample's block equals the Hessian of the call on that one sample, and the blocks
  sum to the default.
- Probe and apply, at three orders.
- The default itself was checked in P12a against the polarization of my dense `phi''`.

**C. The uniform twins.**
- *Setup:*
  - the point from `t3_to_ut3`, padded beyond every size (`N = 5`, `n = r = 4`);
  - the frame from `t3frame_to_ut3frame`;
  - **garbage (1e3) in every padded slot** of the point, the frame, the directions (`K + C`) and the variations, and in
    the mode padding of a packed sample and of a packed probe residual.
- *Each twin is compared with its ragged function, both values of the flag, ragged and packed samples:*
  - `ut3_*` (`z`, tangent, corewise transpose with a residual carrying `K`, Hessian);
  - `utv_*` (`Jp` and the transpose).
- *What is asserted:*
  - the real block equals ragged;
  - **every padded entry is exactly zero**, with the real extent read off the ragged result's shape, not a mask;
  - **the mirror**: a ragged sample gives a `len = d` tuple of the ragged shapes, and a packed one gives an
    `np.ndarray` with zero padding.

**D. The step count.** The closure for `Jp` on a corewise call with `core_directions` has 9 rows, each once:
`xi`, `mu`, `xi_dot`, `mu_dot`, `nu`, `_raw.nu_dot`, `_raw.eta_dot`, `eta`, `Jp`. There is no `_frame.*` row, and no
row reads `U`, `R`, `V`, `H` or `Vc`.

## The tests

| Test | What it covers |
| --- | --- |
| `TestTheCorewiseTangent` | the dense oracle first, at the plan's orders and mixed jets; the substitution bitwise; the adjoint pair; the structural count |
| `TestAgainstAutomaticDifferentiation` | the tangent against `jax.jvp`; the Hessian against `jvp(grad)`, in float64 |
| `TestTheUniformTwins` | garbage everywhere, an indicator from the ragged shapes, both flags, the mirror |
| `TestDispatch` | every function jitted, the six tangents included |

That is what the plan asks.

**Runs.**
- `test_sampling_functions`: 22 passed (current env).
- On the floor: `test_sampling_functions`, `test_uniform_probing` and `test_uniform_curvature`, 114 passed (the 2
  skips are pre-existing, in the other files).
- The coordinator's report: the full gate passed (1,824; lowest MemAvailable 3.9 GB).

## Mutants (5, mine; each targets an area outside the implementer's 15)

All five are killed by `tests/test_sampling_functions.py`:
- `ut3_entries_corewise_hessian` ignoring its flag;
- `utv_entries_transpose` ignoring its flag;
- `ut3_probe` always returning packed (the mirror off);
- `utv_probe` always returning packed;
- `ut3_probe_corewise_transpose` not packing a ragged residual.

## Notes (no action)

- The `_by_old_order` adapter in `test_uniform_curvature` goes with the `*_from_sweep` functions at P20, as the
  handover records.
- The tangent's docstring identity (Euler, `J x = 2d z`) is a good doctest: it checks the scale of the derivative
  without a reference.
