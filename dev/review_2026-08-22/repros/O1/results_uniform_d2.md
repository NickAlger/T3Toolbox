| op | struct | repr | C | W | K | sharing | status | max relerr | note |
|---|---|---|---|---|---|---|---|---|---|
| u_to_dense | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_add | d2 | uniform | () | () | () | None | PASS | 1.46e-16 |  |
| u_sub | d2 | uniform | () | () | () | None | PASS | 1.71e-16 |  |
| u_scalar_mul | d2 | uniform | () | () | () | None | PASS | 1.57e-16 |  |
| u_inner | d2 | uniform | () | () | () | None | PASS | 1.10e-15 |  |
| u_norm | d2 | uniform | () | () | () | None | FAIL | 3.84e+02 |  |
| u_reverse | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_t3svd_lossless | d2 | uniform | () | () | () | None | PASS | 8.95e-16 |  |
| u_rank_adjustment_sweep | d2 | uniform | () | () | () | None | PASS | 1.11e-15 |  |
| u_t3svd_trunc_vs_ragged | d2 | uniform | () | () | () | None | PASS | 3.66e-16 |  |
| u_orthogonal_representations | d2 | uniform | () | () | () | None | PASS | 1.08e-15 |  |
| u_apply | d2 | uniform | () | () | () | None | PASS | 1.59e-16 |  |
| u_entries | d2 | uniform | () | () | () | None | PASS | 2.10e-16 |  |
| u_probe | d2 | uniform | () | () | () | None | PASS | 1.26e-16 |  |
| u_apply_derivatives | d2 | uniform | () | () | () | None | PASS | 3.36e-16 |  |
| u_entries_derivatives | d2 | uniform | () | () | () | None | PASS | 3.61e-16 |  |
| u_probe_derivatives | d2 | uniform | () | () | () | None | PASS | 5.29e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 1.78e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 1.78e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 1.09e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 8.23e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 2.97e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 1.09e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 8.23e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | () | () | () | None | PASS | 2.97e-17 |  |
| utv_apply | d2 | uniform | () | () | () | None | PASS | 1.96e-15 |  |
| utv_entries | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe | d2 | uniform | () | () | () | None | PASS | 1.57e-16 |  |
| utv_apply_derivatives | d2 | uniform | () | () | () | None | PASS | 1.39e-16 |  |
| utv_entries_derivatives | d2 | uniform | () | () | () | None | PASS | 1.74e-16 |  |
| utv_probe_derivatives | d2 | uniform | () | () | () | None | PASS | 2.13e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | () | () | () | None | PASS | 2.19e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | () | () | () | None | PASS | 2.19e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | () | () | None | PASS | 1.88e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | () | () | None | PASS | 1.23e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | () | () | None | PASS | 7.62e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | () | () | None | PASS | 1.88e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | () | () | None | PASS | 1.23e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | () | () | None | PASS | 7.62e-16 |  |
| utv_apply | d2 | uniform | () | () | (2,) | None | PASS | 8.46e-17 |  |
| utv_entries | d2 | uniform | () | () | (2,) | None | PASS | 7.16e-17 |  |
| utv_probe | d2 | uniform | () | () | (2,) | None | PASS | 2.56e-16 |  |
| utv_apply_derivatives | d2 | uniform | () | () | (2,) | None | PASS | 2.00e-16 |  |
| utv_entries_derivatives | d2 | uniform | () | () | (2,) | None | PASS | 1.58e-16 |  |
| utv_probe_derivatives | d2 | uniform | () | () | (2,) | None | PASS | 1.68e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | () | () | (2,) | None | PASS | 3.98e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | () | () | (2,) | None | PASS | 1.46e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | () | () | (2,) | None | PASS | 2.52e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | () | () | (2,) | None | PASS | 3.98e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | () | () | (2,) | None | PASS | 1.46e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | () | () | (2,) | None | PASS | 2.52e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | () | (2,) | None | PASS | 3.15e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | () | (2,) | None | PASS | 3.15e-15 |  |
| u_apply | d2 | uniform | () | (3,) | () | None | PASS | 3.21e-16 |  |
| u_entries | d2 | uniform | () | (3,) | () | None | PASS | 7.30e-16 |  |
| u_probe | d2 | uniform | () | (3,) | () | None | PASS | 1.71e-16 |  |
| u_apply_derivatives | d2 | uniform | () | (3,) | () | None | PASS | 7.09e-16 |  |
| u_entries_derivatives | d2 | uniform | () | (3,) | () | None | PASS | 9.27e-16 |  |
| u_probe_derivatives | d2 | uniform | () | (3,) | () | None | PASS | 3.00e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 1.23e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 1.98e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 5.26e-17 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 2.13e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 3.33e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 4.61e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 7.31e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 7.29e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 9.62e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 1.48e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 1.95e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | () | (3,) | () | None | PASS | 2.41e-16 |  |
| utv_apply | d2 | uniform | () | (3,) | () | None | PASS | 2.04e-16 |  |
| utv_entries | d2 | uniform | () | (3,) | () | None | PASS | 3.00e-17 |  |
| utv_probe | d2 | uniform | () | (3,) | () | None | PASS | 2.10e-16 |  |
| utv_apply_derivatives | d2 | uniform | () | (3,) | () | None | PASS | 2.74e-16 |  |
| utv_entries_derivatives | d2 | uniform | () | (3,) | () | None | PASS | 5.34e-16 |  |
| utv_probe_derivatives | d2 | uniform | () | (3,) | () | None | PASS | 2.06e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | () | None | PASS | 1.97e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | () | None | PASS | 1.40e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | () | None | PASS | 1.97e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | () | None | PASS | 1.40e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | () | None | PASS | 9.84e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | () | None | PASS | 4.26e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | () | None | PASS | 7.00e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | () | None | PASS | 2.95e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | () | None | PASS | 6.39e-16 |  |
| utv_apply | d2 | uniform | () | (3,) | (2,) | None | PASS | 6.59e-16 |  |
| utv_entries | d2 | uniform | () | (3,) | (2,) | None | PASS | 1.56e-16 |  |
| utv_probe | d2 | uniform | () | (3,) | (2,) | None | PASS | 2.01e-16 |  |
| utv_apply_derivatives | d2 | uniform | () | (3,) | (2,) | None | PASS | 2.15e-16 |  |
| utv_entries_derivatives | d2 | uniform | () | (3,) | (2,) | None | PASS | 1.41e-16 |  |
| utv_probe_derivatives | d2 | uniform | () | (3,) | (2,) | None | PASS | 2.55e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | (2,) | None | PASS | 5.94e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | (2,) | None | PASS | 3.18e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | (2,) | None | PASS | 1.04e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | (2,) | None | PASS | 9.53e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (3,) | (2,) | None | PASS | 4.42e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | (2,) | None | PASS | 2.81e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (3,) | (2,) | None | PASS | 1.47e-16 |  |
| u_apply | d2 | uniform | () | (2, 2) | () | None | PASS | 3.18e-16 |  |
| u_entries | d2 | uniform | () | (2, 2) | () | None | PASS | 5.65e-16 |  |
| u_probe | d2 | uniform | () | (2, 2) | () | None | PASS | 3.40e-16 |  |
| u_apply_derivatives | d2 | uniform | () | (2, 2) | () | None | PASS | 4.32e-16 |  |
| u_entries_derivatives | d2 | uniform | () | (2, 2) | () | None | PASS | 4.40e-16 |  |
| u_probe_derivatives | d2 | uniform | () | (2, 2) | () | None | PASS | 3.27e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 1.24e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 8.46e-17 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 3.65e-17 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 2.84e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 6.65e-17 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 1.26e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 6.99e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 1.15e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 1.27e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 2.34e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 1.26e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | () | (2, 2) | () | None | PASS | 3.56e-16 |  |
| utv_apply | d2 | uniform | () | (2, 2) | () | None | PASS | 2.52e-16 |  |
| utv_entries | d2 | uniform | () | (2, 2) | () | None | PASS | 1.06e-16 |  |
| utv_probe | d2 | uniform | () | (2, 2) | () | None | PASS | 2.11e-16 |  |
| utv_apply_derivatives | d2 | uniform | () | (2, 2) | () | None | PASS | 1.08e-16 |  |
| utv_entries_derivatives | d2 | uniform | () | (2, 2) | () | None | PASS | 1.07e-16 |  |
| utv_probe_derivatives | d2 | uniform | () | (2, 2) | () | None | PASS | 1.98e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | () | None | PASS | 1.50e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | () | None | PASS | 1.43e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | () | None | PASS | 1.50e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | () | None | PASS | 2.85e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | () | None | PASS | 1.85e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | () | None | PASS | 1.18e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | () | None | PASS | 2.82e-16 |  |
| utv_apply | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 1.45e-16 |  |
| utv_entries | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 1.91e-16 |  |
| utv_probe | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 2.44e-16 |  |
| utv_apply_derivatives | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 2.13e-16 |  |
| utv_entries_derivatives | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 1.38e-16 |  |
| utv_probe_derivatives | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 2.55e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 3.67e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 1.83e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 3.65e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 1.70e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 4.36e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 4.95e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 4.36e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | () | (2, 2) | (2,) | None | PASS | 8.25e-16 |  |
| u_manifold_inner | d2 | uniform | () | () | () | None | PASS | 1.22e-14 |  |
| u_manifold_norm | d2 | uniform | () | () | () | None | FAIL | 1.68e+02 |  |
| u_gauge_project_idempotent | d2 | uniform | () | () | () | None | PASS | 2.83e-16 |  |
| u_tangent_add_scale | d2 | uniform | () | () | () | None | PASS | 2.17e-16 |  |
| u_tangent_reverse | d2 | uniform | () | () | () | None | PASS | 1.05e-16 |  |
| u_retract_zero | d2 | uniform | () | () | () | None | PASS | 8.17e-16 |  |
| u_retract_fd_jacobian | d2 | uniform | () | () | () | None | PASS | 1.27e-07 | ratio=4.00 |
| u_retract_vs_ragged | d2 | uniform | () | () | () | None | PASS | 1.35e-15 |  |
| u_project_ambient | d2 | uniform | () | () | () | None | PASS | 2.10e-16 |  |
| u_transport_identity | d2 | uniform | () | () | () | None | PASS | 1.08e-15 |  |
| u_transport_vs_ragged_projection | d2 | uniform | () | () | () | None | PASS | 2.13e-16 |  |
| u_manifold_inner | d2 | uniform | () | () | (2,) | None | PASS | 8.18e-16 |  |
| u_manifold_norm | d2 | uniform | () | () | (2,) | None | FAIL | 2.52e+01 |  |
| u_gauge_project_idempotent | d2 | uniform | () | () | (2,) | None | PASS | 6.68e-16 |  |
| u_tangent_add_scale | d2 | uniform | () | () | (2,) | None | PASS | 1.32e-16 |  |
| u_tangent_reverse | d2 | uniform | () | () | (2,) | None | PASS | 7.22e-17 |  |
| u_retract_zero | d2 | uniform | () | () | (2,) | None | PASS | 8.17e-16 |  |
| u_retract_fd_jacobian | d2 | uniform | () | () | (2,) | None | PASS | 1.27e-07 | ratio=4.00 |
| u_project_ambient | d2 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 6 into shape (2,2,3,1) |
| u_transport_identity | d2 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 6 into shape (2,2,3,1) |
| u_transport_vs_ragged_projection | d2 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 6 into shape (2,2,3,1) |
| u_to_dense | d2 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d2 | uniform | (2,) | () | () | None | PASS | 1.17e-16 |  |
| u_sub | d2 | uniform | (2,) | () | () | None | PASS | 1.10e-16 |  |
| u_scalar_mul | d2 | uniform | (2,) | () | () | None | PASS | 1.26e-16 |  |
| u_inner | d2 | uniform | (2,) | () | () | None | PASS | 1.94e-16 |  |
| u_norm | d2 | uniform | (2,) | () | () | None | FAIL | 1.35e+01 |  |
| u_reverse | d2 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_t3svd_lossless | d2 | uniform | (2,) | () | () | None | PASS | 5.28e-16 |  |
| u_rank_adjustment_sweep | d2 | uniform | (2,) | () | () | None | PASS | 5.29e-16 |  |
| u_t3svd_trunc_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 6.41e-16 |  |
| u_orthogonal_representations | d2 | uniform | (2,) | () | () | None | PASS | 4.55e-16 |  |
| u_apply | d2 | uniform | (2,) | () | () | None | PASS | 1.52e-16 |  |
| u_entries | d2 | uniform | (2,) | () | () | None | PASS | 4.42e-17 |  |
| u_probe | d2 | uniform | (2,) | () | () | None | PASS | 1.89e-16 |  |
| u_apply_derivatives | d2 | uniform | (2,) | () | () | None | PASS | 4.59e-16 |  |
| u_entries_derivatives | d2 | uniform | (2,) | () | () | None | PASS | 4.38e-16 |  |
| u_probe_derivatives | d2 | uniform | (2,) | () | () | None | PASS | 1.92e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 2.60e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 1.62e-17 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 2.80e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 2.60e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 1.62e-17 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 2.80e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 3.84e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 7.48e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 1.06e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 3.84e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 7.48e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 1.06e-16 |  |
| utv_apply | d2 | uniform | (2,) | () | () | None | PASS | 1.57e-16 |  |
| utv_entries | d2 | uniform | (2,) | () | () | None | PASS | 1.53e-16 |  |
| utv_probe | d2 | uniform | (2,) | () | () | None | PASS | 3.18e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2,) | () | () | None | PASS | 1.12e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2,) | () | () | None | PASS | 1.48e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2,) | () | () | None | PASS | 2.10e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | () | None | PASS | 1.37e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | () | None | PASS | 1.26e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | () | None | PASS | 3.77e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | () | None | PASS | 1.37e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | () | None | PASS | 1.26e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | () | None | PASS | 3.77e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | () | None | PASS | 7.29e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | () | None | PASS | 7.29e-16 |  |
| utv_apply | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.59e-16 |  |
| utv_entries | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.53e-16 |  |
| utv_probe | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.39e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.14e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.11e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.02e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | (2,) | None | PASS | 1.11e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.90e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.00e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | (2,) | None | PASS | 1.11e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.90e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.00e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.09e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.68e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | () | (2,) | None | PASS | 3.18e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.09e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.68e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | () | (2,) | None | PASS | 3.18e-16 |  |
| u_apply | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.05e-16 |  |
| u_entries | d2 | uniform | (2,) | (3,) | () | None | PASS | 3.53e-17 |  |
| u_probe | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.76e-16 |  |
| u_apply_derivatives | d2 | uniform | (2,) | (3,) | () | None | PASS | 2.77e-16 |  |
| u_entries_derivatives | d2 | uniform | (2,) | (3,) | () | None | PASS | 2.51e-16 |  |
| u_probe_derivatives | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.95e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 9.53e-17 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 2.40e-17 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.03e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 9.38e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.04e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.13e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.86e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.31e-16 |  |
| utv_apply | d2 | uniform | (2,) | (3,) | () | None | PASS | 2.51e-16 |  |
| utv_entries | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.07e-16 |  |
| utv_probe | d2 | uniform | (2,) | (3,) | () | None | PASS | 2.35e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2,) | (3,) | () | None | PASS | 2.28e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.09e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2,) | (3,) | () | None | PASS | 2.08e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | () | None | PASS | 2.01e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.74e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.16e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | () | None | PASS | 3.90e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | () | None | PASS | 5.31e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | () | None | PASS | 5.31e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | () | None | PASS | 1.53e-16 |  |
| utv_apply | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 3.34e-16 |  |
| utv_entries | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 3.32e-16 |  |
| utv_probe | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.57e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.08e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.98e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.35e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.54e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.53e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 4.12e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 6.98e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.79e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.82e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.16e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.79e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.82e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.84e-16 |  |
| u_entries | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 2.33e-17 |  |
| u_probe | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.46e-16 |  |
| u_apply_derivatives | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 2.18e-16 |  |
| u_entries_derivatives | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.98e-16 |  |
| u_probe_derivatives | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.90e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 9.97e-17 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.28e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.28e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.38e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.12e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.91e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 7.03e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 5.58e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.06e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.74e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.50e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.45e-16 |  |
| utv_apply | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.34e-16 |  |
| utv_entries | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.19e-16 |  |
| utv_probe | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 2.28e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 2.21e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 2.34e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 2.10e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.19e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.18e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.40e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 2.37e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.07e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.14e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 1.95e-14 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 4.89e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | () | None | PASS | 2.21e-14 |  |
| utv_apply | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.30e-16 |  |
| utv_entries | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.98e-16 |  |
| utv_probe | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.53e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.30e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.89e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.25e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.59e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.13e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.31e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 7.18e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.13e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.31e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.04e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.74e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.04e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.74e-16 |  |
| u_manifold_inner | d2 | uniform | (2,) | () | () | None | PASS | 2.57e-15 |  |
| u_manifold_norm | d2 | uniform | (2,) | () | () | None | FAIL | 4.66e+00 |  |
| u_gauge_project_idempotent | d2 | uniform | (2,) | () | () | None | PASS | 3.45e-16 |  |
| u_tangent_add_scale | d2 | uniform | (2,) | () | () | None | PASS | 1.91e-16 |  |
| u_tangent_reverse | d2 | uniform | (2,) | () | () | None | PASS | 1.06e-16 |  |
| u_retract_zero | d2 | uniform | (2,) | () | () | None | PASS | 4.82e-16 |  |
| u_retract_fd_jacobian | d2 | uniform | (2,) | () | () | None | PASS | 9.21e-08 | ratio=4.00 |
| u_retract_vs_ragged | d2 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d2 | uniform | (2,) | () | () | None | PASS | 1.69e-17 |  |
| u_transport_identity | d2 | uniform | (2,) | () | () | None | PASS | 4.82e-16 |  |
| u_transport_vs_ragged_projection | d2 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.64e-16 |  |
| u_manifold_norm | d2 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| u_gauge_project_idempotent | d2 | uniform | (2,) | () | (2,) | None | PASS | 2.58e-16 |  |
| u_tangent_add_scale | d2 | uniform | (2,) | () | (2,) | None | PASS | 1.53e-16 |  |
| u_tangent_reverse | d2 | uniform | (2,) | () | (2,) | None | PASS | 8.67e-17 |  |
| u_retract_zero | d2 | uniform | (2,) | () | (2,) | None | PASS | 4.82e-16 |  |
| u_retract_fd_jacobian | d2 | uniform | (2,) | () | (2,) | None | PASS | 4.25e-07 | ratio=4.00 |
| u_project_ambient | d2 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 12 into shape (2,2,2,3,1) |
| u_transport_identity | d2 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 12 into shape (2,2,2,3,1) |
| u_transport_vs_ragged_projection | d2 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 12 into shape (2,2,2,3,1) |
| u_to_dense | d2 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d2 | uniform | (2, 3) | () | () | None | PASS | 8.20e-17 |  |
| u_sub | d2 | uniform | (2, 3) | () | () | None | PASS | 5.30e-17 |  |
| u_scalar_mul | d2 | uniform | (2, 3) | () | () | None | PASS | 1.02e-16 |  |
| u_inner | d2 | uniform | (2, 3) | () | () | None | PASS | 1.03e-15 |  |
| u_norm | d2 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| u_reverse | d2 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_t3svd_lossless | d2 | uniform | (2, 3) | () | () | None | PASS | 8.82e-16 |  |
| u_rank_adjustment_sweep | d2 | uniform | (2, 3) | () | () | None | PASS | 7.09e-16 |  |
| u_t3svd_trunc_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 2.71e-16 |  |
| u_orthogonal_representations | d2 | uniform | (2, 3) | () | () | None | PASS | 1.12e-15 |  |
| u_apply | d2 | uniform | (2, 3) | () | () | None | PASS | 2.11e-16 |  |
| u_entries | d2 | uniform | (2, 3) | () | () | None | PASS | 1.74e-16 |  |
| u_probe | d2 | uniform | (2, 3) | () | () | None | PASS | 2.13e-16 |  |
| u_apply_derivatives | d2 | uniform | (2, 3) | () | () | None | PASS | 2.17e-16 |  |
| u_entries_derivatives | d2 | uniform | (2, 3) | () | () | None | PASS | 2.03e-16 |  |
| u_probe_derivatives | d2 | uniform | (2, 3) | () | () | None | PASS | 2.05e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 8.97e-17 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 1.65e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 7.99e-17 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 8.97e-17 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 1.65e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 7.99e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 2.02e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 6.43e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 1.74e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 2.02e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 6.43e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 1.74e-16 |  |
| utv_apply | d2 | uniform | (2, 3) | () | () | None | PASS | 3.08e-16 |  |
| utv_entries | d2 | uniform | (2, 3) | () | () | None | PASS | 1.85e-16 |  |
| utv_probe | d2 | uniform | (2, 3) | () | () | None | PASS | 2.14e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2, 3) | () | () | None | PASS | 4.47e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2, 3) | () | () | None | PASS | 4.92e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2, 3) | () | () | None | PASS | 2.38e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | () | None | PASS | 2.51e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | () | None | PASS | 2.51e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | () | None | PASS | 2.32e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | () | None | PASS | 3.20e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | () | None | PASS | 2.32e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | () | None | PASS | 3.20e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 3.48e-16 |  |
| utv_entries | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.20e-16 |  |
| utv_probe | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 2.45e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.97e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 2.06e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 2.12e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.21e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.88e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 4.20e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.21e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.88e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 4.20e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 3.07e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 3.56e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.76e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 3.07e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 3.56e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.76e-16 |  |
| u_apply | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 4.20e-16 |  |
| u_entries | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.28e-16 |  |
| u_probe | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.87e-16 |  |
| u_apply_derivatives | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.60e-16 |  |
| u_entries_derivatives | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.73e-16 |  |
| u_probe_derivatives | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.95e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 4.27e-17 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 7.09e-17 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 7.84e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 4.01e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.25e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.38e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 7.18e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.66e-16 |  |
| utv_apply | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.77e-16 |  |
| utv_entries | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.63e-16 |  |
| utv_probe | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 2.41e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 3.32e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 3.39e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 2.44e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 3.78e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.26e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 2.16e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.89e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 1.69e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 2.16e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 9.30e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 3.68e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 4.52e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 7.44e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 7.36e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.93e-16 |  |
| utv_entries | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.90e-16 |  |
| utv_probe | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.18e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.30e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.32e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.17e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.48e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.67e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 7.93e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.48e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.67e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.17e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.03e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.24e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.35e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.71e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.87e-16 |  |
| u_apply | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.67e-16 |  |
| u_entries | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.22e-16 |  |
| u_probe | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.32e-16 |  |
| u_apply_derivatives | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.02e-16 |  |
| u_entries_derivatives | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.94e-16 |  |
| u_probe_derivatives | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.23e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.40e-17 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.33e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.12e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.06e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.98e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 9.69e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.43e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.14e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.04e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.19e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.39e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.70e-16 |  |
| utv_apply | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.28e-16 |  |
| utv_entries | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.64e-16 |  |
| utv_probe | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.43e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.30e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.09e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.40e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 7.25e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.12e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 6.04e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.12e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.81e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 7.52e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.04e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 9.02e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.85e-16 |  |
| utv_apply | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.52e-16 |  |
| utv_entries | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.80e-16 |  |
| utv_probe | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.31e-16 |  |
| utv_apply_derivatives | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.44e-16 |  |
| utv_entries_derivatives | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.23e-16 |  |
| utv_probe_derivatives | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.21e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.81e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.41e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.18e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.05e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.94e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.61e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.05e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 5.89e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.21e-16 |  |
| u_manifold_inner | d2 | uniform | (2, 3) | () | () | None | PASS | 6.89e-16 |  |
| u_manifold_norm | d2 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| u_gauge_project_idempotent | d2 | uniform | (2, 3) | () | () | None | PASS | 3.00e-16 |  |
| u_tangent_add_scale | d2 | uniform | (2, 3) | () | () | None | PASS | 2.09e-16 |  |
| u_tangent_reverse | d2 | uniform | (2, 3) | () | () | None | PASS | 7.66e-17 |  |
| u_retract_zero | d2 | uniform | (2, 3) | () | () | None | PASS | 4.62e-16 |  |
| u_retract_fd_jacobian | d2 | uniform | (2, 3) | () | () | None | PASS | 1.09e-03 | ratio=3.85 |
| u_retract_vs_ragged | d2 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d2 | uniform | (2, 3) | () | () | None | PASS | 1.80e-16 |  |
| u_transport_identity | d2 | uniform | (2, 3) | () | () | None | PASS | 4.62e-16 |  |
| u_transport_vs_ragged_projection | d2 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.17e-15 |  |
| u_manifold_norm | d2 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| u_gauge_project_idempotent | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 5.33e-16 |  |
| u_tangent_add_scale | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.76e-16 |  |
| u_tangent_reverse | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 8.14e-17 |  |
| u_retract_zero | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 4.62e-16 |  |
| u_retract_fd_jacobian | d2 | uniform | (2, 3) | () | (2,) | None | PASS | 1.50e-03 | ratio=3.82 |
| u_project_ambient | d2 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 36 into shape (2,2,2,3,3,1) |
| u_transport_identity | d2 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 36 into shape (2,2,2,3,3,1) |
| u_transport_vs_ragged_projection | d2 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 36 into shape (2,2,2,3,3,1) |
| u_to_dense | d2 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_add | d2 | uniform+pad | () | () | () | None | PASS | 1.46e-16 |  |
| u_sub | d2 | uniform+pad | () | () | () | None | PASS | 1.71e-16 |  |
| u_scalar_mul | d2 | uniform+pad | () | () | () | None | PASS | 1.57e-16 |  |
| u_inner | d2 | uniform+pad | () | () | () | None | PASS | 2.19e-16 |  |
| u_norm | d2 | uniform+pad | () | () | () | None | FAIL | 3.84e+02 |  |
| u_reverse | d2 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_t3svd_lossless | d2 | uniform+pad | () | () | () | None | PASS | 8.25e-16 |  |
| u_rank_adjustment_sweep | d2 | uniform+pad | () | () | () | None | PASS | 6.45e-16 |  |
| u_t3svd_trunc_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 1.31e-15 |  |
| u_orthogonal_representations | d2 | uniform+pad | () | () | () | None | PASS | 6.97e-16 |  |
| u_apply | d2 | uniform+pad | () | () | () | None | PASS | 1.59e-16 |  |
| u_entries | d2 | uniform+pad | () | () | () | None | PASS | 2.10e-16 |  |
| u_probe | d2 | uniform+pad | () | () | () | None | PASS | 1.26e-16 |  |
| u_apply_derivatives | d2 | uniform+pad | () | () | () | None | PASS | 4.16e-16 |  |
| u_entries_derivatives | d2 | uniform+pad | () | () | () | None | PASS | 4.81e-16 |  |
| u_probe_derivatives | d2 | uniform+pad | () | () | () | None | PASS | 5.29e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 1.78e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 1.26e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 1.78e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 1.26e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 2.46e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 2.09e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 1.36e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 2.46e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 2.09e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 1.36e-16 |  |
| utv_apply | d2 | uniform+pad | () | () | () | None | PASS | 1.77e-16 |  |
| utv_entries | d2 | uniform+pad | () | () | () | None | PASS | 1.40e-16 |  |
| utv_probe | d2 | uniform+pad | () | () | () | None | PASS | 1.87e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | () | () | () | None | PASS | 1.01e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | () | () | () | None | PASS | 1.86e-17 |  |
| utv_probe_derivatives | d2 | uniform+pad | () | () | () | None | PASS | 1.63e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | () | None | PASS | 6.40e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | () | None | PASS | 1.69e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | () | None | PASS | 6.40e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | () | None | PASS | 1.69e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | () | None | PASS | 4.02e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | () | None | PASS | 1.71e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | () | None | PASS | 4.02e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | () | None | PASS | 1.71e-15 |  |
| utv_apply | d2 | uniform+pad | () | () | (2,) | None | PASS | 1.25e-16 |  |
| utv_entries | d2 | uniform+pad | () | () | (2,) | None | PASS | 2.84e-16 |  |
| utv_probe | d2 | uniform+pad | () | () | (2,) | None | PASS | 2.66e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | () | () | (2,) | None | PASS | 2.36e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | () | () | (2,) | None | PASS | 1.80e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | () | () | (2,) | None | PASS | 2.03e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | (2,) | None | PASS | 1.30e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | (2,) | None | PASS | 3.12e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | (2,) | None | PASS | 1.30e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | (2,) | None | PASS | 3.12e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | (2,) | None | PASS | 1.93e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | (2,) | None | PASS | 3.06e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | () | (2,) | None | PASS | 4.00e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | (2,) | None | PASS | 1.93e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | (2,) | None | PASS | 3.06e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | () | (2,) | None | PASS | 4.00e-15 |  |
| u_apply | d2 | uniform+pad | () | (3,) | () | None | PASS | 3.70e-16 |  |
| u_entries | d2 | uniform+pad | () | (3,) | () | None | PASS | 7.30e-16 |  |
| u_probe | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.71e-16 |  |
| u_apply_derivatives | d2 | uniform+pad | () | (3,) | () | None | PASS | 7.07e-16 |  |
| u_entries_derivatives | d2 | uniform+pad | () | (3,) | () | None | PASS | 9.20e-16 |  |
| u_probe_derivatives | d2 | uniform+pad | () | (3,) | () | None | PASS | 2.92e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.50e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 6.96e-17 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 3.12e-17 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.33e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.39e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 8.12e-18 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.21e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.09e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 6.90e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 4.78e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.45e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (3,) | () | None | PASS | 2.21e-16 |  |
| utv_apply | d2 | uniform+pad | () | (3,) | () | None | PASS | 2.66e-16 |  |
| utv_entries | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.55e-16 |  |
| utv_probe | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.96e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | () | (3,) | () | None | PASS | 2.24e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | () | (3,) | () | None | PASS | 2.51e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.75e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | () | None | PASS | 2.52e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | () | None | PASS | 2.30e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | () | None | PASS | 2.52e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | () | None | PASS | 4.07e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.44e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | () | None | PASS | 6.11e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.14e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | () | None | PASS | 1.44e-16 |  |
| utv_apply | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.17e-15 |  |
| utv_entries | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.26e-16 |  |
| utv_probe | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.93e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.52e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.35e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.88e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 9.42e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.20e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 4.87e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.88e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.20e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 7.30e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.91e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.49e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.03e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.91e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.49e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.03e-16 |  |
| u_apply | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 3.97e-16 |  |
| u_entries | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 5.65e-16 |  |
| u_probe | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 4.20e-16 |  |
| u_apply_derivatives | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 4.42e-16 |  |
| u_entries_derivatives | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 4.24e-16 |  |
| u_probe_derivatives | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 3.08e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 2.56e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.72e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.67e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 4.50e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 2.72e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 3.35e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.09e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.15e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.30e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 3.67e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.41e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 4.47e-16 |  |
| utv_apply | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 2.05e-16 |  |
| utv_entries | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.49e-16 |  |
| utv_probe | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.66e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.93e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.89e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.88e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 3.54e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.69e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 3.54e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.69e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 1.54e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 5.79e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 2.89e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | () | None | PASS | 5.79e-16 |  |
| utv_apply | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 6.06e-16 |  |
| utv_entries | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.67e-16 |  |
| utv_probe | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.88e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.77e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.76e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.59e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.64e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.31e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.28e-13 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.43e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.31e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 4.26e-14 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.45e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.20e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 5.70e-16 |  |
| u_manifold_inner | d2 | uniform+pad | () | () | () | None | PASS | 1.15e-15 |  |
| u_manifold_norm | d2 | uniform+pad | () | () | () | None | FAIL | 1.66e+02 |  |
| u_gauge_project_idempotent | d2 | uniform+pad | () | () | () | None | PASS | 4.98e-16 |  |
| u_tangent_add_scale | d2 | uniform+pad | () | () | () | None | PASS | 1.69e-16 |  |
| u_tangent_reverse | d2 | uniform+pad | () | () | () | None | PASS | 7.50e-17 |  |
| u_retract_zero | d2 | uniform+pad | () | () | () | None | PASS | 5.51e-16 |  |
| u_retract_fd_jacobian | d2 | uniform+pad | () | () | () | None | PASS | 1.14e-08 | ratio=4.00 |
| u_retract_vs_ragged | d2 | uniform+pad | () | () | () | None | PASS | 1.85e-15 |  |
| u_project_ambient | d2 | uniform+pad | () | () | () | None | PASS | 9.06e-16 |  |
| u_transport_identity | d2 | uniform+pad | () | () | () | None | PASS | 3.19e-16 |  |
| u_transport_vs_ragged_projection | d2 | uniform+pad | () | () | () | None | PASS | 3.40e-16 |  |
| u_manifold_inner | d2 | uniform+pad | () | () | (2,) | None | PASS | 3.34e-16 |  |
| u_manifold_norm | d2 | uniform+pad | () | () | (2,) | None | FAIL | 1.08e+02 |  |
| u_gauge_project_idempotent | d2 | uniform+pad | () | () | (2,) | None | PASS | 5.39e-16 |  |
| u_tangent_add_scale | d2 | uniform+pad | () | () | (2,) | None | PASS | 1.81e-16 |  |
| u_tangent_reverse | d2 | uniform+pad | () | () | (2,) | None | PASS | 1.01e-16 |  |
| u_retract_zero | d2 | uniform+pad | () | () | (2,) | None | PASS | 5.51e-16 |  |
| u_retract_fd_jacobian | d2 | uniform+pad | () | () | (2,) | None | PASS | 3.75e-08 | ratio=4.00 |
| u_project_ambient | d2 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 10 into shape (2,2,5,1) |
| u_transport_identity | d2 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 10 into shape (2,2,5,1) |
| u_transport_vs_ragged_projection | d2 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 10 into shape (2,2,5,1) |
| u_to_dense | d2 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.17e-16 |  |
| u_sub | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.10e-16 |  |
| u_scalar_mul | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.26e-16 |  |
| u_inner | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.32e-15 |  |
| u_norm | d2 | uniform+pad | (2,) | () | () | None | FAIL | 1.35e+01 |  |
| u_reverse | d2 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_t3svd_lossless | d2 | uniform+pad | (2,) | () | () | None | PASS | 5.68e-16 |  |
| u_rank_adjustment_sweep | d2 | uniform+pad | (2,) | () | () | None | PASS | 5.93e-16 |  |
| u_t3svd_trunc_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 9.37e-16 |  |
| u_orthogonal_representations | d2 | uniform+pad | (2,) | () | () | None | PASS | 6.83e-16 |  |
| u_apply | d2 | uniform+pad | (2,) | () | () | None | PASS | 2.73e-16 |  |
| u_entries | d2 | uniform+pad | (2,) | () | () | None | PASS | 4.42e-17 |  |
| u_probe | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.89e-16 |  |
| u_apply_derivatives | d2 | uniform+pad | (2,) | () | () | None | PASS | 4.59e-16 |  |
| u_entries_derivatives | d2 | uniform+pad | (2,) | () | () | None | PASS | 4.38e-16 |  |
| u_probe_derivatives | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.92e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.96e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.57e-17 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 3.08e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.96e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.57e-17 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 3.08e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 7.72e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 7.48e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.14e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 7.72e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 7.48e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.14e-16 |  |
| utv_apply | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.87e-16 |  |
| utv_entries | d2 | uniform+pad | (2,) | () | () | None | PASS | 2.70e-16 |  |
| utv_probe | d2 | uniform+pad | (2,) | () | () | None | PASS | 2.02e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2,) | () | () | None | PASS | 4.97e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2,) | () | () | None | PASS | 6.95e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2,) | () | () | None | PASS | 3.07e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.11e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | () | None | PASS | 3.65e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.18e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.11e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | () | None | PASS | 3.65e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.18e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | () | None | PASS | 2.23e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | () | None | PASS | 9.01e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.50e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | () | None | PASS | 2.23e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | () | None | PASS | 9.01e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.50e-16 |  |
| utv_apply | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.69e-16 |  |
| utv_entries | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 7.06e-16 |  |
| utv_probe | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.55e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.74e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.41e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.36e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 6.00e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 6.00e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 8.98e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 6.34e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 8.98e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 6.34e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.14e-16 |  |
| u_entries | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 7.89e-17 |  |
| u_probe | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.96e-16 |  |
| u_apply_derivatives | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.64e-16 |  |
| u_entries_derivatives | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.48e-16 |  |
| u_probe_derivatives | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.05e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.19e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.47e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.78e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.74e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.41e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.81e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.61e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.04e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.96e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.19e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.58e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.79e-16 |  |
| utv_apply | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 6.67e-17 |  |
| utv_entries | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.85e-16 |  |
| utv_probe | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.93e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.18e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.49e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.60e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.38e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.02e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.25e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.38e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.02e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.25e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.17e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.90e-16 |  |
| utv_entries | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 8.63e-17 |  |
| utv_probe | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.77e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.58e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.49e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.43e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.54e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.76e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 9.51e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.54e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.76e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.19e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.27e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.92e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.37e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 4.24e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.37e-16 |  |
| u_apply | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.81e-16 |  |
| u_entries | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 5.20e-17 |  |
| u_probe | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.41e-16 |  |
| u_apply_derivatives | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.16e-16 |  |
| u_entries_derivatives | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.89e-16 |  |
| u_probe_derivatives | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.90e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.06e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.33e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.12e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.54e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.39e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.55e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.09e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 7.10e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.39e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.69e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.62e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.84e-16 |  |
| utv_apply | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.12e-16 |  |
| utv_entries | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.00e-16 |  |
| utv_probe | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.73e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.56e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.32e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.66e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.78e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.38e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.26e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.51e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 6.76e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 8.38e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 5.73e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.12e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.76e-14 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 8.60e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.23e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.00e-14 |  |
| utv_apply | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.91e-16 |  |
| utv_entries | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 7.64e-17 |  |
| utv_probe | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.42e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.10e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.13e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.43e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.98e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.27e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.17e-14 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.98e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.70e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.74e-14 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 5.63e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.12e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.30e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 9.38e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 5.20e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 4.61e-16 |  |
| u_manifold_inner | d2 | uniform+pad | (2,) | () | () | None | PASS | 5.50e-16 |  |
| u_manifold_norm | d2 | uniform+pad | (2,) | () | () | None | FAIL | 2.16e+01 |  |
| u_gauge_project_idempotent | d2 | uniform+pad | (2,) | () | () | None | PASS | 2.83e-16 |  |
| u_tangent_add_scale | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.88e-16 |  |
| u_tangent_reverse | d2 | uniform+pad | (2,) | () | () | None | PASS | 1.26e-16 |  |
| u_retract_zero | d2 | uniform+pad | (2,) | () | () | None | PASS | 7.20e-16 |  |
| u_retract_fd_jacobian | d2 | uniform+pad | (2,) | () | () | None | PASS | 4.62e-07 | ratio=4.00 |
| u_retract_vs_ragged | d2 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d2 | uniform+pad | (2,) | () | () | None | PASS | 6.75e-16 |  |
| u_transport_identity | d2 | uniform+pad | (2,) | () | () | None | PASS | 4.03e-16 |  |
| u_transport_vs_ragged_projection | d2 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.18e-15 |  |
| u_manifold_norm | d2 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| u_gauge_project_idempotent | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.30e-16 |  |
| u_tangent_add_scale | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.04e-16 |  |
| u_tangent_reverse | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.30e-16 |  |
| u_retract_zero | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 7.20e-16 |  |
| u_retract_fd_jacobian | d2 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.50e-06 | ratio=4.00 |
| u_project_ambient | d2 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 20 into shape (2,2,2,5,1) |
| u_transport_identity | d2 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 20 into shape (2,2,2,5,1) |
| u_transport_vs_ragged_projection | d2 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 20 into shape (2,2,2,5,1) |
| u_to_dense | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 8.20e-17 |  |
| u_sub | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 5.30e-17 |  |
| u_scalar_mul | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.02e-16 |  |
| u_inner | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.45e-15 |  |
| u_norm | d2 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| u_reverse | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_t3svd_lossless | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.06e-15 |  |
| u_rank_adjustment_sweep | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.26e-15 |  |
| u_t3svd_trunc_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.88e-15 |  |
| u_orthogonal_representations | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 8.98e-16 |  |
| u_apply | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.07e-16 |  |
| u_entries | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.74e-16 |  |
| u_probe | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.61e-16 |  |
| u_apply_derivatives | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.49e-16 |  |
| u_entries_derivatives | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.62e-16 |  |
| u_probe_derivatives | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.27e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.07e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.74e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.28e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.07e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.74e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.28e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.55e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 3.19e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.97e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.55e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 3.19e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.97e-16 |  |
| utv_apply | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.81e-16 |  |
| utv_entries | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.68e-16 |  |
| utv_probe | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.16e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.58e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.66e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.43e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.18e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.64e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.29e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.18e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.64e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.29e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 3.39e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.34e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.74e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 3.39e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.34e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.74e-16 |  |
| utv_apply | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.61e-16 |  |
| utv_entries | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.18e-16 |  |
| utv_probe | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.91e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.06e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.64e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.77e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.12e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.40e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.12e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.40e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.87e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.99e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.32e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.87e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.99e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.32e-16 |  |
| u_apply | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 4.34e-16 |  |
| u_entries | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 7.33e-17 |  |
| u_probe | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.38e-16 |  |
| u_apply_derivatives | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.60e-16 |  |
| u_entries_derivatives | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.57e-16 |  |
| u_probe_derivatives | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.87e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.71e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.40e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.42e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.71e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.91e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.73e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.69e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 8.34e-17 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.58e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.97e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.45e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.84e-16 |  |
| utv_apply | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.29e-16 |  |
| utv_entries | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.67e-16 |  |
| utv_probe | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.81e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.54e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.49e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.02e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.26e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.91e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.13e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.91e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.24e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.15e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.81e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.48e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 9.45e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.33e-16 |  |
| utv_entries | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.69e-16 |  |
| utv_probe | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.17e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.66e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.62e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.25e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.57e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.55e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.45e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.06e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 4.64e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.45e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.56e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.63e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.20e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.56e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.26e-16 |  |
| u_apply | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.21e-16 |  |
| u_entries | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 9.05e-17 |  |
| u_probe | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.50e-16 |  |
| u_apply_derivatives | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.10e-16 |  |
| u_entries_derivatives | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.99e-16 |  |
| u_probe_derivatives | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.13e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.43e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.40e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.57e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.65e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.42e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.19e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.47e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.18e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.33e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.21e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.40e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.88e-16 |  |
| utv_apply | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.33e-16 |  |
| utv_entries | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.69e-16 |  |
| utv_probe | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.11e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.11e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.00e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.05e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.91e-14 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.71e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.59e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 4.00e-14 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.35e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.86e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.14e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 8.54e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.86e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.14e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 4.27e-16 |  |
| utv_apply | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.45e-16 |  |
| utv_entries | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.79e-16 |  |
| utv_probe | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.96e-16 |  |
| utv_apply_derivatives | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.46e-16 |  |
| utv_entries_derivatives | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.46e-16 |  |
| utv_probe_derivatives | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.24e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.28e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.21e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.88e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.28e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.21e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.37e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.01e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.33e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.69e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d2 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.89e-16 |  |
| u_manifold_inner | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 6.67e-16 |  |
| u_manifold_norm | d2 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| u_gauge_project_idempotent | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 5.11e-16 |  |
| u_tangent_add_scale | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.56e-16 |  |
| u_tangent_reverse | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.14e-16 |  |
| u_retract_zero | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 1.07e-15 |  |
| u_retract_fd_jacobian | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 2.76e-03 | ratio=3.71 |
| u_retract_vs_ragged | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 7.59e-16 |  |
| u_transport_identity | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 5.60e-16 |  |
| u_transport_vs_ragged_projection | d2 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 6.57e-16 |  |
| u_manifold_norm | d2 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| u_gauge_project_idempotent | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.88e-16 |  |
| u_tangent_add_scale | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.90e-16 |  |
| u_tangent_reverse | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.07e-16 |  |
| u_retract_zero | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.07e-15 |  |
| u_retract_fd_jacobian | d2 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 5.05e-03 | ratio=3.59 |
| u_project_ambient | d2 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 60 into shape (2,2,2,3,5,1) |
| u_transport_identity | d2 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 60 into shape (2,2,2,3,5,1) |
| u_transport_vs_ragged_projection | d2 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 60 into shape (2,2,2,3,5,1) |
