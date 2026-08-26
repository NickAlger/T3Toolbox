| op | struct | repr | C | W | K | sharing | status | max relerr | note |
|---|---|---|---|---|---|---|---|---|---|
| u_to_dense | d3 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_add | d3 | uniform | () | () | () | None | PASS | 1.08e-16 |  |
| u_sub | d3 | uniform | () | () | () | None | PASS | 1.11e-16 |  |
| u_scalar_mul | d3 | uniform | () | () | () | None | PASS | 2.23e-16 |  |
| u_inner | d3 | uniform | () | () | () | None | PASS | 2.57e-14 |  |
| u_norm | d3 | uniform | () | () | () | None | PASS | 1.41e-15 |  |
| u_reverse | d3 | uniform | () | () | () | None | PASS | 1.26e-16 |  |
| u_t3svd_lossless | d3 | uniform | () | () | () | None | PASS | 2.92e-15 |  |
| u_rank_adjustment_sweep | d3 | uniform | () | () | () | None | PASS | 3.09e-15 |  |
| u_t3svd_trunc_vs_ragged | d3 | uniform | () | () | () | None | PASS | 4.35e-16 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | d3 | uniform | () | () | () | None | FAIL | 1.36e-01 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | d3 | uniform | () | () | () | None | PASS | 3.03e-15 |  |
| u_apply | d3 | uniform | () | () | () | None | PASS | 3.80e-15 |  |
| u_entries | d3 | uniform | () | () | () | None | PASS | 4.46e-13 |  |
| u_probe | d3 | uniform | () | () | () | None | PASS | 5.26e-15 |  |
| u_apply_derivatives | d3 | uniform | () | () | () | None | PASS | 6.54e-16 |  |
| u_entries_derivatives | d3 | uniform | () | () | () | None | PASS | 2.97e-16 |  |
| u_probe_derivatives | d3 | uniform | () | () | () | None | PASS | 2.07e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| utv_apply | d3 | uniform | () | () | () | None | PASS | 2.19e-16 |  |
| utv_entries | d3 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe | d3 | uniform | () | () | () | None | PASS | 2.94e-16 |  |
| utv_apply_derivatives | d3 | uniform | () | () | () | None | PASS | 3.80e-16 |  |
| utv_entries_derivatives | d3 | uniform | () | () | () | None | PASS | 3.01e-16 |  |
| utv_probe_derivatives | d3 | uniform | () | () | () | None | PASS | 3.38e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | () | () | () | None | PASS | 3.94e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | () | () | () | None | PASS | 1.25e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | () | () | () | None | PASS | 2.75e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | () | () | () | None | PASS | 3.94e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | () | () | () | None | PASS | 1.25e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | () | () | () | None | PASS | 2.75e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | () | () | None | PASS | 4.92e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | () | () | None | PASS | 3.01e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | () | () | None | PASS | 1.42e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | () | () | None | PASS | 4.92e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | () | () | None | PASS | 3.01e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | () | () | None | PASS | 1.42e-15 |  |
| utv_apply | d3 | uniform | () | () | (2,) | None | PASS | 3.93e-16 |  |
| utv_entries | d3 | uniform | () | () | (2,) | None | PASS | 3.88e-16 |  |
| utv_probe | d3 | uniform | () | () | (2,) | None | PASS | 4.55e-16 |  |
| utv_apply_derivatives | d3 | uniform | () | () | (2,) | None | PASS | 2.34e-16 |  |
| utv_entries_derivatives | d3 | uniform | () | () | (2,) | None | PASS | 2.32e-16 |  |
| utv_probe_derivatives | d3 | uniform | () | () | (2,) | None | PASS | 3.89e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | () | () | (2,) | None | PASS | 1.35e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | () | () | (2,) | None | PASS | 2.51e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | () | () | (2,) | None | PASS | 1.34e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | () | () | (2,) | None | PASS | 1.35e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | () | () | (2,) | None | PASS | 2.51e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | () | () | (2,) | None | PASS | 1.34e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | () | (2,) | None | PASS | 2.07e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | () | (2,) | None | PASS | 2.51e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | () | (2,) | None | PASS | 2.07e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | () | (2,) | None | PASS | 2.51e-16 |  |
| u_apply | d3 | uniform | () | (3,) | () | None | PASS | 1.18e-14 |  |
| u_entries | d3 | uniform | () | (3,) | () | None | PASS | 5.67e-15 |  |
| u_probe | d3 | uniform | () | (3,) | () | None | PASS | 9.50e-15 |  |
| u_apply_derivatives | d3 | uniform | () | (3,) | () | None | PASS | 2.49e-15 |  |
| u_entries_derivatives | d3 | uniform | () | (3,) | () | None | PASS | 2.44e-15 |  |
| u_probe_derivatives | d3 | uniform | () | (3,) | () | None | PASS | 3.80e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| utv_apply | d3 | uniform | () | (3,) | () | None | PASS | 2.17e-16 |  |
| utv_entries | d3 | uniform | () | (3,) | () | None | PASS | 2.01e-16 |  |
| utv_probe | d3 | uniform | () | (3,) | () | None | PASS | 4.31e-16 |  |
| utv_apply_derivatives | d3 | uniform | () | (3,) | () | None | PASS | 7.86e-16 |  |
| utv_entries_derivatives | d3 | uniform | () | (3,) | () | None | PASS | 7.30e-16 |  |
| utv_probe_derivatives | d3 | uniform | () | (3,) | () | None | PASS | 3.75e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | () | None | PASS | 2.08e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | () | None | PASS | 3.11e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | () | None | PASS | 1.68e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | () | None | PASS | 4.15e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | () | None | PASS | 1.56e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | () | None | PASS | 2.40e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | () | None | PASS | 4.64e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | () | None | PASS | 3.78e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | () | None | PASS | 3.28e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | () | None | PASS | 2.32e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | () | None | PASS | 7.56e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | () | None | PASS | 1.64e-16 |  |
| utv_apply | d3 | uniform | () | (3,) | (2,) | None | PASS | 2.52e-16 |  |
| utv_entries | d3 | uniform | () | (3,) | (2,) | None | PASS | 2.14e-16 |  |
| utv_probe | d3 | uniform | () | (3,) | (2,) | None | PASS | 3.14e-16 |  |
| utv_apply_derivatives | d3 | uniform | () | (3,) | (2,) | None | PASS | 3.69e-16 |  |
| utv_entries_derivatives | d3 | uniform | () | (3,) | (2,) | None | PASS | 4.00e-16 |  |
| utv_probe_derivatives | d3 | uniform | () | (3,) | (2,) | None | PASS | 3.44e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | (2,) | None | PASS | 1.94e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | (2,) | None | PASS | 1.42e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | (2,) | None | PASS | 3.88e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | (2,) | None | PASS | 1.40e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | (2,) | None | PASS | 1.42e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (3,) | (2,) | None | PASS | 1.36e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | (2,) | None | PASS | 4.51e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | (2,) | None | PASS | 1.42e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (3,) | (2,) | None | PASS | 4.09e-16 |  |
| u_apply | d3 | uniform | () | (2, 2) | () | None | PASS | 2.20e-15 |  |
| u_entries | d3 | uniform | () | (2, 2) | () | None | PASS | 3.65e-15 |  |
| u_probe | d3 | uniform | () | (2, 2) | () | None | PASS | 8.20e-15 |  |
| u_apply_derivatives | d3 | uniform | () | (2, 2) | () | None | PASS | 2.39e-15 |  |
| u_entries_derivatives | d3 | uniform | () | (2, 2) | () | None | PASS | 2.31e-15 |  |
| u_probe_derivatives | d3 | uniform | () | (2, 2) | () | None | PASS | 4.71e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| utv_apply | d3 | uniform | () | (2, 2) | () | None | PASS | 5.23e-16 |  |
| utv_entries | d3 | uniform | () | (2, 2) | () | None | PASS | 1.86e-16 |  |
| utv_probe | d3 | uniform | () | (2, 2) | () | None | PASS | 3.76e-16 |  |
| utv_apply_derivatives | d3 | uniform | () | (2, 2) | () | None | PASS | 3.77e-16 |  |
| utv_entries_derivatives | d3 | uniform | () | (2, 2) | () | None | PASS | 2.73e-16 |  |
| utv_probe_derivatives | d3 | uniform | () | (2, 2) | () | None | PASS | 3.69e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | () | None | PASS | 1.11e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | () | None | PASS | 3.00e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | () | None | PASS | 8.31e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | () | None | PASS | 2.08e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | () | None | PASS | 4.50e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | () | None | PASS | 3.52e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | () | None | PASS | 3.45e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | () | None | PASS | 5.29e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | () | None | PASS | 1.72e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | () | None | PASS | 5.46e-16 |  |
| utv_apply | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 6.66e-16 |  |
| utv_entries | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 1.73e-16 |  |
| utv_probe | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 3.76e-16 |  |
| utv_apply_derivatives | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 3.57e-16 |  |
| utv_entries_derivatives | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 4.48e-16 |  |
| utv_probe_derivatives | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 3.40e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 1.52e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 4.37e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 1.30e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 1.75e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 1.30e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 2.88e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d3 | uniform | () | () | () | None | PASS | 4.86e-16 |  |
| u_manifold_norm | d3 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_gauge_project_idempotent | d3 | uniform | () | () | () | None | PASS | 1.98e-16 |  |
| u_tangent_add_scale | d3 | uniform | () | () | () | None | PASS | 1.87e-16 |  |
| u_tangent_reverse | d3 | uniform | () | () | () | None | PASS | 2.08e-16 |  |
| u_retract_zero | d3 | uniform | () | () | () | None | PASS | 8.89e-16 |  |
| u_retract_fd_jacobian | d3 | uniform | () | () | () | None | PASS | 6.33e-07 | ratio=4.00 |
| u_retract_vs_ragged | d3 | uniform | () | () | () | None | PASS | 1.13e-15 |  |
| u_project_ambient | d3 | uniform | () | () | () | None | PASS | 9.30e-16 |  |
| u_transport_identity | d3 | uniform | () | () | () | None | PASS | 3.83e-16 |  |
| u_transport_vs_ragged_projection | d3 | uniform | () | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | d3 | uniform | () | () | (2,) | None | PASS | 5.36e-16 |  |
| u_manifold_norm | d3 | uniform | () | () | (2,) | None | PASS | 2.01e-16 |  |
| u_gauge_project_idempotent | d3 | uniform | () | () | (2,) | None | PASS | 2.53e-16 |  |
| u_tangent_add_scale | d3 | uniform | () | () | (2,) | None | PASS | 2.24e-16 |  |
| u_tangent_reverse | d3 | uniform | () | () | (2,) | None | PASS | 1.63e-16 |  |
| u_retract_zero | d3 | uniform | () | () | (2,) | None | PASS | 8.89e-16 |  |
| u_retract_fd_jacobian | d3 | uniform | () | () | (2,) | None | PASS | 2.00e-07 | ratio=4.00 |
| u_project_ambient | d3 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 9 into shape (3,2,3,1) |
| u_transport_identity | d3 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 9 into shape (3,2,3,1) |
| u_transport_vs_ragged_projection | d3 | uniform | () | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | d3 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d3 | uniform | (2,) | () | () | None | PASS | 1.06e-16 |  |
| u_sub | d3 | uniform | (2,) | () | () | None | PASS | 9.30e-17 |  |
| u_scalar_mul | d3 | uniform | (2,) | () | () | None | PASS | 1.50e-16 |  |
| u_inner | d3 | uniform | (2,) | () | () | None | PASS | 2.38e-16 |  |
| u_norm | d3 | uniform | (2,) | () | () | None | PASS | 4.93e-17 |  |
| u_reverse | d3 | uniform | (2,) | () | () | None | PASS | 1.25e-16 |  |
| u_t3svd_lossless | d3 | uniform | (2,) | () | () | None | PASS | 6.29e-16 |  |
| u_rank_adjustment_sweep | d3 | uniform | (2,) | () | () | None | PASS | 6.64e-16 |  |
| u_t3svd_trunc_vs_ragged | d3 | uniform | (2,) | () | () | None | PASS | 3.73e-16 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | d3 | uniform | (2,) | () | () | None | FAIL | 6.21e-01 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | d3 | uniform | (2,) | () | () | None | PASS | 6.68e-16 |  |
| u_apply | d3 | uniform | (2,) | () | () | None | PASS | 1.80e-15 |  |
| u_entries | d3 | uniform | (2,) | () | () | None | PASS | 7.70e-16 |  |
| u_probe | d3 | uniform | (2,) | () | () | None | PASS | 1.50e-15 |  |
| u_apply_derivatives | d3 | uniform | (2,) | () | () | None | PASS | 1.40e-15 |  |
| u_entries_derivatives | d3 | uniform | (2,) | () | () | None | PASS | 1.38e-15 |  |
| u_probe_derivatives | d3 | uniform | (2,) | () | () | None | PASS | 1.08e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| utv_apply | d3 | uniform | (2,) | () | () | None | PASS | 1.77e-16 |  |
| utv_entries | d3 | uniform | (2,) | () | () | None | PASS | 2.19e-16 |  |
| utv_probe | d3 | uniform | (2,) | () | () | None | PASS | 2.40e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2,) | () | () | None | PASS | 2.31e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2,) | () | () | None | PASS | 2.02e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2,) | () | () | None | PASS | 3.87e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | () | None | PASS | 1.41e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | () | None | PASS | 7.60e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | () | None | PASS | 5.50e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | () | None | PASS | 1.41e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | () | None | PASS | 7.60e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | () | None | PASS | 5.50e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | () | None | PASS | 1.21e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | () | None | PASS | 2.95e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | () | None | PASS | 1.21e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | () | None | PASS | 2.95e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply | d3 | uniform | (2,) | () | (2,) | None | PASS | 3.63e-16 |  |
| utv_entries | d3 | uniform | (2,) | () | (2,) | None | PASS | 3.67e-16 |  |
| utv_probe | d3 | uniform | (2,) | () | (2,) | None | PASS | 3.08e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2,) | () | (2,) | None | PASS | 4.13e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2,) | () | (2,) | None | PASS | 4.08e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2,) | () | (2,) | None | PASS | 4.05e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | (2,) | None | PASS | 3.43e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | (2,) | None | PASS | 3.46e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | (2,) | None | PASS | 1.20e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | (2,) | None | PASS | 3.43e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | (2,) | None | PASS | 3.46e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | (2,) | None | PASS | 1.20e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | (2,) | None | PASS | 1.19e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | (2,) | None | PASS | 1.30e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | (2,) | None | PASS | 1.19e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | (2,) | None | PASS | 1.30e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | d3 | uniform | (2,) | (3,) | () | None | PASS | 1.39e-15 |  |
| u_entries | d3 | uniform | (2,) | (3,) | () | None | PASS | 3.47e-16 |  |
| u_probe | d3 | uniform | (2,) | (3,) | () | None | PASS | 1.50e-15 |  |
| u_apply_derivatives | d3 | uniform | (2,) | (3,) | () | None | PASS | 1.63e-15 |  |
| u_entries_derivatives | d3 | uniform | (2,) | (3,) | () | None | PASS | 1.81e-15 |  |
| u_probe_derivatives | d3 | uniform | (2,) | (3,) | () | None | PASS | 1.17e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| utv_apply | d3 | uniform | (2,) | (3,) | () | None | PASS | 5.03e-16 |  |
| utv_entries | d3 | uniform | (2,) | (3,) | () | None | PASS | 2.15e-16 |  |
| utv_probe | d3 | uniform | (2,) | (3,) | () | None | PASS | 3.33e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2,) | (3,) | () | None | PASS | 2.74e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2,) | (3,) | () | None | PASS | 2.44e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2,) | (3,) | () | None | PASS | 3.22e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | () | None | PASS | 3.10e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | () | None | PASS | 4.09e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | () | None | PASS | 5.26e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | () | None | PASS | 6.20e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | () | None | PASS | 8.17e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | () | None | PASS | 1.17e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | () | None | PASS | 1.97e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | () | None | PASS | 1.17e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | () | None | PASS | 1.19e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | () | None | PASS | 1.97e-16 |  |
| utv_apply | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 5.25e-16 |  |
| utv_entries | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.01e-16 |  |
| utv_probe | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 4.18e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 4.56e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 4.45e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 4.29e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 6.47e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.62e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.14e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 5.82e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.95e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 7.95e-14 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.47e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.65e-14 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (3,) | (2,) | None | PASS | 6.96e-16 |  |
| u_apply | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 8.61e-16 |  |
| u_entries | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 1.96e-15 |  |
| u_probe | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 7.41e-16 |  |
| u_apply_derivatives | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 5.17e-16 |  |
| u_entries_derivatives | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 4.17e-16 |  |
| u_probe_derivatives | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 6.97e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| utv_apply | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 3.32e-16 |  |
| utv_entries | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 2.57e-16 |  |
| utv_probe | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 2.91e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 3.58e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 3.40e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 3.02e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 3.61e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 1.35e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 3.61e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 7.21e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 3.00e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 3.60e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | () | None | PASS | 1.86e-16 |  |
| utv_apply | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.35e-16 |  |
| utv_entries | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.95e-16 |  |
| utv_probe | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.26e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.07e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.80e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.68e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.58e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.36e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.79e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.36e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.98e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 9.14e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.57e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.04e-14 |  |
| u_manifold_inner | d3 | uniform | (2,) | () | () | None | PASS | 1.45e-15 |  |
| u_manifold_norm | d3 | uniform | (2,) | () | () | None | PASS | 5.48e-16 |  |
| u_gauge_project_idempotent | d3 | uniform | (2,) | () | () | None | PASS | 3.73e-16 |  |
| u_tangent_add_scale | d3 | uniform | (2,) | () | () | None | PASS | 1.92e-16 |  |
| u_tangent_reverse | d3 | uniform | (2,) | () | () | None | PASS | 1.52e-16 |  |
| u_retract_zero | d3 | uniform | (2,) | () | () | None | PASS | 6.54e-16 |  |
| u_retract_fd_jacobian | d3 | uniform | (2,) | () | () | None | PASS | 1.93e-07 | ratio=4.00 |
| u_retract_vs_ragged | d3 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d3 | uniform | (2,) | () | () | None | PASS | 1.23e-15 |  |
| u_transport_identity | d3 | uniform | (2,) | () | () | None | PASS | 9.82e-16 |  |
| u_transport_vs_ragged_projection | d3 | uniform | (2,) | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | d3 | uniform | (2,) | () | (2,) | None | PASS | 1.86e-15 |  |
| u_manifold_norm | d3 | uniform | (2,) | () | (2,) | None | PASS | 4.11e-16 |  |
| u_gauge_project_idempotent | d3 | uniform | (2,) | () | (2,) | None | PASS | 5.23e-16 |  |
| u_tangent_add_scale | d3 | uniform | (2,) | () | (2,) | None | PASS | 2.09e-16 |  |
| u_tangent_reverse | d3 | uniform | (2,) | () | (2,) | None | PASS | 1.51e-16 |  |
| u_retract_zero | d3 | uniform | (2,) | () | (2,) | None | PASS | 6.54e-16 |  |
| u_retract_fd_jacobian | d3 | uniform | (2,) | () | (2,) | None | PASS | 3.43e-07 | ratio=4.00 |
| u_project_ambient | d3 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1) |
| u_transport_identity | d3 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1) |
| u_transport_vs_ragged_projection | d3 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | d3 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d3 | uniform | (2, 3) | () | () | None | PASS | 8.19e-17 |  |
| u_sub | d3 | uniform | (2, 3) | () | () | None | PASS | 9.01e-17 |  |
| u_scalar_mul | d3 | uniform | (2, 3) | () | () | None | PASS | 1.03e-16 |  |
| u_inner | d3 | uniform | (2, 3) | () | () | None | PASS | 3.00e-15 |  |
| u_norm | d3 | uniform | (2, 3) | () | () | None | PASS | 1.09e-15 |  |
| u_reverse | d3 | uniform | (2, 3) | () | () | None | PASS | 1.46e-16 |  |
| u_t3svd_lossless | d3 | uniform | (2, 3) | () | () | None | PASS | 9.17e-16 |  |
| u_rank_adjustment_sweep | d3 | uniform | (2, 3) | () | () | None | PASS | 8.84e-16 |  |
| u_t3svd_trunc_vs_ragged | d3 | uniform | (2, 3) | () | () | None | PASS | 5.21e-16 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | d3 | uniform | (2, 3) | () | () | None | FAIL | 7.28e-01 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | d3 | uniform | (2, 3) | () | () | None | PASS | 1.03e-15 |  |
| u_apply | d3 | uniform | (2, 3) | () | () | None | PASS | 1.38e-15 |  |
| u_entries | d3 | uniform | (2, 3) | () | () | None | PASS | 7.25e-16 |  |
| u_probe | d3 | uniform | (2, 3) | () | () | None | PASS | 1.15e-15 |  |
| u_apply_derivatives | d3 | uniform | (2, 3) | () | () | None | PASS | 1.28e-15 |  |
| u_entries_derivatives | d3 | uniform | (2, 3) | () | () | None | PASS | 1.28e-15 |  |
| u_probe_derivatives | d3 | uniform | (2, 3) | () | () | None | PASS | 8.72e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| utv_apply | d3 | uniform | (2, 3) | () | () | None | PASS | 5.05e-16 |  |
| utv_entries | d3 | uniform | (2, 3) | () | () | None | PASS | 3.65e-16 |  |
| utv_probe | d3 | uniform | (2, 3) | () | () | None | PASS | 3.19e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2, 3) | () | () | None | PASS | 3.05e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2, 3) | () | () | None | PASS | 1.90e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2, 3) | () | () | None | PASS | 4.16e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | () | None | PASS | 1.19e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | () | None | PASS | 3.96e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | () | None | PASS | 3.70e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | () | None | PASS | 1.19e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | () | None | PASS | 3.96e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | () | None | PASS | 3.70e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | () | None | PASS | 2.18e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | () | None | PASS | 1.82e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | () | None | PASS | 2.18e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | () | None | PASS | 1.82e-15 |  |
| utv_apply | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 5.96e-16 |  |
| utv_entries | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 3.31e-16 |  |
| utv_probe | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 3.40e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 3.91e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 3.63e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 3.67e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 1.46e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 7.03e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 1.71e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 1.46e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 7.03e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 1.71e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 7.19e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 2.56e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 7.19e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 2.56e-16 |  |
| u_apply | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 2.00e-15 |  |
| u_entries | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 1.44e-15 |  |
| u_probe | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 1.18e-15 |  |
| u_apply_derivatives | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 6.52e-16 |  |
| u_entries_derivatives | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 6.60e-16 |  |
| u_probe_derivatives | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 1.14e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| utv_apply | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 5.39e-16 |  |
| utv_entries | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 2.27e-16 |  |
| utv_probe | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 3.11e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 3.31e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 3.17e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 3.29e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 1.31e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 1.61e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 6.56e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 8.81e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 1.20e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 1.16e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 5.52e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 8.03e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.53e-16 |  |
| utv_entries | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.18e-16 |  |
| utv_probe | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.01e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.30e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.19e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.75e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.15e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.02e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.86e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.31e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 9.07e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.04e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 8.64e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 8.39e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 9.42e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.32e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.19e-16 |  |
| u_apply | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.11e-15 |  |
| u_entries | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.43e-15 |  |
| u_probe | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 8.97e-16 |  |
| u_apply_derivatives | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 5.70e-16 |  |
| u_entries_derivatives | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 6.82e-16 |  |
| u_probe_derivatives | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 7.96e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| utv_apply | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.05e-16 |  |
| utv_entries | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.84e-16 |  |
| utv_probe | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.18e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.95e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.56e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.63e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.05e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.53e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.37e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.57e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 7.91e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.87e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | () | None | PASS | 5.28e-16 |  |
| utv_apply | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.84e-16 |  |
| utv_entries | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.31e-16 |  |
| utv_probe | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 4.13e-16 |  |
| utv_apply_derivatives | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 4.23e-16 |  |
| utv_entries_derivatives | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 4.31e-16 |  |
| utv_probe_derivatives | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.72e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.89e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 5.62e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 5.84e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.25e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.98e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 8.53e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 4.12e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.98e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.71e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d3 | uniform | (2, 3) | () | () | None | PASS | 8.03e-16 |  |
| u_manifold_norm | d3 | uniform | (2, 3) | () | () | None | PASS | 3.54e-16 |  |
| u_gauge_project_idempotent | d3 | uniform | (2, 3) | () | () | None | PASS | 4.02e-16 |  |
| u_tangent_add_scale | d3 | uniform | (2, 3) | () | () | None | PASS | 2.05e-16 |  |
| u_tangent_reverse | d3 | uniform | (2, 3) | () | () | None | PASS | 1.48e-16 |  |
| u_retract_zero | d3 | uniform | (2, 3) | () | () | None | PASS | 8.51e-16 |  |
| u_retract_fd_jacobian | d3 | uniform | (2, 3) | () | () | None | PASS | 5.21e-05 | ratio=3.99 |
| u_retract_vs_ragged | d3 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d3 | uniform | (2, 3) | () | () | None | PASS | 5.24e-16 |  |
| u_transport_identity | d3 | uniform | (2, 3) | () | () | None | PASS | 7.91e-16 |  |
| u_transport_vs_ragged_projection | d3 | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 6.12e-16 |  |
| u_manifold_norm | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 2.35e-16 |  |
| u_gauge_project_idempotent | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 3.66e-16 |  |
| u_tangent_add_scale | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 2.15e-16 |  |
| u_tangent_reverse | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 1.49e-16 |  |
| u_retract_zero | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 8.51e-16 |  |
| u_retract_fd_jacobian | d3 | uniform | (2, 3) | () | (2,) | None | PASS | 7.88e-05 | ratio=3.98 |
| u_project_ambient | d3 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 54 into shape (3,2,2,3,3,1) |
| u_transport_identity | d3 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 54 into shape (3,2,2,3,3,1) |
| u_transport_vs_ragged_projection | d3 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | d3 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_add | d3 | uniform+pad | () | () | () | None | PASS | 1.08e-16 |  |
| u_sub | d3 | uniform+pad | () | () | () | None | PASS | 1.11e-16 |  |
| u_scalar_mul | d3 | uniform+pad | () | () | () | None | PASS | 2.23e-16 |  |
| u_inner | d3 | uniform+pad | () | () | () | None | PASS | 1.33e-14 |  |
| u_norm | d3 | uniform+pad | () | () | () | None | PASS | 2.56e-16 |  |
| u_reverse | d3 | uniform+pad | () | () | () | None | PASS | 1.26e-16 |  |
| u_t3svd_lossless | d3 | uniform+pad | () | () | () | None | PASS | 2.84e-15 |  |
| u_rank_adjustment_sweep | d3 | uniform+pad | () | () | () | None | PASS | 2.81e-15 |  |
| u_t3svd_trunc_vs_ragged | d3 | uniform+pad | () | () | () | None | PASS | 6.23e-16 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | d3 | uniform+pad | () | () | () | None | FAIL | 1.36e-01 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | d3 | uniform+pad | () | () | () | None | PASS | 2.92e-15 |  |
| u_apply | d3 | uniform+pad | () | () | () | None | PASS | 1.98e-15 |  |
| u_entries | d3 | uniform+pad | () | () | () | None | PASS | 4.22e-13 |  |
| u_probe | d3 | uniform+pad | () | () | () | None | PASS | 5.19e-15 |  |
| u_apply_derivatives | d3 | uniform+pad | () | () | () | None | PASS | 4.92e-16 |  |
| u_entries_derivatives | d3 | uniform+pad | () | () | () | None | PASS | 6.06e-16 |  |
| u_probe_derivatives | d3 | uniform+pad | () | () | () | None | PASS | 1.63e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| utv_apply | d3 | uniform+pad | () | () | () | None | PASS | 1.89e-16 |  |
| utv_entries | d3 | uniform+pad | () | () | () | None | PASS | 4.51e-16 |  |
| utv_probe | d3 | uniform+pad | () | () | () | None | PASS | 5.31e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | () | () | () | None | PASS | 3.12e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | () | () | () | None | PASS | 3.12e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | () | () | () | None | PASS | 3.35e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | () | None | PASS | 5.70e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | () | None | PASS | 5.70e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | () | None | PASS | 1.16e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | () | None | PASS | 1.38e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | () | None | PASS | 1.41e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | () | None | PASS | 1.16e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | () | None | PASS | 1.38e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | () | None | PASS | 1.41e-16 |  |
| utv_apply | d3 | uniform+pad | () | () | (2,) | None | PASS | 1.41e-16 |  |
| utv_entries | d3 | uniform+pad | () | () | (2,) | None | PASS | 1.50e-16 |  |
| utv_probe | d3 | uniform+pad | () | () | (2,) | None | PASS | 3.16e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | () | () | (2,) | None | PASS | 2.34e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | () | () | (2,) | None | PASS | 2.27e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | () | () | (2,) | None | PASS | 5.64e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | (2,) | None | PASS | 1.37e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | (2,) | None | PASS | 2.84e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | (2,) | None | PASS | 1.37e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | (2,) | None | PASS | 2.84e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | (2,) | None | PASS | 7.89e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | () | (2,) | None | PASS | 4.04e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | (2,) | None | PASS | 7.89e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | () | (2,) | None | PASS | 4.04e-16 |  |
| u_apply | d3 | uniform+pad | () | (3,) | () | None | PASS | 9.75e-15 |  |
| u_entries | d3 | uniform+pad | () | (3,) | () | None | PASS | 6.07e-15 |  |
| u_probe | d3 | uniform+pad | () | (3,) | () | None | PASS | 9.33e-15 |  |
| u_apply_derivatives | d3 | uniform+pad | () | (3,) | () | None | PASS | 1.64e-15 |  |
| u_entries_derivatives | d3 | uniform+pad | () | (3,) | () | None | PASS | 1.57e-15 |  |
| u_probe_derivatives | d3 | uniform+pad | () | (3,) | () | None | PASS | 3.48e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2) (3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| utv_apply | d3 | uniform+pad | () | (3,) | () | None | PASS | 3.46e-16 |  |
| utv_entries | d3 | uniform+pad | () | (3,) | () | None | PASS | 6.63e-16 |  |
| utv_probe | d3 | uniform+pad | () | (3,) | () | None | PASS | 3.50e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | () | (3,) | () | None | PASS | 4.48e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | () | (3,) | () | None | PASS | 5.89e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | () | (3,) | () | None | PASS | 3.33e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | () | None | PASS | 4.99e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | () | None | PASS | 1.48e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | () | None | PASS | 3.33e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | () | None | PASS | 8.89e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | () | None | PASS | 8.54e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | () | None | PASS | 3.42e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | () | None | PASS | 8.92e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | () | None | PASS | 4.27e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | () | None | PASS | 1.07e-14 |  |
| utv_apply | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 6.37e-16 |  |
| utv_entries | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.40e-16 |  |
| utv_probe | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 3.25e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 4.08e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 4.29e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 3.54e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 4.27e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 6.44e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.83e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 8.55e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 8.59e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 3.80e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 5.35e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.48e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.27e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 7.14e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.48e-16 |  |
| u_apply | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 2.59e-15 |  |
| u_entries | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 3.86e-15 |  |
| u_probe | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 8.76e-15 |  |
| u_apply_derivatives | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 2.19e-15 |  |
| u_entries_derivatives | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 2.41e-15 |  |
| u_probe_derivatives | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 4.84e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2) (2,2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2) (2,3,3)  |
| utv_apply | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 2.05e-16 |  |
| utv_entries | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 1.09e-16 |  |
| utv_probe | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 3.01e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 4.37e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 4.69e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 2.89e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 3.35e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 2.10e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 1.74e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 2.10e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 8.68e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 1.35e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 4.12e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 8.25e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | () | None | PASS | 1.11e-16 |  |
| utv_apply | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.83e-16 |  |
| utv_entries | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.27e-16 |  |
| utv_probe | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.25e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.33e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.25e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.91e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.99e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.17e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.26e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.26e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.75e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.40e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.45e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.20e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d3 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_norm | d3 | uniform+pad | () | () | () | None | PASS | 3.63e-16 |  |
| u_gauge_project_idempotent | d3 | uniform+pad | () | () | () | None | PASS | 5.81e-16 |  |
| u_tangent_add_scale | d3 | uniform+pad | () | () | () | None | PASS | 2.11e-16 |  |
| u_tangent_reverse | d3 | uniform+pad | () | () | () | None | PASS | 1.48e-16 |  |
| u_retract_zero | d3 | uniform+pad | () | () | () | None | PASS | 4.70e-16 |  |
| u_retract_fd_jacobian | d3 | uniform+pad | () | () | () | None | PASS | 1.93e-07 | ratio=4.00 |
| u_retract_vs_ragged | d3 | uniform+pad | () | () | () | None | PASS | 1.55e-15 |  |
| u_project_ambient | d3 | uniform+pad | () | () | () | None | PASS | 1.39e-15 |  |
| u_transport_identity | d3 | uniform+pad | () | () | () | None | PASS | 7.41e-16 |  |
| u_transport_vs_ragged_projection | d3 | uniform+pad | () | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | d3 | uniform+pad | () | () | (2,) | None | PASS | 5.56e-16 |  |
| u_manifold_norm | d3 | uniform+pad | () | () | (2,) | None | PASS | 2.48e-16 |  |
| u_gauge_project_idempotent | d3 | uniform+pad | () | () | (2,) | None | PASS | 4.66e-16 |  |
| u_tangent_add_scale | d3 | uniform+pad | () | () | (2,) | None | PASS | 1.76e-16 |  |
| u_tangent_reverse | d3 | uniform+pad | () | () | (2,) | None | PASS | 1.57e-16 |  |
| u_retract_zero | d3 | uniform+pad | () | () | (2,) | None | PASS | 4.70e-16 |  |
| u_retract_fd_jacobian | d3 | uniform+pad | () | () | (2,) | None | PASS | 3.01e-07 | ratio=4.00 |
| u_project_ambient | d3 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 9 into shape (3,2,3,1) |
| u_transport_identity | d3 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 9 into shape (3,2,3,1) |
| u_transport_vs_ragged_projection | d3 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | d3 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.06e-16 |  |
| u_sub | d3 | uniform+pad | (2,) | () | () | None | PASS | 9.30e-17 |  |
| u_scalar_mul | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.50e-16 |  |
| u_inner | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.73e-15 |  |
| u_norm | d3 | uniform+pad | (2,) | () | () | None | PASS | 6.24e-16 |  |
| u_reverse | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.25e-16 |  |
| u_t3svd_lossless | d3 | uniform+pad | (2,) | () | () | None | PASS | 5.81e-16 |  |
| u_rank_adjustment_sweep | d3 | uniform+pad | (2,) | () | () | None | PASS | 5.54e-16 |  |
| u_t3svd_trunc_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | PASS | 4.32e-16 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | d3 | uniform+pad | (2,) | () | () | None | FAIL | 6.21e-01 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | d3 | uniform+pad | (2,) | () | () | None | PASS | 6.43e-16 |  |
| u_apply | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.40e-15 |  |
| u_entries | d3 | uniform+pad | (2,) | () | () | None | PASS | 4.63e-16 |  |
| u_probe | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.03e-15 |  |
| u_apply_derivatives | d3 | uniform+pad | (2,) | () | () | None | PASS | 6.03e-16 |  |
| u_entries_derivatives | d3 | uniform+pad | (2,) | () | () | None | PASS | 5.98e-16 |  |
| u_probe_derivatives | d3 | uniform+pad | (2,) | () | () | None | PASS | 7.18e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| utv_apply | d3 | uniform+pad | (2,) | () | () | None | PASS | 2.33e-16 |  |
| utv_entries | d3 | uniform+pad | (2,) | () | () | None | PASS | 8.70e-16 |  |
| utv_probe | d3 | uniform+pad | (2,) | () | () | None | PASS | 5.30e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2,) | () | () | None | PASS | 6.49e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2,) | () | () | None | PASS | 6.32e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2,) | () | () | None | PASS | 3.35e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.06e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.06e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.41e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | () | None | PASS | 3.10e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.41e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | () | None | PASS | 3.10e-16 |  |
| utv_apply | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.91e-16 |  |
| utv_entries | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.96e-16 |  |
| utv_probe | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 4.24e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.37e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.29e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.18e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.34e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.47e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.34e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.47e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.40e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.98e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.21e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.40e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.98e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.21e-15 |  |
| u_apply | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 6.67e-16 |  |
| u_entries | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.72e-16 |  |
| u_probe | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.37e-15 |  |
| u_apply_derivatives | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 8.57e-16 |  |
| u_entries_derivatives | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 8.13e-16 |  |
| u_probe_derivatives | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 6.31e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,3,2) (3,2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| utv_apply | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.49e-16 |  |
| utv_entries | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.14e-16 |  |
| utv_probe | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 5.87e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 5.30e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.71e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.42e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.91e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.88e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.31e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.91e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.76e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.09e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.17e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.08e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.08e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 5.87e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.08e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.08e-16 |  |
| utv_apply | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 5.37e-16 |  |
| utv_entries | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.97e-16 |  |
| utv_probe | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.38e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.82e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.68e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.28e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 5.21e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.61e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 5.88e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.47e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.81e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.92e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.93e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.66e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.66e-15 |  |
| u_apply | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 7.03e-16 |  |
| u_entries | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.45e-15 |  |
| u_probe | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 8.80e-16 |  |
| u_apply_derivatives | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 5.04e-16 |  |
| u_entries_derivatives | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.31e-16 |  |
| u_probe_derivatives | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 8.93e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,3,2) (2,2,2,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,3,2) (2,2,3,3)  |
| utv_apply | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.57e-16 |  |
| utv_entries | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.90e-16 |  |
| utv_probe | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.70e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.22e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.06e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.87e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.87e-14 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.38e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.15e-14 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.11e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 6.16e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.47e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.05e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.46e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.70e-16 |  |
| utv_apply | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.98e-16 |  |
| utv_entries | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.66e-16 |  |
| utv_probe | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.02e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.50e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.43e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.94e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 8.61e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.89e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.32e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.43e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.89e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.58e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.48e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.58e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.48e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.16e-16 |  |
| u_manifold_inner | d3 | uniform+pad | (2,) | () | () | None | PASS | 2.98e-16 |  |
| u_manifold_norm | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.22e-16 |  |
| u_gauge_project_idempotent | d3 | uniform+pad | (2,) | () | () | None | PASS | 2.88e-16 |  |
| u_tangent_add_scale | d3 | uniform+pad | (2,) | () | () | None | PASS | 2.15e-16 |  |
| u_tangent_reverse | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.37e-16 |  |
| u_retract_zero | d3 | uniform+pad | (2,) | () | () | None | PASS | 8.63e-16 |  |
| u_retract_fd_jacobian | d3 | uniform+pad | (2,) | () | () | None | PASS | 1.62e-07 | ratio=4.00 |
| u_retract_vs_ragged | d3 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d3 | uniform+pad | (2,) | () | () | None | PASS | 7.15e-16 |  |
| u_transport_identity | d3 | uniform+pad | (2,) | () | () | None | PASS | 6.21e-16 |  |
| u_transport_vs_ragged_projection | d3 | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.19e-15 |  |
| u_manifold_norm | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.52e-16 |  |
| u_gauge_project_idempotent | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.70e-16 |  |
| u_tangent_add_scale | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.80e-16 |  |
| u_tangent_reverse | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.60e-16 |  |
| u_retract_zero | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 8.63e-16 |  |
| u_retract_fd_jacobian | d3 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.94e-07 | ratio=4.00 |
| u_project_ambient | d3 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1) |
| u_transport_identity | d3 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1) |
| u_transport_vs_ragged_projection | d3 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 8.19e-17 |  |
| u_sub | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 9.01e-17 |  |
| u_scalar_mul | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.03e-16 |  |
| u_inner | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 2.73e-15 |  |
| u_norm | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 7.00e-16 |  |
| u_reverse | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.46e-16 |  |
| u_t3svd_lossless | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.12e-15 |  |
| u_rank_adjustment_sweep | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 7.85e-16 |  |
| u_t3svd_trunc_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.02e-15 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | d3 | uniform+pad | (2, 3) | () | () | None | FAIL | 7.28e-01 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.02e-15 |  |
| u_apply | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 2.16e-15 |  |
| u_entries | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 7.75e-16 |  |
| u_probe | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.04e-15 |  |
| u_apply_derivatives | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 2.70e-16 |  |
| u_entries_derivatives | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.99e-16 |  |
| u_probe_derivatives | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 6.59e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| utv_apply | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 3.59e-16 |  |
| utv_entries | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.28e-16 |  |
| utv_probe | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 3.40e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 6.15e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 6.99e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 2.90e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 8.40e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 5.26e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 8.40e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 5.26e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.85e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 2.34e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.64e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.85e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 2.34e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.64e-16 |  |
| utv_apply | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.14e-16 |  |
| utv_entries | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.56e-16 |  |
| utv_probe | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.63e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.27e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.39e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.48e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.06e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 6.67e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.06e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 6.67e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.17e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.45e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.17e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.45e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.28e-15 |  |
| u_entries | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 8.42e-16 |  |
| u_probe | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.11e-15 |  |
| u_apply_derivatives | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 9.68e-16 |  |
| u_entries_derivatives | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.05e-15 |  |
| u_probe_derivatives | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 9.30e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,3,2) (3,2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| utv_apply | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 6.27e-16 |  |
| utv_entries | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.58e-16 |  |
| utv_probe | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.39e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.58e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.72e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.99e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.16e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.12e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.45e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 5.78e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.12e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 5.54e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.73e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 9.46e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.39e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.73e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.15e-16 |  |
| utv_apply | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 4.58e-16 |  |
| utv_entries | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.15e-16 |  |
| utv_probe | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.40e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 4.28e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 4.44e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.16e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.64e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.22e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.61e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.64e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.22e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.61e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.75e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 4.19e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.33e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 9.02e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 7.55e-15 |  |
| u_apply | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 5.65e-16 |  |
| u_entries | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.16e-15 |  |
| u_probe | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 9.01e-16 |  |
| u_apply_derivatives | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 5.84e-16 |  |
| u_entries_derivatives | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 5.85e-16 |  |
| u_probe_derivatives | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 6.16e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,3,2) (2,2,2,3,2,3,3)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,3,2) (2,3,2,3,3)  |
| utv_apply | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.85e-16 |  |
| utv_entries | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.75e-16 |  |
| utv_probe | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.13e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.93e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 4.22e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.30e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.54e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 6.78e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.40e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.13e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 5.15e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 4.40e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.13e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 5.87e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.13e-16 |  |
| utv_apply | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.75e-16 |  |
| utv_entries | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.54e-16 |  |
| utv_probe | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.95e-16 |  |
| utv_apply_derivatives | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.63e-16 |  |
| utv_entries_derivatives | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.43e-16 |  |
| utv_probe_derivatives | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.33e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.92e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.99e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.46e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.99e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.37e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 9.07e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.26e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.24e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.27e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.53e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d3 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.48e-16 |  |
| u_manifold_inner | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 2.27e-15 |  |
| u_manifold_norm | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 3.20e-16 |  |
| u_gauge_project_idempotent | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 5.61e-16 |  |
| u_tangent_add_scale | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 2.27e-16 |  |
| u_tangent_reverse | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.65e-16 |  |
| u_retract_zero | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 6.86e-16 |  |
| u_retract_fd_jacobian | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 1.96e-04 | ratio=3.96 |
| u_retract_vs_ragged | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 7.35e-16 |  |
| u_transport_identity | d3 | uniform+pad | (2, 3) | () | () | None | PASS | 7.30e-16 |  |
| u_transport_vs_ragged_projection | d3 | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.22e-15 |  |
| u_manifold_norm | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.81e-16 |  |
| u_gauge_project_idempotent | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.26e-16 |  |
| u_tangent_add_scale | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.96e-16 |  |
| u_tangent_reverse | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.51e-16 |  |
| u_retract_zero | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 6.86e-16 |  |
| u_retract_fd_jacobian | d3 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 6.52e-05 | ratio=3.98 |
| u_project_ambient | d3 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 54 into shape (3,2,2,3,3,1) |
| u_transport_identity | d3 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 54 into shape (3,2,2,3,3,1) |
| u_transport_vs_ragged_projection | d3 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | d4 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_add | d4 | uniform | () | () | () | None | PASS | 9.21e-17 |  |
| u_sub | d4 | uniform | () | () | () | None | PASS | 9.17e-17 |  |
| u_scalar_mul | d4 | uniform | () | () | () | None | PASS | 1.93e-16 |  |
| u_inner | d4 | uniform | () | () | () | None | PASS | 1.63e-14 |  |
| u_norm | d4 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_reverse | d4 | uniform | () | () | () | None | PASS | 1.51e-16 |  |
| u_t3svd_lossless | d4 | uniform | () | () | () | None | PASS | 1.39e-15 |  |
| u_rank_adjustment_sweep | d4 | uniform | () | () | () | None | PASS | 1.51e-15 |  |
| u_t3svd_trunc_vs_ragged | d4 | uniform | () | () | () | None | PASS | 1.94e-15 |  |
| u_orthogonal_representations | d4 | uniform | () | () | () | None | PASS | 9.79e-16 |  |
| u_apply | d4 | uniform | () | () | () | None | PASS | 6.96e-16 |  |
| u_entries | d4 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe | d4 | uniform | () | () | () | None | PASS | 6.68e-16 |  |
| u_apply_derivatives | d4 | uniform | () | () | () | None | PASS | 6.76e-16 |  |
| u_entries_derivatives | d4 | uniform | () | () | () | None | PASS | 1.28e-16 |  |
| u_probe_derivatives | d4 | uniform | () | () | () | None | PASS | 8.10e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 8.12e-18 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 2.14e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 8.12e-18 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 2.14e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 1.17e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 2.16e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 2.69e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 1.17e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 2.16e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | () | () | () | None | PASS | 2.69e-16 |  |
| utv_apply | d4 | uniform | () | () | () | None | PASS | 2.33e-15 |  |
| utv_entries | d4 | uniform | () | () | () | None | PASS | 2.10e-16 |  |
| utv_probe | d4 | uniform | () | () | () | None | PASS | 6.29e-16 |  |
| utv_apply_derivatives | d4 | uniform | () | () | () | None | PASS | 4.27e-16 |  |
| utv_entries_derivatives | d4 | uniform | () | () | () | None | PASS | 3.63e-18 |  |
| utv_probe_derivatives | d4 | uniform | () | () | () | None | PASS | 4.65e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | () | () | () | None | PASS | 1.56e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | () | () | () | None | PASS | 3.80e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | () | () | () | None | PASS | 6.77e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | () | () | () | None | PASS | 1.56e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | () | () | () | None | PASS | 3.80e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | () | () | () | None | PASS | 6.77e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | () | () | None | PASS | 2.37e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | () | () | None | PASS | 9.16e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | () | () | None | PASS | 3.39e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | () | () | None | PASS | 2.37e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | () | () | None | PASS | 9.16e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | () | () | None | PASS | 3.39e-16 |  |
| utv_apply | d4 | uniform | () | () | (2,) | None | PASS | 7.84e-16 |  |
| utv_entries | d4 | uniform | () | () | (2,) | None | PASS | 3.25e-16 |  |
| utv_probe | d4 | uniform | () | () | (2,) | None | PASS | 1.26e-15 |  |
| utv_apply_derivatives | d4 | uniform | () | () | (2,) | None | PASS | 8.74e-16 |  |
| utv_entries_derivatives | d4 | uniform | () | () | (2,) | None | PASS | 1.19e-15 |  |
| utv_probe_derivatives | d4 | uniform | () | () | (2,) | None | PASS | 9.09e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | () | () | (2,) | None | PASS | 8.13e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | () | () | (2,) | None | PASS | 2.15e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | () | () | (2,) | None | PASS | 2.48e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | () | () | (2,) | None | PASS | 8.13e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | () | () | (2,) | None | PASS | 2.15e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | () | () | (2,) | None | PASS | 2.48e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | () | (2,) | None | PASS | 3.09e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | () | (2,) | None | PASS | 3.02e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | () | (2,) | None | PASS | 3.09e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | () | (2,) | None | PASS | 3.02e-16 |  |
| u_apply | d4 | uniform | () | (3,) | () | None | PASS | 1.48e-15 |  |
| u_entries | d4 | uniform | () | (3,) | () | None | PASS | 1.50e-16 |  |
| u_probe | d4 | uniform | () | (3,) | () | None | PASS | 1.60e-15 |  |
| u_apply_derivatives | d4 | uniform | () | (3,) | () | None | PASS | 3.08e-16 |  |
| u_entries_derivatives | d4 | uniform | () | (3,) | () | None | PASS | 4.55e-16 |  |
| u_probe_derivatives | d4 | uniform | () | (3,) | () | None | PASS | 5.99e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 3.14e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 9.49e-17 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 1.91e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 3.13e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 2.14e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 2.84e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 1.19e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 1.60e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 2.00e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 2.13e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 2.41e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | () | (3,) | () | None | PASS | 2.82e-16 |  |
| utv_apply | d4 | uniform | () | (3,) | () | None | PASS | 6.67e-16 |  |
| utv_entries | d4 | uniform | () | (3,) | () | None | PASS | 5.86e-16 |  |
| utv_probe | d4 | uniform | () | (3,) | () | None | PASS | 7.02e-16 |  |
| utv_apply_derivatives | d4 | uniform | () | (3,) | () | None | PASS | 2.31e-16 |  |
| utv_entries_derivatives | d4 | uniform | () | (3,) | () | None | PASS | 4.66e-16 |  |
| utv_probe_derivatives | d4 | uniform | () | (3,) | () | None | PASS | 4.96e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | () | None | PASS | 1.47e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | () | None | PASS | 2.59e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | () | None | PASS | 6.97e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | () | None | PASS | 2.59e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | () | None | PASS | 6.97e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | () | None | PASS | 1.96e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | () | None | PASS | 4.09e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | () | None | PASS | 3.87e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | () | None | PASS | 3.93e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | () | None | PASS | 3.41e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply | d4 | uniform | () | (3,) | (2,) | None | PASS | 9.74e-16 |  |
| utv_entries | d4 | uniform | () | (3,) | (2,) | None | PASS | 1.35e-16 |  |
| utv_probe | d4 | uniform | () | (3,) | (2,) | None | PASS | 8.61e-16 |  |
| utv_apply_derivatives | d4 | uniform | () | (3,) | (2,) | None | PASS | 3.86e-16 |  |
| utv_entries_derivatives | d4 | uniform | () | (3,) | (2,) | None | PASS | 6.72e-16 |  |
| utv_probe_derivatives | d4 | uniform | () | (3,) | (2,) | None | PASS | 7.09e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | (2,) | None | PASS | 5.08e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | (2,) | None | PASS | 2.50e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | (2,) | None | PASS | 4.11e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | (2,) | None | PASS | 3.39e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | (2,) | None | PASS | 4.99e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | (2,) | None | PASS | 8.72e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (3,) | (2,) | None | PASS | 2.64e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | (2,) | None | PASS | 4.39e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (3,) | (2,) | None | PASS | 8.81e-16 |  |
| u_apply | d4 | uniform | () | (2, 2) | () | None | PASS | 2.49e-16 |  |
| u_entries | d4 | uniform | () | (2, 2) | () | None | PASS | 1.99e-16 |  |
| u_probe | d4 | uniform | () | (2, 2) | () | None | PASS | 7.37e-16 |  |
| u_apply_derivatives | d4 | uniform | () | (2, 2) | () | None | PASS | 6.61e-16 |  |
| u_entries_derivatives | d4 | uniform | () | (2, 2) | () | None | PASS | 2.07e-16 |  |
| u_probe_derivatives | d4 | uniform | () | (2, 2) | () | None | PASS | 3.95e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 4.79e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 1.50e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 2.13e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 5.22e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 1.50e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 2.46e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 1.09e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 1.94e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 2.05e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 2.08e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 1.87e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | () | (2, 2) | () | None | PASS | 3.39e-16 |  |
| utv_apply | d4 | uniform | () | (2, 2) | () | None | PASS | 1.31e-15 |  |
| utv_entries | d4 | uniform | () | (2, 2) | () | None | PASS | 1.56e-16 |  |
| utv_probe | d4 | uniform | () | (2, 2) | () | None | PASS | 7.24e-16 |  |
| utv_apply_derivatives | d4 | uniform | () | (2, 2) | () | None | PASS | 6.35e-16 |  |
| utv_entries_derivatives | d4 | uniform | () | (2, 2) | () | None | PASS | 1.30e-16 |  |
| utv_probe_derivatives | d4 | uniform | () | (2, 2) | () | None | PASS | 5.23e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | () | None | PASS | 7.87e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | () | None | PASS | 3.01e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | () | None | PASS | 8.14e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | () | None | PASS | 3.37e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | () | None | PASS | 3.01e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | () | None | PASS | 1.09e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | () | None | PASS | 3.14e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | () | None | PASS | 6.91e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | () | None | PASS | 5.50e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | () | None | PASS | 1.57e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | () | None | PASS | 6.91e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | () | None | PASS | 1.38e-16 |  |
| utv_apply | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 4.13e-16 |  |
| utv_entries | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 1.44e-16 |  |
| utv_probe | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 7.31e-16 |  |
| utv_apply_derivatives | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 2.20e-15 |  |
| utv_entries_derivatives | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 7.52e-16 |  |
| utv_probe_derivatives | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 6.48e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 4.22e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 1.31e-14 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 4.75e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 4.22e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 3.93e-14 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 1.07e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 9.81e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 3.47e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 5.63e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 6.13e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 2.31e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | () | (2, 2) | (2,) | None | PASS | 3.76e-16 |  |
| u_manifold_inner | d4 | uniform | () | () | () | None | PASS | 1.67e-15 |  |
| u_manifold_norm | d4 | uniform | () | () | () | None | PASS | 2.49e-16 |  |
| u_gauge_project_idempotent | d4 | uniform | () | () | () | None | PASS | 4.45e-16 |  |
| u_tangent_add_scale | d4 | uniform | () | () | () | None | PASS | 2.58e-16 |  |
| u_tangent_reverse | d4 | uniform | () | () | () | None | PASS | 1.92e-16 |  |
| u_retract_zero | d4 | uniform | () | () | () | None | PASS | 1.07e-15 |  |
| u_retract_fd_jacobian | d4 | uniform | () | () | () | None | PASS | 5.07e-09 |  |
| u_retract_vs_ragged | d4 | uniform | () | () | () | None | PASS | 1.72e-15 |  |
| u_project_ambient | d4 | uniform | () | () | () | None | PASS | 5.78e-16 |  |
| u_transport_identity | d4 | uniform | () | () | () | None | PASS | 1.01e-15 |  |
| u_transport_vs_ragged_projection | d4 | uniform | () | () | () | None | PASS | 5.53e-16 |  |
| u_manifold_inner | d4 | uniform | () | () | (2,) | None | PASS | 2.08e-15 |  |
| u_manifold_norm | d4 | uniform | () | () | (2,) | None | PASS | 1.29e-16 |  |
| u_gauge_project_idempotent | d4 | uniform | () | () | (2,) | None | PASS | 5.29e-16 |  |
| u_tangent_add_scale | d4 | uniform | () | () | (2,) | None | PASS | 2.32e-16 |  |
| u_tangent_reverse | d4 | uniform | () | () | (2,) | None | PASS | 2.06e-16 |  |
| u_retract_zero | d4 | uniform | () | () | (2,) | None | PASS | 1.07e-15 |  |
| u_retract_fd_jacobian | d4 | uniform | () | () | (2,) | None | PASS | 1.23e-08 | ratio=4.00 |
| u_project_ambient | d4 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 16 into shape (4,2,4,1) |
| u_transport_identity | d4 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 16 into shape (4,2,4,1) |
| u_transport_vs_ragged_projection | d4 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 16 into shape (4,2,4,1) |
| u_to_dense | d4 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d4 | uniform | (2,) | () | () | None | PASS | 8.05e-17 |  |
| u_sub | d4 | uniform | (2,) | () | () | None | PASS | 8.21e-17 |  |
| u_scalar_mul | d4 | uniform | (2,) | () | () | None | PASS | 1.46e-16 |  |
| u_inner | d4 | uniform | (2,) | () | () | None | PASS | 1.85e-15 |  |
| u_norm | d4 | uniform | (2,) | () | () | None | PASS | 8.36e-16 |  |
| u_reverse | d4 | uniform | (2,) | () | () | None | PASS | 1.55e-16 |  |
| u_t3svd_lossless | d4 | uniform | (2,) | () | () | None | PASS | 1.00e-15 |  |
| u_rank_adjustment_sweep | d4 | uniform | (2,) | () | () | None | PASS | 1.04e-15 |  |
| u_t3svd_trunc_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 1.29e-15 |  |
| u_orthogonal_representations | d4 | uniform | (2,) | () | () | None | PASS | 1.33e-15 |  |
| u_apply | d4 | uniform | (2,) | () | () | None | PASS | 4.96e-16 |  |
| u_entries | d4 | uniform | (2,) | () | () | None | PASS | 1.36e-16 |  |
| u_probe | d4 | uniform | (2,) | () | () | None | PASS | 1.11e-15 |  |
| u_apply_derivatives | d4 | uniform | (2,) | () | () | None | PASS | 2.49e-15 |  |
| u_entries_derivatives | d4 | uniform | (2,) | () | () | None | PASS | 1.40e-15 |  |
| u_probe_derivatives | d4 | uniform | (2,) | () | () | None | PASS | 8.99e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 1.99e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 9.21e-17 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 1.99e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 9.21e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 1.60e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 1.32e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 2.44e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 1.60e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 1.32e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 2.44e-16 |  |
| utv_apply | d4 | uniform | (2,) | () | () | None | PASS | 1.08e-15 |  |
| utv_entries | d4 | uniform | (2,) | () | () | None | PASS | 1.98e-16 |  |
| utv_probe | d4 | uniform | (2,) | () | () | None | PASS | 5.24e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2,) | () | () | None | PASS | 6.80e-16 |  |
| utv_entries_derivatives | d4 | uniform | (2,) | () | () | None | PASS | 1.05e-15 |  |
| utv_probe_derivatives | d4 | uniform | (2,) | () | () | None | PASS | 4.98e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | () | None | PASS | 1.32e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | () | None | PASS | 1.97e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | () | None | PASS | 9.08e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | () | None | PASS | 1.32e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | () | None | PASS | 1.97e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | () | None | PASS | 9.08e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | () | None | PASS | 3.36e-14 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | () | None | PASS | 1.28e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | () | None | PASS | 3.36e-14 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | () | None | PASS | 1.28e-16 |  |
| utv_apply | d4 | uniform | (2,) | () | (2,) | None | PASS | 7.58e-16 |  |
| utv_entries | d4 | uniform | (2,) | () | (2,) | None | PASS | 4.12e-16 |  |
| utv_probe | d4 | uniform | (2,) | () | (2,) | None | PASS | 3.69e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2,) | () | (2,) | None | PASS | 7.01e-16 |  |
| utv_entries_derivatives | d4 | uniform | (2,) | () | (2,) | None | PASS | 8.44e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2,) | () | (2,) | None | PASS | 7.06e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | (2,) | None | PASS | 2.07e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | (2,) | None | PASS | 3.76e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | (2,) | None | PASS | 1.95e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | (2,) | None | PASS | 2.07e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | (2,) | None | PASS | 3.76e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | (2,) | None | PASS | 1.95e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | (2,) | None | PASS | 5.73e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | () | (2,) | None | PASS | 2.05e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | (2,) | None | PASS | 5.73e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | () | (2,) | None | PASS | 2.05e-16 |  |
| u_apply | d4 | uniform | (2,) | (3,) | () | None | PASS | 7.33e-16 |  |
| u_entries | d4 | uniform | (2,) | (3,) | () | None | PASS | 5.56e-16 |  |
| u_probe | d4 | uniform | (2,) | (3,) | () | None | PASS | 5.19e-16 |  |
| u_apply_derivatives | d4 | uniform | (2,) | (3,) | () | None | PASS | 8.64e-16 |  |
| u_entries_derivatives | d4 | uniform | (2,) | (3,) | () | None | PASS | 4.40e-16 |  |
| u_probe_derivatives | d4 | uniform | (2,) | (3,) | () | None | PASS | 5.41e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 2.10e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 5.63e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 2.78e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 2.64e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 5.71e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 2.46e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 1.31e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 1.57e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 2.08e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 1.28e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 1.59e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (3,) | () | None | PASS | 2.47e-16 |  |
| utv_apply | d4 | uniform | (2,) | (3,) | () | None | PASS | 4.69e-16 |  |
| utv_entries | d4 | uniform | (2,) | (3,) | () | None | PASS | 1.74e-16 |  |
| utv_probe | d4 | uniform | (2,) | (3,) | () | None | PASS | 7.09e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2,) | (3,) | () | None | PASS | 3.96e-16 |  |
| utv_entries_derivatives | d4 | uniform | (2,) | (3,) | () | None | PASS | 3.27e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2,) | (3,) | () | None | PASS | 5.47e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | () | None | PASS | 5.69e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | () | None | PASS | 2.57e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | () | None | PASS | 4.27e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | () | None | PASS | 1.53e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | () | None | PASS | 1.29e-14 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | () | None | PASS | 2.18e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | () | None | PASS | 4.22e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | () | None | PASS | 2.18e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | () | None | PASS | 1.96e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | () | None | PASS | 2.81e-16 |  |
| utv_apply | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 7.29e-16 |  |
| utv_entries | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 4.42e-16 |  |
| utv_probe | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 7.03e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.06e-15 |  |
| utv_entries_derivatives | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 3.67e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 5.87e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 5.68e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.32e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 5.29e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 5.68e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 3.95e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 6.62e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.63e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.35e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.63e-16 |  |
| u_apply | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 1.92e-16 |  |
| u_entries | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 1.08e-15 |  |
| u_probe | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 4.05e-16 |  |
| u_apply_derivatives | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 6.33e-16 |  |
| u_entries_derivatives | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 8.14e-16 |  |
| u_probe_derivatives | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 5.46e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 2.47e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 2.73e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 2.48e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 2.26e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 2.76e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 3.18e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 1.46e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 1.30e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 1.48e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 2.52e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 1.47e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 2.68e-16 |  |
| utv_apply | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 6.33e-16 |  |
| utv_entries | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 2.18e-16 |  |
| utv_probe | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 6.28e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 6.72e-16 |  |
| utv_entries_derivatives | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 5.13e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 4.70e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 1.34e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 4.58e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 5.37e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 3.05e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 6.78e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 1.96e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 1.04e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 9.49e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 1.96e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | () | None | PASS | 8.12e-16 |  |
| utv_apply | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 7.13e-16 |  |
| utv_entries | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.39e-16 |  |
| utv_probe | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 5.16e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.49e-16 |  |
| utv_entries_derivatives | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.87e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.95e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.86e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 5.40e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.72e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.42e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.80e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.04e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.17e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 9.96e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.49e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.17e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 9.96e-16 |  |
| u_manifold_inner | d4 | uniform | (2,) | () | () | None | PASS | 4.23e-16 |  |
| u_manifold_norm | d4 | uniform | (2,) | () | () | None | PASS | 3.79e-16 |  |
| u_gauge_project_idempotent | d4 | uniform | (2,) | () | () | None | PASS | 7.36e-16 |  |
| u_tangent_add_scale | d4 | uniform | (2,) | () | () | None | PASS | 2.21e-16 |  |
| u_tangent_reverse | d4 | uniform | (2,) | () | () | None | PASS | 1.84e-16 |  |
| u_retract_zero | d4 | uniform | (2,) | () | () | None | PASS | 1.26e-15 |  |
| u_retract_fd_jacobian | d4 | uniform | (2,) | () | () | None | PASS | 3.26e-09 |  |
| u_retract_vs_ragged | d4 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d4 | uniform | (2,) | () | () | None | PASS | 7.25e-16 |  |
| u_transport_identity | d4 | uniform | (2,) | () | () | None | PASS | 1.17e-15 |  |
| u_transport_vs_ragged_projection | d4 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d4 | uniform | (2,) | () | (2,) | None | PASS | 1.89e-15 |  |
| u_manifold_norm | d4 | uniform | (2,) | () | (2,) | None | PASS | 4.07e-16 |  |
| u_gauge_project_idempotent | d4 | uniform | (2,) | () | (2,) | None | PASS | 4.91e-16 |  |
| u_tangent_add_scale | d4 | uniform | (2,) | () | (2,) | None | PASS | 2.21e-16 |  |
| u_tangent_reverse | d4 | uniform | (2,) | () | (2,) | None | PASS | 1.90e-16 |  |
| u_retract_zero | d4 | uniform | (2,) | () | (2,) | None | PASS | 1.26e-15 |  |
| u_retract_fd_jacobian | d4 | uniform | (2,) | () | (2,) | None | PASS | 4.22e-09 |  |
| u_project_ambient | d4 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 32 into shape (4,2,2,4,1) |
| u_transport_identity | d4 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 32 into shape (4,2,2,4,1) |
| u_transport_vs_ragged_projection | d4 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 32 into shape (4,2,2,4,1) |
| u_to_dense | d4 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d4 | uniform | (2, 3) | () | () | None | PASS | 1.06e-16 |  |
| u_sub | d4 | uniform | (2, 3) | () | () | None | PASS | 9.59e-17 |  |
| u_scalar_mul | d4 | uniform | (2, 3) | () | () | None | PASS | 1.51e-16 |  |
| u_inner | d4 | uniform | (2, 3) | () | () | None | PASS | 2.38e-15 |  |
| u_norm | d4 | uniform | (2, 3) | () | () | None | PASS | 4.67e-16 |  |
| u_reverse | d4 | uniform | (2, 3) | () | () | None | PASS | 1.46e-16 |  |
| u_t3svd_lossless | d4 | uniform | (2, 3) | () | () | None | PASS | 1.68e-15 |  |
| u_rank_adjustment_sweep | d4 | uniform | (2, 3) | () | () | None | PASS | 1.89e-15 |  |
| u_t3svd_trunc_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 1.37e-15 |  |
| u_orthogonal_representations | d4 | uniform | (2, 3) | () | () | None | PASS | 1.41e-15 |  |
| u_apply | d4 | uniform | (2, 3) | () | () | None | PASS | 2.68e-16 |  |
| u_entries | d4 | uniform | (2, 3) | () | () | None | PASS | 2.05e-16 |  |
| u_probe | d4 | uniform | (2, 3) | () | () | None | PASS | 3.90e-16 |  |
| u_apply_derivatives | d4 | uniform | (2, 3) | () | () | None | PASS | 9.21e-16 |  |
| u_entries_derivatives | d4 | uniform | (2, 3) | () | () | None | PASS | 8.05e-16 |  |
| u_probe_derivatives | d4 | uniform | (2, 3) | () | () | None | PASS | 8.15e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 3.63e-17 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 2.57e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 3.00e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 3.63e-17 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 2.57e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 3.00e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 1.22e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 1.65e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 2.35e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 1.22e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 1.65e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 2.35e-16 |  |
| utv_apply | d4 | uniform | (2, 3) | () | () | None | PASS | 2.07e-16 |  |
| utv_entries | d4 | uniform | (2, 3) | () | () | None | PASS | 1.62e-16 |  |
| utv_probe | d4 | uniform | (2, 3) | () | () | None | PASS | 5.11e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2, 3) | () | () | None | PASS | 4.28e-16 |  |
| utv_entries_derivatives | d4 | uniform | (2, 3) | () | () | None | PASS | 2.77e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2, 3) | () | () | None | PASS | 4.07e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | () | None | PASS | 4.52e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | () | None | PASS | 2.22e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | () | None | PASS | 3.05e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | () | None | PASS | 4.52e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | () | None | PASS | 2.22e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | () | None | PASS | 3.05e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | () | None | PASS | 1.15e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | () | None | PASS | 1.93e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | () | None | PASS | 6.74e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | () | None | PASS | 1.15e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | () | None | PASS | 1.93e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | () | None | PASS | 6.74e-16 |  |
| utv_apply | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 9.48e-16 |  |
| utv_entries | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 2.27e-16 |  |
| utv_probe | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 4.42e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 7.29e-16 |  |
| utv_entries_derivatives | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 3.15e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 4.73e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 3.83e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 7.07e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 6.86e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 3.83e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 7.07e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 6.86e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 2.19e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 9.16e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 7.91e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 2.19e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 9.16e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 7.91e-16 |  |
| u_apply | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 6.63e-16 |  |
| u_entries | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 1.91e-16 |  |
| u_probe | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 5.89e-16 |  |
| u_apply_derivatives | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 3.49e-16 |  |
| u_entries_derivatives | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 3.36e-16 |  |
| u_probe_derivatives | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 5.16e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 2.68e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 2.29e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 2.50e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 2.31e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 3.01e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 2.19e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 1.24e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 1.67e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 2.96e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 1.73e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 1.84e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 3.30e-16 |  |
| utv_apply | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 4.35e-16 |  |
| utv_entries | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 2.28e-16 |  |
| utv_probe | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 5.10e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 1.13e-15 |  |
| utv_entries_derivatives | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 4.27e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 5.34e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 4.47e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 3.58e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 1.66e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 4.47e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 3.58e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 4.54e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 2.96e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 1.13e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | () | None | PASS | 2.27e-16 |  |
| utv_apply | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 6.30e-16 |  |
| utv_entries | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.47e-16 |  |
| utv_probe | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 5.83e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 6.75e-16 |  |
| utv_entries_derivatives | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 5.30e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 6.04e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 9.46e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.28e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 8.16e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.35e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.28e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.85e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 5.49e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.60e-15 |  |
| u_entries | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.24e-16 |  |
| u_probe | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 5.67e-16 |  |
| u_apply_derivatives | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 9.02e-16 |  |
| u_entries_derivatives | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.33e-16 |  |
| u_probe_derivatives | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 5.32e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.41e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.48e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.54e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.52e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.48e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.64e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.89e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.44e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.83e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.32e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.88e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.85e-16 |  |
| utv_apply | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 6.81e-16 |  |
| utv_entries | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.53e-16 |  |
| utv_probe | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.30e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 7.25e-16 |  |
| utv_entries_derivatives | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 5.67e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 5.88e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 5.89e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.19e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 6.65e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 8.83e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.19e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.70e-14 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.49e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.50e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 7.27e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.49e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.00e-16 |  |
| utv_apply | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 9.95e-16 |  |
| utv_entries | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.47e-16 |  |
| utv_probe | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 7.05e-16 |  |
| utv_apply_derivatives | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 8.67e-16 |  |
| utv_entries_derivatives | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 4.30e-16 |  |
| utv_probe_derivatives | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 5.51e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.70e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 6.67e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 8.10e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.67e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.78e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.32e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.71e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 7.16e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.29e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d4 | uniform | (2, 3) | () | () | None | PASS | 2.02e-15 |  |
| u_manifold_norm | d4 | uniform | (2, 3) | () | () | None | PASS | 5.25e-16 |  |
| u_gauge_project_idempotent | d4 | uniform | (2, 3) | () | () | None | PASS | 6.32e-16 |  |
| u_tangent_add_scale | d4 | uniform | (2, 3) | () | () | None | PASS | 2.47e-16 |  |
| u_tangent_reverse | d4 | uniform | (2, 3) | () | () | None | PASS | 1.87e-16 |  |
| u_retract_zero | d4 | uniform | (2, 3) | () | () | None | PASS | 1.40e-15 |  |
| u_retract_fd_jacobian | d4 | uniform | (2, 3) | () | () | None | PASS | 4.58e-08 | ratio=4.00 |
| u_retract_vs_ragged | d4 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d4 | uniform | (2, 3) | () | () | None | PASS | 4.86e-16 |  |
| u_transport_identity | d4 | uniform | (2, 3) | () | () | None | PASS | 1.14e-15 |  |
| u_transport_vs_ragged_projection | d4 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 1.39e-15 |  |
| u_manifold_norm | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 4.37e-16 |  |
| u_gauge_project_idempotent | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 5.74e-16 |  |
| u_tangent_add_scale | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 2.44e-16 |  |
| u_tangent_reverse | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 2.03e-16 |  |
| u_retract_zero | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 1.40e-15 |  |
| u_retract_fd_jacobian | d4 | uniform | (2, 3) | () | (2,) | None | PASS | 4.53e-08 | ratio=4.00 |
| u_project_ambient | d4 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 96 into shape (4,2,2,3,4,1) |
| u_transport_identity | d4 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 96 into shape (4,2,2,3,4,1) |
| u_transport_vs_ragged_projection | d4 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 96 into shape (4,2,2,3,4,1) |
| u_to_dense | d4 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_add | d4 | uniform+pad | () | () | () | None | PASS | 9.21e-17 |  |
| u_sub | d4 | uniform+pad | () | () | () | None | PASS | 9.17e-17 |  |
| u_scalar_mul | d4 | uniform+pad | () | () | () | None | PASS | 1.93e-16 |  |
| u_inner | d4 | uniform+pad | () | () | () | None | PASS | 1.44e-14 |  |
| u_norm | d4 | uniform+pad | () | () | () | None | PASS | 1.69e-16 |  |
| u_reverse | d4 | uniform+pad | () | () | () | None | PASS | 1.51e-16 |  |
| u_t3svd_lossless | d4 | uniform+pad | () | () | () | None | PASS | 1.39e-15 |  |
| u_rank_adjustment_sweep | d4 | uniform+pad | () | () | () | None | PASS | 1.80e-15 |  |
| u_t3svd_trunc_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 1.94e-15 |  |
| u_orthogonal_representations | d4 | uniform+pad | () | () | () | None | PASS | 8.69e-16 |  |
| u_apply | d4 | uniform+pad | () | () | () | None | PASS | 4.18e-16 |  |
| u_entries | d4 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe | d4 | uniform+pad | () | () | () | None | PASS | 6.75e-16 |  |
| u_apply_derivatives | d4 | uniform+pad | () | () | () | None | PASS | 8.23e-16 |  |
| u_entries_derivatives | d4 | uniform+pad | () | () | () | None | PASS | 6.36e-16 |  |
| u_probe_derivatives | d4 | uniform+pad | () | () | () | None | PASS | 6.47e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 3.39e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 2.39e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 4.60e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 3.39e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 2.39e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 4.60e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 2.39e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 2.16e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 2.45e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 2.39e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 2.16e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 2.45e-16 |  |
| utv_apply | d4 | uniform+pad | () | () | () | None | PASS | 1.55e-16 |  |
| utv_entries | d4 | uniform+pad | () | () | () | None | PASS | 1.89e-16 |  |
| utv_probe | d4 | uniform+pad | () | () | () | None | PASS | 6.70e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | () | () | () | None | PASS | 1.44e-15 |  |
| utv_entries_derivatives | d4 | uniform+pad | () | () | () | None | PASS | 3.33e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | () | () | () | None | PASS | 6.41e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | () | None | PASS | 5.59e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | () | None | PASS | 3.42e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | () | None | PASS | 3.60e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | () | None | PASS | 5.59e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | () | None | PASS | 3.42e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | () | None | PASS | 3.60e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | () | None | PASS | 1.18e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | () | None | PASS | 3.41e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | () | None | PASS | 1.96e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | () | None | PASS | 1.18e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | () | None | PASS | 3.41e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | () | None | PASS | 1.96e-16 |  |
| utv_apply | d4 | uniform+pad | () | () | (2,) | None | PASS | 9.51e-16 |  |
| utv_entries | d4 | uniform+pad | () | () | (2,) | None | PASS | 2.58e-16 |  |
| utv_probe | d4 | uniform+pad | () | () | (2,) | None | PASS | 4.60e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | () | () | (2,) | None | PASS | 6.69e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | () | () | (2,) | None | PASS | 2.53e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | () | () | (2,) | None | PASS | 5.78e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | (2,) | None | PASS | 1.18e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | (2,) | None | PASS | 2.29e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | (2,) | None | PASS | 1.90e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | (2,) | None | PASS | 1.18e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | (2,) | None | PASS | 2.29e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | (2,) | None | PASS | 1.90e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | (2,) | None | PASS | 1.89e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | (2,) | None | PASS | 3.11e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | () | (2,) | None | PASS | 1.41e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | (2,) | None | PASS | 1.89e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | (2,) | None | PASS | 3.11e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | () | (2,) | None | PASS | 1.41e-16 |  |
| u_apply | d4 | uniform+pad | () | (3,) | () | None | PASS | 1.89e-15 |  |
| u_entries | d4 | uniform+pad | () | (3,) | () | None | PASS | 1.50e-16 |  |
| u_probe | d4 | uniform+pad | () | (3,) | () | None | PASS | 1.92e-15 |  |
| u_apply_derivatives | d4 | uniform+pad | () | (3,) | () | None | PASS | 4.39e-16 |  |
| u_entries_derivatives | d4 | uniform+pad | () | (3,) | () | None | PASS | 6.15e-16 |  |
| u_probe_derivatives | d4 | uniform+pad | () | (3,) | () | None | PASS | 7.24e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 8.09e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 1.73e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 6.96e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 8.54e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 1.73e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 7.41e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 4.12e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 2.70e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 3.54e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 4.83e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 3.61e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (3,) | () | None | PASS | 5.22e-16 |  |
| utv_apply | d4 | uniform+pad | () | (3,) | () | None | PASS | 2.44e-16 |  |
| utv_entries | d4 | uniform+pad | () | (3,) | () | None | PASS | 5.54e-16 |  |
| utv_probe | d4 | uniform+pad | () | (3,) | () | None | PASS | 4.03e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | () | (3,) | () | None | PASS | 5.40e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | () | (3,) | () | None | PASS | 4.12e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | () | (3,) | () | None | PASS | 5.03e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | () | None | PASS | 4.03e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | () | None | PASS | 2.21e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | () | None | PASS | 4.03e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | () | None | PASS | 3.57e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | () | None | PASS | 4.48e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | () | None | PASS | 6.36e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | () | None | PASS | 3.57e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | () | None | PASS | 2.12e-16 |  |
| utv_apply | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 9.29e-16 |  |
| utv_entries | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 5.25e-16 |  |
| utv_probe | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 6.95e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 6.02e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 4.84e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 4.18e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 5.08e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.59e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 6.77e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.59e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.25e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.69e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 3.22e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 3.39e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 2.44e-16 |  |
| u_entries | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 1.99e-16 |  |
| u_probe | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 8.79e-16 |  |
| u_apply_derivatives | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 6.61e-16 |  |
| u_entries_derivatives | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 1.85e-16 |  |
| u_probe_derivatives | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 5.33e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 4.86e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 2.05e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 3.93e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 4.99e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 2.05e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 4.19e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 3.01e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 2.50e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 2.38e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 3.92e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 2.66e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 4.31e-16 |  |
| utv_apply | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 1.12e-15 |  |
| utv_entries | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 2.26e-16 |  |
| utv_probe | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 7.56e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 1.64e-15 |  |
| utv_entries_derivatives | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 5.26e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 6.89e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 1.74e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 6.37e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 1.74e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 4.77e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 9.78e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 5.41e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 3.64e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | () | None | PASS | 5.41e-16 |  |
| utv_apply | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 9.05e-16 |  |
| utv_entries | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.15e-16 |  |
| utv_probe | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 5.38e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.80e-15 |  |
| utv_entries_derivatives | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.13e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 6.70e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.83e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.39e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.97e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 4.78e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.33e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.74e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.17e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 6.58e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.74e-16 |  |
| u_manifold_inner | d4 | uniform+pad | () | () | () | None | PASS | 1.13e-15 |  |
| u_manifold_norm | d4 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_gauge_project_idempotent | d4 | uniform+pad | () | () | () | None | PASS | 5.53e-16 |  |
| u_tangent_add_scale | d4 | uniform+pad | () | () | () | None | PASS | 2.38e-16 |  |
| u_tangent_reverse | d4 | uniform+pad | () | () | () | None | PASS | 1.95e-16 |  |
| u_retract_zero | d4 | uniform+pad | () | () | () | None | PASS | 1.08e-15 |  |
| u_retract_fd_jacobian | d4 | uniform+pad | () | () | () | None | PASS | 5.47e-09 |  |
| u_retract_vs_ragged | d4 | uniform+pad | () | () | () | None | PASS | 2.86e-15 |  |
| u_project_ambient | d4 | uniform+pad | () | () | () | None | PASS | 2.70e-16 |  |
| u_transport_identity | d4 | uniform+pad | () | () | () | None | PASS | 1.10e-15 |  |
| u_transport_vs_ragged_projection | d4 | uniform+pad | () | () | () | None | PASS | 6.13e-16 |  |
| u_manifold_inner | d4 | uniform+pad | () | () | (2,) | None | PASS | 4.34e-16 |  |
| u_manifold_norm | d4 | uniform+pad | () | () | (2,) | None | PASS | 9.06e-17 |  |
| u_gauge_project_idempotent | d4 | uniform+pad | () | () | (2,) | None | PASS | 4.72e-16 |  |
| u_tangent_add_scale | d4 | uniform+pad | () | () | (2,) | None | PASS | 2.22e-16 |  |
| u_tangent_reverse | d4 | uniform+pad | () | () | (2,) | None | PASS | 2.12e-16 |  |
| u_retract_zero | d4 | uniform+pad | () | () | (2,) | None | PASS | 1.08e-15 |  |
| u_retract_fd_jacobian | d4 | uniform+pad | () | () | (2,) | None | PASS | 1.14e-08 | ratio=4.00 |
| u_project_ambient | d4 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 20 into shape (4,2,5,1) |
| u_transport_identity | d4 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 20 into shape (4,2,5,1) |
| u_transport_vs_ragged_projection | d4 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 20 into shape (4,2,5,1) |
| u_to_dense | d4 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d4 | uniform+pad | (2,) | () | () | None | PASS | 8.05e-17 |  |
| u_sub | d4 | uniform+pad | (2,) | () | () | None | PASS | 8.21e-17 |  |
| u_scalar_mul | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.46e-16 |  |
| u_inner | d4 | uniform+pad | (2,) | () | () | None | PASS | 2.30e-15 |  |
| u_norm | d4 | uniform+pad | (2,) | () | () | None | PASS | 7.36e-16 |  |
| u_reverse | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.55e-16 |  |
| u_t3svd_lossless | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.76e-15 |  |
| u_rank_adjustment_sweep | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.69e-15 |  |
| u_t3svd_trunc_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 8.09e-16 |  |
| u_orthogonal_representations | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.35e-15 |  |
| u_apply | d4 | uniform+pad | (2,) | () | () | None | PASS | 3.29e-16 |  |
| u_entries | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.36e-16 |  |
| u_probe | d4 | uniform+pad | (2,) | () | () | None | PASS | 8.85e-16 |  |
| u_apply_derivatives | d4 | uniform+pad | (2,) | () | () | None | PASS | 2.14e-15 |  |
| u_entries_derivatives | d4 | uniform+pad | (2,) | () | () | None | PASS | 6.27e-16 |  |
| u_probe_derivatives | d4 | uniform+pad | (2,) | () | () | None | PASS | 7.80e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 3.62e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 3.97e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 4.28e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 3.62e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 3.97e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 4.28e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 6.69e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 4.31e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 2.91e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 6.69e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 4.31e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 2.91e-16 |  |
| utv_apply | d4 | uniform+pad | (2,) | () | () | None | PASS | 2.47e-15 |  |
| utv_entries | d4 | uniform+pad | (2,) | () | () | None | PASS | 2.98e-16 |  |
| utv_probe | d4 | uniform+pad | (2,) | () | () | None | PASS | 6.52e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2,) | () | () | None | PASS | 8.08e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2,) | () | () | None | PASS | 8.22e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2,) | () | () | None | PASS | 3.72e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | () | None | PASS | 5.99e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | () | None | PASS | 7.80e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | () | None | PASS | 5.99e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | () | None | PASS | 7.80e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | () | None | PASS | 2.03e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.21e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.60e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | () | None | PASS | 2.03e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.21e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.60e-16 |  |
| utv_apply | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 5.99e-16 |  |
| utv_entries | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.15e-15 |  |
| utv_probe | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 5.62e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.76e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 6.88e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 5.11e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.55e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.01e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 6.31e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.55e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.01e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 6.31e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.41e-14 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.17e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.03e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.41e-14 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.17e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.03e-16 |  |
| u_apply | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 9.02e-16 |  |
| u_entries | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 5.56e-16 |  |
| u_probe | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 5.19e-16 |  |
| u_apply_derivatives | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 7.24e-16 |  |
| u_entries_derivatives | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.69e-16 |  |
| u_probe_derivatives | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.13e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 5.27e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 5.63e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.82e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 5.90e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 5.71e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.67e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.23e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.65e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.47e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.98e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.84e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.45e-16 |  |
| utv_apply | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.93e-16 |  |
| utv_entries | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.79e-16 |  |
| utv_probe | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 6.55e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 6.60e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 6.90e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 5.14e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.46e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.84e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.44e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.73e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.84e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.66e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 7.47e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.82e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 9.89e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 7.47e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.23e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.27e-15 |  |
| utv_apply | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 9.41e-16 |  |
| utv_entries | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.03e-16 |  |
| utv_probe | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 4.55e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 8.04e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 5.75e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 4.68e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 4.47e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 8.18e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.39e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.49e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.64e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.70e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.84e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.92e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 6.32e-16 |  |
| u_apply | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 5.00e-17 |  |
| u_entries | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.08e-15 |  |
| u_probe | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.91e-16 |  |
| u_apply_derivatives | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 8.15e-16 |  |
| u_entries_derivatives | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 9.40e-16 |  |
| u_probe_derivatives | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.79e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.62e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.73e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.77e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.87e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.76e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 5.64e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.40e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.99e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.16e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 5.12e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 6.39e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.85e-16 |  |
| utv_apply | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.75e-16 |  |
| utv_entries | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.36e-16 |  |
| utv_probe | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 5.92e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.30e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 6.14e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 5.84e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 6.74e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.28e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.74e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.37e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.49e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.98e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.11e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.28e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.98e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.11e-16 |  |
| utv_apply | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 7.47e-16 |  |
| utv_entries | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 4.84e-16 |  |
| utv_probe | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 4.95e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 8.70e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 4.97e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 5.61e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 5.49e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 5.52e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.18e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 5.49e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 5.52e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.18e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.15e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 5.44e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 9.88e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.16e-15 |  |
| u_manifold_norm | d4 | uniform+pad | (2,) | () | () | None | PASS | 3.29e-16 |  |
| u_gauge_project_idempotent | d4 | uniform+pad | (2,) | () | () | None | PASS | 4.40e-16 |  |
| u_tangent_add_scale | d4 | uniform+pad | (2,) | () | () | None | PASS | 2.49e-16 |  |
| u_tangent_reverse | d4 | uniform+pad | (2,) | () | () | None | PASS | 2.30e-16 |  |
| u_retract_zero | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.75e-15 |  |
| u_retract_fd_jacobian | d4 | uniform+pad | (2,) | () | () | None | PASS | 5.80e-09 |  |
| u_retract_vs_ragged | d4 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d4 | uniform+pad | (2,) | () | () | None | PASS | 3.57e-16 |  |
| u_transport_identity | d4 | uniform+pad | (2,) | () | () | None | PASS | 1.07e-15 |  |
| u_transport_vs_ragged_projection | d4 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.09e-15 |  |
| u_manifold_norm | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 4.27e-16 |  |
| u_gauge_project_idempotent | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 5.91e-16 |  |
| u_tangent_add_scale | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.19e-16 |  |
| u_tangent_reverse | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.94e-16 |  |
| u_retract_zero | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.75e-15 |  |
| u_retract_fd_jacobian | d4 | uniform+pad | (2,) | () | (2,) | None | PASS | 5.01e-09 |  |
| u_project_ambient | d4 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 40 into shape (4,2,2,5,1) |
| u_transport_identity | d4 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 40 into shape (4,2,2,5,1) |
| u_transport_vs_ragged_projection | d4 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 40 into shape (4,2,2,5,1) |
| u_to_dense | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_add | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.06e-16 |  |
| u_sub | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 9.59e-17 |  |
| u_scalar_mul | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.51e-16 |  |
| u_inner | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 2.69e-15 |  |
| u_norm | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 3.59e-16 |  |
| u_reverse | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.46e-16 |  |
| u_t3svd_lossless | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.43e-15 |  |
| u_rank_adjustment_sweep | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.45e-15 |  |
| u_t3svd_trunc_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.13e-15 |  |
| u_orthogonal_representations | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.51e-15 |  |
| u_apply | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 3.72e-16 |  |
| u_entries | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.91e-16 |  |
| u_probe | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 5.26e-16 |  |
| u_apply_derivatives | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 9.27e-16 |  |
| u_entries_derivatives | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 8.69e-16 |  |
| u_probe_derivatives | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 8.27e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 3.43e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 2.64e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 3.45e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 3.43e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 2.64e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 3.45e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.88e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.74e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 2.99e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.88e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.74e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 2.99e-16 |  |
| utv_apply | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 6.35e-16 |  |
| utv_entries | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 2.08e-16 |  |
| utv_probe | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 6.29e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.02e-15 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.62e-15 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 9.03e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 3.98e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 5.78e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 6.05e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 3.98e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 5.78e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 6.05e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 2.26e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 8.11e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 4.84e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 2.26e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 8.11e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 4.84e-16 |  |
| utv_apply | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.34e-16 |  |
| utv_entries | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.60e-16 |  |
| utv_probe | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.89e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 8.25e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 7.46e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 5.94e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.02e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 7.36e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.02e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 7.36e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.32e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.32e-16 |  |
| u_apply | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 7.34e-16 |  |
| u_entries | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.91e-16 |  |
| u_probe | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 5.41e-16 |  |
| u_apply_derivatives | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.24e-16 |  |
| u_entries_derivatives | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.13e-16 |  |
| u_probe_derivatives | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 4.79e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.06e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.29e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.80e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.89e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.01e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.59e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.77e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.08e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.42e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.18e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.89e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.99e-16 |  |
| utv_apply | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 6.44e-16 |  |
| utv_entries | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.09e-16 |  |
| utv_probe | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 5.57e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.22e-15 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.38e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 6.23e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.81e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 5.67e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 9.13e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.11e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 4.25e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.52e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.35e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 5.24e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 5.70e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 7.69e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 5.24e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 7.72e-16 |  |
| utv_entries | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 4.12e-16 |  |
| utv_probe | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 8.88e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 8.54e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 5.06e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 6.93e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 4.23e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.72e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 5.92e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.11e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 6.19e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.93e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.18e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.12e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.87e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.18e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.62e-15 |  |
| u_entries | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.24e-16 |  |
| u_probe | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 6.01e-16 |  |
| u_apply_derivatives | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 9.76e-16 |  |
| u_entries_derivatives | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.49e-16 |  |
| u_probe_derivatives | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 5.91e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.61e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.48e-16 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.41e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.26e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.48e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.69e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.70e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.59e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.54e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.51e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.89e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.79e-16 |  |
| utv_apply | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 8.80e-16 |  |
| utv_entries | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.54e-16 |  |
| utv_probe | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 7.34e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 6.62e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 4.93e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 6.94e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.46e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.58e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.13e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.05e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.58e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.22e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 5.35e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.24e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.57e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 4.14e-16 |  |
| utv_apply | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 8.02e-16 |  |
| utv_entries | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.47e-16 |  |
| utv_probe | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 5.29e-16 |  |
| utv_apply_derivatives | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 7.96e-16 |  |
| utv_entries_derivatives | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.60e-16 |  |
| utv_probe_derivatives | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 5.16e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.45e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.81e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.53e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.22e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.91e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.84e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 6.35e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.84e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.76e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | d4 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 2.02e-15 |  |
| u_manifold_norm | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 6.46e-16 |  |
| u_gauge_project_idempotent | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 6.61e-16 |  |
| u_tangent_add_scale | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 2.54e-16 |  |
| u_tangent_reverse | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.98e-16 |  |
| u_retract_zero | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.15e-15 |  |
| u_retract_fd_jacobian | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 5.64e-08 | ratio=4.00 |
| u_retract_vs_ragged | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.50e-15 |  |
| u_transport_identity | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 1.42e-15 |  |
| u_transport_vs_ragged_projection | d4 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.16e-15 |  |
| u_manifold_norm | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.79e-16 |  |
| u_gauge_project_idempotent | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 5.19e-16 |  |
| u_tangent_add_scale | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.44e-16 |  |
| u_tangent_reverse | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.05e-16 |  |
| u_retract_zero | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.15e-15 |  |
| u_retract_fd_jacobian | d4 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 5.56e-08 | ratio=4.00 |
| u_project_ambient | d4 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 120 into shape (4,2,2,3,5,1) |
| u_transport_identity | d4 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 120 into shape (4,2,2,3,5,1) |
| u_transport_vs_ragged_projection | d4 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 120 into shape (4,2,2,3,5,1) |
| u_to_dense | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_add | rank1 | uniform | () | () | () | None | PASS | 1.01e-17 |  |
| u_sub | rank1 | uniform | () | () | () | None | PASS | 7.13e-18 |  |
| u_scalar_mul | rank1 | uniform | () | () | () | None | PASS | 6.67e-17 |  |
| u_inner | rank1 | uniform | () | () | () | None | PASS | 1.02e-15 |  |
| u_norm | rank1 | uniform | () | () | () | None | PASS | 2.16e-16 |  |
| u_reverse | rank1 | uniform | () | () | () | None | PASS | 6.79e-17 |  |
| u_t3svd_lossless | rank1 | uniform | () | () | () | None | PASS | 1.49e-16 |  |
| u_rank_adjustment_sweep | rank1 | uniform | () | () | () | None | PASS | 1.04e-16 |  |
| u_t3svd_trunc_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_orthogonal_representations | rank1 | uniform | () | () | () | None | PASS | 1.30e-16 |  |
| u_apply | rank1 | uniform | () | () | () | None | PASS | 2.62e-16 |  |
| u_entries | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe | rank1 | uniform | () | () | () | None | PASS | 4.62e-16 |  |
| u_apply_derivatives | rank1 | uniform | () | () | () | None | PASS | 8.49e-16 |  |
| u_entries_derivatives | rank1 | uniform | () | () | () | None | PASS | 1.50e-15 |  |
| u_probe_derivatives | rank1 | uniform | () | () | () | None | PASS | 3.29e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 2.32e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 1.79e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 2.32e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 1.79e-16 |  |
| utv_apply | rank1 | uniform | () | () | () | None | PASS | 8.62e-16 |  |
| utv_entries | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe | rank1 | uniform | () | () | () | None | PASS | 7.78e-16 |  |
| utv_apply_derivatives | rank1 | uniform | () | () | () | None | PASS | 6.10e-16 |  |
| utv_entries_derivatives | rank1 | uniform | () | () | () | None | PASS | 6.27e-16 |  |
| utv_probe_derivatives | rank1 | uniform | () | () | () | None | PASS | 5.05e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | () | () | () | None | PASS | 1.73e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | () | () | () | None | PASS | 1.53e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | () | () | () | None | PASS | 1.56e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | () | () | () | None | PASS | 1.73e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | () | () | () | None | PASS | 1.53e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | () | () | () | None | PASS | 1.56e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | () | () | None | PASS | 3.67e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | () | () | None | PASS | 1.10e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | () | () | None | PASS | 1.30e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | () | () | None | PASS | 3.67e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | () | () | None | PASS | 1.10e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | () | () | None | PASS | 1.30e-16 |  |
| utv_apply | rank1 | uniform | () | () | (2,) | None | PASS | 3.75e-17 |  |
| utv_entries | rank1 | uniform | () | () | (2,) | None | PASS | 4.23e-17 |  |
| utv_probe | rank1 | uniform | () | () | (2,) | None | PASS | 2.76e-16 |  |
| utv_apply_derivatives | rank1 | uniform | () | () | (2,) | None | PASS | 9.65e-16 |  |
| utv_entries_derivatives | rank1 | uniform | () | () | (2,) | None | PASS | 1.94e-15 |  |
| utv_probe_derivatives | rank1 | uniform | () | () | (2,) | None | PASS | 2.94e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | () | () | (2,) | None | PASS | 1.79e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | () | () | (2,) | None | PASS | 1.18e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | () | () | (2,) | None | PASS | 7.46e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | () | () | (2,) | None | PASS | 1.79e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | () | () | (2,) | None | PASS | 1.18e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | () | () | (2,) | None | PASS | 7.46e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | () | (2,) | None | PASS | 7.52e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | () | (2,) | None | PASS | 1.61e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | () | (2,) | None | PASS | 7.52e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | () | (2,) | None | PASS | 1.61e-16 |  |
| u_apply | rank1 | uniform | () | (3,) | () | None | PASS | 3.84e-16 |  |
| u_entries | rank1 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe | rank1 | uniform | () | (3,) | () | None | PASS | 1.56e-15 |  |
| u_apply_derivatives | rank1 | uniform | () | (3,) | () | None | PASS | 5.74e-16 |  |
| u_entries_derivatives | rank1 | uniform | () | (3,) | () | None | PASS | 5.74e-16 |  |
| u_probe_derivatives | rank1 | uniform | () | (3,) | () | None | PASS | 4.40e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 2.02e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 2.03e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 2.66e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 1.54e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 3.45e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (3,) | () | None | PASS | 3.46e-16 |  |
| utv_apply | rank1 | uniform | () | (3,) | () | None | PASS | 4.77e-16 |  |
| utv_entries | rank1 | uniform | () | (3,) | () | None | PASS | 7.06e-17 |  |
| utv_probe | rank1 | uniform | () | (3,) | () | None | PASS | 5.49e-16 |  |
| utv_apply_derivatives | rank1 | uniform | () | (3,) | () | None | PASS | 1.31e-16 |  |
| utv_entries_derivatives | rank1 | uniform | () | (3,) | () | None | PASS | 1.70e-16 |  |
| utv_probe_derivatives | rank1 | uniform | () | (3,) | () | None | PASS | 2.98e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | () | None | PASS | 2.06e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | () | None | PASS | 1.67e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | () | None | PASS | 1.39e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | () | None | PASS | 2.06e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | () | None | PASS | 1.67e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | () | None | PASS | 3.40e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | () | None | PASS | 3.14e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | () | None | PASS | 4.24e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | () | None | PASS | 4.24e-16 |  |
| utv_apply | rank1 | uniform | () | (3,) | (2,) | None | PASS | 5.11e-16 |  |
| utv_entries | rank1 | uniform | () | (3,) | (2,) | None | PASS | 1.94e-16 |  |
| utv_probe | rank1 | uniform | () | (3,) | (2,) | None | PASS | 2.84e-16 |  |
| utv_apply_derivatives | rank1 | uniform | () | (3,) | (2,) | None | PASS | 4.09e-16 |  |
| utv_entries_derivatives | rank1 | uniform | () | (3,) | (2,) | None | PASS | 3.95e-16 |  |
| utv_probe_derivatives | rank1 | uniform | () | (3,) | (2,) | None | PASS | 3.89e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 1.79e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 2.88e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 3.62e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 1.79e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 1.44e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 3.62e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 5.62e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 2.23e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 7.97e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 7.31e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 9.55e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (3,) | (2,) | None | PASS | 5.98e-16 |  |
| u_apply | rank1 | uniform | () | (2, 2) | () | None | PASS | 5.58e-16 |  |
| u_entries | rank1 | uniform | () | (2, 2) | () | None | PASS | 2.27e-17 |  |
| u_probe | rank1 | uniform | () | (2, 2) | () | None | PASS | 3.21e-16 |  |
| u_apply_derivatives | rank1 | uniform | () | (2, 2) | () | None | PASS | 2.73e-16 |  |
| u_entries_derivatives | rank1 | uniform | () | (2, 2) | () | None | PASS | 1.92e-16 |  |
| u_probe_derivatives | rank1 | uniform | () | (2, 2) | () | None | PASS | 3.48e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 2.13e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 1.76e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 3.51e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 1.26e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 4.90e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | () | (2, 2) | () | None | PASS | 2.65e-16 |  |
| utv_apply | rank1 | uniform | () | (2, 2) | () | None | PASS | 1.88e-16 |  |
| utv_entries | rank1 | uniform | () | (2, 2) | () | None | PASS | 1.83e-16 |  |
| utv_probe | rank1 | uniform | () | (2, 2) | () | None | PASS | 2.56e-16 |  |
| utv_apply_derivatives | rank1 | uniform | () | (2, 2) | () | None | PASS | 2.36e-16 |  |
| utv_entries_derivatives | rank1 | uniform | () | (2, 2) | () | None | PASS | 2.00e-16 |  |
| utv_probe_derivatives | rank1 | uniform | () | (2, 2) | () | None | PASS | 3.33e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | () | None | PASS | 6.46e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | () | None | PASS | 6.46e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | () | None | PASS | 1.13e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | () | None | PASS | 1.24e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | () | None | PASS | 2.05e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | () | None | PASS | 2.48e-16 |  |
| utv_apply | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 3.14e-16 |  |
| utv_entries | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 1.91e-16 |  |
| utv_probe | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 3.68e-16 |  |
| utv_apply_derivatives | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 4.34e-16 |  |
| utv_entries_derivatives | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 2.38e-16 |  |
| utv_probe_derivatives | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 3.35e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 1.77e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 2.63e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 1.77e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 2.63e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 1.77e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 1.14e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 2.47e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 8.84e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 3.79e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | () | (2, 2) | (2,) | None | PASS | 3.70e-16 |  |
| u_manifold_inner | rank1 | uniform | () | () | () | None | PASS | 1.74e-15 |  |
| u_manifold_norm | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_gauge_project_idempotent | rank1 | uniform | () | () | () | None | PASS | 5.38e-16 |  |
| u_tangent_add_scale | rank1 | uniform | () | () | () | None | PASS | 1.66e-16 |  |
| u_tangent_reverse | rank1 | uniform | () | () | () | None | PASS | 9.88e-17 |  |
| u_retract_zero | rank1 | uniform | () | () | () | None | PASS | 5.11e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform | () | () | () | None | PASS | 5.21e-09 |  |
| u_retract_vs_ragged | rank1 | uniform | () | () | () | None | PASS | 4.61e-16 |  |
| u_project_ambient | rank1 | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_transport_identity | rank1 | uniform | () | () | () | None | PASS | 2.05e-16 |  |
| u_transport_vs_ragged_projection | rank1 | uniform | () | () | () | None | PASS | 2.09e-16 |  |
| u_manifold_inner | rank1 | uniform | () | () | (2,) | None | PASS | 5.07e-16 |  |
| u_manifold_norm | rank1 | uniform | () | () | (2,) | None | PASS | 2.24e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform | () | () | (2,) | None | PASS | 1.60e-16 |  |
| u_tangent_add_scale | rank1 | uniform | () | () | (2,) | None | PASS | 1.56e-16 |  |
| u_tangent_reverse | rank1 | uniform | () | () | (2,) | None | PASS | 1.34e-16 |  |
| u_retract_zero | rank1 | uniform | () | () | (2,) | None | PASS | 5.11e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform | () | () | (2,) | None | PASS | 2.68e-08 | ratio=4.00 |
| u_project_ambient | rank1 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 3 into shape (3,2,1,1) |
| u_transport_identity | rank1 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 3 into shape (3,2,1,1) |
| u_transport_vs_ragged_projection | rank1 | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 3 into shape (3,2,1,1) |
| u_to_dense | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_add | rank1 | uniform | (2,) | () | () | None | PASS | 4.22e-17 |  |
| u_sub | rank1 | uniform | (2,) | () | () | None | PASS | 2.59e-17 |  |
| u_scalar_mul | rank1 | uniform | (2,) | () | () | None | PASS | 1.60e-16 |  |
| u_inner | rank1 | uniform | (2,) | () | () | None | PASS | 1.16e-15 |  |
| u_norm | rank1 | uniform | (2,) | () | () | None | PASS | 2.22e-16 |  |
| u_reverse | rank1 | uniform | (2,) | () | () | None | PASS | 8.77e-17 |  |
| u_t3svd_lossless | rank1 | uniform | (2,) | () | () | None | PASS | 3.37e-16 |  |
| u_rank_adjustment_sweep | rank1 | uniform | (2,) | () | () | None | PASS | 3.31e-16 |  |
| u_t3svd_trunc_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_orthogonal_representations | rank1 | uniform | (2,) | () | () | None | PASS | 3.31e-16 |  |
| u_apply | rank1 | uniform | (2,) | () | () | None | PASS | 1.67e-16 |  |
| u_entries | rank1 | uniform | (2,) | () | () | None | PASS | 2.32e-16 |  |
| u_probe | rank1 | uniform | (2,) | () | () | None | PASS | 2.01e-16 |  |
| u_apply_derivatives | rank1 | uniform | (2,) | () | () | None | PASS | 6.09e-16 |  |
| u_entries_derivatives | rank1 | uniform | (2,) | () | () | None | PASS | 5.45e-16 |  |
| u_probe_derivatives | rank1 | uniform | (2,) | () | () | None | PASS | 6.47e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 2.49e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 1.61e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 2.06e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 2.49e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 1.61e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 2.06e-16 |  |
| utv_apply | rank1 | uniform | (2,) | () | () | None | PASS | 8.44e-17 |  |
| utv_entries | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe | rank1 | uniform | (2,) | () | () | None | PASS | 3.80e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2,) | () | () | None | PASS | 3.21e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2,) | () | () | None | PASS | 3.00e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2,) | () | () | None | PASS | 3.85e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | () | None | PASS | 2.14e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | () | None | PASS | 1.40e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | () | None | PASS | 2.14e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | () | None | PASS | 1.40e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | () | None | PASS | 1.26e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | () | None | PASS | 4.96e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | () | None | PASS | 2.36e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | () | None | PASS | 1.26e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | () | None | PASS | 4.96e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | () | None | PASS | 2.36e-16 |  |
| utv_apply | rank1 | uniform | (2,) | () | (2,) | None | PASS | 3.39e-16 |  |
| utv_entries | rank1 | uniform | (2,) | () | (2,) | None | PASS | 1.16e-16 |  |
| utv_probe | rank1 | uniform | (2,) | () | (2,) | None | PASS | 3.00e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2,) | () | (2,) | None | PASS | 3.51e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2,) | () | (2,) | None | PASS | 3.75e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2,) | () | (2,) | None | PASS | 2.29e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 4.29e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 1.72e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 4.29e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 1.72e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 3.33e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 1.13e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 2.35e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 3.33e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 1.13e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | () | (2,) | None | PASS | 2.35e-16 |  |
| u_apply | rank1 | uniform | (2,) | (3,) | () | None | PASS | 8.45e-16 |  |
| u_entries | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.32e-17 |  |
| u_probe | rank1 | uniform | (2,) | (3,) | () | None | PASS | 4.04e-16 |  |
| u_apply_derivatives | rank1 | uniform | (2,) | (3,) | () | None | PASS | 4.66e-16 |  |
| u_entries_derivatives | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.98e-16 |  |
| u_probe_derivatives | rank1 | uniform | (2,) | (3,) | () | None | PASS | 3.07e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.64e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.34e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.64e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.44e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.73e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (3,) | () | None | PASS | 2.52e-17 |  |
| utv_apply | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.93e-16 |  |
| utv_entries | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.33e-16 |  |
| utv_probe | rank1 | uniform | (2,) | (3,) | () | None | PASS | 2.82e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2,) | (3,) | () | None | PASS | 3.15e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2,) | (3,) | () | None | PASS | 2.01e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2,) | (3,) | () | None | PASS | 3.22e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.67e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.58e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 5.83e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 4.93e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 8.51e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 5.83e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 4.93e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | () | None | PASS | 1.49e-15 |  |
| utv_apply | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.53e-16 |  |
| utv_entries | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.25e-16 |  |
| utv_probe | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.37e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 3.86e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 4.06e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.42e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.95e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.35e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.95e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 7.06e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.62e-14 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.09e-14 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.45e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 2.36e-14 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 1.09e-14 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (3,) | (2,) | None | PASS | 3.68e-16 |  |
| u_apply | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 2.01e-16 |  |
| u_entries | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 8.48e-18 |  |
| u_probe | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 4.68e-16 |  |
| u_apply_derivatives | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 2.46e-16 |  |
| u_entries_derivatives | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 2.40e-16 |  |
| u_probe_derivatives | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 2.55e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.40e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 5.14e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.11e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.34e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.14e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 3.51e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 2.72e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.41e-16 |  |
| utv_apply | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 5.01e-16 |  |
| utv_entries | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.25e-16 |  |
| utv_probe | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 2.47e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 3.29e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 4.97e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 2.16e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.88e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 6.65e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 5.65e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.16e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 8.87e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.81e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.46e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.32e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.81e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.46e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | () | None | PASS | 1.32e-16 |  |
| utv_apply | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.75e-16 |  |
| utv_entries | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 5.41e-17 |  |
| utv_probe | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.47e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.34e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.16e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.51e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.81e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.81e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 9.84e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 8.02e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.22e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 6.15e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.26e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.31e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.08e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.13e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2,) | (2, 2) | (2,) | None | PASS | 5.24e-16 |  |
| u_manifold_inner | rank1 | uniform | (2,) | () | () | None | PASS | 2.73e-15 |  |
| u_manifold_norm | rank1 | uniform | (2,) | () | () | None | PASS | 2.19e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform | (2,) | () | () | None | PASS | 1.47e-16 |  |
| u_tangent_add_scale | rank1 | uniform | (2,) | () | () | None | PASS | 1.47e-16 |  |
| u_tangent_reverse | rank1 | uniform | (2,) | () | () | None | PASS | 1.21e-16 |  |
| u_retract_zero | rank1 | uniform | (2,) | () | () | None | PASS | 2.12e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform | (2,) | () | () | None | PASS | 4.71e-08 | ratio=4.00 |
| u_retract_vs_ragged | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | rank1 | uniform | (2,) | () | () | None | PASS | 4.28e-16 |  |
| u_transport_identity | rank1 | uniform | (2,) | () | () | None | PASS | 5.26e-16 |  |
| u_transport_vs_ragged_projection | rank1 | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | rank1 | uniform | (2,) | () | (2,) | None | PASS | 4.62e-16 |  |
| u_manifold_norm | rank1 | uniform | (2,) | () | (2,) | None | PASS | 2.63e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform | (2,) | () | (2,) | None | PASS | 1.35e-16 |  |
| u_tangent_add_scale | rank1 | uniform | (2,) | () | (2,) | None | PASS | 1.46e-16 |  |
| u_tangent_reverse | rank1 | uniform | (2,) | () | (2,) | None | PASS | 1.28e-16 |  |
| u_retract_zero | rank1 | uniform | (2,) | () | (2,) | None | PASS | 2.12e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform | (2,) | () | (2,) | None | PASS | 6.47e-08 | ratio=4.00 |
| u_project_ambient | rank1 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 6 into shape (3,2,2,1,1) |
| u_transport_identity | rank1 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 6 into shape (3,2,2,1,1) |
| u_transport_vs_ragged_projection | rank1 | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 6 into shape (3,2,2,1,1) |
| u_to_dense | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_add | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.68e-17 |  |
| u_sub | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.78e-17 |  |
| u_scalar_mul | rank1 | uniform | (2, 3) | () | () | None | PASS | 8.03e-17 |  |
| u_inner | rank1 | uniform | (2, 3) | () | () | None | PASS | 8.25e-16 |  |
| u_norm | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.89e-17 |  |
| u_reverse | rank1 | uniform | (2, 3) | () | () | None | PASS | 8.43e-17 |  |
| u_t3svd_lossless | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.70e-16 |  |
| u_rank_adjustment_sweep | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.73e-16 |  |
| u_t3svd_trunc_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_orthogonal_representations | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.78e-16 |  |
| u_apply | rank1 | uniform | (2, 3) | () | () | None | PASS | 7.20e-16 |  |
| u_entries | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.16e-17 |  |
| u_probe | rank1 | uniform | (2, 3) | () | () | None | PASS | 4.33e-16 |  |
| u_apply_derivatives | rank1 | uniform | (2, 3) | () | () | None | PASS | 4.24e-16 |  |
| u_entries_derivatives | rank1 | uniform | (2, 3) | () | () | None | PASS | 3.96e-16 |  |
| u_probe_derivatives | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.26e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.53e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.90e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.32e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.53e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.90e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.32e-16 |  |
| utv_apply | rank1 | uniform | (2, 3) | () | () | None | PASS | 5.98e-16 |  |
| utv_entries | rank1 | uniform | (2, 3) | () | () | None | PASS | 4.34e-18 |  |
| utv_probe | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.64e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.06e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.06e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.05e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | () | None | PASS | 3.13e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.56e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | () | None | PASS | 3.13e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.56e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 3.28e-16 |  |
| utv_entries | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 4.06e-17 |  |
| utv_probe | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 2.94e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 2.75e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 2.74e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 2.44e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 1.65e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 2.55e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 3.20e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 1.65e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 2.55e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 3.20e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 3.72e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 3.72e-16 |  |
| u_apply | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 4.10e-16 |  |
| u_entries | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 8.38e-17 |  |
| u_probe | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 2.92e-16 |  |
| u_apply_derivatives | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 2.16e-16 |  |
| u_entries_derivatives | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 1.55e-16 |  |
| u_probe_derivatives | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 3.22e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 1.52e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 1.19e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 1.31e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 2.15e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 1.94e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 2.45e-16 |  |
| utv_apply | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 2.78e-16 |  |
| utv_entries | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 1.11e-16 |  |
| utv_probe | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 2.58e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 3.32e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 3.55e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 2.75e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 5.07e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 6.57e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 1.87e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 3.94e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 6.71e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 5.31e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 9.39e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 1.77e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | () | None | PASS | 2.49e-16 |  |
| utv_apply | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.55e-16 |  |
| utv_entries | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.09e-16 |  |
| utv_probe | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.28e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.06e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.75e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.26e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.62e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.67e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 8.18e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.31e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.34e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.46e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.31e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.20e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.93e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.34e-15 |  |
| u_apply | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.58e-15 |  |
| u_entries | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.10e-16 |  |
| u_probe | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 7.45e-16 |  |
| u_apply_derivatives | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.96e-16 |  |
| u_entries_derivatives | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.47e-16 |  |
| u_probe_derivatives | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.77e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.11e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.84e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.40e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.50e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.56e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 9.71e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.88e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.36e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.31e-16 |  |
| utv_apply | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.26e-16 |  |
| utv_entries | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.32e-16 |  |
| utv_probe | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.53e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.16e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.76e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.81e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.31e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.39e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.50e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.31e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.78e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.25e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.09e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.69e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 9.26e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.41e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.25e-15 |  |
| utv_apply | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.47e-16 |  |
| utv_entries | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.01e-16 |  |
| utv_probe | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.76e-16 |  |
| utv_apply_derivatives | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.44e-16 |  |
| utv_entries_derivatives | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.75e-16 |  |
| utv_probe_derivatives | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.40e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.45e-14 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 4.17e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.01e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 7.06e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.39e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 4.36e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.19e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.74e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.96e-15 |  |
| u_manifold_inner | rank1 | uniform | (2, 3) | () | () | None | PASS | 4.03e-16 |  |
| u_manifold_norm | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.81e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.31e-16 |  |
| u_tangent_add_scale | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.38e-16 |  |
| u_tangent_reverse | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.20e-16 |  |
| u_retract_zero | rank1 | uniform | (2, 3) | () | () | None | PASS | 2.76e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.32e-05 | ratio=4.00 |
| u_retract_vs_ragged | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | rank1 | uniform | (2, 3) | () | () | None | PASS | 1.29e-15 |  |
| u_transport_identity | rank1 | uniform | (2, 3) | () | () | None | PASS | 4.64e-16 |  |
| u_transport_vs_ragged_projection | rank1 | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 7.28e-16 |  |
| u_manifold_norm | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 2.09e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 1.76e-16 |  |
| u_tangent_add_scale | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 1.43e-16 |  |
| u_tangent_reverse | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 1.17e-16 |  |
| u_retract_zero | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 2.76e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform | (2, 3) | () | (2,) | None | PASS | 2.64e-05 | ratio=4.00 |
| u_project_ambient | rank1 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1,1) |
| u_transport_identity | rank1 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1,1) |
| u_transport_vs_ragged_projection | rank1 | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1,1) |
| u_to_dense | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_add | rank1 | uniform+pad | () | () | () | None | PASS | 1.01e-17 |  |
| u_sub | rank1 | uniform+pad | () | () | () | None | PASS | 7.13e-18 |  |
| u_scalar_mul | rank1 | uniform+pad | () | () | () | None | PASS | 6.67e-17 |  |
| u_inner | rank1 | uniform+pad | () | () | () | None | PASS | 1.02e-15 |  |
| u_norm | rank1 | uniform+pad | () | () | () | None | PASS | 2.16e-16 |  |
| u_reverse | rank1 | uniform+pad | () | () | () | None | PASS | 6.79e-17 |  |
| u_t3svd_lossless | rank1 | uniform+pad | () | () | () | None | PASS | 1.49e-16 |  |
| u_rank_adjustment_sweep | rank1 | uniform+pad | () | () | () | None | PASS | 1.04e-16 |  |
| u_t3svd_trunc_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_orthogonal_representations | rank1 | uniform+pad | () | () | () | None | PASS | 1.30e-16 |  |
| u_apply | rank1 | uniform+pad | () | () | () | None | PASS | 2.62e-16 |  |
| u_entries | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe | rank1 | uniform+pad | () | () | () | None | PASS | 4.62e-16 |  |
| u_apply_derivatives | rank1 | uniform+pad | () | () | () | None | PASS | 8.49e-16 |  |
| u_entries_derivatives | rank1 | uniform+pad | () | () | () | None | PASS | 1.50e-15 |  |
| u_probe_derivatives | rank1 | uniform+pad | () | () | () | None | PASS | 3.29e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 2.32e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 1.79e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 2.32e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 1.79e-16 |  |
| utv_apply | rank1 | uniform+pad | () | () | () | None | PASS | 3.36e-16 |  |
| utv_entries | rank1 | uniform+pad | () | () | () | None | PASS | 1.97e-16 |  |
| utv_probe | rank1 | uniform+pad | () | () | () | None | PASS | 1.08e-15 |  |
| utv_apply_derivatives | rank1 | uniform+pad | () | () | () | None | PASS | 8.10e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | () | () | () | None | PASS | 7.93e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | () | () | () | None | PASS | 5.05e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | () | None | PASS | 4.04e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | () | None | PASS | 1.18e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | () | None | PASS | 5.12e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | () | None | PASS | 4.04e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | () | None | PASS | 1.18e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | () | None | PASS | 5.12e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | () | None | PASS | 1.06e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | () | None | PASS | 1.47e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | () | None | PASS | 1.06e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | () | None | PASS | 1.47e-16 |  |
| utv_apply | rank1 | uniform+pad | () | () | (2,) | None | PASS | 7.07e-16 |  |
| utv_entries | rank1 | uniform+pad | () | () | (2,) | None | PASS | 3.19e-17 |  |
| utv_probe | rank1 | uniform+pad | () | () | (2,) | None | PASS | 3.38e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | () | () | (2,) | None | PASS | 1.54e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | () | () | (2,) | None | PASS | 1.60e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | () | () | (2,) | None | PASS | 3.09e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 1.25e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 2.37e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 1.25e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 2.37e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 3.52e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 4.23e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 3.52e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | () | (2,) | None | PASS | 4.23e-16 |  |
| u_apply | rank1 | uniform+pad | () | (3,) | () | None | PASS | 3.44e-16 |  |
| u_entries | rank1 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe | rank1 | uniform+pad | () | (3,) | () | None | PASS | 1.56e-15 |  |
| u_apply_derivatives | rank1 | uniform+pad | () | (3,) | () | None | PASS | 5.74e-16 |  |
| u_entries_derivatives | rank1 | uniform+pad | () | (3,) | () | None | PASS | 5.74e-16 |  |
| u_probe_derivatives | rank1 | uniform+pad | () | (3,) | () | None | PASS | 4.40e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 2.14e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 1.37e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 1.32e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 2.02e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 2.03e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 2.66e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 1.54e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 3.45e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (3,) | () | None | PASS | 3.46e-16 |  |
| utv_apply | rank1 | uniform+pad | () | (3,) | () | None | PASS | 4.22e-16 |  |
| utv_entries | rank1 | uniform+pad | () | (3,) | () | None | PASS | 1.67e-16 |  |
| utv_probe | rank1 | uniform+pad | () | (3,) | () | None | PASS | 5.05e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | () | (3,) | () | None | PASS | 2.77e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | () | (3,) | () | None | PASS | 3.51e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | () | (3,) | () | None | PASS | 3.24e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 1.31e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 4.11e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 4.11e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 3.42e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 1.30e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 4.10e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 1.14e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 1.30e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | () | None | PASS | 5.46e-16 |  |
| utv_apply | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 4.40e-16 |  |
| utv_entries | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.39e-16 |  |
| utv_probe | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.92e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.35e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.30e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.89e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 9.38e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.06e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 7.29e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 6.35e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 1.48e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 2.06e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 3.29e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 4.12e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 3.07e-16 |  |
| u_entries | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 2.27e-17 |  |
| u_probe | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 3.20e-16 |  |
| u_apply_derivatives | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 2.79e-16 |  |
| u_entries_derivatives | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 2.10e-16 |  |
| u_probe_derivatives | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 3.42e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.62e-18 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.24e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 5.56e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 3.76e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.76e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.46e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 5.13e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 4.90e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 2.65e-16 |  |
| utv_apply | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 2.94e-16 |  |
| utv_entries | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.85e-16 |  |
| utv_probe | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 3.67e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 2.78e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 3.17e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 3.71e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.80e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.26e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.15e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.80e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 6.28e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 4.40e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.32e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 5.28e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 2.20e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 1.32e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 5.52e-16 |  |
| utv_entries | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.49e-16 |  |
| utv_probe | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 4.07e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.81e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.53e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.40e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 7.83e-14 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.89e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.34e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.81e-14 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.89e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 4.46e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.18e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.24e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 4.50e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 4.37e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.62e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.38e-16 |  |
| u_manifold_inner | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_norm | rank1 | uniform+pad | () | () | () | None | PASS | 2.98e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform+pad | () | () | () | None | PASS | 1.58e-16 |  |
| u_tangent_add_scale | rank1 | uniform+pad | () | () | () | None | PASS | 1.05e-16 |  |
| u_tangent_reverse | rank1 | uniform+pad | () | () | () | None | PASS | 1.36e-16 |  |
| u_retract_zero | rank1 | uniform+pad | () | () | () | None | PASS | 4.13e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform+pad | () | () | () | None | PASS | 1.44e-08 | ratio=4.00 |
| u_retract_vs_ragged | rank1 | uniform+pad | () | () | () | None | PASS | 8.17e-16 |  |
| u_project_ambient | rank1 | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_transport_identity | rank1 | uniform+pad | () | () | () | None | PASS | 2.51e-16 |  |
| u_transport_vs_ragged_projection | rank1 | uniform+pad | () | () | () | None | PASS | 6.57e-16 |  |
| u_manifold_inner | rank1 | uniform+pad | () | () | (2,) | None | PASS | 6.47e-16 |  |
| u_manifold_norm | rank1 | uniform+pad | () | () | (2,) | None | PASS | 1.34e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform+pad | () | () | (2,) | None | PASS | 1.28e-16 |  |
| u_tangent_add_scale | rank1 | uniform+pad | () | () | (2,) | None | PASS | 1.72e-16 |  |
| u_tangent_reverse | rank1 | uniform+pad | () | () | (2,) | None | PASS | 1.30e-16 |  |
| u_retract_zero | rank1 | uniform+pad | () | () | (2,) | None | PASS | 4.13e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform+pad | () | () | (2,) | None | PASS | 1.83e-08 | ratio=4.00 |
| u_project_ambient | rank1 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 15 into shape (3,2,5,1) |
| u_transport_identity | rank1 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 15 into shape (3,2,5,1) |
| u_transport_vs_ragged_projection | rank1 | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 15 into shape (3,2,5,1) |
| u_to_dense | rank1 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_add | rank1 | uniform+pad | (2,) | () | () | None | PASS | 4.22e-17 |  |
| u_sub | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.59e-17 |  |
| u_scalar_mul | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.60e-16 |  |
| u_inner | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.16e-15 |  |
| u_norm | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.22e-16 |  |
| u_reverse | rank1 | uniform+pad | (2,) | () | () | None | PASS | 8.77e-17 |  |
| u_t3svd_lossless | rank1 | uniform+pad | (2,) | () | () | None | PASS | 3.37e-16 |  |
| u_rank_adjustment_sweep | rank1 | uniform+pad | (2,) | () | () | None | PASS | 3.31e-16 |  |
| u_t3svd_trunc_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_orthogonal_representations | rank1 | uniform+pad | (2,) | () | () | None | PASS | 3.31e-16 |  |
| u_apply | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.67e-16 |  |
| u_entries | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.32e-16 |  |
| u_probe | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.01e-16 |  |
| u_apply_derivatives | rank1 | uniform+pad | (2,) | () | () | None | PASS | 6.09e-16 |  |
| u_entries_derivatives | rank1 | uniform+pad | (2,) | () | () | None | PASS | 5.45e-16 |  |
| u_probe_derivatives | rank1 | uniform+pad | (2,) | () | () | None | PASS | 6.47e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.64e-16 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.22e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.64e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.22e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.49e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.61e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.06e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.49e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.61e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.06e-16 |  |
| utv_apply | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.32e-16 |  |
| utv_entries | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.41e-16 |  |
| utv_probe | rank1 | uniform+pad | (2,) | () | () | None | PASS | 3.25e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.35e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.61e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2,) | () | () | None | PASS | 3.28e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.86e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.62e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 4.76e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.86e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.62e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 4.76e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 3.67e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.40e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.11e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 3.67e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.40e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.11e-16 |  |
| utv_apply | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.17e-16 |  |
| utv_entries | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.10e-16 |  |
| utv_probe | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.73e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.34e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.39e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.59e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 5.07e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 5.07e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 5.97e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.02e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 5.97e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 3.02e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 8.55e-16 |  |
| u_entries | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.32e-17 |  |
| u_probe | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.04e-16 |  |
| u_apply_derivatives | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 4.69e-16 |  |
| u_entries_derivatives | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.08e-16 |  |
| u_probe_derivatives | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.07e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 8.40e-18 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.65e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.22e-18 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.64e-17 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.34e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.94e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.44e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.73e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.81e-16 |  |
| utv_apply | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.07e-16 |  |
| utv_entries | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.30e-16 |  |
| utv_probe | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.42e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.96e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.54e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.11e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.12e-14 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.51e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.65e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.68e-14 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.25e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 3.98e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.29e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 2.73e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | () | None | PASS | 1.47e-16 |  |
| utv_apply | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.09e-16 |  |
| utv_entries | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 7.36e-17 |  |
| utv_probe | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.23e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.78e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.54e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.35e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 7.43e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.06e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.22e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 7.43e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.91e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 6.70e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 6.50e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 5.86e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 6.37e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.25e-16 |  |
| u_apply | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.05e-16 |  |
| u_entries | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 8.48e-18 |  |
| u_probe | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.68e-16 |  |
| u_apply_derivatives | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.76e-16 |  |
| u_entries_derivatives | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.83e-16 |  |
| u_probe_derivatives | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.64e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.17e-17 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.41e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.40e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.31e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.06e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.21e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.57e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.49e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.02e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.07e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.99e-16 |  |
| utv_apply | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 7.09e-16 |  |
| utv_entries | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.54e-17 |  |
| utv_probe | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.34e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.96e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.11e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.96e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.16e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.13e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.16e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.75e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.53e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.84e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.77e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.90e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.84e-16 |  |
| utv_apply | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.39e-16 |  |
| utv_entries | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.33e-16 |  |
| utv_probe | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.57e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 4.30e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 4.11e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.32e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.11e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.37e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 7.86e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.37e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.48e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.85e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.91e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.69e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.93e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.91e-16 |  |
| u_manifold_inner | rank1 | uniform+pad | (2,) | () | () | None | PASS | 5.05e-16 |  |
| u_manifold_norm | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.97e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.69e-16 |  |
| u_tangent_add_scale | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.78e-16 |  |
| u_tangent_reverse | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.12e-16 |  |
| u_retract_zero | rank1 | uniform+pad | (2,) | () | () | None | PASS | 2.75e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform+pad | (2,) | () | () | None | PASS | 9.87e-08 | ratio=4.00 |
| u_retract_vs_ragged | rank1 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | rank1 | uniform+pad | (2,) | () | () | None | PASS | 1.71e-16 |  |
| u_transport_identity | rank1 | uniform+pad | (2,) | () | () | None | PASS | 4.84e-16 |  |
| u_transport_vs_ragged_projection | rank1 | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.71e-15 |  |
| u_manifold_norm | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.75e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.13e-16 |  |
| u_tangent_add_scale | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.49e-16 |  |
| u_tangent_reverse | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 1.12e-16 |  |
| u_retract_zero | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 2.75e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform+pad | (2,) | () | (2,) | None | PASS | 8.26e-08 | ratio=4.00 |
| u_project_ambient | rank1 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 30 into shape (3,2,2,5,1) |
| u_transport_identity | rank1 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 30 into shape (3,2,2,5,1) |
| u_transport_vs_ragged_projection | rank1 | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 30 into shape (3,2,2,5,1) |
| u_to_dense | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_add | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.68e-17 |  |
| u_sub | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.78e-17 |  |
| u_scalar_mul | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 8.03e-17 |  |
| u_inner | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 8.25e-16 |  |
| u_norm | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.89e-17 |  |
| u_reverse | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 8.43e-17 |  |
| u_t3svd_lossless | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.70e-16 |  |
| u_rank_adjustment_sweep | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.73e-16 |  |
| u_t3svd_trunc_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_orthogonal_representations | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.78e-16 |  |
| u_apply | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 7.20e-16 |  |
| u_entries | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.16e-17 |  |
| u_probe | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 4.33e-16 |  |
| u_apply_derivatives | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 4.24e-16 |  |
| u_entries_derivatives | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 3.96e-16 |  |
| u_probe_derivatives | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.26e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.43e-16 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.43e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.53e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.90e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.89e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.53e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.90e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.89e-16 |  |
| utv_apply | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.92e-16 |  |
| utv_entries | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 6.22e-17 |  |
| utv_probe | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 4.02e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 3.37e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 3.41e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.75e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.51e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 3.35e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.51e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 3.35e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.24e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.17e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.42e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.24e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.17e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.42e-16 |  |
| utv_apply | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.29e-16 |  |
| utv_entries | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.90e-17 |  |
| utv_probe | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.73e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.94e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.07e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.67e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.19e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.38e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.56e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.19e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.38e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.56e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.13e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.06e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.13e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.06e-16 |  |
| u_apply | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 4.10e-16 |  |
| u_entries | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 8.38e-17 |  |
| u_probe | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.92e-16 |  |
| u_apply_derivatives | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.14e-16 |  |
| u_entries_derivatives | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.55e-16 |  |
| u_probe_derivatives | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.30e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 4.04e-18 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.68e-17 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.38e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.70e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.15e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.66e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.71e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 9.34e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.91e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.95e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.31e-16 |  |
| utv_apply | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.83e-16 |  |
| utv_entries | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 9.33e-17 |  |
| utv_probe | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.06e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.52e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.95e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.74e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 9.47e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.56e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 7.58e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.25e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.64e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 8.74e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.26e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.79e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.26e-15 |  |
| utv_apply | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.39e-16 |  |
| utv_entries | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 8.33e-17 |  |
| utv_probe | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.86e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 4.23e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.68e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.91e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.21e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.10e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.21e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.10e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.51e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.61e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 5.31e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.34e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.23e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.65e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.34e-16 |  |
| u_apply | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.60e-15 |  |
| u_entries | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.10e-16 |  |
| u_probe | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 7.85e-16 |  |
| u_apply_derivatives | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.96e-16 |  |
| u_entries_derivatives | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 4.46e-16 |  |
| u_probe_derivatives | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.79e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 6.81e-17 |  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 5.02e-17 |  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.68e-16 |  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.84e-16 |  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.48e-16 |  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.50e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.56e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 9.72e-17 |  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.88e-16 |  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.36e-16 |  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.31e-16 |  |
| utv_apply | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.83e-16 |  |
| utv_entries | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 6.11e-17 |  |
| utv_probe | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.09e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.24e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.21e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.46e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 7.53e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.78e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.51e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.88e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 5.55e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.95e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 5.39e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.80e-16 |  |
| utv_apply | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.95e-16 |  |
| utv_entries | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 6.26e-17 |  |
| utv_probe | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.09e-16 |  |
| utv_apply_derivatives | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.92e-16 |  |
| utv_entries_derivatives | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.22e-16 |  |
| utv_probe_derivatives | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.52e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.48e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.69e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.22e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.48e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.69e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.22e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.62e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.65e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.26e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | rank1 | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.10e-15 |  |
| u_manifold_inner | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 4.67e-16 |  |
| u_manifold_norm | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.05e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 2.23e-16 |  |
| u_tangent_add_scale | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.71e-16 |  |
| u_tangent_reverse | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.34e-16 |  |
| u_retract_zero | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 3.25e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 1.05e-05 | ratio=4.00 |
| u_retract_vs_ragged | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 6.85e-16 |  |
| u_transport_identity | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 3.72e-16 |  |
| u_transport_vs_ragged_projection | rank1 | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_manifold_inner | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 5.97e-16 |  |
| u_manifold_norm | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.09e-16 |  |
| u_gauge_project_idempotent | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.58e-16 |  |
| u_tangent_add_scale | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.51e-16 |  |
| u_tangent_reverse | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.31e-16 |  |
| u_retract_zero | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.25e-16 |  |
| u_retract_fd_jacobian | rank1 | uniform+pad | (2, 3) | () | (2,) | None | PASS | 7.14e-06 | ratio=4.00 |
| u_project_ambient | rank1 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 90 into shape (3,2,2,3,5,1) |
| u_transport_identity | rank1 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 90 into shape (3,2,2,3,5,1) |
| u_transport_vs_ragged_projection | rank1 | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 90 into shape (3,2,2,3,5,1) |
| u_to_dense | nonmin | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| u_add | nonmin | uniform | () | () | () | None | PASS | 7.82e-17 |  |
| u_sub | nonmin | uniform | () | () | () | None | PASS | 8.22e-17 |  |
| u_scalar_mul | nonmin | uniform | () | () | () | None | PASS | 1.53e-16 |  |
| u_inner | nonmin | uniform | () | () | () | None | PASS | 2.60e-15 |  |
| u_norm | nonmin | uniform | () | () | () | None | PASS | 1.25e-16 |  |
| u_reverse | nonmin | uniform | () | () | () | None | PASS | 1.80e-16 |  |
| u_t3svd_lossless | nonmin | uniform | () | () | () | None | PASS | 1.03e-15 |  |
| u_rank_adjustment_sweep | nonmin | uniform | () | () | () | None | PASS | 1.02e-15 |  |
| u_t3svd_trunc_vs_ragged | nonmin | uniform | () | () | () | None | PASS | 6.05e-16 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | nonmin | uniform | () | () | () | None | FAIL | 1.06e-02 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | nonmin | uniform | () | () | () | None | PASS | 1.57e-15 |  |
| u_apply | nonmin | uniform | () | () | () | None | PASS | 1.87e-16 |  |
| u_entries | nonmin | uniform | () | () | () | None | PASS | 2.16e-16 |  |
| u_probe | nonmin | uniform | () | () | () | None | PASS | 9.91e-16 |  |
| u_apply_derivatives | nonmin | uniform | () | () | () | None | PASS | 3.21e-16 |  |
| u_entries_derivatives | nonmin | uniform | () | () | () | None | PASS | 3.31e-16 |  |
| u_probe_derivatives | nonmin | uniform | () | () | () | None | PASS | 1.07e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| utv_apply | nonmin | uniform | () | () | () | None | PASS | 2.15e-16 |  |
| utv_entries | nonmin | uniform | () | () | () | None | PASS | 5.64e-16 |  |
| utv_probe | nonmin | uniform | () | () | () | None | PASS | 6.47e-16 |  |
| utv_apply_derivatives | nonmin | uniform | () | () | () | None | PASS | 1.50e-15 |  |
| utv_entries_derivatives | nonmin | uniform | () | () | () | None | PASS | 1.48e-15 |  |
| utv_probe_derivatives | nonmin | uniform | () | () | () | None | PASS | 4.98e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | () | () | () | None | PASS | 7.74e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | () | () | () | None | PASS | 3.39e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | () | () | () | None | PASS | 7.74e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | () | () | () | None | PASS | 3.39e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | () | () | None | PASS | 3.03e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | () | () | None | PASS | 2.41e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | () | () | None | PASS | 3.03e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | () | () | None | PASS | 2.41e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | () | () | None | PASS | 0.00e+00 |  |
| utv_apply | nonmin | uniform | () | () | (2,) | None | PASS | 6.69e-16 |  |
| utv_entries | nonmin | uniform | () | () | (2,) | None | PASS | 1.05e-16 |  |
| utv_probe | nonmin | uniform | () | () | (2,) | None | PASS | 6.76e-16 |  |
| utv_apply_derivatives | nonmin | uniform | () | () | (2,) | None | PASS | 1.45e-16 |  |
| utv_entries_derivatives | nonmin | uniform | () | () | (2,) | None | PASS | 1.47e-16 |  |
| utv_probe_derivatives | nonmin | uniform | () | () | (2,) | None | PASS | 2.82e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | () | () | (2,) | None | PASS | 3.43e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | () | () | (2,) | None | PASS | 3.43e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | () | (2,) | None | PASS | 2.19e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | () | (2,) | None | PASS | 1.57e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | () | (2,) | None | PASS | 2.27e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | () | (2,) | None | PASS | 2.19e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | () | (2,) | None | PASS | 1.57e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | () | (2,) | None | PASS | 2.27e-16 |  |
| u_apply | nonmin | uniform | () | (3,) | () | None | PASS | 2.02e-15 |  |
| u_entries | nonmin | uniform | () | (3,) | () | None | PASS | 1.45e-15 |  |
| u_probe | nonmin | uniform | () | (3,) | () | None | PASS | 1.82e-15 |  |
| u_apply_derivatives | nonmin | uniform | () | (3,) | () | None | PASS | 4.83e-15 |  |
| u_entries_derivatives | nonmin | uniform | () | (3,) | () | None | PASS | 3.79e-15 |  |
| u_probe_derivatives | nonmin | uniform | () | (3,) | () | None | PASS | 1.72e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| utv_apply | nonmin | uniform | () | (3,) | () | None | PASS | 1.41e-15 |  |
| utv_entries | nonmin | uniform | () | (3,) | () | None | PASS | 2.62e-16 |  |
| utv_probe | nonmin | uniform | () | (3,) | () | None | PASS | 7.77e-16 |  |
| utv_apply_derivatives | nonmin | uniform | () | (3,) | () | None | PASS | 6.21e-16 |  |
| utv_entries_derivatives | nonmin | uniform | () | (3,) | () | None | PASS | 1.15e-15 |  |
| utv_probe_derivatives | nonmin | uniform | () | (3,) | () | None | PASS | 3.99e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | () | None | PASS | 8.57e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | () | None | PASS | 1.51e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | () | None | PASS | 1.87e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | () | None | PASS | 6.42e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | () | None | PASS | 4.52e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | () | None | PASS | 5.61e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | () | None | PASS | 1.38e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | () | None | PASS | 1.12e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | () | None | PASS | 4.79e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | () | None | PASS | 1.38e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | () | None | PASS | 1.87e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | () | None | PASS | 3.59e-16 |  |
| utv_apply | nonmin | uniform | () | (3,) | (2,) | None | PASS | 4.07e-16 |  |
| utv_entries | nonmin | uniform | () | (3,) | (2,) | None | PASS | 1.61e-16 |  |
| utv_probe | nonmin | uniform | () | (3,) | (2,) | None | PASS | 3.34e-16 |  |
| utv_apply_derivatives | nonmin | uniform | () | (3,) | (2,) | None | PASS | 2.91e-16 |  |
| utv_entries_derivatives | nonmin | uniform | () | (3,) | (2,) | None | PASS | 3.16e-16 |  |
| utv_probe_derivatives | nonmin | uniform | () | (3,) | (2,) | None | PASS | 3.02e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 1.59e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 8.39e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 3.59e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 1.59e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 8.39e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 4.06e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 3.93e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 1.45e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 2.62e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (3,) | (2,) | None | PASS | 1.45e-16 |  |
| u_apply | nonmin | uniform | () | (2, 2) | () | None | PASS | 7.21e-16 |  |
| u_entries | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.86e-15 |  |
| u_probe | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.12e-15 |  |
| u_apply_derivatives | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.20e-15 |  |
| u_entries_derivatives | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.34e-15 |  |
| u_probe_derivatives | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.39e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| utv_apply | nonmin | uniform | () | (2, 2) | () | None | PASS | 5.40e-16 |  |
| utv_entries | nonmin | uniform | () | (2, 2) | () | None | PASS | 8.05e-17 |  |
| utv_probe | nonmin | uniform | () | (2, 2) | () | None | PASS | 4.33e-16 |  |
| utv_apply_derivatives | nonmin | uniform | () | (2, 2) | () | None | PASS | 2.96e-16 |  |
| utv_entries_derivatives | nonmin | uniform | () | (2, 2) | () | None | PASS | 4.43e-16 |  |
| utv_probe_derivatives | nonmin | uniform | () | (2, 2) | () | None | PASS | 3.40e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | () | None | PASS | 5.94e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.14e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.11e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.49e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.14e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | () | None | PASS | 3.70e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.42e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | () | None | PASS | 3.21e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | () | None | PASS | 2.46e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | () | None | PASS | 2.84e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | () | None | PASS | 3.21e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | () | None | PASS | 1.23e-16 |  |
| utv_apply | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 3.17e-16 |  |
| utv_entries | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 9.05e-17 |  |
| utv_probe | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 3.39e-16 |  |
| utv_apply_derivatives | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 4.18e-16 |  |
| utv_entries_derivatives | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 3.99e-16 |  |
| utv_probe_derivatives | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 3.42e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 2.18e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 4.83e-14 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 1.16e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 2.18e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 1.69e-14 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 6.43e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 1.98e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 3.35e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 5.14e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 3.97e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | () | (2, 2) | (2,) | None | PASS | 7.82e-16 |  |
| u_manifold_inner | nonmin | uniform | () | () | () | None | PASS | 9.14e-16 |  |
| u_manifold_norm | nonmin | uniform | () | () | () | None | PASS | 5.13e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform | () | () | () | None | PASS | 2.65e-16 |  |
| u_tangent_add_scale | nonmin | uniform | () | () | () | None | PASS | 1.50e-16 |  |
| u_tangent_reverse | nonmin | uniform | () | () | () | None | PASS | 1.52e-16 |  |
| u_retract_zero | nonmin | uniform | () | () | () | None | PASS | 1.29e-15 |  |
| u_retract_fd_jacobian | nonmin | uniform | () | () | () | None | PASS | 5.83e-07 | ratio=4.00 |
| u_retract_vs_ragged | nonmin | uniform | () | () | () | None | PASS | 2.15e-15 |  |
| u_project_ambient | nonmin | uniform | () | () | () | None | PASS | 7.88e-16 |  |
| u_transport_identity | nonmin | uniform | () | () | () | None | PASS | 1.20e-15 |  |
| u_transport_vs_ragged_projection | nonmin | uniform | () | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | nonmin | uniform | () | () | (2,) | None | PASS | 2.00e-15 |  |
| u_manifold_norm | nonmin | uniform | () | () | (2,) | None | PASS | 2.90e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform | () | () | (2,) | None | PASS | 2.32e-16 |  |
| u_tangent_add_scale | nonmin | uniform | () | () | (2,) | None | PASS | 2.50e-16 |  |
| u_tangent_reverse | nonmin | uniform | () | () | (2,) | None | PASS | 1.70e-16 |  |
| u_retract_zero | nonmin | uniform | () | () | (2,) | None | PASS | 1.29e-15 |  |
| u_retract_fd_jacobian | nonmin | uniform | () | () | (2,) | None | PASS | 1.97e-06 | ratio=4.00 |
| u_project_ambient | nonmin | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 9 into shape (3,2,3,1) |
| u_transport_identity | nonmin | uniform | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 9 into shape (3,2,3,1) |
| u_transport_vs_ragged_projection | nonmin | uniform | () | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | nonmin | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_add | nonmin | uniform | (2,) | () | () | None | PASS | 1.00e-16 |  |
| u_sub | nonmin | uniform | (2,) | () | () | None | PASS | 8.09e-17 |  |
| u_scalar_mul | nonmin | uniform | (2,) | () | () | None | PASS | 1.50e-16 |  |
| u_inner | nonmin | uniform | (2,) | () | () | None | PASS | 6.45e-15 |  |
| u_norm | nonmin | uniform | (2,) | () | () | None | PASS | 6.76e-16 |  |
| u_reverse | nonmin | uniform | (2,) | () | () | None | PASS | 1.51e-16 |  |
| u_t3svd_lossless | nonmin | uniform | (2,) | () | () | None | PASS | 1.71e-15 |  |
| u_rank_adjustment_sweep | nonmin | uniform | (2,) | () | () | None | PASS | 1.55e-15 |  |
| u_t3svd_trunc_vs_ragged | nonmin | uniform | (2,) | () | () | None | PASS | 1.16e-15 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | nonmin | uniform | (2,) | () | () | None | FAIL | 9.70e-01 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | nonmin | uniform | (2,) | () | () | None | PASS | 1.78e-15 |  |
| u_apply | nonmin | uniform | (2,) | () | () | None | PASS | 4.73e-16 |  |
| u_entries | nonmin | uniform | (2,) | () | () | None | PASS | 2.30e-15 |  |
| u_probe | nonmin | uniform | (2,) | () | () | None | PASS | 1.40e-15 |  |
| u_apply_derivatives | nonmin | uniform | (2,) | () | () | None | PASS | 1.03e-15 |  |
| u_entries_derivatives | nonmin | uniform | (2,) | () | () | None | PASS | 2.50e-15 |  |
| u_probe_derivatives | nonmin | uniform | (2,) | () | () | None | PASS | 1.37e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| utv_apply | nonmin | uniform | (2,) | () | () | None | PASS | 1.72e-15 |  |
| utv_entries | nonmin | uniform | (2,) | () | () | None | PASS | 2.06e-16 |  |
| utv_probe | nonmin | uniform | (2,) | () | () | None | PASS | 3.93e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2,) | () | () | None | PASS | 1.08e-15 |  |
| utv_entries_derivatives | nonmin | uniform | (2,) | () | () | None | PASS | 1.18e-15 |  |
| utv_probe_derivatives | nonmin | uniform | (2,) | () | () | None | PASS | 5.10e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | () | None | PASS | 1.94e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | () | None | PASS | 4.44e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | () | None | PASS | 1.94e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | () | None | PASS | 4.44e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | () | None | PASS | 1.35e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | () | None | PASS | 2.87e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | () | None | PASS | 1.35e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | () | None | PASS | 2.87e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply | nonmin | uniform | (2,) | () | (2,) | None | PASS | 3.25e-16 |  |
| utv_entries | nonmin | uniform | (2,) | () | (2,) | None | PASS | 3.47e-16 |  |
| utv_probe | nonmin | uniform | (2,) | () | (2,) | None | PASS | 3.70e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2,) | () | (2,) | None | PASS | 2.69e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2,) | () | (2,) | None | PASS | 2.67e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2,) | () | (2,) | None | PASS | 3.68e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 4.87e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 4.87e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 1.84e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 4.38e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 1.84e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 4.38e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | nonmin | uniform | (2,) | (3,) | () | None | PASS | 4.11e-16 |  |
| u_entries | nonmin | uniform | (2,) | (3,) | () | None | PASS | 1.80e-15 |  |
| u_probe | nonmin | uniform | (2,) | (3,) | () | None | PASS | 1.69e-15 |  |
| u_apply_derivatives | nonmin | uniform | (2,) | (3,) | () | None | PASS | 6.93e-16 |  |
| u_entries_derivatives | nonmin | uniform | (2,) | (3,) | () | None | PASS | 6.34e-16 |  |
| u_probe_derivatives | nonmin | uniform | (2,) | (3,) | () | None | PASS | 1.33e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| utv_apply | nonmin | uniform | (2,) | (3,) | () | None | PASS | 5.40e-16 |  |
| utv_entries | nonmin | uniform | (2,) | (3,) | () | None | PASS | 1.92e-16 |  |
| utv_probe | nonmin | uniform | (2,) | (3,) | () | None | PASS | 3.90e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2,) | (3,) | () | None | PASS | 3.31e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2,) | (3,) | () | None | PASS | 2.62e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2,) | (3,) | () | None | PASS | 3.73e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 2.06e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 8.01e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 1.33e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 4.12e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 9.62e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 1.33e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 4.96e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 1.53e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 2.14e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 4.96e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 1.53e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | () | None | PASS | 4.28e-16 |  |
| utv_apply | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 1.96e-16 |  |
| utv_entries | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 2.15e-16 |  |
| utv_probe | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 2.71e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 6.61e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 8.23e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 4.13e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 1.54e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 2.82e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 2.24e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 5.64e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 2.24e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 1.68e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 1.84e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 5.03e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 3.69e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| u_apply | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 1.00e-15 |  |
| u_entries | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 1.21e-15 |  |
| u_probe | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 1.40e-15 |  |
| u_apply_derivatives | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 1.45e-15 |  |
| u_entries_derivatives | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 1.31e-15 |  |
| u_probe_derivatives | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 1.50e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| utv_apply | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 3.55e-16 |  |
| utv_entries | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 2.01e-16 |  |
| utv_probe | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 4.17e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 3.43e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 2.97e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 2.90e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 3.56e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 1.62e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 5.35e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 1.33e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 1.43e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 1.33e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.76e-16 |  |
| utv_entries | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.18e-16 |  |
| utv_probe | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.17e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.84e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 4.21e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.91e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.86e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.20e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.70e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 1.86e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 5.77e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.35e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 5.73e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 8.65e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 6.27e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.86e-16 |  |
| u_manifold_inner | nonmin | uniform | (2,) | () | () | None | PASS | 7.22e-16 |  |
| u_manifold_norm | nonmin | uniform | (2,) | () | () | None | PASS | 1.41e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform | (2,) | () | () | None | PASS | 2.57e-16 |  |
| u_tangent_add_scale | nonmin | uniform | (2,) | () | () | None | PASS | 1.90e-16 |  |
| u_tangent_reverse | nonmin | uniform | (2,) | () | () | None | PASS | 1.40e-16 |  |
| u_retract_zero | nonmin | uniform | (2,) | () | () | None | PASS | 1.12e-15 |  |
| u_retract_fd_jacobian | nonmin | uniform | (2,) | () | () | None | PASS | 5.32e-07 | ratio=4.00 |
| u_retract_vs_ragged | nonmin | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | nonmin | uniform | (2,) | () | () | None | PASS | 1.10e-15 |  |
| u_transport_identity | nonmin | uniform | (2,) | () | () | None | PASS | 5.88e-16 |  |
| u_transport_vs_ragged_projection | nonmin | uniform | (2,) | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | nonmin | uniform | (2,) | () | (2,) | None | PASS | 4.80e-16 |  |
| u_manifold_norm | nonmin | uniform | (2,) | () | (2,) | None | PASS | 1.27e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform | (2,) | () | (2,) | None | PASS | 3.54e-16 |  |
| u_tangent_add_scale | nonmin | uniform | (2,) | () | (2,) | None | PASS | 1.87e-16 |  |
| u_tangent_reverse | nonmin | uniform | (2,) | () | (2,) | None | PASS | 1.49e-16 |  |
| u_retract_zero | nonmin | uniform | (2,) | () | (2,) | None | PASS | 1.12e-15 |  |
| u_retract_fd_jacobian | nonmin | uniform | (2,) | () | (2,) | None | PASS | 3.88e-07 | ratio=4.00 |
| u_project_ambient | nonmin | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1) |
| u_transport_identity | nonmin | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1) |
| u_transport_vs_ragged_projection | nonmin | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | nonmin | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_add | nonmin | uniform | (2, 3) | () | () | None | PASS | 9.33e-17 |  |
| u_sub | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.07e-16 |  |
| u_scalar_mul | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.35e-16 |  |
| u_inner | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.67e-15 |  |
| u_norm | nonmin | uniform | (2, 3) | () | () | None | PASS | 4.33e-16 |  |
| u_reverse | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.39e-16 |  |
| u_t3svd_lossless | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.62e-15 |  |
| u_rank_adjustment_sweep | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.41e-15 |  |
| u_t3svd_trunc_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.26e-15 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | nonmin | uniform | (2, 3) | () | () | None | FAIL | 8.47e-01 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.62e-15 |  |
| u_apply | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.16e-15 |  |
| u_entries | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.42e-15 |  |
| u_probe | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.60e-15 |  |
| u_apply_derivatives | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.69e-15 |  |
| u_entries_derivatives | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.68e-15 |  |
| u_probe_derivatives | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.55e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| utv_apply | nonmin | uniform | (2, 3) | () | () | None | PASS | 2.34e-16 |  |
| utv_entries | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.57e-16 |  |
| utv_probe | nonmin | uniform | (2, 3) | () | () | None | PASS | 3.74e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2, 3) | () | () | None | PASS | 3.70e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2, 3) | () | () | None | PASS | 3.83e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2, 3) | () | () | None | PASS | 3.30e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | () | None | PASS | 2.33e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.65e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | () | None | PASS | 3.32e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | () | None | PASS | 2.33e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.65e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | () | None | PASS | 3.32e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | () | None | PASS | 4.62e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | () | None | PASS | 2.25e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | () | None | PASS | 5.23e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | () | None | PASS | 4.62e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | () | None | PASS | 2.25e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | () | None | PASS | 5.23e-16 |  |
| utv_apply | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 3.71e-16 |  |
| utv_entries | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 2.17e-16 |  |
| utv_probe | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 3.74e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 3.57e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 3.67e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 2.88e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 3.45e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 1.23e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 6.25e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 3.45e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 1.23e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 6.25e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 4.66e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 2.54e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 4.84e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 4.66e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 2.54e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 4.84e-16 |  |
| u_apply | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.89e-15 |  |
| u_entries | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.53e-15 |  |
| u_probe | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.74e-15 |  |
| u_apply_derivatives | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.60e-15 |  |
| u_entries_derivatives | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.73e-15 |  |
| u_probe_derivatives | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.56e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| utv_apply | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 3.23e-16 |  |
| utv_entries | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 2.96e-16 |  |
| utv_probe | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 3.40e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.86e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.30e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 3.06e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.31e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.28e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 2.71e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 2.62e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.28e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 2.13e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 1.29e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | () | None | PASS | 5.15e-16 |  |
| utv_apply | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.73e-16 |  |
| utv_entries | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.81e-16 |  |
| utv_probe | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.08e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.34e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.94e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.39e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 2.29e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.38e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.38e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 3.71e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.13e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 4.95e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (3,) | (2,) | None | PASS | 1.71e-16 |  |
| u_apply | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.83e-15 |  |
| u_entries | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.76e-15 |  |
| u_probe | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.72e-15 |  |
| u_apply_derivatives | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.85e-15 |  |
| u_entries_derivatives | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.08e-15 |  |
| u_probe_derivatives | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.83e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| utv_apply | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 5.32e-16 |  |
| utv_entries | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.69e-16 |  |
| utv_probe | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.57e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.33e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.71e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.51e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 2.19e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.17e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 6.25e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 6.25e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.75e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 7.50e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 3.00e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.23e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 4.28e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.13e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 1.23e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 4.93e-16 |  |
| utv_entries | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.15e-16 |  |
| utv_probe | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.34e-16 |  |
| utv_apply_derivatives | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.05e-16 |  |
| utv_entries_derivatives | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.79e-16 |  |
| utv_probe_derivatives | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 3.18e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 7.27e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.63e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 7.27e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.63e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 4.79e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 1.60e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 2.14e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform | (2, 3) | (2, 2) | (2,) | None | PASS | 4.47e-15 |  |
| u_manifold_inner | nonmin | uniform | (2, 3) | () | () | None | PASS | 9.49e-16 |  |
| u_manifold_norm | nonmin | uniform | (2, 3) | () | () | None | PASS | 4.03e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform | (2, 3) | () | () | None | PASS | 3.50e-16 |  |
| u_tangent_add_scale | nonmin | uniform | (2, 3) | () | () | None | PASS | 2.21e-16 |  |
| u_tangent_reverse | nonmin | uniform | (2, 3) | () | () | None | PASS | 1.48e-16 |  |
| u_retract_zero | nonmin | uniform | (2, 3) | () | () | None | PASS | 2.02e-15 |  |
| u_retract_fd_jacobian | nonmin | uniform | (2, 3) | () | () | None | PASS | 2.41e-05 | ratio=4.00 |
| u_retract_vs_ragged | nonmin | uniform | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | nonmin | uniform | (2, 3) | () | () | None | PASS | 8.17e-16 |  |
| u_transport_identity | nonmin | uniform | (2, 3) | () | () | None | PASS | 8.21e-16 |  |
| u_transport_vs_ragged_projection | nonmin | uniform | (2, 3) | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 1.45e-15 |  |
| u_manifold_norm | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 3.16e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 3.55e-16 |  |
| u_tangent_add_scale | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 2.01e-16 |  |
| u_tangent_reverse | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 1.54e-16 |  |
| u_retract_zero | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 2.02e-15 |  |
| u_retract_fd_jacobian | nonmin | uniform | (2, 3) | () | (2,) | None | PASS | 7.26e-06 | ratio=4.00 |
| u_project_ambient | nonmin | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 54 into shape (3,2,2,3,3,1) |
| u_transport_identity | nonmin | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 54 into shape (3,2,2,3,3,1) |
| u_transport_vs_ragged_projection | nonmin | uniform | (2, 3) | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | nonmin | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| u_add | nonmin | uniform+pad | () | () | () | None | PASS | 7.82e-17 |  |
| u_sub | nonmin | uniform+pad | () | () | () | None | PASS | 8.22e-17 |  |
| u_scalar_mul | nonmin | uniform+pad | () | () | () | None | PASS | 1.53e-16 |  |
| u_inner | nonmin | uniform+pad | () | () | () | None | PASS | 3.03e-15 |  |
| u_norm | nonmin | uniform+pad | () | () | () | None | PASS | 3.75e-16 |  |
| u_reverse | nonmin | uniform+pad | () | () | () | None | PASS | 1.80e-16 |  |
| u_t3svd_lossless | nonmin | uniform+pad | () | () | () | None | PASS | 1.28e-15 |  |
| u_rank_adjustment_sweep | nonmin | uniform+pad | () | () | () | None | PASS | 1.12e-15 |  |
| u_t3svd_trunc_vs_ragged | nonmin | uniform+pad | () | () | () | None | PASS | 9.47e-16 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | nonmin | uniform+pad | () | () | () | None | FAIL | 1.00e+00 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | nonmin | uniform+pad | () | () | () | None | PASS | 8.57e-16 |  |
| u_apply | nonmin | uniform+pad | () | () | () | None | PASS | 3.74e-16 |  |
| u_entries | nonmin | uniform+pad | () | () | () | None | PASS | 4.32e-16 |  |
| u_probe | nonmin | uniform+pad | () | () | () | None | PASS | 9.41e-16 |  |
| u_apply_derivatives | nonmin | uniform+pad | () | () | () | None | PASS | 3.09e-16 |  |
| u_entries_derivatives | nonmin | uniform+pad | () | () | () | None | PASS | 3.38e-16 |  |
| u_probe_derivatives | nonmin | uniform+pad | () | () | () | None | PASS | 1.13e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| utv_apply | nonmin | uniform+pad | () | () | () | None | PASS | 3.43e-16 |  |
| utv_entries | nonmin | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe | nonmin | uniform+pad | () | () | () | None | PASS | 4.53e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | () | () | () | None | PASS | 1.94e-15 |  |
| utv_entries_derivatives | nonmin | uniform+pad | () | () | () | None | PASS | 7.50e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | () | () | () | None | PASS | 1.73e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | () | None | PASS | 1.15e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | () | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | () | None | PASS | 1.15e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | () | None | PASS | 6.56e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | () | None | PASS | 4.02e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | () | None | PASS | 1.70e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | () | None | PASS | 6.56e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | () | None | PASS | 4.02e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | () | None | PASS | 1.70e-16 |  |
| utv_apply | nonmin | uniform+pad | () | () | (2,) | None | PASS | 7.00e-16 |  |
| utv_entries | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.56e-16 |  |
| utv_probe | nonmin | uniform+pad | () | () | (2,) | None | PASS | 6.74e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | () | () | (2,) | None | PASS | 4.14e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | () | () | (2,) | None | PASS | 5.08e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | () | () | (2,) | None | PASS | 5.22e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.12e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.32e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.12e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.32e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.96e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 6.83e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 6.92e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.96e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 6.83e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | () | (2,) | None | PASS | 6.92e-16 |  |
| u_apply | nonmin | uniform+pad | () | (3,) | () | None | PASS | 1.65e-15 |  |
| u_entries | nonmin | uniform+pad | () | (3,) | () | None | PASS | 1.11e-15 |  |
| u_probe | nonmin | uniform+pad | () | (3,) | () | None | PASS | 8.87e-16 |  |
| u_apply_derivatives | nonmin | uniform+pad | () | (3,) | () | None | PASS | 5.07e-15 |  |
| u_entries_derivatives | nonmin | uniform+pad | () | (3,) | () | None | PASS | 3.94e-15 |  |
| u_probe_derivatives | nonmin | uniform+pad | () | (3,) | () | None | PASS | 1.97e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,1,2,2) (3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| utv_apply | nonmin | uniform+pad | () | (3,) | () | None | PASS | 3.52e-16 |  |
| utv_entries | nonmin | uniform+pad | () | (3,) | () | None | PASS | 7.95e-17 |  |
| utv_probe | nonmin | uniform+pad | () | (3,) | () | None | PASS | 3.03e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | () | (3,) | () | None | PASS | 9.53e-17 |  |
| utv_entries_derivatives | nonmin | uniform+pad | () | (3,) | () | None | PASS | 6.99e-17 |  |
| utv_probe_derivatives | nonmin | uniform+pad | () | (3,) | () | None | PASS | 4.23e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 2.20e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 1.70e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 6.54e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 2.20e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 1.70e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 9.81e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 7.49e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 9.90e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 1.58e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 1.39e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | () | None | PASS | 1.58e-15 |  |
| utv_apply | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 1.66e-16 |  |
| utv_entries | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 3.29e-16 |  |
| utv_probe | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 3.18e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 4.80e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 6.65e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 4.12e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 1.58e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 2.17e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 1.31e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 1.58e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 2.17e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 1.42e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 4.84e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 2.30e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 7.27e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 2.30e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (3,) | (2,) | None | PASS | 1.90e-16 |  |
| u_apply | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.13e-15 |  |
| u_entries | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.37e-15 |  |
| u_probe | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.13e-15 |  |
| u_apply_derivatives | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.54e-15 |  |
| u_entries_derivatives | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.84e-15 |  |
| u_probe_derivatives | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.37e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,1,2,2) (2,2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | () | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (1,2,2) (1,2,4)  |
| utv_apply | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 9.28e-16 |  |
| utv_entries | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 6.53e-17 |  |
| utv_probe | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 3.23e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 4.47e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 3.61e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 3.13e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 7.79e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.38e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.47e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 6.49e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.38e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 7.37e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 4.29e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.32e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 3.93e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 3.37e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 1.32e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | () | None | PASS | 7.86e-16 |  |
| utv_apply | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 6.86e-16 |  |
| utv_entries | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 2.90e-16 |  |
| utv_probe | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.80e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.36e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.52e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.01e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.82e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.77e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.70e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.53e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.77e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 5.69e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.40e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 7.00e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.64e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 1.40e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 5.60e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | () | (2, 2) | (2,) | None | PASS | 3.27e-16 |  |
| u_manifold_inner | nonmin | uniform+pad | () | () | () | None | PASS | 1.49e-16 |  |
| u_manifold_norm | nonmin | uniform+pad | () | () | () | None | PASS | 1.90e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform+pad | () | () | () | None | PASS | 4.23e-16 |  |
| u_tangent_add_scale | nonmin | uniform+pad | () | () | () | None | PASS | 2.65e-16 |  |
| u_tangent_reverse | nonmin | uniform+pad | () | () | () | None | PASS | 1.58e-16 |  |
| u_retract_zero | nonmin | uniform+pad | () | () | () | None | PASS | 1.93e-15 |  |
| u_retract_fd_jacobian | nonmin | uniform+pad | () | () | () | None | PASS | 4.22e-07 | ratio=4.00 |
| u_retract_vs_ragged | nonmin | uniform+pad | () | () | () | None | PASS | 1.19e-15 |  |
| u_project_ambient | nonmin | uniform+pad | () | () | () | None | PASS | 1.37e-15 |  |
| u_transport_identity | nonmin | uniform+pad | () | () | () | None | PASS | 7.65e-16 |  |
| u_transport_vs_ragged_projection | nonmin | uniform+pad | () | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.97e-16 |  |
| u_manifold_norm | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.16e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform+pad | () | () | (2,) | None | PASS | 3.47e-16 |  |
| u_tangent_add_scale | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.74e-16 |  |
| u_tangent_reverse | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.43e-16 |  |
| u_retract_zero | nonmin | uniform+pad | () | () | (2,) | None | PASS | 1.93e-15 |  |
| u_retract_fd_jacobian | nonmin | uniform+pad | () | () | (2,) | None | PASS | 5.39e-07 | ratio=4.00 |
| u_project_ambient | nonmin | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 9 into shape (3,2,3,1) |
| u_transport_identity | nonmin | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 9 into shape (3,2,3,1) |
| u_transport_vs_ragged_projection | nonmin | uniform+pad | () | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | nonmin | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_add | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.00e-16 |  |
| u_sub | nonmin | uniform+pad | (2,) | () | () | None | PASS | 8.09e-17 |  |
| u_scalar_mul | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.50e-16 |  |
| u_inner | nonmin | uniform+pad | (2,) | () | () | None | PASS | 2.38e-14 |  |
| u_norm | nonmin | uniform+pad | (2,) | () | () | None | PASS | 4.74e-16 |  |
| u_reverse | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.51e-16 |  |
| u_t3svd_lossless | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.33e-15 |  |
| u_rank_adjustment_sweep | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.43e-15 |  |
| u_t3svd_trunc_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.42e-15 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | nonmin | uniform+pad | (2,) | () | () | None | FAIL | 1.00e+00 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | nonmin | uniform+pad | (2,) | () | () | None | PASS | 2.04e-15 |  |
| u_apply | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.51e-15 |  |
| u_entries | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.48e-15 |  |
| u_probe | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.66e-15 |  |
| u_apply_derivatives | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.00e-15 |  |
| u_entries_derivatives | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.77e-15 |  |
| u_probe_derivatives | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.58e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| utv_apply | nonmin | uniform+pad | (2,) | () | () | None | PASS | 6.78e-16 |  |
| utv_entries | nonmin | uniform+pad | (2,) | () | () | None | PASS | 2.30e-16 |  |
| utv_probe | nonmin | uniform+pad | (2,) | () | () | None | PASS | 3.59e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2,) | () | () | None | PASS | 4.45e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2,) | () | () | None | PASS | 4.68e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2,) | () | () | None | PASS | 3.69e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 8.15e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.80e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 5.53e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 8.15e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.80e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 5.53e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 3.58e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.35e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 3.58e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.35e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| utv_apply | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 2.24e-16 |  |
| utv_entries | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 2.28e-16 |  |
| utv_probe | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 4.20e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 1.59e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 1.47e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 3.21e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 3.56e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 2.60e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 3.56e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 2.60e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 3.58e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 2.08e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 1.38e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 3.58e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 2.08e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 1.38e-16 |  |
| u_apply | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 5.91e-16 |  |
| u_entries | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 1.70e-15 |  |
| u_probe | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 1.16e-15 |  |
| u_apply_derivatives | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 6.87e-16 |  |
| u_entries_derivatives | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 7.44e-16 |  |
| u_probe_derivatives | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 1.31e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,1,2,2) (3,2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| utv_apply | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 1.41e-16 |  |
| utv_entries | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 2.08e-16 |  |
| utv_probe | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 3.33e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 3.36e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 2.76e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 4.01e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 1.30e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 4.59e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 1.30e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 3.25e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 1.63e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 2.50e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.09e-16 |  |
| utv_entries | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.72e-16 |  |
| utv_probe | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.75e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.90e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.24e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 3.43e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 6.25e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.87e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.52e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.08e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.87e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 1.52e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 2.29e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 4.13e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 4.58e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (3,) | (2,) | None | PASS | 6.88e-16 |  |
| u_apply | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.48e-15 |  |
| u_entries | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.31e-16 |  |
| u_probe | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.69e-15 |  |
| u_apply_derivatives | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.78e-15 |  |
| u_entries_derivatives | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.37e-15 |  |
| u_probe_derivatives | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.71e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,1,2,2) (2,2,2,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2,) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,1,2,2) (2,1,2,4)  |
| utv_apply | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.24e-16 |  |
| utv_entries | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 7.19e-17 |  |
| utv_probe | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.16e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.15e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.61e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.69e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.02e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 6.46e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 6.89e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.02e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.58e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 7.96e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 1.23e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 3.98e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 4.39e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | () | None | PASS | 2.47e-16 |  |
| utv_apply | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.12e-16 |  |
| utv_entries | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.08e-16 |  |
| utv_probe | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.44e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 4.11e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.65e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 4.66e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 4.51e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 4.36e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.31e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 3.38e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 5.82e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 7.72e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 9.67e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.98e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 2.57e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2,) | (2, 2) | (2,) | None | PASS | 1.98e-16 |  |
| u_manifold_inner | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.09e-15 |  |
| u_manifold_norm | nonmin | uniform+pad | (2,) | () | () | None | PASS | 3.36e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform+pad | (2,) | () | () | None | PASS | 2.67e-16 |  |
| u_tangent_add_scale | nonmin | uniform+pad | (2,) | () | () | None | PASS | 2.69e-16 |  |
| u_tangent_reverse | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.46e-16 |  |
| u_retract_zero | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.03e-15 |  |
| u_retract_fd_jacobian | nonmin | uniform+pad | (2,) | () | () | None | PASS | 3.74e-07 | ratio=4.00 |
| u_retract_vs_ragged | nonmin | uniform+pad | (2,) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | nonmin | uniform+pad | (2,) | () | () | None | PASS | 1.57e-16 |  |
| u_transport_identity | nonmin | uniform+pad | (2,) | () | () | None | PASS | 5.35e-16 |  |
| u_transport_vs_ragged_projection | nonmin | uniform+pad | (2,) | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 1.01e-15 |  |
| u_manifold_norm | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 2.83e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 3.45e-16 |  |
| u_tangent_add_scale | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 2.20e-16 |  |
| u_tangent_reverse | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 1.53e-16 |  |
| u_retract_zero | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 1.03e-15 |  |
| u_retract_fd_jacobian | nonmin | uniform+pad | (2,) | () | (2,) | None | PASS | 5.31e-07 | ratio=4.00 |
| u_project_ambient | nonmin | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1) |
| u_transport_identity | nonmin | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1) |
| u_transport_vs_ragged_projection | nonmin | uniform+pad | (2,) | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_add | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 9.33e-17 |  |
| u_sub | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.07e-16 |  |
| u_scalar_mul | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.35e-16 |  |
| u_inner | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.46e-15 |  |
| u_norm | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 4.30e-16 |  |
| u_reverse | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.39e-16 |  |
| u_t3svd_lossless | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.67e-15 |  |
| u_rank_adjustment_sweep | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.56e-15 |  |
| u_t3svd_trunc_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.28e-15 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | nonmin | uniform+pad | (2, 3) | () | () | None | FAIL | 1.00e+00 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.74e-15 |  |
| u_apply | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.83e-15 |  |
| u_entries | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.27e-15 |  |
| u_probe | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.82e-15 |  |
| u_apply_derivatives | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.65e-15 |  |
| u_entries_derivatives | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.64e-15 |  |
| u_probe_derivatives | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.78e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| utv_apply | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 6.25e-16 |  |
| utv_entries | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 2.07e-16 |  |
| utv_probe | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 4.05e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 4.85e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 5.17e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 4.15e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 3.40e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.31e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 9.02e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 3.40e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.31e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 9.02e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 2.11e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.31e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 2.94e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 2.11e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.31e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 2.94e-16 |  |
| utv_apply | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.76e-16 |  |
| utv_entries | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.91e-16 |  |
| utv_probe | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.57e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.56e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 4.21e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.61e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.51e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.28e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.51e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.28e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.35e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.82e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.81e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.35e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.82e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.81e-15 |  |
| u_apply | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.99e-15 |  |
| u_entries | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.91e-15 |  |
| u_probe | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.74e-15 |  |
| u_apply_derivatives | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.95e-15 |  |
| u_entries_derivatives | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.02e-15 |  |
| u_probe_derivatives | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.04e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,1,2,2) (3,2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (3,) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| utv_apply | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 4.86e-16 |  |
| utv_entries | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.02e-16 |  |
| utv_probe | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.70e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.80e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.50e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.57e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 3.86e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.33e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.54e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 8.85e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 1.76e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.20e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 2.14e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 6.81e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | () | None | PASS | 5.11e-16 |  |
| utv_apply | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.21e-16 |  |
| utv_entries | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.22e-16 |  |
| utv_probe | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.28e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.60e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.42e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 3.92e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.11e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.49e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.21e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 6.14e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.30e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.77e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.54e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 2.60e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (3,) | (2,) | None | PASS | 1.77e-16 |  |
| u_apply | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.73e-15 |  |
| u_entries | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.12e-15 |  |
| u_probe | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.89e-15 |  |
| u_apply_derivatives | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.85e-15 |  |
| u_entries_derivatives | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.19e-15 |  |
| u_probe_derivatives | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.71e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,1,2,2) (2,2,2,3,1,2,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,1,2,2) (2,3,1,2,4)  |
| utv_apply | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.24e-16 |  |
| utv_entries | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.04e-16 |  |
| utv_probe | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.27e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 4.15e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.31e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.38e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 9.71e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.65e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.43e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 3.30e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.79e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 7.73e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 9.95e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 4.48e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 2.49e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | () | None | PASS | 1.54e-16 |  |
| utv_apply | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.84e-16 |  |
| utv_entries | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.16e-16 |  |
| utv_probe | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.31e-16 |  |
| utv_apply_derivatives | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 4.01e-16 |  |
| utv_entries_derivatives | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.06e-16 |  |
| utv_probe_derivatives | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 3.31e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 8.70e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.52e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 5.95e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.16e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 9.09e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.98e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 1.29e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.03e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | nonmin | uniform+pad | (2, 3) | (2, 2) | (2,) | None | PASS | 2.03e-16 |  |
| u_manifold_inner | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.90e-15 |  |
| u_manifold_norm | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 2.63e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 3.69e-16 |  |
| u_tangent_add_scale | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 2.05e-16 |  |
| u_tangent_reverse | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 1.39e-16 |  |
| u_retract_zero | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 8.85e-16 |  |
| u_retract_fd_jacobian | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 3.56e-06 | ratio=4.00 |
| u_retract_vs_ragged | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 0.00e+00 |  |
| u_project_ambient | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 8.60e-16 |  |
| u_transport_identity | nonmin | uniform+pad | (2, 3) | () | () | None | PASS | 6.04e-16 |  |
| u_transport_vs_ragged_projection | nonmin | uniform+pad | (2, 3) | () | () | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_manifold_inner | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 9.09e-16 |  |
| u_manifold_norm | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.45e-16 |  |
| u_gauge_project_idempotent | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 3.87e-16 |  |
| u_tangent_add_scale | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 2.06e-16 |  |
| u_tangent_reverse | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 1.43e-16 |  |
| u_retract_zero | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 8.85e-16 |  |
| u_retract_fd_jacobian | nonmin | uniform+pad | (2, 3) | () | (2,) | None | PASS | 9.45e-06 | ratio=4.00 |
| u_project_ambient | nonmin | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 54 into shape (3,2,2,3,3,1) |
| u_transport_identity | nonmin | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: cannot reshape array of size 54 into shape (3,2,2,3,3,1) |
| u_transport_vs_ragged_projection | nonmin | uniform+pad | (2, 3) | () | (2,) | None | EXC | nan | ValueError: UniformManifoldGeometry.project_ambient requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the Hilbert-Schmidt-orthogonal proj |
| u_to_dense | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.22e-17 |  |
| u_add | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 4.88e-17 |  |
| u_sub | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 6.48e-17 |  |
| u_scalar_mul | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.14e-16 |  |
| u_inner | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.72e-15 |  |
| u_norm | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 2.76e-16 |  |
| u_reverse | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.19e-16 |  |
| u_t3svd_lossless | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 9.73e-16 |  |
| u_rank_adjustment_sweep | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.48e-15 |  |
| u_t3svd_trunc_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.03e-14 |  |
| u_orthogonal_representations | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 5.67e-16 |  |
| u_apply | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.14e-15 |  |
| u_entries | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 3.32e-16 |  |
| u_probe | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 2.95e-16 |  |
| u_apply_derivatives | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 5.15e-16 |  |
| u_entries_derivatives | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 5.97e-16 |  |
| u_probe_derivatives | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 3.41e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| utv_apply | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 4.96e-16 |  |
| utv_entries | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 4.39e-16 |  |
| utv_probe | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 3.31e-16 |  |
| utv_apply_derivatives | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.72e-16 |  |
| utv_entries_derivatives | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 2.51e-17 |  |
| utv_probe_derivatives | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 5.36e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.76e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.76e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 2.12e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 3.34e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 2.12e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 3.34e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 6.08e-16 |  |
| utv_entries | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 2.93e-16 |  |
| utv_probe | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 3.60e-16 |  |
| utv_apply_derivatives | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 5.84e-16 |  |
| utv_entries_derivatives | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 6.33e-16 |  |
| utv_probe_derivatives | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 4.40e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 2.38e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 1.64e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 5.79e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 2.38e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 1.64e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 5.79e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 3.06e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 2.78e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 1.78e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 3.06e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 2.78e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 1.78e-16 |  |
| u_apply | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 3.96e-16 |  |
| u_entries | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 2.16e-16 |  |
| u_probe | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 2.72e-16 |  |
| u_apply_derivatives | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 5.20e-16 |  |
| u_entries_derivatives | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 2.91e-16 |  |
| u_probe_derivatives | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 3.88e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| utv_apply | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 5.27e-16 |  |
| utv_entries | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 2.07e-16 |  |
| utv_probe | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 5.57e-16 |  |
| utv_apply_derivatives | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 2.73e-16 |  |
| utv_entries_derivatives | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 1.99e-16 |  |
| utv_probe_derivatives | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 4.41e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 2.22e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 5.65e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 9.69e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 2.22e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 6.78e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 9.69e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 1.11e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 6.37e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 1.23e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 6.37e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | () | (0, 0, 1) | PASS | 1.17e-14 |  |
| utv_apply | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 6.60e-16 |  |
| utv_entries | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 1.25e-16 |  |
| utv_probe | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 5.81e-16 |  |
| utv_apply_derivatives | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 4.99e-16 |  |
| utv_entries_derivatives | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 5.08e-16 |  |
| utv_probe_derivatives | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 4.70e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 1.04e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 7.14e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 1.11e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 1.81e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 5.36e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 7.37e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 6.12e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 3.40e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 1.44e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 6.12e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (3,) | (2,) | (0, 0, 1) | PASS | 2.87e-16 |  |
| u_apply | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 8.41e-16 |  |
| u_entries | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 1.48e-16 |  |
| u_probe | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 3.11e-16 |  |
| u_apply_derivatives | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 5.73e-16 |  |
| u_entries_derivatives | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 1.18e-15 |  |
| u_probe_derivatives | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 3.67e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| utv_apply | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 4.58e-16 |  |
| utv_entries | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 1.50e-16 |  |
| utv_probe | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 4.37e-16 |  |
| utv_apply_derivatives | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 7.51e-16 |  |
| utv_entries_derivatives | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 1.05e-15 |  |
| utv_probe_derivatives | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 3.55e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 1.90e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 1.51e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 3.30e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 1.90e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 1.51e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 3.30e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 7.78e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 1.54e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 1.78e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 5.56e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 6.16e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.50e-16 |  |
| utv_entries | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.64e-16 |  |
| utv_probe | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.50e-16 |  |
| utv_apply_derivatives | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 5.05e-16 |  |
| utv_entries_derivatives | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.72e-16 |  |
| utv_probe_derivatives | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.49e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.39e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 9.47e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.78e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.37e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.49e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.08e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.17e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.13e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_manifold_inner | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 3.74e-15 |  |
| u_manifold_norm | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_gauge_project_idempotent | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 4.45e-16 |  |
| u_tangent_add_scale | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.97e-16 |  |
| u_tangent_reverse | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.93e-16 |  |
| u_retract_zero | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.20e-15 |  |
| u_retract_fd_jacobian | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.46e-05 | ratio=4.00 |
| u_retract_vs_ragged | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.48e-15 |  |
| u_project_ambient | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 1.09e-15 |  |
| u_transport_identity | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 9.38e-16 |  |
| u_transport_vs_ragged_projection | sh2 | uniform | () | () | () | (0, 0, 1) | PASS | 3.87e-16 |  |
| u_manifold_inner | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 1.50e-15 |  |
| u_manifold_norm | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 2.14e-16 |  |
| u_gauge_project_idempotent | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 4.23e-16 |  |
| u_tangent_add_scale | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 2.28e-16 |  |
| u_tangent_reverse | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 1.38e-16 |  |
| u_retract_zero | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 1.20e-15 |  |
| u_retract_fd_jacobian | sh2 | uniform | () | () | (2,) | (0, 0, 1) | PASS | 4.30e-06 | ratio=4.00 |
| u_project_ambient | sh2 | uniform | () | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 12 into shape (3,2,4,1) |
| u_transport_identity | sh2 | uniform | () | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 12 into shape (3,2,4,1) |
| u_transport_vs_ragged_projection | sh2 | uniform | () | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 9 into shape (3,2,3,1) |
| u_to_dense | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 3.47e-17 |  |
| u_add | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.06e-16 |  |
| u_sub | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 8.91e-17 |  |
| u_scalar_mul | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.78e-16 |  |
| u_inner | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 7.55e-16 |  |
| u_norm | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 2.94e-16 |  |
| u_reverse | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.74e-16 |  |
| u_t3svd_lossless | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 8.34e-16 |  |
| u_rank_adjustment_sweep | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.38e-15 |  |
| u_t3svd_trunc_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.02e-14 |  |
| u_orthogonal_representations | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.21e-15 |  |
| u_apply | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.07e-15 |  |
| u_entries | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.44e-16 |  |
| u_probe | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 6.35e-16 |  |
| u_apply_derivatives | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 4.06e-16 |  |
| u_entries_derivatives | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 4.06e-16 |  |
| u_probe_derivatives | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 5.12e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| utv_apply | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 4.62e-16 |  |
| utv_entries | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 3.50e-17 |  |
| utv_probe | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 5.81e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.86e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.73e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 4.99e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.09e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.50e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.91e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.09e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.50e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.91e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 2.27e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 2.49e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 2.27e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 2.49e-16 |  |
| utv_apply | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 8.38e-16 |  |
| utv_entries | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.42e-16 |  |
| utv_probe | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 4.43e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 4.12e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 3.48e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 4.09e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.60e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 9.99e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.60e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 9.99e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 3.34e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 3.40e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 3.34e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 3.40e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_apply | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.30e-16 |  |
| u_entries | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 1.49e-16 |  |
| u_probe | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 4.10e-16 |  |
| u_apply_derivatives | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 3.18e-16 |  |
| u_entries_derivatives | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.72e-16 |  |
| u_probe_derivatives | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 4.21e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| utv_apply | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 4.85e-16 |  |
| utv_entries | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 1.87e-16 |  |
| utv_probe | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 3.76e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 6.11e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 6.97e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 3.48e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 6.31e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 5.09e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.15e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 4.74e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 5.09e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.15e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.30e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 1.09e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 6.59e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 1.15e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.18e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 4.37e-16 |  |
| utv_entries | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 2.85e-16 |  |
| utv_probe | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 3.26e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 3.98e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 4.43e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 3.50e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 2.07e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 1.53e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 1.48e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 2.07e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 3.06e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 2.29e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 1.71e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 1.40e-15 |  |
| u_apply | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.60e-16 |  |
| u_entries | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.64e-16 |  |
| u_probe | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 4.04e-16 |  |
| u_apply_derivatives | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.66e-16 |  |
| u_entries_derivatives | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.17e-16 |  |
| u_probe_derivatives | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 4.42e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| utv_apply | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.03e-16 |  |
| utv_entries | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 6.38e-16 |  |
| utv_probe | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.39e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.46e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.36e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.46e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 1.06e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 6.31e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.98e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 8.48e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 6.31e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 1.49e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 1.12e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.62e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.19e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 1.12e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 1.31e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.19e-16 |  |
| utv_apply | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.29e-16 |  |
| utv_entries | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.68e-16 |  |
| utv_probe | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.39e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.33e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.83e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.53e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.95e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.28e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 6.88e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.36e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.28e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.13e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.21e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.33e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.35e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.17e-16 |  |
| u_manifold_inner | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 5.10e-16 |  |
| u_manifold_norm | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.35e-16 |  |
| u_gauge_project_idempotent | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 2.51e-16 |  |
| u_tangent_add_scale | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 2.15e-16 |  |
| u_tangent_reverse | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 1.47e-16 |  |
| u_retract_zero | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 7.94e-16 |  |
| u_retract_fd_jacobian | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 8.33e-07 | ratio=4.00 |
| u_retract_vs_ragged | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_project_ambient | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 6.58e-16 |  |
| u_transport_identity | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 7.13e-16 |  |
| u_transport_vs_ragged_projection | sh2 | uniform | (2,) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_manifold_inner | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 3.14e-16 |  |
| u_manifold_norm | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.49e-16 |  |
| u_gauge_project_idempotent | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 3.61e-16 |  |
| u_tangent_add_scale | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 2.07e-16 |  |
| u_tangent_reverse | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.81e-16 |  |
| u_retract_zero | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 7.94e-16 |  |
| u_retract_fd_jacobian | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | PASS | 2.05e-07 | ratio=4.00 |
| u_project_ambient | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 24 into shape (3,2,2,4,1) |
| u_transport_identity | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 24 into shape (3,2,2,4,1) |
| u_transport_vs_ragged_projection | sh2 | uniform | (2,) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 18 into shape (3,2,2,3,1) |
| u_to_dense | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 3.39e-18 |  |
| u_add | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 8.84e-17 |  |
| u_sub | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.11e-16 |  |
| u_scalar_mul | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.59e-16 |  |
| u_inner | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 2.05e-15 |  |
| u_norm | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 3.42e-16 |  |
| u_reverse | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.19e-16 |  |
| u_t3svd_lossless | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 9.87e-16 |  |
| u_rank_adjustment_sweep | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.18e-15 |  |
| u_t3svd_trunc_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.44e-15 |  |
| u_orthogonal_representations | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.14e-15 |  |
| u_apply | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 4.90e-16 |  |
| u_entries | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.56e-16 |  |
| u_probe | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 2.08e-16 |  |
| u_apply_derivatives | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 5.82e-16 |  |
| u_entries_derivatives | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 5.66e-16 |  |
| u_probe_derivatives | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 4.77e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| utv_apply | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 3.61e-16 |  |
| utv_entries | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 2.10e-16 |  |
| utv_probe | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 3.46e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 6.25e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 5.67e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 4.96e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.93e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 4.87e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 4.24e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.93e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 4.87e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 4.24e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.72e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.56e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.99e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.72e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.56e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.99e-16 |  |
| utv_apply | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 6.40e-16 |  |
| utv_entries | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.82e-16 |  |
| utv_probe | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.87e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.28e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.31e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.87e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.93e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 1.22e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.34e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.93e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 1.22e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.34e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.91e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 8.74e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.91e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 8.74e-15 |  |
| u_apply | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 2.32e-16 |  |
| u_entries | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.26e-16 |  |
| u_probe | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.01e-16 |  |
| u_apply_derivatives | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.29e-16 |  |
| u_entries_derivatives | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 2.99e-16 |  |
| u_probe_derivatives | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.88e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| utv_apply | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 6.76e-16 |  |
| utv_entries | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.05e-16 |  |
| utv_probe | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.39e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 5.16e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 5.68e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.29e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.67e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 2.00e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.63e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.00e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.50e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.31e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.32e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.50e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 5.23e-16 |  |
| utv_apply | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 5.59e-16 |  |
| utv_entries | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 3.02e-16 |  |
| utv_probe | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 3.25e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 4.77e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 4.55e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 3.56e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 1.06e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 1.18e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 9.81e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 1.30e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 1.18e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 9.81e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 6.06e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 4.33e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 6.06e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 2.16e-16 |  |
| u_apply | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.89e-16 |  |
| u_entries | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.22e-16 |  |
| u_probe | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.80e-16 |  |
| u_apply_derivatives | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 4.37e-16 |  |
| u_entries_derivatives | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.32e-16 |  |
| u_probe_derivatives | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.34e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| utv_apply | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.62e-16 |  |
| utv_entries | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.15e-16 |  |
| utv_probe | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.39e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 5.00e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 4.96e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.34e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.20e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.83e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.80e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 7.21e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.42e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.88e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.00e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 4.22e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.00e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 7.03e-16 |  |
| utv_apply | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.72e-16 |  |
| utv_entries | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.05e-16 |  |
| utv_probe | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.00e-16 |  |
| utv_apply_derivatives | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.41e-16 |  |
| utv_entries_derivatives | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.90e-16 |  |
| utv_probe_derivatives | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.60e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.74e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 9.88e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.16e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.84e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.95e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.77e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 7.52e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.64e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.59e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 9.40e-16 |  |
| u_manifold_inner | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 6.80e-16 |  |
| u_manifold_norm | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 2.43e-16 |  |
| u_gauge_project_idempotent | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 3.98e-16 |  |
| u_tangent_add_scale | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 2.27e-16 |  |
| u_tangent_reverse | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.62e-16 |  |
| u_retract_zero | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 1.73e-15 |  |
| u_retract_fd_jacobian | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 2.95e-05 | ratio=4.00 |
| u_retract_vs_ragged | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_project_ambient | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 9.13e-16 |  |
| u_transport_identity | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 6.53e-16 |  |
| u_transport_vs_ragged_projection | sh2 | uniform | (2, 3) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_manifold_inner | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 1.06e-15 |  |
| u_manifold_norm | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.08e-16 |  |
| u_gauge_project_idempotent | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 4.59e-16 |  |
| u_tangent_add_scale | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.09e-16 |  |
| u_tangent_reverse | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 1.58e-16 |  |
| u_retract_zero | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 1.73e-15 |  |
| u_retract_fd_jacobian | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 1.78e-04 | ratio=3.98 |
| u_project_ambient | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 72 into shape (3,2,2,3,4,1) |
| u_transport_identity | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 72 into shape (3,2,2,3,4,1) |
| u_transport_vs_ragged_projection | sh2 | uniform | (2, 3) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 54 into shape (3,2,2,3,3,1) |
| u_to_dense | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.22e-17 |  |
| u_add | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 4.88e-17 |  |
| u_sub | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 6.48e-17 |  |
| u_scalar_mul | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.14e-16 |  |
| u_inner | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 2.57e-15 |  |
| u_norm | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_reverse | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.19e-16 |  |
| u_t3svd_lossless | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.03e-15 |  |
| u_rank_adjustment_sweep | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 2.99e-15 |  |
| u_t3svd_trunc_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 2.19e-14 |  |
| u_orthogonal_representations | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 3.26e-15 |  |
| u_apply | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.14e-15 |  |
| u_entries | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.66e-16 |  |
| u_probe | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 3.56e-16 |  |
| u_apply_derivatives | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 5.15e-16 |  |
| u_entries_derivatives | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 5.97e-16 |  |
| u_probe_derivatives | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 3.41e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| utv_apply | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 5.95e-16 |  |
| utv_entries | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 3.64e-16 |  |
| utv_probe | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 4.28e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.11e-15 |  |
| utv_entries_derivatives | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.40e-15 |  |
| utv_probe_derivatives | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 3.81e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 5.84e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.13e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 5.84e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.13e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 7.27e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 3.64e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.55e-14 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 7.27e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 3.64e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.55e-14 |  |
| utv_apply | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 3.49e-16 |  |
| utv_entries | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 6.00e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 3.89e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 3.96e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 3.21e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 3.46e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 4.77e-14 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 6.54e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 3.46e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 4.77e-14 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 6.54e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 1.43e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 1.56e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 1.43e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 1.56e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_apply | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 2.70e-16 |  |
| u_entries | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 2.16e-16 |  |
| u_probe | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 2.72e-16 |  |
| u_apply_derivatives | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 5.20e-16 |  |
| u_entries_derivatives | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 2.91e-16 |  |
| u_probe_derivatives | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 3.88e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,4) (3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| utv_apply | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 3.41e-16 |  |
| utv_entries | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 1.92e-16 |  |
| utv_probe | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 3.23e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 2.73e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 2.40e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 3.72e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 1.59e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 1.32e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 1.59e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 1.74e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 2.30e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 4.56e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 1.40e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 1.52e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 1.58e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | () | (0, 0, 1) | PASS | 1.40e-15 |  |
| utv_apply | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 6.03e-16 |  |
| utv_entries | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 2.30e-16 |  |
| utv_probe | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 5.58e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 3.92e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 3.10e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 3.63e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 1.19e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 1.24e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 2.57e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 3.56e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 3.73e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 1.29e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 1.67e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 7.02e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 2.07e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 1.67e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (3,) | (2,) | (0, 0, 1) | PASS | 3.51e-16 |  |
| u_apply | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 8.42e-16 |  |
| u_entries | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 1.48e-16 |  |
| u_probe | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 3.18e-16 |  |
| u_apply_derivatives | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 5.73e-16 |  |
| u_entries_derivatives | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 1.18e-15 |  |
| u_probe_derivatives | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 3.67e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,4) (2,2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| utv_apply | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 2.86e-16 |  |
| utv_entries | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 3.71e-16 |  |
| utv_probe | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 3.25e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 4.88e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 3.34e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 3.36e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 4.35e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 6.15e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 2.23e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 8.70e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 6.15e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 2.23e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 1.50e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 1.49e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 3.00e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 2.06e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.55e-16 |  |
| utv_entries | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.91e-16 |  |
| utv_probe | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.94e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 7.17e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 5.72e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.57e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.43e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.94e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.83e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.43e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.94e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.83e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.77e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 5.92e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.42e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.48e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.95e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | () | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.45e-15 |  |
| u_manifold_inner | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 5.19e-16 |  |
| u_manifold_norm | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 3.11e-16 |  |
| u_gauge_project_idempotent | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 5.10e-16 |  |
| u_tangent_add_scale | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 2.23e-16 |  |
| u_tangent_reverse | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.86e-16 |  |
| u_retract_zero | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 3.25e-15 |  |
| u_retract_fd_jacobian | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 3.79e-06 | ratio=4.00 |
| u_retract_vs_ragged | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 1.31e-15 |  |
| u_project_ambient | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 2.18e-16 |  |
| u_transport_identity | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 8.78e-16 |  |
| u_transport_vs_ragged_projection | sh2 | uniform+pad | () | () | () | (0, 0, 1) | PASS | 4.90e-16 |  |
| u_manifold_inner | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 7.87e-16 |  |
| u_manifold_norm | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 3.48e-16 |  |
| u_gauge_project_idempotent | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 5.47e-16 |  |
| u_tangent_add_scale | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 2.33e-16 |  |
| u_tangent_reverse | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 1.13e-16 |  |
| u_retract_zero | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 3.25e-15 |  |
| u_retract_fd_jacobian | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | PASS | 1.58e-05 | ratio=4.00 |
| u_project_ambient | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 15 into shape (3,2,5,1) |
| u_transport_identity | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 15 into shape (3,2,5,1) |
| u_transport_vs_ragged_projection | sh2 | uniform+pad | () | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 15 into shape (3,2,5,1) |
| u_to_dense | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 3.47e-17 |  |
| u_add | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.06e-16 |  |
| u_sub | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 8.91e-17 |  |
| u_scalar_mul | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.78e-16 |  |
| u_inner | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.10e-15 |  |
| u_norm | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 2.08e-16 |  |
| u_reverse | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.74e-16 |  |
| u_t3svd_lossless | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.33e-15 |  |
| u_rank_adjustment_sweep | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 2.14e-15 |  |
| u_t3svd_trunc_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.04e-14 |  |
| u_orthogonal_representations | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 7.12e-16 |  |
| u_apply | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 2.20e-15 |  |
| u_entries | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 5.72e-16 |  |
| u_probe | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 5.90e-16 |  |
| u_apply_derivatives | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 4.06e-16 |  |
| u_entries_derivatives | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 4.06e-16 |  |
| u_probe_derivatives | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 5.12e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| utv_apply | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 4.70e-16 |  |
| utv_entries | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 3.94e-16 |  |
| utv_probe | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 3.80e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.88e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.88e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 3.85e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 7.49e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.53e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 7.49e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.53e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.47e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.64e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.47e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.64e-16 |  |
| utv_apply | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 6.12e-16 |  |
| utv_entries | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 7.11e-16 |  |
| utv_probe | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 3.10e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 5.02e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 4.97e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 3.63e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 2.44e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.29e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 2.44e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.29e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 9.13e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 6.20e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.52e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 9.13e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 6.20e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.52e-16 |  |
| u_apply | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 1.64e-16 |  |
| u_entries | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 1.49e-16 |  |
| u_probe | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 4.10e-16 |  |
| u_apply_derivatives | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 3.18e-16 |  |
| u_entries_derivatives | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.70e-16 |  |
| u_probe_derivatives | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 4.08e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,2,4) (3,2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| utv_apply | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 6.13e-16 |  |
| utv_entries | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.11e-16 |  |
| utv_probe | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 4.10e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 3.62e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 3.99e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 3.34e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 6.04e-15 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 1.87e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 6.04e-15 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 3.74e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.18e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.57e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.18e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | () | (0, 0, 1) | PASS | 2.20e-16 |  |
| utv_apply | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 4.97e-16 |  |
| utv_entries | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 3.39e-16 |  |
| utv_probe | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 4.11e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 3.63e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 4.04e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 3.92e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 2.26e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 1.18e-14 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 1.13e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 3.93e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 4.08e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 4.27e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 2.76e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 1.36e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 5.69e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (3,) | (2,) | (0, 0, 1) | PASS | 2.76e-16 |  |
| u_apply | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.32e-16 |  |
| u_entries | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.64e-16 |  |
| u_probe | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 4.55e-16 |  |
| u_apply_derivatives | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.99e-16 |  |
| u_entries_derivatives | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.79e-16 |  |
| u_probe_derivatives | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 4.59e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,2,4) (2,2,2,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,4) (2,4,4)  |
| utv_apply | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.44e-16 |  |
| utv_entries | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.08e-16 |  |
| utv_probe | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.75e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.95e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.28e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.45e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 3.25e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.16e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 1.63e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 2.16e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 1.22e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 4.45e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 1.22e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.75e-16 |  |
| utv_entries | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.33e-16 |  |
| utv_probe | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.13e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.33e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.92e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.58e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.73e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 5.84e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.80e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.73e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.89e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 1.50e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.98e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.13e-15 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.09e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.57e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2,) | (2, 2) | (2,) | (0, 0, 1) | PASS | 9.28e-16 |  |
| u_manifold_inner | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.12e-15 |  |
| u_manifold_norm | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.05e-16 |  |
| u_gauge_project_idempotent | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 4.52e-16 |  |
| u_tangent_add_scale | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 2.31e-16 |  |
| u_tangent_reverse | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 1.46e-16 |  |
| u_retract_zero | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 7.93e-16 |  |
| u_retract_fd_jacobian | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 3.46e-07 | ratio=4.00 |
| u_retract_vs_ragged | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_project_ambient | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 3.23e-16 |  |
| u_transport_identity | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 7.36e-16 |  |
| u_transport_vs_ragged_projection | sh2 | uniform+pad | (2,) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_manifold_inner | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 3.95e-16 |  |
| u_manifold_norm | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_gauge_project_idempotent | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 4.18e-16 |  |
| u_tangent_add_scale | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.93e-16 |  |
| u_tangent_reverse | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.61e-16 |  |
| u_retract_zero | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 7.93e-16 |  |
| u_retract_fd_jacobian | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | PASS | 1.30e-07 | ratio=4.00 |
| u_project_ambient | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 30 into shape (3,2,2,5,1) |
| u_transport_identity | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 30 into shape (3,2,2,5,1) |
| u_transport_vs_ragged_projection | sh2 | uniform+pad | (2,) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 30 into shape (3,2,2,5,1) |
| u_to_dense | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 3.39e-18 |  |
| u_add | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 8.84e-17 |  |
| u_sub | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 1.11e-16 |  |
| u_scalar_mul | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 1.59e-16 |  |
| u_inner | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 2.34e-15 |  |
| u_norm | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 3.51e-16 |  |
| u_reverse | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 1.19e-16 |  |
| u_t3svd_lossless | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 1.09e-15 |  |
| u_rank_adjustment_sweep | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 9.87e-16 |  |
| u_t3svd_trunc_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 3.33e-15 |  |
| u_orthogonal_representations | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 1.79e-15 |  |
| u_apply | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 6.02e-16 |  |
| u_entries | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 2.23e-16 |  |
| u_probe | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 3.41e-16 |  |
| u_apply_derivatives | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 5.85e-16 |  |
| u_entries_derivatives | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 5.70e-16 |  |
| u_probe_derivatives | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 4.77e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| utv_apply | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 3.01e-16 |  |
| utv_entries | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 2.97e-16 |  |
| utv_probe | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 3.37e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 3.52e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 3.50e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 4.07e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 5.04e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 5.04e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 2.36e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 9.55e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 2.73e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 2.36e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 9.55e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 2.73e-16 |  |
| utv_apply | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.26e-16 |  |
| utv_entries | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.29e-16 |  |
| utv_probe | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.42e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 8.87e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 8.84e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 4.42e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 1.25e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.29e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.14e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 1.25e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.29e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.14e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.32e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.41e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 7.42e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 3.32e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.41e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 7.42e-16 |  |
| u_apply | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 2.15e-16 |  |
| u_entries | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.26e-16 |  |
| u_probe | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.00e-16 |  |
| u_apply_derivatives | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.15e-16 |  |
| u_entries_derivatives | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 2.99e-16 |  |
| u_probe_derivatives | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.96e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (3,2,3,2,4) (3,2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| utv_apply | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 4.84e-16 |  |
| utv_entries | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.60e-16 |  |
| utv_probe | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.17e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 4.68e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 4.63e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.40e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.87e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.55e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 4.38e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 5.84e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.89e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 4.89e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 4.40e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 1.95e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 3.26e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 4.54e-16 |  |
| utv_entries | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 2.33e-16 |  |
| utv_probe | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 3.61e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 4.28e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 3.84e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 3.40e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 4.11e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 1.41e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 2.74e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 2.82e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 3.20e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 2.03e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 3.20e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 1.12e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (3,) | (2,) | (0, 0, 1) | PASS | 1.22e-15 |  |
| u_apply | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.58e-16 |  |
| u_entries | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.22e-16 |  |
| u_probe | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.45e-16 |  |
| u_apply_derivatives | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 4.38e-16 |  |
| u_entries_derivatives | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.35e-16 |  |
| u_probe_derivatives | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.62e-16 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,2,2,3,2,4) (2,2,2,3,4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,3,2,4) (2,3,4,4)  |
| utv_apply | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 5.53e-16 |  |
| utv_entries | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.28e-16 |  |
| utv_probe | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.31e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 5.28e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 5.58e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 3.78e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.19e-15 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.42e-15 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.31e-15 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 2.04e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.56e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.26e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.02e-15 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.25e-15 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.26e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | () | (0, 0, 1) | PASS | 1.02e-15 |  |
| utv_apply | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.86e-16 |  |
| utv_entries | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.53e-16 |  |
| utv_probe | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.56e-16 |  |
| utv_apply_derivatives | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.29e-16 |  |
| utv_entries_derivatives | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.77e-16 |  |
| utv_probe_derivatives | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.56e-16 |  |
| utv_apply_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.18e-16 |  |
| utv_entries_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 3.21e-16 |  |
| utv_probe_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.74e-16 |  |
| utv_apply_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.09e-16 |  |
| utv_entries_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.81e-16 |  |
| utv_probe_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 4.10e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 8.41e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.20e-16 |  |
| utv_probe_derivatives_transpose_adjoint(sum=False) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 6.06e-16 |  |
| utv_apply_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 2.80e-16 |  |
| utv_entries_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 0.00e+00 |  |
| utv_probe_derivatives_transpose_adjoint(sum=True) | sh2 | uniform+pad | (2, 3) | (2, 2) | (2,) | (0, 0, 1) | PASS | 6.06e-16 |  |
| u_manifold_inner | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 1.17e-15 |  |
| u_manifold_norm | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 3.60e-16 |  |
| u_gauge_project_idempotent | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 4.67e-16 |  |
| u_tangent_add_scale | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 2.22e-16 |  |
| u_tangent_reverse | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 1.75e-16 |  |
| u_retract_zero | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 2.03e-15 |  |
| u_retract_fd_jacobian | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 4.62e-05 | ratio=4.00 |
| u_retract_vs_ragged | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_project_ambient | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 1.60e-15 |  |
| u_transport_identity | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 9.91e-16 |  |
| u_transport_vs_ragged_projection | sh2 | uniform+pad | (2, 3) | () | () | (0, 0, 1) | PASS | 0.00e+00 |  |
| u_manifold_inner | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 1.36e-15 |  |
| u_manifold_norm | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.80e-16 |  |
| u_gauge_project_idempotent | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 4.75e-16 |  |
| u_tangent_add_scale | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.22e-16 |  |
| u_tangent_reverse | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 1.56e-16 |  |
| u_retract_zero | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 2.03e-15 |  |
| u_retract_fd_jacobian | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | PASS | 7.70e-05 | ratio=3.98 |
| u_project_ambient | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 90 into shape (3,2,2,3,5,1) |
| u_transport_identity | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 90 into shape (3,2,2,3,5,1) |
| u_transport_vs_ragged_projection | sh2 | uniform+pad | (2, 3) | () | (2,) | (0, 0, 1) | EXC | nan | ValueError: cannot reshape array of size 90 into shape (3,2,2,3,5,1) |
| u_to_dense | shall | uniform | () | () | () | (0, 0, 0) | PASS | 7.71e-17 |  |
| u_add | shall | uniform | () | () | () | (0, 0, 0) | PASS | 1.17e-16 |  |
| u_sub | shall | uniform | () | () | () | (0, 0, 0) | PASS | 1.09e-16 |  |
| u_scalar_mul | shall | uniform | () | () | () | (0, 0, 0) | PASS | 2.20e-16 |  |
| u_inner | shall | uniform | () | () | () | (0, 0, 0) | PASS | 1.04e-14 |  |
| u_norm | shall | uniform | () | () | () | (0, 0, 0) | PASS | 2.02e-16 |  |
| u_reverse | shall | uniform | () | () | () | (0, 0, 0) | PASS | 1.49e-16 |  |
| u_t3svd_lossless | shall | uniform | () | () | () | (0, 0, 0) | PASS | 1.45e-15 |  |
| u_rank_adjustment_sweep | shall | uniform | () | () | () | (0, 0, 0) | PASS | 9.71e-16 |  |
| u_t3svd_trunc_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | PASS | 4.16e-14 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | shall | uniform | () | () | () | (0, 0, 0) | FAIL | 4.98e-07 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | shall | uniform | () | () | () | (0, 0, 0) | FAIL | 1.00e+00 |  |
| u_apply | shall | uniform | () | () | () | (0, 0, 0) | PASS | 1.30e-16 |  |
| u_entries | shall | uniform | () | () | () | (0, 0, 0) | PASS | 2.57e-15 |  |
| u_probe | shall | uniform | () | () | () | (0, 0, 0) | PASS | 9.05e-16 |  |
| u_apply_derivatives | shall | uniform | () | () | () | (0, 0, 0) | PASS | 4.61e-15 |  |
| u_entries_derivatives | shall | uniform | () | () | () | (0, 0, 0) | PASS | 1.48e-14 |  |
| u_probe_derivatives | shall | uniform | () | () | () | (0, 0, 0) | PASS | 2.51e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| SWEEP_CRASH | shall | uniform | () | () | () | (0, 0, 0) | EXC | nan | ValueError: UniformManifoldGeometry.project requires an orthogonal frame (the manifold geometry needs an orthono @ File "/home/nick/repos/T3Toolbox/t3toolbox/safety.py", line 195, in require |
| u_to_dense | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 7.71e-17 |  |
| u_add | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 1.17e-16 |  |
| u_sub | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 1.09e-16 |  |
| u_scalar_mul | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 2.20e-16 |  |
| u_inner | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 5.68e-15 |  |
| u_norm | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 2.02e-16 |  |
| u_reverse | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 1.49e-16 |  |
| u_t3svd_lossless | shall | uniform+pad | () | () | () | (0, 0, 0) | FAIL | 1.00e+00 |  |
| u_rank_adjustment_sweep | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 1.07e-15 |  |
| u_t3svd_trunc_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 9.38e-15 |  |
| u_orthogonal_representations_NOT_ORTHOGONAL | shall | uniform+pad | () | () | () | (0, 0, 0) | FAIL | 9.99e-01 | frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead |
| u_orthogonal_representations | shall | uniform+pad | () | () | () | (0, 0, 0) | FAIL | 1.00e+00 |  |
| u_apply | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 7.77e-16 |  |
| u_entries | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 1.40e-15 |  |
| u_probe | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 1.17e-15 |  |
| u_apply_derivatives | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 2.55e-15 |  |
| u_entries_derivatives | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 2.96e-15 |  |
| u_probe_derivatives | shall | uniform+pad | () | () | () | (0, 0, 0) | PASS | 2.13e-15 |  |
| u_apply_corewise_transpose(sum=False)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=False)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=False)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_transpose(sum=True)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_transpose(sum=True)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_transpose(sum=True)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=False)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=False)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=False)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_apply_corewise_derivatives_transpose(sum=True)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_entries_corewise_derivatives_transpose(sum=True)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| u_probe_corewise_derivatives_transpose(sum=True)_vs_ragged | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: operands could not be broadcast together with shapes (2,4) (4,4)  |
| SWEEP_CRASH | shall | uniform+pad | () | () | () | (0, 0, 0) | EXC | nan | ValueError: UniformManifoldGeometry.project requires an orthogonal frame (the manifold geometry needs an orthono @ File "/home/nick/repos/T3Toolbox/t3toolbox/safety.py", line 195, in require |
