| op | struct | repr | C | W | K | sharing | status | max relerr | note |
|---|---|---|---|---|---|---|---|---|---|
| vr_to_dense | vary | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| vr_apply | vary | uniform | (2,) | () | () | None | PASS | 1.59e-16 |  |
| vr_probe | vary | uniform | (2,) | () | () | None | PASS | 1.26e-16 |  |
| vr_entries | vary | uniform | (2,) | () | () | None | PASS | 1.22e-16 |  |
| vr_probe_derivatives | vary | uniform | (2,) | () | () | None | PASS | 3.67e-16 |  |
| vr_apply_derivatives | vary | uniform | (2,) | () | () | None | PASS | 4.40e-16 |  |
| vr_tangent_to_dense | vary | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| vr_tv_apply | vary | uniform | (2,) | () | () | None | PASS | 7.39e-16 |  |
| vr_tv_probe | vary | uniform | (2,) | () | () | None | PASS | 3.14e-16 |  |
| vr_tv_entries | vary | uniform | (2,) | () | () | None | PASS | 7.50e-16 |  |
| vr_tv_probe_derivatives | vary | uniform | (2,) | () | () | None | PASS | 3.69e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=False) | vary | uniform | (2,) | () | () | None | PASS | 1.27e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=True) | vary | uniform | (2,) | () | () | None | PASS | 1.27e-16 |  |
| vr_manifold_norm | vary | uniform | (2,) | () | () | None | FAIL | 3.32e+02 |  |
| vr_retract_per_element | vary | uniform | (2,) | () | () | None | PASS | 0.00e+00 |  |
| vr_tangent_to_dense | vary | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| vr_tv_apply | vary | uniform | (2,) | () | (2,) | None | PASS | 7.08e-16 |  |
| vr_tv_probe | vary | uniform | (2,) | () | (2,) | None | PASS | 3.67e-16 |  |
| vr_tv_entries | vary | uniform | (2,) | () | (2,) | None | PASS | 2.04e-16 |  |
| vr_tv_probe_derivatives | vary | uniform | (2,) | () | (2,) | None | PASS | 3.00e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=False) | vary | uniform | (2,) | () | (2,) | None | PASS | 9.90e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=True) | vary | uniform | (2,) | () | (2,) | None | PASS | 9.90e-16 |  |
| vr_manifold_norm | vary | uniform | (2,) | () | (2,) | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| vr_retract_per_element | vary | uniform | (2,) | () | (2,) | None | PASS | 0.00e+00 |  |
| vr_apply | vary | uniform | (2,) | (3,) | () | None | PASS | 2.76e-16 |  |
| vr_probe | vary | uniform | (2,) | (3,) | () | None | PASS | 2.91e-16 |  |
| vr_entries | vary | uniform | (2,) | (3,) | () | None | PASS | 2.90e-16 |  |
| vr_probe_derivatives | vary | uniform | (2,) | (3,) | () | None | PASS | 4.25e-16 |  |
| vr_apply_derivatives | vary | uniform | (2,) | (3,) | () | None | PASS | 6.60e-16 |  |
| vr_tangent_to_dense | vary | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| vr_tv_apply | vary | uniform | (2,) | (3,) | () | None | PASS | 4.78e-16 |  |
| vr_tv_probe | vary | uniform | (2,) | (3,) | () | None | PASS | 3.28e-16 |  |
| vr_tv_entries | vary | uniform | (2,) | (3,) | () | None | PASS | 8.12e-17 |  |
| vr_tv_probe_derivatives | vary | uniform | (2,) | (3,) | () | None | PASS | 4.09e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=False) | vary | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| vr_tv_probe_transpose_adjoint(sum=True) | vary | uniform | (2,) | (3,) | () | None | PASS | 2.36e-16 |  |
| vr_manifold_norm | vary | uniform | (2,) | (3,) | () | None | FAIL | 1.97e+02 |  |
| vr_retract_per_element | vary | uniform | (2,) | (3,) | () | None | PASS | 0.00e+00 |  |
| vr_tangent_to_dense | vary | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| vr_tv_apply | vary | uniform | (2,) | (3,) | (2,) | None | PASS | 2.75e-16 |  |
| vr_tv_probe | vary | uniform | (2,) | (3,) | (2,) | None | PASS | 3.11e-16 |  |
| vr_tv_entries | vary | uniform | (2,) | (3,) | (2,) | None | PASS | 3.57e-16 |  |
| vr_tv_probe_derivatives | vary | uniform | (2,) | (3,) | (2,) | None | PASS | 3.34e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=False) | vary | uniform | (2,) | (3,) | (2,) | None | PASS | 1.88e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=True) | vary | uniform | (2,) | (3,) | (2,) | None | PASS | 1.88e-16 |  |
| vr_manifold_norm | vary | uniform | (2,) | (3,) | (2,) | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| vr_retract_per_element | vary | uniform | (2,) | (3,) | (2,) | None | PASS | 0.00e+00 |  |
| vr_apply | vary | uniform | (2,) | (2, 2) | () | None | PASS | 2.56e-16 |  |
| vr_probe | vary | uniform | (2,) | (2, 2) | () | None | PASS | 4.62e-16 |  |
| vr_entries | vary | uniform | (2,) | (2, 2) | () | None | PASS | 1.64e-16 |  |
| vr_probe_derivatives | vary | uniform | (2,) | (2, 2) | () | None | PASS | 3.86e-16 |  |
| vr_apply_derivatives | vary | uniform | (2,) | (2, 2) | () | None | PASS | 1.93e-15 |  |
| vr_tangent_to_dense | vary | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| vr_tv_apply | vary | uniform | (2,) | (2, 2) | () | None | PASS | 2.41e-16 |  |
| vr_tv_probe | vary | uniform | (2,) | (2, 2) | () | None | PASS | 2.86e-16 |  |
| vr_tv_entries | vary | uniform | (2,) | (2, 2) | () | None | PASS | 2.14e-16 |  |
| vr_tv_probe_derivatives | vary | uniform | (2,) | (2, 2) | () | None | PASS | 3.31e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=False) | vary | uniform | (2,) | (2, 2) | () | None | PASS | 2.79e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=True) | vary | uniform | (2,) | (2, 2) | () | None | PASS | 2.23e-15 |  |
| vr_manifold_norm | vary | uniform | (2,) | (2, 2) | () | None | FAIL | 1.74e+02 |  |
| vr_retract_per_element | vary | uniform | (2,) | (2, 2) | () | None | PASS | 0.00e+00 |  |
| vr_tangent_to_dense | vary | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
| vr_tv_apply | vary | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.38e-16 |  |
| vr_tv_probe | vary | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.60e-16 |  |
| vr_tv_entries | vary | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.06e-16 |  |
| vr_tv_probe_derivatives | vary | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.74e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=False) | vary | uniform | (2,) | (2, 2) | (2,) | None | PASS | 2.89e-16 |  |
| vr_tv_probe_transpose_adjoint(sum=True) | vary | uniform | (2,) | (2, 2) | (2,) | None | PASS | 3.18e-15 |  |
| vr_manifold_norm | vary | uniform | (2,) | (2, 2) | (2,) | None | EXC | nan | ValueError: Improper number of dimensions to norm. |
| vr_retract_per_element | vary | uniform | (2,) | (2, 2) | (2,) | None | PASS | 0.00e+00 |  |
