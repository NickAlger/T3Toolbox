## Calls of the targets (count; files) by place

| name | library | library doctest | tests | examples | page doctest |
|---|---|---|---|---|---|
| `blocks_to_weight` | 3 (2) | 7 (2) | 46 (14) |  | 2 (1) |
| `data_metric` | 6 (3) | 8 (2) | 41 (5) |  |  |
| `jet_to_dual_jet` |  | 5 (1) | 34 (4) |  |  |
| `dual_jet_to_jet` |  | 2 (1) | 19 (3) |  |  |
| `derivative_weight_to_taylor_weight` |  | 4 (1) | 7 (3) |  |  |
| `prepare_derivative_data` | 1 (1) | 8 (1) | 20 (5) |  | 2 (2) |
| `check_weight` | 1 (1) | 6 (1) | 9 (3) |  |  |
| `check_data` | 1 (1) | 7 (1) | 9 (2) |  |  |
| `pack_data` | 2 (1) | 3 (1) | 4 (2) |  |  |
| `_pack_data_space` | 3 (2) |  |  |  |  |
| `pack_batch` | 1 (1) | 1 (1) | 24 (11) |  |  |
| `LeastSquaresBatch` | 4 (2) | 23 (6) | 75 (18) | 5 (5) | 4 (2) |
| `LeastSquaresOracle` | 2 (1) | 16 (5) | 71 (14) | 5 (5) | 4 (2) |
| `check_batch` | 1 (1) | 2 (1) | 6 (2) |  |  |
| `n_measurements` | 3 (3) |  | 8 (2) |  |  |
| `take` | 16 (4) | 3 (1) | 48 (16) |  |  |
| `flat_draw` | 2 (1) |  | 2 (1) |  |  |
| `_require_draw` | 2 (1) |  |  |  |  |
| `make_newton_display` | 1 (1) | 1 (1) | 15 (1) |  |  |
| `prepare_batch` | 4 (2) |  | 2 (1) |  |  |
| `_is_packed` | 1 (1) |  |  |  |  |
| `_frontend_draw` | 2 (1) |  |  |  |  |
| `sampling_variables` | 38 (6) | 10 (1) | 193 (17) |  |  |
| `computed_names` | 1 (1) | 3 (1) | 5 (1) |  |  |
| `_y_metric_y` | 1 (1) |  |  |  |  |
| `_y_block_metric_y` | 1 (1) |  |  |  |  |
| `_y_times_metric_y` | 2 (1) |  |  |  |  |
| `_arrays_of` | 2 (1) |  |  |  |  |
| `_take_data_space` | 2 (1) |  |  |  |  |
| `_take_W` | 2 (1) |  |  |  |  |

## `weight=` keyword, by callee and place

| callee | library | library doctest | tests | examples | page doctest |
|---|---|---|---|---|---|
| `Jp_norm_sq` |  | 1 |  |  |  |
| `None` |  |  | 8 |  |  |
| `_deriv_fit` |  |  | 7 |  |  |
| `_deriv_fit_away_from_singular` |  |  | 1 |  |  |
| `_fmodel` |  |  | 2 |  |  |
| `_pd_fit` |  |  | 8 |  |  |
| `_plain_fit` |  |  | 4 |  |  |
| `_plain_fit_away_from_singular` |  |  | 1 |  |  |
| `_shared_fit` |  |  | 3 |  |  |
| `call` |  |  | 3 |  |  |
| `computed_names` |  |  | 2 |  |  |
| `derivative_fit` |  |  | 18 |  |  |
| `dict` | 2 | 8 | 14 |  |  |
| `f` |  | 3 | 6 |  |  |
| `fit` |  |  | 4 |  |  |
| `mc_sgd` |  |  | 1 | 1 | 1 |
| `newton_cg` |  |  | 11 | 4 | 1 |
| `old_vocabulary` |  |  | 4 |  |  |
| `r_block_sumsq` |  | 1 |  |  |  |
| `replace` | 1 | 2 | 6 |  |  |
| `sampling_variables` |  |  | 50 |  |  |
| `subTest` |  |  | 21 |  |  |
| **all** | 3 (3 files) | 15 (2 files) | 174 (28 files) | 5 (5 files) | 2 (1 files) |

## Old-form optimizer calls (the translation keeps them working)

- examples, no weight: 4
- examples, with weight=: 5
- page doctest, no weight: 19
- page doctest, with weight=: 2
- tests, no weight: 57
- tests, with weight=: 11

## Attribute reads

- `.kind` (on an oracle), library: 1
- `.kind` (on an oracle), tests: 14
- `.order` (on an oracle), library: 1
- `.order` (on an oracle), tests: 10
- `.weight` (any object), library: 10
- `.weight` (any object), library doctest: 2
- `.weight` (any object), tests: 46

## Tests touched (a `def test_*` containing a site), per file

- `tests/backend/test_curvature.py`: 17
- `tests/backend/test_optimizer_display.py`: 12
- `tests/backend/test_optimizers.py`: 11
- `tests/backend/test_padded_frame.py`: 2
- `tests/backend/test_uniform_curvature.py`: 3
- `tests/backend/test_uniform_fitting.py`: 2
- `tests/test_aligned_padded_frame_frontend.py`: 2
- `tests/test_block_sums.py`: 4
- `tests/test_contractions_sharding.py`: 1
- `tests/test_data_metric.py`: 16
- `tests/test_data_preparation.py`: 1
- `tests/test_dense_oracle.py`: 1
- `tests/test_dispatch.py`: 21
- `tests/test_engine_data_space.py`: 8
- `tests/test_fitting.py`: 8
- `tests/test_jet_contract.py`: 1
- `tests/test_jets.py`: 21
- `tests/test_least_squares_oracle.py`: 13
- `tests/test_manifold.py`: 1
- `tests/test_optimizers_frontend.py`: 30
- `tests/test_oracle_data_space.py`: 11
- `tests/test_sampling_apply.py`: 1
- `tests/test_sampling_corewise.py`: 2
- `tests/test_sampling_definitions.py`: 5
- `tests/test_sampling_entries.py`: 4
- `tests/test_sampling_functions.py`: 1
- `tests/test_sampling_graph.py`: 1
- `tests/test_sampling_identities.py`: 3
- `tests/test_sampling_stacks.py`: 2
- `tests/test_sampling_typed.py`: 2
- `tests/test_sampling_variables.py`: 2
- `tests/test_sharing.py`: 5
- `tests/test_transposes_typed.py`: 1
- `tests/test_type_laws.py`: 6
- **all**: 221 test functions (names, not parametrized nodes) in 34 files

## Page text (prose and fenced non-doctest lines; a plain search)

- `data_metric`: 2 lines in 1 pages
- `pack_data`: 1 lines in 1 pages
- `n_measurements`: 2 lines in 1 pages
- `weight=`: 7 lines in 4 pages
