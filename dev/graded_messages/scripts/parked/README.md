# Parked ideas

Ideas that were assessed and set aside. They are not part of the proposal, and `run_all.sh` does not run them.

## `xi` once per group, for shared sample vectors

Parked by the maintainer on 2026-11-14 as low priority and not worth doing now. It may or may not be revisited.
The write-up is the "Parked ideas" section of `EVALUATOR.md`.

| File | What it is |
| --- | --- |
| `plan_mockup_per_group_xi.py` | The mockup with the prototype: a `sample_groups` argument, and `xi` stored per group and read per site |
| `per_group_xi.diff` | The difference from `../plan_mockup.py` as it was when the idea was parked |
| `shared_sample_vectors_check.py` | The check: identical answers per group and per mode, agreement with the library's shared geometry, and the saving in `xi` |

To run the check:  `python3 parked/shared_sample_vectors_check.py /path/to/T3Toolbox`
