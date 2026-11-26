#!/usr/bin/env python3
"""Part 1's mutation pass (the note's section 6.4 has no row for part 1; these are the reference's twins of part 2's
rows, and the reference's own rules): each mutation applied alone to a
scratch copy of `t3toolbox/` and `tests/`, the named tests run there with `python -B`; a mutation is caught when they
fail. The copy is restored after each mutation.

    python3 part1_mutations.py SCRATCH_DIR [LABEL_PART]   # SCRATCH_DIR: an empty folder; LABEL_PART: only those
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
PY = sys.executable

MUTATIONS = [
    ('the block weight enters unsquared',
     't3toolbox/backend/sampling_definitions.py',
     "        squared = xnp.reshape(row ** 2, tuple(row.shape) + (1,) * (len(shape) - m))  # left-aligned: the jet axes",
     "        squared = xnp.reshape(row, tuple(row.shape) + (1,) * (len(shape) - m))",
     ['tests/test_sampling_definitions.py::TestTheTwoWeights' + '::test_against_brute_force']),
    ('the block row right-aligned (on the trailing axes)',
     't3toolbox/backend/sampling_definitions.py',
     "        squared = xnp.reshape(row ** 2, tuple(row.shape) + (1,) * (len(shape) - m))  # left-aligned: the jet axes",
     "        squared = xnp.reshape(row ** 2, (1,) * (len(shape) - row.ndim) + tuple(row.shape))",
     ['tests/test_sampling_definitions.py::TestTheTwoWeights' + '::test_against_brute_force']),
    ('the weights applied after the turn (M flip(r) for flip(M r))',
     't3toolbox/backend/sampling_definitions.py',
     "    return _map_type_of_z(S, lambda i, jet_TX: _turn(S, _weight_squared(S, block_weight, full_weight, i, jet_TX.shape)\n"
     "                                                     * jet_TX), jet)",
     "    return _map_type_of_z(S, lambda i, jet_TX: _weight_squared(S, block_weight, full_weight, i, jet_TX.shape)\n"
     "                                                     * _turn(S, jet_TX), jet)",
     ['tests/test_sampling_definitions.py::TestTheTwoWeights' + '::test_against_brute_force']),
    ('the full weight may not have a sample axis of 1',
     't3toolbox/backend/sampling_definitions.py',
     "        _check_data_space('the full weight', i, f_X.shape, shape, tuple(range(m, len(S.M))))   # the sample axes W",
     "        _check_data_space('the full weight', i, f_X.shape, shape, ())",
     ['tests/test_sampling_definitions.py::TestTheTwoWeights' + '::test_against_brute_force']),
    ('the full weight may have an edge or a jet axis of 1 (every axis may be 1)',
     't3toolbox/backend/sampling_definitions.py',
     "        _check_data_space('the full weight', i, f_X.shape, shape, tuple(range(m, len(S.M))))   # the sample axes W",
     "        _check_data_space('the full weight', i, f_X.shape, shape, tuple(range(len(shape))))",
     ['tests/test_sampling_definitions.py::TestTheTwoWeights' + '::test_a_weight_of_another_form_raises']),
    ('y may have a sample axis of 1',
     't3toolbox/backend/sampling_definitions.py',
     "        _check_data_space('the data y', i, np.shape(y_TX), np.shape(z_TX), ())",
     "        _check_data_space('the data y', i, np.shape(y_TX), np.shape(z_TX), tuple(range(len(S.order), len(S.M))))",
     ['tests/test_sampling_definitions.py::TestTheTwoWeights' + '::test_data_of_another_form_raises']),
    ('the block weight is not checked',
     't3toolbox/backend/sampling_definitions.py',
     "        _check_block_weight(S, block_weight)\n",
     "",
     ['tests/test_sampling_definitions.py::TestTheTwoWeights' + '::test_a_weight_of_another_form_raises']),
    ('the block weight enters unsquared: seen by the old engine through the reference comparisons',
     't3toolbox/backend/sampling_definitions.py',
     "        squared = xnp.reshape(row ** 2, tuple(row.shape) + (1,) * (len(shape) - m))  # left-aligned: the jet axes",
     "        squared = xnp.reshape(row, tuple(row.shape) + (1,) * (len(shape) - m))",
     ['tests/test_sampling_variables.py::TestByDefinition', 'tests/test_sampling_apply.py::TestByDefinition',
      'tests/test_engine_data_space.py::TestAgainstTheReference']),
    ('the oracle places the block row on the wrong slot (transposed jet index)',
     'tests/oracle_sweeps/oracle.py',
     "        out[slot] = row[slot]",
     "        out[slot] = row[slot[::-1]] if row.ndim > 1 else row[tuple(-1 - k for k in slot)]",
     ['tests/test_dense_oracle.py::TestTwoWeights']),
    ('nu_tilde swaps the two weights in its own call',
     't3toolbox/backend/sampling_definitions.py',
     "    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'nu_tilde'))",
     "    seed_M = _at_nodes(S, _the_seed(S, sample, y, full_weight, block_weight, w_tilde, 'nu_tilde'))",
     ['tests/test_sampling_definitions.py::TestTheTwoWeights::test_every_name_that_reads_the_seed_reads_both_weights']),
    ('V_tilde_dot drops the full weight in its own call',
     't3toolbox/backend/sampling_definitions.py',
     "    V_blocks, H_blocks = _Kp(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'V_tilde_dot'),",
     "    V_blocks, H_blocks = _Kp(S, _the_seed(S, sample, y, block_weight, None, w_tilde, 'V_tilde_dot'),",
     ['tests/test_sampling_definitions.py::TestTheTwoWeights::test_every_name_that_reads_the_seed_reads_both_weights']),
]

def run(copy, tests):
    env = dict(os.environ, PYTHONPATH=copy)
    p = subprocess.run([PY, '-B', '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider'] + tests, cwd=copy, env=env,
                       capture_output=True, text=True)
    return p.returncode, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]


def main():
    scratch = os.path.abspath(sys.argv[1])
    copy = os.path.join(scratch, 'tree')
    if os.path.exists(copy):
        shutil.rmtree(copy)
    for d in ('t3toolbox', 'tests'):
        shutil.copytree(os.path.join(REPO, d), os.path.join(copy, d), ignore=shutil.ignore_patterns('__pycache__'))
    only = sys.argv[2] if len(sys.argv) > 2 else ''          # an optional substring of the labels to run
    mutations = [mu for mu in MUTATIONS if only in mu[0]]
    all_tests = sorted({t for *_, ts in mutations for t in ts})
    code, last = run(copy, all_tests)
    print('UNMUTATED: exit %d, %s' % (code, last))
    for label, path, old, new, tests in mutations:
        target = os.path.join(copy, path)
        text = open(target).read()
        assert text.count(old) == 1, (label, text.count(old))
        open(target, 'w').write(text.replace(old, new))
        code, last = run(copy, tests)
        open(target, 'w').write(text)
        print('%-8s %s -- %s' % ('CAUGHT' if code != 0 else 'SURVIVED', label, last))
    shutil.rmtree(copy)


if __name__ == '__main__':
    main()
