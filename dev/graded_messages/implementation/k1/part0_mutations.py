#!/usr/bin/env python3
"""Part 0's mutation pass (K1_DESIGN_NOTE.md section 6.4, the rows of part 0): each mutation applied alone to a
scratch copy of `t3toolbox/` and `tests/`, the named tests run there with `python -B`; a mutation is caught when they
fail. The copy is restored after each mutation.

    python3 part0_mutations.py SCRATCH_DIR          # SCRATCH_DIR: an empty folder (the copy is made there)
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
PY = sys.executable

MUTATIONS = [
    ('the constructor does not normalize order to Python ints',
     't3toolbox/backend/fitting.py',
     "        order = contractions._read_order(order, 'LeastSquaresBatch')",
     "        contractions._read_order(order, 'LeastSquaresBatch')",
     ['tests/test_least_squares_oracle.py::TestTheBatchConstructor']),
    ('make_newton_display does not refuse a validation batch of another kind or order',
     't3toolbox/backend/optimizer_display.py',
     "        if given != (getattr(batch, 'kind', None), getattr(batch, 'order', None)):",
     "        if False:",
     ['tests/backend/test_optimizer_display.py::TestMakeNewtonDisplay::test_a_validation_batch_of_another_kind_or_order']),
    ('_frontend_draw does not compare a drawn batch',
     't3toolbox/optimizers.py',
     "        _same_kind_and_order(batch, drawn, 'the batch a draw returned')",
     "        pass",
     ['tests/test_optimizers_frontend.py::TestOneKindAndOrderPerFit::test_a_drawn_batch_of_another_kind_or_order']),
    ('the frontend newton_cg does not compare its validation batch',
     't3toolbox/optimizers.py',
     "        _same_kind_and_order(batch, val_batch, 'newton_cg: the validation batch')",
     "        pass",
     ['tests/test_optimizers_frontend.py::TestOneKindAndOrderPerFit::test_a_validation_batch_of_another_kind_or_order']),
    ('_require_draw accepts a callable n_measurements',
     't3toolbox/backend/optimizers.py',
     "    if callable(n):",
     "    if False:",
     ['tests/backend/test_optimizers.py::TestOracleSeam::test_optional_outputs_are_checked_where_needed']),
    ('the oracle reads a fixed kind instead of the batch\'s',
     't3toolbox/backend/fitting.py',
     "            free = sampling_variables.computed_names(batch.kind, frame, batch.sample, engine_want, **args)",
     "            free = sampling_variables.computed_names('probe', frame, batch.sample, engine_want, **args)",
     ['tests/test_least_squares_oracle.py::TestQuery']),
    ('the pytree drops the order from the aux data (rebuilt with order ())',
     't3toolbox/backend/fitting.py',
     "        lambda aux, children: LeastSquaresBatch(*aux, **dict(zip(_BATCH_LEAVES, children))))",
     "        lambda aux, children: LeastSquaresBatch(aux[0], (), **dict(zip(_BATCH_LEAVES, children))))",
     ['tests/test_least_squares_oracle.py::TestTheBatchConstructor', 'tests/test_least_squares_oracle.py::TestTheBatch']),
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
    all_tests = sorted({t for *_, ts in MUTATIONS for t in ts})
    code, last = run(copy, all_tests)
    print('UNMUTATED: exit %d, %s' % (code, last))
    for label, path, old, new, tests in MUTATIONS:
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
