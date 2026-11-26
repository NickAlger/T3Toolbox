"""Reviewer 4: mutations of the built reference (K1 part 1), each run against my dense pins and, when it is my turn
to run tests, against the part's tests.

    python3 ref_mutations.py TREE SCRATCH [--tests "tests/test_sampling_definitions.py ..."] [--only NAME]

TREE: the tree under review (the repository's working tree, or a `git archive` of it). SCRATCH: an empty folder; each
mutant is a copy of TREE's t3toolbox/ and tests/ there, run with `python -B`, then deleted.

MUTATIONS below is filled in when the built text is known (each patch is an exact text of the built file, which must
occur exactly once). The intents, written before the build (REVIEWER4_PLAN.md section 4):
  M1 the block row right-aligned (reshaped onto the trailing axes)       -> pins: w_tilde, f, sums, g, JtJp (F == W == N)
  M2 the block row of another mode (b[(i + 1) % d])                       -> pins: every weighted variable
  M3 the block table transposed (b.T at a square (d,)+F)                  -> pins
  M4 the block weight unsquared, or the full weight unsquared             -> pins
  M5 the block row read after the turn (b[k] where b[order - k] is meant) -> pins: w_tilde, g, Kp, the tilde laws
  M6 one seeded function passes None for the block (pick one: Kp)          -> pins: the law for that function
  M7 the block and the full weight swapped at one call of _the_seed        -> structural refusal, or the law
  M8 y allowed a size-1 W axis (may_be_one for y)                          -> refusal check
  M9 the full weight allowed a size-1 structural axis (the edge)           -> refusal check
  M10 a block weight sequence accepted (the TypeError dropped)             -> refusal check
  M11 the exact check written as "no 1 on a structural axis"               -> the degenerate configuration
"""
import argparse
import os
import shutil
import subprocess
import sys

PY = '/home/nick/miniconda3/envs/t3toolbox/bin/python'
HERE = os.path.dirname(os.path.abspath(__file__))
REF = 't3toolbox/backend/sampling_definitions.py'

MUTATIONS = [
    # (name, path, exact old text, new text)
    ('M1', REF, "squared = xnp.reshape(row ** 2, tuple(row.shape) + (1,) * (len(shape) - m))",
     "squared = xnp.reshape(row ** 2, (1,) * (len(shape) - row.ndim) + tuple(row.shape))"),
    ('M2', REF, "row = xnp.asarray(block_weight if i is None else block_weight[i])",
     "row = xnp.asarray(block_weight if i is None else block_weight[(i + 1) % S.d])"),
    ('M3', REF, "row = xnp.asarray(block_weight if i is None else block_weight[i])",
     "row = xnp.asarray(block_weight if i is None else xnp.asarray(block_weight).T[i])"),
    ('M4b', REF, "squared = xnp.reshape(row ** 2, tuple(row.shape)", "squared = xnp.reshape(row, tuple(row.shape)"),
    ('M4f', REF, "squared = squared * f_X ** 2", "squared = squared * f_X"),
    ('M5', REF, "lambda i, jet_TX: _turn(S, _weight_squared(S, block_weight, full_weight, i, jet_TX.shape)\n"
                "                                                     * jet_TX), jet)",
     "lambda i, jet_TX: _weight_squared(S, block_weight, full_weight, i, jet_TX.shape)\n"
     "                                                     * _turn(S, jet_TX), jet)"),
    ('M6', REF, "_the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'Kp')",
     "_the_seed(S, sample, y, None, full_weight, w_tilde, 'Kp')"),
    ('M7', REF, "_the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'nu_L_tilde')",
     "_the_seed(S, sample, y, full_weight, block_weight, w_tilde, 'nu_L_tilde')"),
    ('M8', REF, "_check_data_space('the data y', i, np.shape(y_TX), np.shape(z_TX), ())",
     "_check_data_space('the data y', i, np.shape(y_TX), np.shape(z_TX), tuple(range(len(S.order), len(S.M))))"),
    ('M9', REF, "tuple(range(m, len(S.M))))   # the sample axes W", "tuple(range(m, len(S.M) + 1)))   # the sample axes W"),
    ('M10', REF, "    if isinstance(block_weight, (tuple, list)):\n        raise TypeError(",
     "    if False:\n        raise TypeError("),
    ('M11', REF, "any(a != b and not (a == 1 and axis in may_be_one)",
     "any((a != b or a == 1) and not (a == 1 and axis in may_be_one)"),
]


def run(cmd, cwd, env=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, shell=isinstance(cmd, str))
    return p.returncode, (p.stdout + p.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tree')
    ap.add_argument('scratch')
    ap.add_argument('--tests', default='')
    ap.add_argument('--only')
    args = ap.parse_args()
    if not MUTATIONS:
        sys.exit('MUTATIONS is empty: fill it from the built text first')
    for name, path, old, new in MUTATIONS:
        if args.only and name != args.only:
            continue
        root = os.path.join(args.scratch, 'mut_' + name)
        if os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root)
        for top in ('t3toolbox', 'tests'):
            shutil.copytree(os.path.join(args.tree, top), os.path.join(root, top),
                            ignore=shutil.ignore_patterns('__pycache__'))
        target = os.path.join(root, path)
        text = open(target).read()
        if text.count(old) != 1:
            print('%-6s the patch text occurs %d times: not applied' % (name, text.count(old)))
            shutil.rmtree(root)
            continue
        open(target, 'w').write(text.replace(old, new))
        env = dict(os.environ, PYTHONPATH=root)
        code, out = run([PY, '-B', os.path.join(HERE, 'ref_dense_pins.py'), '--api', 'new'], root, env)
        nfail = sum(1 for l in out.splitlines() if l.startswith('FAIL'))
        pins = ('caught (%d FAIL lines before the end; %s)' % (nfail, out.strip().splitlines()[-1][:90]) if code
                else 'NOT caught')
        line = '%-6s pins: %s' % (name, pins)
        if args.tests:
            code, out = run('%s -B -m pytest -x -q -n 2 -p no:cacheprovider %s' % (PY, args.tests), root, env)
            tail = [l for l in out.splitlines() if l.strip()][-1:]
            line += ' | tests: %s %s' % ('caught' if code else 'NOT caught', tail[0][:80] if tail else '')
        print(line)
        shutil.rmtree(root)


if __name__ == '__main__':
    main()
