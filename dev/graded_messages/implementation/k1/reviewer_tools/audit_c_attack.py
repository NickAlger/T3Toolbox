"""Reviewer 4: attack the implementer's weights audit (diff_audit.py, audit (c)) with planted conversions.

    python3 audit_c_attack.py PATH/TO/diff_audit.py [--keep DIR]

Builds a scratch git repository laid out like this one (the audit finds the repository four levels above itself),
commits a base test file per case, writes the case's conversion over it, and runs the audit on that file alone with
`--part 1`. Each case says what a sound audit must do: PASS (a right conversion), FAIL (a wrong one it must catch),
or LIST (a blind spot the reviewer must cover by reading: the audit passes, the entry it lists is named).
Standard library only; never imports the library.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HEAD = 'import numpy as np\nimport t3toolbox.backend.jets as jets\nimport t3toolbox.backend.sampling_definitions as sdef\n\n\n'

CASES = [
    # (name, expectation, base body, new body, reason lines, what the audit should show)
    ('block, direct', 'PASS',
     "def test_x():\n    y, b = 1, 2\n    sdef.f('probe', F, S, y=y, weight=jets.blocks_to_weight('probe', (2,), b, y))\n",
     "def test_x():\n    y, b = 1, 2\n    sdef.f('probe', F, S, y=y, block_weight=b)\n", [], 'BTW-OK'),
    ('block, through one assignment', 'PASS',
     "def test_x():\n    y, b = 1, 2\n    w = jets.blocks_to_weight('probe', (2,), b, y)\n    sdef.f('probe', F, S, y=y, weight=w)\n",
     "def test_x():\n    y, b = 1, 2\n    sdef.f('probe', F, S, y=y, block_weight=b)\n", [], 'BTW-OK'),
    ('block passed as a full weight (apply: legal, same numbers)', 'FAIL',
     "def test_x():\n    y, b = 1, 2\n    sdef.f('apply', F, S, y=y, weight=jets.blocks_to_weight('apply', (2,), b, y))\n",
     "def test_x():\n    y, b = 1, 2\n    sdef.f('apply', F, S, y=y, full_weight=b)\n", [], 'BTW-UNPAIRED, F-NOREASON'),
    ('the table of blocks_to_weight passed whole as a full weight', 'FAIL',
     "def test_x():\n    y, b = 1, 2\n    sdef.f('apply', F, S, y=y, weight=jets.blocks_to_weight('apply', (2,), b, y))\n",
     "def test_x():\n    y, b = 1, 2\n    sdef.f('apply', F, S, y=y, full_weight=jets.blocks_to_weight('apply', (2,), b, y))\n", [],
     'F-NOREASON (blocks_to_weight is still in the scope, so BTW pairs nothing)'),
    ('block of another value', 'FAIL',
     "def test_x():\n    y, b, c = 1, 2, 3\n    sdef.f('probe', F, S, y=y, weight=jets.blocks_to_weight('probe', (2,), b, y))\n",
     "def test_x():\n    y, b, c = 1, 2, 3\n    sdef.f('probe', F, S, y=y, block_weight=c)\n", [], 'BTW-UNPAIRED, B-NOREASON'),
    ('a full weight with no reason line', 'FAIL',
     "def test_x():\n    y, fw = 1, 2\n    sdef.f('probe', F, S, y=y, weight=fw)\n",
     "def test_x():\n    y, fw = 1, 2\n    sdef.f('probe', F, S, y=y, full_weight=fw)\n", [], 'F-NOREASON'),
    ('a full weight with a reason line', 'PASS',
     "def test_x():\n    y, fw = 1, 2\n    sdef.f('probe', F, S, y=y, weight=fw)\n",
     "def test_x():\n    y, fw = 1, 2\n    sdef.f('probe', F, S, y=y, full_weight=fw)\n",
     ['{path}\ttest_x\tf\tfull_weight\tfw\tprobe\ta weight per entry'], 'F-REASON'),
    ('a weight dropped (the test loses its subject)', 'LIST',
     "def test_x():\n    y, fw = 1, 2\n    sdef.f('probe', F, S, y=y, weight=fw)\n",
     "def test_x():\n    y, fw = 1, 2\n    sdef.f('probe', F, S, y=y)\n", [], 'W-GONE'),
    ('the product split into the pair', 'PASS',
     "def test_x():\n    y, b, fw = 1, 2, 3\n    sdef.f('probe', F, S, y=y, weight=jets.blocks_to_weight('probe', (2,), b, y) * fw)\n",
     "def test_x():\n    y, b, fw = 1, 2, 3\n    sdef.f('probe', F, S, y=y, block_weight=b, full_weight=fw)\n",
     ['{path}\ttest_x\tf\tfull_weight\tfw\tprobe\tthe full factor of the product'], 'BTW-OK, F-REASON'),
    ('the pair swapped', 'FAIL',
     "def test_x():\n    y, b, fw = 1, 2, 3\n    sdef.f('probe', F, S, y=y, weight=jets.blocks_to_weight('probe', (2,), b, y) * fw)\n",
     "def test_x():\n    y, b, fw = 1, 2, 3\n    sdef.f('probe', F, S, y=y, block_weight=fw, full_weight=b)\n", [],
     'BTW-UNPAIRED, B-NOREASON, F-NOREASON'),
    ('a weight by a string key', 'LIST',
     "def test_x():\n    y, b = 1, 2\n    sdef.f('apply', F, S, y=y, weight=jets.blocks_to_weight('apply', (2,), b, y))\n",
     "def test_x():\n    y, b = 1, 2\n    sdef.f('apply', F, S, y=y, **{'full_weight': b})\n", [],
     'BTW-UNPAIRED at least (the string key itself is invisible)'),
    ('a helper parameter renamed and passed through', 'FAIL',
     "def helper(y, weight):\n    return sdef.f('probe', F, S, y=y, weight=weight)\n\n\ndef test_x():\n    helper(1, 2)\n",
     "def helper(y, full_weight):\n    return sdef.f('probe', F, S, y=y, full_weight=full_weight)\n\n\ndef test_x():\n    helper(1, 2)\n",
     [], 'F-NOREASON (the reason must name the callers)'),
    ('the engine keeps the old weight, the reference gets the pair (part 1 form)', 'PASS',
     "def test_x():\n    y, b = 1, 2\n    w = jets.blocks_to_weight('probe', (2,), b, y)\n    sv.sampling_variables('probe', F, S, ('f',), y=y, weight=w)\n    sdef.f('probe', F, S, y=y, weight=w)\n",
     "def test_x():\n    y, b = 1, 2\n    w = jets.blocks_to_weight('probe', (2,), b, y)\n    sv.sampling_variables('probe', F, S, ('f',), y=y, weight=w)\n    sdef.f('probe', F, S, y=y, block_weight=b)\n",
     [], 'B-PAIRED'),
    ('the engine keeps the old weight, the reference gets ANOTHER block', 'FAIL',
     "def test_x():\n    y, b, c = 1, 2, 3\n    w = jets.blocks_to_weight('probe', (2,), b, y)\n    sv.sampling_variables('probe', F, S, ('f',), y=y, weight=w)\n    sdef.f('probe', F, S, y=y, weight=w)\n",
     "def test_x():\n    y, b, c = 1, 2, 3\n    w = jets.blocks_to_weight('probe', (2,), b, y)\n    sv.sampling_variables('probe', F, S, ('f',), y=y, weight=w)\n    sdef.f('probe', F, S, y=y, block_weight=c)\n",
     [], 'B-NOREASON'),
]


def git(root, *args):
    return subprocess.run(['git', '-C', root] + list(args), check=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audit')
    ap.add_argument('--keep')
    args = ap.parse_args()
    root = args.keep or tempfile.mkdtemp(prefix='audit_c_attack_')
    if os.path.exists(root):
        shutil.rmtree(root)
    tool_dir = os.path.join(root, 'dev', 'graded_messages', 'implementation', 'k1')
    os.makedirs(tool_dir)
    os.makedirs(os.path.join(root, 'tests'))
    shutil.copy(args.audit, os.path.join(tool_dir, 'diff_audit.py'))
    subprocess.run(['git', 'init', '-q', root], check=True)
    paths = []
    for k, (name, _e, base, _n, _r, _w) in enumerate(CASES):
        path = 'tests/test_case%02d.py' % k
        paths.append(path)
        open(os.path.join(root, path), 'w').write(HEAD + base)
    git(root, 'add', '-A')
    git(root, '-c', 'user.name=r', '-c', 'user.email=r@example.invalid', 'commit', '-qm', 'base')
    for k, (name, _e, _b, new, _r, _w) in enumerate(CASES):
        open(os.path.join(root, paths[k]), 'w').write(HEAD + new)
    wrong = 0
    for k, (name, expect, _b, _n, reasons, what) in enumerate(CASES):
        tsv = os.path.join(root, 'reasons_%02d.tsv' % k)
        open(tsv, 'w').write(''.join(r.format(path=paths[k]) + '\n' for r in reasons))
        out = os.path.join(root, 'out_%02d.txt' % k)
        p = subprocess.run([sys.executable, os.path.join(tool_dir, 'diff_audit.py'), 'HEAD', '--part', '1',
                            '--reasons', tsv, '--files', paths[k], '--out', out], capture_output=True, text=True)
        text = open(out).read() if os.path.exists(out) else p.stdout + p.stderr
        entries = sorted({line.split()[0] for line in text.splitlines()[5:] if line.strip() and
                          not line.startswith('LINE')})
        got = 'FAIL' if p.returncode else 'PASS'
        ok = got == expect or (expect == 'LIST' and got == 'PASS')
        if expect == 'LIST' and got == 'FAIL':
            ok = True                                   # caught after all: better than a blind spot
        wrong += not ok
        print('%-4s %-70s expected %-4s got %-4s entries %s   (should show: %s)'
              % ('ok' if ok else 'MISS', name[:70], expect, got, ','.join(entries), what))
    if not args.keep:
        shutil.rmtree(root)
    print('%d of %d cases as a sound audit should be' % (len(CASES) - wrong, len(CASES)))
    sys.exit(1 if wrong else 0)


if __name__ == '__main__':
    main()
