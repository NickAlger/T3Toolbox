"""Scratch copies for mutation testing. Run the tests of a copy from inside it with `python -B` (no stale bytecode).

    python mut.py copy SRC DST              # DST/{t3toolbox,tests,pyproject.toml} from SRC; no __pycache__
    python mut.py step DST STEP EPS         # sampling_steps.STEP's output scaled by (1 + EPS), every array of it
    python mut.py patch DST SPEC.json       # [{"file": rel, "old": text, "new": text}, ...]: each old exactly once
    python mut.py same DST SRC              # every file of DST equal to SRC's but the ones a mutation touched

A step mutant rebinds the module attribute: the engine looks its steps up by name when it runs them
(`getattr(sampling_steps, step_name)`), so a rebinding reaches every call; a step imported by name elsewhere
(`from sampling_steps import x`) would not be reached, so `step` checks that no file of the copy does that.
"""
import json
import os
import re
import shutil
import sys

STEPS = 't3toolbox/backend/sampling_steps.py'


def copy(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for d in ('t3toolbox', 'tests'):
        shutil.copytree(os.path.join(src, d), os.path.join(dst, d),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    shutil.copy(os.path.join(src, 'pyproject.toml'), dst)
    open(os.path.join(dst, '.mutations'), 'w').write('')
    print('copied %s -> %s' % (src, dst))


def log(dst, rel):
    with open(os.path.join(dst, '.mutations'), 'a') as f:
        f.write(rel + '\n')


def step(dst, name, eps):
    src = open(os.path.join(dst, STEPS)).read()
    assert re.search(r'^def %s\(' % name, src, re.M), 'no step %s' % name
    for d, _, fs in os.walk(os.path.join(dst, 't3toolbox')):
        for f in fs:
            if f.endswith('.py'):
                text = open(os.path.join(d, f)).read()
                assert not re.search(r'from t3toolbox\.backend\.sampling_steps import .*\b%s\b' % name, text), f
    with open(os.path.join(dst, STEPS), 'a') as f:
        f.write('\n\n_MUTANT_%s = %s\n\n\ndef %s(*a, **k):\n    out = _MUTANT_%s(*a, **k)\n'
                '    return tuple(x * (1 + %r) for x in out) if isinstance(out, tuple) else out * (1 + %r)\n'
                % (name, name, name, name, eps, eps))
    log(dst, STEPS)
    print('mutant: %s x (1 + %g)' % (name, eps))


def patch(dst, spec_path):
    for item in json.load(open(spec_path)):
        p = os.path.join(dst, item['file'])
        text = open(p).read()
        n = text.count(item['old'])
        assert n == 1, '%s: old text found %d times' % (item['file'], n)
        open(p, 'w').write(text.replace(item['old'], item['new']))
        log(dst, item['file'])
        print('patched %s' % item['file'])


def same(dst, src):
    touched = set(open(os.path.join(dst, '.mutations')).read().split())
    diff = []
    for d in ('t3toolbox', 'tests'):
        for root, _, fs in os.walk(os.path.join(dst, d)):
            if '__pycache__' in root:
                continue
            for f in fs:
                if f.endswith('.py'):
                    rel = os.path.relpath(os.path.join(root, f), dst)
                    other = os.path.join(src, rel)
                    if rel not in touched and (not os.path.exists(other) or open(other).read() !=
                                               open(os.path.join(root, f)).read()):
                        diff.append(rel)
    print('files differing from %s beyond the mutations: %s' % (src, diff or 'none'))


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    {'copy': lambda: copy(*args), 'step': lambda: step(args[0], args[1], float(args[2])),
     'patch': lambda: patch(*args), 'same': lambda: same(*args)}[cmd]()
