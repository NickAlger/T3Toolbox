"""Per-test comparison of two `pytest -v` logs, explained by the tests' own text. Independent of k0/tally.py.

    python tally2.py OLD_LOG[.gz] NEW_LOG[.gz] OLDROOT NEWROOT

1. Each log: node -> (outcome, subtests passed, subtests failed, subtests skipped). Accepts xdist lines
   ("[gw0] [ 3%] PASSED tests/...") and plain ones ("tests/... PASSED"); a node with two outcome lines (a failing
   teardown) keeps both.
2. Each test function of each root: its status from the text of the method, unitdiff.py's units (identical,
   changed, deleted, added). A deleted and an added test of the same class are reported as a possible rename.
3. Every node whose record differs between the logs, with the static status of its test. A difference on an
   IDENTICAL test is flagged: it changed through something else (a helper, a fixture, the library) and needs a
   reason. A CHANGED test with no difference is listed too (its edit kept its counts).
"""
import collections
import gzip
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unitdiff  # noqa: E402

OUT = r'(SUBPASSED|SUBFAILED|SUBSKIPPED|PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)'
XD = re.compile(r'^\[gw\d+\] \[\s*\d+%\] ' + OUT + r'(?:\[.*?\])? ?(?:\(.*?\))? (tests/\S+)')
PLAIN = re.compile(r'^(tests/\S+?)(?: \[.*?\])?(?: \(.*?\))? ' + OUT + r'\b')


def read_log(path):
    opener = gzip.open if path.endswith('.gz') else open
    rec = collections.defaultdict(collections.Counter)
    with opener(path, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('=') and any(w in line for w in ('FAILURES', 'ERRORS', 'warnings summary',
                                                                 'short test summary')):
                break                       # the report sections echo captured lines: not results
            m = XD.match(line)
            if m:
                rec[m.group(2)][m.group(1)] += 1
                continue
            m = PLAIN.match(line)
            if m:
                rec[m.group(1)][m.group(2)] += 1
    return rec


def summary(rec):
    t = collections.Counter()
    for c in rec.values():
        t.update(c)
    return dict(sorted(t.items()))


def test_status(old_root, new_root):
    """{'tests/file.py::Class::test': status} over both roots."""
    status = {}
    rels = sorted({r for r in unitdiff.py_files(old_root) + unitdiff.py_files(new_root) if r.startswith('tests/')})
    for rel in rels:
        po, pn = os.path.join(old_root, rel), os.path.join(new_root, rel)
        uo = unitdiff.units(open(po).read())[0] if os.path.exists(po) else {}
        un = unitdiff.units(open(pn).read())[0] if os.path.exists(pn) else {}
        for k in set(uo) | set(un):
            name = k.split('.')[-1]
            if not name.startswith('test') or k.endswith('<header>'):
                continue
            node = rel + '::' + k.replace('.', '::')
            if k in uo and k in un:
                status[node] = 'identical' if uo[k][2] == un[k][2] else 'CHANGED'
            else:
                status[node] = 'DELETED' if k in uo else 'ADDED'
    return status


def base(node):
    return node.split('[')[0]


def class_bases(roots):
    '''{(file, class): [base class names in the same file]} over the roots (for inherited tests).'''
    import ast
    out = {}
    for root in roots:
        for rel in unitdiff.py_files(root):
            if rel.startswith('tests/'):
                for n in ast.parse(open(os.path.join(root, rel)).read()).body:
                    if isinstance(n, ast.ClassDef):
                        out.setdefault((rel, n.name), [b.id for b in n.bases if isinstance(b, ast.Name)])
    return out


def lookup(node, st, bases):
    '''The static status of a node's test, following base classes of the same file.'''
    key = base(node)
    if key in st:
        return st[key]
    parts = key.split('::')
    if len(parts) == 3:
        rel, cls, name = parts
        todo, seen = list(bases.get((rel, cls), [])), set()
        while todo:
            b = todo.pop(0)
            if b in seen:
                continue
            seen.add(b)
            k = '%s::%s::%s' % (rel, b, name)
            if k in st:
                return st[k] + ' (inherited from %s)' % b
            todo += bases.get((rel, b), [])
    return '? (no test function found)'


def main(old_log, new_log, old_root, new_root):
    a, b = read_log(old_log), read_log(new_log)
    st = test_status(old_root, new_root)
    cb = class_bases((old_root, new_root))
    lost = [n for n in set(a) | set(b) if lookup(n, st, cb).startswith('?')]
    print('nodes with no test function found: %s' % (sorted(lost) or 'none'))
    print('old: %d nodes %s' % (len(a), summary(a)))
    print('new: %d nodes %s' % (len(b), summary(b)))
    groups = collections.defaultdict(list)
    for node in sorted(set(a) | set(b)):
        if a.get(node) != b.get(node):
            s = lookup(node, st, cb)
            groups[s].append(node)
    for s in sorted(groups, key=lambda s: (s.startswith('identical'), s)):
        print('\n== %s: %d nodes differ' % (s + ('  <-- needs a reason' if s.startswith('identical') else ''),
                                            len(groups[s])))
        for node in groups[s]:
            print('  %s\n      old %s\n      new %s' % (node, dict(a.get(node, {})), dict(b.get(node, {}))))
    same_changed = [n for n, s in st.items() if s == 'CHANGED' and n in a and a.get(n) == b.get(n)]
    print('\n== CHANGED tests whose record is the same in both logs: %d' % len(same_changed))
    for n in same_changed:
        print('  ' + n)
    dels = collections.defaultdict(list)
    adds = collections.defaultdict(list)
    for n, s in st.items():
        cls = n.rsplit('::', 1)[0]
        (dels if s == 'DELETED' else adds if s == 'ADDED' else {}).setdefault(cls, []).append(n.rsplit('::', 1)[1]) \
            if s in ('DELETED', 'ADDED') else None
    print('\n== possible renames (a deleted and an added test in one class):')
    for cls in sorted(set(dels) & set(adds)):
        print('  %s: %s -> %s' % (cls, dels[cls], adds[cls]))
    print('\n== totals of the static status: %s' % dict(collections.Counter(st.values())))


if __name__ == '__main__':
    main(*sys.argv[1:5])
