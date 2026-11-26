"""Summarize the strict-forms recorder's JSON: ``python strict_analyze.py <out dir> [events|sites|users|tests|auto]``.

``auto``: the tests (or doctests, pages, scripts) that rely on an AUTOMATIC conversion on the uniform side -- the
library converting on the caller's behalf -- by path and by file. Explicit packing (the caller calling ``pack_batch``,
``pack_sample``, ``pack_data`` or ``pack_if_ragged`` directly) is left out: option (a) keeps it."""
import glob
import json
import sys
from collections import Counter, defaultdict

rows = []
for p in glob.glob(sys.argv[1] + '/strict_*.json'):
    for k, v in json.load(open(p)):
        rows.append((tuple(k), v))
mode = sys.argv[2] if len(sys.argv) > 2 else 'events'


def kind_of(user):
    if user.startswith('tests/'):
        return 'test'
    if user.startswith('<doctest') or user.startswith('t3toolbox/'):
        return 'module doctest'
    if user.startswith('docs/') or user.endswith('.md') or user.endswith('.rst'):
        return 'doc page'
    if user.startswith('examples/'):
        return 'example'
    return 'other:' + user.split(':')[0]


if mode == 'events':        # per event: how many calls, distinct tests, by kind of origin
    agg = defaultdict(lambda: [0, set(), Counter()])
    for (ev, detail, imm, inner, outer, user, test), v in rows:
        a = agg[(ev, detail)]
        a[0] += v
        a[1].add(test)
        a[2][kind_of(user)] += v
    for (ev, detail), (n, tests, kinds) in sorted(agg.items()):
        print('%-60s %-8s calls=%-6d tests=%-4d %s' % (ev, detail, n, len(tests), dict(kinds)))
elif mode == 'sites':       # per event: the library caller relying on it (innermost outside the machinery, else immediate)
    agg = Counter()
    for (ev, detail, imm, inner, outer, user, test), v in rows:
        agg[(ev, detail, imm.split(' ')[0], inner)] += v
    for k, v in sorted(agg.items()):
        print('%-40s %-7s imm=%-50s lib=%s  x%d' % (k + (v,)))
elif mode == 'users':       # per origin file: distinct tests / lines
    agg = defaultdict(lambda: [0, set(), set()])
    for (ev, detail, imm, inner, outer, user, test), v in rows:
        f = user.split(':')[0]
        agg[f][0] += v
        agg[f][1].add(test)
        agg[f][2].add(ev.split(' ')[0])
    for f, (n, tests, evs) in sorted(agg.items(), key=lambda kv: -len(kv[1][1])):
        print('%-60s tests=%-4d calls=%-6d %s' % (f, len(tests), n, sorted(evs)))
elif mode == 'auto':
    def explicit(imm, outer):
        return imm == '-' or (outer.startswith('t3toolbox/backend/uniform_fitting.py')
                              and outer.split(' ')[-1] in ('pack_batch', 'pack_sample', 'pack_data'))
    groups, files, alltests = defaultdict(set), defaultdict(set), set()
    for (ev, detail, imm, inner, outer, user, test), v in rows:
        if not ev.startswith('U:') or explicit(imm, outer):
            continue
        alltests.add(test)
        files[test.split('::')[0]].add(test)
        if 'prepare_batch' in imm or 'prepare_batch' in inner:
            groups['the fitting frontend (prepare_batch)'].add(test)
        elif 'data_preparation' in imm:
            groups['ut3_unscale_coordinates (scales)'].add(test)
        elif 'uniform_fitting.py' in inner or 'backend/fitting.py' in inner:
            groups['the uniform kinds / the backend oracle given a ragged sample'].add(test)
        else:
            groups['sampling functions (engine, twins, frontend methods) given ragged vectors'].add(test)
    print('distinct tests relying on an automatic conversion:', len(alltests))
    for g, t in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print('  %-75s %d' % (g, len(t)))
    for f, t in sorted(files.items(), key=lambda kv: -len(kv[1])):
        print('  %-75s %d' % (f, len(t)))
elif mode == 'tests':
    for t in sorted({k[6] for k, v in rows}):
        print(t)
