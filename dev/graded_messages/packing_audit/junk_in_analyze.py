"""Summarize ``junk_in_plugin``'s JSON: ``python junk_in_analyze.py <out dir>``. One row per public uniform function:
the checks run, how many gave bitwise-equal outputs, how many differed (in the output's padding, or in real entries),
errors, and the functions the tests never called with a corruptible input."""
import glob
import importlib
import json
import sys
from collections import defaultdict

tot = defaultdict(lambda: defaultdict(float))
examples = {}
for p in glob.glob(sys.argv[1] + '/junk_*.json'):
    for name, r in json.load(open(p)).items():
        for k in ('checked', 'equal', 'padding-like', 'real', 'error', 'no-corruptible-input', 'variant-error'):
            tot[name][k] += r.get(k, 0)
        tot[name]['max_diff'] = max(tot[name]['max_diff'], r['max_diff'])
        if r['example'] and name not in examples:
            examples[name] = r['example']
print('%-62s %7s %6s %8s %5s %5s %8s' % ('function', 'checked', 'equal', 'padding', 'real', 'error', 'max diff'))
for name in sorted(tot, key=lambda n: (-(tot[n]['real'] + tot[n]['padding-like'] + tot[n]['error']), n)):
    r = tot[name]
    print('%-62s %7d %6d %8d %5d %5d %8.1e' % (name, r['checked'], r['equal'], r['padding-like'], r['real'], r['error'],
                                               r['max_diff']))
print()
for name, ex in sorted(examples.items()):
    print('example %-55s %s' % (name, ex[:150]))
seen = set(tot)
missing = []
for m in ('ut3_sampling', 'utv_sampling'):
    mod = importlib.import_module('t3toolbox.backend.' + m)
    missing += ['%s.%s' % (m, n) for n in mod.__all__ if '%s.%s' % (m, n) not in seen]
print()
print('never called by the tests run (%d): %s' % (len(missing), ', '.join(sorted(missing))))
never_checked = [n for n in seen if tot[n]['checked'] == 0]
print('called, but no corruptible input (%d): %s' % (len(never_checked), ', '.join(sorted(never_checked))))
