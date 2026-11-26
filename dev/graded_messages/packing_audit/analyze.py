"""Read the JSON the audit plugin writes; see sampling_audit_plugin.py for usage."""
import json, glob, sys
from collections import defaultdict, Counter
A = sys.argv[1]
calls, convs, shadows = Counter(), Counter(), Counter()
for p in glob.glob(A + '/audit_*.json'):
    j = json.load(open(p))
    for k, v in j['calls']:
        name, d0, lay, fs, oc = k
        calls[(name, d0, lay, tuple(fs), tuple(oc))] += v
    for k, v in j.get('shadows', []):
        shadows[tuple(k)] += v
    for name, stack, f, caller, v in j['convs']:
        convs[(name, tuple((s[0], s[1], tuple(s[2])) for s in stack), f, caller)] += v

def cls(fs):
    if not fs: return 'novec'
    if all(x.endswith('=arr') for x in fs): return 'packed'
    if all(x.endswith('=seq') for x in fs): return 'ragged'
    return 'MIXED'

mode = sys.argv[2]
if mode == 'table':
    by = defaultdict(lambda: defaultdict(Counter))
    for (name, d0, lay, fs, oc), v in calls.items():
        by[(lay, name)][cls(fs)][oc] += v
    for (lay, name) in sorted(by):
        row = []
        for c in ('packed', 'ragged', 'novec', 'MIXED'):
            if c in by[(lay, name)]:
                row.append('%s->{%s}' % (c, ' | '.join(','.join(o) for o in by[(lay, name)][c])))
        print('%-8s %-62s %s' % (lay, name, '  '.join(row)))
elif mode == 'shadows':
    for k, v in sorted(shadows.items()):
        print(v, ' '.join(k))
elif mode == 'convs':
    for (name, stack, f, caller), v in sorted(convs.items(), key=lambda kv: (kv[0][0], kv[0][3])):
        print('%s  arg0=%s  at %s  x%d' % (name, f, caller, v))
        print('      stack: ' + ' > '.join('%s[%s;%s]' % (s[0], s[1], ','.join(s[2])) for s in stack))
