"""Planted conversions for diff_audit.audit_c: a right one passes, each wrong one is caught."""
import sys
sys.path.insert(0, '/home/nick/repos/T3Toolbox/dev/graded_messages/implementation/k1')
import diff_audit as da

OLD = '''
def test_x(self):
    b = 0.5 + rng.random(3)
    w = blocks_to_weight('apply', (2,), b, y)
    e = sv.sampling_variables('apply', x, ww, names, order=(2,), y=y, weight=w)
    r = sdef.f('apply', frame, ww, order=(2,), y=y, weight=w)
    s = sdef.g('apply', frame, ww, order=(2,), y=y, weight=per_sample)
'''
RIGHT = '''
def test_x(self):
    b = 0.5 + rng.random(3)
    w = blocks_to_weight('apply', (2,), b, y)
    e = sv.sampling_variables('apply', x, ww, names, order=(2,), y=y, weight=w)
    r = sdef.f('apply', frame, ww, order=(2,), y=y, block_weight=b)
    s = sdef.g('apply', frame, ww, order=(2,), y=y, full_weight=per_sample)
'''
WRONG_FULL = RIGHT.replace('block_weight=b', 'full_weight=w')               # a block table passed as a full weight
WRONG_BLOCK = RIGHT.replace('block_weight=b', 'block_weight=c')              # a block weight of another value
REMOVED = '''
def test_x(self):
    b = 0.5 + rng.random(3)
    e = sv.sampling_variables('apply', x, ww, names, order=(2,), y=y, full_weight=jnk)
    r = sdef.f('apply', frame, ww, order=(2,), y=y, block_weight=c)
    s = sdef.g('apply', frame, ww, order=(2,), y=y, full_weight=per_sample)
'''
reasons = {('tests/t.py', 'test_x', 'g', 'full_weight', 'per_sample'): ('apply', 'a weight per sample')}
for label, new, want_bad in (('right', RIGHT, 0), ('a block table as full_weight', WRONG_FULL, 1),
                             ('a block weight of another value', WRONG_BLOCK, 1),
                             ('the table removed, the engine on full_weight', REMOVED, 3)):
    report, used = [], set()
    bad = da.audit_c('tests/t.py', da.scopes_of('tests/t.py', OLD), da.scopes_of('tests/t.py', new), reasons, used,
                     report)
    print('%-48s bad=%d (want %d) %s' % (label, bad, want_bad, 'OK' if bad == want_bad else 'MISMATCH'))
    for line in report:
        print('    ' + line)
