"""Append a wrapper that scales one step's output by (1 + eps): python mutate.py STEP EPS"""
import sys
step, eps = sys.argv[1], float(sys.argv[2])
with open('t3toolbox/backend/sampling_steps.py', 'a') as f:
    f.write('''

_MUTANT_ORIG = %s
def %s(*a, **k):
    out = _MUTANT_ORIG(*a, **k)
    return tuple(x * (1 + %r) for x in out) if isinstance(out, tuple) else out * (1 + %r)
''' % (step, step, eps, eps))
