"""Reviewer 3: attack the BUILT LeastSquaresBatch and the fieldless oracle (K1 part 0).

    PYTHONPATH=<tree> <python> -B dev/graded_messages/implementation/k1/reviewer_tools/real_batch_attack.py
"""
import copy, dataclasses as dc, pickle, sys
import numpy as np
import jax, jax.numpy as jnp
from t3toolbox.backend.fitting import LeastSquaresBatch as LSB, LeastSquaresOracle


def expect(name, fn, exc, needle=None):
    try:
        fn()
    except exc as e:
        ok = needle is None or needle in str(e)
        print('%s %s -> %s: %s' % ('PASS' if ok else 'FAIL(msg)', name, type(e).__name__, str(e)[:90]))
        return
    except Exception as e:
        print('FAIL %s -> unexpected %s: %s' % (name, type(e).__name__, e)); return
    print('FAIL %s -> no error' % name)


def check(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)


print(sys.version.split()[0], 'jax', jax.__version__, 'numpy', np.__version__)
ww = [np.ones((4, 3)), np.ones((4, 5))]
y = [np.ones((4, 3)), np.ones((4, 5))]
expect('old LSB(ww, y)', lambda: LSB(ww, y), TypeError)
expect('old LSB(ww, y, w)', lambda: LSB(ww, y, np.ones(2)), TypeError, 'kind')
expect('old LSB(ww, y, w, 0.5)', lambda: LSB(ww, y, np.ones(2), 0.5), TypeError, 'kind')
expect('old entries LSB(index, y, w)', lambda: LSB(np.zeros((2, 4), int), np.ones(4), 1.0), TypeError, 'kind')
expect('5th positional', lambda: LSB('probe', (), ww, y, np.ones(2)), TypeError)
expect('bool order', lambda: LSB('probe', (True,), ww, y), TypeError)
expect('list order', lambda: LSB('probe', [1], ww, y), TypeError)
expect('int order', lambda: LSB('probe', 1, ww, y), TypeError)
expect('negative order', lambda: LSB('probe', (-1,), ww, y), ValueError)
expect('entries with jets', lambda: LSB('entries', (1,), ww, y), ValueError)
expect('unknown kind', lambda: LSB('nonsense', (), ww, y), ValueError)
expect('kind None', lambda: LSB(None, (), ww, y), TypeError)
c = LSB('apply', (np.int64(2),), ww, np.ones((3, 4)))
check('order normalized to Python ints', c.order == (2,) and type(c.order[0]) is int)
c2 = LSB('apply', (2,), ww, np.ones((3, 4)))
check('treedef equal for np.int64 and int order', jax.tree_util.tree_structure(c) == jax.tree_util.tree_structure(c2))
r = dc.replace(c, minibatch_scale=0.5)
check('replace keeps kind/order', (r.kind, r.order, r.minibatch_scale) == ('apply', (2,), 0.5))
expect('replace(kind="nope")', lambda: dc.replace(c, kind='nope'), ValueError)
expect('replace(order=[1])', lambda: dc.replace(c, order=[1]), TypeError)
expect('frozen', lambda: setattr(c, 'kind', 'probe'), dc.FrozenInstanceError)
check('deepcopy', copy.deepcopy(c).order == (2,))
check('pickle', pickle.loads(pickle.dumps(c)).kind == 'apply')
expect('== raises', lambda: c == c2, TypeError)
td = jax.tree_util.tree_structure(c)
p = jax.tree_util.tree_unflatten(td, [object()] * td.num_leaves)
check('unflatten with object() leaves', (p.kind, p.order) == ('apply', (2,)))
check('tree_map to in_axes spec', isinstance(jax.tree_util.tree_map(lambda _: 0, c), LSB))
check('aux is (kind, order)', jax.tree_util.tree_flatten(c)[1] == jax.tree_util.tree_flatten(c2)[1])
traces = []


@jax.jit
def f(b):
    traces.append((b.kind, b.order))
    return sum(jnp.sum(w) for w in b.sample) * jnp.sum(b.y)


f(c); f(c2); f(LSB('apply', (2,), [2 * w for w in ww], 3 * np.ones((3, 4))))
e = LSB('apply', (), ww, np.ones(4))
f(e); f(e.take(np.array([0, 1])))         # a drawn batch: another treedef (scale, indices leaves) -> one more trace
f(e.take(np.array([2, 3])))         # same treedef as the previous draw -> no trace
f(LSB('apply', (1,), ww, np.ones((2, 4)))); f(LSB('probe', (2,), ww, np.ones((3, 4))))
check('traces: one per kind/order (+1 for drawn batches): %s' % traces,
      traces == [('apply', (2,)), ('apply', ()), ('apply', ()), ('apply', (1,)), ('probe', (2,))])
d = e.take(np.array([0, 3]))
check('take keeps kind/order; scale n/b', (d.kind, d.order, d.minibatch_scale) == ('apply', (), 2.0))
check('n_measurements is an int property', isinstance(e.n_measurements, int) and e.n_measurements == 4)
o1, o2 = LeastSquaresOracle(), LeastSquaresOracle()
check('fieldless oracles equal and one hash', o1 == o2 and hash(o1) == hash(o2))
expect('old oracle form refused', lambda: LeastSquaresOracle('probe', (2,)), TypeError)


@dc.dataclass(frozen=True, eq=False)
class Sub(LeastSquaresOracle):
    pass


check('a subclass differs in eq and hash', Sub() != o1 and hash(Sub()) != hash(o1))
seen = []
g = jax.jit(lambda o, b: (seen.append(type(o).__name__), jnp.sum(b.y))[1], static_argnums=0)
g(o1, c); g(o2, c); g(Sub(), c)
check('jit static oracle: base twice one trace, subclass another: %s' % seen, seen == ['LeastSquaresOracle', 'Sub'])
