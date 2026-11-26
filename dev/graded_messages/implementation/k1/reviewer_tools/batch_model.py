"""Reviewer 3: try to break the batch of K1's design note §4.6 (a model of the class, outside the library).

Runs on the project env (py3.11, jax 0.10) and the floor env (tttt: py3.9, jax 0.4.30):
    PYTHONPATH=$PWD <python> -B dev/graded_messages/implementation/k1/reviewer_tools/batch_model.py
"""
import copy, dataclasses as dc, numbers, pickle, sys, typing as typ
import numpy as np
import jax, jax.numpy as jnp

KINDS = ('probe', 'apply', 'entries')


def _read_order(order, what):                     # contractions._read_order, verbatim in substance
    if not isinstance(order, tuple):
        raise TypeError('%s: order must be a tuple, got %r' % (what, order))
    for n in order:
        if not isinstance(n, numbers.Integral) or isinstance(n, bool):
            raise TypeError('%s: integer entries, got %r' % (what, order))
        if n < 0:
            raise ValueError('%s: entries >= 0, got %r' % (what, order))
    return tuple(int(n) for n in order)


@dc.dataclass(frozen=True, eq=False, init=False)
class LSB:
    kind: str
    order: typ.Tuple[int, ...]
    sample: typ.Any
    y: typ.Any = None
    block_weight: typ.Any = None
    full_weight: typ.Any = None
    minibatch_scale: typ.Any = None
    drawn_indices: typ.Any = None

    def __init__(self, kind, order, sample, y=None, *, block_weight=None, full_weight=None, minibatch_scale=None,
                 drawn_indices=None):
        if not isinstance(kind, str):
            raise TypeError('LeastSquaresBatch: kind is a str, got %s' % type(kind).__name__)
        if kind not in KINDS:
            raise ValueError('LeastSquaresBatch: unknown kind %r' % kind)
        order = _read_order(order, 'LeastSquaresBatch')
        if kind == 'entries' and order != ():
            raise ValueError('entries: order ()')
        for name, value in (('kind', kind), ('order', order), ('sample', sample), ('y', y),
                            ('block_weight', block_weight), ('full_weight', full_weight),
                            ('minibatch_scale', minibatch_scale), ('drawn_indices', drawn_indices)):
            object.__setattr__(self, name, value)

    def __eq__(self, other):
        raise TypeError('no ==')
    __hash__ = None


LEAVES = ('sample', 'y', 'block_weight', 'full_weight', 'minibatch_scale', 'drawn_indices')
jax.tree_util.register_pytree_node(LSB, lambda b: (tuple(getattr(b, n) for n in LEAVES), (b.kind, b.order)),
                                   lambda aux, ch: LSB(*aux, **dict(zip(LEAVES, ch))))


def expect(name, fn, exc):
    try:
        fn()
    except exc as e:
        print('PASS %s -> %s: %s' % (name, type(e).__name__, str(e)[:70]))
        return
    except Exception as e:
        print('FAIL %s -> unexpected %s: %s' % (name, type(e).__name__, e))
        return
    print('FAIL %s -> no error' % name)


print(sys.version.split()[0], 'jax', jax.__version__, 'numpy', np.__version__)
ww = [np.ones((4, 3)), np.ones((4, 5))]
y = [np.ones((4, 3)), np.ones((4, 5))]
b = LSB('probe', (np.int64(0),) * 0, ww, y, block_weight=np.ones(2))
print('repr ok:', repr(b)[:60])
expect('old LSB(ww, y)', lambda: LSB(ww, y), TypeError)
expect('old LSB(ww, y, w)', lambda: LSB(ww, y, np.ones(2)), TypeError)
expect('old LSB(ww, y, w, 0.5)', lambda: LSB(ww, y, np.ones(2), 0.5), TypeError)
expect('old entries LSB(index_array, y, w)', lambda: LSB(np.zeros((3, 4), int), np.ones(4), 1.0), TypeError)
expect('LSB(kind, order, ww, y, w) 5th positional', lambda: LSB('probe', (), ww, y, np.ones(2)), TypeError)
expect('bool in order', lambda: LSB('probe', (True,), ww, y), TypeError)
expect('entries with jets', lambda: LSB('entries', (1,), ww, y), ValueError)
expect('order a list', lambda: LSB('probe', [1], ww, y), TypeError)
# numpy ints normalized
c = LSB('apply', (np.int64(2),), ww, np.ones((3, 4)))
print('PASS order normalized:', c.order, type(c.order[0]).__name__) if type(c.order[0]) is int else print('FAIL order type')
# replace
r = dc.replace(c, minibatch_scale=0.5)
print('PASS replace keeps kind/order and sets scale' if (r.kind, r.order, r.minibatch_scale) == ('apply', (2,), 0.5) else 'FAIL replace')
expect('replace(kind="nope") re-validates', lambda: dc.replace(c, kind='nope'), ValueError)
expect('replace(weight=...) (an old field name)', lambda: dc.replace(c, weight=1.0), TypeError)
expect('frozen', lambda: setattr(c, 'kind', 'probe'), dc.FrozenInstanceError)
print('PASS deepcopy' if copy.deepcopy(c).order == (2,) else 'FAIL deepcopy')
print('PASS pickle' if pickle.loads(pickle.dumps(c)).kind == 'apply' else 'FAIL pickle')
# placeholder leaves (jax rebuilds with object(), None, ints: in_axes specs)
td = jax.tree_util.tree_structure(c)
n = td.num_leaves
p = jax.tree_util.tree_unflatten(td, [object()] * n)
print('PASS unflatten with object() leaves:', p.kind, p.order)
spec = jax.tree_util.tree_map(lambda _: 0, c)
print('PASS tree_map to an in_axes spec:', type(spec).__name__)
# treedef equality across a numpy-int order and an int order
c2 = LSB('apply', (2,), ww, np.ones((3, 4)))
print('PASS same treedef for (np.int64(2),) and (2,)' if jax.tree_util.tree_structure(c) == jax.tree_util.tree_structure(c2)
      else 'FAIL treedefs differ')
# compile once across two batches of one kind/order; again for another order
traces = []


@jax.jit
def f(batch):
    traces.append((batch.kind, batch.order))
    return sum(jnp.sum(w) for w in batch.sample) * (1.0 if batch.y is None else jnp.sum(batch.y))


f(c); f(c2); f(LSB('apply', (2,), [2 * w for w in ww], 3 * np.ones((3, 4))))
f(LSB('apply', (1,), ww, np.ones((2, 4))))
f(LSB('probe', (2,), ww, np.ones((3, 4))))
print('PASS traces' if traces == [('apply', (2,)), ('apply', (1,)), ('probe', (2,))] else 'FAIL traces %s' % traces)
# jit with block_weight a Python float (entries) and None leaves
e = LSB('entries', (), np.zeros((2, 4), int), np.ones(4), block_weight=0.5)
print('PASS jit over entries batch with a float block weight:', float(jax.jit(lambda bb: bb.block_weight * jnp.sum(bb.y))(e)))
# vmap over a batch is not meaningful (W is not leading on every leaf); just check vmap with a spec does not raise in unflatten
print('PASS hash is None' if LSB.__hash__ is None else 'FAIL hash')
