import dataclasses as dc, typing as typ
import numpy as np, jax, jax.numpy as jnp

@dc.dataclass(frozen=True, eq=False, init=False)
class B:
    kind: str
    order: typ.Tuple[int, ...]
    sample: typ.Any
    y: typ.Any = None
    block_weight: typ.Any = None
    full_weight: typ.Any = None
    minibatch_scale: typ.Any = None
    drawn_indices: typ.Any = None
    def __init__(self, kind, order, sample, y=None, *, block_weight=None, full_weight=None, minibatch_scale=None, drawn_indices=None):
        if kind not in ('probe','apply','entries'): raise ValueError('kind %r' % (kind,))
        for name, value in (('kind', kind), ('order', tuple(int(n) for n in order)), ('sample', sample), ('y', y),
                            ('block_weight', block_weight), ('full_weight', full_weight),
                            ('minibatch_scale', minibatch_scale), ('drawn_indices', drawn_indices)):
            object.__setattr__(self, name, value)
LEAVES = ('sample','y','block_weight','full_weight','minibatch_scale','drawn_indices')
jax.tree_util.register_pytree_node(B, lambda b: (tuple(getattr(b,n) for n in LEAVES), (b.kind, b.order)),
                                   lambda aux, ch: B(*aux, **dict(zip(LEAVES, ch))))
b = B('probe', (1,), [np.ones(3)], [np.ones(3)], block_weight=np.ones((1,2)))
print(dc.replace(b, minibatch_scale=0.5))
try: B('probe', (), 1, 2, 3)
except TypeError as e: print('TypeError', e)
try: B([np.ones(3)], [np.ones(3)])
except (ValueError, TypeError) as e: print(type(e).__name__, e)
c = jax.tree_util.tree_map(jnp.asarray, b); print(type(c.sample[0]), c.kind, c.order)
f = jax.jit(lambda bb: bb.sample[0].sum() * bb.block_weight.sum()); print(f(b))
try: b.kind = 'x'
except dc.FrozenInstanceError as e: print('frozen ok')
print([f.name for f in dc.fields(b)])
