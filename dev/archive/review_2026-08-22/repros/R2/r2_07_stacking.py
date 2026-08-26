"""stacking.stack/unstack/tree_zip/apply_func_to_leaf_subtrees on heterogeneous trees and depth edges."""
import numpy as np
import t3toolbox.backend.stacking as S
def show(label, f):
    try:
        r = f(); print('%-72s -> %s' % (label, r))
    except Exception as e:
        print('%-72s -> %s: %s' % (label, type(e).__name__, str(e)[:100]))
a, b, c = np.arange(3.), np.arange(6.).reshape(2, 3), np.float64(7.)
T0 = (a, (b, c))
# heterogeneous second element: second tree's inner tuple replaced by a bare array whose [0] slice matches
T1 = (a + 10, b + 10)               # not the same structure as T0: b[0] has shape (3,), like a
show('stack(((a,(b,c)), (a2, b2)), (0,)) -- structure mismatch; silently?', lambda: [x.shape if hasattr(x, 'shape') else x for x in S.stack((T0, T1), (0,))[1]])
show('  value of leaf [1][0] second row (is b2[0], not b2)', lambda: S.stack((T0, T1), (0,))[1][0][1])
T2 = (a + 10, (b + 10, c + 10, np.ones(4)))   # extra leaf silently ignored
show('stack with extra leaf in second element', lambda: [x.shape for x in S.stack((T0, T2), (0,))[1]])
show('tree_depth(())', lambda: S.tree_depth(()))
show('tree_depth(((),))', lambda: S.tree_depth(((),)))
show('tree_depth("ab") (a str is a Sequence)', lambda: S.tree_depth('ab'))
show('tree_depth(((a,b),(c,)))', lambda: S.tree_depth(((a, b), (c,))))
show('tree_depth(((a,b),((c,),)))  (depth follows [0] path only)', lambda: S.tree_depth(((a, b), ((c,),))))
show('get_first_leaf(())', lambda: S.get_first_leaf(()))
show('tree_zip((1,2,3),(4,5)) truncates', lambda: S.tree_zip((1, 2, 3), (4, 5)))
show('tree_zip((1,(2,3)), (4, 5)) leaf vs subtree', lambda: S.tree_zip((1, (2, 3)), (4, 5)))
show('apply_func_to_leaf_subtrees((1,2), f, (None,(None,None)))', lambda: S.apply_func_to_leaf_subtrees((1, 2), lambda t: t, (None, (None, None))))
show('unstack negative axis, leaves of different ndim: A(4,2,3), B(2,3)', lambda: [x.shape for x in S.unstack((np.zeros((4, 2, 3)), np.zeros((2, 3))), axes=(-2,))[0]])
show('unstack axes=(0,) on leaves (2,3),(2,5)', lambda: [x.shape for x in S.unstack((np.zeros((2, 3)), np.zeros((2, 5))), axes=(0,))[1]])
show('unstack of an all-non-array tree', lambda: S.unstack((1, 2), axes=()))
# jax: stack with moveaxis(source=jnp.arange) eager and under jit
import jax, jax.numpy as jnp
TJ = ((jnp.ones((3, 2)), jnp.ones(4)), (jnp.zeros((3, 2)), jnp.zeros(4)))
show('jax stack axes=(1,)', lambda: [x.shape for x in S.stack(TJ, (1,))])
show('jax stack under jit axes=(1,)', lambda: [x.shape for x in jax.jit(lambda t: S.stack(t, (1,)))(TJ)])
show('jax unstack under jit', lambda: len(jax.jit(lambda s: S.unstack(s, (0,)))((jnp.ones((2, 3)), jnp.ones((2, 5))))))
# round trip with nested stack axes
T = (((a, (b, c)), (a, (b, c)), (a, (b, c))), ((a, (b, c)), (a, (b, c)), (a, (b, c))))
St = S.stack(T, (0, 1)); back = S.unstack(St, (0, 1))
show('stack/unstack round trip (2,3) equal', lambda: all(np.array_equal(x, y) for (x, (y1, z)), (x2, (y2, z2)) in zip([t for row in T for t in row], [t for row in back for t in row]) for x, y in [(x, x2), (y1, y2), (z, z2)]))
