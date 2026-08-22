"""common.randn / save_core_families / load_core_families / tree predicates / xcat edge cases."""
import io, numpy as np
import t3toolbox.backend.common as c
def show(label, f):
    try:
        r = f(); print('%-68s -> %s' % (label, r))
    except Exception as e:
        print('%-68s -> %s: %s' % (label, type(e).__name__, str(e)[:90]))
np.random.seed(0); a = c.randn(2, 3, use_jax=False); np.random.seed(0); b = c.randn(2, 3, use_jax=True)
show('randn numpy vs jax same stream (jax is np.random wrapped)', lambda: (type(a).__name__, type(b).__name__, bool(np.allclose(a, np.asarray(b)))))
show('randn dtype under jax (float32 default)', lambda: b.dtype)
fams = ((np.ones((2, 3)), np.zeros(4)), (np.arange(3.),))
buf = io.BytesIO(); c.save_core_families(buf, fams); buf.seek(0)
show('save/load round trip 2 families', lambda: [[x.shape for x in f] for f in c.load_core_families(buf)])
buf = io.BytesIO(); c.save_core_families(buf, ((np.ones(2),), (), (np.ones(3),))); buf.seek(0)
show('save/load with an EMPTY middle family', lambda: [[x.shape for x in f] for f in c.load_core_families(buf)])
buf = io.BytesIO(); c.save_core_families(buf, ((np.ones(2),), ())); buf.seek(0)
show('save/load with an EMPTY last family (family count)', lambda: len(c.load_core_families(buf)))
buf = io.BytesIO(); c.save_core_families(buf, ((np.ones(2),) * 12,)); buf.seek(0)
show('save/load 12 cores (key sort numeric, not lexicographic)', lambda: len(c.load_core_families(buf)[0]))
import jax.numpy as jnp
buf = io.BytesIO(); c.save_core_families(buf, ((jnp.ones(2),),)); buf.seek(0)
show('save jax cores -> loads as', lambda: type(c.load_core_families(buf)[0][0]).__name__)
show('tree_contains_jax("abc") (str is a Sequence)', lambda: c.tree_contains_jax('abc'))
show('tree_contains_jax((np.ones(2), "left"))', lambda: c.tree_contains_jax((np.ones(2), 'left')))
show('tree_to_jax((np.ones(2), "ab"))', lambda: c.tree_to_jax((np.ones(2), 'ab')))
show('tree_to_jax({"a": np.ones(2)}) (dict is not a Sequence)', lambda: type(c.tree_to_jax({'a': np.ones(2)})).__name__)
show('xcat((), [1,2]) returns list unchanged', lambda: c.xcat((), [1, 2]))
show('xcat(np.ones(2), [1.0]) (array + list)', lambda: c.xcat(np.ones(2), [1.0]))
show('items_are_uniform', lambda: (c.items_are_uniform((np.ones(2), np.ones(2))), c.items_are_uniform((np.ones(2), np.ones(3))), c.items_are_uniform(())))
show('partition/rebuild round trip', lambda: c.rebuild_static(*c.partition_static(((np.ones(2), 3, np.zeros(2, bool)), [True, np.ones(1)]))))
show('partition: numpy int array is dynamic, bool array static', lambda: c.partition_static((np.array([1, 2]), np.array([True])))[1].tree)
show('prefix_mask(np.array([[1,2],[0,3]]), 3).shape', lambda: c.prefix_mask(np.array([[1, 2], [0, 3]]), 3).shape)
show('prefix_mask with python tuple ranks', lambda: c.prefix_mask((1, 2), 3).astype(int).tolist())
