"""Do fixed-slot tuples with None for absent grades pass through the library's static split and jit?"""
import sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, sys.argv[1])
import t3toolbox.backend.common as common

mask = np.array([True, True, False])
msg = (jnp.ones((4, 3)), None, 2.0 * jnp.ones((4, 3)), None)          # (plain, t, z, tz)
sweep = ((msg, msg), (5, 3), mask)                                    # components, shape, mask

dyn, skel = common.partition_static(sweep)
print('tuple sweep: dynamic leaves', [None if a is None else a.shape for a in dyn])
back = common.rebuild_static(dyn, skel)
print('  round trip keeps None slots:', back[0][0][1] is None and back[0][0][3] is None,
      '| shape static:', back[1] == (5, 3), '| mask is host numpy:', type(back[2]) is np.ndarray)

@jax.jit
def f(dyn):
    s = common.rebuild_static(dyn, skel)
    plain, t, z, tz = s[0][0]
    n = s[1][0]                                                      # host int, usable in shape arithmetic
    return plain.sum() + z.sum() + n * int(s[2].sum()) + (0.0 if t is None else t.sum())
print('  under jit:', float(f(dyn)))

as_dict = (({'': jnp.ones((4, 3)), 'z': jnp.ones((4, 3)), 'shape': (5, 3), 'mask': mask},), (5, 3), mask)
dyn2, skel2 = common.partition_static(as_dict)
print('dict sweep: dynamic leaves', [type(a).__name__ for a in dyn2],
      '-> the dict is one leaf, with its shape and mask inside')
