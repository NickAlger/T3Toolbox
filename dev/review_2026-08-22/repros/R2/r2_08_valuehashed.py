"""ValueHashedFields / ValueHashedMasks hash/eq edge cases."""
import dataclasses as dc, numpy as np
from t3toolbox.backend.common import ValueHashedFields, ValueHashedMasks
@dc.dataclass(frozen=True, eq=False)
class F(ValueHashedFields):
    a: object = None
    b: object = None
@dc.dataclass(frozen=True, eq=False)
class M(ValueHashedMasks):
    data: tuple = ()
def show(label, f):
    try:
        r = f(); print('%-70s -> %s' % (label, r))
    except Exception as e:
        print('%-70s -> %s: %s' % (label, type(e).__name__, str(e)[:90]))
m1 = M((np.array([True, False]),)); m2 = M((np.array([1, 0], dtype=np.int8),))
show('Masks: bool vs int8 same values: eq', lambda: m1 == m2)
show('Masks: bool vs int8 same values: hash equal', lambda: hash(m1) == hash(m2))
m3 = M((np.array([True, False]), np.array([True]))); m4 = M((np.array([True, False]),))
show('Masks: different number of masks (zip truncates): eq', lambda: m3 == m4)
show('Masks: shape-only (2,3) vs (3,2): eq', lambda: M((np.zeros((2, 3), bool),)) == M((np.zeros((3, 2), bool),)))
show('Fields: shape-only (2,3) vs (3,2): eq', lambda: F(np.zeros((2, 3))) == F(np.zeros((3, 2))))
show('Fields: dtype-only float32 vs float64 zeros: eq', lambda: F(np.zeros(2, np.float32)) == F(np.zeros(2)))
show('Fields: list vs tuple field: eq', lambda: F([1, 2]) == F((1, 2)))
show('Fields: 0-d array vs python float: eq', lambda: F(np.array(1.0)) == F(1.0))
show('Fields: 0-d arrays equal: eq/hash', lambda: (F(np.array(1.0)) == F(np.array(1.0)), hash(F(np.array(1.0))) == hash(F(np.array(1.0)))))
show('Fields: nested tuple of arrays equal: eq', lambda: F((np.ones(2), (np.zeros(1),))) == F((np.ones(2), (np.zeros(1),))))
show('Fields: -0.0 vs 0.0 arrays: eq', lambda: F(np.array([-0.0])) == F(np.array([0.0])))
show('Fields: NaN array: self == copy', lambda: F(np.array([np.nan])) == F(np.array([np.nan])))
show('Fields: NaN python float: self == copy', lambda: F(float("nan")) == F(float("nan")))
show('Fields: NaN python float: x == x (identity fast path)', lambda: (lambda x: x == x)(F(float("nan"))))
import jax.numpy as jnp
show('Fields: jax array field: hash', lambda: hash(F(jnp.ones(2))))
show('Fields: jax array field equal values: eq', lambda: F(jnp.ones(2)) == F(jnp.ones(2)))
show('Fields: dict field: hash', lambda: hash(F({'a': 1})))
show('Fields: bool vs int field True vs 1: eq', lambda: F(True) == F(1))
show('Fields: numpy bool array vs int array same values: eq', lambda: F(np.array([True])) == F(np.array([1])))
