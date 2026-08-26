"""R9-E: negative `axis` on the uniform variation/tangent stack sums: silent wrong answer when K == d,
obscure structural error otherwise; the ragged twin accepts negative axes (numpy semantics)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.manifold as t3m
import t3toolbox.backend.ufv_operations as ufvo
import t3toolbox.backend.utv_operations as utvo

def rep(msg, ok, val=None):
    print(('PASS ' if ok else 'FAIL ') + msg + ('' if val is None else '  [%s]' % (val,)))

for d, K in ((3, (3,)), (3, (4,))):
    np.random.seed(0)
    shape = (4, 5, 6)[:d]; n = (2, 3, 2)[:d]; r = (1, 2, 2, 1)[:d] + (1,)
    x = t3.TuckerTensorTrain.randn(shape, n, r, stack_shape=(2,))
    frame, _ = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
    v = ut3m.UNIFORM_COREWISE.randn(frame, stack_shape=K)           # stack K + C = K + (2,)
    dense = np.asarray(v.to_dense())                                 # (K, 2, *shape)
    print('--- d=%d, K=%s, C=(2,): variations stack %s' % (d, K, v.variations.stack_shape))
    # ragged twin: T3Variations.sum_stack(axis=-1) sums the LAST stack axis (C)
    rv = t3m.COREWISE.randn(t3m.COREWISE.frame(x), stack_shape=K)
    rs = rv.variations.sum_stack(axis=-1)
    rep('ragged T3Variations.sum_stack(axis=-1): sums last stack axis -> stack', rs.stack_shape == K, rs.stack_shape)
    # uniform
    try:
        us = v.variations.sum_stack(axis=-1)
        dense_us = np.asarray(ut3m.UT3Tangent(frame, us).to_dense()) if us.stack_shape == (2,) else None
        rep('uniform UT3Variations.sum_stack(axis=-1) returned stack', us.stack_shape == K, us.stack_shape)
        expected = dense.sum(axis=len(K))                            # sum over C -> (K, *shape)
        got = np.asarray(us.tucker_variations)
        # what it actually did: summed axis 0 = the MODE axis d, and relabeled K as d
        print('    tucker_variations summed over axis 1+(-1)=0 (the mode axis d); result shape', got.shape,
              'vs expected (d,)+K+(nD,N)', (d,) + K + v.variations.tucker_variations.shape[-2:])
        rep('    SILENT WRONG: result validated (K == d) but is the mode-axis sum, not the C sum',
            False, 'max|got - sum_over_d| = %.3g' % np.abs(got - v.variations.tucker_variations.sum(axis=0)).max())
    except ValueError as e:
        rep('    uniform sum_stack(axis=-1) raises an obscure structural error (K != d)', False, str(e).splitlines()[0][:100])
    try:
        st = v.sum_tangents(axis=-1)
        rep('uniform UT3Tangent.sum_tangents(axis=-1) returned tangent stack', st.tangent_stack_shape == K[:-1], st.stack_shape)
    except ValueError as e:
        rep('    uniform sum_tangents(axis=-1) raises', False, str(e).splitlines()[0][:100])
    # positive axis is fine
    us0 = v.variations.sum_stack(axis=0)
    rep('uniform sum_stack(axis=0) sums K -> stack (2,) and matches dense', us0.stack_shape == (2,) and np.allclose(np.asarray(ut3m.UT3Tangent(frame, us0).to_dense()), dense.sum(axis=0)))
# backend twins
print('--- backend: ufv_variations_sum_stack / utv_sum_tangent_stack with axis=-1 use stack axis 1+(-1)=0')
import inspect
print('   ', [l.strip() for l in inspect.getsource(ufvo.ufv_variations_sum_stack).splitlines() if 'stack_axes =' in l])
print('   ', [l.strip() for l in inspect.getsource(utvo.utv_sum_tangent_stack).splitlines() if 'k_axes =' in l])
