"""R9-E2: UT3Variations.sum_stack(axis=-1) (and the backend ufv_variations_sum_stack / utv_sum_tangent_stack)
map a negative axis to array axis 1+axis, i.e. axis 0 = the MODE axis d. When K == d the result is
shape-valid and silently wrong; otherwise an obscure 'Inconsistent shapes' error. Ragged accepts -1."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.manifold as t3m
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))      # d = 3, C = (2,)
frame, _ = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
v = ut3m.UNIFORM_COREWISE.randn(frame, stack_shape=(3,))                                   # K = (3,) == d
V = v.variations
print('variations supercore shape (d,)+K+C+(nD,N) =', V.tucker_variations.shape, ' stack_shape =', V.stack_shape)
right = V.tucker_variations.sum(axis=2)                   # sum over C (the LAST stack axis): (d,)+K+(nD,N)
got = V.sum_stack(axis=-1)
print('sum_stack(axis=-1).stack_shape =', got.stack_shape, ' (expected K = (3,))')
print('got.tucker_variations.shape =', got.tucker_variations.shape, ' correct (sum over C) shape =', right.shape)
print('max|got - sum over the MODE axis d| =', np.abs(np.asarray(got.tucker_variations) - V.tucker_variations.sum(axis=0)).max(), ' <- what it computed')
print('got.masks == OR over the mode axis?', all(np.array_equal(m, np.any(m0, axis=0)) for m, m0 in zip(got.masks.data, V.masks.data)))
rv = t3m.COREWISE.randn(t3m.COREWISE.frame(x), stack_shape=(3,))
print('ragged T3Variations.sum_stack(axis=-1).stack_shape =', rv.variations.sum_stack(axis=-1).stack_shape, ' (numpy semantics: last stack axis)')
print('uniform sum_stack(axis=1).stack_shape =', V.sum_stack(axis=1).stack_shape, '(positive index works)')
