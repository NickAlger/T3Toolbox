import numpy as np
import t3toolbox.backend.jets as jets
from t3toolbox.backend.ut3_operations import pack_vectors

shape = (4, 5, 6)
rng = np.random.default_rng(0)
ww = [rng.standard_normal((7, N)) for N in shape]        # ragged sample: a list of d arrays, W = (7,)
pp = [rng.standard_normal((7, N)) for N in shape]
ww_packed, pp_packed = pack_vectors(ww), pack_vectors(pp)  # packed sample: one array (d,) + W + (N,)

print('1. A function with no object, today: the container is read, to return the same form')
out = jets.affine_jet(ww, pp)
print('   affine_jet(list, list)     ->', type(out).__name__, [a.shape for a in out])
out = jets.affine_jet(ww_packed, pp_packed)
print('   affine_jet(array, array)   ->', type(out).__name__, out.shape)
try:
    jets.affine_jet(ww, pp_packed)
except ValueError as e:
    print('   affine_jet(list, array)    -> ValueError:', str(e)[:70], '...')

print('2. Where the container cannot say: one array, two readings')
a = np.ones((3, 3, 7, 6))
print('   an array of shape', a.shape, 'is apply data with F=(3,), W=(3,7,6),')
print('   and it is packed probe data with d=3, F=(3,), W=(7,), N=6')
as_apply = jets.derivatives_to_taylor('apply', (2,), a)
as_probe = np.stack(jets.derivatives_to_taylor('probe', (2,), a))
print('   read as apply: the factors 1/k! go on axis 0:', as_apply[:, 0, 0, 0].tolist())
print('   read as probe: the factors 1/k! go on axis 1:', as_probe[0, :, 0, 0].tolist(), '(axis 0:', as_probe[:, 0, 0, 0].tolist(), ')')
