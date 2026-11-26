# Sanity for E1: with the exit masking also disabled, junk in the sample's padding must show up (so E1's 0 is real).
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.ut3_operations as uo
sv._UNIFORM_MASKS = dict(sv._UNIFORM_MASKS); sv._UNIFORM_MASKS['w'] = ()
orig = sv._mask_variable
rng = np.random.default_rng(0)
shape, N = (4, 5, 6), 7
x = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=N)
fr = ut3m.UNIFORM_MANIFOLD.frame(ux).data
ww = uo.pack_vectors([rng.standard_normal((3, n)) for n in shape], N)
junk = ww.copy()
for i, n in enumerate(shape): junk[i, ..., n:] = 1e3
y = uo.pack_vectors([rng.standard_normal((3, n)) for n in shape], N)
for name in ('z', 'g', 'xi', 'f', 'w'):
    for exitmask in (True, False):
        sv._mask_variable = orig if exitmask else (lambda n, v, s: v)
        (a,) = sv.sampling_variables('probe', fr, ww, (name,), y=y)
        (b,) = sv.sampling_variables('probe', fr, junk, (name,), y=y)
        fl = lambda v: [q for u in v for q in fl(u)] if isinstance(v, (tuple, list)) else [np.asarray(v)]
        print(name, 'exit-masked' if exitmask else 'no exit mask', max(float(np.max(np.abs(p - q))) for p, q in zip(fl(a), fl(b))))
