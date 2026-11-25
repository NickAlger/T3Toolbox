# E6: the "typed faces" do not fix the layer: the ragged twin runs on a uniform point (and returns the packed form),
# and the uniform twin runs on a ragged point (and returns tuples). The plan's 1.5 fixes kind, call type and want only.
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.probing as probing
import t3toolbox.backend.ut3_sampling as ut3_sampling
import t3toolbox.backend.ut3_operations as uo
rng = np.random.default_rng(0)
shape = (4, 5, 6)
x = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
ww = [rng.standard_normal((3, n)) for n in shape]
z1 = probing.t3_probe(ux.data, uo.pack_vectors(ww))           # the RAGGED twin given a uniform point
print('probing.t3_probe on a uniform point ->', type(z1).__name__, getattr(z1, 'shape', None))
z2 = ut3_sampling.ut3_probe(x.data, ww)                        # the UNIFORM twin given a ragged point
print('ut3_sampling.ut3_probe on a ragged point ->', type(z2).__name__, [a.shape for a in z2])
