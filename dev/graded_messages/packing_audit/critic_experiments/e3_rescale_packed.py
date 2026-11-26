# E3: (a) rescale_coordinates on packed probe data (the plan's K6 makes probe_rescale_coordinates parametric over
# the packed form): zero padding in, 0/0 = NaN padding out; then the engine's multiplicative mask keeps the NaN and f
# is NaN. (b) today's data_metric refuses a (J+1,) per-slot vector on a ragged probe r at d = J+1; a left-aligned
# block weight (the plan) accepts it as a per-mode weight.
import warnings
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.data_preparation as prep
import t3toolbox.backend.ut3_operations as uo
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.jets as jets
warnings.simplefilter('ignore')
rng = np.random.default_rng(0)
shape, N = (4, 5, 6), 6
x = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=N)
ww = [rng.standard_normal((3, n)) for n in shape]
y = [rng.standard_normal((3, n)) for n in shape]
scales = [10.0 ** rng.standard_normal(n) for n in shape]
# ragged: fine
ww2, y2 = prep.rescale_coordinates('probe', (), ww, y, scales)
(f_r,) = sv.sampling_variables('probe', x.data, ww2, ('f',), y=y2)
# packed (what K6's parametric probe_rescale_coordinates would do: the same elementwise division)
pw, py, ps = uo.pack_vectors(ww, N), uo.pack_vectors(y, N), uo.pack_vectors(scales, N)
pw2, py2 = pw * ps, py / ps
print('packed y padding after division (mode 0, entries 4..5):', py2[0, 0, 4:])
(f_u,) = sv.sampling_variables('probe', ux.data, pw2, ('f',), y=py2)
print('f ragged', float(f_r), ' f uniform from packed-rescaled data', float(f_u))

# (b) data_metric and a per-slot vector at d = J + 1
J = 2
r = [rng.standard_normal((J + 1, 4, n)) for n in shape]           # probe, order (J,), F = (3,), W = (4,), d = 3
per_slot = np.array([1.0, 10.0, 100.0])                           # meant: one weight per slot, the same in each mode
try:
    jets.data_metric(r, per_slot, None)
    print('today: accepted')
except ValueError as e:
    print('today: refused ->', str(e)[:110])
# the plan's left alignment, as written in 2.2: row i of a (d,)+F block weight against r[i]; a (d,) vector passes the
# only structural test available without the kind (its first axis is d) and is applied per MODE:
left = [per_slot[i].reshape((1,) * 0) * ri for i, ri in enumerate(r)]   # what a left-aligned rule computes
print('left-aligned reading weights mode 2 by', per_slot[2], 'instead of slot 2 of every mode')
