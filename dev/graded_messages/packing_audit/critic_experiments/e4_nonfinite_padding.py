# E4: "liberal in" tolerates only FINITE junk: the engine masks by multiplication, so inf/nan in a padding survives
# the mask, and nan reaches REAL outputs through the zeroed core columns (0 * inf = nan inside a contraction).
# A relative full weight computed on packed data, 1/|y|, is the everyday way to put inf there.
import warnings
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ut3_operations as uo
import t3toolbox.backend.sampling_variables as sv
from t3toolbox.backend.sampling_variables import HeldVariables
warnings.simplefilter('ignore')
rng = np.random.default_rng(0)
shape, N = (4, 5, 6), 6
x = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=N)
ww = uo.pack_vectors([rng.standard_normal((3, n)) for n in shape], N)
y = uo.pack_vectors([rng.standard_normal((3, n)) for n in shape], N)
inf_y = y.copy(); inf_y[0, :, 4:] = np.inf
(f,) = sv.sampling_variables('probe', ux.data, ww, ('f',), y=y)
(f_inf,) = sv.sampling_variables('probe', ux.data, ww, ('f',), y=inf_y)
print('f with zero padding in y:', float(f), '  with inf in y padding:', float(f_inf))
# the weight path: a relative weight 1/|y| on the packed y (zero padding -> inf padding)
w_rel = 1.0 / np.abs(y)
print('relative weight padding:', w_rel[0, 0, 4:])
(f_w,) = sv.sampling_variables('probe', ux.data, ww, ('f',), y=y, weight=w_rel)
print('f with that weight:', float(f_w))
# a held c (the transposes' route) with inf padding: the REAL blocks of g go nan
c = y.copy(); c_inf = c.copy(); c_inf[0, :, 4:] = np.inf
(b,) = sv.sampling_variables('probe', ux.data, ww, ('JtJp',), held=HeldVariables(Jp=c))
(b_inf,) = sv.sampling_variables('probe', ux.data, ww, ('JtJp',), held=HeldVariables(Jp=c_inf))
print('JtJp real part finite with zero padding:', bool(np.isfinite(b[0]).all()), ' with inf padding:',
      bool(np.isfinite(b_inf[0]).all()), '(nan count in the Tucker blocks:', int(np.isnan(b_inf[0]).sum()), ')')
