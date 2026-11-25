# E5: the backend batch's readers on a batch whose per-mode fields are a frontend wrapper (a stand-in for the plan's
# PackedVectors: a frozen dataclass with .array and .shape). The plan's workflow "pack once (fitting.pack_batch), draw
# with take" calls LeastSquaresOracle.n_measurements / take / LeastSquaresBatch.data_metric on such a batch.
import dataclasses as dc
import numpy as np
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
import t3toolbox.backend.ut3_operations as uo

@dc.dataclass(frozen=True, eq=False)
class PackedVectors:
    array: np.ndarray
    shape: tuple

rng = np.random.default_rng(0)
shape = (4, 5, 6)
ww = [rng.standard_normal((10, n)) for n in shape]
y = [rng.standard_normal((10, n)) for n in shape]
pb = LeastSquaresBatch(PackedVectors(uo.pack_vectors(ww), shape), PackedVectors(uo.pack_vectors(y), shape))
oracle = LeastSquaresOracle('probe', order=())
for what, call in (('n_measurements', lambda: oracle.n_measurements(pb)),
                   ('take', lambda: oracle.take(pb, np.array([0, 3]))),
                   ('batch.data_metric', lambda: pb.data_metric(pb.y))):
    try:
        out = call()
        print(what, '->', out if not hasattr(out, 'sample') else type(out.sample).__name__)
    except Exception as e:
        print(what, '-> %s: %s' % (type(e).__name__, str(e)[:100]))
print('np.shape(wrapper) =', np.shape(pb.sample), ' np.ndim =', np.ndim(pb.sample), ' _w_axes =', oracle._w_axes(pb),
      ' _W =', oracle._W(pb), ' (the true W is (10,))')
