"""Tangent transposes on the SAME frame (uniform frame converted from the ragged one)."""
import numpy as np, t3toolbox as t3t
import t3toolbox.manifold as t3m, t3toolbox.uniform_manifold as ut3m, t3toolbox.uniform_frame_variations_format as ubv
np.random.seed(0)
shape = (5, 6, 7)
x = t3t.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
frame = t3m.MANIFOLD.frame(x); uframe = ubv.UT3Frame.from_t3frame(frame)
def rel(a, b): return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))
ww = [np.random.randn(4, N) for N in shape]; zt = [np.random.randn(4, N) for N in shape]; c = np.random.randn(4)
pp = [np.random.randn(4, N) for N in shape]
zj = [np.random.RandomState(2 + i).randn(3, 4, N) for i, N in enumerate(shape)]
cj = np.random.RandomState(1).randn(3, 4)
for sop in (False, True):
    print(f'sum_over_probes={sop}')
    print('  probe_transpose            rel %.1e' % rel(t3m.T3Tangent.probe_transpose(zt, ww, frame, sum_over_probes=sop).to_dense(), ut3m.UT3Tangent.probe_transpose(zt, ww, uframe, sum_over_probes=sop).to_dense()))
    print('  apply_transpose            rel %.1e' % rel(t3m.T3Tangent.apply_transpose(c, ww, frame, sum_over_probes=sop).to_dense(), ut3m.UT3Tangent.apply_transpose(c, ww, uframe, sum_over_probes=sop).to_dense()))
    print('  apply_derivatives_transpose rel %.1e' % rel(t3m.T3Tangent.apply_derivatives_transpose(cj, ww, pp, frame, 2, sum_over_probes=sop).to_dense(), ut3m.UT3Tangent.apply_derivatives_transpose(cj, ww, pp, uframe, 2, sum_over_probes=sop).to_dense()))
    for cs in (None, 2):
        print(f'  probe_derivatives_transpose(chunk_size={cs}) rel %.1e' % rel(t3m.T3Tangent.probe_derivatives_transpose(zj, ww, pp, frame, 2, sum_over_probes=sop, chunk_size=cs).to_dense(), ut3m.UT3Tangent.probe_derivatives_transpose(zj, ww, pp, uframe, 2, sum_over_probes=sop, chunk_size=cs).to_dense()))
