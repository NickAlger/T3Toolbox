"""E1: an engine call with _mask_variable patched to identity is a 'bare seam' (real block bitwise, padding not zero).
E2: per plain kind, the uniform corewise Hessian with the directions' entry masks AND the exit mask removed has a
corrupted real block (the engine analog of the old negative control)."""
import unittest.mock as mock
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.ut3_sampling as ut3s
import t3toolbox.backend.utv_sampling as utvs
import t3toolbox.backend.ut3_operations as uops
import t3toolbox.backend.ut3_masking as ut3mask

np.random.seed(0)
shape = (5, 6, 4)
x = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
xu = ut3.UniformTuckerTensorTrain.from_t3(x, N=7, n=4, r=4)
W = (2,)
ww = [np.random.randn(*(W + (n,))) for n in shape]
index = np.stack([np.random.randint(0, n, size=W) for n in shape])
S = W + (2,)
z = [np.random.randn(*(S + (n,))) for n in shape]
c = np.random.randn(*S)
identity = lambda name, value, setting: value
frame = ut3m.UNIFORM_MANIFOLD.frame(xu).data

def realpad(out, ref_masks_out):
    pass

for kind, sample, res in (('probe', ww, z), ('apply', ww, c), ('entries', index, c)):
    for sop in (True, False):
        # corewise transposes on the point
        op = getattr(ut3s, 'ut3_%s_corewise_transpose' % kind)
        out = op(xu.data, sample, res, sum_over_samples=sop)
        held = sv.HeldVariables(Jp=uops.pack_if_ragged(res, 7) if kind == 'probe' else res)
        with mock.patch.object(sv, '_mask_variable', identity):
            (bare,) = sv.sampling_variables(kind, xu.data, sample, ('JtJp',), held=held, sum_over_samples=sop)
        nz = [np.abs(np.asarray(b) - np.asarray(o)).max() for o, b in zip(out, bare)]
        same_real = all(np.array_equal(np.asarray(o)[np.asarray(o) != 0], np.asarray(b)[np.asarray(o) != 0]) for o, b in zip(out, bare))
        print('corewise', kind, sop, 'real block bitwise:', same_real, ' bare padding max:', ['%.2e' % v for v in nz])
        # tangent transposes on the frame
        top = getattr(utvs, 'utv_%s_transpose' % kind)
        tout = top(frame, sample, res, sum_over_samples=sop)
        with mock.patch.object(sv, '_mask_variable', identity):
            ((V, H),) = sv.sampling_variables(kind, frame, sample, ('JtJp',), held=held, sum_over_samples=sop)
        nz = [np.abs(np.asarray(b) - np.asarray(o)).max() for o, b in zip(tout[:2], (V, H))]
        same_real = all(np.array_equal(np.asarray(o)[np.asarray(o) != 0], np.asarray(b)[np.asarray(o) != 0]) for o, b in zip(tout[:2], (V, H)))
        print('tangent ', kind, sop, 'real block bitwise:', same_real, ' bare padding max:', ['%.2e' % v for v in nz])

# E2
rng = np.random.default_rng(1)
D = (rng.standard_normal(xu.data[0].shape), rng.standard_normal(xu.data[1].shape))   # garbage everywhere padded
for kind, sample, res in (('probe', ww, [zi[:, 0] for zi in z]), ('apply', ww, c[:, 0]), ('entries', index, c[:, 0])):
    xs = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)), N=7, n=4, r=4)
    hess = getattr(ut3s, 'ut3_%s_corewise_hessian' % kind)
    Dd = (D[0][:, 0], D[1][:, 0])
    good = hess(xs.data, sample, res, Dd)
    table = dict(sv._UNIFORM_MASKS, D_dot=(), L_dot=())
    with mock.patch.object(sv, '_UNIFORM_MASKS', table):
        with mock.patch.object(sv, '_mask_variable', identity):
            bare = hess(xs.data, sample, res, Dd)
    bare_m = ut3mask.ut3_apply_masks((bare[0], bare[1], xs.data[2], xs.data[3]))
    rel = np.linalg.norm(bare_m[1] - good[1]) / np.linalg.norm(good[1])
    with mock.patch.object(sv, '_UNIFORM_MASKS', table):
        only_dirs = hess(xs.data, sample, res, Dd)                  # exit mask kept: still corrupt?
    rel2 = np.linalg.norm(only_dirs[1] - good[1]) / np.linalg.norm(good[1])
    print('E2', kind, 'real block rel. change without direction+exit masks (re-masked): %.2e; without direction masks only: %.2e' % (rel, rel2))
