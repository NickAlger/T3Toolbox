"""H6(b): one wrong-shape input per __post_init__/validate contract; does it raise, and does the message name the problem?"""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.fitting as fit

def expect(label, fn):
    try:
        fn()
        print('%-58s NO RAISE  <-- silently accepted' % label)
    except Exception as e:
        first = str(e).replace('\n', ' | ')[:150]
        print('%-58s %s: %s' % (label, type(e).__name__, first))

np.random.seed(0)
shape, tr, ttr = (4, 5, 3), (2, 3, 2), (1, 2, 3, 1)
x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
B, G = x.data
print('== TuckerTensorTrain')
expect('len mismatch', lambda: t3.TuckerTensorTrain(B[:2], G))
expect('empty', lambda: t3.TuckerTensorTrain((), ()))
expect('tt core ndim<3', lambda: t3.TuckerTensorTrain(B, (G[0][0],) + G[1:]))
expect('tt rank chain broken', lambda: t3.TuckerTensorTrain(B, (G[0], np.ones((3, 3, 3)), G[2])))
expect('tucker ndim<2', lambda: t3.TuckerTensorTrain((B[0][0],) + B[1:], G))
expect('tucker rank != tt middle', lambda: t3.TuckerTensorTrain((np.ones((3, 4)),) + B[1:], G))
expect('stack mismatch', lambda: t3.TuckerTensorTrain((B[0][None],) + B[1:], G))

print('== T3Frame')
fr = bvf.T3Frame.random_orthogonal(shape, tr, ttr)
U, D, L, R = fr.data
expect('len mismatch', lambda: bvf.T3Frame(U[:2], D, L, R))
expect('U ndim<2', lambda: bvf.T3Frame((U[0][0],) + U[1:], D, L, R))
expect('L ndim<3', lambda: bvf.T3Frame(U, D, (L[0][0],) + L[1:], R))
expect('stack mismatch', lambda: bvf.T3Frame((U[0][None],) + U[1:], D, L, R))
expect('rL chain broken', lambda: bvf.T3Frame(U, D, (L[0], np.ones((2, 3, 7)), L[2]), R))
expect('rR chain broken', lambda: bvf.T3Frame(U, D, L, (R[0], np.ones((7, 3, 1)), R[2])))
expect('tucker rank U vs L (unstacked)', lambda: bvf.T3Frame((np.ones((9, 4)),) + U[1:], D, L, R))
frs = bvf.T3Frame.random_orthogonal(shape, tr, ttr, stack_shape=(2,))
Us, Ds, Ls, Rs = frs.data
expect('tucker rank U vs L (STACKED (2,)) -- message numbers', lambda: bvf.T3Frame((np.ones((2, 9, 4)),) + Us[1:], Ds, Ls, Rs))
expect('D left rank', lambda: bvf.T3Frame(U, (np.ones((7,) + D[0].shape[1:]),) + D[1:], L, R))
expect('D right rank (message says T3Base)', lambda: bvf.T3Frame(U, (np.ones(D[0].shape[:2] + (7,)),) + D[1:], L, R))
expect('D middle (down rank) = 9 (free rank store; accepted?)', lambda: bvf.T3Frame(U, (np.ones((D[0].shape[0], 9, D[0].shape[2])),) + D[1:], L, R))

print('== T3Variations')
v = t3m.MANIFOLD.randn(fr)
V, H = v.variations.data
expect('len mismatch', lambda: bvf.T3Variations(V[:2], H))
expect('V ndim<2', lambda: bvf.T3Variations((V[0][0],) + V[1:], H))
expect('H ndim<3', lambda: bvf.T3Variations(V, (H[0][0],) + H[1:]))
expect('stack mismatch (message says T3Frame)', lambda: bvf.T3Variations((V[0][None],) + V[1:], H))

print('== (T3Frame, T3Variations) pair via T3Tangent')
expect('frame stack not suffix of var stack', lambda: t3m.T3Tangent(frs, v.variations))
expect('V hole mismatch (message says T3Base - T3Variation)', lambda: t3m.T3Tangent(fr, bvf.T3Variations((np.ones((V[0].shape[0] + 1, V[0].shape[1])),) + V[1:], H)))
expect('H hole mismatch', lambda: t3m.T3Tangent(fr, bvf.T3Variations(V, (np.ones((H[0].shape[0], H[0].shape[1] + 1, H[0].shape[2])),) + H[1:])))

print('== T3Weights / T3FrameWeights')
expect('T3Weights tt len', lambda: t3.T3Weights(tuple(np.ones(n) for n in tr), tuple(np.ones(r) for r in ttr[:-1])))
expect('T3Weights stack mismatch', lambda: t3.T3Weights((np.ones((2, 2)),) + tuple(np.ones(n) for n in tr[1:]), tuple(np.ones(r) for r in ttr)))
expect('T3FrameWeights len', lambda: bvf.T3FrameWeights(tuple(np.ones(2) for _ in range(3)), tuple(np.ones(2) for _ in range(2)), tuple(np.ones(2) for _ in range(3)), tuple(np.ones(2) for _ in range(3))))
expect('T3FrameWeights scalar (ndim 0)', lambda: bvf.T3FrameWeights((np.float64(1.0),) * 3, (np.ones(2),) * 3, (np.ones(2),) * 3, (np.ones(2),) * 3))

print('== Uniform')
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
tk, tt, shp, (tkm, ttm) = ux.data
expect('UT3 non-bool mask', lambda: ut3.UniformTuckerTensorTrain(tk, tt, shp, ut3.UT3Masks(tkm.astype(int), ttm)))
expect('UT3 shape len', lambda: ut3.UniformTuckerTensorTrain(tk, tt, shp[:2], ut3.UT3Masks(tkm, ttm)))
expect('UT3 shape entry > N', lambda: ut3.UniformTuckerTensorTrain(tk, tt, (99,) + shp[1:], ut3.UT3Masks(tkm, ttm)))
expect('UT3 tt supercore wrong r', lambda: ut3.UniformTuckerTensorTrain(tk, tt[..., :2], shp, ut3.UT3Masks(tkm, ttm)))
expect('UT3 tt mask wrong len', lambda: ut3.UniformTuckerTensorTrain(tk, tt, shp, ut3.UT3Masks(tkm, ttm[:-1])))
expect('UT3 tucker supercore stack mismatch', lambda: ut3.UniformTuckerTensorTrain(tk[:, None], tt, shp, ut3.UT3Masks(tkm, ttm)))
ufr = ubv.UT3Frame.from_t3frame(fr)
uU, uD, uL, uR, ushp, umasks = ufr.data
expect('UT3Frame down supercore wrong rR', lambda: ubv.UT3Frame(uU, uD[..., :1], uL, uR, ushp, ufr.masks))
expect('UT3Frame left supercore rL != rL', lambda: ubv.UT3Frame(uU, uD, uL[..., :1], uR, ushp, ufr.masks))
expect('UT3Frame shape len', lambda: ubv.UT3Frame(uU, uD, uL, uR, ushp[:2], ufr.masks))
expect('UT3Frame up supercore ndim 2 (no d axis)', lambda: ubv.UT3Frame(uU[0], uD, uL, uR, ushp, ufr.masks))
uv = ut3m.UT3Tangent.from_t3tangent(v)
uV, uH = uv.variations.data[:2]
expect('UT3Variations tt wrong nU', lambda: ubv.UT3Variations(uV, uH[..., :1, :], uv.variations.shape, uv.variations.masks))
expect('UT3Tangent pair: variations of other frame ranks', lambda: ut3m.UT3Tangent(ubv.UT3Frame.from_t3frame(bvf.T3Frame.random_orthogonal(shape, (1, 1, 1), (1, 1, 1, 1))), uv.variations))
expect('UT3Tangent pair: stack not suffix', lambda: ut3m.UT3Tangent(ubv.UT3Frame.from_t3frame(frs), uv.variations))

print('== GaussNewtonModel (fitting) structural')
ww = [np.random.randn(6, n) for n in shape]
r_ok = x.apply(ww)
expect('apply_model residual wrong W (5 vs 6)', lambda: fit.apply_model(t3m.MANIFOLD, x, ww, r_ok[:5]).gradient)
expect('apply_model ww wrong d (2 vectors)', lambda: fit.apply_model(t3m.MANIFOLD, x, ww[:2], r_ok).gradient)
expect('apply_model ww wrong N at mode 1', lambda: fit.apply_model(t3m.MANIFOLD, x, [ww[0], ww[1][:, :4], ww[2]], r_ok).gradient)
m = fit.apply_model(t3m.MANIFOLD, x, ww, r_ok)
other = t3m.MANIFOLD.randn(bvf.T3Frame.random_orthogonal(shape, (2, 2, 2), (1, 2, 2, 1)))
expect('gn_hessian(p) p of different STRUCTURE', lambda: m.gn_hessian(other))
