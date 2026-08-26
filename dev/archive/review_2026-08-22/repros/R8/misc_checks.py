"""R8 misc: save/load keeps masks host numpy (incl. jax supercores); ut3svd_supercores direct call vs
ut3svd; `+`/inner with different padded N; minimal_ranks type under jax; masks stay host after jax ops;
is_*_orthogonal at d=1 (separate); UT3Masks hash/eq; stacking varying-pad leaves."""
import numpy as np, tempfile, os
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ut3_svd as usvd
import t3toolbox.backend.ut3_masking as um
import t3toolbox.backend.ut3_conversions as uc
np.random.seed(0)
def relerr(a, b): return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) / max(np.linalg.norm(b), 1e-300))

x = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 3, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=4, r=4)

# --- save/load round trip (numpy and jax), masks host numpy bool, shape preserved, gappy masks preserved
tmp = tempfile.mkdtemp()
f = os.path.join(tmp, 'u.npz')
s = ux + ux  # gappy masks
s.save(f)
s2 = ut3.UniformTuckerTensorTrain.load(f)
print('save/load gappy masks equal:', s2.masks == s.masks, '| dense', relerr(s2.to_dense(), s.to_dense()), '| shape', s2.shape == s.shape)
print('  mask types:', [type(m).__name__ + ':' + str(m.dtype) for m in s2.masks.data])
try:
    import jax, jax.numpy as jnp
    sj = s.to_jax(); sj.save(f)
    s3 = ut3.UniformTuckerTensorTrain.load(f, use_jax=True)
    print('jax save/load: supercore type', type(s3.tucker_supercore).__name__, '| masks', [type(m).__name__ for m in s3.masks.data], '| dense', relerr(np.asarray(s3.to_dense()), s.to_dense()))
    # masks stay host after jax ops
    for nm, fn in [('t3svd', lambda u: u.t3svd()[0]), ('add', lambda u: u + u), ('left', lambda u: u.left_orthogonalize_tt_cores()),
                   ('ras', lambda u: u.t3svd()[0].rank_adjustment_sweep()), ('sum_stack', lambda u: ut3.UniformTuckerTensorTrain.stack((u, u)).sum_stack()),
                   ('reverse', lambda u: u.reverse()), ('squash', lambda u: u.squash_tails())]:
        out = fn(ux.to_jax())
        print('  jax %-9s masks host numpy: %s | supercore jax: %s' % (nm, all(type(m) is np.ndarray for m in out.masks.data), isinstance(out.tucker_supercore, jnp.ndarray)))
    mn = ux.to_jax().minimal_ranks
    print('  minimal_ranks on a jax UT3 returns:', [type(m).__name__ for m in mn], '(masks/ranks are supposed to be host np)')
    print('  has_minimal_ranks on jax UT3:', type(ux.to_jax().has_minimal_ranks).__name__)
except ImportError:
    print('no jax')

# --- ut3svd_supercores direct (public, untested): vs ut3svd at the same caps, and its skip_orthogonalization claim
data = ux.data
masked = um.ut3_apply_masks(data)
caps = um.ut3_make_masks(ux.tucker_ranks, ux.tt_ranks, ux.n, ux.r)
(tk, tt), sk, st = usvd.ut3svd_supercores(masked, caps)
dense_direct = uc.ut3_to_dense((tk, tt, ux.shape, caps))
print('ut3svd_supercores: dense err vs x', relerr(dense_direct, x.to_dense()), '| shapes keep input pad', tk.shape == ux.tucker_supercore.shape and tt.shape == ux.tt_supercore.shape)
xs, rsk, rst = x.t3svd()
print('  tucker svals vs ragged:', [relerr(sk[i][caps[0][i]], rsk[i]) for i in range(3)])
print('  tt svals vs ragged (leading real slots; the uniform keeps the cap width, extra slots zero):', [(relerr(st[i][caps[1][i]][:len(rst[i])], rst[i]), float(np.abs(st[i][caps[1][i]][len(rst[i]):]).max(initial=0))) for i in range(4)])
# skip_orthogonalization with a right-orthogonal input
ro = ux.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores()
(tk2, tt2), sk2, st2 = usvd.ut3svd_supercores(um.ut3_apply_masks(ro.data), um.ut3_make_masks(ro.tucker_ranks, ro.tt_ranks, ro.n, ro.r), skip_orthogonalization=True)
print('  skip_orth on right-orth input dense err', relerr(uc.ut3_to_dense((tk2, tt2, ro.shape, ro.masks.data)), x.to_dense()))
# squash_tails_first=False with a squashed input should be identical
(tk3, tt3), sk3, st3 = usvd.ut3svd_supercores(masked, caps, squash_tails_first=False)
print('  squash_tails_first=False (already squashed input) dense err', relerr(uc.ut3_to_dense((tk3, tt3, ux.shape, caps)), x.to_dense()))
# garbage input (docstring says "assumed masked")
from t3toolbox.backend.common import prefix_mask
g = (masked[0] + 10 * np.random.randn(*masked[0].shape) * (~(caps[0][..., :, None] & prefix_mask(ux.shape, ux.N)[:, None, :])), masked[1])
(tk4, tt4), _, _ = usvd.ut3svd_supercores(g, caps)
print('  unmasked (garbage tucker) input dense err', relerr(uc.ut3_to_dense((tk4, tt4, ux.shape, caps)), x.to_dense()), '(documented: assumes masked input)')

# --- + / inner with DIFFERENT padded N (documented "pass N to force a larger pad"), same real shape
ua = ut3.UniformTuckerTensorTrain.from_t3(x, N=8)
ub = ut3.UniformTuckerTensorTrain.from_t3(x)
for nm, fn in [('+', lambda: ua + ub), ('inner', lambda: ua.inner(ub)), ('weighted_inner', lambda: ut3.ut3_weighted_inner(ua, ut3.UT3Weights.from_ut3svd(ua), ub, ut3.UT3Weights.from_ut3svd(ub)))]:
    try:
        fn(); print('different padded N: %s ok' % nm)
    except Exception as e:
        print('different padded N: %s -> %s: %s' % (nm, type(e).__name__, str(e)[:140]))
# stack of leaves with different padded sizes
try:
    ut3.UniformTuckerTensorTrain.stack((ua, ub))
    print('stack of different-pad leaves: ok')
except Exception as e:
    print('stack of different-pad leaves -> %s: %s' % (type(e).__name__, str(e)[:140]))

# --- is_left_orthogonal / orthogonality residual at d=1
x1 = t3.TuckerTensorTrain.randn((5,), (3,), (1, 1))
u1 = ut3.UniformTuckerTensorTrain.from_t3(x1).down_orthogonalize_tucker_cores()
import t3toolbox.backend.t3_orthogonalization as t3o
print('ragged d=1 t3_orthogonality_residual left:', float(t3o.t3_orthogonality_residual(x1.down_orthogonalize_tucker_cores().data, 'left')))
for side in ('left', 'right'):
    try:
        print('uniform d=1 is_%s_orthogonal:' % side, getattr(u1, 'is_%s_orthogonal' % side)())
    except Exception as e:
        print('uniform d=1 is_%s_orthogonal -> %s: %s' % (side, type(e).__name__, str(e)[:100]))

# --- UT3Masks value hash/eq
m1 = ut3.UT3Masks(*[m.copy() for m in ux.masks.data])
print('UT3Masks value eq/hash:', m1 == ux.masks, hash(m1) == hash(ux.masks))
