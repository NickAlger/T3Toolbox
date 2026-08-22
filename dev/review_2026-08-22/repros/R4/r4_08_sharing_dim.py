"""R4-8: why do shared_manifold(sharing).randn(frame) tangents span MORE than the unshared tangent space?"""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.shared_geometry as sg
import t3toolbox.backend.tv_operations as tv
from r4_common import tangent_basis_dense

np.random.seed(6)
def nrank(D, tol=1e-9):
    sv = np.linalg.svd(D, compute_uv=False); return int(np.sum(sv > tol * sv[0])), sv

for shape, tr, ttr, sharing in [((5, 5, 5), (3, 3, 3), (1, 3, 3, 1), (0, 0, 0)),
                                ((5, 5, 6), (2, 2, 2), (1, 2, 2, 1), (0, 0, 1))]:
    x0 = t3.TuckerTensorTrain.randn(shape, tr, ttr)
    x = x0.share(sharing)
    print(f'\n=== {shape} {tr} {ttr} sharing={sharing} ===')
    print('  x.structure', x.structure, ' tied:', x.has_shared_tucker_factors(sharing))
    geo = sg.shared_manifold(sharing)
    frame = geo.frame(x)
    print('  shared frame:', frame, ' has_minimal_ranks', frame.has_minimal_ranks, ' orthogonal', bool(frame.is_orthogonal()))
    fr_un = t3m.MANIFOLD.frame(x)
    print('  plain  frame:', fr_un, ' has_minimal_ranks', fr_un.has_minimal_ranks)
    md_un = t3m.manifold_dim((shape, tr, ttr)); md_sh = t3m.manifold_dim((shape, tr, ttr), sharing=sharing)
    print('  manifold_dim unshared', md_un, ' shared', md_sh)
    # dense rank of the unit-variation basis at the plain frame and at the shared frame
    print('  unit-basis rank @plain frame  ', nrank(tangent_basis_dense(fr_un))[0])
    print('  unit-basis rank @shared frame ', nrank(tangent_basis_dense(frame))[0])
    n = 3 * md_un
    D_un = np.stack([t3m.MANIFOLD.randn(fr_un).to_dense().reshape(-1) for _ in range(n)])
    D_sh = np.stack([geo.randn(frame).to_dense().reshape(-1) for _ in range(n)])
    D_sh_plainframe = np.stack([geo.randn(fr_un).to_dense().reshape(-1) for _ in range(n)]) if True else None
    r1, s1 = nrank(D_un); r2, s2 = nrank(D_sh); r3, s3 = nrank(D_sh_plainframe)
    print('  MANIFOLD.randn @plain frame  dense rank', r1)
    print('  shared.randn  @shared frame  dense rank', r2, ' singular values around the cut:', np.round(s2[md_sh - 2:md_sh + 3], 6), '... tail', s2[-3:])
    print('  shared.randn  @plain  frame  dense rank', r3)
    # are the shared randn tangents tangent at all?  Project one onto the plain tangent space (lstsq) and
    # measure the residual; a true tangent vector has zero residual.
    B = tangent_basis_dense(fr_un)
    v = geo.randn(frame); vd = v.to_dense().reshape(-1)
    c, *_ = np.linalg.lstsq(B, vd, rcond=1e-10)
    print('  shared.randn tangent: residual off the plain tangent space =', np.linalg.norm(B @ c - vd) / np.linalg.norm(vd))
    print('  shared.randn tangent gauged:', bool(v.is_gauged()), ' frame is shared frame:', v.frame is frame,
          ' tangent type', type(v).__name__, ' stack', v.stack_shape)
    # realization via the shared-aware backend (tied embedding) vs the plain T3Tangent.to_dense
    try:
        import t3toolbox.backend.sharing as bs
        sd = bs.fv_shared_frame_data(frame.data, sharing) if hasattr(bs, 'fv_shared_frame_data') else None
        emb = tv.tv_to_t3(frame.data, v.variations.data, shared_data=sd)
        vd2 = t3.TuckerTensorTrain(*emb).to_dense().reshape(-1)
        print('  tied-embedding to_dense vs T3Tangent.to_dense relerr =', np.linalg.norm(vd2 - vd) / np.linalg.norm(vd))
        c, *_ = np.linalg.lstsq(B, vd2, rcond=1e-10)
        print('  tied-embedding realization: residual off plain tangent space =', np.linalg.norm(B @ c - vd2) / np.linalg.norm(vd2))
        D_tied = np.stack([t3.TuckerTensorTrain(*tv.tv_to_t3(frame.data, geo.randn(frame).variations.data, shared_data=sd)).to_dense().reshape(-1) for _ in range(n)])
        print('  tied-embedding dense rank of shared.randn =', nrank(D_tied)[0], ' (manifold_dim(sharing) =', md_sh, ')')
    except Exception as e:
        print('  tied embedding check failed:', type(e).__name__, str(e)[:120])
