"""R4-6: MANIFOLD.{project,project_oblique,retract,transport,project_ambient,inner,norm} and the COREWISE twin
at asymmetric shapes with C and K stacks, vs dense ground truth; manifold_dim vs dense tangent-space rank
(incl. sharing=)."""
import itertools
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
import t3toolbox.shared_geometry as sg
from r4_common import STRUCTS, NONMIN, tangent_basis_dense, dense_project_onto_tangent, hs_inner, relerr, leaf

np.random.seed(4)
worst = {}
def note(k, e): worst[k] = max(worst.get(k, 0.0), float(e))

for (shape, tr, ttr), C, K in itertools.product(STRUCTS, [(), (2,)], [(), (3,)]):
    d = len(shape)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
    frame = t3m.MANIFOLD.frame(x)
    key = f'd={d} C={C} K={K}'
    try:
        frames_c = frame.unstack() if C else (frame,)
        bases = [tangent_basis_dense(f) for f in frames_c]               # dense tangent bases per base point
        # ---- inner / norm == HS, on gauged tangents (K+C) ----
        v = t3m.MANIFOLD.randn(frame, stack_shape=K); w = t3m.MANIFOLD.randn(frame, stack_shape=K)
        Vd, Wd = v.to_dense(), w.to_dense()
        note(key + ' inner=HS', relerr(hs_inner(Vd, Wd, d), t3m.MANIFOLD.inner(v, w)))
        note(key + ' norm=HS', relerr(np.sqrt(hs_inner(Vd, Vd, d)), t3m.MANIFOLD.norm(v)))
        assert t3m.MANIFOLD.norm(v).shape == K + C
        # ---- project (Pi): gauged, idempotent, and <Pi g, w> = <g, w> for every gauged w ----
        g = t3m.COREWISE.randn(frame, stack_shape=K)
        Pg = t3m.MANIFOLD.project(g)
        note(key + ' Pi gauged', np.max(Pg.gauge_residual))
        note(key + ' Pi idempotent', cw.corewise_relerr(Pg.variations.data, t3m.MANIFOLD.project(Pg).variations.data))
        note(key + ' Pi self-adjoint on gauged', relerr(g.corewise_inner(w), Pg.corewise_inner(w)))
        # ---- project_oblique: same vector, gauged ----
        Og = t3m.MANIFOLD.project_oblique(g)
        note(key + ' oblique same vector', relerr(g.to_dense(), Og.to_dense()))
        note(key + ' oblique gauged', np.max(Og.gauge_residual))
        note(key + ' oblique norm = HS norm of g', relerr(np.sqrt(hs_inner(g.to_dense(), g.to_dense(), d)), t3m.MANIFOLD.norm(Og)))
        # ---- project_ambient (dense, both methods, and T3) vs lstsq projection on the dense tangent basis ----
        Z = np.random.randn(*(K + C + shape))
        y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=K + C)
        PZ = t3m.MANIFOLD.project_ambient(frame, Z)
        PZ2 = t3m.MANIFOLD.project_ambient(frame, Z, method='t3svd')
        Py = t3m.MANIFOLD.project_ambient(frame, y)
        assert PZ.tangent_stack_shape == K and PZ.frame_stack_shape == C
        PZd, PZ2d, Pyd, Yd = PZ.to_dense(), PZ2.to_dense(), Py.to_dense(), y.to_dense()
        for kk in (np.ndindex(*K) if K else [()]):
            for ci, cc in enumerate(np.ndindex(*C) if C else [()]):
                B = bases[ci]
                ref = dense_project_onto_tangent(B, Z[kk + cc].reshape(-1)).reshape(shape)
                note(key + ' project_ambient(dense,contraction)', relerr(ref, PZd[kk + cc]))
                note(key + ' project_ambient(dense,t3svd)', relerr(ref, PZ2d[kk + cc]))
                refy = dense_project_onto_tangent(B, Yd[kk + cc].reshape(-1)).reshape(shape)
                note(key + ' project_ambient(T3)', relerr(refy, Pyd[kk + cc]))
        note(key + ' project_ambient gauged', np.max(PZ.gauge_residual))
        # ---- transport: to the frame of another point ----
        x2 = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
        frame2 = t3m.MANIFOLD.frame(x2)
        Tv = t3m.MANIFOLD.transport(v, frame2)
        Tvd = Tv.to_dense()
        bases2 = [tangent_basis_dense(f) for f in (frame2.unstack() if C else (frame2,))]
        for kk in (np.ndindex(*K) if K else [()]):
            for ci, cc in enumerate(np.ndindex(*C) if C else [()]):
                ref = dense_project_onto_tangent(bases2[ci], Vd[kk + cc].reshape(-1)).reshape(shape)
                note(key + ' transport', relerr(ref, Tvd[kk + cc]))
        # ---- retract: zero step = base point; first order; K-stacked step gives a K+C-stacked point ----
        zero = t3m.T3Tangent.zeros(frame, stack_shape=K)
        R0 = t3m.MANIFOLD.retract(zero)
        note(key + ' retract(0)=x', relerr(np.broadcast_to(x.to_dense(), K + C + shape), R0.to_dense()))
        assert R0.stack_shape == K + C, R0.stack_shape
        xd = np.broadcast_to(x.to_dense(), K + C + shape)
        errs = []
        for t_ in (1e-2, 1e-3):
            Rt = t3m.MANIFOLD.retract(t_ * v).to_dense()
            errs.append(np.linalg.norm((Rt - (xd + t_ * Vd)).reshape(-1)) / np.linalg.norm((t_ * Vd).reshape(-1)))
        note(key + ' retract first-order ratio (err(1e-2)/err(1e-3) ~ 10)', abs(errs[0] / max(errs[1], 1e-300) - 10) / 10)
        # ---- COREWISE twin ----
        cf = t3m.COREWISE.frame(x)
        cv = t3m.COREWISE.randn(cf, stack_shape=K)
        note(key + ' COREWISE.inner = corewise dot', relerr(cw.corewise_stack_dot(cv.variations.data, cv.variations.data, len(K + C)), t3m.COREWISE.inner(cv, cv)))
        Rc = t3m.COREWISE.retract(cv)
        U, G = x.data
        ref = t3.TuckerTensorTrain(tuple(u + du for u, du in zip(U, cv.variations.tucker_variations)),
                                   tuple(g_ + dg for g_, dg in zip(G, cv.variations.tt_variations)))
        note(key + ' COREWISE.retract additive', relerr(ref.to_dense(), Rc.to_dense()))
        assert Rc.stack_shape == K + C
        # corewise tangent dense == first-order change of the cores (d/dt of to_dense(cores + t dv))
        eps = 1e-6
        fd_ = (t3.TuckerTensorTrain(tuple(u + eps * du for u, du in zip(U, cv.variations.tucker_variations)),
                                    tuple(g_ + eps * dg for g_, dg in zip(G, cv.variations.tt_variations))).to_dense() - xd) / eps
        note(key + ' COREWISE tangent = directional derivative (1e-6 fd)', relerr(fd_, cv.to_dense()) / 1e-4)
    except Exception as e:
        print('CRASH', key, type(e).__name__, str(e)[:100]); worst[key + ' CRASH'] = 1.0

# ---- manifold_dim vs dense tangent-space rank, asymmetric, minimal + NON-minimal frames, + sharing ----
print('manifold_dim checks:')
for (shape, tr, ttr) in STRUCTS + [NONMIN]:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
    frame = t3m.MANIFOLD.frame(x)
    B = tangent_basis_dense(frame)
    sv = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(sv > 1e-9 * sv[0]))
    md = t3m.manifold_dim((shape, tr, ttr))
    print(f'  {shape} {tr} {ttr}: dense rank {rank}  manifold_dim {md}  tangent_space_dimension {t3m.T3Tangent.zeros(frame).tangent_space_dimension}'
          f'  has_minimal_ranks={frame.has_minimal_ranks}', 'OK' if rank == md else 'MISMATCH')
    note(f'manifold_dim {shape}', abs(rank - md))
for shape, tr, ttr, sharing in [((5, 5, 6), (2, 2, 3), (1, 3, 2, 1), (0, 0, 1)),
                                ((4, 6, 4, 6), (2, 3, 2, 3), (1, 2, 3, 2, 1), ('a', 'b', 'a', 'b')),
                                ((5, 5, 5), (3, 3, 3), (1, 3, 3, 1), (0, 0, 0))]:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr).share(sharing)
    geo = sg.shared_manifold(sharing)
    frame = geo.frame(x)
    n = 3 * t3m.manifold_dim((shape, tr, ttr))
    D = np.stack([geo.randn(frame).to_dense().reshape(-1) for _ in range(n)])
    sv = np.linalg.svd(D, compute_uv=False)
    rank = int(np.sum(sv > 1e-9 * sv[0]))
    md = t3m.manifold_dim((shape, tr, ttr), sharing=sharing)
    print(f'  sharing={sharing} {shape} {tr} {ttr}: dense rank {rank}  manifold_dim(sharing) {md}', 'OK' if rank == md else 'MISMATCH')
    note(f'manifold_dim sharing {sharing} {shape}', abs(rank - md))

bad = {k: e for k, e in worst.items() if not (e < 1e-8)}
print('checked', len(worst), 'quantities; max =', max(worst.values()))
print('FAILURES:' if bad else 'geometry vs dense: all pass', bad)
