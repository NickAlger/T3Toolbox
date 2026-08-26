"""MANIFOLD.retract first-order check at a point whose orthogonal frame is NOT minimal-rank.
Two ways to get there: (a) x.share(...) (the group basis exceeds what the TT core spans), (b) a plain randn
with a structurally non-minimal Tucker rank (n0 > r0*r1)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
import t3toolbox as t3t

def fd_ladder(geom, frame, v, Vd, hs=(1e-2, 1e-3, 1e-4, 1e-5)):
    out = []
    for h in hs:
        rp = np.asarray(geom.retract(v * h).to_dense()); rm = np.asarray(geom.retract(v * (-h)).to_dense())
        out.append(np.linalg.norm(((rp - rm) / (2 * h) - Vd).reshape(-1)) / np.linalg.norm(Vd.reshape(-1)))
    return out

np.random.seed(1)
cases = []
x = t3.TuckerTensorTrain.randn((4, 4, 4), (2, 2, 2), (1, 2, 3, 1)).share((0, 0, 0)); cases.append(('share((0,0,0)) of (4,4,4),(2,2,2),(1,2,3,1)', x, (0, 0, 0)))
x = t3.TuckerTensorTrain.randn((5, 5, 4, 4), (2, 2, 3, 3), (1, 2, 3, 2, 1)).share(('a', 'a', 'b', 'b')); cases.append(("share(aabb) of (5,5,4,4),(2,2,3,3),(1,2,3,2,1)", x, ('a', 'a', 'b', 'b')))
x = t3.TuckerTensorTrain.randn((5, 5, 4), (4, 2, 2), (1, 2, 2, 1)); cases.append(('plain randn (5,5,4),(4,2,2),(1,2,2,1): n0=4 > r0*r1=2', x, None))
x = t3.TuckerTensorTrain.randn((5, 5, 4), (2, 2, 2), (1, 2, 2, 1)); cases.append(('plain randn (5,5,4),(2,2,2),(1,2,2,1): minimal (control)', x, None))
for name, x, sh in cases:
    frame, _ = bvf.t3_orthogonal_representations(x)
    print('\n==', name)
    print('   x ranks', x.tucker_ranks, x.tt_ranks, '| frame up/down/left/right ranks', frame.up_ranks, frame.down_ranks, frame.left_ranks, frame.right_ranks)
    print('   frame.has_minimal_ranks =', frame.has_minimal_ranks, '| numerically minimal =', bool(np.all(frame.has_numerically_minimal_ranks())),
          '| is_orthogonal =', bool(np.all(frame.is_orthogonal())))
    np.random.seed(4)
    v = t3m.MANIFOLD.randn(frame); Vd = np.asarray(v.to_dense())
    print('   MANIFOLD.retract FD relerr ladder h=1e-2..1e-5:', ['%.2e' % e for e in fd_ladder(t3m.MANIFOLD, frame, v, Vd)])
    r = t3m.MANIFOLD.retract(v * 1e-3)
    print('   retract(1e-3 v) ranks', r.tucker_ranks, r.tt_ranks)
    if sh is not None:
        geom = t3t.shared_manifold(sh)
        vs = geom.project(v); Vsd = np.asarray(vs.to_dense())
        print('   shared_manifold.retract FD ladder (tied tangent):', ['%.2e' % e for e in fd_ladder(geom, frame, vs, Vsd)])
