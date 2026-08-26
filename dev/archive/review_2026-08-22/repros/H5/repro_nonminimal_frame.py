"""ut3_orthogonal_representations on NON-minimal-rank input (the contract says it must hold there too).
Compare the uniform frame's masks/ranks and orthogonality against the ragged T3Frame.from_t3."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.ranks as ranks
import t3toolbox.safety as safety
np.random.seed(0)
def rel(a, b): return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))
structs = [
    ((5, 7, 6), (2, 3, 2), (1, 2, 3, 1), 'bond 2 rank 3 > n2*r3 = 2 (non-minimal TT)'),
    ((6, 4), (3, 2), (1, 2, 1), 'tucker n0 = 3 > r0*r1 = 2 (non-minimal Tucker)'),
    ((5, 7, 6), (2, 3, 2), (1, 2, 2, 1), 'minimal (control)'),
    ((6, 6, 6), (2, 2, 2), (1, 3, 3, 1), 'the uniform_fitting doctest non-minimal example'),
]
for shape, tr, ttr, why in structs:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
    print('\n=== %s %s %s : %s' % (shape, tr, ttr, why))
    print('  ragged has_minimal_ranks:', x.has_minimal_ranks)
    rf = bvf.T3Frame.from_t3(x)
    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    uf = ut3m.UNIFORM_MANIFOLD.frame(ux)
    print('  ragged frame ranks   up %s down %s left %s right %s' % (tuple(rf.up_ranks), tuple(rf.down_ranks), tuple(rf.left_ranks), tuple(rf.right_ranks)))
    print('  uniform frame ranks  up %s down %s left %s right %s' % (tuple(np.asarray(uf.up_ranks).tolist()), tuple(np.asarray(uf.down_ranks).tolist()), tuple(np.asarray(uf.left_ranks).tolist()), tuple(np.asarray(uf.right_ranks).tolist())))
    print('  compute_orthogonal_representation_ranks:', ranks.compute_orthogonal_representation_ranks(shape, tr, ttr))
    print('  ragged frame is_orthogonal: %s   uniform frame is_orthogonal: %s (residual %.2e)' % (rf.is_orthogonal(), bool(uf.is_orthogonal()), float(np.max(uf.orthogonality_residual))))
    print('  uniform frame.to_dense == x: relerr %.2e ; ragged frame.to_dense == x: %.2e' % (rel(uf.to_dense(), x.to_dense()), rel(rf.to_dense(), x.to_dense())))
    # Where does the uniform frame fail orthogonality?  check each family's masked Gram vs diag(mask)
    up, down, left, right = uf.apply_masks().supercores
    um, dm, lm, rm = uf.masks.data
    G_up = np.einsum('dio,djo->dij', up, up)
    print('  up Gram vs diag(mask) max dev per mode:', [float(np.max(np.abs(G_up[i] - np.diag(um[i])))) for i in range(len(shape))])
    G_dn = np.einsum('diaj,dibj->dab', down, down)
    print('  down Gram vs diag(mask) max dev per mode:', [float(np.max(np.abs(G_dn[i] - np.diag(dm[i])))) for i in range(len(shape))])
    G_l = np.einsum('diaj,diak->djk', left[:-1], left[:-1])
    print('  left Gram vs diag(mask[1:]) per mode:', [float(np.max(np.abs(G_l[i] - np.diag(lm[i + 1])))) for i in range(len(shape) - 1)])
    G_r = np.einsum('diaj,dkaj->dik', right[1:], right[1:])
    print('  right Gram vs diag(mask) per mode:', [float(np.max(np.abs(G_r[i] - np.diag(rm[i + 1])))) for i in range(len(shape) - 1)])
    # is the content inside the mask?  (dense through mask == dense through to_t3frame)
    tf = uf.to_t3frame()
    print('  to_t3frame().to_dense == x: %.2e ; to_t3frame().is_orthogonal: %s' % (rel(tf.to_dense(), x.to_dense()), tf.is_orthogonal()))
    print('  to_t3frame core shapes: up %s down %s left %s right %s' % ([c.shape for c in tf.up_tucker_cores], [c.shape for c in tf.down_tt_cores], [c.shape for c in tf.left_tt_cores], [c.shape for c in tf.right_tt_cores]))
    print('  ragged frame core shapes: up %s down %s left %s right %s' % ([c.shape for c in rf.up_tucker_cores], [c.shape for c in rf.down_tt_cores], [c.shape for c in rf.left_tt_cores], [c.shape for c in rf.right_tt_cores]))
    # tangent ops at the uniform frame, unsafe, vs ragged
    with safety.unsafe():
        v = ut3m.UNIFORM_MANIFOLD.randn(uf)
        rv = v.to_t3tangent()
        print('  [unsafe] uniform randn tangent -> to_t3tangent.is_gauged: %s ; is_orthogonal(ragged frame of it): %s' % (rv.is_gauged(), rv.is_orthogonal()))
        y = ut3m.UNIFORM_MANIFOLD.retract(v)
        yr = t3m.MANIFOLD.retract(t3m.MANIFOLD.project(t3m.T3Tangent(rf, rv.variations)) if False else rv) if rv.is_orthogonal() else None
        print('  [unsafe] retract(v) dense vs ragged retract(to_t3tangent(v)): %s' % ('%.2e' % rel(y.to_dense(), yr.to_dense()) if yr is not None else 'n/a (ragged frame of v not orthogonal)'))
        nrm = ut3m.UNIFORM_MANIFOLD.norm(v)
        print('  [unsafe] uniform MANIFOLD.norm(v) = %.6f ; dense HS norm of v = %.6f' % (float(nrm), np.linalg.norm(v.to_dense())))
