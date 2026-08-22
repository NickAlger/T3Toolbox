"""R4-4: fv_operations constructors / fv_variation_shapes / fv_frame_reverse (untested), and
fv_conversions.t3_orthogonal_representations vs Appendix A Algorithm 11; plus T3Frame.validate message."""
import itertools
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.t3_orthogonalization as ro
import t3toolbox.backend.tt_orthogonalization as tto
from r4_common import STRUCTS, NONMIN, relerr

np.random.seed(2)
ok = True
def check(name, cond, info=''):
    global ok
    ok &= bool(cond)
    print(('  ok   ' if cond else '  FAIL ') + name, info)

for (shape, tr, ttr), C in itertools.product(STRUCTS + [NONMIN], [(), (2, 3)]):
    d = len(shape)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
    frame, variations = bvf.t3_orthogonal_representations(x)
    key = f'd={d} tr={tr} C={C}:'
    # fv_variation_shapes == T3Frame.variation_shapes
    check(key + ' fv_variation_shapes', fvo.fv_variation_shapes(frame.data) == frame.variation_shapes,
          str(fvo.fv_variation_shapes(frame.data)))
    vs = frame.variation_shapes
    # zeros / randn shapes with an extra K
    K = (2,)
    z = fvo.fv_variations_zeros(vs, K + C)
    r = fvo.fv_variations_randn(vs, K + C)
    check(key + ' zeros/randn shapes',
          all(c.shape == K + C + s for c, s in zip(z[0], vs[0])) and all(c.shape == K + C + s for c, s in zip(z[1], vs[1]))
          and all(c.shape == K + C + s for c, s in zip(r[0], vs[0])) and all(c.shape == K + C + s for c, s in zip(r[1], vs[1]))
          and float(cw.corewise_norm(z)) == 0.0)
    # unit: single 1 broadcast over the stack
    i = d - 1; idx = tuple(s - 1 for s in vs[1][i])
    u = fvo.fv_variations_unit(vs, (True, i, idx), K + C)
    check(key + ' unit', float(np.sum(u[1][i])) == np.prod(K + C) and np.isclose(float(cw.corewise_norm(u)) ** 2, np.prod(K + C))
          and bool(np.all(u[1][i][(Ellipsis,) + idx] == 1.0)))
    # from_vector round trip with t3_to_vector on a K+C stack
    flat = t3c.t3_to_vector(r)
    back = fvo.fv_variations_from_vector(flat, vs, K + C)
    check(key + ' from_vector roundtrip', cw.corewise_relerr(r, back) == 0.0)
    # T3Tangent.from_vector / to_vector
    tng = t3m.COREWISE.randn(frame, stack_shape=K)
    tng2 = t3m.T3Tangent.from_vector(tng.to_vector(), frame, tangent_stack_shape=K)
    check(key + ' T3Tangent.from_vector roundtrip', cw.corewise_relerr(tng.variations.data, tng2.variations.data) == 0.0)

    # fv_frame_reverse
    rf = bvf.T3Frame(*fvo.fv_frame_reverse(frame.data))
    check(key + ' reverse frame orthogonal', bool(rf.is_orthogonal().all()))
    check(key + ' reverse frame consistent', bool(rf.is_consistent().all()))
    perm = tuple(range(len(C))) + tuple(len(C) + d - 1 - m for m in range(d))
    check(key + ' reverse frame base point', relerr(np.transpose(frame.to_dense(), perm), rf.to_dense()) < 1e-12)
    check(key + ' reverse involution', cw.corewise_relerr(frame.data, fvo.fv_frame_reverse(rf.data)) == 0.0)
    check(key + ' reverse variation_shapes fit', rf.variation_shapes == bvf.T3Variations(*variations.reverse().data).variation_shapes)
    tr_ = t3m.T3Tangent(frame, variations).reverse()
    check(key + ' T3Tangent.reverse commutes with to_dense',
          relerr(np.transpose(tv.tv_to_dense(frame.data, variations.data), perm), tr_.to_dense()) < 1e-12)

    # t3_orthogonal_representations: every single-core term reconstructs x (Alg 11 output contract)
    errs = []
    for i_ in range(d):
        for tt_ in (True, False):
            errs.append(relerr(x.to_dense(), bvf.fv_to_t3((tt_, i_), frame, variations).to_dense()))
    check(key + ' all 2d terms reconstruct x', max(errs) < 1e-12, f'max={max(errs):.1e}')
    check(key + ' frame orthogonal', bool(frame.is_orthogonal().all()))
    # already_left_orthogonal path
    UL = ro.t3_left_orthogonalize(x.data)
    f2, v2 = fvc.t3_orthogonal_representations(UL, already_left_orthogonal=True)
    f2 = bvf.T3Frame(*f2); v2 = bvf.T3Variations(*v2)
    check(key + ' already_left_orthogonal path', bool(f2.is_orthogonal().all()) and
          relerr(x.to_dense(), bvf.fv_to_t3((False, 0), f2, v2).to_dense()) < 1e-12)
    # paper Algorithm 11 order (right sweep THEN left sweep) -- equivalent orthogonal representation?
    U1, G1 = ro.t3_down_orthogonalize_tucker_cores((x.data[0], x.data[1]))
    Q = tto.tt_right_orthogonalize(G1)
    P, Gt = tto.tt_left_orthogonalize(Q, return_variation_cores=True)
    V1, O1 = ro.t3_up_orthogonalize_tt_cores((U1, Gt))
    fp = bvf.T3Frame(U1, O1, P, Q); vp = bvf.T3Variations(V1, Gt)
    errs = [relerr(x.to_dense(), bvf.fv_to_t3((tt_, i_), fp, vp).to_dense()) for i_ in range(d) for tt_ in (True, False)]
    check(key + ' paper-order Alg11 equivalent', bool(fp.is_orthogonal().all()) and max(errs) < 1e-12,
          f'structures equal: {fp.structure == frame.structure}')

# squash_tails=False with boundary ranks > 1
x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 2, 4), (2, 2, 3, 3))
f3, v3 = bvf.t3_orthogonal_representations(x, squash_tails=False)
errs = [relerr(x.to_dense(), bvf.fv_to_t3((tt_, i_), f3, v3).to_dense()) for i_ in range(3) for tt_ in (True, False)]
check('squash_tails=False boundary ranks (2,..,3)', max(errs) < 1e-12, f'left_ranks={f3.left_ranks} right_ranks={f3.right_ranks} orth={bool(f3.is_orthogonal())}')

# T3Frame.validate error message on a STACKED frame with a Tucker-rank mismatch prints shape[0]/shape[1]
x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 2, 4), (1, 2, 3, 1), stack_shape=(2, 3))
frame, _ = bvf.t3_orthogonal_representations(x)
U, O, L, R = frame.data
badU = list(U); badU[1] = np.random.randn(2, 3, 5, 6)     # nU=5 instead of 2
try:
    bvf.T3Frame(tuple(badU), O, L, R)
except ValueError as e:
    print('validate message for stacked mismatch (actual ranks: U 5, L 2, R 2):\n   ', str(e).replace('\n', ' | '))

print('ALL OK' if ok else 'SOME FAILURES')
