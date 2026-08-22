"""R4-2: K-over-C broadcast in tv_operations + the four tv_{stack,unstack}_{tangent,frame}_stack converters
and the T3Tangent stack methods. Per-element references are built by manual slicing."""
import itertools
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.stacking as stacking
from r4_common import STRUCTS, leaf, relerr

np.random.seed(0)
TOL = 1e-10
worst = {}


def note(key, err):
    worst[key] = max(worst.get(key, 0.0), err)


def slice_tree(data, idx):
    """slice every core of a data tree at leading index tuple idx"""
    if isinstance(data, (tuple, list)):
        return tuple(slice_tree(c, idx) for c in data)
    return data[idx]


for (shape, tr, ttr), K, C in itertools.product(STRUCTS[1:],
                                                 [(), (2,), (2, 3)],
                                                 [(), (3,), (2, 2)]):
    d = len(shape)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
    frame = t3m.MANIFOLD.frame(x)
    v = t3m.COREWISE.randn(frame, stack_shape=K)           # ungauged, K+C
    assert v.tangent_stack_shape == K and v.frame_stack_shape == C
    fd, vd = frame.data, v.variations.data
    key = f'd={d} K={K} C={C}'

    # reference per element
    def ref_elem(fn, kk, cc):
        return fn(slice_tree(fd, cc), slice_tree(vd, kk + cc))

    # --- tv_to_dense (both shifts), gauge projections, gauge residual, retract, to_t3 -----------------
    dense = tv.tv_to_dense(fd, vd)
    assert dense.shape == K + C + shape, (dense.shape, K, C, shape)
    dense_s = tv.tv_to_dense(fd, vd, include_shift=True)
    og = tv.tv_orthogonal_gauge_projection(fd, vd)
    ob = tv.tv_oblique_gauge_projection(fd, vd)
    gr = tv.tv_gauge_residual(fd, vd)
    assert gr.shape == K + C, (gr.shape, K, C)
    ret = tv.tv_retract(fd, vd)
    emb = tv.tv_to_t3(fd, vd, include_shift=True)
    # project a K+C-stacked T3 / dense onto the C-stacked frame
    y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=K + C)
    pt3 = tv.tv_project_t3_onto_tangent_space(fd, y.data)
    Z = np.random.randn(*(K + C + shape))
    pdn = tv.tv_project_dense_onto_tangent_space(fd, Z)
    for kk in np.ndindex(*K) if K else [()]:
        for cc in np.ndindex(*C) if C else [()]:
            idx = kk + cc
            note(key + ' to_dense', relerr(ref_elem(tv.tv_to_dense, kk, cc), dense[idx]))
            note(key + ' to_dense+shift', relerr(ref_elem(lambda a, b: tv.tv_to_dense(a, b, include_shift=True), kk, cc), dense_s[idx]))
            r_og = ref_elem(tv.tv_orthogonal_gauge_projection, kk, cc)
            note(key + ' orth_gauge', cw.corewise_relerr(r_og, slice_tree(og, idx)))
            r_ob = ref_elem(tv.tv_oblique_gauge_projection, kk, cc)
            note(key + ' oblique_gauge', cw.corewise_relerr(r_ob, slice_tree(ob, idx)))
            note(key + ' gauge_residual', abs(float(ref_elem(tv.tv_gauge_residual, kk, cc)) - float(gr[idx])))
            r_ret = ref_elem(tv.tv_retract, kk, cc)
            note(key + ' retract(dense)', relerr(t3.TuckerTensorTrain(*r_ret).to_dense(),
                                                t3.TuckerTensorTrain(*slice_tree(ret, idx)).to_dense()))
            r_emb = ref_elem(lambda a, b: tv.tv_to_t3(a, b, include_shift=True), kk, cc)
            note(key + ' to_t3+shift(dense)', relerr(t3.TuckerTensorTrain(*r_emb).to_dense(),
                                                    t3.TuckerTensorTrain(*slice_tree(emb, idx)).to_dense()))
            r_pt3 = tv.tv_project_t3_onto_tangent_space(slice_tree(fd, cc), slice_tree(y.data, idx))
            note(key + ' project_t3', cw.corewise_relerr(r_pt3, slice_tree(pt3, idx)))
            r_pdn = tv.tv_project_dense_onto_tangent_space(slice_tree(fd, cc), Z[idx])
            note(key + ' project_dense', cw.corewise_relerr(r_pdn, slice_tree(pdn, idx)))

    # --- the four stack converters --------------------------------------------------------------------
    vt = tv.tv_unstack_tangent_stack(fd, vd)                    # K-shaped tree of variations (stack C)
    if K:
        for kk in np.ndindex(*K):
            note(key + ' unstack_tangent leaf', cw.corewise_relerr(slice_tree(vd, kk), leaf(vt, kk)))
        vd2 = tv.tv_stack_tangent_stack(vt)
        note(key + ' stack_tangent roundtrip', cw.corewise_relerr(vd, vd2))
    pt = tv.tv_unstack_frame_stack(fd, vd)                      # C-shaped tree of (frame, variations)
    if C:
        for cc in np.ndindex(*C):
            fl, vl = leaf(pt, cc)
            note(key + ' unstack_frame leaf(frame)', cw.corewise_relerr(slice_tree(fd, cc), fl))
            # variations leaf must be the K-stacked slice at c: vd[..., c, ...] with K leading
            ref_vl = tuple(tuple(np.moveaxis(c_, list(range(len(K), len(K) + len(C))), list(range(len(C))))[cc]
                                 for c_ in fam) for fam in vd)
            note(key + ' unstack_frame leaf(vars)', cw.corewise_relerr(ref_vl, vl))
        fd2, vd2 = tv.tv_stack_frame_stack(pt)
        note(key + ' stack_frame roundtrip(frame)', cw.corewise_relerr(fd, fd2))
        note(key + ' stack_frame roundtrip(vars)', cw.corewise_relerr(vd, vd2))
    else:
        # C=() : unstack_frame_stack must give a single pair
        fl, vl = pt
        note(key + ' unstack_frame C=() pair', cw.corewise_relerr(fd, fl) + cw.corewise_relerr(vd, vl))
        fd2, vd2 = tv.tv_stack_frame_stack(pt)
        note(key + ' stack_frame C=() roundtrip', cw.corewise_relerr(fd, fd2) + cw.corewise_relerr(vd, vd2))

    # --- frontend round trips ---------------------------------------------------------------------------
    if K:
        tt = v.unstack_tangents()
        v2 = t3m.T3Tangent.stack_tangents(tt)
        note(key + ' T3Tangent.stack_tangents roundtrip', cw.corewise_relerr(vd, v2.variations.data))
        assert all(t_.frame is frame for t_ in t3m._flatten_tangents(tt))
    if C:
        ft = v.unstack_frame()
        v3 = t3m.T3Tangent.stack_frame(ft)
        note(key + ' T3Tangent.stack_frame roundtrip', cw.corewise_relerr(vd, v3.variations.data)
             + cw.corewise_relerr(fd, v3.frame.data))
        assert all(t_.tangent_stack_shape == K and t_.frame_stack_shape == () for t_ in t3m._flatten_tangents(ft))
    else:
        ft = v.unstack_frame()
        assert isinstance(ft, t3m.T3Tangent), type(ft)

bad = {k: e for k, e in worst.items() if not (e < TOL)}
print('checked', len(worst), 'quantities; max err =', max(worst.values()))
print('FAILURES:' if bad else 'all K-over-C checks pass', bad)
