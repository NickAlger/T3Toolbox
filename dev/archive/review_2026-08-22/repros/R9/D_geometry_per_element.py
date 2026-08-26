"""R9-D: UniformManifoldGeometry / UniformCorewiseGeometry vs the ragged twins PER ELEMENT at a hetero
C=(2,) (different ranks) with K=(3,), three prongs: dense-vs-ragged, exact output masks, garbage robustness
(frame AND variations padding corrupted) for every geometry op + to_ut3 + gauge_residual + corewise_inner."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.manifold as t3m
import t3toolbox.frame_variations_format as bvf
import t3toolbox.backend.utv_operations as utvo

def rep(msg, ok, val=None):
    print(('PASS ' if ok else 'FAIL ') + msg + ('' if val is None else '  [%s]' % (val,)))

np.random.seed(0)
HET = [((4, 5, 6, 3), (2, 2, 2, 2), (1, 2, 3, 2, 1)), ((4, 5, 6, 3), (3, 3, 2, 2), (1, 1, 2, 2, 1))]
PAD = dict(N=6, nU=4, nD=5, rL=4, rR=4)
K = (3,)
us, rs, xs = [], [], []
for s in HET:
    x = t3.TuckerTensorTrain.randn(*s); xs.append(x)
    rb, rv = bvf.t3_orthogonal_representations(x)
    us.append(ut3m.UT3Tangent(ubv.UT3Frame.from_t3frame(rb, **PAD), ubv.UT3Variations.from_t3variations(rv, **PAD)))
    rs.append(t3m.T3Tangent(rb, rv))
vC = ut3m.UT3Tangent.stack_frame(tuple(us))
# K-stacked raw variations at the C-stacked frame, and the SAME directions ragged per element
np.random.seed(1)
vKC = ut3m.UNIFORM_COREWISE.randn(vC.frame, stack_shape=K)          # raw randn, masks bcast over K
def ragged_leaf(vu, k, c):   # the (k,c) leaf of a K+C uniform tangent as a ragged tangent at rs[c].frame
    leaf = vu.unstack_tangents()[k].unstack_frame()[c]
    return t3m.T3Tangent(rs[c].frame, leaf.variations.to_t3variations())
def dense_u(t): return np.asarray(t.to_dense())

def corrupt(obj, scale=1e3):
    scs = obj.supercores
    if isinstance(obj, ubv.UT3Variations):
        ind = ubv.UT3Variations(*[np.ones_like(s) for s in scs], obj.shape, obj.masks).apply_masks().supercores
        new = [sc + scale * (1.0 - i) for sc, i in zip(scs, ind)]
        return ubv.UT3Variations(new[0], new[1], obj.shape, obj.masks)
    if isinstance(obj, ubv.UT3Frame):
        ind = ubv.UT3Frame(*[np.ones_like(s) for s in scs], obj.shape, obj.masks).apply_masks().supercores
        new = [sc + scale * (1.0 - i) for sc, i in zip(scs, ind)]
        return ubv.UT3Frame(*new, obj.shape, obj.masks)
    ind = ut3.UniformTuckerTensorTrain(*[np.ones_like(s) for s in scs], obj.shape, obj.masks).apply_masks().supercores
    new = [sc + scale * (1.0 - i) for sc, i in zip(scs, ind)]
    return ut3.UniformTuckerTensorTrain(new[0], new[1], obj.shape, obj.masks)

vG = ut3m.UT3Tangent(corrupt(vKC.frame), corrupt(vKC.variations))   # garbage in BOTH paddings
M, CW = ut3m.UNIFORM_MANIFOLD, ut3m.UNIFORM_COREWISE
rM, rCW = t3m.MANIFOLD, t3m.COREWISE

def per_element(msg, uni_dense, ragged_fn, tol=1e-9):
    worst = 0.0
    for k in range(K[0]):
        for c in range(2):
            dr = np.asarray(ragged_fn(k, c))
            worst = max(worst, np.linalg.norm(uni_dense[k, c] - dr) / max(1e-300, np.linalg.norm(dr)))
    rep(msg, worst < tol, worst)

try:
    # ---- project (orthogonal gauge) -------------------------------------------------------------------------
    pu = M.project(vKC); pg = M.project(vG)
    per_element('MANIFOLD.project == ragged per element (dense)', dense_u(pu), lambda k, c: rM.project(ragged_leaf(vKC, k, c)).to_dense())
    rep('MANIFOLD.project garbage-robust (frame+variations padding)', np.allclose(dense_u(pg), dense_u(pu), atol=1e-8))
    rep('MANIFOLD.project output masks == input masks', pu.variations.masks == vKC.variations.masks)
    # ---- project_oblique -------------------------------------------------------------------------------------
    ou = M.project_oblique(vKC); og = M.project_oblique(vG)
    rep('project_oblique preserves the vector (dense)', np.allclose(dense_u(ou), dense_u(vKC), atol=1e-8))
    per_element('project_oblique == ragged per element (dense)', dense_u(ou), lambda k, c: rM.project_oblique(ragged_leaf(vKC, k, c)).to_dense())
    rep('project_oblique garbage-robust', np.allclose(dense_u(og), dense_u(vKC), atol=1e-8))
    gres = np.asarray(ou.gauge_residual); rep('project_oblique gauged per (k,c)', gres.shape == (3, 2) and gres.max() < 1e-9, gres.max())
    # ---- inner / norm (on the gauged tangents) vs ragged per element --------------------------------------------
    inn = np.asarray(M.inner(pu, ou)); nrm = np.asarray(M.norm(pu))
    worst = max(abs(float(inn[k, c]) - float(rM.inner(rM.project(ragged_leaf(vKC, k, c)), rM.project_oblique(ragged_leaf(vKC, k, c))))) for k in range(3) for c in range(2))
    rep('MANIFOLD.inner per (k,c) == ragged', inn.shape == (3, 2) and worst < 1e-9, worst)
    worst = max(abs(float(nrm[k, c]) - float(rM.norm(rM.project(ragged_leaf(vKC, k, c))))) for k in range(3) for c in range(2))
    rep('MANIFOLD.norm per (k,c) == ragged', worst < 1e-9, worst)
    # HS faithfulness: inner == dense dot
    worst = max(abs(float(inn[k, c]) - float(np.sum(dense_u(pu)[k, c] * dense_u(ou)[k, c]))) for k in range(3) for c in range(2))
    rep('MANIFOLD.inner == dense HS dot per element', worst < 1e-9, worst)
    ing = np.asarray(pg.corewise_inner(og))
    rep('corewise_inner garbage-robust', np.allclose(ing, np.asarray(pu.corewise_inner(ou)), atol=1e-8))
    # ---- retract ---------------------------------------------------------------------------------------------------
    yu = M.retract(pu); yg = M.retract(pg)
    per_element('MANIFOLD.retract == ragged per element (dense)', np.asarray(yu.to_dense()), lambda k, c: rM.retract(rM.project(ragged_leaf(vKC, k, c))).to_dense(), tol=1e-8)
    rep('MANIFOLD.retract garbage-robust', np.allclose(np.asarray(yg.to_dense()), np.asarray(yu.to_dense()), atol=1e-7))
    def is_prefix(m): return bool(np.array_equal(m, np.arange(m.shape[-1]) < m.sum(axis=-1)[..., None]))
    rep('retract output masks prefix', is_prefix(yu.masks.tucker_edge_mask) and is_prefix(yu.masks.tt_edge_mask))
    exp_tk = np.broadcast_to(vC.frame.masks.up_mask.sum(-1)[:, None], (4, 3, 2)); exp_tt = np.broadcast_to(vC.frame.masks.frame_left_mask.sum(-1)[:, None], (5, 3, 2))
    rep('retract output ranks == frame ranks per element, bcast over K', np.array_equal(yu.masks.tucker_edge_mask.sum(-1), exp_tk) and np.array_equal(yu.masks.tt_edge_mask.sum(-1), exp_tt))
    print('    frame up ranks per C  :', vC.frame.masks.up_mask.sum(-1).T.tolist(), ' left:', vC.frame.masks.frame_left_mask.sum(-1).T.tolist())
    print('    retract tucker ranks  :', yu.masks.tucker_edge_mask.sum(-1)[:, 0].T.tolist(), ' tt:', yu.masks.tt_edge_mask.sum(-1)[:, 0].T.tolist())
    print('    frame has_minimal_ranks per C:', vC.frame.has_minimal_ranks, ' (HET[1] is deliberately non-minimal: n0=3 > r0*r1=1)')
    # ---- to_ut3 (exact doubled masks, garbage) ------------------------------------------------------------------------
    eu = vKC.to_ut3(); eg = vG.to_ut3()
    rep('to_ut3 garbage-robust (dense)', np.allclose(np.asarray(eg.to_dense()), np.asarray(eu.to_dense()), atol=1e-8))
    per_element('to_ut3(include_shift=True) == ragged', np.asarray(vKC.to_ut3(True).to_dense()), lambda k, c: ragged_leaf(vKC, k, c).to_t3(True).to_dense())
    # ---- project_ambient / transport -----------------------------------------------------------------------------------
    np.random.seed(2)
    zs = [t3.TuckerTensorTrain.randn(*HET[c]) for c in range(2)]
    z = ut3.UniformTuckerTensorTrain.stack(tuple(ut3.UniformTuckerTensorTrain.from_t3(zz, N=6, n=5, r=4) for zz in zs))
    pa = M.project_ambient(vC.frame, z); pag = M.project_ambient(corrupt(vC.frame), corrupt(z))
    worst = max(np.linalg.norm(np.asarray(pa.to_dense())[c] - np.asarray(rM.project_ambient(rs[c].frame, zs[c]).to_dense())) for c in range(2))
    rep('project_ambient == ragged per element', worst < 1e-9, worst)
    rep('project_ambient garbage-robust', np.allclose(np.asarray(pag.to_dense()), np.asarray(pa.to_dense()), atol=1e-8))
    rep('project_ambient output masks == frame gauge masks', pa.variations.masks == ubv.UT3Variations._variation_masks_of(vC.frame))
    gres = np.asarray(pa.gauge_residual); rep('project_ambient gauged', gres.max() < 1e-9, gres.max())
    # transport of the K+C tangent to a new C-stacked frame
    np.random.seed(3)
    x2s = [t3.TuckerTensorTrain.randn(*HET[c]) for c in range(2)]
    f2 = ubv.UT3Frame.stack(tuple(ubv.UT3Frame.from_t3frame(bvf.T3Frame.from_t3(xx), **PAD) for xx in x2s))
    try:
        tu = M.transport(pu, f2)
        rep('transport of a K+C tangent to a C frame works', True, tu.stack_shape)
    except Exception as e:
        rep('CHARACTERIZE: UNIFORM_MANIFOLD.transport(K+C tangent, C frame) CRASHES', False, '%s: %s' % (type(e).__name__, str(e)[:90]))
    try:
        rvK = rM.randn(rs[0].frame, stack_shape=(3,))
        rt = rM.transport(rvK, bvf.T3Frame.from_t3(x2s[0]))
        rep('  ragged MANIFOLD.transport(K tangent, frame) works', True, (rt.tangent_stack_shape, rt.frame_stack_shape))
    except Exception as e:
        rep('  ragged MANIFOLD.transport(K tangent) also raises', False, '%s: %s' % (type(e).__name__, str(e)[:90]))
    try:
        rpa = rM.project_ambient(rs[0].frame, rvK.to_t3())
        rep('  ragged MANIFOLD.project_ambient(frame, K-stacked T3) works', True, rpa.tangent_stack_shape)
    except Exception as e:
        rep('  ragged MANIFOLD.project_ambient(frame, K-stacked T3) raises', False, '%s: %s' % (type(e).__name__, str(e)[:90]))
    try:
        upa = M.project_ambient(vC.frame, vKC.to_ut3())
        rep('  uniform project_ambient(frame C, grad K+C) works', True, upa.stack_shape)
    except Exception as e:
        rep('  uniform project_ambient(frame C, grad K+C) raises', False, '%s: %s' % (type(e).__name__, str(e)[:90]))
    # transport with K=() per element (the supported shape) + garbage
    puC = M.project(vC); pgC = M.project(ut3m.UT3Tangent(corrupt(vC.frame), corrupt(vC.variations)))
    tu = M.transport(puC, f2); tg = M.transport(pgC, corrupt(f2))
    worst = max(np.linalg.norm(np.asarray(tu.to_dense())[c] - np.asarray(rM.transport(rM.project(rs[c]), bvf.T3Frame.from_t3(x2s[c])).to_dense())) for c in range(2))
    rep('transport (C only) == ragged per element', worst < 1e-9, worst)
    rep('transport garbage-robust', np.allclose(np.asarray(tg.to_dense()), np.asarray(tu.to_dense()), atol=1e-8))
    # ---- corewise geometry ---------------------------------------------------------------------------------------------
    xC = ut3.UniformTuckerTensorTrain.stack(tuple(ut3.UniformTuckerTensorTrain.from_t3(xx, N=6, n=5, r=4) for xx in xs))
    cf = CW.frame(xC); vc = CW.randn(cf, stack_shape=K); vcg = ut3m.UT3Tangent(corrupt(cf), corrupt(vc.variations))
    yc = CW.retract(vc); ycg = CW.retract(vcg)
    worst = 0.0
    for k in range(3):
        for c in range(2):
            leaf = vc.unstack_tangents()[k].unstack_frame()[c]
            rt = t3m.T3Tangent(rCW.frame(xs[c]), leaf.variations.to_t3variations())
            worst = max(worst, np.linalg.norm(np.asarray(yc.to_dense())[k, c] - np.asarray(rCW.retract(rt).to_dense())))
    rep('COREWISE.retract == ragged per element', worst < 1e-9, worst)
    rep('COREWISE.retract garbage-robust', np.allclose(np.asarray(ycg.to_dense()), np.asarray(yc.to_dense()), atol=1e-7))
    rep('COREWISE.retract output masks == point masks bcast over K', np.array_equal(yc.masks.tucker_edge_mask, np.broadcast_to(xC.masks.tucker_edge_mask[:, None], (4, 3, 2, 5))))
    rep('COREWISE.inner per (k,c) shape', np.asarray(CW.inner(vc, vc)).shape == (3, 2))
    # ---- gauge_residual per element vs ragged, garbage --------------------------------------------------------------------
    gr = np.asarray(vKC.gauge_residual); grg = np.asarray(vG.gauge_residual)
    worst = max(abs(float(gr[k, c]) - float(ragged_leaf(vKC, k, c).gauge_residual)) for k in range(3) for c in range(2))
    rep('gauge_residual per (k,c) == ragged', worst < 1e-9, worst)
    rep('gauge_residual garbage-robust', np.allclose(grg, gr, atol=1e-8))
    # ---- sum_tangents per element ---------------------------------------------------------------------------------------
    st = vKC.sum_tangents()
    rep('sum_tangents dense == sum over K of dense', np.allclose(np.asarray(st.to_dense()), dense_u(vKC).sum(axis=0)))
    # ---- normalized ---------------------------------------------------------------------------------------------------
    nn = np.asarray(vKC.normalized().corewise_norm()); rep('normalized() has unit corewise norm per (k,c)', np.allclose(nn, 1.0))
    # ---- weighted norm on K+C with a C metric ------------------------------------------------------------------------------
    Wt = ubv.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(vC.frame.to_ut3()))
    try:
        wn = np.asarray(pu.weighted_norm(Wt)); rep('weighted_norm on K+C with C metric -> shape (3,2)', wn.shape == (3, 2), wn.shape)
    except Exception as e:
        rep('weighted_norm on K+C with C metric', False, '%s: %s' % (type(e).__name__, str(e)[:200]))
except Exception:
    traceback.print_exc()
