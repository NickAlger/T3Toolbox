"""R9-A2: tied-tangent (shared) uniform ops at C=(2,), K=(3,): stacked vs UNSTACKED-uniform per element on
the SAME frame slices (the unstacked path is the one test_sharing vouches for vs ragged), plus the
companion spectrum vs ragged (gauge-invariant), garbage prongs, and the frontend untie characterization."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.manifold as t3m
import t3toolbox.frame_variations_format as bvf
import t3toolbox.backend.sharing as sharing
import t3toolbox.backend.utv_operations as utvo
import t3toolbox.backend.tv_operations as tvo
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.shared_geometry as sg

def rep(msg, ok, val=None):
    print(('PASS ' if ok else 'FAIL ') + msg + ('' if val is None else '  [%s]' % (val,)))

np.random.seed(0)
shape, n, r, spec = (5, 6, 5, 6), (2, 3, 2, 3), (1, 2, 3, 2, 1), ('a', 'b', 'a', 'b')
C, K = (2,), (3,)
groups = sharing.validate_sharing(spec, shape)
import t3toolbox.backend.ranks as ranks
assert ranks.compute_minimal_ranks(shape, n, r, sharing=spec) == (n, r)

def tied(shape, n, r):
    x = t3.TuckerTensorTrain.randn(shape, n, r); tk = list(x.tucker_cores)
    for g in groups:
        for ii in g[1:]: tk[ii] = tk[g[0]]
    return t3.TuckerTensorTrain(tuple(tk), x.tt_cores)
xs = [tied(shape, n, r) for _ in range(C[0])]
x = t3.TuckerTensorTrain.stack(tuple(xs))
u = ut3.UniformTuckerTensorTrain.from_t3(x)
frame_u, _ = ubv.ut3_orthogonal_representations(u)
sfd_u = sharing.ufv_shared_frame_data(frame_u.data, groups)
rep('stacked frame tied per element', bool(np.all(np.asarray(frame_u.to_ut3().has_shared_tucker_factors(spec)))))

# unstacked uniform frames from the SAME supercore slices + their companions
frames_c = frame_u.unstack()
sfd_c = [sharing.ufv_shared_frame_data(f.data, groups) for f in frames_c]
rep('unstacked frame slices are bitwise the stacked slices', all(np.array_equal(frames_c[c].left_tt_supercore, frame_u.left_tt_supercore[:, c]) for c in range(2)))
# companion spectrum: stacked vs unstacked vs ragged (own frame) -- gauge-invariant
ok = True; worst = 0.0
for gi in range(len(sharing.nontrivial_groups(groups))):
    for c in range(2):
        s_st = np.asarray(sfd_u.svd_s[gi])[c]; s_un = np.asarray(sfd_c[c].svd_s[gi])
        fr_r, _ = fvc.t3_orthogonal_representations(xs[c].data)
        s_r = np.asarray(sharing.fv_shared_frame_data(fr_r, groups).svd_s[gi])
        worst = max(worst, np.abs(s_st - s_un).max(), np.abs(s_st[:s_r.size] - s_r).max(), np.abs(s_st[s_r.size:]).max() if s_st.size > s_r.size else 0)
rep('companion spectrum: stacked == unstacked == ragged per element (padded tail 0)', worst < 1e-9, worst)
rep('companion centers: stacked == unstacked per element', all(np.allclose(np.asarray(sfd_u.centers[gi][jj])[c], np.asarray(sfd_c[c].centers[gi][jj])) for gi in range(2) for jj in range(2) for c in range(2)))

v_raw = ut3m.UNIFORM_COREWISE.randn(frame_u, stack_shape=K)   # K+C raw
def leaf_var(vu, k, c):
    return vu.unstack_tangents()[k].unstack_frame()[c].variations

# (a) tied gauge projection: stacked vs unstacked-uniform per (k,c), raw coordinates (same frame slices)
proj = utvo.utv_orthogonal_gauge_projection(frame_u.data, v_raw.variations.data, shared_data=sfd_u)
worst = 0.0
for k in range(3):
    for c in range(2):
        pl = utvo.utv_orthogonal_gauge_projection(frames_c[c].data, leaf_var(v_raw, k, c).data, shared_data=sfd_c[c])
        worst = max(worst, np.abs(proj[0][:, k, c] - pl[0]).max(), np.abs(proj[1][:, k, c] - pl[1]).max())
rep('(a) tied gauge projection K+C == unstacked uniform per (k,c)', worst < 1e-10, worst)
res = np.asarray(sharing.ufv_tied_variations_residual(proj, sfd_u)); rep('(a) tied residual shape K+C and == 0', res.shape == (3, 2) and res.max() < 1e-10, (res.shape, res.max()))
gr = np.asarray(utvo.utv_gauge_residual(frame_u.data, proj)); rep('(a) gauged', gr.max() < 1e-10, gr.max())
rep('(a) masks unchanged', all(np.array_equal(a, b) for a, b in zip(proj[3], v_raw.variations.masks.data)))
tu = ut3m.UT3Tangent(frame_u, ut3m._ut3variations_from_data(proj))
# ragged-oracle (dense): the ragged layer's own frame differs in gauge, so compare the TIED projection of the
# SAME ambient direction via project_ambient (gauge-invariant) below in (f); here also the 'post-pass == separate' identity:
sep = sharing.ufv_share_tucker_variations(utvo.utv_orthogonal_gauge_projection(frame_u.data, v_raw.variations.data), sfd_u)
rep('(a) shared_data= threading == gauge then separate post-pass (stacked)', np.allclose(sep[0], proj[0]) and np.allclose(sep[1], proj[1]))
rep('(a) post-pass idempotent (stacked)', np.allclose(sharing.ufv_share_tucker_variations(proj, sfd_u)[0], proj[0]))

# (b) tied retract stacked vs unstacked per (k,c) (dense) + masks + tied output
ret = utvo.utv_retract(frame_u.data, proj, shared_data=sfd_u); ru = ut3m._ut3_from_data(ret); rd = np.asarray(ru.to_dense())
worst = 0.0
for k in range(3):
    for c in range(2):
        rl = ut3m._ut3_from_data(utvo.utv_retract(frames_c[c].data, leaf_var(tu, k, c).data, shared_data=sfd_c[c]))
        worst = max(worst, np.linalg.norm(rd[k, c] - np.asarray(rl.to_dense())) / np.linalg.norm(rd[k, c]))
rep('(b) tied retract K+C == unstacked uniform per (k,c) (dense)', worst < 1e-10, worst)
def is_prefix(m): return bool(np.array_equal(m, np.arange(m.shape[-1]) < m.sum(axis=-1)[..., None]))
rep('(b) retract masks prefix', is_prefix(ret[3][0]) and is_prefix(ret[3][1]))
rep('(b) retract ranks == frame ranks bcast over K', np.array_equal(ret[3][0].sum(-1), np.broadcast_to(frame_u.masks.up_mask.sum(-1)[:, None], (4, 3, 2))) and np.array_equal(ret[3][1].sum(-1), np.broadcast_to(frame_u.masks.frame_left_mask.sum(-1)[:, None], (5, 3, 2))))
rep('(b) retracted points tied per (k,c)', bool(np.all(np.asarray(ru.has_shared_tucker_factors(spec)))))
# ragged oracle for retract is gauge-invariant given the same represented tangent; use (f)-style: ragged retract of
# the ragged projection of the same ambient tensor -- covered by test_sharing unstacked; skip.

# (c) tied embedding stacked vs unstacked
for shift in (False, True):
    emb = ut3m._ut3_from_data(utvo.utv_to_ut3(frame_u.data, proj, include_shift=shift, shared_data=sfd_u)); de = np.asarray(emb.to_dense())
    worst = 0.0
    for k in range(3):
        for c in range(2):
            el = ut3m._ut3_from_data(utvo.utv_to_ut3(frames_c[c].data, leaf_var(tu, k, c).data, include_shift=shift, shared_data=sfd_c[c]))
            worst = max(worst, np.linalg.norm(de[k, c] - np.asarray(el.to_dense())) / np.linalg.norm(de[k, c]))
    rep('(c) tied utv_to_ut3(shift=%s) K+C == unstacked per (k,c)' % shift, worst < 1e-10, worst)
    # and the tied embedding represents the SAME tangent as the untied embedding of the tied coordinates
    plain = np.asarray(ut3m._ut3_from_data(utvo.utv_to_ut3(frame_u.data, proj, include_shift=shift)).to_dense())
    rep('(c) tied embedding == plain embedding of tied coordinates (shift=%s)' % shift, np.allclose(de, plain, atol=1e-9), np.abs(de - plain).max())
    rep('(c) tied embedding masks prefix? (tucker)', is_prefix(emb.masks.tucker_edge_mask), emb.masks.tucker_edge_mask.sum(-1)[:, 0, 0].tolist())

# (d) corewise post-pass stacked vs unstacked
cf = ut3m.UNIFORM_COREWISE.frame(u); vc = ut3m.UNIFORM_COREWISE.randn(cf, stack_shape=K)
tc = sharing.ufv_share_tucker_variations_corewise(vc.variations.data, groups)
cfs = cf.unstack(); worst = 0.0
for k in range(3):
    for c in range(2):
        tl = sharing.ufv_share_tucker_variations_corewise(leaf_var(vc, k, c).data, groups)
        worst = max(worst, np.abs(tc[0][:, k, c] - tl[0]).max())
rep('(d) corewise tied post-pass K+C == unstacked per (k,c)', worst < 1e-12, worst)

# (e) garbage prongs
def corrupt_var(v, scale=1e3):
    scs = v.supercores; ind = ubv.UT3Variations(*[np.ones_like(s) for s in scs], v.shape, v.masks).apply_masks().supercores
    return ubv.UT3Variations(*[sc + scale * (1 - i) for sc, i in zip(scs, ind)], v.shape, v.masks)
def corrupt_frame(f, scale=1e3):
    scs = f.supercores; ind = ubv.UT3Frame(*[np.ones_like(s) for s in scs], f.shape, f.masks).apply_masks().supercores
    return ubv.UT3Frame(*[sc + scale * (1 - i) for sc, i in zip(scs, ind)], f.shape, f.masks)
pg = utvo.utv_orthogonal_gauge_projection(frame_u.data, corrupt_var(v_raw.variations).data, shared_data=sfd_u)
rep('(e) tied projection robust to VARIATION garbage', np.allclose(pg[0], proj[0], atol=1e-9) and np.allclose(pg[1], proj[1], atol=1e-9))
rg = utvo.utv_retract(frame_u.data, pg, shared_data=sfd_u)
rep('(e) tied retract robust to VARIATION garbage', np.allclose(np.asarray(ut3m._ut3_from_data(rg).to_dense()), rd, atol=1e-8))
rep('(e) tied residual robust to variation garbage', np.asarray(sharing.ufv_tied_variations_residual(corrupt_var(tu.variations).data, sfd_u)).max() < 1e-9)
rep('(e) corewise post-pass robust to variation garbage', np.allclose(sharing.ufv_share_tucker_variations_corewise(corrupt_var(vc.variations).data, groups)[0], tc[0]))
fg = corrupt_frame(frame_u); sfd_g = sharing.ufv_shared_frame_data(fg.data, groups)
print('    spectrum clean  :', np.round(np.asarray(sfd_u.svd_s[0])[0], 4)); print('    spectrum garbage:', np.round(np.asarray(sfd_g.svd_s[0])[0], 4))
pgf = utvo.utv_orthogonal_gauge_projection(fg.data, v_raw.variations.data, shared_data=sfd_g)
rep('(e) CHARACTERIZE: tied projection with FRAME padding garbage (companion as-stored) == clean', np.allclose(pgf[0], proj[0], atol=1e-8), np.abs(pgf[0] - proj[0]).max())
fm = frame_u.apply_masks(); sfd_m = sharing.ufv_shared_frame_data(fm.data, groups)
pm = utvo.utv_orthogonal_gauge_projection(fm.data, v_raw.variations.data, shared_data=sfd_m)
print('    spectrum zero-padded frame:', np.round(np.asarray(sfd_m.svd_s[0])[0], 4))
rep('(e) CHARACTERIZE: tied projection on a ZERO-padded (apply_masks) frame == clean', np.allclose(pm[0], proj[0], atol=1e-8), np.abs(pm[0] - proj[0]).max())
# the ragged companion built on the to_t3frame() leaves (the sliced cores) -- does its tied projection agree?
fr_leaves = frame_u.to_t3frame(); bad = 0.0
for c in range(2):
    sfr = sharing.fv_shared_frame_data(fr_leaves[c].data, groups)
    for k in range(3):
        pr = tvo.tv_orthogonal_gauge_projection(fr_leaves[c].data, leaf_var(v_raw, k, c).to_t3variations().data, shared_data=sfr)
        pu_ = ut3m._ut3variations_from_data((proj[0][:, k, c], proj[1][:, k, c], proj[2], tuple(m[:, k, c] for m in proj[3]))).to_t3variations()
        bad = max(bad, max(np.abs(np.asarray(a) - np.asarray(b)).max() for a, b in zip(pr[0], pu_.tucker_variations)))
rep('(e) CHARACTERIZE: ragged companion on UT3Frame.to_t3frame() leaves gives the same tied projection', bad < 1e-9, bad)

# (f) frontend: shared project_ambient / transport untie?
G = sg.shared_manifold(spec)
z = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, n, r, stack_shape=C), n=u.tucker_supercore.shape[-2], r=u.tt_supercore.shape[-1])
pa = G.project_ambient(frame_u, z)
rep('(f) frontend SharedGeometry.project_ambient (uniform) is TIED', np.asarray(sharing.ufv_tied_variations_residual(pa.variations.data, sfd_u)).max() < 1e-10)
zr = z.to_t3(); worst = 0.0
for c in range(2):
    fr_r, _ = fvc.t3_orthogonal_representations(xs[c].data); sfr = sharing.fv_shared_frame_data(fr_r, groups)
    vr = tvo.tv_project_t3_onto_tangent_space(fr_r, zr[c].data, shared_data=sfr)
    dr = np.asarray(t3.TuckerTensorTrain(*tvo.tv_to_t3(fr_r, vr)).to_dense())
    worst = max(worst, np.linalg.norm(np.asarray(pa.to_dense())[c] - dr) / np.linalg.norm(dr))
rep('(f) frontend shared project_ambient (C=(2,)) == ragged shared per element (dense, own frames)', worst < 1e-9, worst)
raw_b = utvo.utv_project_ut3_onto_tangent_space(frame_u.data, z.data)
rep('(f) CHARACTERIZE backlog: backend utv_project_ut3_onto_tangent_space alone is UNTIED', np.asarray(sharing.ufv_tied_variations_residual(raw_b, sfd_u)).max() > 1e-3, np.asarray(sharing.ufv_tied_variations_residual(raw_b, sfd_u)).max())
x2 = t3.TuckerTensorTrain.stack(tuple(tied(shape, n, r) for _ in range(2))); frame2 = G.frame(ut3.UniformTuckerTensorTrain.from_t3(x2))
try:
    tr = G.transport(tu, frame2)
    rep('(f) frontend SharedGeometry.transport(K+C tangent) TIED', np.asarray(sharing.ufv_tied_variations_residual(tr.variations.data, sharing.ufv_shared_frame_data(frame2.data, groups))).max() < 1e-10, tr.stack_shape)
except Exception as e:
    rep('(f) frontend SharedGeometry.transport(K+C tangent) raises', False, '%s: %s' % (type(e).__name__, str(e)[:80]))
trC = G.transport(ut3m.UT3Tangent(frame_u, ut3m._ut3variations_from_data(tuple(p[:, 0] for p in proj[:2]) + (proj[2], tuple(m[:, 0] for m in proj[3])))), frame2)
rep('(f) frontend SharedGeometry.transport(C tangent) TIED', np.asarray(sharing.ufv_tied_variations_residual(trC.variations.data, sharing.ufv_shared_frame_data(frame2.data, groups))).max() < 1e-10, trC.stack_shape)
rr = G.retract(tu); rep('(f) frontend SharedGeometry.retract (K+C) == backend', np.allclose(np.asarray(rr.to_dense()), rd))
rp = G.project(v_raw); rep('(f) frontend SharedGeometry.project (K+C) == backend', np.allclose(rp.variations.tucker_variations, proj[0]))
