"""R9-A: the uniform TIED-tangent (shared) ops with C=(2,) and K=(3,), per element vs ragged, plus
garbage robustness, plus: does shared project_ambient / transport (frontend) silently untie?"""
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
import t3toolbox.shared_geometry as sg
import t3toolbox.safety as safety

np.random.seed(0)
shape, n, r, spec = (5, 6, 5, 6), (2, 3, 2, 3), (1, 2, 3, 2, 1), ('a', 'b', 'a', 'b')
C, K = (2,), (3,)
groups = sharing.validate_sharing(spec, shape)

def tied_stack(shape, n, r, spec, C):
    leaves = []
    for _ in range(int(np.prod(C))):
        x = t3.TuckerTensorTrain.randn(shape, n, r)
        tk = list(x.tucker_cores)
        for g in groups:
            for ii in g[1:]:
                tk[ii] = tk[g[0]]
        leaves.append(t3.TuckerTensorTrain(tuple(tk), x.tt_cores))
    return t3.TuckerTensorTrain.stack(tuple(leaves))

x = tied_stack(shape, n, r, spec, C)
u = ut3.UniformTuckerTensorTrain.from_t3(x)
frame_u, _ = ubv.ut3_orthogonal_representations(u)
sfd_u = sharing.ufv_shared_frame_data(frame_u.data, groups)
assert frame_u.stack_shape == C

# K-stacked raw variations at the uniform frame
v_raw = ut3m.UNIFORM_COREWISE.randn(frame_u, stack_shape=K)       # stack K+C
assert v_raw.stack_shape == K + C

# ragged leaves of the SAME frame (same gauge: to_t3frame slices the masked content)
frames_r = frame_u.to_t3frame()                # tuple over C of T3Frame
sfd_r = [sharing.fv_shared_frame_data(fr.data, groups) for fr in frames_r]

def ragged_leaves(vu):   # vu: UT3Tangent stack K+C -> dict (k,c) -> T3Variations
    out = {}
    ktree = vu.unstack_tangents()
    for k, tk_ in enumerate(ktree):
        ctree = tk_.unstack_frame()
        for c, leaf in enumerate(ctree):
            out[(k, c)] = leaf.variations.to_t3variations()
    return out

def rep(msg, ok, val=None):
    print(('PASS ' if ok else 'FAIL ') + msg + ('' if val is None else '  [%s]' % (val,)))

# ---- (a) tied orthogonal gauge projection, stacked, vs per-element ragged ---------------------------
proj_u = utvo.utv_orthogonal_gauge_projection(frame_u.data, v_raw.variations.data, shared_data=sfd_u)
tu = ut3m.UT3Tangent(frame_u, ut3m._ut3variations_from_data(proj_u))
dense_u = np.asarray(tu.to_dense())                       # (K+C)+shape
raw_leaves = ragged_leaves(v_raw)
worst = 0.0
for (k, c), vr in raw_leaves.items():
    pr = tvo.tv_orthogonal_gauge_projection(frames_r[c].data, vr.data, shared_data=sfd_r[c])
    dr = np.asarray(t3m.T3Tangent(frames_r[c], bvf.T3Variations(*pr)).to_dense())
    worst = max(worst, np.linalg.norm(dense_u[k, c] - dr) / np.linalg.norm(dr))
rep('(a) tied gauge projection stacked K+C == per-element ragged (dense)', worst < 1e-10, worst)
res = np.asarray(sharing.ufv_tied_variations_residual(proj_u, sfd_u))
rep('(a) tied residual of projection == 0 on every (k,c)', res.shape == K + C and res.max() < 1e-10, (res.shape, res.max()))
res_raw = np.asarray(sharing.ufv_tied_variations_residual(v_raw.variations.data, sfd_u))
rep('(a) raw residual nonzero (sanity)', res_raw.min() > 1e-3, res_raw.min())
rep('(a) output masks unchanged', all(np.array_equal(a, b) for a, b in zip(proj_u[3], v_raw.variations.masks.data)))
gr = np.asarray(utvo.utv_gauge_residual(frame_u.data, proj_u))
rep('(a) tied projection is gauged', gr.max() < 1e-10, gr.max())

# ---- (b) tied retraction stacked vs per-element ragged; tied output; prefix masks -------------------
ret_u = utvo.utv_retract(frame_u.data, proj_u, shared_data=sfd_u)
ru = ut3m._ut3_from_data(ret_u)
rd = np.asarray(ru.to_dense())
tied_leaves = ragged_leaves(tu)
worst = 0.0
for (k, c), vr in tied_leaves.items():
    cores = tvo.tv_retract(frames_r[c].data, vr.data, shared_data=sfd_r[c])
    dr = np.asarray(t3.TuckerTensorTrain(*cores).to_dense())
    worst = max(worst, np.linalg.norm(rd[k, c] - dr) / np.linalg.norm(dr))
rep('(b) tied retract stacked K+C == per-element ragged (dense)', worst < 1e-9, worst)
def is_prefix(m): return bool(np.array_equal(m, np.arange(m.shape[-1]) < m.sum(axis=-1)[..., None]))
rep('(b) retract output masks are prefix', is_prefix(ret_u[3][0]) and is_prefix(ret_u[3][1]))
rep('(b) retract output ranks == frame ranks (bcast over K)',
    np.array_equal(ret_u[3][0].sum(-1), np.broadcast_to(frame_u.masks.up_mask.sum(-1)[:, None], (len(shape),) + K + C)) and
    np.array_equal(ret_u[3][1].sum(-1), np.broadcast_to(frame_u.masks.frame_left_mask.sum(-1)[:, None], (len(shape)+1,) + K + C)))
shared_ok = np.asarray(ru.has_shared_tucker_factors(spec))
rep('(b) retracted point has tied Tucker factors per element', bool(np.all(shared_ok)), shared_ok)

# ---- (c) tied doubled embedding stacked vs ragged -------------------------------------------------------
for shift in (False, True):
    emb = utvo.utv_to_ut3(frame_u.data, proj_u, include_shift=shift, shared_data=sfd_u)
    de = np.asarray(ut3m._ut3_from_data(emb).to_dense())
    worst = 0.0
    for (k, c), vr in tied_leaves.items():
        dr = np.asarray(t3.TuckerTensorTrain(*tvo.tv_to_t3(frames_r[c].data, vr.data, include_shift=shift, shared_data=sfd_r[c])).to_dense())
        worst = max(worst, np.linalg.norm(de[k, c] - dr) / np.linalg.norm(dr))
    rep('(c) tied utv_to_ut3(include_shift=%s) stacked == ragged' % shift, worst < 1e-10, worst)

# ---- (d) corewise tied post-pass stacked vs ragged ------------------------------------------------------
cf_u = ut3m.UNIFORM_COREWISE.frame(u)
vc = ut3m.UNIFORM_COREWISE.randn(cf_u, stack_shape=K)
tied_c = sharing.ufv_share_tucker_variations_corewise(vc.variations.data, groups)
cf_r = cf_u.to_t3frame()
leaves_c = ragged_leaves(vc)
worst = 0.0
for (k, c), vr in leaves_c.items():
    tr = sharing.fv_share_tucker_variations_corewise(vr.data, groups)
    uv = ut3m._ut3variations_from_data(tied_c).to_t3variations()[k][c]
    for a, b in zip(uv.tucker_variations, tr[0]):
        worst = max(worst, np.linalg.norm(np.asarray(a) - np.asarray(b)))
rep('(d) corewise tied post-pass stacked == ragged', worst < 1e-12, worst)

# ---- (e) garbage robustness ---------------------------------------------------------------------------
def corrupt_var(v, scale=1e3):
    scs = v.supercores
    ind = ubv.UT3Variations(*[np.ones_like(s) for s in scs], v.shape, v.masks).apply_masks().supercores
    return ubv.UT3Variations(*[sc + scale * (1 - i) for sc, i in zip(scs, ind)], v.shape, v.masks)
def corrupt_frame(f, scale=1e3):
    scs = f.supercores
    ind = ubv.UT3Frame(*[np.ones_like(s) for s in scs], f.shape, f.masks).apply_masks().supercores
    return ubv.UT3Frame(*[sc + scale * (1 - i) for sc, i in zip(scs, ind)], f.shape, f.masks)
vg = corrupt_var(v_raw.variations)
pg = utvo.utv_orthogonal_gauge_projection(frame_u.data, vg.data, shared_data=sfd_u)
dg = np.asarray(ut3m.UT3Tangent(frame_u, ut3m._ut3variations_from_data(pg)).to_dense())
rep('(e) tied projection robust to VARIATION padding garbage', np.allclose(dg, dense_u, atol=1e-9), np.abs(dg - dense_u).max())
rg = utvo.utv_retract(frame_u.data, pg, shared_data=sfd_u)
rep('(e) tied retract robust to VARIATION padding garbage', np.allclose(np.asarray(ut3m._ut3_from_data(rg).to_dense()), rd, atol=1e-8))
resg = np.asarray(sharing.ufv_tied_variations_residual(corrupt_var(tu.variations).data, sfd_u))
rep('(e) tied residual robust to variation garbage', resg.max() < 1e-9, resg.max())
# frame padding garbage: the companion is documented "as stored, NOT re-masked" -> characterize
fg = corrupt_frame(frame_u)
try:
    sfd_g = sharing.ufv_shared_frame_data(fg.data, groups)
    s_clean = np.asarray(sfd_u.svd_s[0]); s_g = np.asarray(sfd_g.svd_s[0])
    print('    companion spectrum clean frame  :', np.round(s_clean[0], 4))
    print('    companion spectrum garbage frame:', np.round(s_g[0], 4))
    pgf = utvo.utv_orthogonal_gauge_projection(fg.data, v_raw.variations.data, shared_data=sfd_g)
    dgf = np.asarray(ut3m.UT3Tangent(fg, ut3m._ut3variations_from_data(pgf)).to_dense())
    rep('(e) CHARACTERIZE: tied projection with FRAME padding garbage (companion as-stored) equals clean',
        np.allclose(dgf, dense_u, atol=1e-8), np.abs(dgf - dense_u).max())
    # contrast: the unshared gauge projection IS frame-garbage robust (mask-once)
    pgu = utvo.utv_orthogonal_gauge_projection(fg.data, v_raw.variations.data)
    pcu = utvo.utv_orthogonal_gauge_projection(frame_u.data, v_raw.variations.data)
    rep('(e) contrast: UNSHARED gauge projection robust to frame garbage', np.allclose(pgu[0], pcu[0]) and np.allclose(pgu[1], pcu[1]))
    # a masked (zero-padded) frame: the doc says re-masking breaks the pairing -> check
    fm = frame_u.apply_masks()
    sfd_m = sharing.ufv_shared_frame_data(fm.data, groups)
    pm = utvo.utv_orthogonal_gauge_projection(fm.data, v_raw.variations.data, shared_data=sfd_m)
    dm = np.asarray(ut3m.UT3Tangent(fm, ut3m._ut3variations_from_data(pm)).to_dense())
    rep('(e) CHARACTERIZE: tied projection on a ZERO-padded (apply_masks) frame equals clean', np.allclose(dm, dense_u, atol=1e-8), np.abs(dm - dense_u).max())
    print('    zero-padded-frame companion spectrum:', np.round(np.asarray(sfd_m.svd_s[0])[0], 4))
except Exception:
    traceback.print_exc()

# ---- (f) frontend SharedGeometry on the uniform base: project_ambient / transport untie? -------------
G = sg.shared_manifold(spec)
z = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, n, r, stack_shape=C), n=u.tucker_supercore.shape[-2], r=u.tt_supercore.shape[-1])
pa = G.project_ambient(frame_u, z)                       # frontend shared uniform
res_pa = np.asarray(sharing.ufv_tied_variations_residual(pa.variations.data, sfd_u))
rep('(f) frontend SharedGeometry.project_ambient (uniform) output is TIED', res_pa.max() < 1e-10, res_pa.max())
# vs ragged shared project_ambient per element (dense)
pa_d = np.asarray(pa.to_dense())
worst = 0.0
zr = z.to_t3()
for c in range(C[0]):
    vr = tvo.tv_project_t3_onto_tangent_space(frames_r[c].data, zr[c].data, shared_data=sfd_r[c])
    dr = np.asarray(t3m.T3Tangent(frames_r[c], bvf.T3Variations(*vr)).to_dense())
    worst = max(worst, np.linalg.norm(pa_d[c] - dr) / np.linalg.norm(dr))
rep('(f) frontend shared project_ambient == ragged shared per element (dense)', worst < 1e-9, worst)
# the BACKEND utv_project_ut3_onto_tangent_space alone (no shared_data=) -> untied (the backlog item)
raw_b = utvo.utv_project_ut3_onto_tangent_space(frame_u.data, z.data)
res_b = np.asarray(sharing.ufv_tied_variations_residual(raw_b, sfd_u))
rep('(f) CHARACTERIZE backlog: backend utv_project_ut3_onto_tangent_space alone is UNTIED (expected)', res_b.max() > 1e-3, res_b.max())
# transport: tied?
x2 = tied_stack(shape, n, r, spec, C)
frame2 = G.frame(ut3.UniformTuckerTensorTrain.from_t3(x2))
tr = G.transport(tu, frame2)
sfd2 = sharing.ufv_shared_frame_data(frame2.data, groups)
res_tr = np.asarray(sharing.ufv_tied_variations_residual(tr.variations.data, sfd2))
rep('(f) frontend SharedGeometry.transport (uniform) output is TIED (K+C stack)', res_tr.shape == K + C and res_tr.max() < 1e-10, (res_tr.shape, res_tr.max()))
# frontend retract at K+C
rr = G.retract(tu)
rep('(f) frontend SharedGeometry.retract (uniform, K+C) == backend', np.allclose(np.asarray(rr.to_dense()), rd))
