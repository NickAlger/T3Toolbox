"""R9-F: (1) shared frontend on the UNIFORM base at C=(2,)/K=(3,): project_ambient / transport tied?
(2) the cross-layer route UT3Frame.to_t3frame() -> ragged shared companion: is the tied projection right?
(3) ragged-based shared geometry handed a UT3Frame: error quality. (4) has_shared_tucker_factors on
unequal group ranks: raise vs False, both layers."""
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
import t3toolbox.backend.ranks as ranks

def rep(msg, ok, val=None):
    print(('PASS ' if ok else 'FAIL ') + msg + ('' if val is None else '  [%s]' % (val,)))

np.random.seed(0)
shape, n, r, spec = (5, 6, 5, 6), (2, 3, 2, 3), (1, 2, 3, 2, 1), ('a', 'b', 'a', 'b')
C, K = (2,), (3,)
groups = sharing.validate_sharing(spec, shape)
def tied(shape, n, r):
    x = t3.TuckerTensorTrain.randn(shape, n, r); tk = list(x.tucker_cores)
    for g in groups:
        for ii in g[1:]: tk[ii] = tk[g[0]]
    return t3.TuckerTensorTrain(tuple(tk), x.tt_cores)
xs = [tied(shape, n, r) for _ in range(2)]; x = t3.TuckerTensorTrain.stack(tuple(xs))
u = ut3.UniformTuckerTensorTrain.from_t3(x)
GU = sg.shared(ut3m.UNIFORM_MANIFOLD, spec); GR = sg.shared_manifold(spec)
print('GU.is_uniform', GU.is_uniform, ' GR.is_uniform', GR.is_uniform)
frame_u = GU.frame(u); sfd_u = GU.shared_frame_data(frame_u)
zs = [t3.TuckerTensorTrain.randn(shape, n, r) for _ in range(2)]
z = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.stack(tuple(zs)), n=u.tucker_supercore.shape[-2], r=u.tt_supercore.shape[-1])

# (1) frontend uniform shared project_ambient / transport
pa = GU.project_ambient(frame_u, z)
rep('(1) GU.project_ambient(C frame, C grad) TIED', np.asarray(sharing.ufv_tied_variations_residual(pa.variations.data, sfd_u)).max() < 1e-10)
pad = np.asarray(pa.to_dense()); worst = 0.0; ragged_dense = []
for c in range(2):
    fr = GR.frame(xs[c]); pr = GR.project_ambient(fr, zs[c]); dr = np.asarray(pr.to_dense()); ragged_dense.append(dr)
    worst = max(worst, np.linalg.norm(pad[c] - dr) / np.linalg.norm(dr))
rep('(1) GU.project_ambient == GR.project_ambient per element (dense)', worst < 1e-9, worst)
v = GU.randn(frame_u, stack_shape=K)
rep('(1) GU.randn(K) tied + gauged', np.asarray(sharing.ufv_tied_variations_residual(v.variations.data, sfd_u)).max() < 1e-10 and np.asarray(v.gauge_residual).max() < 1e-10)
x2 = t3.TuckerTensorTrain.stack(tuple(tied(shape, n, r) for _ in range(2))); frame2 = GU.frame(ut3.UniformTuckerTensorTrain.from_t3(x2))
try:
    tr = GU.transport(v, frame2); rep('(1) GU.transport(K+C tangent) works', True, tr.stack_shape)
except Exception as e:
    rep('(1) GU.transport(K+C tangent) raises (same K+C limitation as UNIFORM_MANIFOLD.transport)', False, '%s: %s' % (type(e).__name__, str(e)[:80]))
vC = ut3m.UT3Tangent(frame_u, ut3m._ut3variations_from_data((v.variations.tucker_variations[:, 0], v.variations.tt_variations[:, 0], v.shape, tuple(m[:, 0] for m in v.variations.masks.data))))
trC = GU.transport(vC, frame2)
rep('(1) GU.transport(C tangent) TIED at the new frame', np.asarray(sharing.ufv_tied_variations_residual(trC.variations.data, GU.shared_frame_data(frame2))).max() < 1e-10)
rr = GU.retract(v); rep('(1) GU.retract(K+C) tied per element', bool(np.all(np.asarray(rr.has_shared_tucker_factors(spec)))))

# (2) cross-layer route: UT3Frame.to_t3frame() leaves + RAGGED shared companion (the documented route for dense grads)
fr_leaves = frame_u.to_t3frame()
for c in range(2):
    fl = fr_leaves[c]
    sfd_leaf = GR.shared_frame_data(fl)
    s_leaf = np.asarray(sfd_leaf.svd_s[0]); s_true = np.asarray(sfd_u.svd_s[0])[c]
    rep('(2) c=%d ragged companion on to_t3frame() leaf: spectrum == true' % c, np.allclose(s_leaf, s_true[:s_leaf.size], rtol=1e-9), (np.round(s_leaf, 3), np.round(s_true, 3)))
    pl = GR.project_ambient(fl, zs[c])          # the cross-layer route for a shared ambient projection
    dl = np.asarray(pl.to_dense())
    rep('(2) c=%d GR.project_ambient(to_t3frame leaf, z) == true tied projection (dense)' % c,
        np.linalg.norm(dl - ragged_dense[c]) / np.linalg.norm(ragged_dense[c]) < 1e-9,
        np.linalg.norm(dl - ragged_dense[c]) / np.linalg.norm(ragged_dense[c]))
    # is the leaf frame tied (a precondition) and orthogonal?
    rep('(2) c=%d leaf frame tied + orthogonal' % c, bool(fl.is_orthogonal()) and bool(t3.TuckerTensorTrain(fl.up_tucker_cores, fl.left_tt_cores).has_shared_tucker_factors(spec)))
    # dense projection through the pure-ragged own frame (already 'ragged_dense'); and the raw unshared projection, for scale
    raw = np.asarray(t3m.MANIFOLD.project_ambient(fl, zs[c]).to_dense())
    print('    |tied - unshared| / |unshared| =', np.linalg.norm(ragged_dense[c] - raw) / np.linalg.norm(raw))
    # is the leaf result at least tied w.r.t. ITS companion (self-consistent) -- i.e. a projection onto SOME subspace?
    rep('(2) c=%d leaf result tied w.r.t. the leaf companion' % c, float(np.asarray(sharing.fv_tied_variations_residual(pl.variations.data, sfd_leaf)).max()) < 1e-10)
    # and dense: does the leaf result lie in the tied tangent space? check via the true ragged projection being idempotent on it
    frx = GR.frame(xs[c]); reproj = np.asarray(GR.project_ambient(frx, t3.TuckerTensorTrain(*tvo.tv_to_t3(fl.data, pl.variations.data))).to_dense())
    rep('(2) c=%d leaf result lies in the TRUE tied tangent space (re-projection fixes it)' % c, np.linalg.norm(reproj - dl) / np.linalg.norm(dl) < 1e-9, np.linalg.norm(reproj - dl) / np.linalg.norm(dl))

# (3) ragged-based shared geometry handed a UT3Frame -> error quality
try:
    GR.project_ambient(frame_u, z); rep('(3) GR(ragged base).project_ambient(UT3Frame) raises', False)
except Exception as e:
    rep('(3) CHARACTERIZE: GR(ragged base).project_ambient(UT3Frame, UT3) error', True, '%s: %s' % (type(e).__name__, str(e)[:70]))
try:
    t3m.MANIFOLD.project_ambient(frame_u, z); rep('(3) MANIFOLD.project_ambient(UT3Frame) raises', False)
except Exception as e:
    rep('(3) CHARACTERIZE: plain MANIFOLD.project_ambient(UT3Frame, UT3) error', True, '%s: %s' % (type(e).__name__, str(e)[:70]))

# (4) has_shared_tucker_factors with unequal group ranks: raise vs False
xu = t3.TuckerTensorTrain.randn((6, 6, 4), (2, 4, 2), (1, 2, 2, 1))
for name, obj in (('ragged', xu), ('uniform', ut3.UniformTuckerTensorTrain.from_t3(xu))):
    try:
        rep('(4) %s has_shared_tucker_factors((0,0,1)) on unequal group ranks returns' % name, True, obj.has_shared_tucker_factors((0, 0, 1)))
    except ValueError as e:
        rep('(4) %s has_shared_tucker_factors on unequal group ranks RAISES' % name, True, str(e)[:70])
