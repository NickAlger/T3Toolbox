"""R9-B: uniform_fitting -- the six Uniform*Kind (adjoint identity per kind, forward vs the frontend
UT3Tangent op and vs ragged, point_forward vs ragged, take consistency), pack_sample/pack_data round trip,
uniform_minimal(sharing=), the shared-minimal gate, and two ergonomic probes."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.manifold as t3m
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.ut3_operations as ut3o
import t3toolbox.backend.utv_operations as utvo
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.geometry as geom
import t3toolbox.backend.sharing as sharing

def rep(msg, ok, val=None):
    print(('PASS ' if ok else 'FAIL ') + msg + ('' if val is None else '  [%s]' % (val,)))

np.random.seed(0)
shape, n, r = (5, 6, 4, 3), (2, 3, 2, 2), (1, 2, 3, 2, 1)
assert ranks.compute_minimal_ranks(shape, n, r) == (n, r), ranks.compute_minimal_ranks(shape, n, r)
W = (7,)
d = len(shape)
order = 2

def make_sample(name, C):
    ww = [np.random.randn(*(W + (Ni,))) for Ni in shape]
    pp = [np.random.randn(*(W + (Ni,))) for Ni in shape]
    index = np.stack([np.random.randint(0, Ni, size=W) for Ni in shape])
    if name == 'apply': return ww
    if name == 'probe': return ww
    if name == 'entries': return index
    if name == 'apply_derivatives': return (ww, pp)
    if name == 'probe_derivatives': return (ww, pp)
    if name == 'entries_derivatives': return (index, pp)

def frontend_forward(name, v, sample):
    if name == 'apply': return v.apply(sample)
    if name == 'probe': return v.probe(sample)
    if name == 'entries': return v.entries(sample)
    if name == 'apply_derivatives': return v.apply_derivatives(sample[0], sample[1], order)
    if name == 'probe_derivatives': return v.probe_derivatives(sample[0], sample[1], order)
    if name == 'entries_derivatives': return v.entries_derivatives(sample[0], sample[1], order)

def point_forward_ragged(name, x, sample):
    if name == 'apply': return x.apply(sample)
    if name == 'probe': return x.probe(sample)
    if name == 'entries': return x.entries(sample)
    if name == 'apply_derivatives': return x.apply_derivatives(sample[0], sample[1], order)
    if name == 'probe_derivatives': return x.probe_derivatives(sample[0], sample[1], order)
    if name == 'entries_derivatives': return x.entries_derivatives(sample[0], sample[1], order)

names = ['apply', 'entries', 'probe', 'apply_derivatives', 'entries_derivatives', 'probe_derivatives']
for C in [(), (2,)]:
    for gname in ['manifold', 'corewise']:
        for name in names:
            try:
                np.random.seed(1)
                x = t3.TuckerTensorTrain.randn(shape, n, r, stack_shape=C)
                u = ut3.UniformTuckerTensorTrain.from_t3(x)
                sample = make_sample(name, C)
                N = u.tucker_supercore.shape[-1]
                # data of the right shape: use point_forward of a random other point
                x_sc = (u.data[0], u.data[1])
                prob = uf.uniform_least_squares_problem(gname, name, u, sample, None, order=order)
                kind, G = prob.kind, prob.geom
                psample = prob.sample
                data = kind.point_forward(x_sc, psample)
                frame = G.frame(x_sc)
                sweep = kind.precompute(frame, psample)
                # a random masked variation (bare supercore pair) at the frame
                vm = G.var_masks
                vshape_tk = frame[0].shape[:1] + C + (frame[1].shape[-2], N)
                vshape_tt = frame[0].shape[:1] + C + (frame[2].shape[-1], frame[0].shape[-2], frame[3].shape[-1])
                v_sc = (np.random.randn(*vshape_tk), np.random.randn(*vshape_tt))
                from t3toolbox.backend import ufv_masking
                v_sc = ufv_masking.ufv_apply_variations_masks((v_sc[0], v_sc[1], tuple(shape), vm))
                out = kind.forward(v_sc, psample, frame, sweep)
                rr = np.random.randn(*np.asarray(out).shape)
                JTr = kind.transpose(rr, psample, frame, sweep)
                lhs = float(np.sum(rr * np.asarray(out)))
                rhs = float(np.sum(np.asarray(G.inner(JTr, v_sc))))   # sums the C stack too
                rep('C=%s %s %-20s adjoint identity' % (C, gname, name), abs(lhs - rhs) < 1e-8 * max(1, abs(lhs)), (lhs, rhs))
                # forward vs the frontend UT3Tangent op on the same (frame, variations)
                vt = ut3m.UT3Tangent(ut3m._ut3frame_from_data(frame), ut3m._ut3variations_from_data(G._variations(v_sc)))
                ff = frontend_forward(name, vt, sample)
                if name in ('probe', 'probe_derivatives'):
                    ff = ut3o.pack_vectors(ff, N)
                rep('C=%s %s %-20s kind.forward == frontend op' % (C, gname, name), np.allclose(np.asarray(out), np.asarray(ff), atol=1e-10))
                # point_forward vs ragged
                pf = kind.point_forward(x_sc, psample)
                pr = point_forward_ragged(name, x, sample)
                if name in ('probe', 'probe_derivatives'):
                    pr = ut3o.pack_vectors(pr, N)
                rep('C=%s %s %-20s point_forward == ragged' % (C, gname, name), np.allclose(np.asarray(pf), np.asarray(pr), atol=1e-10))
                # take: forward on a minibatch == gather of the full forward
                idx = np.array([5, 1, 3])
                sub_sample, sub_data = kind.take(psample, data, idx)
                sweep_sub = kind.precompute(frame, sub_sample)
                out_sub = kind.forward(v_sc, sub_sample, frame, sweep_sub)
                waxis = kind.w_axes(psample)
                full = np.asarray(out)
                if name in ('probe',):
                    gathered = full[:, idx]
                elif name == 'probe_derivatives':
                    gathered = full[:, :, idx]
                elif name in ('apply_derivatives', 'entries_derivatives'):
                    gathered = full[:, idx]
                else:
                    gathered = full[idx]
                rep('C=%s %s %-20s take -> forward == gather(forward)' % (C, gname, name), np.allclose(np.asarray(out_sub), gathered, atol=1e-10))
                dfull = np.asarray(data)
                if name in ('probe',): gd = dfull[:, idx]
                elif name == 'probe_derivatives': gd = dfull[:, :, idx]
                elif name in ('apply_derivatives', 'entries_derivatives'): gd = dfull[:, idx]
                else: gd = dfull[idx]
                rep('C=%s %s %-20s take(data) == gather(data)' % (C, gname, name), np.allclose(np.asarray(sub_data), gd))
            except Exception as e:
                print('ERROR C=%s %s %s: %r' % (C, gname, name, e)); traceback.print_exc()

# ---- pack_sample / pack_data round trip --------------------------------------------------------------
np.random.seed(2)
ww = [np.random.randn(*(W + (Ni,))) for Ni in shape]; pp = [np.random.randn(*(W + (Ni,))) for Ni in shape]
N = max(shape)
for name in names:
    s = make_sample(name, ())
    ps = uf.pack_sample(name, s, N)
    ps2 = uf.pack_sample(name, ps, N)   # mirror-tolerant: packing twice is a no-op
    def eq(a, b):
        if isinstance(a, tuple): return all(eq(x, y) for x, y in zip(a, b))
        return np.array_equal(np.asarray(a), np.asarray(b))
    rep('pack_sample(%s) idempotent on packed input' % name, eq(ps, ps2))
    # unpack recovers ragged
    if name in ('apply', 'probe'):
        rep('pack_sample(%s) round trip' % name, all(np.array_equal(a, b) for a, b in zip(ut3o.unpack_vectors(ps, shape), s)))
pd_ = uf.pack_data('probe', [np.random.randn(*(W + (Ni,))) for Ni in shape], N)
rep('pack_data(probe) shape (d,)+W+(N,)', pd_.shape == (d,) + W + (N,), pd_.shape)
rep('pack_data(apply) passthrough', uf.pack_data('apply', 3.0, N) == 3.0)
try:
    uf.pack_sample('bogus', ww, N); rep('pack_sample bogus raises', False)
except ValueError as e:
    rep('pack_sample bogus raises ValueError', True)

# ---- uniform_minimal(sharing=) and the gate ------------------------------------------------------------
np.random.seed(3)
shape2, n2, r2, spec2 = (6, 6, 4), (4, 4, 2), (1, 2, 2, 1), (0, 0, 1)   # shared-minimal; per-mode minimal is (2,4,2)
print('per-mode minimal:', ranks.compute_minimal_ranks(shape2, n2, r2), ' shared-minimal:', ranks.compute_minimal_ranks(shape2, n2, r2, sharing=spec2))
xs = t3.TuckerTensorTrain.randn(shape2, n2, r2).share(spec2)
us = ut3.UniformTuckerTensorTrain.from_t3(xs)
rep('start is tied', bool(us.has_shared_tucker_factors(spec2)))
um = uf.uniform_minimal(us, sharing=spec2)
rep('uniform_minimal(sharing) returns x0 itself when shared-minimal', um is us)
um0 = uf.uniform_minimal(us)   # per-mode: clips the group rank -> unties
rep('CHARACTERIZE uniform_minimal(no sharing) on a shared start changes ranks', tuple(um0.tucker_ranks) != tuple(us.tucker_ranks), (tuple(um0.tucker_ranks), tuple(us.tucker_ranks)))
rep('  ... and the same tensor', np.allclose(np.asarray(um0.to_dense()), np.asarray(us.to_dense())))
try:
    rep('  ... and is now UNTIED', not bool(um0.has_shared_tucker_factors(spec2)))
except ValueError as e:
    rep('  ... has_shared_tucker_factors RAISES on the untied (unequal-rank) result', True, str(e)[:80])
# a non-shared-minimal shared start: nominal TT rank 3 unrealizable
x_nm = t3.TuckerTensorTrain.randn((6, 6, 4), (2, 2, 2), (1, 3, 3, 1)).share((0, 0, 1))
u_nm = ut3.UniformTuckerTensorTrain.from_t3(x_nm)
sampleA = [np.random.randn(5, Ni) for Ni in (6, 6, 4)]
try:
    uf.uniform_least_squares_problem('manifold', 'apply', u_nm, sampleA, np.random.randn(5), sharing=(0, 0, 1))
    rep('gate: non-minimal shared start raises', False)
except ValueError as e:
    rep('gate: non-minimal shared start raises ValueError', True, str(e)[-60:])
u_nm_min = uf.uniform_minimal(u_nm, sharing=(0, 0, 1))
rep('uniform_minimal(sharing) -> same tensor', np.allclose(np.asarray(u_nm_min.to_dense()), np.asarray(u_nm.to_dense())))
rep('uniform_minimal(sharing) -> tied', bool(u_nm_min.has_shared_tucker_factors((0, 0, 1))))
rep('uniform_minimal(sharing) -> shared-minimal ranks', (tuple(u_nm_min.tucker_ranks), tuple(u_nm_min.tt_ranks)) == tuple(tuple(int(v) for v in q) for q in ranks.compute_minimal_ranks((6,6,4), (2,2,2), (1,3,3,1), sharing=(0,0,1))), (tuple(u_nm_min.tucker_ranks), tuple(u_nm_min.tt_ranks)))
prob = uf.uniform_least_squares_problem('manifold', 'apply', u_nm_min, sampleA, np.random.randn(5), sharing=(0, 0, 1))
rep('gate passes after uniform_minimal(sharing)', prob.geom.groups == ((0, 1), (2,)), prob.geom.groups)
# gate with sharing given but a start that is per-mode-minimal yet NOT shared-minimal? (shared minimal <= per-mode minimal always) skip

# ---- ergonomic probes ---------------------------------------------------------------------------------
np.random.seed(4)
x = t3.TuckerTensorTrain.randn(shape, n, r); u = ut3.UniformTuckerTensorTrain.from_t3(x)
p_typo = uf.uniform_least_squares_problem('Manifold', 'apply', u, make_sample('apply', ()), np.zeros(W))
rep('CHARACTERIZE: geometry="Manifold" (typo) silently gives', False, type(p_typo.geom).__name__)
try:
    uf.uniform_least_squares_problem('manifold', 'apply_derivatives', u, make_sample('apply_derivatives', ()), np.zeros((3,) + W))
    rep('order=None for a derivative kind raises', False)
except Exception as e:
    rep('CHARACTERIZE: order=None for a derivative kind raises', True, '%s: %s' % (type(e).__name__, str(e)[:120]))
try:
    uf.uniform_sampling_kind('apply', u.data, weight=np.ones(3))
    rep('weight on apply raises', False)
except ValueError as e:
    rep('weight on apply raises ValueError', True)
