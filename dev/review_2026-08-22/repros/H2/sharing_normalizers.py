"""The sharing-partition normalizers: None vs all-singleton vs list/np/str labels must agree everywhere."""
import numpy as np, t3toolbox as t3t
import t3toolbox.backend.ranks as ranks, t3toolbox.manifold as t3m, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.shared_geometry as sg, t3toolbox.optimizers as opt, t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.geometry as bgeo, t3toolbox.backend.sharing as bsh
np.random.seed(0)
shape = (5, 5, 6)
x = t3t.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)) + t3t.TuckerTensorTrain.randn(shape, (2, 2, 3), (1, 3, 2, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
def rec(s):
    out = {}
    out['min'] = ranks.compute_minimal_ranks(shape, x.tucker_ranks, x.tt_ranks, sharing=s)
    out['dim'] = t3m.manifold_dim((shape, x.tucker_ranks, x.tt_ranks), sharing=s)
    a = x.t3svd(sharing=s)[0]; out['t3svd'] = (a.tucker_ranks, a.tt_ranks, round(float(np.linalg.norm(a.to_dense() - x.to_dense())), 10))
    b = x.rank_adjustment_sweep(sharing=s); out['ras'] = (b.tucker_ranks, b.tt_ranks)
    ua = ux.t3svd(sharing=s)[0]; out['ut3svd'] = (round(float(np.linalg.norm(ua.to_dense() - x.to_dense())), 10),)
    ub = ux.rank_adjustment_sweep(sharing=s); out['uras'] = (round(float(np.linalg.norm(ub.to_dense() - x.to_dense())), 10),)
    out['cont'] = x.continuation_ranks(sharing=s) if s is not None or True else None
    out['frame_min'] = bool(ranks.frame_has_minimal_ranks(shape, a.tucker_ranks, a.tucker_ranks, a.tt_ranks, a.tt_ranks, sharing=s))
    out['cg'] = bsh.canonical_groups(s, shape)
    return out
specs = {'None': None, 'tuple': (0, 1, 2), 'list': [0, 1, 2], 'np': np.array([0, 1, 2]), 'str': ('a', 'b', 'c')}
base = rec(None)
for name, s in specs.items():
    r = rec(s)
    diff = {k: (base[k], r[k]) for k in base if str(base[k]) != str(r[k])}
    print(f'all-singleton spec {name:6s}: agrees with None? {not diff}', diff if diff else '')
print('--- nontrivial partition in several spellings')
nt = {'tuple': (0, 0, 1), 'list': [0, 0, 1], 'np': np.array([0, 0, 1]), 'str': ('a', 'a', 'b'), 'np-int32': np.array([0, 0, 1], dtype=np.int32)}
xs = x.share((0, 0, 1)) if hasattr(x, 'share') else x
uxs = ut3.UniformTuckerTensorTrain.from_t3(xs)
def rec2(s):
    out = {}
    out['min'] = ranks.compute_minimal_ranks(shape, xs.tucker_ranks, xs.tt_ranks, sharing=s)
    out['dim'] = t3m.manifold_dim((shape, xs.tucker_ranks, xs.tt_ranks), sharing=s)
    a = xs.t3svd(sharing=s)[0]; out['t3svd'] = (a.tucker_ranks, a.tt_ranks, round(float(np.linalg.norm(a.to_dense() - xs.to_dense())), 10))
    ua = uxs.t3svd(sharing=s)[0]; out['ut3svd'] = round(float(np.linalg.norm(ua.to_dense() - xs.to_dense())), 10)
    out['cg'] = bsh.canonical_groups(s, shape)
    g = sg.shared_manifold(s); out['geom_eq_tuple'] = (g == sg.shared_manifold((0, 0, 1)), hash(g) == hash(sg.shared_manifold((0, 0, 1))))
    bg = bgeo.ManifoldGeometryOps().with_sharing(s, shape); out['bgeom_eq'] = (bg == bgeo.ManifoldGeometryOps().with_sharing((0, 0, 1), shape), hash(bg) == hash(bgeo.ManifoldGeometryOps().with_sharing((0, 0, 1), shape)))
    return out
b2 = rec2((0, 0, 1))
for name, s in nt.items():
    try:
        r = rec2(s)
        diff = {k: (b2[k], r[k]) for k in b2 if str(b2[k]) != str(r[k])}
        print(f'spelling {name:8s}: agrees with tuple? {not diff}', diff if diff else '')
    except Exception as e:
        print(f'spelling {name:8s}: RAISED {type(e).__name__}: {str(e)[:100]}')
print('--- SharedGeometry with an all-singleton partition vs the plain geometry (fit results)')
ww = [np.random.randn(20, N) for N in shape]; b = x.apply(ww)
x0 = t3t.TuckerTensorTrain.zeros(shape, (2, 2, 2), (1, 2, 2, 1))
xa, sa = opt.newton_cg(t3m.MANIFOLD, 'apply', ww, b, x0, max_newton=3)
xb, sb = opt.newton_cg(sg.shared_manifold((0, 1, 2)), 'apply', ww, b, x0, max_newton=3)
print('plain vs shared(all-singleton): dense rel diff %.1e' % (np.linalg.norm(xa.to_dense() - xb.to_dense()) / np.linalg.norm(xa.to_dense())))
print('precompute(frame) on all-singleton shared manifold:', type(sg.shared_manifold((0, 1, 2)).precompute(t3m.MANIFOLD.frame(x))).__name__)
print('canonical_groups((0,1,2)) =', bsh.canonical_groups((0, 1, 2), shape), '; SharedGeometry.groups =', sg.shared_manifold((0, 1, 2)).groups(shape))
