"""Does any op mutate its operands in place? Snapshot core bytes before/after each op."""
import numpy as np, hashlib
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_frame_variations_format as ubv, t3toolbox.uniform_manifold as ut3m

def digest(arrs):
    h = hashlib.md5()
    for a in arrs: h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()[:10]

np.random.seed(1)
shape, tr, ttr = (3, 5, 4), (2, 3, 2), (1, 2, 3, 1)
x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
y = t3.TuckerTensorTrain.randn(shape, tr, ttr)
frame, _ = bvf.t3_orthogonal_representations(x)
ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=8, n=5, r=5); uy = ut3.UniformTuckerTensorTrain.from_t3(y, N=8, n=5, r=5)
uframe, _ = ubv.ut3_orthogonal_representations(ux)
objs = {'x': lambda: x.data[0] + x.data[1], 'frame': lambda: sum((tuple(f) for f in frame.data), ()),
        'ux': lambda: ux.supercores, 'uframe': lambda: uframe.supercores}
ww = [np.random.randn(4, N) for N in shape]; pp = [np.random.randn(4, N) for N in shape]
ops = [
    ('ux.to_dense()', lambda: ux.to_dense()), ('ux + uy', lambda: ux + uy), ('ux * 2.5', lambda: ux * 2.5),
    ('ux.inner(uy)', lambda: ux.inner(uy)), ('ux.norm()', lambda: ux.norm()), ('ux.reverse()', lambda: ux.reverse()),
    ('ux.t3svd()', lambda: ux.t3svd()), ('ux.rank_adjustment_sweep()', lambda: ux.t3svd()[0].rank_adjustment_sweep()),
    ('ux.apply(ww)', lambda: ux.apply(ww)), ('ux.probe_derivatives', lambda: ux.probe_derivatives(ww, pp, 3)),
    ('ux.apply_corewise_transpose', lambda: ux.apply_corewise_transpose(np.random.randn(4), ww)),
    ('ux.probe_corewise_derivatives_transpose', lambda: ux.probe_corewise_derivatives_transpose([np.random.randn(4, 4, N) for N in shape], ww, pp, 3)),
    ('UNIFORM_MANIFOLD.randn(uframe)', lambda: ut3m.UNIFORM_MANIFOLD.randn(uframe)),
    ('UNIFORM_MANIFOLD.randn(uframe, K)', lambda: ut3m.UNIFORM_MANIFOLD.randn(uframe, stack_shape=(2,))),
    ('x.apply(ww)', lambda: x.apply(ww)), ('x.t3svd()', lambda: x.t3svd()), ('x + y', lambda: x + y), ('x * y', lambda: x * y),
    ('MANIFOLD.randn(frame)', lambda: t3m.MANIFOLD.randn(frame)), ('MANIFOLD.project_ambient(frame, y)', lambda: t3m.MANIFOLD.project_ambient(frame, y)),
    ('MANIFOLD.retract(randn)', lambda: t3m.MANIFOLD.retract(t3m.MANIFOLD.randn(frame))),
    ('x.probe_corewise_derivatives_transpose', lambda: x.probe_corewise_derivatives_transpose([np.random.randn(4, 4, N) for N in shape], ww, pp, 3)),
]
before = {k: f() for k, f in ((k, lambda f=f: digest(f())) for k, f in objs.items())}
for name, op in ops:
    try: op()
    except Exception as e: print('%-45s RAISED %s: %s' % (name, type(e).__name__, str(e)[:80]))
    after = {k: digest(f()) for k, f in objs.items()}
    changed = [k for k in objs if after[k] != before[k]]
    print('%-45s %s' % (name, 'MUTATED: ' + ', '.join(changed) if changed else 'ok'))
    before = after
