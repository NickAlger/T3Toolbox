"""R9-B2: the shared-minimal gate with an explicitly TIED non-minimal start (not share()-projected);
uniform_minimal(sharing=) output ranks; geometry typo; order=None Problem first use."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.sharing as sharing
import t3toolbox.backend.optimizers as bopt
np.random.seed(0)
def tied(shape, n, r, spec):
    x = t3.TuckerTensorTrain.randn(shape, n, r); tk = list(x.tucker_cores)
    for g in sharing.validate_sharing(spec, shape):
        for ii in g[1:]: tk[ii] = tk[g[0]]
    return t3.TuckerTensorTrain(tuple(tk), x.tt_cores)
shape, n, r, spec = (6, 6, 4), (2, 2, 2), (1, 3, 3, 1), (0, 0, 1)
print('shared-minimal for', n, r, '->', ranks.compute_minimal_ranks(shape, n, r, sharing=spec), ' per-mode ->', ranks.compute_minimal_ranks(shape, n, r))
u = ut3.UniformTuckerTensorTrain.from_t3(tied(shape, n, r, spec))
print('u ranks', tuple(map(int, u.tucker_ranks)), tuple(map(int, u.tt_ranks)), 'tied:', bool(u.has_shared_tucker_factors(spec)))
ww = [np.random.randn(5, Ni) for Ni in shape]
try:
    uf.uniform_least_squares_problem('manifold', 'apply', u, ww, np.random.randn(5), sharing=spec); print('FAIL gate did not raise')
except ValueError as e:
    print('PASS gate raises:', str(e)[:90])
um = uf.uniform_minimal(u, sharing=spec)
print('uniform_minimal(sharing) ranks', tuple(map(int, um.tucker_ranks)), tuple(map(int, um.tt_ranks)), ' tied:', bool(um.has_shared_tucker_factors(spec)),
      ' same tensor:', np.allclose(np.asarray(um.to_dense()), np.asarray(u.to_dense())))
prob = uf.uniform_least_squares_problem('manifold', 'apply', um, ww, np.random.randn(5), sharing=spec)
print('PASS gate passes after uniform_minimal(sharing); groups =', prob.geom.groups)
# the earlier confusion: .share() changes ranks
xs = t3.TuckerTensorTrain.randn(shape, n, r).share(spec)
print('NOTE: randn(%s,%s).share(%s) has ranks' % (n, r, spec), xs.tucker_ranks, xs.tt_ranks)
# geometry typo
p = uf.uniform_least_squares_problem('Manifold', 'apply', um, ww, np.random.randn(5), sharing=spec)
print('geometry="Manifold" (typo) ->', type(p.geom).__name__, ' (no error)')
# order=None derivative kind: Problem builds, first use?
pp = [np.random.randn(5, Ni) for Ni in shape]
try:
    p2 = uf.uniform_least_squares_problem('manifold', 'apply_derivatives', um, (ww, pp), np.random.randn(3, 5))
    print('order=None: Problem built; kind.order =', p2.kind.order)
    try:
        val = p2.objective((um.data[0], um.data[1]))
        print('order=None: objective =', val)
    except Exception as e:
        print('order=None: first use raises %s: %s' % (type(e).__name__, str(e)[:110]))
except Exception as e:
    print('order=None: construction raises %s: %s' % (type(e).__name__, str(e)[:110]))
