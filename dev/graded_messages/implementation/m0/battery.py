"""M0's behaviour battery: the library's outputs, and every einsum call it makes, on one tree.

    PYTHONPATH=<tree> python battery.py OUT_PREFIX            # writes OUT_PREFIX.npz and OUT_PREFIX.calls.txt
    python battery.py --compare PREFIX_A PREFIX_B             # bitwise comparison of two runs

Run once on the untouched tree (a `git archive` of HEAD) and once on the renamed one: the call logs must be identical
and every output `np.array_equal`, dtype included. Not a test (no `test_` name): the record of slice M0.

It uses only names that the rename leaves alone (`sampling_variables` and its keywords, the frontend), so the same
script runs on both trees. Covered:
- every public variable of the three kinds, on a manifold frame and at a corewise point, ragged and uniform;
- the jet shapes (), (1,), (2,), (1, 1), (2, 1) (entries: () only), with input lines (sizes 2) and a mode that
  does not move (size 1);
- the stacks K = () and (2,), C = () and (2,); sum_over_samples True and False;
- a short newton_cg on probe data with jets, ragged and uniform;
- one jitted jax run (float64), with the jnp.einsum calls of its trace logged.
"""
import sys

import numpy as np


def compare(a, b):
    la, lb = open(a + '.calls.txt').read(), open(b + '.calls.txt').read()
    za, zb = np.load(a + '.npz'), np.load(b + '.npz')
    print('calls: %d vs %d lines, identical: %s' % (la.count('\n'), lb.count('\n'), la == lb))
    keys_equal = sorted(za.files) == sorted(zb.files)
    bad = [k for k in za.files if k in zb.files and not (za[k].dtype == zb[k].dtype and za[k].shape == zb[k].shape
                                                          and np.array_equal(za[k], zb[k], equal_nan=True))]
    print('outputs: %d vs %d arrays, same keys: %s, bitwise different: %d' % (len(za.files), len(zb.files),
                                                                               keys_equal, len(bad)))
    for k in bad[:20]:
        print('   ', k)
    ok = la == lb and keys_equal and not bad
    print('BITWISE IDENTICAL' if ok else 'DIFFERENT')
    return 0 if ok else 1


if sys.argv[1] == '--compare':
    sys.exit(compare(sys.argv[2], sys.argv[3]))

prefix = sys.argv[1]
calls = []
_np_einsum = np.einsum


def _logged_np_einsum(*args, **kwargs):
    calls.append('np %r %r %r' % (args[0], [np.shape(a) for a in args[1:]], kwargs.get('optimize')))
    return _np_einsum(*args, **kwargs)


np.einsum = _logged_np_einsum

import jax                                                       # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                          # noqa: E402
_jnp_einsum = jnp.einsum


def _logged_jnp_einsum(*args, **kwargs):
    calls.append('jnp %r %r' % (args[0], [tuple(np.shape(a)) for a in args[1:]]))
    return _jnp_einsum(*args, **kwargs)


jnp.einsum = _logged_jnp_einsum

import t3toolbox.tucker_tensor_train as t3                       # noqa: E402
import t3toolbox.uniform_tucker_tensor_train as ut3              # noqa: E402
import t3toolbox.manifold as t3m                                 # noqa: E402
import t3toolbox.uniform_manifold as ut3m                        # noqa: E402
import t3toolbox.optimizers as optimizers                        # noqa: E402
import t3toolbox.backend.sampling_variables as sv                # noqa: E402
from t3toolbox.backend.jets import affine_jet                    # noqa: E402
from t3toolbox.fitting import LeastSquaresOracle, LeastSquaresBatch   # noqa: E402

out = {}


def store(key, value):
    if value is None:
        return
    if isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            store('%s/%d' % (key, i), v)
        return
    if isinstance(value, dict):
        for k in sorted(value):
            store('%s/%s' % (key, k), value[k])
        return
    if isinstance(value, (int, float, np.floating, np.integer, np.ndarray)) or hasattr(value, 'shape'):
        out[key] = np.asarray(value)


SHAPE, TUCKER, TT, W = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1), (3,)
d = len(SHAPE)
ORDERS = ((), (1,), (2,), (1, 1), (2, 1))


def sample_of(rng, order, C):
    """Input jets: a line (two slots) in every variable but on mode 1, which does not move in the last variable."""
    ww = [rng.standard_normal(W + (n,)) for n in SHAPE]
    if not order:
        return ww
    m = len(order)
    out_ = []
    for i, w in enumerate(ww):
        sizes = tuple(1 if (i == 1 and j == m - 1) else 2 for j in range(m))
        jet = np.zeros(sizes + w.shape)
        jet[(0,) * m] = w
        for j in range(m):
            if sizes[j] == 2:
                jet[tuple(1 if jj == j else 0 for jj in range(m))] = rng.standard_normal(w.shape)
        out_.append(jet)
    return out_


def data_of(rng, kind, order):
    F = tuple(n + 1 for n in order)
    if kind == 'probe':
        return [rng.standard_normal(F + W + (n,)) for n in SHAPE], 0.5 + rng.random((d,) + F)
    if kind == 'apply':
        return rng.standard_normal(F + W), 0.5 + rng.random(F)
    return rng.standard_normal(W), np.asarray(0.7)


def directions_of(rng, frame, call, K, uniform):
    """Random variations (manifold) and core directions, of the frame's shapes with K in front."""
    if call == 'corewise':
        B, G = frame[0], frame[1]
        if uniform:
            return None, (rng.standard_normal(np.shape(B)[:1] + K + np.shape(B)[1:]),
                          rng.standard_normal(np.shape(G)[:1] + K + np.shape(G)[1:]))
        return None, (tuple(rng.standard_normal(K + b.shape) for b in B), tuple(rng.standard_normal(K + g.shape) for g in G))
    D, U, L, R = frame[:4]
    if uniform:
        def rnd(a):
            return rng.standard_normal(np.shape(a)[:1] + K + np.shape(a)[1:])
        V = rng.standard_normal(np.shape(U)[:1] + K + np.shape(U)[1:-3] + (np.shape(U)[-2], np.shape(D)[-1]))
        H = rnd(L)
        return (V, H), (rnd(D), rnd(L))
    V = tuple(rng.standard_normal(K + u.shape[:-3] + (u.shape[-2], dd.shape[-1])) for u, dd in zip(U, D))
    H = tuple(rng.standard_normal(K + l.shape[:-3] + (l.shape[-3], dd.shape[-2], r.shape[-1]))
              for l, dd, r in zip(L, D, R))
    return (V, H), (tuple(rng.standard_normal(K + a.shape) for a in D), tuple(rng.standard_normal(K + a.shape) for a in L))


def frames(rng, C):
    x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT, stack_shape=C)
    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    return {('ragged', 'manifold'): t3m.MANIFOLD.frame(x).data, ('ragged', 'corewise'): x.data,
            ('uniform', 'manifold'): ut3m.UNIFORM_MANIFOLD.frame(ux).data, ('uniform', 'corewise'): ux.data}


def all_wanted(kind, frame, sample, order, args):
    """Every public variable this call can compute (one name at a time through computed_names)."""
    names = sorted(sv._public(sv._GRAPHS[kind, '_' + sv.frame_form(frame)[1]]))
    keep = []
    for name in names:
        try:
            sv.computed_names(kind, frame, sample, (name,), order=order, **args)
            keep.append(name)
        except (ValueError, KeyError):
            pass
    return tuple(keep)


np.random.seed(0)
rng = np.random.default_rng(0)
for C in ((), (2,)):
    fr = frames(rng, C)
    for (layout, call), frame in fr.items():
        for kind in ('probe', 'apply', 'entries'):
            for order in (ORDERS if kind != 'entries' else ((),)):
                if kind == 'entries':
                    sample = np.stack([rng.integers(0, n, size=W) for n in SHAPE])
                else:
                    sample = sample_of(rng, order, C)
                y, weight = data_of(rng, kind, order)
                for K in ((), (2,)):
                    variations, core_directions = directions_of(rng, frame, call, K, layout == 'uniform')
                    args = dict(y=y, weight=weight, core_directions=core_directions)
                    if call == 'manifold':
                        args['variations'] = variations
                    for sos in (True, False):
                        key = '%s/%s/%s/C%s/K%s/%s/sum%s' % (layout, call, kind, C, K, order, sos)
                        want = all_wanted(kind, frame, sample, order, args)
                        values = sv.sampling_variables(kind, frame, sample, want, order=order,
                                                       sum_over_samples=sos, **args)
                        store(key, dict(zip(want, values)))
                        calls.append('# ' + key + ' ' + ','.join(want))

# a short newton_cg on jetted probe data, ragged and uniform
rng = np.random.default_rng(1)
x_true = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
ww, pp = ([rng.standard_normal((20, n)) for n in SHAPE] for _ in range(2))
data = x_true.probe(affine_jet(ww, pp), (2,))
x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)
oracle = LeastSquaresOracle('probe', order=(2,))
batch = LeastSquaresBatch(affine_jet(ww, pp), data, None)
xr, stats_r = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x0, max_newton=3)
store('newton/ragged', xr.data)
store('newton/ragged_stats', {k: v for k, v in stats_r.items() if isinstance(v, (int, float, np.ndarray))})
ux, stats_u = optimizers.newton_cg(ut3m.UNIFORM_MANIFOLD, oracle, batch, ut3.UniformTuckerTensorTrain.from_t3(x0),
                                   max_newton=3)
store('newton/uniform', ux.data[:2])
calls.append('# newton done')

# one jitted jax run: uniform probe at a manifold frame, order (2, 1), K = (2,), C = (2,)
rng = np.random.default_rng(2)
fr = frames(rng, (2,))
frame = fr['uniform', 'manifold']
sample = sample_of(rng, (2, 1), (2,))
y, weight = data_of(rng, 'probe', (2, 1))
variations, core_directions = directions_of(rng, frame, 'manifold', (2,), True)
args = dict(y=y, weight=weight, variations=variations, core_directions=core_directions)
want = all_wanted('probe', frame, sample, (2, 1), args)
supercores = tuple(jnp.asarray(a) for a in frame[:4])
statics = frame[4:]


def run(cores, sample_, y_, weight_, V, H, Dd, Ld):
    return sv.sampling_variables('probe', tuple(cores) + tuple(statics), list(sample_), want, order=(2, 1), y=list(y_),
                                 weight=weight_, variations=(V, H), core_directions=(Dd, Ld))


values = jax.jit(run)(supercores, [jnp.asarray(s) for s in sample], [jnp.asarray(a) for a in y], jnp.asarray(weight),
                      *(jnp.asarray(a) for a in variations + core_directions))
store('jax/uniform/probe', dict(zip(want, values)))
calls.append('# jax done')

np.savez(prefix + '.npz', **out)
with open(prefix + '.calls.txt', 'w') as fh:
    fh.write('\n'.join(calls) + '\n')
print('%d arrays, %d einsum calls' % (len(out), sum(1 for c in calls if not c.startswith('#'))))
