"""M2's battery: the weighted outputs of the library on one tree, before and after the switch to the rule.

    PYTHONPATH=<tree> python battery.py OUT_PREFIX old         # on a `git archive` of 85c6e54d: block weights,
                                                               #   a y shared over a stack without its C axes
    PYTHONPATH=<tree> python battery.py OUT_PREFIX new         # on the M2 tree: the same blocks through
                                                               #   blocks_to_weight, the shared y with size-1 C axes
    python battery.py --compare OLD_PREFIX NEW_PREFIX           # the comparison, under the expectations below

Not a test (no `test_` name): the record of M2's gate (1), "every weighted output reproduces its pre-M2 numbers"
(M2_DESIGN_NOTE.md section 2, with the design review's finding 2). The same script runs on both trees; the only
differences are the forms the arguments are passed in (`weight_form`, `y_form`, `taylor_weight`). Every weight
stored is normalized to the full form (`blocks_to_weight`, present on both trees since M1), so that the two runs
have the same keys and shapes.

Covered:
- the engine: every public variable of the three kinds, on a manifold frame and at a corewise point, ragged and
  uniform, at the jet shapes (), (1,), (2,), (1, 1), (2, 1) (entries ()), K = () and (2,), C = () and (2,) (one y
  shared over the stack), sum_over_samples True and False, with a nonuniform weight per block;
- the oracle: f, g, JtJp, Jp_norm_sq, Kp, r, Jp, r_block_sumsq, y_block_sumsq, f_resolution, on full and drawn
  batches, the three kinds, ragged manifold, ragged corewise and uniform manifold, C = () and (2,);
- prepare_derivative_data (a weight and None) and derivative_weight_to_taylor_weight;
- short weighted runs of newton_cg and mc_sgd, ragged and uniform, in the new call form and in the old (translated)
  one;
- one jitted jax run of the engine (float64).

The expectations (`--compare`):
- **bitwise** (dtype, shape and every bit): everything not listed below. The seed `w_tilde` is the same product
  `(omega*omega) * r` in the same order (M1's bridge test), so every tilde variable, `g`, `JtJp`, `Kp` follow; `z`,
  `r`, `Jp`, `r_block_sumsq`, `y_block_sumsq` are untouched by M2 (a sum over an extra size-1 axis adds nothing);
- **to rounding**, relative error at most 1e-13: `f`, `Jp_norm_sq` (`r * (omega**2 * r)` in place of
  `omega**2 * (r * r)`), `f_resolution` (`<y, M y>` over z's shape in place of block sums);
- **the jitted run**, to 1e-12: XLA's fusion of a changed program need not keep a reduction bitwise;
- **the optimizer runs**: the iteration counts and the stop reason exactly; the losses and the iterates to 1e-10
  (`mc_sgd` with `check_every=1`, so that its full-batch loss is recorded at every step, not only the endpoint).
  Rounding-level changes in `f` and `Jp_norm_sq` can only move them through the line search and the CG tests; three
  Newton steps from the zero tensor amplify a 1e-16 change by far less than 1e-10. If a run sits on a knife edge (an
  acceptance decided at rounding level), the run is changed, not the tolerance.
"""
import sys

import numpy as np

ROUNDING = {'f', 'Jp_norm_sq', 'f_resolution'}
RTOL_ROUNDING, RTOL_JAX, RTOL_OPT = 1e-13, 1e-12, 1e-10
EXACT_OPT = ('newton_steps', 'newton_accepted', 'stop_reason', 'n_iter')


def _rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)) if a.size else 0.0


def compare(a, b):
    za, zb = np.load(a + '.npz'), np.load(b + '.npz')
    keys_a, keys_b = set(za.files), set(zb.files)
    bad, worst = [], {}
    if keys_a != keys_b:
        bad.append('keys differ: only in old %s, only in new %s'
                   % (sorted(keys_a - keys_b)[:10], sorted(keys_b - keys_a)[:10]))
    counts = dict(bitwise=0, rounding=0, jax=0, opt_exact=0, opt=0)
    for key in sorted(keys_a & keys_b):
        x, y = za[key], zb[key]
        parts = key.split('/')
        if x.shape != y.shape or (x.dtype != y.dtype and x.dtype.kind != 'U'):
            bad.append('%s: shape/dtype %s %s vs %s %s' % (key, x.shape, x.dtype, y.shape, y.dtype))
            continue
        if parts[0] == 'opt':
            if parts[-1] in EXACT_OPT:
                counts['opt_exact'] += 1
                if not np.array_equal(x, y):
                    bad.append('%s: %s vs %s' % (key, x, y))
            else:
                counts['opt'] += 1
                err = _rel(y, x)
                worst['opt'] = max(worst.get('opt', 0.0), err)
                if err > RTOL_OPT:
                    bad.append('%s: relative error %.2e' % (key, err))
        elif parts[0] == 'jax':
            counts['jax'] += 1
            err = _rel(y, x)
            worst['jax'] = max(worst.get('jax', 0.0), err)
            if err > RTOL_JAX:
                bad.append('%s: relative error %.2e' % (key, err))
        elif ROUNDING & set(parts):
            counts['rounding'] += 1
            err = _rel(y, x)
            worst['rounding'] = max(worst.get('rounding', 0.0), err)
            if err > RTOL_ROUNDING:
                bad.append('%s: relative error %.2e' % (key, err))
        else:
            counts['bitwise'] += 1
            if not np.array_equal(x, y, equal_nan=True):
                bad.append('%s: not bitwise (relative error %.2e)' % (key, _rel(y, x)))
    print('arrays: %d old, %d new; checked %s' % (len(keys_a), len(keys_b), counts))
    print('largest relative errors: %s' % {k: '%.2e' % v for k, v in worst.items()})
    for line in bad[:40]:
        print('   ', line)
    print('AS EXPECTED' if not bad else '%d UNEXPECTED' % len(bad))
    return 0 if not bad else 1


if sys.argv[1] == '--compare':
    sys.exit(compare(sys.argv[2], sys.argv[3]))

prefix, FORM = sys.argv[1], sys.argv[2]
assert FORM in ('old', 'new'), FORM

import jax                                                       # noqa: E402
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                          # noqa: E402

import t3toolbox.tucker_tensor_train as t3                       # noqa: E402
import t3toolbox.uniform_tucker_tensor_train as ut3              # noqa: E402
import t3toolbox.manifold as t3m                                 # noqa: E402
import t3toolbox.uniform_manifold as ut3m                        # noqa: E402
import t3toolbox.optimizers as optimizers                        # noqa: E402
import t3toolbox.backend.jets as jets                            # noqa: E402
import t3toolbox.backend.sampling_variables as sv                # noqa: E402
import t3toolbox.backend.uniform_fitting as ufit                 # noqa: E402
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
    if isinstance(value, str):
        out[key] = np.asarray(value)
        return
    if isinstance(value, (int, float, np.floating, np.integer, np.ndarray)) or hasattr(value, 'shape'):
        out[key] = np.asarray(value)


# ---------------------------------------------------------------------------------------------- the two forms

def y_form(kind, order, y, C):
    """A y shared over the stack C: as it is on the old tree; with size-1 C axes (after F + W) on the new one."""
    if FORM == 'old' or not C:
        return y
    if kind == 'probe':
        return [np.reshape(y_i, np.shape(y_i)[:-1] + (1,) * len(C) + np.shape(y_i)[-1:]) for y_i in y]
    return np.reshape(y, np.shape(y) + (1,) * len(C))


def weight_form(kind, order, blocks, y):
    """One weight per block: as it is on the old tree; through blocks_to_weight on the new one (y in its form)."""
    if FORM == 'old' or blocks is None:
        return blocks
    return jets.blocks_to_weight(kind, order, blocks, y)


def full(kind, order, weight, y):
    """A weight returned by the library, stored in the full form (the old tree returns blocks)."""
    return jets.blocks_to_weight(kind, order, weight, y) if FORM == 'old' else weight


def taylor_weight(kind, order, weight):
    if FORM == 'old':
        return jets.derivative_weight_to_taylor_weight(weight, order)
    return jets.derivative_weight_to_taylor_weight(kind, order, weight)


# ---------------------------------------------------------------------------------------------- the problems

SHAPE, TUCKER, TT, W = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1), (3,)
d = len(SHAPE)
ORDERS = ((), (1,), (2,), (1, 1), (2, 1))


def sample_of(rng, order):
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
    """(y, blocks): the data without C, and one nonuniform weight per block."""
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
        return None, (tuple(rng.standard_normal(K + b.shape) for b in B),
                      tuple(rng.standard_normal(K + g.shape) for g in G))
    D, U, L, R = frame[:4]
    if uniform:
        def rnd(a):
            return rng.standard_normal(np.shape(a)[:1] + K + np.shape(a)[1:])
        V = rng.standard_normal(np.shape(U)[:1] + K + np.shape(U)[1:-3] + (np.shape(U)[-2], np.shape(D)[-1]))
        return (V, rnd(L)), (rnd(D), rnd(L))
    V = tuple(rng.standard_normal(K + u.shape[:-3] + (u.shape[-2], dd.shape[-1])) for u, dd in zip(U, D))
    H = tuple(rng.standard_normal(K + l.shape[:-3] + (l.shape[-3], dd.shape[-2], r.shape[-1]))
              for l, dd, r in zip(L, D, R))
    return (V, H), (tuple(rng.standard_normal(K + a.shape) for a in D),
                    tuple(rng.standard_normal(K + a.shape) for a in L))


def frames(C):
    x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT, stack_shape=C)
    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    return x, {('ragged', 'manifold'): t3m.MANIFOLD.frame(x).data, ('ragged', 'corewise'): x.data,
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


# ---------------------------------------------------------------------------------------------- 1. the engine

np.random.seed(0)
rng = np.random.default_rng(0)
for C in ((), (2,)):
    _, fr = frames(C)
    for (layout, call), frame in fr.items():
        for kind in ('probe', 'apply', 'entries'):
            for order in (ORDERS if kind != 'entries' else ((),)):
                if kind == 'entries':
                    sample = np.stack([rng.integers(0, n, size=W) for n in SHAPE])
                else:
                    sample = sample_of(rng, order)
                y0, blocks = data_of(rng, kind, order)
                y = y_form(kind, order, y0, C)
                weight = weight_form(kind, order, blocks, y)
                for K in ((), (2,)):
                    variations, core_directions = directions_of(rng, frame, call, K, layout == 'uniform')
                    args = dict(y=y, weight=weight, core_directions=core_directions)
                    if call == 'manifold':
                        args['variations'] = variations
                    for sos in (True, False):
                        key = 'eng/%s/%s/%s/C%s/K%s/%s/sum%s' % (layout, call, kind, C, K, order, sos)
                        want = all_wanted(kind, frame, sample, order, args)
                        values = sv.sampling_variables(kind, frame, sample, want, order=order,
                                                       sum_over_samples=sos, **args)
                        store(key, dict(zip(want, values)))

# ---------------------------------------------------------------------------------------------- 2. the oracle

ORACLE_WANT = ('f', 'g', 'JtJp', 'Jp_norm_sq', 'r', 'Jp', 'r_block_sumsq', 'y_block_sumsq', 'f_resolution')
rng = np.random.default_rng(3)
for C in ((), (2,)):
    x, fr = frames(C)
    for (layout, call), frame in fr.items():
        if (layout, call) == ('uniform', 'corewise'):
            continue
        for kind in ('probe', 'apply', 'entries'):
            for order in (((), (2,), (1, 1)) if kind != 'entries' else ((),)):
                sample = (np.stack([rng.integers(0, n, size=W) for n in SHAPE]) if kind == 'entries'
                          else sample_of(rng, order))
                y0, blocks = data_of(rng, kind, order)
                y = y_form(kind, order, y0, C)
                batch = LeastSquaresBatch(sample, y, weight_form(kind, order, blocks, y))
                oracle = LeastSquaresOracle(kind, order)
                if layout == 'uniform':
                    batch = ufit.pack_batch(kind, order, batch, int(np.shape(frame[0])[-1]))
                variations, core_directions = directions_of(rng, frame, call, (), layout == 'uniform')
                if call == 'corewise':
                    variations = core_directions
                for drawn in (False, True):
                    b = oracle.take(batch, np.array([2, 0])) if drawn else batch
                    want = ORACLE_WANT + (('Kp',) if call == 'corewise' or layout == 'ragged' else ())
                    values, _ = oracle.query(frame, b, want, variations=variations, core_directions=core_directions)
                    store('orc/%s/%s/%s/C%s/%s/drawn%s' % (layout, call, kind, C, order, drawn),
                          dict(zip(want, values)))

# ---------------------------------------------------------------------------------------------- 3. derivative data

rng = np.random.default_rng(4)
for kind in ('probe', 'apply', 'entries'):
    for order in (((2,), (1, 2)) if kind != 'entries' else ((),)):
        F = tuple(n + 1 for n in order)
        y_d, blocks = data_of(rng, kind, order)
        val = [2.0 * a for a in y_d] if kind == 'probe' else 2.0 * y_d
        for given in (True, False):
            weight = weight_form(kind, order, blocks, y_d) if given else None
            y_t, w_t, v_t = jets.prepare_derivative_data(kind, order, y_d, weight, val)
            store('drv/%s/%s/given%s' % (kind, order, given), dict(y=y_t, weight=full(kind, order, w_t, y_d),
                                                                   validation=v_t))
        store('drv/%s/%s/taylor_weight' % (kind, order),
              full(kind, order, taylor_weight(kind, order, weight_form(kind, order, blocks, y_d)), y_d))

# ---------------------------------------------------------------------------------------------- 4. optimizer runs

rng = np.random.default_rng(1)
x_true = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
ww, pp = ([rng.standard_normal((20, n)) for n in SHAPE] for _ in range(2))
data = x_true.probe(affine_jet(ww, pp), (2,))
data = [z + 0.01 * rng.standard_normal(z.shape) for z in data]              # noisy: no exact fit to reach
x0 = t3.TuckerTensorTrain.zeros(SHAPE, TUCKER, TT)
ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)
omega_blocks = 0.5 + rng.random((d, 3))                                     # per mode and slot, Taylor
oracle = LeastSquaresOracle('probe', order=(2,))
batch = LeastSquaresBatch(affine_jet(ww, pp), data, weight_form('probe', (2,), omega_blocks, data))


def store_run(key, x, stats):
    store(key + '/x', x.data[:2])
    store(key, {k: v for k, v in stats.items() if k in ('losses',) + EXACT_OPT})


store_run('opt/newton/ragged/new', *optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x0, max_newton=3))
store_run('opt/newton/uniform/new', *optimizers.newton_cg(ut3m.UNIFORM_MANIFOLD, oracle, batch, ux0, max_newton=3))
# mc_sgd starts at a seeded random point (part 5): the cores' sign gauge of the SVD retraction can flip under a
# rounding-level change of the step, from any start (the part-5 review saw it in 4 of 8 runs from generic starts),
# while the tensors and losses agree; this seeded run happens not to flip. Losses and counts are the robust checks.
_state = np.random.get_state()                                              # the later sections' draws unchanged
np.random.seed(5)
x0_mc = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
np.random.set_state(_state)
store_run('opt/mc_sgd/ragged/new', *optimizers.mc_sgd(t3m.MANIFOLD, oracle, batch, x0_mc, np.random.default_rng(7), 8,
                                                      max_iter=4, check_every=1))
data_derivatives = jets.taylor_to_derivatives('probe', (2,), data)
omega_derivatives = 0.5 + rng.random((d, 3))                                # the old form: ω[mode, order]
store_run('opt/newton/ragged/old', *optimizers.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp),
                                                         data_derivatives, x0, order=2, weight=omega_derivatives,
                                                         max_newton=3))
data_plain = x_true.probe(ww)
omega_mode = 0.5 + rng.random(d)                                            # the old plain-probe form: (d,)
store_run('opt/mc_sgd/ragged/old', *optimizers.mc_sgd(t3m.MANIFOLD, 'probe', ww, data_plain, x0_mc,
                                                      np.random.default_rng(8), 8, weight=omega_mode, max_iter=4,
                                                      check_every=1))

# ---------------------------------------------------------------------------------------------- 5. one jitted run

rng = np.random.default_rng(2)
_, fr = frames((2,))
frame = fr['uniform', 'manifold']
sample = sample_of(rng, (2, 1))
y0, blocks = data_of(rng, 'probe', (2, 1))
y = y_form('probe', (2, 1), y0, (2,))
weight = weight_form('probe', (2, 1), blocks, y)
variations, core_directions = directions_of(rng, frame, 'manifold', (2,), True)
args = dict(y=y, weight=weight, variations=variations, core_directions=core_directions)
want = all_wanted('probe', frame, sample, (2, 1), args)
supercores = tuple(jnp.asarray(a) for a in frame[:4])
statics = frame[4:]


def run(cores, sample_, y_, weight_, V, H, Dd, Ld):
    return sv.sampling_variables('probe', tuple(cores) + tuple(statics), list(sample_), want, order=(2, 1), y=list(y_),
                                 weight=weight_, variations=(V, H), core_directions=(Dd, Ld))


values = jax.jit(run)(supercores, [jnp.asarray(s) for s in sample], [jnp.asarray(a) for a in y],
                      jnp.asarray(weight), *(jnp.asarray(a) for a in variations + core_directions))
store('jax/uniform/probe', dict(zip(want, values)))

np.savez(prefix + '.npz', **out)
print('%s form: %d arrays' % (FORM, len(out)))
