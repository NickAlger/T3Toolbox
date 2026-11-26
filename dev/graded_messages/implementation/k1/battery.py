#!/usr/bin/env python3
"""K1's battery: the weighted outputs of the library on one tree, before K1, after part 0, after part 5.

    PYTHONPATH=<tree> python -B battery.py OUT_PREFIX old      # on a `git archive` of 282b46a9 (K1's start)
    PYTHONPATH=<tree> python -B battery.py OUT_PREFIX part0    # on the tree after part 0 (the batch has kind, order;
                                                               #   one field `weight`)
    PYTHONPATH=<tree> python -B battery.py OUT_PREFIX new      # on the tree after part 5 (block_weight, full_weight)
    python battery.py --compare OLD_PREFIX NEW_PREFIX [--expect part0|new]

Not a test: the record of K1's gate (K1_DESIGN_NOTE.md section 7). The same script runs on every tree; only the
adapter below (`_Api`) differs, by FORM: how an oracle and a batch are made, how a batch is drawn, counted and packed,
and how a weight is passed (`old`, `part0`: a block weight through `jets.blocks_to_weight`, a full weight as it is, both
as their product; `new`: `block_weight=`, `full_weight=`). Every input is drawn from a seeded generator per case, so the
three trees see the same numbers.

Covered (a key per output, `case/name[/i...]`):
- the engine: `f`, `g`, `r_block_sumsq`, `w_tilde`, `r` and, with a direction, `Jp`, `Jp_norm_sq`, `JtJp`, `Kp`; the
  three kinds at their jet shapes; a manifold frame and a corewise point (ragged), a uniform frame; `C = ()` and
  `C = (2,)` with `y` shared (`C` of 1); the weights none, block, full (per entry), both;
- the oracle: the same outputs plus `y_block_sumsq` and `f_resolution`, on the full batch and on a drawn one (`take`
  at fixed indices), ragged manifold, ragged corewise, uniform (`pack_batch`);
- the batch: `n_measurements`, `take`'s fields, `check`'s verdict on a good and a bad batch, `pack_batch`'s fields;
- the default draw (`flat_draw`) at a fixed generator;
- the display (`make_newton_display` with a validation batch) through a short backend `newton_cg`;
- short frontend optimizer runs: `newton_cg` (ragged probe with a block weight; uniform apply), `mc_sgd` (default draw),
  `adam` (corewise), `gradient_descent`, and the old (translated) call form with a probe-derivative weight;
- one jitted query (jax, float32 by default).

Expectations (`--compare`):
- `--expect part0`: every key bitwise (dtype, shape, every bit), the optimizer paths included; the keys equal.
- `--expect new`: bitwise but for the keys listed by the note's section 7.2 (the "both" rows, the uniform
  `y_block_sumsq` and `f_resolution`, the jitted rows), which are compared to their tolerances.
"""
import json
import sys
import traceback

import numpy as np

ROUNDING_TAGS = ('/both/', 'uniform/y_block_sumsq', 'uniform/f_resolution')
RTOL_ROUNDING, RTOL_JAX = 1e-13, 1e-12


# ---------------------------------------------------------------------------------------------------------------------
# The adapter: the one place the three trees differ
# ---------------------------------------------------------------------------------------------------------------------
class _Api:
    def __init__(self, form):
        assert form in ('old', 'part0', 'new'), form
        self.form = form
        import t3toolbox.backend.fitting as bfit
        import t3toolbox.backend.jets as jets
        import t3toolbox.backend.uniform_fitting as uf
        import t3toolbox.backend.optimizers as bopt
        self.bfit, self.jets, self.uf, self.bopt = bfit, jets, uf, bopt

    def oracle(self, kind, order):
        if self.form == 'old':
            return self.bfit.LeastSquaresOracle(kind, order)
        return self.bfit.LeastSquaresOracle()

    def _weight(self, kind, order, y, block, full):
        '''old and part0: the one weight of the type of z.'''
        w = None if block is None else self.jets.blocks_to_weight(kind, order, block, y)
        if full is None:
            return w
        if w is None:
            return full
        if kind == 'probe':
            return [w[i] * full[i] for i in range(len(full))]
        return w * full

    def batch(self, kind, order, sample, y, block=None, full=None, scale=None, indices=None):
        B = self.bfit.LeastSquaresBatch
        if self.form == 'new':
            return B(kind, order, sample, y, block_weight=block, full_weight=full, minibatch_scale=scale,
                     drawn_indices=indices)
        w = self._weight(kind, order, y, block, full)
        if self.form == 'old':
            return B(sample, y, w, scale, indices)
        return B(kind, order, sample, y, weight=w, minibatch_scale=scale, drawn_indices=indices)

    def engine_weights(self, kind, order, y, block, full):
        if self.form == 'new':
            return dict(block_weight=block, full_weight=full)
        return dict(weight=self._weight(kind, order, y, block, full))

    def take(self, oracle, batch, indices):
        return oracle.take(batch, indices) if self.form == 'old' else batch.take(indices)

    def n_measurements(self, oracle, batch):
        return oracle.n_measurements(batch) if self.form == 'old' else batch.n_measurements

    def check(self, oracle, batch, shape, stack_shape=()):
        if self.form == 'old':
            return oracle.check_batch(batch, shape, stack_shape=stack_shape)
        return batch.check(shape, stack_shape=stack_shape)

    def pack(self, kind, order, batch, N):
        if self.form == 'old':
            return self.uf.pack_batch(kind, order, batch, N)
        return self.uf.pack_batch(batch, N)

    def flat_draw(self, oracle, batch, size):
        if self.form == 'old':
            return self.bopt.flat_draw(oracle, batch, size)
        return self.bopt.flat_draw(batch, size)


# ---------------------------------------------------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------------------------------------------------
class _Record:
    def __init__(self):
        self.arrays, self.notes = {}, {}

    def put(self, key, value):
        if value is None:
            self.notes[key] = 'None'
        elif isinstance(value, (tuple, list)):
            self.notes[key] = 'seq%d' % len(value)
            for i, v in enumerate(value):
                self.put('%s/%d' % (key, i), v)
        elif isinstance(value, dict):
            for k in sorted(value):
                self.put('%s/%s' % (key, k), value[k])
        elif isinstance(value, str):
            self.notes[key] = value
        else:
            self.arrays[key] = np.asarray(value)

    def error(self, key, exc):
        self.notes[key] = 'raises %s' % type(exc).__name__


def _frame_data(x, which):
    import t3toolbox.manifold as t3m
    import t3toolbox.uniform_frame_variations_format as ubv
    if which == 'manifold':
        return t3m.MANIFOLD.frame(x).data
    if which == 'corewise':
        return x.data
    return ubv.UT3Frame.from_t3frame(t3m.MANIFOLD.frame(x)).data


def _directions(rng, frame, which, K=()):
    if which == 'manifold':
        D, U, L, R = frame
        C = tuple(D[0].shape[:-2])
        V = tuple(rng.standard_normal(K + C + (U[i].shape[-2], D[i].shape[-1])) for i in range(len(D)))
        H = tuple(rng.standard_normal(K + C + (L[i].shape[-3], D[i].shape[-2], R[i].shape[-1])) for i in range(len(D)))
        Dd = tuple(rng.standard_normal(K + d.shape) for d in D)
        Ld = tuple(rng.standard_normal(K + l.shape) for l in L)
        return dict(variations=(V, H), core_directions=(Dd, Ld))
    B, G = frame
    return dict(core_directions=(tuple(rng.standard_normal(K + b.shape) for b in B),
                                 tuple(rng.standard_normal(K + g.shape) for g in G)))


SHAPE, TUCKER, TT = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1)
CASES = [('probe', ()), ('probe', (1,)), ('probe', (2,)), ('apply', ()), ('apply', (2,)), ('apply', (1, 1)),
         ('entries', ())]


def _problem(seed, kind, order, C=(), W=7):
    '''A point, its sample, its data y (shared by the stack: C of 1), a block table and a full weight (per entry).'''
    import t3toolbox.tucker_tensor_train as t3
    import t3toolbox.backend.jets as jets
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    x = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT, stack_shape=C)
    F = tuple(n + 1 for n in order)
    m = len(order)
    if kind == 'entries':
        sample = np.stack([rng.integers(0, n, size=(W,)) for n in SHAPE])
    elif m:
        ww = [rng.standard_normal((W, n)) for n in SHAPE]
        pp = [[rng.standard_normal((W, n)) for n in SHAPE] for _ in range(m)]
        sample = jets.affine_jet(ww, *pp)
    else:
        sample = [rng.standard_normal((W, n)) for n in SHAPE]
    ones_C = (1,) * len(C)
    if kind == 'probe':
        y = [rng.standard_normal(F + (W,) + ones_C + (n,)) for n in SHAPE]
        full = [0.5 + rng.random(F + (W,) + ones_C + (n,)) for n in SHAPE]
        block = 0.5 + rng.random((len(SHAPE),) + F)
    else:
        y = rng.standard_normal(F + (W,) + ones_C)
        full = 0.5 + rng.random(F + (W,) + ones_C)
        block = 0.5 + rng.random(F) if F else np.asarray(0.5 + rng.random())
    return rng, x, sample, y, block, full


WEIGHTS = ('none', 'block', 'full', 'both')


def _weights(label, block, full):
    return dict(none=(None, None), block=(block, None), full=(None, full), both=(block, full))[label]


def engine_rows(api, rec):
    import t3toolbox.backend.sampling_variables as sv
    for kind, order in CASES:
        for which in ('manifold', 'corewise', 'uniform'):
            for C in ((), (2,)):
                if which == 'uniform' and C:
                    continue
                for wl in WEIGHTS:
                    case = 'engine/%s%s/%s/C%d/%s' % (kind, order, which, len(C), wl)
                    try:
                        rng, x, sample, y, block, full = _problem(11, kind, order, C)
                        if C:                                   # the full weight per element of C is not shared
                            pass
                        frame = _frame_data(x, which)
                        b, f = _weights(wl, block, full)
                        kw = api.engine_weights(kind, order, y, b, f)
                        want = ('r', 'f', 'g', 'r_block_sumsq', 'w_tilde')
                        args = dict(order=order, y=y, **kw)
                        if which != 'uniform':
                            args.update(_directions(rng, frame, which))
                            want = want + ('Jp', 'Jp_norm_sq', 'Kp') + (('JtJp',) if which == 'manifold' else ())
                        out = sv.sampling_variables(kind, frame, sample, want, **args)
                        for name, value in zip(want, out):
                            rec.put('%s/%s' % (case, name), value)
                    except Exception as exc:
                        rec.error(case, exc)


def oracle_rows(api, rec):
    want_all = ('f', 'g', 'r', 'r_block_sumsq', 'y_block_sumsq', 'f_resolution')
    for kind, order in CASES:
        for which in ('manifold', 'corewise', 'uniform'):
            for C in ((), (2,)):
                if which == 'uniform' and C:
                    continue
                for wl in WEIGHTS:
                    case = 'oracle/%s%s/%s/C%d/%s' % (kind, order, which, len(C), wl)
                    try:
                        rng, x, sample, y, block, full = _problem(23, kind, order, C)
                        frame = _frame_data(x, which)
                        b, f = _weights(wl, block, full)
                        oracle = api.oracle(kind, order)
                        batch = api.batch(kind, order, sample, y, b, f)
                        rec.put(case + '/n_measurements', np.asarray(api.n_measurements(oracle, batch)))
                        if which == 'uniform':
                            batch = api.pack(kind, order, batch, max(SHAPE))
                            rec.put(case + '/packed_sample', batch.sample)
                        want = want_all
                        extra = {}
                        if which != 'uniform':
                            d = _directions(rng, frame, which)
                            extra = dict(d)
                            want = want_all + ('Jp', 'Jp_norm_sq', 'Kp') + (('JtJp',) if which == 'manifold' else ())
                        values, held = oracle.query(frame, batch, want, **extra)
                        for name, value in zip(want, values):
                            rec.put('%s/full/%s' % (case, name), value)
                        idx = np.array([0, 2, 5])
                        sub = api.take(oracle, batch, idx)
                        rec.put(case + '/take/minibatch_scale', np.asarray(sub.minibatch_scale))
                        rec.put(case + '/take/y', sub.y)
                        values, _ = oracle.query(frame, sub, want, **extra)
                        for name, value in zip(want, values):
                            rec.put('%s/drawn/%s' % (case, name), value)
                    except Exception as exc:
                        rec.error(case, exc)


def batch_rows(api, rec):
    for kind, order in CASES:
        case = 'batch/%s%s' % (kind, order)
        rng, x, sample, y, block, full = _problem(31, kind, order)
        oracle = api.oracle(kind, order)
        good = api.batch(kind, order, sample, y, block, None)
        try:
            api.check(oracle, good, SHAPE)
            rec.put(case + '/check_good', 'ok')
        except Exception as exc:
            rec.error(case + '/check_good', exc)
        bad_y = [a[..., :-1] for a in y] if kind == 'probe' else y[..., :-1]
        try:
            api.check(oracle, api.batch(kind, order, sample, bad_y), SHAPE)
            rec.put(case + '/check_bad', 'ok')
        except Exception as exc:
            rec.error(case + '/check_bad', exc)
        if kind != 'entries':
            packed = api.pack(kind, order, good, 7)
            rec.put(case + '/pack/sample', packed.sample)
            rec.put(case + '/pack/y', packed.y)
        draw = api.flat_draw(oracle, good, 3)
        drawn = draw(np.random.default_rng(5))
        rec.put(case + '/draw/indices', drawn.drawn_indices)
        rec.put(case + '/draw/scale', np.asarray(drawn.minibatch_scale))
        rec.put(case + '/draw/y', drawn.y)


def display_rows(api, rec):
    import t3toolbox.tucker_tensor_train as t3
    import t3toolbox.backend.optimizer_display as bdisp
    from t3toolbox.backend.geometry import ManifoldGeometryOps
    np.random.seed(0)
    target = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
    rng = np.random.default_rng(3)
    ww, wv = ([rng.standard_normal((W, N)) for N in target.shape] for W in (60, 20))
    oracle = api.oracle('probe', ())
    block = 0.5 + rng.random(3)
    batch = api.batch('probe', (), ww, list(target.probe(ww)), block)
    val = api.batch('probe', (), wv, list(target.probe(wv)), block)
    cb, records = bdisp.make_newton_display(oracle, batch, val, print_fn=None)
    np.random.seed(1)
    x0 = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)).data
    x, stats = api.bopt.newton_cg(ManifoldGeometryOps(), oracle, batch, x0, max_newton=4, callback=cb)
    for i, r in enumerate(records):
        rec.put('display/%d/train_err' % i, r['train_err'])
        rec.put('display/%d/val_err' % i, r['val_err'])
    rec.put('display/x', x)


def _history(stats):
    out = {}
    for key in ('losses',):
        if key in stats:
            out[key] = np.asarray(stats[key], dtype=float)
    hist = stats.get('history')
    if hist:
        for field in ('f', 'gnorm', 'objective', 'cg_iters', 'step'):
            vals = [h.get(field) for h in hist if isinstance(h, dict)]
            if vals and all(isinstance(v, (int, float, np.floating, np.integer)) for v in vals):
                out['history_' + field] = np.asarray(vals, dtype=float)
        out['n_history'] = np.asarray(len(hist))
    for key in ('stop_reason', 'converged'):
        if key in stats:
            out[key] = str(stats[key])
    return out


def optimizer_rows(api, rec):
    import t3toolbox.tucker_tensor_train as t3
    import t3toolbox.uniform_tucker_tensor_train as ut3
    import t3toolbox.manifold as t3m
    import t3toolbox.uniform_manifold as ut3m
    import t3toolbox.optimizers as topt
    import t3toolbox.backend.jets as jets

    def run(case, fn):
        try:
            x, stats = fn()
            rec.put(case + '/dense', x.to_dense() if not isinstance(x, ut3.UniformTuckerTensorTrain)
                    else x.to_t3().to_dense())
            rec.put(case + '/stats', _history(stats))
        except Exception as exc:
            rec.error(case, exc)
            traceback.print_exc()

    np.random.seed(0)
    target = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
    rng = np.random.default_rng(7)
    ww = [rng.standard_normal((40, n)) for n in SHAPE]
    np.random.seed(2)
    x0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
    block = 0.5 + rng.random(3)
    probe = api.batch('probe', (), ww, list(target.probe(ww)), block)
    apply_ = api.batch('apply', (), ww, np.asarray(target.apply(ww)))
    run('opt/newton_probe_block', lambda: topt.newton_cg(t3m.MANIFOLD, api.oracle('probe', ()), probe, x0,
                                                           max_newton=5))
    ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)
    run('opt/newton_uniform_apply', lambda: topt.newton_cg(ut3m.UNIFORM_MANIFOLD, api.oracle('apply', ()), apply_,
                                                             ux0, max_newton=5))
    run('opt/mc_sgd_probe_block', lambda: topt.mc_sgd(t3m.MANIFOLD, api.oracle('probe', ()), probe, x0,
                                                       np.random.default_rng(1), 10, max_iter=30, check_every=5))
    run('opt/adam_apply', lambda: topt.adam(t3m.COREWISE, api.oracle('apply', ()), apply_, x0,
                                            np.random.default_rng(1), 10, max_iter=30))
    run('opt/gd_probe_block', lambda: topt.gradient_descent(t3m.MANIFOLD, api.oracle('probe', ()), probe, x0,
                                                            n_iter=5))
    wv = [rng.standard_normal((15, n)) for n in SHAPE]
    val = api.batch('probe', (), wv, list(target.probe(wv)), block)
    run('opt/newton_probe_val', lambda: topt.newton_cg(t3m.MANIFOLD, api.oracle('probe', ()), probe, x0, max_newton=3,
                                                       val_batch=val))
    po = api.oracle('probe', ())
    user_draw = lambda g: api.take(po, probe, g.choice(40, size=10, replace=False))
    run('opt/mc_sgd_user_draw', lambda: topt.mc_sgd(t3m.MANIFOLD, po, probe, x0, np.random.default_rng(1), 10,
                                                    draw=user_draw, max_iter=20, check_every=5))
    run('opt/mc_sgd_user_draw_uniform', lambda: topt.mc_sgd(ut3m.UNIFORM_MANIFOLD, po, probe, ux0,
                                                            np.random.default_rng(1), 10, draw=user_draw,
                                                            max_iter=20, check_every=5))
    # the old call form (translated): probe derivatives with a per-(mode, order) weight, and plain probe per mode
    pp = [rng.standard_normal((40, n)) for n in SHAPE]
    data = [np.stack([a, b]) for a, b in zip(target.probe(ww), target.probe(pp))]   # any derivative-shaped data
    wmat = 0.5 + rng.random((3, 2))
    run('opt/old_form_probe_derivatives', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp), data,
                                                                  x0, order=1, weight=wmat, max_newton=4))
    run('opt/old_form_probe', lambda: topt.newton_cg(t3m.MANIFOLD, 'probe', ww, list(target.probe(ww)), x0,
                                                     weight=block, max_newton=4))


def jit_rows(api, rec):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        rec.put('jit', 'no jax')
        return
    rng, x, sample, y, block, full = _problem(41, 'probe', (1,))
    frame = _frame_data(x, 'manifold')
    to_j = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
    oracle = api.oracle('probe', (1,))
    batch = to_j(api.batch('probe', (1,), sample, y, block, None))
    frame = to_j(frame)

    @jax.jit
    def q(frame, batch):
        (f, g), _ = oracle.query(frame, batch, ('f', 'g'))
        return f, g
    rec.put('jit/probe(1,)/block', q(frame, batch))


def main_record(prefix, form):
    api = _Api(form)
    rec = _Record()
    for part in (engine_rows, oracle_rows, batch_rows, display_rows, optimizer_rows, jit_rows):
        part(api, rec)
    np.savez_compressed(prefix + '.npz', **rec.arrays)
    with open(prefix + '.json', 'w') as f:
        json.dump(rec.notes, f, indent=0, sort_keys=True)
    print('%s: %d arrays, %d notes (%d raises)' % (prefix, len(rec.arrays), len(rec.notes),
                                                   sum(1 for v in rec.notes.values() if v.startswith('raises'))))


def compare(a, b, expect):
    za, zb = np.load(a + '.npz'), np.load(b + '.npz')
    na, nb = json.load(open(a + '.json')), json.load(open(b + '.json'))
    bad = []
    if set(za.files) != set(zb.files):
        bad.append('array keys differ: only in %s %s; only in %s %s' % (a, sorted(set(za.files) - set(zb.files))[:20],
                                                                        b, sorted(set(zb.files) - set(za.files))[:20]))
    if na != nb:
        diff = sorted(k for k in set(na) | set(nb) if na.get(k) != nb.get(k))
        bad.append('notes differ at %d keys, e.g. %s' % (len(diff), [(k, na.get(k), nb.get(k)) for k in diff[:10]]))
    counts = dict(bitwise=0, rounding=0, jax=0)
    for key in sorted(set(za.files) & set(zb.files)):
        x, y = za[key], zb[key]
        if x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes():
            counts['bitwise'] += 1
            continue
        if expect == 'new':
            tol = RTOL_JAX if key.startswith('jit') else (RTOL_ROUNDING if any(t in key for t in ROUNDING_TAGS)
                                                         else None)
            if tol is not None and x.shape == y.shape:
                rel = float(np.linalg.norm((x - y).ravel()) / max(np.linalg.norm(y.ravel()), 1e-300))
                if rel <= tol:
                    counts['rounding' if tol == RTOL_ROUNDING else 'jax'] += 1
                    continue
        bad.append('%s: %s %s vs %s %s' % (key, x.dtype, x.shape, y.dtype, y.shape))
    print('compared %d keys: %s; %d notes equal: %s' % (len(set(za.files) & set(zb.files)), counts, len(na), na == nb))
    for line in bad[:50]:
        print('MISMATCH', line)
    print('RESULT:', 'PASS' if not bad else 'FAIL (%d)' % len(bad))
    return not bad


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        expect = sys.argv[sys.argv.index('--expect') + 1] if '--expect' in sys.argv else 'part0'
        sys.exit(0 if compare(sys.argv[2], sys.argv[3], expect) else 1)
    main_record(sys.argv[1], sys.argv[2])
