"""R4-5: every safety.require site vs docs/numerical_contracts.md -- raises in safe mode, silent under
safety.unsafe(), silent under jit; plus set_default_safety contextvar semantics (threads)."""
import threading
import numpy as np
import jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.safety as safety

np.random.seed(3)
s = ((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))   # minimal ranks, so from_t3svd weights fit the frame
x = t3.TuckerTensorTrain.randn(*s)
frame = t3m.MANIFOLD.frame(x)
bad_frame = t3m.COREWISE.frame(x)                         # non-orthogonal
other_frame = t3m.MANIFOLD.frame(t3.TuckerTensorTrain.randn(*s))
v_g = t3m.MANIFOLD.randn(frame)                            # gauged at frame
v_r = t3m.COREWISE.randn(frame)                            # raw (ungauged) at frame
w_o = t3m.MANIFOLD.randn(other_frame)                      # at a different frame
vb = t3m.COREWISE.randn(bad_frame)                         # at a non-orthogonal frame
W = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(x))
ww = tuple(np.random.randn(N) for N in s[0])

cases = [  # (name, doc-claimed precondition, callable, expected substring in message)
    ('T3Tangent.__add__',       'SF',    lambda: v_g + w_o,                                  'different tangent spaces'),
    ('T3Tangent.__sub__',       'SF',    lambda: v_g - w_o,                                  'different tangent spaces'),
    ('T3Tangent.allclose',      'SF',    lambda: v_g.allclose(w_o),                          'different tangent spaces'),
    ('T3Tangent.corewise_inner','SF',    lambda: v_g.corewise_inner(w_o),                    'different tangent spaces'),
    ('T3Tangent.weighted_inner','SF',    lambda: v_g.weighted_inner(w_o, W),                 'different tangent spaces'),
    ('T3Tangent.stack_tangents','SF',    lambda: t3m.T3Tangent.stack_tangents((v_g, w_o)),  'same frame'),
    ('MANIFOLD.inner',          'SF',    lambda: t3m.MANIFOLD.inner(v_g, w_o),               'different tangent spaces'),
    ('MANIFOLD.inner',          'ORTH',  lambda: t3m.MANIFOLD.inner(vb, vb),                 'orthogonal frame'),
    ('MANIFOLD.inner',          'GAUGE', lambda: t3m.MANIFOLD.inner(v_g, v_r),               'gauged'),
    ('MANIFOLD.norm',           'ORTH',  lambda: t3m.MANIFOLD.norm(vb),                      'orthogonal frame'),
    ('MANIFOLD.norm',           'GAUGE', lambda: t3m.MANIFOLD.norm(v_r),                     'gauged'),
    ('COREWISE.inner',          'SF',    lambda: t3m.COREWISE.inner(v_g, w_o),               'different tangent spaces'),
    ('MANIFOLD.project',        'ORTH',  lambda: t3m.MANIFOLD.project(vb),                   'orthogonal frame'),
    ('MANIFOLD.project_oblique','ORTH',  lambda: t3m.MANIFOLD.project_oblique(vb),           'orthogonal frame'),
    ('MANIFOLD.retract',        'ORTH',  lambda: t3m.MANIFOLD.retract(vb),                   'orthogonal frame'),
    ('MANIFOLD.project_ambient(T3)',   'ORTH', lambda: t3m.MANIFOLD.project_ambient(bad_frame, x),            'orthogonal frame'),
    ('MANIFOLD.project_ambient(dense)','ORTH', lambda: t3m.MANIFOLD.project_ambient(bad_frame, x.to_dense()), 'orthogonal frame'),
    ('MANIFOLD.transport',      'ORTH',  lambda: t3m.MANIFOLD.transport(v_g, bad_frame),     'orthogonal frame'),
    ('MANIFOLD.randn',          'ORTH',  lambda: t3m.MANIFOLD.randn(bad_frame),              'orthogonal frame'),
    ('MANIFOLD.randn_like',     'ORTH',  lambda: t3m.MANIFOLD.randn_like(vb),                'orthogonal frame'),
]
must_not_raise = [
    ('COREWISE.inner on non-orth ungauged', lambda: t3m.COREWISE.inner(vb, vb)),
    ('COREWISE.norm',                       lambda: t3m.COREWISE.norm(vb)),
    ('COREWISE.retract',                    lambda: t3m.COREWISE.retract(vb)),
    ('T3Tangent.corewise_norm',             lambda: vb.corewise_norm()),
    ('T3Tangent.to_dense / to_t3',          lambda: (vb.to_dense(), vb.to_t3())),
    ('T3Tangent.probe/apply',               lambda: (vb.probe(ww), vb.apply(ww))),
    ('T3Tangent.normalized / sum_tangents', lambda: (vb.normalized(), vb.sum_tangents())),
    ('T3Tangent.weighted_norm (ungauged)',  lambda: v_r.weighted_norm(W)),
    ('MANIFOLD.frame / COREWISE.frame',     lambda: (t3m.MANIFOLD.frame(x), t3m.COREWISE.frame(x))),
    ('jit-roundtrip frame + (SF must pass)', lambda: (v_g + t3m.T3Tangent(jax.jit(lambda f: f)(frame.to_jax()), v_g.variations.to_jax()))),
]

allok = True
print(f'{"site":36s} {"pre":6s} safe-raises  unsafe-silent  jit-silent')
for name, pre, f, sub in cases:
    try:
        f(); r1 = 'NO RAISE'
    except ValueError as e:
        r1 = 'raises' if sub in str(e) else f'raises(other msg: {str(e)[:50]!r})'
    try:
        with safety.unsafe():
            f(); r2 = 'silent'
    except Exception as e:
        r2 = f'RAISES {type(e).__name__}'
    try:
        jax.jit(lambda z: (f(), z)[1])(jnp.zeros(1)); r3 = 'silent'   # trace with closed-over numpy operands
    except Exception as e:
        r3 = f'RAISES {type(e).__name__}: {str(e)[:60]}'
    good = r1 == 'raises' and r2 == 'silent' and r3 == 'silent'
    allok &= good
    print(f'{name:36s} {pre:6s} {r1:12s} {r2:14s} {r3}')
for name, f in must_not_raise:
    try:
        f(); print(f'{name:36s} {"--":6s} no-raise OK')
    except Exception as e:
        allok = False; print(f'{name:36s} {"--":6s} RAISED {type(e).__name__}: {str(e)[:80]}')

# jit with traced (jax) operands: the tangent is a pytree, checks must skip and compile
vb_j = vb.to_jax(); v_r_j = v_r.to_jax(); w_o_j = w_o.to_jax(); v_g_j = v_g.to_jax()
for name, f in [('jit MANIFOLD.norm(ungauged, non-orth)', lambda: jax.jit(lambda t: t3m.MANIFOLD.norm(t))(vb_j)),
                ('jit MANIFOLD.inner(diff frames)', lambda: jax.jit(lambda a, b: t3m.MANIFOLD.inner(a, b))(v_g_j, w_o_j)),
                ('jit MANIFOLD.project(non-orth)', lambda: jax.jit(lambda t: t3m.MANIFOLD.project(t).variations.tucker_variations[0])(vb_j)),
                ('jit __add__(diff frames)', lambda: jax.jit(lambda a, b: (a + b).variations.tucker_variations[0])(v_g_j, w_o_j))]:
    try:
        f(); print(f'{name:36s}        traced-operand jit silent OK')
    except Exception as e:
        allok = False; print(f'{name:36s}        RAISED {type(e).__name__}: {str(e)[:80]}')

print('\n--- set_default_safety: contextvar semantics ---')
safety.set_default_safety(rtol_numpy=1e-12, rtol_jax=1e-3)
print('main thread after set_default_safety:', safety.current_safety())
res = {}
t_ = threading.Thread(target=lambda: res.update(s=safety.current_safety())); t_.start(); t_.join()
print('worker thread sees:               ', res['s'], '  (== module default:', res['s'] == safety._DEFAULT, ')')
with safety.safe():
    safety.set_default_safety(rtol_numpy=1e-7, rtol_jax=1e-2)   # "script-level default" set inside a scoped block
print('after set_default_safety inside safe() block exits:', safety.current_safety(), '(the 1e-7 was discarded)')
print('can set_default_safety make UNSAFE the default?  signature only accepts tolerances ->', 'no')
print('\nALL SITES OK' if allok else '\nSOME SITE MISMATCHES')
