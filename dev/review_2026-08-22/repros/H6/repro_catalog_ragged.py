"""H6(a): catalog-vs-code table, ragged + shared + GaussNewtonModel.
For each op with a claimed precondition: violating input in safe mode (expect raise), under safety.unsafe() (expect no raise),
under jax.jit (expect no raise). Ops claimed precondition-free are run on violating inputs to confirm they do NOT raise."""
import numpy as np, jax, jax.numpy as jnp
import t3toolbox.safety as safety
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.shared_geometry as sg
import t3toolbox.fitting as fit
from t3toolbox.backend.common import tree_to_jax

def run(label, fn, expect_raise):
    def one(ctx):
        try:
            with ctx():
                fn()
            return 'pass'
        except ValueError as e:
            return 'RAISE'
        except Exception as e:
            return 'ERR:%s' % type(e).__name__
    import contextlib
    safe_r = one(contextlib.nullcontext); unsafe_r = one(safety.unsafe)
    flag = '' if (safe_r == 'RAISE') == expect_raise and unsafe_r == 'pass' else '   <-- MISMATCH vs catalog'
    print('%-52s safe=%-6s unsafe=%-6s (catalog: %s)%s' % (label, safe_r, unsafe_r, 'raise' if expect_raise else 'free', flag))

def jit_ok(label, fn, *args):
    try:
        jax.jit(fn)(*args); print('%-52s jit=pass' % label)
    except Exception as e:
        print('%-52s jit=%s: %s   <-- check not skipped under trace?' % (label, type(e).__name__, str(e)[:80]))

np.random.seed(0)
shape, tr, ttr = (4, 5, 3), (2, 3, 2), (1, 2, 3, 1)
x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
fr = t3m.MANIFOLD.frame(x)                 # orthogonal
fr2 = bvf.T3Frame.random_orthogonal_like(fr)  # a DIFFERENT frame, same structure
nonorth = t3m.COREWISE.frame(x)            # (U,G,G,G): not orthogonal
v1, v2 = t3m.MANIFOLD.randn(fr), t3m.MANIFOLD.randn(fr)
v_other = t3m.MANIFOLD.randn(fr2)
raw = t3m.T3Tangent(fr, bvf.T3Variations.randn(fr.variation_shapes, (), False))  # ungauged at orth frame
raw_no = t3m.T3Tangent(nonorth, bvf.T3Variations.randn(nonorth.variation_shapes, (), False))
M = t3m.MANIFOLD; CW = t3m.COREWISE

print('== T3Tangent (SF ops)')
run('v1 + v_other (SF)', lambda: v1 + v_other, True)
run('v1 - v_other (SF)', lambda: v1 - v_other, True)
run('v1.corewise_inner(v_other) (SF)', lambda: v1.corewise_inner(v_other), True)
run('v1.allclose(v_other) (SF)', lambda: v1.allclose(v_other), True)
run('stack_tangents([v1, v_other]) (SF)', lambda: t3m.T3Tangent.stack_tangents([v1, v_other]), True)
run('v1 * 2, -v1, normalized, corewise_norm (free)', lambda: ((raw_no * 2), -raw_no, raw_no.normalized(), raw_no.corewise_norm()), False)
run('to_dense / to_t3 on non-orth frame (free)', lambda: (raw_no.to_dense(), raw_no.to_t3()), False)
run('probe/apply on non-orth ungauged (free)', lambda: (raw_no.apply([np.ones(n) for n in shape]),), False)
run('stack_frame([v1, v_other]) (free)', lambda: t3m.T3Tangent.stack_frame([v1, v_other]), False)

print('== MANIFOLD')
run('inner: different frames (SF)', lambda: M.inner(v1, v_other), True)
run('inner: orth frame, t2 ungauged (GAUGE)', lambda: M.inner(v1, raw), True)
run('inner: non-orth frame (ORTH)', lambda: M.inner(raw_no, raw_no), True)
run('norm: ungauged (GAUGE)', lambda: M.norm(raw), True)
run('norm: non-orth (ORTH)', lambda: M.norm(raw_no), True)
run('project: non-orth (ORTH)', lambda: M.project(raw_no), True)
run('project_oblique: non-orth (ORTH)', lambda: M.project_oblique(raw_no), True)
run('retract: non-orth (ORTH)', lambda: M.retract(raw_no), True)
run('retract: orth but UNGAUGED (gauge-invariant: free)', lambda: M.retract(raw), False)
run('project_ambient: non-orth (ORTH)', lambda: M.project_ambient(nonorth, x), True)
run('transport: non-orth new_frame (ORTH)', lambda: M.transport(v1, nonorth), True)
run('randn: non-orth (ORTH)', lambda: M.randn(nonorth), True)
run('randn_like: non-orth (ORTH)', lambda: M.randn_like(raw_no), True)
run('frame(x) (free)', lambda: M.frame(x), False)
print('== COREWISE')
run('COREWISE.inner different frames (SF)', lambda: CW.inner(v1, v_other), True)
run('COREWISE.norm ungauged/non-orth (free)', lambda: CW.norm(raw_no), False)
run('COREWISE.project/retract/randn (free)', lambda: (CW.project(raw_no), CW.retract(raw_no), CW.randn(nonorth)), False)

print('== GaussNewtonModel (SF on p)')
ww = [np.random.randn(6, n) for n in shape]
m = fit.apply_model(M, x, ww, x.apply(ww))
p_bad = M.randn(bvf.T3Frame.random_orthogonal_like(m.frame))
run('jacobian(p at other frame)', lambda: m.jacobian(p_bad), True)
run('gn_hessian(p at other frame)', lambda: m.gn_hessian(p_bad), True)
run('gn_quadratic(p at other frame)', lambda: m.gn_quadratic(p_bad), True)
run('evaluate(p at other frame)', lambda: m.evaluate(p_bad), True)
p_ok = M.randn(m.frame)
run('gn_hessian(p at SAME frame, value-equal copy)', lambda: m.gn_hessian(t3m.T3Tangent(bvf.T3Frame(*[tuple(np.array(c) for c in fam) for fam in m.frame.data]), p_ok.variations)), False)

print('== Sharing (TIED)')
xs = t3.TuckerTensorTrain.randn((4, 4, 3), (2, 2, 2), (1, 2, 2, 1))   # untied
sh = (0, 0, 1)
run('t3svd(sharing) untied', lambda: xs.t3svd(sharing=sh), True)
run('rank_adjustment_sweep(sharing) untied', lambda: xs.rank_adjustment_sweep(sharing=sh), True)
run('resize(sharing) untied', lambda: xs.resize(xs.shape, (2, 2, 2), (1, 2, 2, 1), sharing=sh), True)
run('continuation_ranks(sharing) untied', lambda: xs.continuation_ranks(sharing=sh), True)
run('get_minimal_ranks(sharing) untied (free)', lambda: xs.get_minimal_ranks(sharing=sh), False)
run('share(sharing) untied (free)', lambda: xs.share(sh), False)
SM = sg.shared_manifold(sh)
run('shared.frame(untied x)  [doc: TIED]', lambda: SM.frame(xs), True)
xt = xs.share(sh); frt = SM.frame(xt); vt = SM.randn(frt)
fr_untied = M.frame(xs)
run('shared.transport(v, untied new_frame) (TIED)', lambda: SM.transport(vt, fr_untied), True)
run('shared.retract(untied-frame tangent) (TIED frame)', lambda: SM.retract(M.randn(fr_untied)), True)
run('shared.retract(tied frame, UNTIED tangent) (TIED tan)', lambda: SM.retract(M.randn(frt)), True)
run('shared.retract(tied tangent) (ok)', lambda: SM.retract(vt), False)
run('shared.project(non-orth) (ORTH)', lambda: SM.project(t3m.T3Tangent(CW.frame(xt), vt.variations)), True)
run('shared.project(untied frame, orth) (free)', lambda: SM.project(M.randn(fr_untied)), False)
run('shared.project_ambient(non-orth) (ORTH)', lambda: SM.project_ambient(CW.frame(xt), xt), True)
SC = sg.shared_corewise(sh)
run('shared_corewise.retract untied (free)', lambda: SC.retract(CW.randn(CW.frame(xs))), False)

print('== under jit (checks must skip)')
v1j = jax.tree_util.tree_map(jnp.asarray, v1); v_oj = jax.tree_util.tree_map(jnp.asarray, v_other)
rawj = jax.tree_util.tree_map(jnp.asarray, raw); raw_noj = jax.tree_util.tree_map(jnp.asarray, raw_no)
jit_ok('jit: v1 + v_other', lambda a, b: (a + b).variations.data, v1j, v_oj)
jit_ok('jit: MANIFOLD.inner different frames', lambda a, b: M.inner(a, b), v1j, v_oj)
jit_ok('jit: MANIFOLD.norm ungauged', lambda a: M.norm(a), rawj)
jit_ok('jit: MANIFOLD.project non-orth', lambda a: M.project(a).variations.data, raw_noj)
jit_ok('jit: MANIFOLD.retract non-orth', lambda a: M.retract(a).data, raw_noj)
xsj = jax.tree_util.tree_map(jnp.asarray, xs)
jit_ok('jit: t3svd(sharing) untied', lambda a: a.t3svd(sharing=sh, max_tucker_ranks=2, max_tt_ranks=(1,2,2,1))[0].data, xsj)
jit_ok('jit: shared.retract untied tangent', lambda a: SM.retract(a).data, jax.tree_util.tree_map(jnp.asarray, M.randn(frt)))
mj = fit.apply_model(M, jax.tree_util.tree_map(jnp.asarray, x), [jnp.asarray(w) for w in ww], jnp.asarray(x.apply(ww)))
jit_ok('jit: gn_hessian p at other frame', lambda mm, p: mm.gn_hessian(p).variations.data, mj, jax.tree_util.tree_map(jnp.asarray, p_bad))

print('== jax EAGER float32: GAUGE false failure at modest scale (rtol_jax=1e-5 vs absolute residual)')
frj = bvf.T3Frame(*tree_to_jax(fr.data)); vj = M.randn(frj)
for s in [1.0, 10.0, 100.0, 1e3, 1e4]:
    w = vj * s
    try:
        M.norm(w); r = 'OK'
    except ValueError as e:
        r = 'RAISES (gauge)'
    print('  jax eager scale %6.0e: gauge_residual=%.2e rel=%.1e -> MANIFOLD.norm %s' % (s, float(w.gauge_residual), float(w.gauge_residual) / float(w.corewise_norm()), r))
