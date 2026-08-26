"""H6(a): catalog-vs-code table, UNIFORM mirror (UT3Tangent, UNIFORM_MANIFOLD/COREWISE, uniform sharing, shared uniform geometry)."""
import numpy as np, jax, jax.numpy as jnp, contextlib
import t3toolbox.safety as safety
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as um
import t3toolbox.shared_geometry as sg
import t3toolbox.fitting as fit

def run(label, fn, expect_raise):
    def one(ctx):
        try:
            with ctx(): fn()
            return 'pass'
        except ValueError: return 'RAISE'
        except Exception as e: return 'ERR:%s' % type(e).__name__
    s, u = one(contextlib.nullcontext), one(safety.unsafe)
    flag = '' if (s == 'RAISE') == expect_raise and u == 'pass' else '   <-- MISMATCH vs catalog'
    print('%-54s safe=%-6s unsafe=%-6s (catalog: %s)%s' % (label, s, u, 'raise' if expect_raise else 'free', flag))
def jit_ok(label, fn, *args):
    try: jax.jit(fn)(*args); print('%-54s jit=pass' % label)
    except Exception as e: print('%-54s jit=%s: %s   <-- check not skipped?' % (label, type(e).__name__, str(e)[:80]))

np.random.seed(0)
shape, tr, ttr = (4, 5, 3), (2, 3, 2), (1, 2, 2, 1)
x = t3.TuckerTensorTrain.randn(shape, tr, ttr); ux = ut3.UniformTuckerTensorTrain.from_t3(x)
UM, UC = um.UNIFORM_MANIFOLD, um.UNIFORM_COREWISE
fr = UM.frame(ux); fr2 = ubv.UT3Frame.from_t3frame(bvf.T3Frame.random_orthogonal_like(fr.to_t3frame())); nonorth = UC.frame(ux)
v1, v2, v_other = UM.randn(fr), UM.randn(fr), UM.randn(fr2)
raw = um.UT3Tangent.from_t3tangent(t3m.T3Tangent(fr.to_t3frame(), bvf.T3Variations.randn(fr.to_t3frame().variation_shapes, (), False)))
raw = um.UT3Tangent(fr, raw.variations)     # ungauged at the orthogonal uniform frame
raw_no = UC.randn(nonorth)
print('== UT3Tangent')
run('v1 + v_other (SF)', lambda: v1 + v_other, True)
run('v1 - v_other (SF)', lambda: v1 - v_other, True)
run('corewise_inner different frames (SF)', lambda: v1.corewise_inner(v_other), True)
run('allclose different frames (SF)', lambda: v1.allclose(v_other), True)
run('stack_tangents different frames (SF)', lambda: um.UT3Tangent.stack_tangents([v1, v_other]), True)
run('scale/neg/normalized/corewise_norm/to_dense (free)', lambda: (raw_no * 2, -raw_no, raw_no.normalized(), raw_no.corewise_norm(), raw_no.to_dense()), False)
print('== UNIFORM_MANIFOLD')
run('inner different frames (SF)', lambda: UM.inner(v1, v_other), True)
run('inner t2 ungauged (GAUGE)', lambda: UM.inner(v1, raw), True)
run('inner non-orth (ORTH)', lambda: UM.inner(raw_no, raw_no), True)
run('norm ungauged (GAUGE)', lambda: UM.norm(raw), True)
run('norm non-orth (ORTH)', lambda: UM.norm(raw_no), True)
run('project non-orth (ORTH)', lambda: UM.project(raw_no), True)
run('project_oblique non-orth (ORTH)', lambda: UM.project_oblique(raw_no), True)
run('retract non-orth (ORTH)', lambda: UM.retract(raw_no), True)
run('retract orth ungauged (free)', lambda: UM.retract(raw), False)
run('project_ambient non-orth (ORTH)', lambda: UM.project_ambient(nonorth, ux), True)
run('transport non-orth new_frame (ORTH)', lambda: UM.transport(v1, nonorth), True)
run('randn non-orth (ORTH)', lambda: UM.randn(nonorth), True)
run('randn_like non-orth (ORTH)', lambda: UM.randn_like(raw_no), True)
print('== UNIFORM_COREWISE')
run('UC.inner different frames (SF)', lambda: UC.inner(v1, v_other), True)
run('UC.norm / project / retract (free)', lambda: (UC.norm(raw_no), UC.project(raw_no), UC.retract(raw_no)), False)
print('== uniform GaussNewtonModel (SF)')
ww = [np.random.randn(6, n) for n in shape]
m = fit.apply_model(UM, ux, ww, np.asarray(ux.apply(ww)))
p_bad = UM.randn(ubv.UT3Frame.from_t3frame(bvf.T3Frame.random_orthogonal_like(m.frame.to_t3frame())))
run('jacobian(p other frame)', lambda: m.jacobian(p_bad), True)
run('gn_hessian(p other frame)', lambda: m.gn_hessian(p_bad), True)
run('evaluate(p other frame)', lambda: m.evaluate(p_bad), True)
print('== uniform sharing (TIED)')
xs = t3.TuckerTensorTrain.randn((4, 4, 3), (2, 2, 2), (1, 2, 2, 1)); sh = (0, 0, 1); uxs = ut3.UniformTuckerTensorTrain.from_t3(xs)
run('ut3svd(sharing) untied', lambda: uxs.t3svd(sharing=sh), True)
run('rank_adjustment_sweep(sharing) untied', lambda: uxs.rank_adjustment_sweep(sharing=sh), True)
USM = sg.shared(UM, sh); uxt = ut3.UniformTuckerTensorTrain.from_t3(xs.share(sh)); frt = USM.frame(uxt); vt = USM.randn(frt)
fr_untied = UM.frame(uxs)
run('shared.frame(untied) [doc: TIED]', lambda: USM.frame(uxs), True)
run('shared.transport(v, untied new_frame) (TIED)', lambda: USM.transport(vt, fr_untied), True)
run('shared.retract(untied-frame tangent) (TIED frame)', lambda: USM.retract(UM.randn(fr_untied)), True)
run('shared.retract(tied frame, UNTIED tangent)', lambda: USM.retract(UM.randn(frt)), True)
run('shared.retract(tied tangent) (ok)', lambda: USM.retract(vt), False)
run('shared.project(non-orth) (ORTH)', lambda: USM.project(UC.randn(UC.frame(uxt))), True)
run('shared.project_ambient(non-orth) (ORTH)', lambda: USM.project_ambient(UC.frame(uxt), uxt), True)
print('== jit')
tm = lambda o: jax.tree_util.tree_map(jnp.asarray, o)
jit_ok('jit: v1 + v_other', lambda a, b: (a + b).variations.data[:2], tm(v1), tm(v_other))
jit_ok('jit: UM.norm ungauged', lambda a: UM.norm(a), tm(raw))
jit_ok('jit: UM.project non-orth', lambda a: UM.project(a).variations.data[:2], tm(raw_no))
jit_ok('jit: UM.retract non-orth', lambda a: UM.retract(a).data[:2], tm(raw_no))
jit_ok('jit: ut3svd(sharing) untied', lambda a: a.t3svd(sharing=sh)[0].data[:2], tm(uxs))
jit_ok('jit: shared.retract untied tangent', lambda a: USM.retract(a).data[:2], tm(UM.randn(frt)))
