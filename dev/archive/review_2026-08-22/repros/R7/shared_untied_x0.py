"""R7: SharedGeometry.frame ties an untied x silently (docs claim a safe-mode check); the backend geometry the
optimizers use does NOT tie -- what happens to an untied x0 under topt.newton_cg(shared_*)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt
import t3toolbox.fitting as fitting
import t3toolbox.shared_geometry as sg
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.sharing as bsh
import t3toolbox.corewise as cw

np.random.seed(0)
shape, tr, rr = (5, 5, 4), (2, 2, 2), (1, 2, 2, 1)
sharing = (0, 0, 1)
A = t3.TuckerTensorTrain.randn(shape, tr, rr).share(sharing)
ww = [np.random.randn(150, N) for N in shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
b = A.apply(ww)
x_untied = t3.TuckerTensorTrain.randn(shape, tr, rr) * 0.3   # NOT tied
print("x_untied.has_shared_tucker_factors:", bool(np.all(x_untied.has_shared_tucker_factors(sharing))))

# 1. frontend SharedGeometry.frame in SAFE mode: docs (class docstring l.85-87, sharing.md l.250-252, doctest l.100,
#    fitting.py l.467 comment) say tied factors are checked at `frame`; the method docstring says it ties silently.
for gm in (sg.shared_manifold(sharing), sg.shared_corewise(sharing)):
    try:
        fr = gm.frame(x_untied)
        print("%s.frame(untied x) in safe mode -> NO error; frame factors tied: %s" % (gm, bool(np.all(
            t3.TuckerTensorTrain(fr.up_tucker_cores, fr.left_tt_cores).has_shared_tucker_factors(sharing)))))
    except Exception as e:
        print("%s.frame(untied x) ->" % gm, type(e).__name__, str(e)[:80])

# 2. backend geometry (what the optimizers use): frame does not tie
for g in (bgeo.ManifoldGeometryOps().with_sharing(sharing, shape), bgeo.CorewiseGeometryOps().with_sharing(sharing, shape)):
    fr = g.frame(x_untied.data)
    tied = bool(np.all(t3.TuckerTensorTrain(*g.base_point(fr)).has_shared_tucker_factors(sharing)))
    print("backend %s.frame(untied).base_point tied: %s" % (type(g).__name__, tied))

# 3. through the optimizer, untied x0 vs the mean-tied x0: does it raise / stay tied / converge the same?
x_tied = t3.TuckerTensorTrain(*bsh.t3_tie_tucker_factors(x_untied.data, sharing))
for name, gm in (("shared_manifold", sg.shared_manifold(sharing)), ("shared_corewise", sg.shared_corewise(sharing))):
    res = {}
    for lab, xs in (("untied x0", x_untied), ("tied x0", x_tied)):
        try:
            x, st = topt.newton_cg(gm, 'apply', ww, b, xs, max_newton=2, gtol_rel=1e-12)
            tied = bool(np.all(x.has_shared_tucker_factors(sharing)))
            err = np.linalg.norm(x.to_dense() - A.to_dense()) / np.linalg.norm(A.to_dense())
            res[lab] = x.to_dense()
            print("  %-16s %-10s -> ok; iterate tied=%s  relerr=%.2e  history[0].objective=%.4f" % (name, lab, tied, err, st['history'][0]['objective']))
        except Exception as e:
            print("  %-16s %-10s -> %s: %s" % (name, lab, type(e).__name__, str(e)[:90]))
    if len(res) == 2:
        print("     untied-vs-tied start, after 2 Newton steps: rel diff = %.2e" % (np.linalg.norm(res['untied x0'] - res['tied x0']) / np.linalg.norm(res['tied x0'])))

# 4. corewise retract on an UNTIED base: frontend ties the SUM (mean); backend aliases member 0 of group
gc = bgeo.CorewiseGeometryOps().with_sharing(sharing, shape)
fr_b = gc.frame(x_untied.data)
v = t3m.COREWISE.randn(t3m.COREWISE.frame(x_untied)).variations.data
back = gc.retract(fr_b, v)
front = sg.shared_corewise(sharing).retract(t3m.T3Tangent(t3m.COREWISE.frame(x_untied), __import__('t3toolbox.frame_variations_format', fromlist=['x']).T3Variations(*v)))
U0, U1 = x_untied.data[0][0], x_untied.data[0][1]
print("\ncorewise shared retract on an UNTIED base point:")
print("  backend result U[0] is U[1]: %s ; equals x.U[0]+mean(V): %s ; equals mean(x.U)+mean(V): %s" % (
    back[0][0] is back[0][1],
    np.allclose(back[0][0], U0 + 0.5 * (v[0][0] + v[0][1])),
    np.allclose(back[0][0], 0.5 * (U0 + U1) + 0.5 * (v[0][0] + v[0][1]))))
print("  frontend result U[0] equals mean(x.U)+mean(V): %s" % np.allclose(front.data[0][0], 0.5 * (U0 + U1) + 0.5 * (v[0][0] + v[0][1])))
print("  frontend vs backend dense rel diff: %.2e" % (np.linalg.norm(front.to_dense() - t3.TuckerTensorTrain(*back).to_dense()) / np.linalg.norm(front.to_dense())))

# 5. on a TIED base point the four SharedGeometry.retract branches must match the backend twins bit-for-bit
import t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as um
import t3toolbox.backend.uniform_fitting as uf
print("\nretract branches vs backend twins (TIED base):")
for base_name, base, bg in (("MANIFOLD", t3m.MANIFOLD, bgeo.ManifoldGeometryOps().with_sharing(sharing, shape)),
                            ("COREWISE", t3m.COREWISE, bgeo.CorewiseGeometryOps().with_sharing(sharing, shape))):
    gm = sg.shared(base, sharing)
    frF = gm.frame(x_tied); auxF = gm.precompute(frF)
    vF = gm.project(base.randn(frF), shared_data=auxF) if base is t3m.MANIFOLD else gm.project(base.randn(frF))
    yF = gm.retract(vF, shared_data=auxF)
    frB = bg.frame(x_tied.data); auxB = bg.precompute(frB)
    yB = bg.retract(frB, vF.variations.data, aux=auxB)
    diff = max(float(np.max(np.abs(a - b_))) for a, b_ in zip(yF.data[0] + yF.data[1], yB[0] + yB[1]))
    print("  ragged %-8s frontend vs backend retract max abs core diff: %.1e" % (base_name, diff))
ux = ut3.UniformTuckerTensorTrain.from_t3(x_tied)
uxm = uf.uniform_minimal(ux, sharing=sharing)
for base_name, base, cls in (("UNIFORM_MANIFOLD", um.UNIFORM_MANIFOLD, bgeo.UniformManifoldGeometryOps),
                             ("UNIFORM_COREWISE", um.UNIFORM_COREWISE, bgeo.UniformCorewiseGeometryOps)):
    gm = sg.shared(base, sharing)
    frF = gm.frame(uxm); auxF = gm.precompute(frF)
    vF = gm.project(base.randn(frF), shared_data=auxF) if base is um.UNIFORM_MANIFOLD else gm.project(base.randn(frF))
    yF = gm.retract(vF, shared_data=auxF)
    bg = cls.from_point(uxm.data, sharing)
    frB = bg.frame((uxm.tucker_supercore, uxm.tt_supercore)); auxB = bg.precompute(frB)
    yB = bg.retract(frB, (vF.variations.data[0], vF.variations.data[1]), aux=auxB)
    dF = yF.to_t3().to_dense(); dB = ut3.UniformTuckerTensorTrain(yB[0], yB[1], uxm.shape, uxm.masks).to_t3().to_dense()
    print("  %-18s frontend vs backend retract dense rel diff: %.1e" % (base_name, np.linalg.norm(dF - dB) / np.linalg.norm(dF)))
