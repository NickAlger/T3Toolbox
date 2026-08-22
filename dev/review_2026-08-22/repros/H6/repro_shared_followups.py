"""H6 follow-ups: (1) shared.retract with an UNTIED tangent at a tied frame -- does the TIED-tangent check fire?
(2) shared.project on a non-orth frame of the right structure; (3) silent broadcast of different-structure tangents in unsafe mode;
(4) same-frame guard on STACKED frames differing in one element; (5) get_minimal_ranks(sharing=)."""
import numpy as np
import t3toolbox.safety as safety
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.shared_geometry as sg
import t3toolbox.backend.sharing as bsh
np.random.seed(0)
M, CW = t3m.MANIFOLD, t3m.COREWISE
xs = t3.TuckerTensorTrain.randn((4, 4, 3), (2, 2, 2), (1, 2, 2, 1)); sh = (0, 0, 1)
SM = sg.shared_manifold(sh); xt = xs.share(sh); frt = SM.frame(xt)
sd = SM.shared_frame_data(frt)
v_untied = M.randn(frt); v_tied = SM.randn(frt)
print('(1) tied-variations residual: M.randn(frt) =', float(bsh.fv_tied_variations_residual(v_untied.variations.data, sd)),
      '; SM.randn(frt) =', float(bsh.fv_tied_variations_residual(v_tied.variations.data, sd)))
print('    tucker_variations[0] is tucker_variations[1]?', v_untied.variations.tucker_variations[0] is v_untied.variations.tucker_variations[1],
      '')
try:
    SM.retract(v_untied); print('    SM.retract(untied tangent) -> NO RAISE')
except ValueError as e:
    print('    SM.retract(untied tangent) -> RAISE', str(e)[:60])
# is the projected-then-retracted point different from the raw-retracted one? (does the 'silent tie' matter)
y_raw = SM.retract(v_untied); y_proj = SM.retract(SM.project(v_untied))
print('    retract(untied) vs retract(project(untied)) dense diff:', float(np.linalg.norm(y_raw.to_dense() - y_proj.to_dense())))
# how does the residual behave for a visibly untied variation: zero mode-1 variation
V, H = v_untied.variations.data
V2 = (V[0], np.zeros_like(V[1]), V[2])
r2 = float(bsh.fv_tied_variations_residual((V2, H), sd)); print('    residual with V1 := 0 (clearly untied):', r2)
print('    group spectrum svd_s:', [np.round(s, 3) for s in sd.svd_s] if hasattr(sd, 'svd_s') else 'n/a')

print('(2) shared.project on non-orth frame, same structure')
nonorth = CW.frame(xt)
raw_no = t3m.T3Tangent(nonorth, bvf.T3Variations.randn(nonorth.variation_shapes, (), False))
for ctx, name in ((__import__('contextlib').nullcontext, 'safe'), (safety.unsafe, 'unsafe')):
    try:
        with ctx(): SM.project(raw_no); print('    %s: pass' % name)
    except Exception as e:
        print('    %s: %s %s' % (name, type(e).__name__, str(e)[:70]))

print('(3) different-structure tangents with BROADCASTABLE holes, unsafe mode')
frA = bvf.T3Frame.random_orthogonal((4, 5, 3), (2, 2, 2), (1, 2, 2, 1))
frB = bvf.T3Frame.random_orthogonal((4, 5, 3), (1, 1, 1), (1, 1, 1, 1))
a, b = M.randn(frA), M.randn(frB)
print('    holes A:', frA.variation_shapes, '\n    holes B:', frB.variation_shapes)
with safety.unsafe():
    try:
        c = a + b; print('    unsafe a+b -> NO ERROR, result structure = A\'s; corewise_inner(a,b) =', float(a.corewise_inner(b)))
    except Exception as e:
        print('    unsafe a+b ->', type(e).__name__, str(e)[:80])

print('(4) same-frame guard on stacked frames differing in ONE stack element')
frS = bvf.T3Frame.random_orthogonal((4, 5, 3), (2, 2, 2), (1, 2, 2, 1), stack_shape=(3,))
U, D, L, R = frS.data
U2 = list(np.array(u) for u in U); U2[0][1] *= -1.0     # flip sign of element 1's first up core (still orthogonal)
frS2 = bvf.T3Frame(tuple(U2), D, L, R)
s1, s2 = M.randn(frS), M.randn(frS2)
try:
    s1 + s2; print('    stacked a+b with one differing element -> NO RAISE  <-- guard misses per-element difference?')
except ValueError as e:
    print('    stacked a+b with one differing element -> RAISE (good)')

print('(5) get_minimal_ranks(sharing=)')
try:
    print('   ', t3.TuckerTensorTrain.get_minimal_ranks(xs.shape, xs.tucker_ranks, xs.tt_ranks, sharing=sh))
except Exception as e:
    print('   ', type(e).__name__, e)

print('(6) TIED-tangent check on other structures: is M.randn(tied frame) ever untied?')
for shape, tr, ttr in [((5, 5, 4), (2, 2, 2), (1, 2, 2, 1)), ((6, 6, 3, 3), (3, 3, 2, 2), (1, 3, 4, 2, 1)), ((4, 4), (2, 2), (1, 2, 1))]:
    sh2 = (0, 0) + (1,) * (len(shape) - 2) if len(shape) > 2 else (0, 0)
    if len(shape) == 4: sh2 = (0, 0, 1, 1)
    x2 = t3.TuckerTensorTrain.randn(shape, tr, ttr).share(sh2); S2 = sg.shared_manifold(sh2); f2 = S2.frame(x2); sd2 = S2.shared_frame_data(f2)
    u = M.randn(f2)
    res = float(bsh.fv_tied_variations_residual(u.variations.data, sd2))
    Vv, Hh = u.variations.data; Vz = tuple(np.zeros_like(Vv[i]) if i == 1 else Vv[i] for i in range(len(Vv)))
    resz = float(bsh.fv_tied_variations_residual((Vz, Hh), sd2))
    try:
        S2.retract(t3m.T3Tangent(f2, bvf.T3Variations(Vz, Hh))); rz = 'NO RAISE'
    except ValueError: rz = 'RAISE'
    print('    %s tr=%s ttr=%s sharing=%s: residual(M.randn)=%.1e  residual(V1:=0)=%.1e -> retract(V1:=0) %s ; tangent dims: full=%d shared=%d'
          % (shape, tr, ttr, sh2, res, resz, rz, u.tangent_space_dimension, t3m.manifold_dim((shape, tr, ttr), sharing=sh2) if 'sharing' in t3m.manifold_dim.__code__.co_varnames else -1))
