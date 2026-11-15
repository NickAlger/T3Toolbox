import sys, importlib.util, traceback
import numpy as np
sys.path.append('/home/claude/repo/T3Toolbox')
spec = importlib.util.spec_from_file_location('pm', '/home/claude/graded_messages_handoff/scripts/plan_mockup.py')
pm = importlib.util.module_from_spec(spec); sys.modules['pm'] = pm; spec.loader.exec_module(pm)
x, w, y, frame, centers, p, plan_fill, _ = pm.user_example()
n = 300
ws, ys = [a[:n] for a in w], [a[:n] for a in y]
plan = pm.Plan(samples=64)   # 300 not divisible by 64
flat = lambda v: np.concatenate([np.ravel(a) for part in v for a in part])
rel = lambda a, b: np.linalg.norm(flat(a) - flat(b)) / np.linalg.norm(flat(b))
def attempt(label, f):
    try:
        out = f(); print('OK   ', label, '->', out if not isinstance(out, dict) else sorted(out))
        return out
    except Exception as e:
        print('FAIL ', label, '->', type(e).__name__, str(e)[:150])
full = pm.evaluate(frame, centers, ws, plan, want=pm.POINT + pm.POINT_DATA + ('f', 'g'), y=ys)
g0 = full['g']
# (a) gradient from have = FRAME + r, without y  ("today's behaviour")
have = {k: full[k] for k in pm.FRAME + ('r',)}
attempt('g from have=FRAME+r, no y', lambda: pm.evaluate(frame, centers, ws, plan, want=('g',), have=have))
out = attempt('g from have=FRAME+r, with y', lambda: pm.evaluate(frame, centers, ws, plan, want=('g',), have=have, y=ys))
# (b) y different from the y that made r: which wins?
y2 = [a + 1.0 for a in ys]
g_b = pm.evaluate(frame, centers, ws, plan, want=('g',), have=have, y=y2)['g']
g_c = pm.evaluate(frame, centers, ws, plan, want=('g', 'f'), have=have, y=y2)
print('     have r (from y) + different y2: g equals g(y)? %.1e ; f uses have r: f=%.6e vs f(y)=%.6e' % (rel(g_b, g0), g_c['f'], full['f']))
# (c) unknown names
attempt('typo in have (xis)', lambda: pm.evaluate(frame, centers, ws, plan, want=('f',), y=ys, have={'xis': full['xi']}))
attempt('unknown want', lambda: pm.evaluate(frame, centers, ws, plan, want=('mu_tz',), y=ys))
attempt('want Hp without p', lambda: pm.evaluate(frame, centers, ws, plan, want=('Hp',)))
# (d) stale have from another point: silently wrong
x2 = pm.random_point((60, 50, 70, 40), (6, 7, 6, 5), (1, 8, 9, 8, 1))
frame2, centers2 = pm.orthogonal_frame(x2)
try:
    g_stale = pm.evaluate(frame2, centers2, ws, plan, want=('g',), y=ys, have={'xi': full['xi'], 'mu': full['mu']})['g']
    g_true = pm.evaluate(frame2, centers2, ws, plan, want=('g',), y=ys)['g']
    print('     stale have from another point: no error; relative error of g %.1e' % rel(g_stale, g_true))
except Exception as e:
    print('     stale have ->', type(e).__name__, e)
# (e) have for a different number of samples
attempt('have with wrong W (first 100 samples)', lambda: pm.evaluate(frame, centers, ws, plan, want=('f',), y=ys, have={'xi': [a[:100] for a in full['xi']]}))
# (f) centers inconsistent with frame: silently wrong objective, right gradient?
Vc, Hc = centers
bad = ([1.001 * v for v in Vc], Hc)
fb = pm.evaluate(frame, bad, ws, plan, want=('f', 'g'), y=ys)
print('     centers scaled by 1.001: f changes by %.1e relative, g by %.1e; no error raised' % (abs(fb['f'] - full['f']) / full['f'], rel(fb['g'], g0)))
# (g) mapped outputs that are wanted and also handed in
o = pm.evaluate(frame, centers, ws, plan, want=('xi', 'f'), y=ys, have={'xi': full['xi']})
print('     want xi with xi in have: returned object is the handed-in list?', o['xi'] is full['xi'], '| arrays identical objects?', all(a is b for a, b in zip(o['xi'], full['xi'])))
# (h) wanting direction components and z
attempt('want z, Jp, Jp.xi_z, eta_t', lambda: pm.evaluate(frame, centers, ws, plan, want=('z', 'Jp', 'Jp.xi_z', 'eta_t'), p=p))
attempt('Plan(samples=0)', lambda: pm.evaluate(frame, centers, ws, pm.Plan(samples=0), want=('f',), y=ys))
attempt("Plan(path='typo')", lambda: pm.evaluate(frame, centers, ws, pm.Plan(path='typo'), want=('f',), y=ys))
attempt("Plan(orders='nonsense', executor='shard_map')", lambda: pm.evaluate(frame, centers, ws, pm.Plan(orders='nonsense', executor='shard_map'), want=('f',), y=ys))
# (i) does evaluate project p / g?  p not gauged
praw = pm.tree_map(lambda a: np.random.default_rng(5).standard_normal(a.shape), pm.zero_tangent(frame))
H1 = pm.evaluate(frame, centers, ws, plan, want=('Hp',), p=praw)['Hp']
H2 = pm.evaluate(frame, centers, ws, plan, want=('Hp',), p=pm.project(frame, praw))['Hp']
print('     Hp(raw p) vs Hp(projected p): %.1e  (evaluate projects p itself)' % rel(H1, H2))
# (j) Jp wanted: is it J applied to projected p?
