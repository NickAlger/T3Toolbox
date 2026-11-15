import sys, importlib.util, tracemalloc, math
import numpy as np
sys.path.append('/home/claude/repo/T3Toolbox')
spec = importlib.util.spec_from_file_location('pm', '/home/claude/graded_messages_handoff/scripts/plan_mockup.py')
pm = importlib.util.module_from_spec(spec); sys.modules['pm'] = pm; spec.loader.exec_module(pm)
x, w, y, frame, centers, p, plan_fill, _ = pm.user_example()
W = w[0].shape[0]; mb = lambda b: b / 2**20
print('fill plan:', plan_fill.samples)

def peak_of(call):
    tracemalloc.start(); out = call(); pk = tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return pk, out

# (1) replicate fill's affine model and its prediction
def probe(b, keep=pm.INPUTS):
    ws = [a[:2 * b] for a in w]
    plan = pm.Plan(samples=b)
    have = pm.evaluate(frame, centers, ws, plan, want=tuple(keep))
    pz = pm.zero_tangent(frame)
    tracemalloc.start(); pm.evaluate(frame, centers, ws, plan, want=('Hp',), p=pz, have=have)
    pk = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return pk, pm.tree_bytes(have) / (2 * b)
(m1, _), (m2, kps) = probe(256), probe(512)
slope = (m2 - m1) / 256; fixed = m1 - slope * 256
budget = 8 * 2**20
room = 0.9 * budget - kps * W - fixed
print('slope B/sample %.1f fixed %.0f kept/sample %.1f room/slope %.1f' % (slope, fixed, kps, room / slope))
pred = fixed + slope * 2048
have = pm.evaluate(frame, centers, w, plan_fill, want=pm.INPUTS)
pk, _ = peak_of(lambda: pm.evaluate(frame, centers, w, plan_fill, want=('Hp',), p=p, have=have))
print('predicted Hp peak at 2048: %.2f MiB; measured %.2f MiB; kept %.2f; target 0.9*budget = %.2f; total measured %.2f'
      % (mb(pred), mb(pk), mb(pm.tree_bytes(have)), 0.9 * 8, mb(pk + pm.tree_bytes(have))))
# affine check across block sizes
for b in (128, 256, 512, 1024, 2048, 4096):
    hv = pm.evaluate(frame, centers, w, pm.Plan(samples=b), want=pm.INPUTS)
    pk, _ = peak_of(lambda: pm.evaluate(frame, centers, w, pm.Plan(samples=b), want=('Hp',), p=p, have=hv))
    print('  block %5d: measured %.3f MiB, model %.3f MiB, ratio %.3f' % (b, mb(pk), mb(fixed + slope * b), pk / (fixed + slope * b)))

# (2) peaks of the other calls of the user example under the fill plan
for want in (('f',), ('f', 'g') + pm.INPUTS, ('f', 'g'), ('r',), ('f','g','r'), ('f','g','xi_z','mu_z','nu_z')):
    pk, _ = peak_of(lambda: pm.evaluate(frame, centers, w, plan_fill, want=want, y=y, have=have if 'xi' not in want else None))
    print('  fill plan, want=%-40s peak %.2f MiB (+kept %.2f) budget 8.00' % (want, mb(pk), mb(pm.tree_bytes(have))))
