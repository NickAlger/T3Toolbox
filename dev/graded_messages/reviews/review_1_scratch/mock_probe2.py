import sys, importlib.util, tracemalloc, gc
import numpy as np
sys.path.append('/home/claude/repo/T3Toolbox')
spec = importlib.util.spec_from_file_location('pm', '/home/claude/graded_messages_handoff/scripts/plan_mockup.py')
pm = importlib.util.module_from_spec(spec); sys.modules['pm'] = pm; spec.loader.exec_module(pm)
x, w, y, frame, centers, p, plan_fill, _ = pm.user_example()
mb = lambda b: b / 2**20
def peak_of(call):
    gc.collect(); tracemalloc.start(); out = call(); cur, pk = tracemalloc.get_traced_memory(); tracemalloc.stop(); return pk, cur, out
pz = pm.zero_tangent(frame)
print('peak of Hp at block 256 as a function of total W (have = xi):')
for Wt in (512, 1024, 2048, 4096):
    ws = [a[:Wt] for a in w]
    hv = pm.evaluate(frame, centers, ws, pm.Plan(samples=256), want=pm.INPUTS)
    for name, pp in (('zero p', pz), ('real p', p)):
        pk, cur, out = peak_of(lambda: pm.evaluate(frame, centers, ws, pm.Plan(samples=256), want=('Hp',), p=pp, have=hv))
        print('   W=%5d %s: peak %.3f MiB, still allocated after return %.3f MiB' % (Wt, name, mb(pk), mb(cur)))
# cycles left behind by one call
gc.collect(); gc.disable()
ws = [a[:512] for a in w]
pm.evaluate(frame, centers, ws, pm.Plan(samples=64), want=('Hp',), p=p)
n = gc.collect(); gc.enable()
print('unreachable objects found by gc after one Hp call with 8 blocks (cycles):', n)
gc.collect(); gc.set_debug(gc.DEBUG_SAVEALL)
pm.evaluate(frame, centers, ws, pm.Plan(samples=64), want=('Hp',), p=p)
gc.collect()
kinds = {}
for o in gc.garbage:
    kinds[type(o).__name__] = kinds.get(type(o).__name__, 0) + 1
print(kinds)
arrs = [o for o in gc.garbage if isinstance(o, np.ndarray)]
print('ndarrays in cyclic garbage:', len(arrs), 'bytes owned', sum(a.nbytes for a in arrs if a.base is None))
gc.set_debug(0); gc.garbage.clear()
