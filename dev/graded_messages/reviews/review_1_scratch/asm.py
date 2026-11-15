import sys, importlib.util, tracemalloc, gc
sys.path.append('/home/claude/repo/T3Toolbox')
spec = importlib.util.spec_from_file_location('pm', '/home/claude/graded_messages_handoff/scripts/plan_mockup.py')
pm = importlib.util.module_from_spec(spec); sys.modules['pm'] = pm; spec.loader.exec_module(pm)
x, w, y, frame, centers, p, plan_fill, _ = pm.user_example()
def peak(plan, want, **kw):
    gc.collect(); tracemalloc.start(); pm.evaluate(frame, centers, w, plan, want=want, **kw); pk = tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return pk / 2**20
for a in (None, 256, 64, 16):
    print('samples=512 assembly=%-5s Hp peak %.3f MiB   (f,g) peak %.3f MiB' % (a, peak(pm.Plan(samples=512, assembly=a), ('Hp',), p=p), peak(pm.Plan(samples=512, assembly=a), ('f', 'g'), y=y)))
for a in (None, 512, 64):
    print('samples=None assembly=%-5s Hp peak %.3f MiB' % (a, peak(pm.Plan(assembly=a), ('Hp',), p=p)))
