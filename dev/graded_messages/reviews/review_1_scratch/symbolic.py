import importlib.util, sys, io, contextlib, types
import numpy
def load():
    spec = importlib.util.spec_from_file_location('cg', '/home/claude/graded_messages_handoff/scripts/component_graphs.py')
    cg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cg); return cg
def out_of(cg):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cg.main('/home/claude/repo/T3Toolbox')
    return buf.getvalue()
a = out_of(load())
cg = load()
class NP:
    def __getattr__(self, k):
        return (lambda x: True) if k == 'any' else getattr(numpy, k)
cg.np = NP()          # the value-dependent "skip zero coefficients" test in gmul_traced is disabled
b = out_of(cg)
al, bl = a.splitlines(), b.splitlines()
print('lines', len(al), len(bl), 'identical apart from the nu_z shortcut line:', [x for x, y in zip(al, bl) if x != y])
