import re, importlib, sys, inspect
sys.path.insert(0, '/home/nick/repos/T3Toolbox')
import t3toolbox
src = open('/home/nick/repos/T3Toolbox/docs/api_reference.rst').read()
# 1. every :py:xxx:`~target` resolves
targets = re.findall(r':py:(class|func|mod):`~([\w\.]+)`', src)
for kind, t in targets:
    try:
        if kind == 'mod':
            importlib.import_module(t)
        else:
            mod, name = t.rsplit('.', 1)
            getattr(importlib.import_module(mod), name)
    except Exception as e:
        print('UNRESOLVED', kind, t, e)
# 2. "Everything below is importable directly from t3toolbox" — check frontend section names
frontend = src.split('The backend surface')[0]
names = set(re.findall(r':py:(?:class|func):`~[\w\.]+\.(\w+)`', frontend)) | set(re.findall(r'``(\w+)``', frontend))
root = set(t3toolbox.__all__)
print('root __all__ size', len(root))
for n in sorted(names):
    if n not in root and not hasattr(t3toolbox, n):
        print('NOT AT ROOT:', n)
    elif n not in root and hasattr(t3toolbox, n):
        print('attr but not in __all__:', n)
# 3. root __all__ names not mentioned anywhere in api_reference
for n in sorted(root):
    if n not in src:
        print('ROOT NAME NOT IN API REF:', n)
# 4. backend modules not listed
import pkgutil, t3toolbox.backend as B
for m in pkgutil.iter_modules(B.__path__):
    if f'backend.{m.name}' not in src:
        print('BACKEND MODULE NOT LISTED:', m.name)
import os
for f in os.listdir('/home/nick/repos/T3Toolbox/t3toolbox'):
    if f.endswith('.py') and f != '__init__.py':
        if f't3toolbox.{f[:-3]}' not in src:
            print('FRONTEND MODULE NOT LISTED:', f)
# 5. frontend module __all__ names not at root
for modname in ['tucker_tensor_train','uniform_tucker_tensor_train','frame_variations_format','uniform_frame_variations_format','manifold','uniform_manifold','fitting','optimizers','shared_geometry','safety','corewise']:
    m = importlib.import_module('t3toolbox.'+modname)
    missing = [n for n in getattr(m,'__all__',[]) if n not in root]
    print(modname, 'in __all__ but not root:', missing)
