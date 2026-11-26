"""pytest plugin: record the forms (packed / ragged) of every sampling call's inputs and outputs, and every
pack / unpack / layout conversion made inside a sampling call. Read-only instrumentation; writes JSON per process.

Written for the packing investigation (``dev/graded_messages/PACKING_INVESTIGATION.md``, section 6). Usage, from the
repository root::

    AUDIT_OUT=<an empty directory> PYTHONPATH=$PWD:dev/graded_messages/packing_audit \
        python -m pytest -q -n 2 -p sampling_audit_plugin -p no:cacheprovider <test files>
    python dev/graded_messages/packing_audit/analyze.py <that directory> table     # forms per function
    python dev/graded_messages/packing_audit/analyze.py <that directory> shadows   # the packed re-runs
    python dev/graded_messages/packing_audit/analyze.py <that directory> convs     # conversions inside calls

What it does:

- wraps every public function of ``ut3_sampling``, ``utv_sampling``, ``probing``, ``apply``, ``entries``,
  ``sampling_derivatives`` (not the ``compute_*`` / ``assemble_*`` steps), the engine ``sampling_variables``, the
  sampling methods of the four frontend classes and the methods of the fitting kinds; records per call the layout
  (ragged / uniform, read off the cores or the frame), the form of each per-mode vector argument (``seq`` or
  ``arr``) and flags on the output (``seq_d:*`` = a tuple of exactly ``d`` arrays: a ragged per-mode output, or a
  coincidence when another tuple has length ``d``, e.g. the two blocks at ``d = 2``);
- wraps every converter (``pack_vectors``, ``unpack_vectors``, ``_pack_data_space``, ``t3_to_ut3``, ``ut3_to_t3``,
  the frame / variation conversions, ``fitting._arrays_of``, ``pack_sample`` / ``pack_data`` / ``pack_batch``) and
  records each call made inside a sampling call, with the stack of sampling calls and the caller's line;
- "shadow" calls: when a test calls a uniform function with ragged per-mode vectors, it calls the function again with
  them packed, and records the output form and whether the result agrees with the ragged one (real part equal,
  padding exactly zero).

Known limits: conversions done inline, not through a named converter, are not seen; layout detection is heuristic;
the shadow cannot find ``N`` for functions given only a sweep (``noN``) and does not pack an entries kind's
``(index, pp)`` pair (``nothing-to-pack``)."""
import functools
import importlib
import t3toolbox
import inspect
import json
import os
import sys
from collections import Counter

OUT_DIR = os.environ.get('AUDIT_OUT', '.')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(t3toolbox.__file__)))

CALLS = Counter()        # (name, depth0, layout, forms, outclass) -> count
CONVS = Counter()        # (converter, stack-signature, caller) -> count
STACK = []               # [(name, layout, forms)]

PER_MODE = {'ww', 'pp', 'ztildes', 's', 'vecs'}                # per-mode in every kind
PER_MODE_PROBE = {'c', 'residual', 'r', 'y', 'weight', 'sample', 'z'}   # per-mode for the probe kind only
DATA_PARAMS = ('x', 'data', 'frame', 'frame_data', 'core_pair', 'variations', 'variation', 'variations_data',
               'v', 'v_sc', 'x_sc', 'x_cores')


def is_arr(v):
    return hasattr(v, 'shape') and hasattr(v, 'dtype') and not isinstance(v, (tuple, list))


def form(v):
    if v is None:
        return None
    if isinstance(v, (tuple, list)):
        return 'seq'
    if is_arr(v):
        return 'arr'
    return type(v).__name__


def d_of_data(data):
    """(layout, d, N) of a core / frame data tuple, or None."""
    if hasattr(data, 'data') and not is_arr(data):
        data = data.data
    if not isinstance(data, (tuple, list)) or not data:
        return None
    f0 = data[0]
    if hasattr(f0, 'data') and not is_arr(f0):
        return d_of_data(f0)
    if is_arr(f0) and len(f0.shape) >= 2:
        return ('uniform', int(f0.shape[0]), int(f0.shape[-1]))
    if isinstance(f0, (tuple, list)) and f0:
        if is_arr(f0[0]):
            return ('ragged', len(f0), None)
        return d_of_data(f0)
    return None


def detect_layout(name, bound, self_obj):
    if self_obj is not None:
        obj = self_obj.frame if hasattr(self_obj, 'frame') else self_obj
        r = d_of_data(obj)
        cn = type(self_obj).__name__
        lay = 'uniform' if (cn.startswith('Uniform') or cn.startswith('UT3')) else 'ragged'
        if r is None and hasattr(self_obj, 'masks'):        # a uniform kind: frame comes as an argument
            r = None
        if r is None:
            for p in ('frame', 'frame_data', 'x_sc', 'x_cores'):
                if p in bound:
                    r = d_of_data(bound[p])
                    if r:
                        break
        return lay, (r[1] if r else None)
    short = name.split('.')[-1]
    if short.startswith('ut3_') or short.startswith('utv_'):
        lay = 'uniform'
    elif short.startswith('dense_') or 'ambient_transpose' in short:
        return 'dense', None
    else:
        lay = None
    for p in DATA_PARAMS + ('frame_sweep', 'sweep'):
        if p in bound:
            r = d_of_data(bound[p])
            if r:
                return (lay or r[0]), r[1]
    return lay or '?', None


def N_of(bound, self_obj):
    for obj in ([self_obj.frame if hasattr(self_obj, 'frame') else self_obj] if self_obj is not None else []) + [
            bound.get(p) for p in DATA_PARAMS + ('frame_sweep', 'sweep')]:
        if obj is None:
            continue
        r = d_of_data(obj)
        if r and r[0] == 'uniform':
            return r[2]
    return None


def _d_of_self(obj):
    try:
        data = obj.data
        if hasattr(obj, 'frame') and not hasattr(data[0], 'shape'):       # tangents: (frame, variations)
            pass
        f0 = data[0]
        if is_arr(f0):
            return int(f0.shape[0])
        if isinstance(f0, (tuple, list)):
            g = f0[0]
            return int(g.shape[0]) if is_arr(g) else len(f0)
    except Exception:
        return None
    return None


def forms_of(name, bound, kind_probe):
    out = []
    for p in sorted(bound):
        v = bound[p]
        if p in PER_MODE or (kind_probe and p in PER_MODE_PROBE):
            if p == 's' and isinstance(v, (tuple, list)) and len(v) == 2:   # a derivatives kind's (ww, pp)
                out.append('s.ww=%s' % form(v[0]))
                out.append('s.pp=%s' % form(v[1]))
            else:
                f = form(v)
                if f is not None:
                    out.append('%s=%s' % (p, f))
    return tuple(out)


def outclass(out, d):
    """Flags over the output pytree: ragged per-mode tuples, tuples of d equal arrays, arrays."""
    flags = set()

    def walk(v, depth=0):
        if depth > 6:
            return
        if isinstance(v, (tuple, list)):
            arrs = [a for a in v if is_arr(a)]
            if d is not None and len(v) == d and len(arrs) == d and d > 1:
                shapes = {tuple(a.shape) for a in arrs}
                lasts = {tuple(a.shape)[-1:] for a in arrs}
                flags.add('seq_d:ragged' if len(lasts) > 1 else ('seq_d:equal' if len(shapes) == 1 else 'seq_d:mixedshape'))
            for a in v:
                walk(a, depth + 1)
        elif is_arr(v):
            flags.add('arr')
        elif v is None:
            pass
        else:
            flags.add(type(v).__name__)
    walk(out)
    return tuple(sorted(flags))


def wrap(name, fn, is_method=False):
    sig = inspect.signature(fn)
    kind_probe = 'probe' in name.lower()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            b = sig.bind_partial(*args, **kwargs).arguments
        except TypeError:
            b = {}
        self_obj = b.get('self') if is_method else None
        kp = kind_probe or (self_obj is not None and 'Probe' in type(self_obj).__name__) or b.get('kind') == 'probe'
        if b.get('kind') in ('probe', 'apply') and 'sample' in b:
            b = dict(b); b['ww'] = b.pop('sample')
        layout, d = detect_layout(name, b, self_obj)
        fs = forms_of(name, b, kp)
        depth0 = len(STACK) == 0 or (bool(SHADOW) and len(STACK) == 0)
        STACK.append((('SHADOW:' if SHADOW else '') + name, layout, fs))
        try:
            out = fn(*args, **kwargs)
        finally:
            STACK.pop()
        try:
            CALLS[(('SHADOW:' if SHADOW else '') + name, depth0, layout, fs, outclass(out, d))] += 1
        except Exception as e:
            CALLS[(name, depth0, layout, fs, ('ERR', str(e)[:40]))] += 1
        if name.endswith('sampling_variables') and layout == 'uniform' and fs and all(f.endswith('=arr') for f in fs):
            oc = outclass(out, d)
            if any(f.startswith('seq_d') for f in oc):
                import os as _os
                SHADOWS[('SEQOUT', 'want=%s d=%s out=%s test=%s' % (b.get('want'), d,
                         [[tuple(getattr(a, 'shape', ())) for a in v] if isinstance(v, (tuple, list)) else tuple(getattr(v, 'shape', ())) for v in out],
                         _os.environ.get('PYTEST_CURRENT_TEST', '?')))] += 1
        if depth0 and layout == 'uniform' and any(f.endswith('=seq') for f in fs) and not SHADOW:
            shadow(name, fn, sig, args, kwargs, b, self_obj, kp, d, out)
        return out
    wrapper.__audit_wrapped__ = True
    return wrapper


SHADOW = []
SHADOWS = Counter()


def _leaves(v):
    if isinstance(v, (tuple, list)):
        return [a for x in v for a in _leaves(x)]
    return [v] if is_arr(v) else []


def shadow(name, fn, sig, args, kwargs, b, self_obj, kp, d, ref):
    """Call ``fn`` again with every ragged per-mode vector packed; record forms, conversions, and agreement."""
    import numpy as np
    from t3toolbox.backend import ut3_operations as uo
    N = N_of(b, self_obj)
    if N is None:
        SHADOWS[(name, 'noN')] += 1
        return
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError:
        SHADOWS[(name, 'bindfail')] += 1
        return
    order = bound.arguments.get('order', ())
    len_P = len(order) if isinstance(order, tuple) else 0
    packed_names = []
    for p, v in list(bound.arguments.items()):
        per_mode = p in PER_MODE or (kp and p in PER_MODE_PROBE) or (p == 'sample' and b.get('kind') in ('probe', 'apply'))
        if not per_mode or not isinstance(v, (tuple, list)) or not v:
            continue
        if p == 's' and len(v) == 2 and isinstance(v[0], (tuple, list)):
            bound.arguments[p] = tuple(uo.pack_vectors(list(x), N, len_P=len_P) if isinstance(x, (tuple, list)) else x for x in v)
        elif all(is_arr(a) for a in v) and len(v) == d:
            lp = len_P if p in ('ww', 'sample') else 0
            try:
                if p in ('y', 'weight'):
                    bound.arguments[p] = uo._pack_data_space(list(v), N)
                else:
                    bound.arguments[p] = uo.pack_vectors(list(v), N, len_P=lp)
            except Exception as e:
                SHADOWS[(name, 'pack-failed %s: %s' % (p, str(e)[:60]))] += 1
                return
        else:
            continue
        packed_names.append(p)
    if not packed_names:
        SHADOWS[(name, 'nothing-to-pack')] += 1
        return
    SHADOW.append(1)
    STACK.append(('SHADOW:' + name, 'uniform', forms_of(name, dict(bound.arguments), kp)))
    try:
        out = fn(*bound.args, **bound.kwargs)
    except Exception as e:
        SHADOWS[(name, 'raised: %s' % str(e)[:80])] += 1
        return
    finally:
        STACK.pop()
        SHADOW.pop()
    # agreement: unpack per-mode outputs of the original (ragged) against the packed output
    try:
        ok = _agree(ref, out, N)
    except Exception as e:
        ok = 'cmp-error: %s' % str(e)[:60]
    SHADOWS[(name, 'packed=' + ','.join(sorted(packed_names)), 'out=' + ','.join(outclass(out, d)), 'agree=%s' % ok)] += 1
    if ok is not True and name.endswith('sampling_variables'):
        import numpy as np, os as _os
        want = b.get('want')
        diffs = []
        for nm, a, c in zip(want or (), ref, out):
            try:
                diffs.append('%s:%s/%s maxdiff=%.2e' % (nm, np.shape(a), np.shape(c), float(np.max(np.abs(np.asarray(a) - np.asarray(c))))))
            except Exception as e:
                diffs.append('%s:%s' % (nm, str(e)[:40]))
        info = 'kind=%s order=%s y=%s weight=%s | %s | test=%s' % (b.get('kind'), b.get('order'),
            [np.shape(a) for a in b['y']] if isinstance(b.get('y'), (tuple, list)) else np.shape(b.get('y')),
            [np.shape(a) for a in b['weight']] if isinstance(b.get('weight'), (tuple, list)) else (None if b.get('weight') is None else np.shape(b.get('weight'))),
            '; '.join(diffs), _os.environ.get('PYTEST_CURRENT_TEST', '?'))
        SHADOWS[('DISAGREE', info)] += 1


def _agree(ref, out, N):
    import numpy as np
    if isinstance(ref, (tuple, list)) and is_arr(out) and ref and all(is_arr(a) for a in ref):
        o = np.asarray(out)
        return all(np.allclose(np.asarray(a), o[i, ..., :np.shape(a)[-1]], atol=1e-10) and
                   np.all(o[i, ..., np.shape(a)[-1]:] == 0) for i, a in enumerate(ref))
    if isinstance(ref, (tuple, list)) and isinstance(out, (tuple, list)):
        if len(ref) != len(out):
            return 'len %d vs %d' % (len(ref), len(out))
        return all(_agree(a, c, N) is True for a, c in zip(ref, out))
    if is_arr(ref) and is_arr(out):
        return bool(np.shape(ref) == np.shape(out) and np.allclose(np.asarray(ref), np.asarray(out), atol=1e-10))
    if not is_arr(ref) and not isinstance(ref, (tuple, list)):
        return True
    return 'structure %s vs %s' % (type(ref).__name__, type(out).__name__)


def caller_site():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if 'sampling_audit_plugin' not in fn and not fn.endswith('ut3_operations.py') and not fn.endswith('functools.py'):
            return '%s:%d %s' % (os.path.relpath(fn, REPO_ROOT), f.f_lineno, f.f_code.co_name)
        f = f.f_back
    return '?'


def wrap_converter(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if STACK:
            first = args[0] if args else None
            CONVS[(name, tuple(STACK), form(first), caller_site())] += 1
        return fn(*args, **kwargs)
    wrapper.__audit_wrapped__ = True
    return wrapper


BACKEND = ['ut3_sampling', 'utv_sampling', 'probing', 'apply', 'entries', 'sampling_derivatives']
CONVERTERS = {
    'ut3_operations': ['pack_vectors', 'unpack_vectors', '_pack_data_space'],
    'ut3_conversions': ['t3_to_ut3', 'ut3_to_t3', 'ut3_to_dense'],
    'ufv_conversions': ['ut3frame_to_t3frame', 't3frame_to_ut3frame', 'ut3variations_to_t3variations',
                        't3variations_to_ut3variations'],
    'fitting': ['_arrays_of'],
    'uniform_fitting': ['pack_sample', 'pack_data', 'pack_batch'],
}
FRONTEND = {
    'tucker_tensor_train': ('TuckerTensorTrain', ['entries', 'apply', 'probe', 'probe_ambient_transpose',
                            'probe_corewise_transpose', 'apply_ambient_transpose', 'entries_ambient_transpose',
                            'apply_corewise_transpose', 'entries_corewise_transpose', 'probe_corewise_tangent',
                            'apply_corewise_tangent', 'entries_corewise_tangent', 'probe_corewise_hessian',
                            'apply_corewise_hessian', 'entries_corewise_hessian']),
    'manifold': ('T3Tangent', ['probe', 'probe_transpose', 'apply', 'entries', 'apply_transpose', 'entries_transpose']),
    'uniform_tucker_tensor_train': ('UniformTuckerTensorTrain', ['entries', 'apply', 'probe',
                            'apply_corewise_transpose', 'entries_corewise_transpose', 'probe_corewise_transpose',
                            'probe_corewise_tangent', 'probe_corewise_hessian', 'apply_corewise_tangent',
                            'apply_corewise_hessian', 'entries_corewise_tangent', 'entries_corewise_hessian']),
    'uniform_manifold': ('UT3Tangent', ['probe', 'apply', 'entries', 'probe_transpose', 'apply_transpose',
                         'entries_transpose']),
}
KIND_METHODS = ['precompute', 'forward', 'transpose', 'point_forward', 'take', 'corewise_hessian']


def install():
    for m in BACKEND:
        mod = importlib.import_module('t3toolbox.backend.' + m)
        for n in mod.__all__:
            if n.startswith('compute_') or n.startswith('assemble_') or n in (
                    'estimate_chunk_size', 'max_chunk_size_within', 'check_perturbation_vectors',
                    'check_perturbation_index', 'build_input_jets'):
                continue
            f = getattr(mod, n)
            if callable(f) and not getattr(f, '__audit_wrapped__', False):
                setattr(mod, n, wrap('%s.%s' % (m, n), f))
    sv = importlib.import_module('t3toolbox.backend.sampling_variables')
    sv.sampling_variables = wrap('sampling_variables.sampling_variables', sv.sampling_variables)
    for m, names in CONVERTERS.items():
        mod = importlib.import_module('t3toolbox.backend.' + m)
        for n in names:
            setattr(mod, n, wrap_converter('%s.%s' % (m, n), getattr(mod, n)))
    for m, (cls_name, names) in FRONTEND.items():
        cls = getattr(importlib.import_module('t3toolbox.' + m), cls_name)
        for n in names:
            if n in cls.__dict__:
                raw = cls.__dict__[n]
                if isinstance(raw, staticmethod):
                    setattr(cls, n, staticmethod(wrap('%s.%s' % (cls_name, n), raw.__func__)))
                elif isinstance(raw, classmethod):
                    setattr(cls, n, classmethod(wrap('%s.%s' % (cls_name, n), raw.__func__)))
                else:
                    setattr(cls, n, wrap('%s.%s' % (cls_name, n), raw, is_method=True))
    for m in ('fitting', 'uniform_fitting'):
        mod = importlib.import_module('t3toolbox.backend.' + m)
        for cname in dir(mod):
            cls = getattr(mod, cname)
            if inspect.isclass(cls) and cls.__module__ == mod.__name__ and cname.endswith('Kind'):
                for n in KIND_METHODS:
                    if n in cls.__dict__ and inspect.isfunction(cls.__dict__[n]):
                        setattr(cls, n, wrap('%s.%s' % (cname, n), cls.__dict__[n], is_method=True))


def pytest_configure(config):
    install()


def pytest_unconfigure(config):
    path = os.path.join(OUT_DIR, 'audit_%d.json' % os.getpid())
    with open(path, 'w') as fh:
        json.dump({'calls': [[list(k[:3]) + [list(k[3]), list(k[4])], v] for k, v in CALLS.items()],
                   'shadows': [[list(k), v] for k, v in SHADOWS.items()],
                   'convs': [[k[0], [list(s[:2]) + [list(s[2])] for s in k[1]], k[2], k[3], v]
                             for k, v in CONVS.items()]}, fh)
