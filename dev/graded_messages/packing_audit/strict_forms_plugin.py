"""pytest plugin: the blast radius of refusing the other form, option (a) of the packing investigation
(``dev/graded_messages/PACKING_INVESTIGATION.md``). Records, without raising, every place where a strict library would
refuse an input:

- ``U:*`` -- ragged (unpacked) per-mode vectors reaching a conversion on the uniform layout: ``pack_if_ragged`` or
  ``_pack_data_space`` given a sequence, and the fitting frontend's ``pack_batch``;
- ``R:*`` -- a packed array where the ragged layout takes a per-mode sequence: the engine's sample, ``y``, weight and
  held variables, and the per-mode arguments of the ragged public sampling functions. Each is flagged ``equal`` when
  its trailing width equals every ``N_i`` (an array read as ``d`` same-length vectors: the possible exception) or
  ``padded`` otherwise;
- ``O:*`` -- informational: the jets helpers (no object, so no layout) given a packed probe array.

Each event records the immediate library caller, the innermost library frame outside the sampling machinery, the
outermost library frame (the public function the user called) and the first non-library frame (the test, doctest or
example line), with the pytest test id. Usage, from the repository root::

    STRICT_OUT=<an empty directory> PYTHONPATH=$PWD:dev/graded_messages/packing_audit \\
        python -m pytest -q -n 2 -p strict_forms_plugin -p no:cacheprovider tests/
    python dev/graded_messages/packing_audit/strict_analyze.py <that directory>

For a script (an example): ``python dev/graded_messages/packing_audit/strict_run_script.py <script> <out dir>``.
"""
import functools
import importlib
import inspect
import json
import os
import sys
from collections import Counter

import t3toolbox

OUT_DIR = os.environ.get('STRICT_OUT', '.')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(t3toolbox.__file__)))
LIB = os.path.join(REPO_ROOT, 't3toolbox') + os.sep
# the sampling machinery itself: a frame in one of these is not "a library caller relying on the conversion"
MACHINERY = {'ut3_sampling.py', 'utv_sampling.py', 'sampling_variables.py', 'probing.py', 'apply.py', 'entries.py',
             'sampling_derivatives.py', 'ut3_operations.py', 'sampling_steps.py', 'jets.py', 'contractions.py',
             'common.py', 'uniform_tucker_tensor_train.py', 'tucker_tensor_train.py', 'manifold.py',
             'uniform_manifold.py'}
EVENTS = Counter()
SCRIPT = {'name': None}


def is_arr(v):
    return hasattr(v, 'shape') and hasattr(v, 'dtype') and not isinstance(v, (tuple, list))


def _rel(fn):
    try:
        return os.path.relpath(fn, REPO_ROOT)
    except ValueError:
        return fn


def _frames():
    f = sys._getframe(2)
    immediate = user = inner = outer = None
    while f is not None:
        fn = f.f_code.co_filename
        if 'strict_forms_plugin' in fn or fn.endswith('functools.py'):
            f = f.f_back
            continue
        site = '%s:%d %s' % (_rel(fn), f.f_lineno, f.f_code.co_name)
        if fn.startswith(LIB):
            if immediate is None:
                immediate = site
            if os.path.basename(fn) not in MACHINERY and inner is None:
                inner = site
            outer = site
        elif user is None and ('site-packages' not in fn and '/lib/python' not in fn):
            user = '%s:%d' % (_rel(fn), f.f_lineno)
        f = f.f_back
    return immediate, inner, outer, user


def record(event, detail=''):
    immediate, inner, outer, user = _frames()
    test = os.environ.get('PYTEST_CURRENT_TEST', SCRIPT['name'] or '?').rsplit(' (', 1)[0]
    EVENTS[(event, detail, immediate or '-', inner or '-', outer or '-', user or '-', test)] += 1


def _ragged_widths(frame_or_x):
    """The real widths N_i of a ragged point / frame .data, or None."""
    try:
        D = frame_or_x[0]
        if isinstance(D, (tuple, list)) and D and is_arr(D[0]):
            return [int(B.shape[-1]) for B in D]
    except Exception:
        pass
    return None


def _flag(arr, widths):
    if widths is None:
        return 'widths?'
    return 'equal' if all(int(arr.shape[-1]) == n for n in widths) else 'padded'


# ---- uniform side: conversions of ragged input ----

def _wrap_pack_if_ragged(fn, is_packed):
    @functools.wraps(fn)
    def wrapper(vectors, *args, **kwargs):
        if not is_packed(vectors):
            record('U:pack_if_ragged')
        return fn(vectors, *args, **kwargs)
    return wrapper


def _wrap_pack_data_space(fn, is_packed):
    @functools.wraps(fn)
    def wrapper(value, *args, **kwargs):
        if not is_packed(value):
            record('U:_pack_data_space')
        return fn(value, *args, **kwargs)
    return wrapper


def _wrap_pack_batch(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        record('U:pack_batch')
        return fn(*args, **kwargs)
    return wrapper


# ---- ragged side: packed arrays where per-mode sequences are taken ----

def _wrap_engine(fn):
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            b = sig.bind_partial(*args, **kwargs).arguments
            frame, kind = b.get('frame'), b.get('kind')
            widths = _ragged_widths(frame) if frame is not None else None
            if widths is not None:                                  # the ragged layout
                sample = b.get('sample')
                if kind in ('probe', 'apply') and is_arr(sample):
                    record('R:engine sample', _flag(sample, widths))
                if kind == 'probe':
                    for name in ('y', 'weight'):
                        v = b.get(name)
                        if is_arr(v) and v.ndim >= 1 and v.shape[0] == len(widths):
                            record('R:engine ' + name, _flag(v, widths) if v.shape[-1] != 1 else 'edge1')
                held = b.get('held')
                if held is not None and kind == 'probe':
                    for name in ('Jp', 'r'):
                        try:
                            v = held[name] if name in held else None
                        except Exception:
                            v = None
                        if is_arr(v):
                            record('R:engine held ' + name, _flag(v, widths))
        except Exception as e:                                       # never disturb the call
            record('PLUGIN-ERROR engine', str(e)[:80])
        return fn(*args, **kwargs)
    return wrapper


PER_MODE = ('ww', 'pp', 'ztildes')
PER_MODE_PROBE = ('c', 'residual', 'r')
DATA_ARGS = ('x', 'frame', 'core_pair', 'variation', 'variations', 'data')


def _wrap_ragged_public(name, fn):
    sig = inspect.signature(fn)
    probe = 'probe' in name

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            b = sig.bind_partial(*args, **kwargs).arguments
            widths = None
            for p in DATA_ARGS:
                if p in b:
                    widths = _ragged_widths(b[p])
                    if widths is not None:
                        break
            if widths is not None:
                for p in PER_MODE + (PER_MODE_PROBE if probe else ()):
                    if is_arr(b.get(p)):
                        record('R:%s %s' % (name, p), _flag(b[p], widths))
        except Exception as e:
            record('PLUGIN-ERROR ' + name, str(e)[:80])
        return fn(*args, **kwargs)
    return wrapper


# ---- informational: object-less jets helpers given a packed probe array ----

def _wrap_jets(name, fn):
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            b = sig.bind_partial(*args, **kwargs).arguments
            if b.get('kind') == 'probe':
                for p in ('jet', 'dual_jet', 'jets', 'y', 'r', 'weight'):
                    if is_arr(b.get(p)):
                        record('O:jets.%s %s' % (name, p))
        except Exception as e:
            record('PLUGIN-ERROR jets.' + name, str(e)[:80])
        return fn(*args, **kwargs)
    return wrapper


def install():
    uo = importlib.import_module('t3toolbox.backend.ut3_operations')
    is_packed = uo.is_packed
    uo.pack_if_ragged = _wrap_pack_if_ragged(uo.pack_if_ragged, is_packed)
    uo._pack_data_space = _wrap_pack_data_space(uo._pack_data_space, is_packed)
    uf = importlib.import_module('t3toolbox.backend.uniform_fitting')
    uf.pack_batch = _wrap_pack_batch(uf.pack_batch)
    sv = importlib.import_module('t3toolbox.backend.sampling_variables')
    sv.sampling_variables = _wrap_engine(sv.sampling_variables)
    for m in ('probing', 'apply', 'entries', 'sampling_derivatives'):
        mod = importlib.import_module('t3toolbox.backend.' + m)
        for n in mod.__all__:
            if n.startswith(('compute_', 'assemble_', 'dense_')) or 'ambient' in n:
                continue
            f = getattr(mod, n)
            if inspect.isfunction(f):
                setattr(mod, n, _wrap_ragged_public('%s.%s' % (m, n), f))
    jets = importlib.import_module('t3toolbox.backend.jets')
    for n in ('jet_to_dual_jet', 'dual_jet_to_jet', 'derivatives_to_taylor', 'taylor_to_derivatives',
              'prepare_derivative_data', 'blocks_to_weight'):
        if hasattr(jets, n):
            setattr(jets, n, _wrap_jets(n, getattr(jets, n)))


def dump():
    path = os.path.join(OUT_DIR, 'strict_%d.json' % os.getpid())
    with open(path, 'w') as fh:
        json.dump([[list(k), v] for k, v in EVENTS.items()], fh)


def pytest_configure(config):
    install()


def pytest_unconfigure(config):
    dump()
