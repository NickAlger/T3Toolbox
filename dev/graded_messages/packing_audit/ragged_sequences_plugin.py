"""pytest plugin: the blast radius of requiring sequences on the ragged layer (``PACKING_INVESTIGATION.md`` §10.9, O2).

Nick's candidate rule (2026-11-24): on the ragged layer every per-mode object is a sequence of ``d`` arrays; a single
array of stacked slices is refused (the user unstacks it first). Records, without raising, every place an array is
given where the rule requires a sequence:

- ``A:engine <arg>``: the engine on a ragged frame or point given an array sample (probe, apply), or an array ``y``,
  weight or held data-space variable (probe);
- ``A:<module>.<function> <param>``: a ragged or dense public sampling function given an array per-mode argument;
- ``A:data_metric weight``: a tuple ``r`` with an array weight;
- ``A:jets.<function> <param>`` / ``A:data_preparation.<function> <param>``: a probe call holding a ragged object and,
  beside it, an array per-mode object;
- ``A:blocks_to_weight array for a ragged y``: the constructor returning one array where the rule wants a tuple;
- ``I:entries index``: informational, the engine's entries kind on a ragged frame (the index is ``(d,)+W``, one array:
  is it covered by the rule?).

Each event carries a flag (``edge1``: the last axis is 1; ``equal``: every array row has its mode's width; else
``other``), the library frames and the test, doctest or script line. Usage as ``strict_forms_plugin``:

    RAGSEQ_OUT=<dir> PYTHONPATH=$PWD:dev/graded_messages/packing_audit \\
        python -m pytest -q -n 2 -p ragged_sequences_plugin -p no:cacheprovider tests/
    python dev/graded_messages/packing_audit/strict_analyze.py <dir> users     # the same JSON layout
"""
import functools
import importlib
import inspect
import json
import os
import sys
from collections import Counter

import t3toolbox

OUT_DIR = os.environ.get('RAGSEQ_OUT', '.')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(t3toolbox.__file__)))
LIB = os.path.join(REPO_ROOT, 't3toolbox') + os.sep
MACHINERY = {'ut3_sampling.py', 'utv_sampling.py', 'sampling_variables.py', 'probing.py', 'apply.py', 'entries.py',
             'sampling_derivatives.py', 'ut3_operations.py', 'sampling_steps.py', 'jets.py', 'contractions.py',
             'common.py', 'uniform_tucker_tensor_train.py', 'tucker_tensor_train.py', 'manifold.py',
             'uniform_manifold.py', 'data_preparation.py'}
EVENTS = Counter()
SCRIPT = {'name': None}


def is_arr(v):
    return hasattr(v, 'shape') and hasattr(v, 'dtype') and not isinstance(v, (tuple, list))


def is_seq(v):
    return isinstance(v, (tuple, list))


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
        if 'ragged_sequences_plugin' in fn or fn.endswith('functools.py'):
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


def _widths(frame_or_x):
    try:
        D = frame_or_x[0]
        if is_seq(D) and D and is_arr(D[0]):
            return [int(B.shape[-1]) for B in D]
    except Exception:
        pass
    return None


def _flag(arr, widths=None):
    if arr.ndim and arr.shape[-1] == 1:
        return 'edge1'
    if widths is not None and all(int(arr.shape[-1]) == n for n in widths):
        return 'equal'
    return 'other'


def _guard(name, fn, check):
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            check(sig.bind_partial(*args, **kwargs).arguments)
        except Exception as e:
            record('PLUGIN-ERROR ' + name, str(e)[:80])
        return fn(*args, **kwargs)
    wrapper.__ragseq__ = True
    return wrapper


def _check_engine(b):
    widths = _widths(b.get('frame'))
    if widths is None:
        return
    kind = b.get('kind')
    if kind == 'entries':
        record('I:entries index')
        return
    if is_arr(b.get('sample')):
        record('A:engine sample', _flag(b['sample'], widths))
    if kind == 'probe':
        for name in ('y', 'weight'):
            if is_arr(b.get(name)):
                record('A:engine ' + name, _flag(b[name], widths))
        held = b.get('held')
        if held is not None:
            for name in ('Jp', 'r', 'z', 'w', 'w_tilde'):
                try:
                    v = held[name] if name in held else None
                except Exception:
                    v = None
                if is_arr(v):
                    record('A:engine held ' + name, _flag(v, widths))


PER_MODE = ('ww', 'pp', 'ztildes')
PER_MODE_PROBE = ('c', 'residual', 'r', 'z')
DATA_ARGS = ('x', 'frame', 'core_pair', 'variation', 'variations')


def _check_ragged_public(name):
    probe = 'probe' in name
    dense = '.dense_' in name or 'ambient' in name

    def check(b):
        widths = None
        for p in DATA_ARGS:
            if p in b:
                widths = _widths(b[p])
                if widths is not None:
                    break
        if widths is None and not dense:
            return
        for p in PER_MODE + (PER_MODE_PROBE if probe else ()):
            if is_arr(b.get(p)):
                record('A:%s %s' % (name, p), _flag(b[p], widths))
    return check


def _check_data_metric(b):
    if is_seq(b.get('r')) and is_arr(b.get('weight')):
        record('A:data_metric weight', _flag(b['weight']))


HELPER_OBJECTS = ('jet', 'dual_jet', 'jets', 'y', 'weight', 'sample', 'validation')


def _check_helper(name):
    def check(b):
        if b.get('kind') != 'probe':
            return
        present = [p for p in HELPER_OBJECTS if p in b and b[p] is not None]
        if not any(is_seq(b[p]) for p in present):
            return
        for p in present:
            if is_arr(b[p]):
                record('A:%s %s' % (name, p), _flag(b[p]))
        if name.endswith('blocks_to_weight') and is_seq(b.get('y')):
            record('A:blocks_to_weight array for a ragged y')
    return check


def install():
    sv = importlib.import_module('t3toolbox.backend.sampling_variables')
    sv.sampling_variables = _guard('engine', sv.sampling_variables, _check_engine)
    for m in ('probing', 'apply', 'entries', 'sampling_derivatives'):
        mod = importlib.import_module('t3toolbox.backend.' + m)
        for n in mod.__all__:
            if n.startswith(('compute_', 'assemble_')) or n in ('estimate_chunk_size', 'max_chunk_size_within',
                                                                 'check_perturbation_vectors',
                                                                 'check_perturbation_index', 'build_input_jets'):
                continue
            f = getattr(mod, n)
            if inspect.isfunction(f):
                setattr(mod, n, _guard('%s.%s' % (m, n), f, _check_ragged_public('%s.%s' % (m, n))))
    jets = importlib.import_module('t3toolbox.backend.jets')
    jets.data_metric = _guard('data_metric', jets.data_metric, _check_data_metric)
    for n in ('jet_to_dual_jet', 'dual_jet_to_jet', 'derivatives_to_taylor', 'taylor_to_derivatives',
              'prepare_derivative_data', 'blocks_to_weight', 'check_sample', 'check_data', 'check_weight'):
        setattr(jets, n, _guard('jets.' + n, getattr(jets, n), _check_helper('jets.' + n)))
    dp = importlib.import_module('t3toolbox.backend.data_preparation')
    for n in ('data_block_norms', 'rescale_direction', 'rescale_coordinates'):
        setattr(dp, n, _guard('data_preparation.' + n, getattr(dp, n), _check_helper('data_preparation.' + n)))
    # the frontend re-exports these by name (bound at import): point its names at the same wrappers, so frontend
    # calls are recorded too and the frontend's identity test (a name IS the backend object) still holds
    originals = {}
    for mod in (jets, dp):
        for n in dir(mod):
            f = getattr(mod, n)
            if getattr(f, '__ragseq__', False):
                originals[id(f.__wrapped__)] = f
    for front in ('t3toolbox.fitting', 't3toolbox.optimizers'):
        fm = importlib.import_module(front)
        for n in dir(fm):
            g = getattr(fm, n)
            if id(g) in originals:
                setattr(fm, n, originals[id(g)])


def dump():
    path = os.path.join(OUT_DIR, 'strict_%d.json' % os.getpid())     # strict_analyze.py reads strict_*.json
    with open(path, 'w') as fh:
        json.dump([[list(k), v] for k, v in EVENTS.items()], fh)


def pytest_configure(config):
    install()


def pytest_unconfigure(config):
    dump()
