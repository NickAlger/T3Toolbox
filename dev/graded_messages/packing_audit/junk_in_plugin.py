"""pytest plugin: the prototype of guard 4 (``PACKING_INVESTIGATION.md`` §10.9, O6), "junk in".

For every public uniform sampling function a test calls (``ut3_sampling``, ``utv_sampling``, the engine
``sampling_variables``), the plugin calls it twice more: once with every input it can clean made canonical (padding
zeroed by the input's own masks), and once with finite garbage (``GARBAGE``) written into that padding. A function that
supports junk-in gives bitwise the same output both times. Inputs handled:

- a uniform point ``(B, G, shape, masks)``, frame ``(D, U, L, R, shape, masks)`` or variations ``(V, H, shape, masks)``,
  by their own masks;
- ``core_directions``: by the point's masks, or the frame's down and left masks;
- packed per-mode arrays (``ww``, ``pp``, ``ztildes``, ``c``, ``residual``, ``r``, ``y``, ``weight``, ``sample``):
  beyond each mode's ``N_i`` on the last axis (an edge of 1 has no padding; jet axes are data, left alone).

Left alone (not corrupted): sweeps, held variables, centers ``Vc``, bare supercores without masks, and a frame's
core directions with a stack ``K`` (the frame masker does not broadcast ``K``). Residual-type arguments (``c``, ``r``,
``y``, the weight) are per mode only for the probe kind: they are chosen by the kind, never by their shape (an apply
residual ``(order+1,)+W+C`` can have ``d`` rows and ``N`` columns by coincidence; the first version of this plugin
was caught by exactly that). Calls inside a jax
trace and calls with jax arrays are skipped. At most ``JUNK_MAX`` checks per function per process (default 25).

A difference is classified by where it falls in the clean output: ``padding-like`` (the clean output is exactly 0 there:
garbage reached the output's padding, i.e. the output is not canonical w.r.t. junk-in) or ``real`` (garbage reached a
nonzero entry: junk leaks into the result). The test is a heuristic (a real entry may be exactly 0). Usage:

    JUNK_OUT=<dir> PYTHONPATH=$PWD:dev/graded_messages/packing_audit \\
        python -m pytest -q -n 2 -p junk_in_plugin -p no:cacheprovider <test files>
    python dev/graded_messages/packing_audit/junk_in_analyze.py <dir>
"""
import functools
import importlib
import inspect
import json
import os
from collections import defaultdict

import numpy as np

OUT_DIR = os.environ.get('JUNK_OUT', '.')
JUNK_MAX = int(os.environ.get('JUNK_MAX', '25'))
GARBAGE = 1.0e3
RESULTS = defaultdict(lambda: {'checked': 0, 'equal': 0, 'padding-like': 0, 'real': 0, 'error': 0,
                               'no-corruptible-input': 0, 'max_diff': 0.0, 'example': None})
COUNTS = defaultdict(int)
BUSY = []


def is_arr(v):
    return hasattr(v, 'shape') and hasattr(v, 'dtype') and not isinstance(v, (tuple, list))


def is_jax(v):
    return is_arr(v) and type(v).__module__.split('.')[0] in ('jax', 'jaxlib')


def _leaves(v):
    if isinstance(v, (tuple, list)):
        return [a for x in v for a in _leaves(x)]
    return [v] if is_arr(v) else []


def _shape_tuple(v):
    return isinstance(v, tuple) and len(v) > 0 and all(isinstance(n, (int, np.integer)) for n in v)


def _form(v):
    """'point' | 'frame' | 'variations' | None for a uniform .data tuple."""
    if not isinstance(v, tuple) or not v or not is_arr(v[0]) or v[0].ndim < 3:
        return None
    if len(v) == 6 and _shape_tuple(v[4]) and len(v[5]) == 4:
        return 'frame'
    if len(v) == 4 and _shape_tuple(v[2]) and isinstance(v[3], tuple):
        return 'point' if len(v[3]) == 2 else ('variations' if len(v[3]) == 4 else None)
    return None


def _apply(form, data):
    from t3toolbox.backend import ut3_masking, ufv_masking
    if form == 'point':
        return tuple(ut3_masking.ut3_apply_masks(data))
    if form == 'frame':
        return tuple(ufv_masking.ufv_apply_frame_masks(data))
    return tuple(ufv_masking.ufv_apply_variations_masks(data))


def _clean_and_garbage(form, data):
    n = {'point': 2, 'frame': 4, 'variations': 2}[form]
    sc = tuple(np.asarray(a, dtype=float) for a in data[:n])
    ind = _apply(form, tuple(np.ones_like(a) for a in sc) + tuple(data[n:]))
    clean = _apply(form, sc + tuple(data[n:]))
    garbage = tuple(a + GARBAGE * (1.0 - i) for a, i in zip(clean, ind))
    return tuple(clean) + tuple(data[n:]), tuple(garbage) + tuple(data[n:])


PACKED = ('ww', 'pp', 'ztildes', 'sample')                     # per-mode in probe and apply
PACKED_PROBE = ('c', 'residual', 'r', 'y', 'weight')           # per-mode in probe only: apply's and entries' are one array
                                                               #   with no mode axis (read by kind, never by shape)


def _packed_pair(v, shape, N):
    a = np.asarray(v, dtype=float) if np.issubdtype(np.asarray(v).dtype, np.floating) else None
    if a is None or a.ndim < 2 or a.shape[0] != len(shape) or a.shape[-1] != N or N == 1:
        return None
    clean, garbage = a.copy(), a.copy()
    for i, n in enumerate(shape):
        clean[i, ..., n:] = 0.0
        garbage[i, ..., n:] = GARBAGE
    return clean, garbage


def _variants(b, probe):
    """(clean args, garbage args) as dicts, or None when nothing could be corrupted. ``probe``: the call is of the
    probe kind (its data-space arguments are per mode)."""
    clean, garbage = dict(b), dict(b)
    touched = False
    shape = N = None
    point_masks = frame = None
    for name, v in b.items():
        form = _form(v)
        if form is None:
            continue
        c, g = _clean_and_garbage(form, v)
        clean[name], garbage[name] = c, g
        touched = True
        if shape is None:
            shape = tuple(int(n) for n in (v[4] if form == 'frame' else v[2]))
            N = int(np.shape(v[0])[-1])
        if form == 'point':
            point_masks = v
        elif form == 'frame':
            frame = v
    cd = b.get('core_directions')
    if isinstance(cd, tuple) and len(cd) == 2 and all(is_arr(a) for a in cd):
        if point_masks is not None:
            c, g = _clean_and_garbage('point', tuple(cd) + tuple(point_masks[2:]))
            clean['core_directions'], garbage['core_directions'] = c[:2], g[:2]
            touched = True
        elif frame is not None and np.shape(cd[0]) == np.shape(frame[0]):  # the frame masker broadcasts no K
            fake = (cd[0], frame[1], cd[1], frame[3]) + tuple(frame[4:])
            c, g = _clean_and_garbage('frame', fake)
            clean['core_directions'], garbage['core_directions'] = (c[0], c[2]), (g[0], g[2])
            touched = True
    if shape is not None:
        if b.get('kind') == 'entries':
            names = ()                                                 # the index has no padding
        else:
            names = PACKED + (PACKED_PROBE if probe else ())
        for name in names:
            if is_arr(b.get(name)):
                pair = _packed_pair(b[name], shape, N)
                if pair is not None:
                    clean[name], garbage[name] = pair
                    touched = True
    return (clean, garbage) if touched else None


def _compare(name, out_c, out_g, test):
    r = RESULTS[name]
    lc, lg = _leaves(out_c), _leaves(out_g)
    if len(lc) != len(lg):
        r['error'] += 1
        return
    worst = None
    for a, c in zip(lc, lg):
        a, c = np.asarray(a), np.asarray(c)
        if a.shape != c.shape:
            worst = 'real'
            break
        if np.array_equal(a, c, equal_nan=True):
            continue
        diff = np.abs(a.astype(float) - c.astype(float))
        r['max_diff'] = max(r['max_diff'], float(np.nanmax(diff)))
        where = diff > 0
        kind = 'padding-like' if np.all(a[where] == 0) else 'real'
        worst = 'real' if (worst == 'real' or kind == 'real') else kind
    if worst is None:
        r['equal'] += 1
    else:
        r[worst] += 1
        if r['example'] is None:
            r['example'] = test


def wrap(name, fn):
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if BUSY or COUNTS[name] >= JUNK_MAX:
            return out
        try:
            b = sig.bind_partial(*args, **kwargs).arguments
        except TypeError:
            return out
        if any(is_jax(a) for v in b.values() for a in _leaves(v)) or any(is_jax(a) for a in _leaves(out)):
            return out
        try:
            variants = _variants(b, 'probe' in name or b.get('kind') == 'probe')
        except Exception as e:                                          # never disturb the test
            RESULTS[name]['variant-error'] = RESULTS[name].get('variant-error', 0) + 1
            if RESULTS[name]['example'] is None:
                RESULTS[name]['example'] = 'variant: %s: %s' % (type(e).__name__, str(e)[:80])
            return out
        test = os.environ.get('PYTEST_CURRENT_TEST', '?').rsplit(' (', 1)[0]
        if variants is None:
            RESULTS[name]['no-corruptible-input'] += 1
            return out
        COUNTS[name] += 1
        BUSY.append(1)
        try:
            ca, ga = variants
            bc, bg = sig.bind_partial(**ca), sig.bind_partial(**ga)
            out_c = fn(*bc.args, **bc.kwargs)
            out_g = fn(*bg.args, **bg.kwargs)
            RESULTS[name]['checked'] += 1
            _compare(name, out_c, out_g, test)
        except Exception as e:
            RESULTS[name]['error'] += 1
            if RESULTS[name]['example'] is None:
                RESULTS[name]['example'] = '%s: %s (%s)' % (type(e).__name__, str(e)[:80], test)
        finally:
            BUSY.pop()
        return out
    return wrapper


def install():
    for m in ('ut3_sampling', 'utv_sampling'):
        mod = importlib.import_module('t3toolbox.backend.' + m)
        for n in mod.__all__:
            f = getattr(mod, n)
            if inspect.isfunction(f):
                setattr(mod, n, wrap('%s.%s' % (m, n), f))
    sv = importlib.import_module('t3toolbox.backend.sampling_variables')
    sv.sampling_variables = wrap('sampling_variables.sampling_variables', sv.sampling_variables)


def pytest_configure(config):
    install()


def pytest_unconfigure(config):
    with open(os.path.join(OUT_DIR, 'junk_%d.json' % os.getpid()), 'w') as fh:
        json.dump(dict(RESULTS), fh)
