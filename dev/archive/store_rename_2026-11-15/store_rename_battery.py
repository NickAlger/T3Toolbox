"""The store rename's bitwise battery (plan ``dev/store_rename_plan_2026-11-10.md`` §5.1).

The same seeded calls run on two trees in one environment: the reference, a ``git worktree`` of ``v2026.4.0``
(R1), and the renamed tree (after R2, R3, R4b and R6). The rename changes no number, so every recorded array
must be BITWISE equal between the two runs. This is what proves the swap is a rename.

Usage (the tree under test is whatever ``t3toolbox`` is first on ``PYTHONPATH``)::

    PYTHONPATH=<tree> python dev/store_rename_battery.py b9-save --out <dir>          # old tree: B9's files
    PYTHONPATH=<tree> python dev/store_rename_battery.py run --out <dir> [--b9 <dir>]  # -> <dir>/battery.npz
    python dev/store_rename_battery.py compare <ref_dir> <new_dir>                    # the pass criterion

``run`` writes ONE file, ``<dir>/battery.npz``: every array under its key, every text under ``text:<key>`` (a 0-d
unicode array), and ``__tree_is_new__`` / ``__letters_new__`` (the adapter's two flags, below). Nothing else.

**Keys are physical.** Every record is keyed by its slot in the object's ``.data`` tuple (or by a slot index
``s0``..``s3`` = the down/Tucker store, the up/TT-shaped store, left, right), never by a store word, so the old and
the renamed tree produce the same key set. Attributes whose spelling changes are read through the adapter's
``attr`` by their NEW name and keyed by the slot they read.

**The adapter** is the only tree-dependent code:

- ``NEW``: ``'up_tt_cores' in {f.name for f in dataclasses.fields(bvf.T3Frame)}`` (``hasattr`` is False on both
  trees: a field without a default is no class attribute).
- ``LETTERS_NEW``: whether ``update`` is the 2026.5.0 letter tuple (D-2(c)), read from the default of
  ``padded_frames.tv_align_padded_frame``'s ``update`` (a ``str`` on the old letter API). This second flag is NOT in
  §5.1: after R2 and before R3 the tree is ``NEW`` but its letter API is still the old one (R2 leaves letter strings
  old by design), so ``NEW`` alone cannot pick ``upd``'s form.
- ``upd(stores)``: stores named in the NEW meaning (``'down'`` = the Tucker-factor store, ``'up'`` = the TT-shaped
  store, ``'left'``, ``'right'``), in the order given. The new letter API gets the tuple of new letters
  (``down->D, up->U, left->L, right->R``); the old one gets the old letter string (``down->U, up->O, left->L,
  right->R``). Never a tuple of old letters (form (c) gives them a new meaning).
- ``attr(obj, new_name)``: the NEW spelling on a ``NEW`` tree, else the old spelling (A1, A2 and A3 reversed).
- Widths are passed positionally (``from_t3frame(F, 8, 7, 5, 6, 3)``: N, the down store's, the up store's, rL, rR).

**Pass criterion** (``compare``): the key sets are identical; every array has the same dtype and shape and
``np.array_equal(a, b, equal_nan=True)`` (the uniform movements are NaN-padded); every text is equal, the reference's
texts first mapped by ``map_text_a_tiers`` from ``dev/store_rename_apply.py`` when the reference is an old tree and
the other a ``NEW`` tree. Host-clock fields are excluded by name, at record and at compare time: ``wall_time``
(``stats['history']``) and any key containing ``time``. Integer and boolean stats are arrays like any other.
Numeric history fields are stacked into one array per field (``None`` -> NaN when mixed with numbers).

Every call is guarded: an exception is recorded as the text ``error:<key>`` (type and message) and printed, so one
broken call does not hide the rest; a run with errors is reported as such and should be fixed before it is trusted.
"""
import argparse
import dataclasses
import importlib.util
import inspect
import itertools
import os
import sys
import time
import traceback
import typing as typ

import numpy as np

NDArray = np.ndarray

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.optimizers as topt
import t3toolbox.shared_geometry as tsg
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.padded_frames as padded_frames
import t3toolbox.backend.uniform_padded_frames as uniform_padded_frames
import t3toolbox.backend.fv_operations as fv_operations
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.ufv_masking as ufv_masking
import t3toolbox.backend.sharing as bsharing
import t3toolbox.backend.utv_operations as utv_operations
import t3toolbox.backend.tv_operations as tv_operations
import t3toolbox.backend.probing as probing
import t3toolbox.backend.sampling_derivatives as sampling_derivatives


# ============================================================================== the adapter

NEW = 'up_tt_cores' in {f.name for f in dataclasses.fields(bvf.T3Frame)}
LETTERS_NEW = not isinstance(
    inspect.signature(padded_frames.tv_align_padded_frame).parameters['update'].default, str)

_NEW_LETTER = {'down': 'D', 'up': 'U', 'left': 'L', 'right': 'R'}   # D-2(c), the new meaning
_OLD_LETTER = {'down': 'U', 'up': 'O', 'left': 'L', 'right': 'R'}   # 2026.4.0: 'U' Tucker store, 'O' TT-shaped store
_SLOT = {'down': 0, 'up': 1, 'left': 2, 'right': 3}                 # physical position in T3Frame.data

# NEW spelling -> 2026.4.0 spelling, for the attributes the battery reads (A1, A2 and A3 of §1.2, reversed)
_OLD_SPELLING = {
    'down_tucker_cores': 'up_tucker_cores',              # A1
    'up_tt_cores': 'down_tt_cores',
    'down_tucker_supercore': 'up_tucker_supercore',
    'up_tt_supercore': 'down_tt_supercore',
    'down_ranks': 'up_ranks', 'up_ranks': 'down_ranks',  # A2 swaps
    'down_mask': 'up_mask', 'up_mask': 'down_mask',
    'variations_down_mask': 'variations_up_mask', 'variations_up_mask': 'variations_down_mask',
    'down_weights': 'up_weights', 'up_weights': 'down_weights',
    'down_weight_supercore': 'up_weight_supercore', 'up_weight_supercore': 'down_weight_supercore',
    'nD': 'nU', 'nU': 'nD',                              # A3 swap
}


def upd(
        stores:  typ.Sequence[str],   # store words in the NEW meaning, any order, e.g. ('down', 'left')
):                                     # the tree's `update` value: new letter tuple, or the old letter string
    """The tree's ``update`` argument for these stores (see the module docstring)."""
    if LETTERS_NEW:
        return tuple(_NEW_LETTER[s] for s in stores)
    return ''.join(_OLD_LETTER[s] for s in stores)


def attr(
        obj:       typ.Any,
        new_name:  str,       # the 2026.5.0 spelling
) -> typ.Any:
    """``obj.<new_name>`` on a NEW tree, else the 2026.4.0 spelling of the same attribute."""
    return getattr(obj, new_name if NEW else _OLD_SPELLING.get(new_name, new_name))


def slots_key(
        stores:  typ.Sequence[str],   # store words in the NEW meaning
) -> str:                              # e.g. 's2+s0' (order kept: a permuted tuple is its own key)
    return '+'.join('s%d' % _SLOT[s] for s in stores)


# ============================================================================== recording

class Recorder:
    """``{key: array}`` and ``{key: text}``, flattened from nested results by position."""

    def __init__(self):
        self.arrays: typ.Dict[str, NDArray] = {}
        self.texts: typ.Dict[str, str] = {}
        self.errors: typ.List[str] = []

    def text(self, key: str, value: str) -> None:
        self._fresh(key)
        self.texts[key] = str(value)

    def _fresh(self, key: str) -> None:
        if key in self.arrays or key in self.texts:
            raise KeyError('battery key recorded twice: %r' % key)

    def rec(self, key: str, obj: typ.Any) -> None:
        """Record ``obj`` under ``key``: arrays and scalars as arrays, sequences by position, dicts by key,
        library objects by their ``.data`` (or a dataclass's fields in order), texts as texts."""
        if 'time' in key.rsplit('/', 1)[-1]:           # host-clock fields, excluded by name
            return
        if obj is None:
            self.text(key, 'None')
        elif isinstance(obj, str):
            self.text(key, obj)
        elif isinstance(obj, (bool, int, float, np.bool_, np.integer, np.floating)):
            self._fresh(key)
            self.arrays[key] = np.asarray(obj)
        elif hasattr(obj, 'shape') and hasattr(obj, 'dtype') and hasattr(obj, '__array__'):
            self._fresh(key)
            self.arrays[key] = np.array(np.asarray(obj))   # numpy or jax -> a host numpy copy
        elif isinstance(obj, dict):
            for k in sorted(obj, key=str):
                if k == 'history':
                    self.rec_history(key + '/history', obj[k])
                else:
                    self.rec(key + '/' + str(k), obj[k])
        elif isinstance(obj, (tuple, list)):
            if obj and all(isinstance(e, (bool, int, float, np.bool_, np.integer, np.floating)) for e in obj):
                self._fresh(key)
                self.arrays[key] = np.asarray(obj)
            else:
                self.text(key + '/len', str(len(obj)))
                for i, e in enumerate(obj):
                    self.rec('%s/%d' % (key, i), e)
        elif hasattr(obj, 'data') and not isinstance(obj, np.ndarray):
            self.rec(key, obj.data)
        elif dataclasses.is_dataclass(obj):
            for i, f in enumerate(dataclasses.fields(obj)):
                self.rec('%s/%d' % (key, i), getattr(obj, f.name))
        else:
            raise TypeError('battery cannot record %s at %r' % (type(obj).__name__, key))

    def rec_history(self, key: str, history: typ.Sequence[dict]) -> None:
        """The numeric history fields stacked into one array per field; texts joined; host clocks excluded."""
        names = sorted({k for h in history for k in h})
        self.text(key + '/len', str(len(history)))
        for name in names:
            if 'time' in name:
                continue
            vals = [h.get(name) for h in history]
            if any(isinstance(v, str) for v in vals) or all(v is None for v in vals):
                self.text(key + '/' + name, repr(vals))
            elif all(isinstance(v, (bool, np.bool_)) for v in vals):
                self._fresh(key + '/' + name)
                self.arrays[key + '/' + name] = np.asarray(vals, dtype=bool)
            elif all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in vals):
                self._fresh(key + '/' + name)
                self.arrays[key + '/' + name] = np.asarray(vals, dtype=np.int64)
            else:
                self._fresh(key + '/' + name)
                self.arrays[key + '/' + name] = np.asarray(
                    [np.nan if v is None else float(np.asarray(v)) for v in vals], dtype=np.float64)

    def call(self, key: str, fn: typ.Callable[[], typ.Any]) -> typ.Any:
        """``fn()`` recorded under ``key``; an exception becomes the text ``error:<key>`` (and is printed)."""
        try:
            out = fn()
        except Exception as e:
            msg = '%s: %s' % (type(e).__name__, e)
            self.text('error:' + key, msg)
            self.errors.append(key)
            print('[battery] ERROR at %s: %s' % (key, msg), file=sys.stderr)
            traceback.print_exc(limit=3, file=sys.stderr)
            return None
        self.rec(key, out)
        return out

    def save(self, path: str) -> None:
        payload = dict(self.arrays)
        for k, v in self.texts.items():
            payload['text:' + k] = np.asarray(v, dtype=np.str_)
        payload['__tree_is_new__'] = np.asarray(NEW)
        payload['__letters_new__'] = np.asarray(LETTERS_NEW)
        np.savez(path, **payload)


def seed() -> None:
    """Every fixture and every random draw starts from ``np.random.seed(0)`` (§5.1), so a key does not depend on
    the calls before it."""
    np.random.seed(0)


def randn_like(cores: typ.Sequence[NDArray]) -> typ.Tuple[NDArray, ...]:   # seeded, the same shapes
    seed()
    return tuple(np.random.randn(*c.shape) for c in cores)


# ============================================================================== fixtures

STORES = ('down', 'up', 'left', 'right')
SUBSETS = tuple(s for k in range(1, 5) for s in itertools.combinations(STORES, k))   # the 15 non-empty subsets
SHARING = (0, 0, 1)
PAD = (8, 7, 5, 6, 3)   # positional (N, down store's width, up store's width, rL, rR): all different
TK1, TT1 = (3, 5, 3), (1, 3, 3, 1)                       # the minimal target of x_old
TK_SLACK, TT_SLACK = (4, 5, 4), (1, 2, 3, 1)             # the slack target: stores (4, 5, 4) / (2, 5, 3)
TK_O, TT_O = (5, 5, 4, 3), (1, 3, 5, 3, 1)               # the slack old frame's target: (5, 5, 4, 3) / (3, 5, 4, 3)
TK_G1, TT_G1 = (6, 6, 3), (1, 4, 3, 1)                   # a shared padded target containing xG's frame (not in §5.1)


class Fixtures:
    """§5.1's fixtures, each built after ``np.random.seed(0)``."""

    def __init__(self):
        seed(); self.xS = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 6, 2), (1, 2, 5, 1))
        seed(); self.xC = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 6, 2), (1, 2, 5, 1), stack_shape=(2,))
        seed(); self.x_old = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
        self.x0 = self.x_old.resize((5, 6, 7), TK1, TT1)
        seed(); self.xG = t3.TuckerTensorTrain.randn((6, 6, 7), (3, 3, 2), (1, 3, 2, 1)).share(SHARING)
        seed(); self.y = t3.TuckerTensorTrain.randn((5, 6, 7), (5, 6, 7), (1, 5, 7, 1))
        seed(); self.yG = t3.TuckerTensorTrain.randn((6, 6, 7), (6, 6, 7), (1, 6, 7, 1))      # xG's hint/data source
        seed(); self.xO = t3.TuckerTensorTrain.randn((5, 6, 6, 5), (4, 3, 3, 2), (1, 2, 3, 2, 1))
        seed(); self.yO = t3.TuckerTensorTrain.randn((5, 6, 6, 5), (5, 6, 6, 5), (1, 5, 6, 5, 1))  # xO's hint
        seed()
        self.ww = [np.random.randn(4, n) for n in (5, 6, 7)]
        self.pp = [np.random.randn(4, n) for n in (5, 6, 7)]
        self.idx = np.stack([np.random.randint(0, n, size=(4,)) for n in (5, 6, 7)])
        self.zt = [np.random.randn(4, n) for n in (5, 6, 7)]            # probe residuals, W+(Ni,)
        self.zt_jets = [np.random.randn(3, 4, n) for n in (5, 6, 7)]    # residual jets, (order+1,)+W+(Ni,)
        seed()
        self.wwG = [np.random.randn(4, n) for n in (6, 6, 7)]
        self.data = self.y.probe(self.ww)
        self.dataG = self.yG.probe(self.wwG)
        # the frames and tangents the calls share
        self.F, self.V = bvf.t3_orthogonal_representations(self.xS)
        self.FG, self.VG = bvf.t3_orthogonal_representations(self.xG)
        self.FO, _ = bvf.t3_orthogonal_representations(self.xO)
        self.v = t3m.MANIFOLD.project_ambient(self.F, self.y)
        self.vG = t3m.MANIFOLD.project_ambient(self.FG, self.yG)
        self.ux = ut3.UniformTuckerTensorTrain.from_t3(self.xS)
        self.uxC = ut3.UniformTuckerTensorTrain.from_t3(self.xC)
        self.ux_old = ut3.UniformTuckerTensorTrain.from_t3(self.x_old)
        self.uy = ut3.UniformTuckerTensorTrain.from_t3(self.y, N=self.ux_old.N)
        self.uxG = ut3.UniformTuckerTensorTrain.from_t3(self.xG)
        self.uyG = ut3.UniformTuckerTensorTrain.from_t3(self.yG, N=self.uxG.N)
        self.uxO = ut3.UniformTuckerTensorTrain.from_t3(self.xO)
        self.uyO = ut3.UniformTuckerTensorTrain.from_t3(self.yO, N=self.uxO.N)


def per_store_frame_weights(F: 'bvf.T3Frame') -> 'bvf.T3FrameWeights':
    """Seeded positive weights built per store from ``F``'s ranks (§5.2), positionally, so a swapped weight store
    shows: slot 0 on the TT variations' middle leg (the down store's width), slot 1 on the Tucker variations' rows
    (the up store's width), then the variation bonds."""
    tucker_holes, tt_holes = F.variation_shapes
    sizes = (tuple(h[1] for h in tt_holes), tuple(h[0] for h in tucker_holes),
             tuple(h[0] for h in tt_holes), tuple(h[2] for h in tt_holes))
    seed()
    return bvf.T3FrameWeights(*(tuple(np.abs(np.random.randn(n)) + 0.5 for n in fam) for fam in sizes))


# ============================================================================== B1 .. B8

def b1_representations(R: Recorder, X: Fixtures) -> None:
    """B1: representations and conversions."""
    for name in ('xS', 'xC', 'xG'):
        R.call('B1/orth_reps/' + name, lambda name=name: bvf.t3_orthogonal_representations(getattr(X, name)))
    R.call('B1/T3Frame.from_t3/xS', lambda: bvf.T3Frame.from_t3(X.xS))
    R.call('B1/MANIFOLD.frame/xS', lambda: t3m.MANIFOLD.frame(X.xS))
    R.call('B1/random_orthogonal', lambda: (seed(), bvf.T3Frame.random_orthogonal(
        (5, 6, 7), (3, 6, 2), (1, 2, 5, 1), (2,)))[1])
    R.call('B1/t3svd_orth_reps/xS', lambda: bvf.t3svd_orthogonal_representations(X.xS))
    R.call('B1/zero_point_frame/xS', lambda: fv_operations.fv_zero_point_frame(X.F.data))
    # with groups: xS's modes have different sizes, so the grouped draw takes xG's frame (equal sizes in the group)
    R.call('B1/zero_point_frame/xG_groups', lambda: fv_operations.fv_zero_point_frame(
        X.FG.data, groups=bsharing.canonical_groups(SHARING, X.xG.shape)))
    R.call('B1/ut3_orth_reps/ux', lambda: ubv.ut3_orthogonal_representations(X.ux))
    R.call('B1/ut3_orth_reps/uxC', lambda: ubv.ut3_orthogonal_representations(X.uxC))
    UF = R.call('B1/UT3Frame.from_t3frame/pad', lambda: ubv.UT3Frame.from_t3frame(X.F, *PAD))
    R.call('B1/UT3Frame.to_t3frame/pad', lambda: UF.to_t3frame())
    R.call('B1/UT3Variations.from_t3variations/pad', lambda: ubv.UT3Variations.from_t3variations(X.V, *PAD))
    R.call('B1/UT3Frame.from_t3frame/default', lambda: ubv.UT3Frame.from_t3frame(X.F))
    R.call('B1/UT3Variations.from_t3variations/default', lambda: ubv.UT3Variations.from_t3variations(X.V))


def b2_attributes(R: Recorder, X: Fixtures) -> None:
    """B2: the renamed attributes, read by their NEW name through ``attr`` and keyed by physical slot."""
    F = X.F
    for i, name in enumerate(('down_tucker_cores', 'up_tt_cores', 'left_tt_cores', 'right_tt_cores')):
        R.call('B2/T3Frame/cores/s%d' % i, lambda name=name: attr(F, name))
    for i, name in enumerate(('down_ranks', 'up_ranks', 'left_ranks', 'right_ranks')):
        R.call('B2/T3Frame/ranks/s%d' % i, lambda name=name: attr(F, name))
    R.text('B2/T3Frame/repr', repr(F))

    def weights_block(tag: str, W: 'bvf.T3FrameWeights') -> None:
        for i, name in enumerate(('down_weights', 'up_weights', 'left_weights', 'right_weights')):
            R.call('B2/T3FrameWeights/%s/fields/s%d' % (tag, i), lambda name=name: attr(W, name))
        for i, name in enumerate(('down_ranks', 'up_ranks', 'left_ranks', 'right_ranks')):
            R.call('B2/T3FrameWeights/%s/ranks/s%d' % (tag, i), lambda name=name: attr(W, name))
        R.text('B2/T3FrameWeights/%s/repr' % tag, repr(W))

    FWsvd = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(X.xS))   # as §5.1 writes it (equal stores)
    weights_block('from_t3svd', FWsvd)
    FW = per_store_frame_weights(F)                                              # unequal stores (§5.2)
    weights_block('per_store', FW)

    UF = ubv.UT3Frame.from_t3frame(F, *PAD)
    for i, name in enumerate(('down_tucker_supercore', 'up_tt_supercore', 'left_tt_supercore', 'right_tt_supercore')):
        R.call('B2/UT3Frame/supercores/s%d' % i, lambda name=name: attr(UF, name))
    for i, name in enumerate(('down_ranks', 'up_ranks', 'left_ranks', 'right_ranks')):
        R.call('B2/UT3Frame/ranks/s%d' % i, lambda name=name: attr(UF, name))
    for i, name in enumerate(('nD', 'nU', 'rL', 'rR')):
        R.call('B2/UT3Frame/widths/s%d' % i, lambda name=name: attr(UF, name))
    for i, name in enumerate(('down_mask', 'up_mask', 'frame_left_mask', 'frame_right_mask')):
        R.call('B2/UT3Frame/masks/s%d' % i, lambda name=name: attr(UF.masks, name))
    R.text('B2/UT3Frame/repr', repr(UF))

    UV = ubv.UT3Variations.from_t3variations(X.V, *PAD)
    for i, name in enumerate(('down_ranks', 'up_ranks', 'variation_left_ranks', 'variation_right_ranks')):
        R.call('B2/UT3Variations/ranks/s%d' % i, lambda name=name: attr(UV, name))
    for i, name in enumerate(('nD', 'nU', 'rL', 'rR')):
        R.call('B2/UT3Variations/widths/s%d' % i, lambda name=name: attr(UV, name))
    for i, name in enumerate(('variations_down_mask', 'variations_up_mask', 'variations_left_mask',
                              'variations_right_mask')):
        R.call('B2/UT3Variations/masks/s%d' % i, lambda name=name: attr(UV.masks, name))
    R.text('B2/UT3Variations/repr', repr(UV))

    UFW = ubv.UT3FrameWeights.from_t3frameweights(FW, *PAD[1:])   # positional (down, up, rL, rR) widths
    for i, name in enumerate(('down_weight_supercore', 'up_weight_supercore', 'left_weight_supercore',
                              'right_weight_supercore')):
        R.call('B2/UT3FrameWeights/supercores/s%d' % i, lambda name=name: attr(UFW, name))
    for i, name in enumerate(('down_ranks', 'up_ranks', 'left_ranks', 'right_ranks')):
        R.call('B2/UT3FrameWeights/ranks/s%d' % i, lambda name=name: attr(UFW, name))
    for i, name in enumerate(('nD', 'nU', 'rL', 'rR')):
        R.call('B2/UT3FrameWeights/widths/s%d' % i, lambda name=name: attr(UFW, name))
    R.text('B2/UT3FrameWeights/repr', repr(UFW))
    R.text('B2/T3Tangent/repr', repr(X.v))


def b3_ranks_and_masks(R: Recorder, X: Fixtures) -> None:
    """B3: rank and mask functions, positional."""
    R.call('B3/orth_rep_ranks/xS', lambda: ranks.compute_orthogonal_representation_ranks(
        (5, 6, 7), (3, 6, 2), (1, 2, 5, 1)))
    R.call('B3/orth_rep_ranks/stacked', lambda: ranks.compute_orthogonal_representation_ranks(
        (5, 6, 7), np.array([[3, 2], [6, 4], [2, 2]]), np.array([[1, 1], [2, 2], [5, 3], [1, 1]])))
    stores = (X.F.structure[1], X.F.structure[2], X.F.structure[3], X.F.structure[4])   # physical order
    F0 = bvf.t3_orthogonal_representations(X.x0)[0]
    stores0 = tuple(F0.structure[1:5])
    R.call('B3/frame_has_minimal_ranks/xS', lambda: ranks.frame_has_minimal_ranks((5, 6, 7), *stores, None))
    R.call('B3/frame_has_minimal_ranks/x0', lambda: ranks.frame_has_minimal_ranks((5, 6, 7), *stores0, None))
    R.call('B3/ufv_make_frame_masks', lambda: ufv_masking.ufv_make_frame_masks(
        *(np.asarray(s) for s in stores), *PAD[1:]))
    for args in ((8, 6, 5), (7, 6, 2), (5, 9, 4)):
        R.call('B3/ufv_frame_widths/%d_%d_%d' % args, lambda args=args: ufv_masking.ufv_frame_widths(*args))


def b4_orthogonalizations(R: Recorder, X: Fixtures) -> None:
    """B4: orthogonalizations and SVD, ragged and uniform."""
    for tag, x in (('ragged', X.xS), ('uniform', X.ux)):
        for m in ('down_orthogonalize_tucker_cores', 'up_orthogonalize_tt_cores', 'left_orthogonalize_tt_cores',
                  'right_orthogonalize_tt_cores', 't3svd'):
            R.call('B4/%s/%s' % (tag, m), lambda x=x, m=m: getattr(x, m)())
        R.call('B4/%s/t3svd_max_tt_2' % tag, lambda x=x: x.t3svd(max_tt_ranks=2))
    R.call('B4/xG/t3svd', lambda: X.xG.t3svd())


def b5_tangents(R: Recorder, X: Fixtures) -> None:
    """B5: tangents, weights, the uniform twins, the shared and tied paths."""
    M, UM = t3m.MANIFOLD, ut3m.UNIFORM_MANIFOLD
    v = X.v
    R.rec('B5/v', v)
    F0 = R.call('B5/frame_x0', lambda: M.frame(X.x0))
    R.call('B5/retract', lambda: M.retract(v))
    R.call('B5/transport', lambda: M.transport(v, F0))
    R.call('B5/inner', lambda: M.inner(v, v))
    R.call('B5/norm', lambda: M.norm(v))
    R.call('B5/to_dense', lambda: v.to_dense())
    R.call('B5/to_t3', lambda: v.to_t3())
    R.call('B5/gauge_residual', lambda: v.gauge_residual)          # a cached property, not a method
    vK = R.call('B5/K/stack', lambda: t3m.T3Tangent.stack_tangents((v, v * 2)))
    R.call('B5/K/to_dense', lambda: vK.to_dense())
    R.call('B5/K/inner', lambda: M.inner(vK, vK))
    R.call('B5/K/retract', lambda: M.retract(vK))
    FWsvd_F, FWsvd_V, FWsvd_W = bvf.t3svd_orthogonal_representations(X.xS)
    R.call('B5/absorb/from_t3svd', lambda: bvf.fv_absorb_weights(
        FWsvd_V, bvf.T3FrameWeights.from_t3weights(FWsvd_W)))
    FW = per_store_frame_weights(X.F)
    R.call('B5/absorb/per_store', lambda: bvf.fv_absorb_weights(v.variations, FW))
    R.call('B5/COREWISE.frame/xS', lambda: t3m.COREWISE.frame(X.xS))

    # uniform twins
    uv = R.call('B5/u/from_t3tangent/pad', lambda: ut3m.UT3Tangent.from_t3tangent(v, *PAD))
    R.call('B5/u/from_t3tangent/default', lambda: ut3m.UT3Tangent.from_t3tangent(v))
    ux0 = ut3.UniformTuckerTensorTrain.from_t3(X.x0, N=PAD[0])
    UF0 = R.call('B5/u/frame_x0', lambda: UM.frame(ux0))
    R.call('B5/u/retract', lambda: UM.retract(uv))
    R.call('B5/u/transport', lambda: UM.transport(uv, UF0))
    R.call('B5/u/inner', lambda: UM.inner(uv, uv))
    R.call('B5/u/norm', lambda: UM.norm(uv))
    R.call('B5/u/to_dense', lambda: uv.to_dense())
    R.call('B5/u/to_ut3', lambda: uv.to_ut3())
    R.call('B5/u/gauge_residual', lambda: uv.gauge_residual)
    uvK = R.call('B5/u/K/stack', lambda: ut3m.UT3Tangent.stack_tangents((uv, uv * 2)))
    R.call('B5/u/K/to_dense', lambda: uvK.to_dense())
    R.call('B5/u/K/inner', lambda: UM.inner(uvK, uvK))
    R.call('B5/u/K/retract', lambda: UM.retract(uvK))
    UFW = ubv.UT3FrameWeights.from_t3frameweights(FW, *PAD[1:])
    R.call('B5/u/absorb/per_store', lambda: ubv.ufv_absorb_weights(uv.variations, UFW))
    # the stacked uniform point uxC: its own tangent space, the point projected onto it
    UFC = ubv.ut3_orthogonal_representations(X.uxC)[0]
    uvC = R.call('B5/uC/project_ambient', lambda: UM.project_ambient(UFC, X.uxC))
    R.call('B5/uC/to_dense', lambda: uvC.to_dense())
    R.call('B5/uC/inner', lambda: UM.inner(uvC, uvC))
    R.call('B5/uC/retract', lambda: UM.retract(uvC))

    # shared and tied paths: a uniform frame of xG with unequal padded widths (pad = 1, utv_operations.py:230-254)
    groups = bsharing.validate_sharing(SHARING, X.xG.shape)   # SharedGeometry.groups' form
    UFG = R.call('B5/tied/UFG', lambda: ubv.UT3Frame.from_t3frame(X.FG, None, 7, 6))
    UVG = ubv.UT3Variations.from_t3variations(X.vG.variations, None, 7, 6)
    sd = R.call('B5/tied/ufv_shared_frame_data', lambda: bsharing.ufv_shared_frame_data(UFG.data, groups))
    gauged = R.call('B5/tied/utv_orthogonal_gauge_projection', lambda: utv_operations.utv_orthogonal_gauge_projection(
        UFG.data, UVG.data, shared_data=sd))
    R.call('B5/tied/utv_retract', lambda: utv_operations.utv_retract(UFG.data, gauged, shared_data=sd))
    SM = tsg.shared(M, SHARING)
    FGs = R.call('B5/shared/frame', lambda: SM.frame(X.xG))
    pv = R.call('B5/shared/project', lambda: SM.project(M.project_ambient(FGs, X.yG)))
    R.call('B5/shared/retract', lambda: SM.retract(pv))
    SU = tsg.shared(UM, SHARING)
    UFGs = R.call('B5/ushared/frame', lambda: SU.frame(X.uxG))
    upv = R.call('B5/ushared/project', lambda: SU.project(UM.project_ambient(UFGs, X.uyG)))
    R.call('B5/ushared/retract', lambda: SU.retract(upv))


def b6_sampling(R: Recorder, X: Fixtures) -> None:
    """B6: sampling on tangents and points, ragged and uniform, and the 14 keyword-renamed kernels positionally."""
    v, F, ww, pp = X.v, X.F, X.ww, X.pp
    for tag, tan, Fr, cls in (('ragged', v, F, t3m.T3Tangent),
                              ('uniform', ut3m.UT3Tangent.from_t3tangent(v, *PAD),
                               ubv.UT3Frame.from_t3frame(F, *PAD), ut3m.UT3Tangent)):
        R.call('B6/%s/probe' % tag, lambda tan=tan: tan.probe(ww))
        R.call('B6/%s/apply' % tag, lambda tan=tan: tan.apply(ww))
        R.call('B6/%s/entries' % tag, lambda tan=tan: tan.entries(X.idx))
        R.call('B6/%s/probe_derivatives' % tag, lambda tan=tan: tan.probe_derivatives(ww, pp, 2))
        R.call('B6/%s/probe_transpose' % tag, lambda cls=cls, Fr=Fr: cls.probe_transpose(X.zt, ww, Fr))
        R.call('B6/%s/probe_derivatives_transpose' % tag,
               lambda cls=cls, Fr=Fr: cls.probe_derivatives_transpose(X.zt_jets, ww, pp, Fr, 2))
    R.call('B6/ragged/point_probe_derivatives', lambda: X.xS.probe_derivatives(ww, pp, 2))
    R.call('B6/uniform/point_probe_derivatives', lambda: X.ux.probe_derivatives(ww, pp, 2))

    P, S = probing, sampling_derivatives
    U0, U1, U2, U3 = F.data                   # physical slots: Tucker store, TT-shaped store, left, right
    Vt, Ht = v.variations.data
    # probing, the frame chain at F (tv_precompute_probe_frame_sweep, tv_probe_from_sweep, the transpose)
    xis = R.call('B6/k/compute_xi', lambda: P.compute_xi(U0, ww))
    mus, nus = P.compute_mu(U2, xis), P.compute_nu(U3, xis)
    R.call('B6/k/compute_eta', lambda: P.compute_eta(U1, mus, nus))
    dxis = P.compute_dxi(Vt, ww)
    R.call('B6/k/compute_sigma', lambda: P.compute_sigma(Ht, U3, U1, xis, dxis, mus))
    R.call('B6/k/compute_tau', lambda: P.compute_tau(Ht, U2, U1, xis, dxis, nus))
    deta_t = R.call('B6/k/compute_deta_tilde', lambda: P.compute_deta_tilde(U0, X.zt))
    tau_t = P.compute_tau_tilde(deta_t, U2, xis, mus)
    sigma_t = P.compute_sigma_tilde(deta_t, U3, xis, nus)
    R.call('B6/k/compute_dxi_tilde', lambda: P.compute_dxi_tilde(sigma_t, tau_t, U1, mus, nus))
    # probing, the corewise chain at (B, G, G, G) (t3_probe_corewise_hessian_from_sweep's steps 1-5)
    tk, tt = X.xS.data
    tkd, ttd = randn_like(tk), randn_like(tt)
    cxis = P.compute_xi(tk, ww)
    cmus, cnus = P.compute_mu(tt, cxis), P.compute_nu(tt, cxis)
    cdeta_t = P.compute_deta_tilde(tk, X.zt)
    ctau_t = P.compute_tau_tilde(cdeta_t, tt, cxis, cmus)
    csigma_t = P.compute_sigma_tilde(cdeta_t, tt, cxis, cnus)
    cxis_dot = P.compute_dxi(tkd, ww)
    cmus_dot = P.compute_sigma(ttd, tt, tt, cxis, cxis_dot, cmus)
    cnus_dot = P.compute_tau(ttd, tt, tt, cxis, cxis_dot, cnus)
    cdeta_t_dot = R.call('B6/k/compute_deta_tilde_tangent', lambda: P.compute_deta_tilde_tangent(tk, tkd, X.zt))
    ctau_t_dot = P.compute_tau_tilde_tangent(tt, ttd, cxis, cxis_dot, cdeta_t, cdeta_t_dot, cmus, cmus_dot, ctau_t)
    csigma_t_dot = P.compute_sigma_tilde_tangent(tt, ttd, cxis, cxis_dot, cdeta_t, cdeta_t_dot, cnus, cnus_dot,
                                                 csigma_t)
    R.call('B6/k/compute_dxi_tilde_tangent', lambda: P.compute_dxi_tilde_tangent(
        tt, ttd, cmus, cmus_dot, cnus, cnus_dot, csigma_t, csigma_t_dot, ctau_t, ctau_t_dot))

    # sampling_derivatives, the frame chain at F (order 2)
    order = 2
    xi_j = S.build_input_jets(P.compute_xi(U0, ww), P.compute_xi(U0, pp))
    mu_j, nu_j = S.compute_mu_jets(U2, xi_j, order), S.compute_nu_jets(U3, xi_j, order)
    dxi_j = S.build_input_jets(P.compute_xi(Vt, ww), P.compute_xi(Vt, pp))
    R.call('B6/k/compute_sigma_jets', lambda: S.compute_sigma_jets(Ht, U3, U1, xi_j, dxi_j, mu_j))
    R.call('B6/k/compute_tau_jets', lambda: S.compute_tau_jets(Ht, U2, U1, xi_j, dxi_j, nu_j))
    deta_tj = R.call('B6/k/compute_deta_tilde_jets', lambda: S.compute_deta_tilde_jets(U0, X.zt_jets))
    tau_tj = S.compute_tau_tilde_jets(U2, xi_j, deta_tj, mu_j)
    sigma_tj = S.compute_sigma_tilde_jets(U3, xi_j, deta_tj, nu_j)
    R.call('B6/k/compute_dxi_tilde_jets', lambda: S.compute_dxi_tilde_jets(U1, mu_j, nu_j, sigma_tj, tau_tj))
    # sampling_derivatives, the corewise chain (t3_probe_corewise_derivatives_hessian_from_sweep's steps 1-5)
    cxi_j = S.build_input_jets(P.compute_xi(tk, ww), P.compute_xi(tk, pp))
    cmu_j, cnu_j = S.compute_mu_jets(tt, cxi_j, order), S.compute_nu_jets(tt, cxi_j, order)
    cdeta_tj = S.compute_deta_tilde_jets(tk, X.zt_jets)
    ctau_tj = S.compute_tau_tilde_jets(tt, cxi_j, cdeta_tj, cmu_j)
    csigma_tj = S.compute_sigma_tilde_jets(tt, cxi_j, cdeta_tj, cnu_j)
    cxi_j_dot = S.build_input_jets(P.compute_xi(tkd, ww), P.compute_xi(tkd, pp))
    cmu_j_dot = S.compute_sigma_jets(ttd, tt, tt, cxi_j, cxi_j_dot, cmu_j)
    cnu_j_dot = S.compute_tau_jets(ttd, tt, tt, cxi_j, cxi_j_dot, cnu_j)
    cdeta_tj_dot = R.call('B6/k/compute_deta_tilde_tangent_jets',
                          lambda: S.compute_deta_tilde_tangent_jets(tk, tkd, X.zt_jets))
    ctau_tj_dot = S.compute_tau_tilde_tangent_jets(tt, ttd, cxi_j, cxi_j_dot, cdeta_tj, cdeta_tj_dot, cmu_j,
                                                   cmu_j_dot, ctau_tj)
    csigma_tj_dot = S.compute_sigma_tilde_tangent_jets(tt, ttd, cxi_j, cxi_j_dot, cdeta_tj, cdeta_tj_dot, cnu_j,
                                                       cnu_j_dot, csigma_tj)
    R.call('B6/k/compute_dxi_tilde_tangent_jets', lambda: S.compute_dxi_tilde_tangent_jets(
        tt, ttd, cmu_j, cmu_j_dot, cnu_j, cnu_j_dot, csigma_tj, csigma_tj_dot, ctau_tj, ctau_tj_dot))


def _old_ranks(frame_data: typ.Tuple) -> typ.Tuple[typ.Tuple[int, ...], typ.Tuple[int, ...]]:
    """The old blocks' extents the backend loop passes its step: the Tucker store's widths and the left bonds."""
    tucker = tuple(int(c.shape[-2]) for c in frame_data[0])
    left = frame_data[2]
    return tucker, tuple(int(c.shape[-3]) for c in left) + (int(left[-1].shape[-1]),)


def b7_padded_frames(R: Recorder, X: Fixtures) -> None:
    """B7: padded frames, the alignment step over every store subset, the loops, ``padded_start``."""
    M, UM = t3m.MANIFOLD, ut3m.UNIFORM_MANIFOLD
    old_tk, old_tt = X.x_old.tucker_ranks, X.x_old.tt_ranks
    targets = (('min', TK1, TT1), ('slack', TK_SLACK, TT_SLACK))
    for tname, tk, tt in targets:
        Fp = R.call('B7/T3Frame.padded/%s' % tname, lambda tk=tk, tt=tt: bvf.T3Frame.padded(X.x_old, tk, tt))
        UFp, split = R.call('B7/UT3Frame.padded/%s' % tname,
                            lambda tk=tk, tt=tt: ubv.UT3Frame.padded(X.ux_old, tk, tt))
        vp = M.project_ambient(Fp, X.y)
        uvp = UM.project_ambient(UFp, X.uy)
        for stores in SUBSETS + (('left', 'down'),):              # the 15 subsets, then the permuted ('L', 'D')
            key = slots_key(stores)
            R.call('B7/align/ragged/%s/%s' % (tname, key),
                   lambda stores=stores: vp.align_padded(old_tk, old_tt, update=upd(stores)))
            R.call('B7/align/uniform/%s/%s' % (tname, key),
                   lambda stores=stores: uvp.align_padded(split, update=upd(stores)))
    R.call('B7/T3Frame.padded/sharing', lambda: bvf.T3Frame.padded(X.xG, TK_G1, TT_G1, sharing=SHARING))

    for m, a in ((0, True), (3, True), (3, False), (10, True)):
        R.call('B7/front_loop/ragged/%d_%s' % (m, a), lambda m=m, a=a: bvf.fv_aligned_padded_frame(
            X.x_old, TK1, TT1, X.y, max_passes=m, alternate=a))
        R.call('B7/front_loop/uniform/%d_%s' % (m, a), lambda m=m, a=a: ubv.ufv_aligned_padded_frame(
            X.ux_old, TK1, TT1, X.uy, max_passes=m, alternate=a))
    R.call('B7/front_loop/uniform/jit', lambda: ubv.ufv_aligned_padded_frame(
        X.ux_old, TK1, TT1, X.uy, use_jit=True))

    # the slack OLD frame (xO's orthogonal representation, stores (4,3,3,2)/(2,3,3,2)): backend step and loop
    old = X.FO.data
    o_tk, o_tt = _old_ranks(old)
    project = tv_operations.tv_project_t3_onto_tangent_space
    hint = X.yO.data
    start = R.call('B7/slack_old/start', lambda: padded_frames.fv_padded_frame(old, TK_O, TT_O))
    var = project(start, hint)
    for stores in (STORES, ('down', 'left'), ('up', 'right')):
        R.call('B7/slack_old/step/%s' % slots_key(stores), lambda stores=stores: padded_frames.tv_align_padded_frame(
            start, var, o_tk, o_tt, update=upd(stores)))
    R.call('B7/slack_old/loop', lambda: padded_frames.fv_aligned_padded_frame(old, TK_O, TT_O, hint, project))
    # its uniform twin
    uold = ubv.ut3_orthogonal_representations(X.uxO)[0].data
    uproject = utv_operations.utv_project_ut3_onto_tangent_space
    uhint = X.uyO.data
    ustart, usplit = R.call('B7/uslack_old/start', lambda: uniform_padded_frames.ufv_padded_frame(uold, TK_O, TT_O))
    uvar = uproject(ustart, uhint)
    for stores in (STORES, ('down', 'left'), ('up', 'right')):
        R.call('B7/uslack_old/step/%s' % slots_key(stores),
               lambda stores=stores: uniform_padded_frames.utv_align_padded_frame(
                   ustart, uvar, usplit, update=upd(stores)))
    widths = ufv_masking.ufv_frame_widths(X.uxO.N, max(TK_O), max(TT_O))
    R.call('B7/uslack_old/loop', lambda: uniform_padded_frames.ufv_aligned_padded_frame(
        uold, TK_O, TT_O, uhint, uproject, widths))
    R.call('B7/uslack_old/loop_jit', lambda: uniform_padded_frames.ufv_aligned_padded_frame(
        uold, TK_O, TT_O, uhint, uproject, widths, use_jit=True))

    setup = topt.least_squares_problem(M, 'probe', X.ww, X.data, X.x0)
    R.call('B7/padded_start', lambda: bopt.padded_start(setup.problem, setup.x0_cores, X.x_old.data))


def b8_optimizers(R: Recorder, X: Fixtures) -> None:
    """B8: the optimizers; ``x.data``, the movements and every non-clock stat."""
    M, UM = t3m.MANIFOLD, ut3m.UNIFORM_MANIFOLD
    args = ('probe', X.ww, X.data)
    R.call('B8/newton', lambda: topt.newton_cg(M, *args, X.x0, max_newton=3))
    R.call('B8/newton_padded', lambda: topt.newton_cg(M, *args, X.x0, max_newton=3, padded_from=X.x_old))
    R.call('B8/newton_cone', lambda: topt.newton_cg(M, *args, X.x0, max_newton=3, g0_reference='cone'))
    R.call('B8/newton_exact', lambda: topt.newton_cg(M, *args, X.x0, max_newton=3, hessian='exact'))
    ux0 = ut3.UniformTuckerTensorTrain.from_t3(X.x0)
    uold = ut3.UniformTuckerTensorTrain.from_t3(X.x_old, N=ux0.N)
    R.call('B8/unewton_padded', lambda: topt.newton_cg(UM, *args, ux0, max_newton=3, padded_from=uold))
    R.call('B8/newton_shared', lambda: topt.newton_cg(
        tsg.shared(M, SHARING), 'probe', X.wwG, X.dataG, X.xG, max_newton=3))
    R.call('B8/gd_cone', lambda: topt.gradient_descent(M, *args, X.x0, g0_reference='cone'))


# ============================================================================== B9 files across versions

B9_FILES = ('T3Frame', 'T3Variations', 'T3Tangent', 'UT3Frame', 'UT3Variations')


def _b9_objects(X: Fixtures) -> typ.Dict[str, typ.Any]:
    return {'T3Frame': X.F, 'T3Variations': X.V, 'T3Tangent': X.v,
            'UT3Frame': ubv.UT3Frame.from_t3frame(X.F, *PAD),
            'UT3Variations': ubv.UT3Variations.from_t3variations(X.V, *PAD)}


def b9_save(out: str) -> None:
    """The old tree saves the five objects of xS; the new tree loads them in ``run --b9``."""
    os.makedirs(out, exist_ok=True)
    X = Fixtures()
    for name, obj in _b9_objects(X).items():
        obj.save(os.path.join(out, name + '.npz'))
    print('[battery] B9 files saved to %s (tree NEW=%s)' % (out, NEW))


def b9_load(R: Recorder, X: Fixtures, b9_dir: str) -> None:
    """B9: load the other tree's files; their ``.data`` must equal this tree's own B1/B5 objects."""
    loaders = {'T3Frame': bvf.T3Frame.load, 'T3Variations': bvf.T3Variations.load, 'T3Tangent': t3m.T3Tangent.load,
               'UT3Frame': ubv.UT3Frame.load, 'UT3Variations': ubv.UT3Variations.load}
    own = _b9_objects(X)
    for name in B9_FILES:
        loaded = R.call('B9/%s' % name, lambda name=name: loaders[name](os.path.join(b9_dir, name + '.npz')))
        if loaded is None:
            continue
        a, b = Recorder(), Recorder()
        a.rec('x', loaded)
        b.rec('x', own[name])
        same = (a.arrays.keys() == b.arrays.keys() and a.texts == b.texts
                and all(a.arrays[k].dtype == b.arrays[k].dtype and a.arrays[k].shape == b.arrays[k].shape
                        and np.array_equal(a.arrays[k], b.arrays[k], equal_nan=True) for k in a.arrays))
        R.rec('B9/%s/equals_own' % name, bool(same))
        if not same:
            R.errors.append('B9/%s/equals_own' % name)
            print('[battery] ERROR: B9 %s loaded != own' % name, file=sys.stderr)


# ============================================================================== run / compare

GROUPS = (('B1', b1_representations), ('B2', b2_attributes), ('B3', b3_ranks_and_masks),
          ('B4', b4_orthogonalizations), ('B5', b5_tangents), ('B6', b6_sampling), ('B7', b7_padded_frames),
          ('B8', b8_optimizers))


def run(out: str, b9_dir: typ.Optional[str]) -> int:
    import t3toolbox
    print('[battery] tree: %s  NEW=%s  LETTERS_NEW=%s' % (os.path.dirname(t3toolbox.__file__), NEW, LETTERS_NEW))
    R = Recorder()
    X = Fixtures()
    for name, fn in GROUPS:
        t0 = time.time()
        before = len(R.arrays) + len(R.texts)
        fn(R, X)
        print('[battery] %s: %d keys, %.1f s' % (name, len(R.arrays) + len(R.texts) - before, time.time() - t0))
    if b9_dir is not None:
        b9_load(R, X, b9_dir)
    os.makedirs(out, exist_ok=True)
    R.save(os.path.join(out, 'battery.npz'))
    print('[battery] %d arrays, %d texts -> %s; %d errors%s'
          % (len(R.arrays), len(R.texts), os.path.join(out, 'battery.npz'), len(R.errors),
             (': ' + ', '.join(R.errors)) if R.errors else ''))
    return 1 if R.errors else 0


def _load(d: str) -> typ.Tuple[typ.Dict[str, NDArray], typ.Dict[str, str], bool]:
    with np.load(os.path.join(d, 'battery.npz'), allow_pickle=False) as z:
        new = bool(z['__tree_is_new__'])
        arrays, texts = {}, {}
        for k in z.files:
            if k.startswith('__'):
                continue
            if k.startswith('text:'):
                texts[k[5:]] = str(z[k])
            else:
                arrays[k] = z[k]
    return arrays, texts, new


def _map_text_a_tiers() -> typ.Callable[[str], str]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'store_rename_apply.py')
    if not os.path.exists(path):
        raise RuntimeError('compare: an old reference against a NEW tree needs map_text_a_tiers from %s, which '
                           'does not exist' % path)
    spec = importlib.util.spec_from_file_location('store_rename_apply', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'map_text_a_tiers'):
        raise RuntimeError('compare: %s defines no map_text_a_tiers(text: str) -> str' % path)
    return mod.map_text_a_tiers


def compare(ref_dir: str, new_dir: str) -> int:
    """The §5.1 pass criterion. Returns 0 on PASS."""
    ra, rt, r_new = _load(ref_dir)
    na, nt, n_new = _load(new_dir)
    if r_new and not n_new:
        print('[compare] the reference is a NEW tree and the other an old one: swap the arguments')
        return 2
    mapper = _map_text_a_tiers() if (n_new and not r_new) else (lambda s: s)
    clock = lambda k: 'time' in k.rsplit('/', 1)[-1]
    problems = []
    for kind, a, b in (('array', ra, na), ('text', rt, nt)):
        ka = {k for k in a if not clock(k)}
        kb = {k for k in b if not clock(k)}
        for k in sorted(ka - kb):
            problems.append('%s key only in the reference: %s' % (kind, k))
        for k in sorted(kb - ka):
            problems.append('%s key only in the new run: %s' % (kind, k))
        for k in sorted(ka & kb):
            if kind == 'array':
                x, y = a[k], b[k]
                if x.dtype != y.dtype or x.shape != y.shape:
                    problems.append('%s: dtype/shape %s%s vs %s%s' % (k, x.dtype, x.shape, y.dtype, y.shape))
                elif not np.array_equal(x, y, equal_nan=True):
                    problems.append('%s: values differ (max |diff| %s)'
                                    % (k, np.nanmax(np.abs(x.astype(float) - y.astype(float))) if x.size else '-'))
            elif mapper(a[k]) != b[k]:
                problems.append('%s: text differs:\n    ref: %r\n    new: %r' % (k, mapper(a[k])[:300], b[k][:300]))
    errs = sorted(k for k in set(rt) | set(nt) if k.startswith('error:'))
    print('[compare] reference %s (NEW=%s), new %s (NEW=%s): %d arrays, %d texts compared'
          % (ref_dir, r_new, new_dir, n_new, len(ra), len(rt)))
    if errs:
        print('[compare] note: %d error records (equal on both sides counts as equal, but fix them): %s'
              % (len(errs), ', '.join(errs)))
    for p in problems:
        print('[compare] ' + p)
    print('[compare] %s' % ('PASS (bitwise)' if not problems else 'FAIL: %d problems' % len(problems)))
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run'); p.add_argument('--out', required=True); p.add_argument('--b9', default=None)
    p = sub.add_parser('b9-save'); p.add_argument('--out', required=True)
    p = sub.add_parser('compare'); p.add_argument('ref'); p.add_argument('new')
    a = ap.parse_args()
    if a.cmd == 'run':
        return run(a.out, a.b9)
    if a.cmd == 'b9-save':
        b9_save(a.out)
        return 0
    return compare(a.ref, a.new)


if __name__ == '__main__':
    sys.exit(main())
