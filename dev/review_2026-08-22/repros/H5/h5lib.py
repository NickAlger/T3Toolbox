"""Shared helpers for the H5 (uniform masks: phantom rank and garbage) hunter lane."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m


def relerr(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    den = np.linalg.norm(b.reshape(-1))
    return float(np.linalg.norm((a - b).reshape(-1)) / (den if den > 0 else 1.0))


def prefix(ranks, size):
    return np.arange(size) < np.asarray(ranks)[..., None]


def is_prefix(mask):
    return bool(np.array_equal(mask, prefix(mask.sum(axis=-1), mask.shape[-1])))


def corrupt_ut3(ux, scale=1e3, seed=1):
    """Garbage (scale * randn) in the padding of a UniformTuckerTensorTrain, real region untouched.
    Uses the object's own (correct-by-construction) mask: for INPUT robustness this is the right notion."""
    rng = np.random.RandomState(seed)
    ind = ut3.UniformTuckerTensorTrain(*[np.ones_like(s) for s in ux.supercores], ux.shape, ux.masks).apply_masks().supercores
    new = [sc + scale * rng.randn(*sc.shape) * (1.0 - i) for sc, i in zip(ux.supercores, ind)]
    return ut3.UniformTuckerTensorTrain(new[0], new[1], ux.shape, ux.masks)


def corrupt_frame(fr, scale=1e3, seed=2):
    rng = np.random.RandomState(seed)
    ind = ubv.UT3Frame(*[np.ones_like(s) for s in fr.supercores], fr.shape, fr.masks).apply_masks().supercores
    new = [sc + scale * rng.randn(*sc.shape) * (1.0 - i) for sc, i in zip(fr.supercores, ind)]
    return ubv.UT3Frame(new[0], new[1], new[2], new[3], fr.shape, fr.masks)


def corrupt_variations(v, scale=1e3, seed=3):
    rng = np.random.RandomState(seed)
    ind = ubv.UT3Variations(*[np.ones_like(s) for s in v.supercores], v.shape, v.masks).apply_masks().supercores
    new = [sc + scale * rng.randn(*sc.shape) * (1.0 - i) for sc, i in zip(v.supercores, ind)]
    return ubv.UT3Variations(new[0], new[1], v.shape, v.masks)


# asymmetric structures: distinct mode sizes, distinct Tucker ranks, distinct TT ranks, non-square cores
CASES = [
    # (shape, tucker_ranks, tt_ranks, stack_shape)
    ((5, 7, 6), (2, 3, 2), (1, 2, 2, 1), ()),
    ((5, 7, 6, 4), (2, 3, 4, 2), (1, 2, 4, 2, 1), ()),
    ((5, 7, 6), (2, 3, 2), (1, 2, 2, 1), (2,)),
    ((6, 4), (2, 2), (1, 2, 1), (2, 3)),
    ((5,), (3,), (1, 1), ()),                      # d = 1
]
PAD = dict(N=8, n=5, r=5)   # forced-larger padding so EVERY core has a padded region


def varying_stack(seed=0):
    """A (2,)-stack whose two elements have DIFFERENT ranks (the variety), with clean padding; returns
    (ustacked, [xa, xb]) where xa/xb are the ragged originals."""
    np.random.seed(seed)
    xa = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 3, 2), (1, 2, 2, 1))
    xb = t3.TuckerTensorTrain.randn((6, 7, 5), (4, 2, 3), (1, 3, 2, 1))
    ua = ut3.UniformTuckerTensorTrain.from_t3(xa, N=7, n=4, r=3)
    ub = ut3.UniformTuckerTensorTrain.from_t3(xb, N=7, n=4, r=3)
    return ut3.UniformTuckerTensorTrain.stack([ua, ub]), [xa, xb]


def dense_list(ux):
    """Dense per stack element as a flat list (stack flattened), for comparing stacks element-wise."""
    D = ux.to_dense()
    S = int(np.prod(ux.stack_shape)) if ux.stack_shape else 1
    return [d for d in D.reshape((S,) + ux.shape)]
