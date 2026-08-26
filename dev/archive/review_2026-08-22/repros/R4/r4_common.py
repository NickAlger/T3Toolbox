"""Shared helpers for the R4 repros (asymmetric structures, per-element references)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw

# asymmetric structures: distinct mode sizes / Tucker ranks / TT ranks, d in {1,2,3,4}
STRUCTS = [
    ((5,), (1,), (1, 1)),                               # d=1 minimal
    ((6, 7), (3, 2), (1, 3, 1)),                        # d=2
    ((5, 6, 7), (3, 2, 4), (1, 2, 3, 1)),               # d=3 (non-square everywhere)
    ((4, 5, 6, 7), (2, 3, 2, 3), (1, 2, 3, 2, 1)),      # d=4
]
NONMIN = ((6, 6, 6), (4, 4, 4), (1, 2, 2, 1))           # over-ranked Tucker -> up != down slack


def dense_mode_axes(Z, d):
    return tuple(range(Z.ndim - d, Z.ndim))


def hs_inner(A, B, d):
    return np.sum(A * B, axis=dense_mode_axes(A, d))


def tangent_basis_dense(frame):
    """Dense images of all unit variations at an (unstacked) frame: columns span T_xM (minimal frame)."""
    tucker_shapes, tt_shapes = frame.variation_shapes
    cols = []
    for fam, shapes in ((False, tucker_shapes), (True, tt_shapes)):
        for i, sh in enumerate(shapes):
            for idx in np.ndindex(*sh):
                u = t3m.T3Tangent.unit(frame, (fam, i, idx))
                cols.append(u.to_dense().reshape(-1))
    return np.stack(cols, axis=1)   # (prod N, ncoords)


def dense_project_onto_tangent(B, z):
    """Orthogonal projection of dense vector z onto span(B) via lstsq."""
    c, *_ = np.linalg.lstsq(B, z, rcond=1e-10)
    return B @ c


def unstack_frame_c(frame):
    """tree (over C) of unstacked T3Frames, as nested tuples."""
    return frame.unstack() if frame.stack_shape else frame


def leaf(tree, idx):
    for i in idx:
        tree = tree[i]
    return tree


def relerr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.linalg.norm((a - b).reshape(-1)) / max(1e-300, np.linalg.norm(a.reshape(-1))))
