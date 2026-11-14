# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The frontend frame/variations classes: ``T3Frame`` (orthogonal frame) + ``T3Variations``.

A (frame, variations) pair is the orthogonal representation of a tangent direction to the
fixed-rank T3 manifold (T4S Appendix A). ``T3Frame.data = (down, up, left, right) = (D, U, L, R)``, each
store named for the side of its edges that its cores occupy (:doc:`/frame_variations`, "Four rank stores");
structural validation runs in ``__post_init__``. ``t3_orthogonal_representations`` here returns
the frontend objects; the same-named backend function (``backend.fv_conversions``) returns raw data.
"""
import math
import numpy as np
import typing as typ
import t3toolbox.backend.common as common
import t3toolbox.safety as safety
import functools as ft
from dataclasses import dataclass

import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.t3_operations as t3_operations
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.fv_operations as fv_operations
import t3toolbox.backend.padded_frames as padded_frames
import t3toolbox.backend.tv_operations as tv_operations
from t3toolbox.backend.fv_operations import ZERO_FRAME_SEED
from t3toolbox.backend.padded_frames import (ALIGN_OVERSAMPLING, ALIGN_OVERSAMPLED_PASSES, ALIGN_MAX_PASSES,
                                             ALIGN_TOL, ALIGN_ALTERNATE)
import t3toolbox.corewise as cw
from t3toolbox.backend.common import *


__all__ = [
    'T3Frame',
    'T3Variations',
    'T3FrameWeights',
    'fv_absorb_weights',
    'fv_to_t3',
    'check_fw_pair',
    't3_orthogonal_representations',
    't3svd_orthogonal_representations',
    'fv_aligned_padded_frame',
]


@dataclass(frozen=True, eq=False)   # eq=False -> the ExplicitEquality mixin stands. (Until 2026-08-24
class T3Frame(common.ExplicitEquality):   # this was IDENTITY eq/hash, needed while the frame rode as jax
                                          # aux_data; the frame is a pytree LEAF since the safety redesign,
                                          # so == now raises like every runtime class -- see the mixin.)
    """Frame for frame-variations representation of TuckerTensorTrains

    Often, one works with TuckerTensorTrains of the following forms::

        1--(H0)--R1---R2---1    1---L0--(H1)--R2---1    1---L0---L1--(H2)--1
            |    |    |             |    |    |             |    |    |
            D0   D1   D2            D0   D1   D2            D0   D1   D2
            |    |    |             |    |    |             |    |    |

        1---U0---R1---R2---1    1---L0---U1---R2---1    1---L0---L1---U2---1
            |    |    |             |    |    |             |    |    |
           (V0)  D1   D2            D0  (V1)  D2            D0   D1  (V2)
            |    |    |             |    |    |             |    |    |

    In each of these, there is a special "variation" core, indicated by parentheses (X), surrounded by "frame" cores.

    The components of T3Frame are the "frame cores":
        - down_tucker_cores = (D0, ..., D(d-1)), elm_shape=(nDi, Ni)
        - up_tt_cores       = (U0, ..., U(d-1)), elm_shape=(rLi, nUi, rR(i+1))
        - left_tt_cores     = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))
        - right_tt_cores    = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))

    The components of T3Variations are the "variation cores":
        - tucker_variations = (V0, ..., V(d-1)), elm_shape=(nUi, Ni)
        - tt_variations     = (H0, ..., H(d-1)), elm_shape=(rLi, nDi, rRi)

    Note that Ld and R0 are not used in these diagrams. (Why keep them, then? They hold the base
    point as one extra variation term so the frame remembers where it is attached, and they give every
    family a uniform length d for code reuse / off-by-one safety. See ``docs/frame_variations.md`` for
    the full rationale, and for how the gauged variations act as coordinates -- which is what the
    weighted-layer metric ``T3FrameWeights`` reweights.)

    The edge ranks are shown in the following diagrams::

           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ----- (H1) ----- ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nD0     | nD1                | nD(d-1)
                 |         |                    |
                 D0        D1                   Dd
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |

    and::

           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ------ U1 ------ ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nD0     | nU1                | nD(d-1)
                 |         |                    |
                 D0       (V1)                   Dd
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |


    A tangent vector can be written as the sum of all the tensor diagrams above.
    In this case, the frame cores are representations of the point where the
    tangent space attaches to the manifold, and the variation cores define the
    tangent vector with respect to the frame.

    Often, it is desirable for the frame cores to be **orthogonal** as follows:
        - down_tucker_cores = (D0,...,D(d-1)), down-orthogonal:  D_ia D_ja = delta_ij
        - up_tt_cores       = (U0,...,U(d-1)), up-orthogonal:    U_aib U_ajb = delta_ij
        - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij
        - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal: R_ibc R_jbc = delta_ij

    Often, it is desirable for the variations to satisfy the following **Gauge conditions**:
        - D_ia V_ja = 0    (all V)
        - L_abi H_abj = 0  (all but the last H)

    If these conditions are satisfied, then one can do "dumb" corewise linear algebra
    (add, scale, dot product, etc) with the variations, and those core faithfully correspond
    to linear algebra with the N1 x ... x Nd tangent vectors represented by the variations.

    The orthogonal representations (44)-(45) and gauge conditions (47)-(48) are defined in Appendix
    A.3 of Alger, Christierson, Chen & Ghattas (2026), "Tucker Tensor Train Taylor Series"
    (arXiv:2603.21141).

    See Also
    --------
    T3Variations
    check_fv_pair
    t3_orthogonal_representations
    t3svd_orthogonal_representations

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.frame_variations_format as bvf
    >>> ss = (2, 3)                                       # frame/core stack C, shared by every core
    >>> down_tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
    >>> up_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
    >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3, 12, 5)))
    >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
    >>> frame = bvf.T3Frame(down_tucker_cores, up_tt_cores, left_tt_cores, right_tt_cores)
    >>> print(frame.structure)   # (shape, down_ranks, up_ranks, left_ranks, right_ranks, stack_shape)
    ((14, 15, 16), (10, 11, 12), (9, 8, 7), (1, 2, 3, 5), (2, 4, 5, 1), (2, 3))
    >>> print(frame.variation_shapes)   # the (tucker, tt) holes a fitting T3Variations must fill
    (((9, 14), (8, 15), (7, 16)), ((1, 10, 4), (2, 11, 5), (3, 12, 1)))
    """
    down_tucker_cores:  typ.Tuple[NDArray,...]  # len=d. D_xo D_yo   = I_xy, Di.shape = stack_shape+(nDi, Ni)
    up_tt_cores:        typ.Tuple[NDArray,...]  # len=d. U_ixj U_iyj = I_xy  Ui.shape = stack_shape+(rLi, nUi, rR(i+1))
    left_tt_cores:      typ.Tuple[NDArray,...]  # len=d. L_iax L_iay = I_xy, Li.shape = stack_shape+(rLi, nDi, rL(i+1))
    right_tt_cores:     typ.Tuple[NDArray,...]  # len=d. R_xaj R_yaj = I_xy  Ri.shape = stack_shape+(rRi, nDi, rR(i+1))

    @ft.cached_property
    def d(self) -> int:
        return len(self.down_tucker_cores)

    @ft.cached_property
    def shape(self) -> typ.Tuple[int,...]:
        return tuple([D.shape[-1] for D in self.down_tucker_cores])

    @ft.cached_property
    def down_ranks(self) -> typ.Tuple[int,...]:
        return tuple([D.shape[-2] for D in self.down_tucker_cores])

    @ft.cached_property
    def up_ranks(self) -> typ.Tuple[int,...]:
        return tuple([G.shape[-2] for G in self.up_tt_cores])

    @ft.cached_property
    def left_ranks(self) -> typ.Tuple[int,...]:
        return tuple([G.shape[-3] for G in self.left_tt_cores]) + (self.left_tt_cores[-1].shape[-1],)

    @ft.cached_property
    def right_ranks(self) -> typ.Tuple[int, ...]:
        return tuple([G.shape[-3] for G in self.right_tt_cores]) + (self.right_tt_cores[-1].shape[-1],)

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:   # C (frame/core stack): the batch of base points, on every core
        """The frame/core stack ``C`` -- a batch of base points, shared by every core."""
        return self.down_tucker_cores[0].shape[:-2]

    @ft.cached_property
    def structure(self) -> typ.Tuple[
        typ.Tuple[int, ...], # shape
        typ.Tuple[int, ...], # down_ranks
        typ.Tuple[int, ...], # up_ranks
        typ.Tuple[int, ...], # left_ranks
        typ.Tuple[int, ...], # right_ranks
        typ.Tuple[int, ...], # stack_shape
    ]:
        return (
            self.shape,
            self.down_ranks, self.up_ranks,
            self.left_ranks, self.right_ranks,
            self.stack_shape,
        )

    @ft.cached_property
    def variation_shapes(
            self,
    ) -> typ.Tuple[
        typ.Tuple[typ.Tuple[int, ...], ...],  # tucker_variation_shapes. len=d. elm_len=2
        typ.Tuple[typ.Tuple[int, ...], ...],  # tt_variation_shapes. len=d. elm_len=3
    ]:
        '''T3Variations shapes that fit with this T3Frame.

        Shapes of the "holes" in the following tensor diagrams::

            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 D0     D1     D2     D3
                 |      |      |      |

            1 -- L0 -- L1 -- U2 -- R3 -- 1
                 |     |     |     |
                 D0    D1    ( )   D3
                 |     |     |     |

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.frame_variations_format as bvf
        >>> ss = (2, 3)                                   # stack C -- NOT part of variation_shapes
        >>> down_tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
        >>> up_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
        >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3, 12, 5)))
        >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
        >>> frame = bvf.T3Frame(down_tucker_cores, up_tt_cores, left_tt_cores, right_tt_cores)
        >>> tucker_holes, tt_holes = frame.variation_shapes   # (tucker hole shapes, tt hole shapes)
        >>> print(tucker_holes)   # one (nUi, Ni) per mode
        ((9, 14), (8, 15), (7, 16))
        >>> print(tt_holes)       # one (rLi, nDi, rRi) per mode
        ((1, 10, 4), (2, 11, 5), (3, 12, 1))
        '''
        tucker_variation_shapes = tuple([(nU, N) for nU, N in zip(self.up_ranks, self.shape)])
        tt_variation_shapes = tuple([
            (rL, nD, rR) for rL, nD, rR
            in zip(self.left_ranks[:-1], self.down_ranks, self.right_ranks[1:])])

        return tucker_variation_shapes, tt_variation_shapes

    @ft.cached_property
    def data(self) -> typ.Tuple[
        typ.Tuple[NDArray,...],  # down_tucker_cores
        typ.Tuple[NDArray, ...], # up_tt_cores
        typ.Tuple[NDArray,...],  # left_tt_cores
        typ.Tuple[NDArray,...],  # right_tt_cores
    ]:
        return self.down_tucker_cores, self.up_tt_cores, self.left_tt_cores, self.right_tt_cores

    def to_jax(self) -> 'T3Frame':
        """Copy with all frame cores converted to jax arrays."""
        return T3Frame(*[tuple(to_jax(c) for c in fam) for fam in self.data])

    def to_numpy(self) -> 'T3Frame':
        """Copy with all frame cores converted to numpy arrays."""
        return T3Frame(*[tuple(to_numpy(c) for c in fam) for fam in self.data])

    def copy(self) -> 'T3Frame':
        """Deep copy (copies every frame core)."""
        return T3Frame(*[tuple(c.copy() for c in fam) for fam in self.data])

    @ft.cached_property
    def contains_jax(self) -> bool:
        """True if any frame core is a jax array."""
        return tree_contains_jax(self.data)

    @ft.cached_property
    def size(self) -> int:
        """Number of elements of the represented (base-point) dense tensor (``prod(shape)``)."""
        return math.prod(self.shape)

    @ft.cached_property
    def data_size(self) -> int:
        """Number of stored core entries (size on disk)."""
        return sum(int(c.size) for fam in self.data for c in fam)

    def __repr__(self) -> str:
        ss = f", stack_shape={self.stack_shape}" if self.stack_shape else ""
        return (f"T3Frame(shape={self.shape}, down_ranks={self.down_ranks}, "
                f"left_ranks={self.left_ranks}{ss})")

    @ft.cached_property
    def orthogonality_residual(self) -> NDArray:  # shape = stack_shape (scalar/0-d when unstacked)
        '''Max absolute deviation of the orthogonal cores from identity, **per stack element** (shape
        ``stack_shape``; atol-independent; **cached**).

        The expensive part of :py:meth:`is_orthogonal` -- a fixed frame reused across an inner loop (e.g.
        the safe-mode ORTH precondition of :py:meth:`~t3toolbox.manifold.ManifoldGeometry.project` on the
        same frame every matvec) is contracted **once**, not per call.'''
        return fv_operations.fv_frame_orthogonality_residual(self.data)

    @staticmethod
    def padded(
            x_old:         't3.TuckerTensorTrain',             # the old point (reduced to minimal ranks here); rank 0 = the zero tensor
            tucker_ranks:  typ.Sequence[int],                  # len=d, target
            tt_ranks:      typ.Sequence[int],                  # len=d+1, target (boundary 1)

            sharing:       typ.Optional[typ.Sequence] = None,  # len=d group labels; None = unshared
            seed:          int = ZERO_FRAME_SEED,              # the keyed seed streams of the completions
    ) -> 'T3Frame':                                            # the seeded padded frame (old frame in the leading slots)
        """The seeded padded frame of ``x_old`` at larger ranks: the start of an alignment, hint-free.

        At a zero-padded rank increase the point ``x_old`` gains zero slots, and its orthonormal frame needs new
        directions there. This frame keeps ``x_old``'s own frame in the leading slots of every store and fills the new
        slots by a data-independent convention: Gaussian seeds from keyed streams, one stream row per slot, completed
        against the old block (so successive starts never reuse a seed). :py:func:`fv_aligned_padded_frame` then aligns
        the new slots with a tangent hint; this is its start, and its ``max_passes=0`` result.

        ``x_old`` is first reduced to minimal ranks (the same tensor). A STRUCTURALLY ZERO ``x_old`` -- any Tucker rank
        or interior TT rank 0, e.g. ``TuckerTensorTrain.zeros(shape, (0,)*d, (1,)+(0,)*(d-1)+(1,))`` -- is the zero
        tensor, whose frame is empty. The target ranks must contain ``x_old``'s structure. In safe mode, a shared
        ``x_old`` must have tied factors.

        Each store's new slot ``s`` is seeded by row ``s`` of a keyed Gaussian stream. So the start is isotropic (a
        random subspace of the old block's complement), well conditioned, and successive starts never reuse a seed.
        It is a function of ``x_old``'s frame AS REPRESENTED, slot signs included: another basis of the same frame (at a
        singular-value tie, or a flipped slot sign) gives another, equally valid start. Do not compare hint-free
        starts across representations or BLAS builds; the aligned frames are comparable. See
        :doc:`/aligned_padded_frame`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.frame_variations_format as bvf
        >>> np.random.seed(0)
        >>> x_old = t3.TuckerTensorTrain.randn((5, 6, 5), (2, 2, 2), (1, 2, 2, 1))
        >>> F = bvf.T3Frame.padded(x_old, (3, 4, 3), (1, 3, 3, 1))
        >>> F.down_ranks, bool(F.is_orthogonal())
        ((3, 4, 3), True)

        The new frame represents the zero-padded point: its base point is still ``x_old``.

        >>> x0 = x_old.resize((5, 6, 5), (3, 4, 3), (1, 3, 3, 1))
        >>> bool(t3.TuckerTensorTrain(F.down_tucker_cores, F.left_tt_cores).allclose(x0))
        True
        """
        return T3Frame(*padded_frames.fv_padded_frame(_old_frame_data(x_old, sharing, 'T3Frame.padded'),
                                                      tucker_ranks, tt_ranks, sharing=sharing, seed=seed))

    def is_orthogonal(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = stack_shape (scalar unstacked)
        '''True (per stack element) if the frame cores are orthogonal in their respective senses.

        Checks (each stacked block; max absolute deviation from identity <= atol):
            - down_tucker D_i (all i):  ``einsum('...io,...jo->...ij', D, D) = I``
            - up/outer U_i (all i):     ``einsum('...iaj,...ibj->...ab', U, U) = I``
            - left L_i (i = 0..d-2):    ``einsum('...iaj,...iak->...jk', L, L) = I``
            - right R_i (i = 1..d-1):   ``einsum('...iaj,...kaj->...ik', R, R) = I``

        The last left core and the first right core are the (non-orthogonal) boundary remainders
        and are not checked. This is a non-enforcing convenience checker; ``T3Frame`` does not
        require orthogonality at construction. **Returns a per-stack-element bool array** (shape
        ``stack_shape``; a scalar when unstacked) -- different base points in a stack can differ; reduce
        with ``.all()`` for a single verdict.

        Orthogonal cores (left/right/outer/Tucker) are defined in Appendix A.1 of Alger et al.
        (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.frame_variations_format as bvf
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        >>> frame, _ = bvf.t3_orthogonal_representations(x)   # this frame IS orthogonal by construction
        >>> print(frame.is_orthogonal())          # unstacked -> a scalar bool
        True

        Stacked: a per-element bool array. Stack a good frame with a deliberately non-orthogonal one to
        show the elements differ:

        >>> good, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn((5, 6), (2, 2), (1, 2, 1)))
        >>> bad = bvf.T3Frame(tuple(np.ones_like(D) for D in good.down_tucker_cores),  # ones cores: not orthogonal
        ...                   good.up_tt_cores, good.left_tt_cores, good.right_tt_cores)
        >>> stacked = bvf.T3Frame.stack([good, bad])
        >>> print(stacked.is_orthogonal().shape, stacked.is_orthogonal())   # one bool per stack element
        (2,) [ True False]
        '''
        return self.orthogonality_residual <= atol

    @ft.cached_property
    def minimal_ranks(self) -> typ.Tuple[typ.Tuple[int, ...], typ.Tuple[int, ...]]:
        """Structural minimal ranks ``(min_tucker_ranks, min_tt_ranks)`` for this frame's shape/ranks.

        **Structural**: computed from the rank tuples + shape, not the numerical core values (see
        :py:meth:`TuckerTensorTrain.get_minimal_ranks`).
        """
        return t3.TuckerTensorTrain.get_minimal_ranks(self.shape, self.down_ranks, self.left_ranks)

    @ft.cached_property
    def has_minimal_ranks(self) -> bool:
        '''True if the frame has minimal ranks.

        Minimal-rank frame means:
            - ``left_ranks == right_ranks``,
            - ``down_ranks == up_ranks``, and
            - those ranks (tucker_ranks=down_ranks, tt_ranks=left_ranks) are minimal for a regular
              Tucker tensor train of this shape (see :py:meth:`TuckerTensorTrain.get_minimal_ranks`).

        This is the **structural** minimal-rank check (cheap integer arithmetic on the ranks); for the
        **numerical** one (no stored rank numerically redundant) see :py:meth:`has_numerically_minimal_ranks`.
        Empirically (``docs/numerical_contracts.md``) minimal rank is **not** a correctness
        precondition for any verified operation -- ``inner``/``norm``-as-HS and ``manifold_dim`` are exact
        on a non-minimal orthonormal frame, and ``retract`` only loses *strict* rank preservation (it
        drops the redundant rank, staying a valid retraction). So this is a non-enforcing checker;
        ``T3Frame`` does not require minimal ranks at construction.

        Minimal (non-degenerate) ranks and their connection to matricizations and matrix unfoldings
        are discussed in Appendix A.2 of Alger et al. (2026), "Tucker Tensor Train Taylor Series"
        (arXiv:2603.21141).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.frame_variations_format as bvf
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))  # minimal ranks
        >>> frame, _ = bvf.t3_orthogonal_representations(x)
        >>> print(frame.has_minimal_ranks)
        True
        >>> x2 = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))  # Tucker rank 4 > 1*3
        >>> frame2, _ = bvf.t3_orthogonal_representations(x2)
        >>> print(frame2.has_minimal_ranks)
        False
        '''
        return ranks.frame_has_minimal_ranks(
            self.shape, self.down_ranks, self.up_ranks, self.left_ranks, self.right_ranks)

    def has_numerically_minimal_ranks(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = stack_shape
        '''True (per stack element) if the frame is **numerically** minimal -- certified *without* an SVD.

        An orthonormal frame's cores are full-rank, so an **orthogonal + structurally-minimal** frame is
        automatically numerically minimal (no ``t3svd`` -- and a frame is not a tensor to SVD anyway). So
        this returns ``is_orthogonal(atol) and has_minimal_ranks``. A **non-orthogonal** frame returns
        ``False``: the SVD certification path for non-orthogonal frames is intentionally not implemented
        (frames that need numerical minimality are expected to be orthonormal). Distinct from the
        structural :py:attr:`has_minimal_ranks`; for tensors see
        :py:meth:`TuckerTensorTrain.has_numerically_minimal_ranks` (the SVD version).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.frame_variations_format as bvf
        >>> np.random.seed(0)
        >>> frame, _ = bvf.t3_orthogonal_representations(
        ...     t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1)))     # orthogonal + minimal
        >>> print(frame.has_numerically_minimal_ranks())
        True
        >>> nb, _ = bvf.t3_orthogonal_representations(
        ...     t3.TuckerTensorTrain.randn((10, 11, 12), (4, 5, 4), (1, 2, 3, 1)))  # orthogonal, NON-minimal
        >>> print(nb.is_orthogonal(), nb.has_minimal_ranks, nb.has_numerically_minimal_ranks())
        True False False
        '''
        # is_orthogonal is per-stack-element (bool array); has_minimal_ranks is a structural scalar -- use
        # `&` (element-wise, broadcasting the scalar) so the result is per-element, NOT Python `and`.
        return self.is_orthogonal(atol=atol) & self.has_minimal_ranks

    def validate(self) -> None:
        '''Check rank and shape consistency of Tucker tensor train frame (`T3Frame`).

        Raises
        ------
        ValueError
            Error raised if the cores of the T3Frame have inconsistent shapes.

        See Also
        --------
        T3Frame
        T3Variations
        '''
        DD, UU, LL, RR = self.data

        d = len(DD)
        if not (len(LL) == d and len(RR) == d and len(UU) == d):
            raise ValueError(
                'Inconsistent T3Frame.\n'
                + 'All core sequences must have length d=' + str(d) + '.\n'
                + 'len(DD)=' + str(len(DD))
                + ', len(UU)=' + str(len(UU))
                + ', len(LL)=' + str(len(LL))
                + ', len(RR)=' + str(len(RR))

            )

        for ii, D in enumerate(DD):
            if len(D.shape) < 2:
                raise ValueError(
                    'Inconsistent T3Frame.\n'
                    + 'tucker_cores[' + str(ii) + '] is not a (stacked) matrix. shape=' + str(D.shape)
                )

        for name, CC in zip(["left_tt", "right_tt", "up_tt"], [LL, RR, UU]):
            for ii, C in enumerate(CC):
                if len(C.shape) < 3:
                    raise ValueError(
                        'Inconsistent T3Frame.\n'
                        + name + '_cores[' + str(ii) + '] is not a (stacked) 3-tensor. '
                        + 'shape=' + str(C.shape)
                    )

        down_stack_shapes   = tuple([B.shape[:-2] for B in self.down_tucker_cores])
        left_stack_shapes   = tuple([G.shape[:-3] for G in self.left_tt_cores])
        right_stack_shapes  = tuple([G.shape[:-3] for G in self.right_tt_cores])
        up_stack_shapes     = tuple([G.shape[:-3] for G in self.up_tt_cores])

        if not (
                down_stack_shapes
                == up_stack_shapes
                == left_stack_shapes
                == right_stack_shapes
                == (self.stack_shape,)*self.d
        ):
            raise ValueError(
                'Inconsistent T3Frame.\n'
                + str(down_stack_shapes) + ' = down_stack_shapes.\n'
                + str(up_stack_shapes) + ' = up_stack_shapes.\n'
                + str(left_stack_shapes) + ' = left_stack_shapes.\n'
                + str(right_stack_shapes) + ' = right_stack_shapes.'
            )

        rLl = tuple([int(LL[0].shape[-3])] + [int(L.shape[-1]) for L in LL])
        rLr = tuple([int(L.shape[-3]) for L in LL] + [int(LL[-1].shape[-1])])
        if rLl != rLr:
            raise ValueError(
                'Inconsistent T3Frame.\n'
                + str(rLl) + ' = rL_left != rL_right = ' + str(rLr)
            )

        rRl = tuple([int(RR[0].shape[-3])] + [int(R.shape[-1]) for R in RR])
        rRr = tuple([int(R.shape[-3]) for R in RR] + [int(RR[-1].shape[-1])])
        if rRl != rRr:
            raise ValueError(
                'Inconsistent T3Frame.\n'
                + str(rRl) + ' = rR_left != rR_right = ' + str(rRr)
            )

        for ii in range(d):
            D, L, R, U = DD[ii], LL[ii], RR[ii], UU[ii]

            if not (D.shape[-2] == L.shape[-2] == R.shape[-2]):
                raise ValueError(
                    'Inconsistent T3Frame.\n'
                    + 'Tucker rank mismatch at index ' + str(ii)
                    + ': D.shape[-2]=' + str(D.shape[-2])
                    + ', L.shape[-2]=' + str(L.shape[-2])
                    + ', R.shape[-2]=' + str(R.shape[-2])
                )

            if U.shape[-3] != L.shape[-3]:
                raise ValueError(
                    'Inconsistent T3Frame.\n'
                    + 'Up TT core left rank mismatch at index ' + str(ii)
                    + ': U.shape[-3]=' + str(U.shape[-3])
                    + ' != L.shape[-3]=' + str(L.shape[-3])
                )

            if U.shape[-1] != R.shape[-1]:
                raise ValueError(
                    'Inconsistent T3Frame.\n'
                    + 'Up TT core right rank mismatch at index ' + str(ii)
                    + ': U.shape[-1]=' + str(U.shape[-1])
                    + ' != R.shape[-1]=' + str(R.shape[-1])
                )

    def __post_init__(self):
        self.validate()

    def unstack(self):
        """Unstack into an array-like tree.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.frame_variations_format as bvf
        >>> import t3toolbox.corewise as cw
        >>> np.random.seed(0)
        >>> rnd = lambda x: np.random.randn(*x)
        >>> ss = (2, 3)                                   # stack C: unstack splits these leading axes
        >>> down_tucker_cores = (rnd(ss+(10, 14)), rnd(ss+(11, 15)), rnd(ss+(12, 16)))
        >>> up_tt_cores = (rnd(ss+(1, 9, 4)), rnd(ss+(2, 8, 5)), rnd(ss+(3, 7, 1)))
        >>> left_tt_cores = (rnd(ss+(1, 10, 2)), rnd(ss+(2, 11, 3)), rnd(ss+(3, 12, 5)))
        >>> right_tt_cores = (rnd(ss+(2, 10, 4)), rnd(ss+(4, 11, 5)), rnd(ss+(5, 12, 1)))
        >>> frame = bvf.T3Frame(down_tucker_cores, up_tt_cores, left_tt_cores, right_tt_cores)
        >>> S = frame.unstack()
        >>> print(len(S), len(S[0]))                      # nested tree shaped like the stack (2, 3)
        2 3
        >>> ii, jj = 1, 2                                 # the [ii][jj] leaf is just the cores sliced at [ii, jj]
        >>> Sij = S[ii][jj]
        >>> sliced = bvf.T3Frame(
        ...     tuple(c[ii, jj] for c in down_tucker_cores), tuple(c[ii, jj] for c in up_tt_cores),
        ...     tuple(c[ii, jj] for c in left_tt_cores), tuple(c[ii, jj] for c in right_tt_cores))
        >>> print(np.allclose(cw.corewise_norm(cw.corewise_sub(Sij.data, sliced.data)), 0.0))
        True
        """
        return stacking.apply_func_to_leaf_subtrees(
            stacking.basic_ragged_unstack(self.data, 2),
            lambda x: T3Frame(*x),
            self.data, # leaf_structure
        )

    @staticmethod
    def stack(
            xx, # Array-like tree of T3Frame
    ):
        """Stack array-like tree of T3Frame into a single T3Frame.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.frame_variations_format as bvf
        >>> import t3toolbox.corewise as cw
        >>> np.random.seed(0)
        >>> rnd = lambda x: np.random.randn(*x)
        >>> ss = (2, 3)                                   # stack C
        >>> down_tucker_cores = (rnd(ss+(10, 14)), rnd(ss+(11, 15)), rnd(ss+(12, 16)))
        >>> up_tt_cores = (rnd(ss+(1, 9, 4)), rnd(ss+(2, 8, 5)), rnd(ss+(3, 7, 1)))
        >>> left_tt_cores = (rnd(ss+(1, 10, 2)), rnd(ss+(2, 11, 3)), rnd(ss+(3, 12, 5)))
        >>> right_tt_cores = (rnd(ss+(2, 10, 4)), rnd(ss+(4, 11, 5)), rnd(ss+(5, 12, 1)))
        >>> x = bvf.T3Frame(down_tucker_cores, up_tt_cores, left_tt_cores, right_tt_cores)
        >>> x2 = bvf.T3Frame.stack(x.unstack())           # stack is the inverse of unstack
        >>> print(np.allclose(cw.corewise_norm(cw.corewise_sub(x.data, x2.data)), 0.0))
        True
        """
        xx_tuples = stacking.apply_func_to_leaf_subtrees(
            xx,
            lambda x: x.data,
            None,  # leaf_structure
        )
        result = stacking.basic_ragged_stack(xx_tuples)
        return T3Frame(*result)

    @staticmethod
    def from_t3(x: 't3.TuckerTensorTrain') -> 'T3Frame':
        """Orthogonal representation (frame) of a TuckerTensorTrain ``x`` -- the orthogonal frame at the
        point ``x`` (the frame part of :py:func:`t3_orthogonal_representations`)."""
        return t3_orthogonal_representations(x)[0]

    @staticmethod
    def random_orthogonal(
            shape:          typ.Sequence[int],              # (N0,...,N(d-1))
            tucker_ranks:   typ.Sequence[int],              # (n0,...,n(d-1))
            tt_ranks:       typ.Sequence[int],              # (1,r1,...,r(d-1),1)
            stack_shape:    typ.Tuple[int, ...] = (),       # C (frame/core stack)
            use_jax:        bool = False,
    ) -> 'T3Frame':
        """Orthogonal representation of a *random* T3 -- a genuine random base point (orthogonal,
        consistent), **not** iid-random cores. Equals ``from_t3(TuckerTensorTrain.randn(...))``.

        Requesting a rank above what the rest of the network can support is allowed and does not raise:
        an orthonormal core cannot carry more rank than its own shape admits, so the excess is carried as
        **slack** between the frame's four rank stores rather than in any one core. Asking for
        ``tucker_ranks=(4,4,4)`` with ``tt_ranks=(1,2,2,1)`` returns ``down_ranks=(4,4,4)`` but
        ``up_ranks=(2,4,2)``; asking for ``tt_ranks=(1,9,9,1)`` returns different ``left_ranks`` and
        ``right_ranks``. The frame is exactly orthogonal either way and every tangent operation on it is
        exact -- only :py:attr:`has_minimal_ranks` reports ``False`` (a diagnostic, never a
        precondition; see ``docs/frame_variations.md``)."""
        x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks, stack_shape=stack_shape, use_jax=use_jax)
        return t3_orthogonal_representations(x)[0]

    @staticmethod
    def random_orthogonal_like(frame: 'T3Frame') -> 'T3Frame':
        """A random orthogonal frame with the same shape/ranks/stack as ``frame``."""
        return T3Frame.random_orthogonal(frame.shape, frame.down_ranks, frame.left_ranks,
                                         stack_shape=frame.stack_shape, use_jax=frame.contains_jax)

    def save(self, file) -> None:
        """Save the frame cores to a ``.npz`` file (load with :py:meth:`load`)."""
        save_core_families(file, self.data)

    @staticmethod
    def load(
            file,                       # path or file-like (.npz saved by save)
            use_jax:    bool = False,
    ) -> 'T3Frame':
        """Load a frame saved by :py:meth:`save`."""
        f = load_core_families(file)
        b = T3Frame(f[0], f[1], f[2], f[3])
        return b.to_jax() if use_jax else b

    def reverse(self) -> 'T3Frame':
        """Reverse the mode order. Left/right cores **swap roles** (reversing a left-orthogonal chain
        yields a right-orthogonal one), so ``new_left = reverse(old_right)`` and vice versa; the
        redundant L/R store makes this exact with no re-orthogonalization."""
        return T3Frame(*fv_operations.fv_frame_reverse(self.data))

    def allclose(
            self,
            other: 'T3Frame',

            rtol:  typ.Optional[float] = None,  # None: the ambient jax-aware default (safety.comparison_rtol)
            atol:  typ.Optional[float] = None,  # None: 0.0
    ) -> NDArray:  # bool, shape=stack_shape (0-d unstacked); reduce with .all()
        """True where ``other`` represents the same BASE POINT as ``self`` (gauge-invariant), per
        stack element -- the ragged twin of :py:meth:`UT3Frame.allclose`, via ``to_t3()`` difference
        norms. Three distinct equality questions exist for a frame; say which you mean: this method
        (same base point, any gauge); ``safety.frames_equal(a.data, b.data)`` (the same-FRAME /
        same-tangent-space question, gauge included); :py:meth:`corewise_equal` (bitwise
        representation). ``==`` is intentionally not defined."""
        return self.to_t3().allclose(other.to_t3(), rtol=rtol, atol=atol)

    def corewise_equal(
            self,
            other: 'T3Frame',
    ) -> bool:
        """Bitwise equality of the stored frame cores (``False`` on any mismatch, never raises)."""
        return type(other) is type(self) and cw.corewise_equal(self.data, other.data)

    def to_t3(self) -> 't3.TuckerTensorTrain':
        """The base point this frame represents, as a :py:class:`TuckerTensorTrain` (natural ranks).

        Reconstructed in right-canonical form (the Tucker factors over the right-orthogonal core-TT).
        For a **consistent** frame (e.g. from :py:func:`t3_orthogonal_representations`) this is the base
        point and equals the left-canonical reconstruction; for a hand-built inconsistent frame it is
        specifically this form. No consistency check is performed (verifying it would mean densifying
        multiple reconstructions -- the kind of expensive check the library avoids).
        """
        return t3.TuckerTensorTrain(self.down_tucker_cores, self.right_tt_cores)

    def to_dense(self) -> NDArray:
        """Dense tensor of the base point this frame represents (``= to_t3().to_dense()``)."""
        return self.to_t3().to_dense()

    def orthogonalize(self) -> 'T3Frame':
        """Orthogonal, minimal-rank representation of the base point this frame reconstructs to.

        Equivalent to ``T3Frame.from_t3(self.to_t3())``: reconstruct the base point (right-canonical,
        see :py:meth:`to_t3`) and recompute its orthogonal representation via
        :py:func:`t3_orthogonal_representations`. For a frame that is already orthogonal and consistent
        this returns an equivalent orthogonal frame; for a hand-built or drifted one it returns a
        genuinely orthogonal frame for the *right-canonical* base point (each direction's ranks capped
        at what the network supports, so an over-ranked input comes back carrying down/up slack --
        see :py:attr:`has_minimal_ranks`).
        """
        return T3Frame.from_t3(self.to_t3())

    def is_consistent(self, rtol: float = 1e-9) -> NDArray:  # bool array, shape = stack_shape (scalar unstacked)
        """``True`` (per stack element) if the left- and right-canonical reconstructions of the base point agree.

        A frame stores both the left- and right-orthogonal core-TTs (see :py:meth:`to_t3`); for a
        **consistent** frame they reconstruct the same base point. This checks
        ``||left - right|| <= rtol * ||right||`` in the dense Frobenius norm.

        EXPENSIVE -- densifies both reconstructions. Deliberately **not** part of :py:meth:`validate`
        (which is structural and cheap). For a frame from :py:func:`t3_orthogonal_representations` (or
        :py:meth:`from_t3`/:py:meth:`orthogonalize`) consistency holds by construction; this is for
        sanity-checking hand-built bases.
        """
        return fv_operations.fv_frame_consistency_residual(self.data) <= rtol

    def allclose(self, other: 'T3Frame', rtol: float = 1e-9, atol: float = 0.0) -> NDArray:  # bool array, stack_shape
        """``True`` (per stack element) if ``other`` represents the same base point as ``self`` (gauge-invariant).

        Compares the *represented* base points, not the cores: ``||self.to_t3() - other.to_t3()|| <=
        atol + rtol * ||other.to_t3()||`` in the dense Frobenius norm (via
        :py:meth:`TuckerTensorTrain.norm`, no densification), **per stack element** (reduce with ``.all()``
        for a single verdict). Invariant to the orthogonal/gauge
        representation -- two different orthogonal bases for the same point compare equal.
        """
        dn = (self.to_t3() - other.to_t3()).norm()
        rn = other.to_t3().norm()
        return dn <= atol + rtol * rn




@dataclass(frozen=True, eq=False)   # eq=False -> the ExplicitEquality mixin stands
class T3Variations(common.ExplicitEquality):
    """
    Tuple containing variation cores for frame-variation representations of TuckerTensorTrains.

    *Components*
        - tucker_variations    = (V0, ..., V(d-1)), elm_shape=stack_shape+(nUi, Ni)
        - tt_variations        = (H0, ..., H(d-1)), elm_shape=stack_shape+(rLi, nDi, rRi)

    The variations should fit in the "holes" of a T3Frame.

    See Also
    --------
    T3Frame

    Examples
    --------
    Build variations that fill the holes of a base point (so ``check_fv_pair`` accepts them):

    >>> import numpy as np
    >>> import t3toolbox.frame_variations_format as bvf
    >>> ss = (2, 3)                                       # stack shape, shared by frame and variations
    >>> down_tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
    >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3, 12, 5)))
    >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
    >>> up_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
    >>> frame = bvf.T3Frame(down_tucker_cores, up_tt_cores, left_tt_cores, right_tt_cores)
    >>> tucker_shapes, tt_shapes = frame.variation_shapes  # the holes to fill
    >>> tucker_variations = tuple(np.ones(ss + s) for s in tucker_shapes)
    >>> tt_variations = tuple(np.ones(ss + s) for s in tt_shapes)
    >>> variations = bvf.T3Variations(tucker_variations, tt_variations)
    >>> print(variations.variation_shapes == frame.variation_shapes)   # fits the frame's holes exactly
    True
    """
    tucker_variations: typ.Tuple[NDArray,...]  # len=d, elm_shape=stack_shape+(nUi, Ni)
    tt_variations:     typ.Tuple[NDArray,...]  # len=d, elm_shape=stack_shape+(rLi, nDi, rRi)

    @ft.cached_property
    def d(self) -> int:
        return len(self.tucker_variations)

    @ft.cached_property
    def shape(self) -> typ.Tuple[int,...]:
        return tuple([V.shape[-1] for V in self.tucker_variations])

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:   # full leading stack K + C (split-agnostic on its own)
        """The full leading stack ``K + C`` shared by every variation core.

        ``T3Variations`` is **split-agnostic**: the tangent-stack ``K`` vs frame-stack ``C`` split is
        fixed only when paired with a :py:class:`T3Frame` (``check_fv_pair`` requires ``C`` to be the
        trailing/inner part of this stack; :py:class:`~t3toolbox.manifold.T3Tangent` then exposes the
        two parts). See ``docs/batching_and_stacking.md``.
        """
        return self.tucker_variations[0].shape[:-2]

    @ft.cached_property
    def variation_shapes(
            self,
    ) -> typ.Tuple[
        typ.Tuple[typ.Tuple[int, ...], ...],  # tucker_variation_shapes. len=d. elm_len=2
        typ.Tuple[typ.Tuple[int, ...], ...],  # tt_variation_shapes. len=d. elm_len=3
    ]:
        '''T3Variations shapes that fit with this T3Frame.

        Shapes of the "holes" in the following tensor diagrams::

            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 D0     D1     D2     D3
                 |      |      |      |

            1 -- L0 -- L1 -- U2 -- R3 -- 1
                 |     |     |     |
                 D0    D1    ( )   D3
                 |     |     |     |
        '''
        tucker_variation_shapes = tuple([V.shape[-2:] for V in self.tucker_variations])
        tt_variation_shapes = tuple([H.shape[-3:] for H in self.tt_variations])
        return tucker_variation_shapes, tt_variation_shapes

    @ft.cached_property
    def data(self) -> typ.Tuple[
        typ.Tuple[NDArray,...], # tucker_variations
        typ.Tuple[NDArray,...], # tt_variations
    ]:
        return self.tucker_variations, self.tt_variations

    def to_jax(self) -> 'T3Variations':
        """Copy with all variation cores converted to jax arrays."""
        return T3Variations(*[tuple(to_jax(c) for c in fam) for fam in self.data])

    def to_numpy(self) -> 'T3Variations':
        """Copy with all variation cores converted to numpy arrays."""
        return T3Variations(*[tuple(to_numpy(c) for c in fam) for fam in self.data])

    def copy(self) -> 'T3Variations':
        """Deep copy (copies every variation core)."""
        return T3Variations(*[tuple(c.copy() for c in fam) for fam in self.data])

    @ft.cached_property
    def contains_jax(self) -> bool:
        """True if any variation core is a jax array."""
        return tree_contains_jax(self.data)

    @ft.cached_property
    def size(self) -> int:
        """Number of elements of the represented dense tensor (``prod(shape)``)."""
        return math.prod(self.shape)

    @ft.cached_property
    def data_size(self) -> int:
        """Number of stored core entries (size on disk)."""
        return sum(int(c.size) for fam in self.data for c in fam)

    def __repr__(self) -> str:
        ss = f", stack_shape={self.stack_shape}" if self.stack_shape else ""
        return f"T3Variations(shape={self.shape}{ss})"

    def validate(self) -> None:
        '''Check rank and shape consistency of Tucker tensor train variations (`T3Variations`).

        Parameters
        ----------
        self : T3Variations

        Raises
        ------
        ValueError
            Error raised if the cores of the T3Variations have inconsistent shapes.

        See Also
        --------
        T3Frame
        T3Variations
        '''
        VV, HH = self.data

        d = len(VV)
        if len(HH) != d:
            raise ValueError(
                'Inconsistent T3Variations.\n'
                + 'All core sequences must have length d=' + str(d) + '.\n'
                + 'len(VV)=' + str(len(VV))
                + ', len(HH)=' + str(len(HH))
            )

        for ii, V in enumerate(VV):
            if len(V.shape) < 2:
                raise ValueError(
                    'Inconsistent T3Variations.\n'
                    + 'tucker_cores[' + str(ii) + '] is not a (stacked) matrix. shape=' + str(V.shape)
                )

        for ii, H in enumerate(HH):
            if len(H.shape) < 3:
                raise ValueError(
                    'Inconsistent T3Variations.\n'
                    + 'tt_cores[' + str(ii) + '] is not a (stacked) 3-tensor. '
                    + 'shape=' + str(H.shape)
                )

        tucker_stack_shapes = tuple([V.shape[:-2] for V in self.tucker_variations])
        tt_stack_shapes = tuple([H.shape[:-3] for H in self.tt_variations])

        if not (tucker_stack_shapes == tt_stack_shapes == (self.stack_shape,)*self.d):
            raise ValueError(
                'Inconsistent T3Variations.\n'
                + str(tucker_stack_shapes) + ' = tucker_stack_shapes.\n'
                + str(tt_stack_shapes) + ' = tt_stack_shapes.\n'
            )

    def __post_init__(self):
        self.validate()

    def unstack(self):
        """Unstack into an array-like tree.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.frame_variations_format as bvf
        >>> import t3toolbox.corewise as cw
        >>> np.random.seed(0)
        >>> rnd = lambda x: np.random.randn(*x)
        >>> ss = (2, 3)                                   # stack: unstack splits these leading axes
        >>> NN, nnD, nnU, rrL, rrR = (10, 11, 12), (8, 7, 8), (6, 7, 8), (2, 3, 4, 3), (5, 4, 6, 1)
        >>> tucker_variations = tuple(rnd(ss+(n, N)) for n, N in zip(nnU, NN))
        >>> tt_variations = tuple(rnd(ss+(rL, n, rR)) for rL, n, rR in zip(rrL[:-1], nnD, rrR[1:]))
        >>> V = bvf.T3Variations(tucker_variations, tt_variations)
        >>> VV = V.unstack()
        >>> print(len(VV), len(VV[0]))                    # nested tree shaped like the stack (2, 3)
        2 3
        >>> ii, jj = 1, 2                                 # the [ii][jj] leaf is the cores sliced at [ii, jj]
        >>> sliced = bvf.T3Variations(
        ...     tuple(c[ii, jj] for c in tucker_variations), tuple(c[ii, jj] for c in tt_variations))
        >>> print(np.allclose(cw.corewise_norm(cw.corewise_sub(VV[ii][jj].data, sliced.data)), 0.0))
        True
        """
        return stacking.apply_func_to_leaf_subtrees(
            stacking.basic_ragged_unstack(self.data, 2),
            lambda x: T3Variations(*x),
            self.data, # leaf_structure
        )

    @staticmethod
    def stack(
            xx, # Array-like tree of T3Variations
    ):
        """Stack array-like tree of T3Variations into a single T3Variations.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.frame_variations_format as bvf
        >>> import t3toolbox.corewise as cw
        >>> np.random.seed(0)
        >>> rnd = lambda x: np.random.randn(*x)
        >>> ss = (2, 3)                                   # stack
        >>> NN, nnD, nnU, rrL, rrR = (10, 11, 12), (8, 7, 8), (6, 7, 8), (2, 3, 4, 3), (5, 4, 6, 1)
        >>> tucker_variations = tuple(rnd(ss+(n, N)) for n, N in zip(nnU, NN))
        >>> tt_variations = tuple(rnd(ss+(rL, n, rR)) for rL, n, rR in zip(rrL[:-1], nnD, rrR[1:]))
        >>> V = bvf.T3Variations(tucker_variations, tt_variations)
        >>> V2 = bvf.T3Variations.stack(V.unstack())      # stack is the inverse of unstack
        >>> print(np.allclose(cw.corewise_norm(cw.corewise_sub(V.data, V2.data)), 0.0))
        True
        """
        xx_tuples = stacking.apply_func_to_leaf_subtrees(
            xx,
            lambda x: x.data,
            None,  # leaf_structure
        )
        result = stacking.basic_ragged_stack(xx_tuples)
        return T3Variations(*result)

    @staticmethod
    def zeros(
            variation_shapes,                           # (tucker_variation_shapes, tt_variation_shapes)
            stack_shape:    typ.Tuple[int, ...] = (),   # full leading stack K + C
            use_jax:        bool = False,
    ) -> 'T3Variations':
        """Zero variations of the given structure (additive identity).

        ``variation_shapes = (tucker_variation_shapes, tt_variation_shapes)`` -- e.g. a frame's
        :py:attr:`T3Frame.variation_shapes`. (See :py:meth:`zeros_like` to take the structure from an object.)
        """
        return T3Variations(*fv_operations.fv_variations_zeros(variation_shapes, stack_shape, use_jax))

    @staticmethod
    def randn(
            variation_shapes,                           # (tucker_variation_shapes, tt_variation_shapes)
            stack_shape:    typ.Tuple[int, ...] = (),   # full leading stack K + C
            use_jax:        bool = False,
    ) -> 'T3Variations':
        """Variations with i.i.d. N(0,1) core entries (corewise, ungauged). See :py:meth:`randn_like`."""
        return T3Variations(*fv_operations.fv_variations_randn(variation_shapes, stack_shape, use_jax))

    @staticmethod
    def unit(
            variation_shapes,                                         # (tucker_variation_shapes, tt_variation_shapes)
            index:          typ.Tuple[bool, int, typ.Sequence[int]],  # (use_tt_coordinate, i, within_index)
            stack_shape:    typ.Tuple[int, ...] = (),                 # full leading stack K + C
            use_jax:        bool = False,
    ) -> 'T3Variations':
        """Canonical unit variation: zero except a single core entry set to 1.

        ``index = (use_tt_coordinate, i, within_index)`` selects the core (a tt-variation if
        ``use_tt_coordinate`` else a tucker-variation), its position ``i``, and the within-core index.
        These units are the standard basis of the variation cores -- an **overcomplete, non-ambient-
        orthogonal** generating set of the tangent space, not an orthonormal basis.
        """
        return T3Variations(*fv_operations.fv_variations_unit(variation_shapes, index, stack_shape, use_jax))

    @staticmethod
    def zeros_like(x) -> 'T3Variations':
        """Zero variations matching the structure (shapes + stack) of ``x`` (a T3Frame or T3Variations)."""
        return T3Variations.zeros(x.variation_shapes, stack_shape=x.stack_shape, use_jax=x.contains_jax)

    def allclose(
            self,
            other: 'T3Variations',

            rtol:  typ.Optional[float] = None,  # None: the ambient jax-aware default (safety.comparison_rtol)
            atol:  typ.Optional[float] = None,  # None: 0.0
    ) -> NDArray:  # bool, shape = stack_shape (K+C); scalar unstacked; reduce with .all()
        """True where the variation cores are numerically equal (coordinate-level), per stack
        element: ``||self - other|| <= atol + rtol * max(||self||, ||other||)`` in the corewise
        norm. Structural mismatches raise. Bitwise: :py:meth:`corewise_equal`; ``==`` is undefined."""
        if rtol is None:
            rtol = safety.comparison_rtol(self.data, other.data)
        atol = 0.0 if atol is None else atol
        n_stack = np.ndim(self.data[0][0]) - 2                  # tucker variation core = stack+(nU, N)
        use_jax = common.tree_contains_jax((self.data, other.data))
        xnp, _, _ = common.get_backend(False, use_jax)
        diff = cw.corewise_sub(self.data, other.data)
        dn = cw.corewise_stack_dot(diff, diff, n_stack) ** 0.5
        rn = xnp.maximum(cw.corewise_stack_dot(self.data, self.data, n_stack),
                         cw.corewise_stack_dot(other.data, other.data, n_stack)) ** 0.5
        return dn <= atol + rtol * rn

    def corewise_equal(
            self,
            other: 'T3Variations',
    ) -> bool:
        """Bitwise equality of the stored variation cores (``False`` on mismatch, never raises)."""
        return type(other) is type(self) and cw.corewise_equal(self.data, other.data)

    @staticmethod
    def randn_like(x) -> 'T3Variations':
        """Random variations matching the structure (shapes + stack) of ``x`` (a T3Frame or T3Variations)."""
        return T3Variations.randn(x.variation_shapes, stack_shape=x.stack_shape, use_jax=x.contains_jax)

    def to_vector(self) -> NDArray:
        """Flatten the variation cores to a 1D vector (the tangent's degrees of freedom)."""
        return t3_conversions.t3_to_vector(self.data)

    @staticmethod
    def from_vector(
            flat,                                       # shape=(size,)
            variation_shapes,                           # (tucker_variation_shapes, tt_variation_shapes)
            stack_shape:    typ.Tuple[int, ...] = (),   # full leading stack K + C
    ) -> 'T3Variations':
        """Inverse of :py:meth:`to_vector`: rebuild variations of the given structure from a 1D vector.

        ``variation_shapes = (tucker_variation_shapes, tt_variation_shapes)`` (e.g. a frame's
        :py:attr:`T3Frame.variation_shapes`); ``stack_shape`` is the full leading stack ``K + C``.
        """
        return T3Variations(*fv_operations.fv_variations_from_vector(flat, variation_shapes, stack_shape))

    def save(self, file) -> None:
        """Save the variation cores to a ``.npz`` file (load with :py:meth:`load`)."""
        save_core_families(file, self.data)

    @staticmethod
    def load(
            file,                       # path or file-like (.npz saved by save)
            use_jax:    bool = False,
    ) -> 'T3Variations':
        """Load variations saved by :py:meth:`save`."""
        f = load_core_families(file)
        v = T3Variations(f[0], f[1])
        return v.to_jax() if use_jax else v

    def reverse(self) -> 'T3Variations':
        """Reverse the mode order (corewise): reverse the tucker-variation order and reverse+transpose
        the tt-variations (bond swap), matching :py:meth:`T3Frame.reverse`."""
        return T3Variations(tuple(V.copy() for V in self.tucker_variations[::-1]),
                            tt_operations.tt_reverse(self.tt_variations))

    def sum_stack(self, axis=None) -> 'T3Variations':
        """Corewise sum over stack axes (a batch of variations -> their sum). ``axis`` indexes the stack
        (default: the whole stack). For variations the corewise sum *is* the tangent sum, by linearity."""
        return T3Variations(*cw.corewise_stack_sum(self.data, axis, len(self.stack_shape)))

    def __add__(self, other: 'T3Variations') -> 'T3Variations':
        """Corewise sum (variations form a vector space)."""
        return T3Variations(*cw.corewise_add(self.data, other.data))

    def __sub__(self, other: 'T3Variations') -> 'T3Variations':
        """Corewise difference."""
        return T3Variations(*cw.corewise_sub(self.data, other.data))

    def __mul__(self, scalar) -> 'T3Variations':
        """Corewise scalar multiplication."""
        return T3Variations(*cw.corewise_scale(self.data, scalar))

    __rmul__ = __mul__

    def __neg__(self) -> 'T3Variations':
        """Corewise negation."""
        return T3Variations(*cw.corewise_neg(self.data))

    def allclose(self, other: 'T3Variations', rtol: float = 1e-9, atol: float = 0.0) -> NDArray:  # bool array, stack
        """``True`` (per stack element) if ``other`` holds the same variations as ``self``, corewise.

        Checks ``||self - other|| <= atol + rtol * ||other||`` in the **stack-vectorized** corewise norm
        (:py:func:`t3toolbox.corewise.corewise_stack_norm`), one verdict per stack slice (reduce with
        ``.all()`` for a single bool). Split-agnostic, like all :py:class:`T3Variations` operations.
        """
        dn = cw.corewise_stack_norm((self - other).data, len(self.stack_shape))
        rn = cw.corewise_stack_norm(other.data, len(other.stack_shape))
        return dn <= atol + rtol * rn


def check_fv_pair(
        frame:      T3Frame,        # stack_shape = C (frame/core stack)
        variations: T3Variations,   # stack_shape = K + C (frame stack is its inner/trailing part)
) -> None:
    """Check rank and shape consistency between T3Frame and T3Variations.

    This ensures that the variation cores (V, H) have the correct dimensions to interface with the
    frame cores (D, U, L, R), and that their stacks are compatible.

    Stacking: a variation may carry extra *outer* tangent-stack axes -- a batch of tangent vectors
    sharing the same base point -- so its ``stack_shape`` is ``tangent_stack_shape + frame_stack_shape``
    (extra axes outermost, frame stack innermost). Consistency requires ``frame.stack_shape`` to be the
    trailing (inner) part of ``variations.stack_shape``; ``tangent_stack_shape`` may be empty, which
    is the common case.

    Examples
    --------
    Variations whose stack matches the frame and whose cores fill the frame's holes are consistent
    (``check_fv_pair`` returns ``None`` and raises nothing):

    >>> import numpy as np
    >>> import t3toolbox.frame_variations_format as bvf
    >>> ss = (2, 3)                                       # frame stack shape C
    >>> down_tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
    >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3, 12, 5)))
    >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
    >>> up_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
    >>> frame = bvf.T3Frame(down_tucker_cores, up_tt_cores, left_tt_cores, right_tt_cores)
    >>> tucker_shapes, tt_shapes = frame.variation_shapes
    >>> variations = bvf.T3Variations(tuple(np.ones(ss + s) for s in tucker_shapes),
    ...                               tuple(np.ones(ss + s) for s in tt_shapes))
    >>> print(bvf.check_fv_pair(frame, variations))   # consistent -> returns None
    None

    A variation may carry extra *outer* tangent-stack axes ``K`` (here ``K = (4,)``, so its stack is
    ``(4,) + (2, 3)``) -- still consistent, since the frame stack is the inner suffix of the variation stack:

    >>> vss = (4,) + ss                                   # tangent_stack_shape K + frame_stack_shape C
    >>> v_stacked = bvf.T3Variations(tuple(np.ones(vss + s) for s in tucker_shapes),
    ...                              tuple(np.ones(vss + s) for s in tt_shapes))
    >>> print(bvf.check_fv_pair(frame, v_stacked))
    None

    Gotcha -- variation cores that do not fit the frame's holes raise (structural error):

    >>> bad_shapes = ((tucker_shapes[0][0] + 1, tucker_shapes[0][1]),) + tucker_shapes[1:]
    >>> bad = bvf.T3Variations(tuple(np.ones(ss + s) for s in bad_shapes),
    ...                        tuple(np.ones(ss + s) for s in tt_shapes))
    >>> bvf.check_fv_pair(frame, bad)                  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError
    """
    frame_stack = frame.stack_shape
    var_stack = variations.stack_shape
    # The frame stack must be the trailing (inner) part of the variation stack; the variation may
    # carry extra *outer* tangent-stack axes (variation stack = tangent_stack_shape + frame_stack_shape).
    if var_stack[len(var_stack) - len(frame_stack):] != frame_stack:
        raise ValueError(
            'Inconsistent (T3Frame, T3Variations) pair.\n'
            'The frame stack_shape must be the trailing (inner) part of the variation stack_shape.\n'
            + str(frame_stack) + ' = frame.stack_shape is not a suffix of '
            + str(var_stack) + ' = variations.stack_shape'
        )

    xVV, xHH = frame.variation_shapes
    yVV, yHH = variations.variation_shapes
    if len(yVV) != len(xVV) or len(yHH) != len(xHH):
        raise ValueError(
            'Inconsistent (T3Frame, T3Variations) pair: the frame has d=%d cores, the variations have d=%d '
            '(a zip over the holes would silently compare only the first min(d) of them).'
            % (len(xVV), len(yVV)))

    for ii, (xV, yV) in enumerate(zip(xVV, yVV)):
        if xV != yV:
            raise ValueError(
                'Inconsistent (T3Frame, T3Variations) pair.\n'
                + str(ii) + '-th Tucker variation shape' + str(yV)
                + ' does not fit frame hole ' + str(xV)
            )

    for ii, (xH, yH) in enumerate(zip(xHH, yHH)):
        if xH != yH:
            raise ValueError(
                'Inconsistent (T3Frame, T3Variations) pair.\n'
                + str(ii) + '-th tensor train variation shape' + str(yH)
                + ' does not fit frame hole ' + str(xH)
            )


def fv_to_t3(
        index:      typ.Tuple[
            bool,  # TT core (True) or Tucker core (False)
            int,   # number of the non-orthogonal core, 1...d-1
        ],
        frame:      T3Frame,        # stack_shape = C (frame/core stack)
        variations: T3Variations,   # stack_shape = K + C (frame stack is its inner/trailing part)
) -> t3.TuckerTensorTrain:
    '''Convert frame-variations representation to TuckerTensorTrain.

    If replacement_ind=1, replace_tt=True::

        1 -- L0 --(H1)-- R2 -- R3 -- 1
             |     |     |     |
             D0    D1    D2    D3
             |     |     |     |

    If replacement_ind=2, replace_tt=False::

        1 -- L0 -- L1 -- U2 -- R3 -- 1
             |     |     |     |
             D0    D1   (V2)   D3
             |     |     |     |

    These are the single-core variation terms summed in equation (46), Appendix A.3, of Alger et al.
    (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).

    Parameters
    ----------
    ii: int
        Index of variation. 0 <= replacement_ind < num_cores
    replace_tt: bool
        Indicates whether to use TT variation (True) or a Tucker variation (False)
    frame: T3Frame
        Frame cores
    variations: T3Variations
        Variation cores

    Raises
    ------
    RuntimeError
        - Error raised if the frame and variations do not fit with each other

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.frame_variations_format as bvf
    >>> import t3toolbox.corewise as cw
    >>> np.random.seed(0)
    >>> randn = np.random.randn
    >>> (D0, D1, D2) = (randn(10, 14), randn(11, 15), randn(12, 16))
    >>> (L0, L1, L2) = (randn(1, 10, 2), randn(2, 11, 3), randn(3, 12, 4))
    >>> (R0, R1, R2) = (randn(2, 10, 4), randn(4, 11, 5), randn(5, 12, 1))
    >>> (U0, U1, U2) = (randn(1, 9, 4), randn(2, 8, 5), randn(3, 7, 1))
    >>> frame = bvf.T3Frame((D0, D1, D2), (U0, U1, U2), (L0, L1, L2), (R0, R1, R2))
    >>> (V0, V1, V2) = (randn(9, 14), randn(8, 15), randn(7, 16))
    >>> (H0, H1, H2) = (randn(1, 10, 4), randn(2, 11, 5), randn(3, 12, 1))
    >>> variations = bvf.T3Variations((V0, V1, V2), (H0, H1, H2))

    Replacing the index-1 TT-core swaps ``H1`` into the right-orthogonal chain ``L0, ?, R2``; the
    Tucker (down) cores are unchanged:

    >>> tt_term = bvf.fv_to_t3((True, 1), frame, variations)
    >>> expected = ((D0, D1, D2), (L0, H1, R2))      # down cores untouched; TT chain = L0, H1, R2
    >>> print(np.allclose(cw.corewise_norm(cw.corewise_sub(tt_term.data, expected)), 0.0))
    True

    Replacing the index-1 Tucker core swaps ``V1`` into the down cores and the up core ``U1`` into the chain:

    >>> tucker_term = bvf.fv_to_t3((False, 1), frame, variations)
    >>> expected = ((D0, V1, D2), (L0, U1, R2))
    >>> print(np.allclose(cw.corewise_norm(cw.corewise_sub(tucker_term.data, expected)), 0.0))
    True
    '''
    check_fv_pair(frame, variations)
    # The term mixes a V+G-stacked variation core with G-stacked frame cores (when the variation
    # carries an extra tangent stack K); broadcast all cores to the common K+C stack so the result is
    # a valid (uniform-stack) TuckerTensorTrain. A no-op when there is no tangent stack (K=()).
    cores = t3_operations.t3_broadcast_to_common_stack(
        *fv_conversions.fv_to_t3(index, frame.data, variations.data)
    )
    return t3.TuckerTensorTrain(*cores)


def t3_orthogonal_representations(
        x: t3.TuckerTensorTrain,
        already_left_orthogonal: bool = False,
        squash_tails: bool = True,
) -> typ.Tuple[
    T3Frame,  # orthogonal frame
    T3Variations,  # variations
]:
    '''Construct frame-variation representations of TuckerTensorTrain with orthogonal frame.

    Input TuckerTensorTrain::

                  1 -- G0 -- G1 -- G2 -- G3 -- 1
        X    =         |     |     |     |
                       B0    B1    B2    B3
                       |     |     |     |

    Frame-variation representation with non-orthogonal TT-core H1::

                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       D0    D1    D2    D3
                       |     |     |     |

    Frame-variation representation with non-orthogonal tucker core V2::

                  1 -- L0 -- L1 -- U2 -- R3 -- 1
        X    =         |     |     |     |
                       D0    D1    V2    D3
                       |     |     |     |

    The input tensor train x is defined by:
        - x_tucker_cores    = (B0, B1, B2, B3)
        - x_tt_cores        = (G0, G1, G2, G3)
    The "frame cores" are:
        - down_tucker_cores = (D0, D1, D2, D3), down orthogonal
        - up_tt_cores       = (U0, U1, U2, U3), up orthogonal
        - left_tt_cores     = (L0, L1, L2),     left orthogonal
        - right_tt_cores    = (R1, R2, R3),     right orthogonal
    The "variation cores" are:
        - tucker_variations  = (V0, V1, V2, V3)
        - tt_variations     = (H0, H1, H2, H3)

    Implements the sweeping orthogonalization (Algorithm 11), producing the representations
    (44)-(45), in Appendix A.3 of Alger et al. (2026), "Tucker Tensor Train Taylor Series"
    (arXiv:2603.21141). NOTE: the left/right orthogonalization sweep order here differs from
    Algorithm 11 (left-then-right vs the paper's right-then-left); the resulting orthogonal
    representations are equivalent.

    Parameters
    ----------
    x: TuckerTensorTrain
        Input TuckerTensorTrain
        x = (x_tucker_cores, x_tt_cores)
        x_tucker_cores = (B0, ..., B(d-1))
        x_tt_cores = (G0, ..., G(d-1))

    Returns
    -------
    T3Frame
        Orthogonal frame for frame-variation representations of x.
    T3Variations
        Variations for frame-variation representations of x.

    Examples
    --------
    Orthogonalize a (stacked) T3. The frame reconstructs the *same* tensor x -- either by dropping the
    index-1 TT variation H1 into the chain, or the index-1 Tucker variation V1 (these are two of the
    single-core terms of :py:func:`fv_to_t3`):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.frame_variations_format as bvf
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (3, 3, 2, 1), stack_shape=(2, 3))
    >>> frame, variations = bvf.t3_orthogonal_representations(x)
    >>> x_tt = bvf.fv_to_t3((True, 1), frame, variations)    # frame with TT-variation H1 in the chain
    >>> print(np.allclose(x.to_dense(), x_tt.to_dense()))    # still represents the original tensor
    True
    >>> x_tk = bvf.fv_to_t3((False, 1), frame, variations)   # frame with Tucker-variation V1
    >>> print(np.allclose(x.to_dense(), x_tk.to_dense()))
    True

    The frame cores are orthogonal in their respective senses (the point of this routine). ``frame`` is
    ``(2, 3)``-stacked, so ``is_orthogonal()`` returns a per-element bool array; ``.all()`` summarizes it:

    >>> print(frame.is_orthogonal().shape, frame.is_orthogonal().all())
    (2, 3) True
    >>> print(frame.shape, frame.stack_shape)                 # shape and stack are preserved
    (14, 15, 16) (2, 3)
    '''
    result = fv_conversions.t3_orthogonal_representations(
        x.data, already_left_orthogonal=already_left_orthogonal, squash_tails=squash_tails,
    )
    return T3Frame(*result[0]), T3Variations(*result[1])


@dataclass(frozen=True, eq=False)   # eq=False -> the ExplicitEquality mixin stands
class T3FrameWeights(common.ExplicitEquality):
    """Diagonal weights defining a **metric on the tangent coordinates** -- a Grasedyck-Kramer-style
    reweighting of the ``d`` variation directions (penalise poorly-informed ones with e.g. ``1/sigma``).

    Four families, each ``len=d`` (one per variation core, Approach-1 / metric-on-variations):
    ``down`` (on ``H``'s ``nD`` leg), ``up`` (on ``V``'s ``nU`` leg), ``left`` (``H``'s ``rL``),
    ``right`` (``H``'s ``rR``). :py:meth:`T3Tangent.weighted_norm` / :py:meth:`~T3Tangent.weighted_inner`
    absorb these into the variation cores and take the coordinate norm/inner -- the frame stays orthonormal
    and untouched.

    **Batching: a weight is FRAME-like** (it is *absorbed into* the variations, but it *batches with* the
    frame -- do not conflate the two). Every vector is ``stack_shape + (rank,)``, and that stack is the frame
    stack ``C``, like :py:class:`T3Frame` -- **not** the variations' ``K + C``: a weight is one metric per
    base point, shared by all ``K`` tangent vectors at that frame, and it broadcasts over ``K`` for free
    (``C`` is innermost). Pairing with variations alone follows the same trailing-stack rule as
    :py:func:`check_fv_pair` (the weight's stack is the trailing/inner part of the variation stack); at the
    tangent level, where the frame is present too, :py:func:`check_fw_pair` enforces the exact
    ``weights.stack_shape == frame.stack_shape``.
    """
    down_weights:  typ.Tuple[NDArray, ...]  # len=d, elm_shape=stack_shape+(nDi,)
    up_weights:    typ.Tuple[NDArray, ...]  # len=d, elm_shape=stack_shape+(nUi,)
    left_weights:  typ.Tuple[NDArray, ...]  # len=d, elm_shape=stack_shape+(rLi,)
    right_weights: typ.Tuple[NDArray, ...]  # len=d, elm_shape=stack_shape+(rRi,)

    @ft.cached_property
    def data(self) -> typ.Tuple[typ.Tuple[NDArray, ...], ...]:  # (down, up, left, right)
        return self.down_weights, self.up_weights, self.left_weights, self.right_weights

    def allclose(
            self,
            other: 'T3FrameWeights',

            rtol:  typ.Optional[float] = None,  # None: the ambient jax-aware default (safety.comparison_rtol)
            atol:  typ.Optional[float] = None,  # None: 0.0
    ) -> NDArray:  # bool, shape = stack_shape (C); scalar unstacked; reduce with .all()
        """True where the four weight families are numerically equal, per stack element (norm-based:
        ``atol + rtol * max(||self||, ||other||)``). Structural mismatches raise. Bitwise:
        :py:meth:`corewise_equal`; ``==`` is undefined."""
        if rtol is None:
            rtol = safety.comparison_rtol(self.data, other.data)
        atol = 0.0 if atol is None else atol
        n_stack = np.ndim(self.down_weights[0]) - 1
        use_jax = common.tree_contains_jax((self.data, other.data))
        xnp, _, _ = common.get_backend(False, use_jax)
        diff = cw.corewise_sub(self.data, other.data)
        dn = cw.corewise_stack_dot(diff, diff, n_stack) ** 0.5
        rn = xnp.maximum(cw.corewise_stack_dot(self.data, self.data, n_stack),
                         cw.corewise_stack_dot(other.data, other.data, n_stack)) ** 0.5
        return dn <= atol + rtol * rn

    def corewise_equal(
            self,
            other: 'T3FrameWeights',
    ) -> bool:
        """Bitwise equality of the stored weight vectors (``False`` on mismatch, never raises)."""
        return type(other) is type(self) and cw.corewise_equal(self.data, other.data)

    @ft.cached_property
    def d(self) -> int:
        return len(self.down_weights)

    @ft.cached_property
    def down_ranks(self) -> typ.Tuple[int, ...]:
        return tuple(int(w.shape[-1]) for w in self.down_weights)

    @ft.cached_property
    def up_ranks(self) -> typ.Tuple[int, ...]:
        return tuple(int(w.shape[-1]) for w in self.up_weights)

    @ft.cached_property
    def left_ranks(self) -> typ.Tuple[int, ...]:
        return tuple(int(w.shape[-1]) for w in self.left_weights)

    @ft.cached_property
    def right_ranks(self) -> typ.Tuple[int, ...]:
        return tuple(int(w.shape[-1]) for w in self.right_weights)

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int, ...]:
        return self.down_weights[0].shape[:-1]

    def validate(self):
        """Structural: four families each of length d, and one uniform stack_shape on every vector."""
        ss = self.stack_shape
        for name, fam in (('down', self.down_weights), ('up', self.up_weights),
                          ('left', self.left_weights), ('right', self.right_weights)):
            if len(fam) != self.d:
                raise ValueError('Inconsistent T3FrameWeights.\n' + str(self.d) + ' = d, but len('
                                 + name + '_weights) = ' + str(len(fam)))
            for ii, w in enumerate(fam):
                if w.ndim < 1 or w.shape[:-1] != ss:
                    raise ValueError('Inconsistent T3FrameWeights.\n' + name + '_weights[' + str(ii)
                                     + '].shape = ' + str(w.shape) + ' is not stack_shape ' + str(ss) + ' + (rank,).')

    def __post_init__(self):
        self.validate()

    def is_consistent_with(self, tangent) -> bool:
        """True iff these weights match the variation ranks + stack of a ``T3Tangent`` (or ``T3Variations``)."""
        variations = tangent.variations if hasattr(tangent, 'variations') else tangent
        return fv_operations.fv_weights_consistent(variations.data, self.data)

    def reciprocal(self) -> 'T3FrameWeights':
        """Elementwise ``1/w`` on every family (e.g. inverse-singular-value weights)."""
        return T3FrameWeights(*[tuple(1.0 / w for w in fam) for fam in self.data])

    def sqrt(self) -> 'T3FrameWeights':
        """Elementwise ``sqrt`` on every family."""
        xnp, _, _ = get_backend(False, tree_contains_jax(self.data))
        return T3FrameWeights(*[tuple(xnp.sqrt(w) for w in fam) for fam in self.data])

    def reverse(self) -> 'T3FrameWeights':
        """Reverse the mode order, swapping ``left``<->``right`` (mirrors :py:meth:`T3Frame.reverse`)."""
        return T3FrameWeights(self.down_weights[::-1], self.up_weights[::-1],
                              self.right_weights[::-1], self.left_weights[::-1])

    def concatenate(self, other: 'T3FrameWeights') -> 'T3FrameWeights':
        """Per-edge concatenation (the ``+`` combine; ranks add)."""
        return T3FrameWeights(*fv_operations.fv_concatenate_weights(self.data, other.data))

    def kronecker(self, other: 'T3FrameWeights') -> 'T3FrameWeights':
        """Per-edge Kronecker product (the Hadamard combine; ranks multiply)."""
        return T3FrameWeights(*fv_operations.fv_kronecker_weights(self.data, other.data))

    def unstack(self):
        """Unstack a stack of frame-weights into an array-like tree (mirrors ``T3Variations.unstack``)."""
        result_tuple = stacking.basic_ragged_unstack(self.data, 1)
        return stacking.apply_func_to_leaf_subtrees(result_tuple, lambda x: T3FrameWeights(*x), self.data)

    @staticmethod
    def stack(xx) -> 'T3FrameWeights':
        """Stack an array-like tree of frame-weights into one (mirrors ``T3Variations.stack``)."""
        xx_tuples = stacking.apply_func_to_leaf_subtrees(xx, lambda x: x.data, None)
        return T3FrameWeights(*stacking.basic_ragged_stack(xx_tuples))

    @classmethod
    def from_t3weights(cls, t3_weights: 't3.T3Weights') -> 'T3FrameWeights':
        """Build a tangent metric from base-point edge weights (e.g. ``T3Weights.from_t3svd(x)``):
        ``down = up = tucker_weights``, ``left = tt_weights[:-1]``, ``right = tt_weights[1:]``. The TT
        slicing follows the ``Hᵢ`` bond convention (``Hᵢ``'s left bond is TT bond ``i``, right bond
        ``i+1``) -- simple but convention-dependent, hence a named method. The result pairs with a
        **minimal-rank** tangent at ``x`` (where the complement rank ``nU`` equals the Tucker rank ``nD``,
        as for ``t3svd`` output); the Grasedyck–Kramer metric is
        ``T3FrameWeights.from_t3weights(T3Weights.from_t3svd(x)).reciprocal()``."""
        return cls(*fv_operations.t3weights_to_t3frameweights(t3_weights.data))


def t3svd_orthogonal_representations(
        x: 't3.TuckerTensorTrain',
        **t3svd_kwargs,                 # passed to TuckerTensorTrain.t3svd (max_*_ranks, rtol, atol, sharing, ...)
) -> typ.Tuple[
    T3Frame,           # orthogonal frame at the t3svd result, in the t3svd GAUGE (its Tucker basis = the singular basis)
    T3Variations,      # the variations of that representation
    't3.T3Weights',    # the singular values, ready for T3FrameWeights.from_t3weights (one SVD, not two)
]:
    '''The orthogonal frame of ``x`` **in the T3-SVD gauge**, with the singular values it came with.

    Composes ``x.t3svd(**t3svd_kwargs)`` with :py:func:`t3_orthogonal_representations` called with
    ``already_left_orthogonal=True`` -- the flag that matters: a T3-SVD result is left-orthogonal, and the
    default sweep would re-SVD its already-orthonormal Tucker factors, whose spectrum is degenerate, so the
    frame's Tucker basis would come out **rotated by an arbitrary orthogonal matrix** relative to the
    singular basis. Per-coordinate singular-value weights (``T3FrameWeights.from_t3weights``, the
    Grasedyck-Kramer metric of ``docs/weighting.md``) are only meaningful in the singular basis, which this
    frame carries and the default frame does not (the 2026-08-22 review, S14). One SVD instead of two
    (``T3Weights.from_t3svd`` discards the train it decomposed).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.frame_variations_format as bvf
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))
    >>> frame, variations, sigma = bvf.t3svd_orthogonal_representations(x)
    >>> xs, _, _ = x.t3svd()
    >>> print(all(np.allclose(D, Bx) for D, Bx in zip(frame.down_tucker_cores, xs.tucker_cores)))  # same gauge
    True
    >>> W = bvf.T3FrameWeights.from_t3weights(sigma)     # the sigma-metric on this frame's coordinates
    >>> print(W.is_consistent_with(variations))
    True
    '''
    xs, tucker_svals, tt_svals = x.t3svd(**t3svd_kwargs)
    frame, variations = t3_orthogonal_representations(xs, already_left_orthogonal=True)
    return frame, variations, t3.T3Weights(tuple(tucker_svals), tuple(tt_svals))


def check_fw_pair(
        frame:   T3Frame,          # stack_shape = C (frame/core stack)
        weights: T3FrameWeights,   # stack_shape = C -- a weight is FRAME-LIKE: one metric per base point
) -> None:
    """Check that ``weights`` is a metric on the tangent coordinates **at this frame**.

    The weight<->frame analog of :py:func:`check_fv_pair`. A :py:class:`T3FrameWeights` is **frame-like**:
    it carries the frame stack ``C`` (one metric per base point, shared by every tangent there), and its
    four families weight the variation holes ``frame`` leaves -- ``down``<->``nD``, ``up``<->``nU``,
    ``left``<->``rL``, ``right``<->``rR``. So the stack must match ``frame.stack_shape`` **exactly** (not
    merely be a trailing part of it, as when pairing with variations alone).

    Why the stricter check lives here: absorption only needs the weight's stack to be the *trailing* part
    of the variation stack (:py:func:`~t3toolbox.backend.fv_operations.fv_weights_consistent`, which is
    blind to the frame, like the variations themselves). A ``K + C``-stacked weight also satisfies that --
    it reads as ``C_w = K + C`` (that many base points, one tangent each), a legitimate absorption -- so it
    would silently weight a ``C``-framed tangent's ``K`` coordinates with ``K`` *different* metrics. This
    is the only place where both objects are present, hence the only place with enough information to
    reject it. Structural (shapes only) -> raises in both safety modes; jit-safe.
    """
    if weights.stack_shape != frame.stack_shape:
        raise ValueError(
            'Inconsistent (T3Frame, T3FrameWeights) pair.\n'
            'A T3FrameWeights is a metric at a base point, so it carries the FRAME stack C exactly (the\n'
            'variations carry K + C; a K-batch of tangents at one frame shares the one metric).\n'
            + str(weights.stack_shape) + ' = weights.stack_shape != '
            + str(frame.stack_shape) + ' = frame.stack_shape'
        )

    tucker_holes, tt_holes = frame.variation_shapes   # (nU, N) per core; (rL, nD, rR) per core
    families = (
        ('down',  weights.down_ranks,  tuple(h[1] for h in tt_holes)),
        ('up',    weights.up_ranks,    tuple(h[0] for h in tucker_holes)),
        ('left',  weights.left_ranks,  tuple(h[0] for h in tt_holes)),
        ('right', weights.right_ranks, tuple(h[2] for h in tt_holes)),
    )
    for name, weight_ranks, hole_ranks in families:
        if weight_ranks != hole_ranks:
            raise ValueError(
                'Inconsistent (T3Frame, T3FrameWeights) pair.\n'
                + str(weight_ranks) + ' = weights.' + name + '_ranks does not match '
                + str(hole_ranks) + ' = the frame\'s ' + name + ' variation-hole ranks.'
            )


def fv_absorb_weights(variations: T3Variations, weights: T3FrameWeights) -> T3Variations:
    """Absorb the metric ``weights`` into the variation cores (``up``->V, ``down``/``left``/``right``->H),
    returning the weighted :py:class:`T3Variations` (the frame is unchanged). The coordinate norm
    (``corewise_norm``) of the result is the weighted tangent norm -- see
    :py:meth:`~t3toolbox.manifold.T3Tangent.weighted_norm`, and the helper method
    :py:meth:`~t3toolbox.manifold.T3Tangent.absorb_weights`. Frontend of
    :py:func:`t3toolbox.backend.fv_operations.fv_absorb_weights`."""
    return T3Variations(*fv_operations.fv_absorb_weights(variations.data, weights.data))


if jax_available:
    import jax

    # Register as jax pytrees so they can be jit/vmap/grad-ed. Leaves = the cores (x.data); no aux.
    jax.tree_util.register_pytree_node(
        T3Frame,
        lambda x: (x.data, None),
        lambda aux_data, children: T3Frame(*children),
    )
    jax.tree_util.register_pytree_node(
        T3Variations,
        lambda x: (x.data, None),
        lambda aux_data, children: T3Variations(*children),
    )
    jax.tree_util.register_pytree_node(
        T3FrameWeights,
        lambda x: (x.data, None),
        lambda aux_data, children: T3FrameWeights(*children),
    )


# --------------------------------------------------------------------------------------------------
# The aligned padded frame (frontend twins of backend.padded_frames)
# --------------------------------------------------------------------------------------------------
def _structurally_zero(
        tucker_ranks:  typ.Sequence[int],   # len=d
        tt_ranks:      typ.Sequence[int],   # len=d+1
) -> bool:
    """Any Tucker rank or interior TT rank 0: the zero tensor."""
    return any(int(n) == 0 for n in tucker_ranks) or any(int(r) == 0 for r in tt_ranks[1:-1])


def _old_frame_data(
        x_old:    't3.TuckerTensorTrain',
        sharing:  typ.Optional[typ.Sequence],
        who:      str,
) -> typ.Tuple:                            # the old frame's data (the empty frame for a structurally zero x_old)
    """The old point's frame: checked, reduced to minimal ranks (sharing-aware), orthogonalized."""
    if not isinstance(x_old, t3.TuckerTensorTrain):
        hint = ' (convert a UniformTuckerTensorTrain with .to_t3())' if hasattr(x_old, 'to_t3') else ''
        raise TypeError('%s: x_old must be a TuckerTensorTrain, got %s%s' % (who, type(x_old).__name__, hint))
    if _structurally_zero(x_old.tucker_ranks, x_old.tt_ranks):
        return padded_frames.fv_empty_frame(x_old.shape, x_old.stack_shape,
                                            use_jax=common.tree_contains_jax(x_old.data))
    if sharing is not None and safety.checks_active(*x_old.data[0]):
        safety.require(bool(np.all(x_old.has_shared_tucker_factors(sharing))),
                       '%s: x_old does not have tied Tucker factors under sharing=%r (share it first: '
                       'x_old.share(sharing))' % (who, tuple(sharing)))
    minimal = t3.TuckerTensorTrain.get_minimal_ranks(x_old.shape, x_old.tucker_ranks, x_old.tt_ranks, sharing=sharing)
    if (tuple(x_old.tucker_ranks), tuple(x_old.tt_ranks)) != (tuple(minimal[0]), tuple(minimal[1])):
        x_old = x_old.t3svd(sharing=sharing)[0].rank_adjustment_sweep('right_to_left', sharing=sharing)
    return t3_orthogonal_representations(x_old)[0].data


def fv_aligned_padded_frame(
        x_old:               't3.TuckerTensorTrain',              # the old point; rank 0 = the zero tensor
        tucker_ranks:        typ.Sequence[int],                   # len=d, target
        tt_ranks:            typ.Sequence[int],                   # len=d+1, target
        hint:                typ.Any,                             # TuckerTensorTrain / dense array, or anything with `project`
        project:             typ.Optional[typ.Callable] = None,   # project(frame.data, hint) -> variations data; None = by the hint's class

        sharing:             typ.Optional[typ.Sequence] = None,   # len=d group labels; None = unshared
        oversampling:        int   = ALIGN_OVERSAMPLING,          # the backend's named defaults
        oversampled_passes:  int   = ALIGN_OVERSAMPLED_PASSES,
        max_passes:          int   = ALIGN_MAX_PASSES,
        tol:                 float = ALIGN_TOL,
        alternate:           bool  = ALIGN_ALTERNATE,
        seed:                int   = ZERO_FRAME_SEED,
        use_jit:             bool  = False,                       # jax inputs: a memoized jitted step in the host loop
) -> typ.Tuple[
    'T3Frame',                 # the aligned padded frame at the target ranks
    typ.Tuple[NDArray, ...],   # movements, one per pass
]:
    """The padded frame of ``x_old`` at larger ranks, its new directions aligned with a hint.

    **Purpose.** At a zero-padded rank increase the point ``x_old`` gains zero slots, and its orthonormal frame needs
    new directions in every rank store (the four families of frame cores; see :doc:`/frame_variations`, "Four rank
    stores"). Any completion represents the same point, but the tangent space -- and so the first Newton step from it --
    depends on the choice. This function biases the completion toward a HINT: a tensor whose tangent projection should
    be as large as possible. ``optimizers.newton_cg(padded_from=x_old)`` uses the misfit gradient as the hint; here
    the hint is yours. The user guide is :doc:`/aligned_padded_frame`.

    **Method.** A fixed-point iteration on projections. It starts from the hint-free padded frame
    (:py:meth:`T3Frame.padded`). Each pass projects the hint onto the current frame's tangent space and, at every store,
    replaces the new slots by the leading new directions of that projection's content there, given the other stores'
    current completions, with the old slots held fixed (:py:meth:`~t3toolbox.manifold.T3Tangent.align_padded` is one
    pass). The schedule:

    - ``oversampled_passes`` passes at the target ranks plus ``oversampling``;
    - one truncating pass to the target;
    - then two-colour pairs: a ``('D', 'L')`` pass (the down and left stores) and a ``('U', 'R')`` pass (the up and
      right stores), the other colour transported. A simultaneous update can cycle with period 2, which the pairs avoid.

    The loop stops after a pair whose two movements are both below ``tol``, or at ``max_passes`` in all. The movement
    of a pass is how far the new slots turned: the largest, over the stores, of ``‖g_old − P⁺P⁺ᵀ g_old‖_F``.

    **What the fixed point satisfies.** At a fixed point, every store's completion spans the leading new directions of
    its own content, given the other stores' completions, so one more pass moves nothing. The completions are then a
    stationary point of the captured hint energy ``‖Π_{T(F)} hint‖²`` (proved for matrices, measured on T3s) but
    generally not a maximum of it: rotating one store's completion alone never lowers that energy and generically raises
    it at second order. In measurements on small trees, they are a strict local maximum of the cone merit, one of many.
    Updating any single store alone moves nothing either; the first example shows both. The cone merit is the share of
    the hint's normal part that the new directions represent on one side of every edge; its best value does not depend
    on which side. See the math note ``docs/padded_frame_cone.pdf``.

    **Properties.**

    - **No global guarantee.** The merit has many strict local maxima in the measurements (an excursion across several
      edges is captured only when every edge it crosses is aligned with it), and the one reached depends on the start
      and the schedule.
      Oversampling sees such excursions through each edge and usually helps (see the second example).
    - **Representation dependence.** The hint-free start is a function of ``x_old``'s frame AS REPRESENTED, slot signs
      included: its Gaussian seeds do not flip with a slot's sign, and bases chosen differently at singular-value
      ties (a ``t3svd`` output, a shared point) give different starts. Different BLAS builds can therefore give
      different hint-free starts. The aligned output does not have this dependence, up to ``tol``.
    - **Uninformative hints.** Where the hint has no content at an edge (its projection there has rank below the new
      slots), the remaining directions are an arbitrary valid completion.
    - The old slots are bitwise ``x_old``'s frame, and the frame represents ``x_old`` zero-padded.

    **Parameters.**

    - ``hint``: a ``TuckerTensorTrain`` (projected by ``tv_project_t3_onto_tangent_space``) or a dense array of
      ``x_old``'s shape (``tv_project_dense_onto_tangent_space``). Anything else needs ``project(frame_data, hint)``,
      returning raw variation data. A ``UniformTuckerTensorTrain`` hint must be converted first (``hint.to_t3()``);
      passing one raises a ``TypeError`` naming the conversion.
    - ``oversampling`` and ``oversampled_passes``: passes at wider working ranks before the truncation. The phase is
      skipped when the working ranks give the target's stores.
    - ``max_passes``: the cap, the truncating pass included. ``0`` returns the hint-free start.
    - ``tol``: the pair stop. The default ``0.05`` is a coarse alignment; with a small ``tol`` the cap is often the
      effective stop.
    - ``alternate``: the two-colour switch. ``False`` runs simultaneous all-store passes, stopping on one movement.
    - ``seed``: the hint-free start's seed streams.
    - ``use_jit``: see the gotchas.
    - Returns ``(frame, movements)``: the aligned ``T3Frame`` and one movement per pass run (each of shape ``C``).
      When the target adds nothing to ``x_old``'s structure, the frame is ``x_old``'s and there are no passes.

    **Gotchas.**

    - **Eager jax is expensive.** On jax arrays without ``use_jit``, every new structure compiles hundreds of
      operations one by one: about 22 s for the first step at d = 4, and 8,000–9,500 process memory mappings per
      alignment. A rank continuation that aligns eagerly at every level can reach Linux's ``vm.max_map_count`` after
      about seven levels and abort inside XLA. With jax inputs, pass ``use_jit=True``: each step becomes a memoized
      jitted program, two to four per level. numpy inputs need nothing.
    - ``max_passes`` counts the truncating pass, and the stop is tested on pairs only, never on the truncating pass.
    - A NaN movement never stops the loop.
    - Zero passes at the zero tensor give the hint-free start, which is not ``fv_zero_point_frame``'s frame. So in
      ``newton_cg``, ``g0_reference='cone'`` with ``cone_max_passes=0`` is a different reference from ``'zero'``.

    ``x_old`` is reduced to minimal ranks first. A structurally zero ``x_old`` (any Tucker or interior TT rank 0) is
    the zero tensor, with an empty frame.

    Examples
    --------
    Align a padded frame with a hint, and check what the fixed point satisfies:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.frame_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> np.random.seed(0)
    >>> shape, old, target = (5, 6, 5, 6), ((2, 2, 2, 2), (1, 2, 2, 2, 1)), ((3, 4, 4, 3), (1, 3, 4, 3, 1))
    >>> x_old = t3.TuckerTensorTrain.randn(shape, *old)
    >>> hint = t3.TuckerTensorTrain.randn(shape, (5, 6, 5, 6), (1, 5, 25, 6, 1))
    >>> captured = lambda F: float(t3m.MANIFOLD.norm(t3m.MANIFOLD.project_ambient(F, hint)))   # ‖Π_T(F) hint‖
    >>> F, movements = bvf.fv_aligned_padded_frame(x_old, *target, hint, tol=1e-10, max_passes=300)
    >>> F.down_ranks, len(movements) < 300                        # stopped on tol, not on the cap
    ((3, 4, 4, 3), True)

    One more pass moves nothing, and neither does a pass updating any one store alone (``align_padded`` is one pass;
    the stores not in ``update`` are transported): each completion already spans the leading directions of its
    content.

    >>> tangent = t3m.MANIFOLD.project_ambient(F, hint)
    >>> bool(float(tangent.align_padded(*old)[1]) < 1e-8)
    True
    >>> [bool(float(tangent.align_padded(*old, update=(store,))[1]) < 1e-8) for store in ('D', 'U', 'L', 'R')]
    [True, True, True, True]

    The aligned frame's tangent space holds far more of the hint than the hint-free start's:

    >>> captured(F) > 2 * captured(bvf.T3Frame.padded(x_old, *target))
    True

    Oversampling helps at an equal (small) number of passes. Here, on a d = 5 point and a low-rank hint, two passes with
    the first one oversampled capture well over 1.2 times what two passes at the target ranks capture (the margin
    held for every start seed tried):

    >>> np.random.seed(0)
    >>> y_old = t3.TuckerTensorTrain.randn((5,) * 5, (2,) * 5, (1, 2, 2, 2, 2, 1))
    >>> y_hint = t3.TuckerTensorTrain.randn((5,) * 5, (4,) * 5, (1, 4, 6, 6, 4, 1))
    >>> y_target = ((3,) * 5, (1, 3, 3, 3, 3, 1))
    >>> y_captured = lambda F: float(t3m.MANIFOLD.norm(t3m.MANIFOLD.project_ambient(F, y_hint)))
    >>> F0, _ = bvf.fv_aligned_padded_frame(y_old, *y_target, y_hint, oversampling=0, max_passes=2, tol=0.0)
    >>> F5, _ = bvf.fv_aligned_padded_frame(y_old, *y_target, y_hint, oversampling=5, max_passes=2, tol=0.0)
    >>> y_captured(F5) > 1.2 * y_captured(F0)
    True
    """
    old = _old_frame_data(x_old, sharing, 'fv_aligned_padded_frame')
    if project is None:
        if isinstance(hint, t3.TuckerTensorTrain):
            project, hint = tv_operations.tv_project_t3_onto_tangent_space, hint.data
        elif common.is_ndarray(hint):
            if tuple(hint.shape[-len(x_old.shape):]) != tuple(x_old.shape):
                raise ValueError('fv_aligned_padded_frame: a dense hint of shape %r; x_old has shape %r'
                                 % (tuple(hint.shape), tuple(x_old.shape)))
            project = tv_operations.tv_project_dense_onto_tangent_space
        elif hasattr(hint, 'to_t3'):
            raise TypeError('fv_aligned_padded_frame: a %s hint on the ragged layer; convert it with hint.to_t3()'
                            % type(hint).__name__)
        else:
            raise TypeError('fv_aligned_padded_frame: no default projection for a %s hint; pass project=' %
                            type(hint).__name__)
    frame, movements = padded_frames.fv_aligned_padded_frame(
        old, tucker_ranks, tt_ranks, hint, project, sharing=sharing, oversampling=oversampling,
        oversampled_passes=oversampled_passes, max_passes=max_passes, tol=tol, alternate=alternate, seed=seed,
        use_jit=use_jit)
    return T3Frame(*frame), movements
