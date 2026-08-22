# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""
Uniform Tucker tensor trains (UT3): a stacked-supercore + boolean-mask representation of (a stack of)
ragged :py:class:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain` s, laid out for GPU/jit efficiency.

A UT3 pads the cores of a Tucker tensor train up to common sizes ``n`` (Tucker rank), ``r`` (TT rank),
``N`` (mode dimension), stacks the ``d`` cores onto a leading axis to form *supercores*, and records the
real extent of each padded edge with boolean masks. It is, by design, a *faster representation of the
ragged layer* -- see ``docs/uniform_equivalence_contract.md`` and the other ``docs/uniform_*.md`` notes.

NOTE: this module is being built incrementally (slice 1: foundation). Linear algebra, orthogonalization,
SVD, sampling, and the jax pytree registration land in later slices.
"""
import typing as typ
import numpy as np
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence, Tuple

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.ut3_conversions as ut3_conversions
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ut3_operations as ut3_operations
import t3toolbox.backend.ut3_orthogonalization as ut3_orthogonalization
import t3toolbox.backend.ut3_linalg as ut3_linalg
import t3toolbox.backend.ut3_sampling as ut3_sampling
import t3toolbox.backend.sampling_derivatives as sampling_derivatives
import t3toolbox.backend.ut3_svd as ut3_svd
import t3toolbox.backend.ut3_constructors as ut3_constructors
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.sharing as backend_sharing
import t3toolbox.backend.common as common
import t3toolbox.safety as safety
from t3toolbox.backend.common import NDArray

if common.jax_available:
    import jax

__all__ = [
    'UT3Masks',
    'UniformTuckerTensorTrain',
    'UT3Weights',
    'ut3_absorb_weights',
    'ut3_weighted_norm',
    'ut3_weighted_inner',
]

@dataclass(frozen=True, eq=False)  # eq=False -> the mixin's VALUE-based __hash__/__eq__ stand (a bare
class UT3Masks(common.ValueHashedMasks):  # eq=True would fail on arrays). See ValueHashedMasks.
    """The static rank structure of a uniform Tucker tensor train: its two boolean edge masks.

    Slot ``j`` of an edge is real iff its mask is ``True`` there (the prefix/canonical form). Held as a
    separate object so it can ride as jax ``aux_data``. Hash/eq are **value-based** (the
    :py:class:`~t3toolbox.backend.common.ValueHashedMasks` mixin), so a rebuilt-but-identical holder is the
    *same* jit cache key -- no per-iteration recompile in optimization loops; see
    ``docs/contributor/uniform_pytree_composition.md``. (The physical ``shape`` is a separate static int tuple on
    :py:class:`UniformTuckerTensorTrain` -- not a mask, and already value-hashable.)
    """
    tucker_edge_mask: NDArray  # HOST bool, static, shape=(d,)   + stack_shape + (n,)
    tt_edge_mask:     NDArray  # HOST bool, static, shape=(d+1,) + stack_shape + (r,)

    @property
    def data(self) -> Tuple[NDArray, NDArray]:
        """The two raw rank-mask arrays, ``(tucker_edge_mask, tt_edge_mask)``."""
        return self.tucker_edge_mask, self.tt_edge_mask


@dataclass(frozen=True)
class UniformTuckerTensorTrain:
    """A uniform Tucker tensor train: two supercores (the data) + a :py:class:`UT3Masks` holder (the
    static structure).

    - ``tucker_supercore``: shape ``(d,) + stack_shape + (n, N)``
    - ``tt_supercore``: shape ``(d,) + stack_shape + (r, n, r)``
    - ``masks``: the :py:class:`UT3Masks` (shape mask + the two rank masks)

    The mode index ``d`` leads (outside the stack) so sweeps compile to ``lax.scan`` over axis 0
    (``docs/uniform_supercore_layout.md``). Ranks may differ across the stack; the physical shape may
    not (``docs/uniform_ranks_and_varieties.md``).
    """
    tucker_supercore: NDArray         # shape=(d,)+stack_shape+(n,N)
    tt_supercore:     NDArray         # shape=(d,)+stack_shape+(r,n,r)
    shape:            Tuple[int, ...] # len=d; (N0,...,N(d-1)) real mode dims, shared across the stack
    masks:            UT3Masks        # static rank structure (the two edge masks)

    # ----------------------------------------------------------------- views
    @cached_property
    def supercores(self) -> Tuple[NDArray, NDArray]:
        """``(tucker_supercore, tt_supercore)``."""
        return self.tucker_supercore, self.tt_supercore

    @cached_property
    def data(self) -> Tuple[NDArray, NDArray, Tuple[int, ...], Tuple[NDArray, NDArray]]:
        """Raw-array view, mirroring the fields: ``(tucker_supercore, tt_supercore, shape, (2 rank masks))``.

        Backend ``ut3_*`` functions take this layout (supercore-only ops use ``.data[:2]``; the static
        ``shape`` is ``.data[2]``; mask-using ops unpack ``.data[3]``). The ``UT3Masks`` holder stays a
        frontend concern.
        """
        return self.tucker_supercore, self.tt_supercore, self.shape, self.masks.data

    # ------------------------------------------------- padded (uniform) structure
    @cached_property
    def d(self) -> int:
        """Number of modes."""
        return self.tucker_supercore.shape[0]

    @cached_property
    def n(self) -> int:
        """Padded Tucker rank (``n >= max`` of the real Tucker ranks)."""
        return self.tucker_supercore.shape[-2]

    @cached_property
    def N(self) -> int:
        """Padded mode dimension (``N >= max`` of the real shapes)."""
        return self.tucker_supercore.shape[-1]

    @cached_property
    def r(self) -> int:
        """Padded TT rank (``r >= max`` of the real TT ranks)."""
        return self.tt_supercore.shape[-1]

    @cached_property
    def stack_shape(self) -> Tuple[int, ...]:
        """Stack shape (``()`` if unstacked). Lives at axes ``1 .. len(stack_shape)`` (``d`` is axis 0)."""
        return self.tucker_supercore.shape[1:-2]

    @cached_property
    def uniform_structure(self) -> Tuple[int, int, int, int, Tuple[int, ...]]:
        """``(d, N, n, r, stack_shape)`` -- the padded structure."""
        return self.d, self.N, self.n, self.r, self.stack_shape

    # ------------------------------------------------- original (real) structure
    # (`shape` is a stored field above -- the real mode dims, shared across the stack.)

    @cached_property
    def tucker_ranks(self) -> NDArray:  # dtype=int, shape=(d,)+stack_shape
        """Real Tucker ranks (from ``tucker_edge_mask``; may vary across the stack)."""
        return self.masks.tucker_edge_mask.sum(axis=-1)

    @cached_property
    def tt_ranks(self) -> NDArray:  # dtype=int, shape=(d+1,)+stack_shape
        """Real TT ranks (from ``tt_edge_mask``; may vary across the stack)."""
        return self.masks.tt_edge_mask.sum(axis=-1)

    @cached_property
    def structure(self) -> Tuple[Tuple[int, ...], NDArray, NDArray, Tuple[int, ...]]:
        """``(shape, tucker_ranks, tt_ranks, stack_shape)`` -- the real structure."""
        return self.shape, self.tucker_ranks, self.tt_ranks, self.stack_shape

    # ----------------------------------------------------------------- validation
    def validate(self):
        """Check the structural invariants (shapes mutually consistent, rank masks boolean, ``shape`` a
        length-``d`` tuple of mode dims within the padded ``N``). Raises ValueError."""
        tkm, ttm = self.masks.data
        for m, name in ((tkm, 'tucker_edge_mask'), (ttm, 'tt_edge_mask')):
            if not common.is_boolean_ndarray(m):
                raise ValueError(
                    'UniformTuckerTensorTrain: %s must be a boolean array (got %s).'
                    % (name, getattr(m, 'dtype', type(m))))

        d, stack, n, N, r = self.d, self.stack_shape, self.n, self.N, self.r
        if len(self.shape) != d:
            raise ValueError(
                'UniformTuckerTensorTrain: shape=%s has length %d, expected d=%d.'
                % (self.shape, len(self.shape), d))
        if any(Ni < 0 or Ni > N for Ni in self.shape):
            raise ValueError(
                'UniformTuckerTensorTrain: every entry of shape=%s must be in [0, padded N=%d].'
                % (self.shape, N))

        expected = {
            'tt_supercore':     (d,) + stack + (r, n, r),
            'tucker_edge_mask': (d,) + stack + (n,),
            'tt_edge_mask':     (d + 1,) + stack + (r,),
        }
        actual = {
            'tt_supercore':     tuple(self.tt_supercore.shape),
            'tucker_edge_mask': tuple(tkm.shape),
            'tt_edge_mask':     tuple(ttm.shape),
        }
        for k in expected:
            if actual[k] != expected[k]:
                raise ValueError(
                    'Inconsistent UniformTuckerTensorTrain: %s.shape = %s, expected %s '
                    '(d=%d, stack_shape=%s, n=%d, N=%d, r=%d).'
                    % (k, actual[k], expected[k], d, stack, n, N, r))

    def __post_init__(self):
        self.validate()

    def __repr__(self) -> str:
        ss = ', stack_shape=%s' % (self.stack_shape,) if self.stack_shape else ''
        return ('UniformTuckerTensorTrain(shape=%s, N=%d, n=%d, r=%d%s)'
                % (self.shape, self.N, self.n, self.r, ss))

    # ----------------------------------------------------------------- operations
    def apply_masks(self) -> 'UniformTuckerTensorTrain':
        """Zero the padded ("garbage") regions of the supercores (the masks are unchanged)."""
        mtk, mtt = ut3_masking.ut3_apply_masks(self.data)
        return UniformTuckerTensorTrain(mtk, mtt, self.shape, self.masks)

    def to_dense(self) -> NDArray:
        """Form the dense tensor, ``shape = stack_shape + (N0,...,N(d-1))``. (Inspection/tests only.)"""
        return ut3_conversions.ut3_to_dense(self.data)

    # ----------------------------------------------------------------- ragged <-> uniform conversions
    @staticmethod
    def from_t3(
            x:  t3.TuckerTensorTrain,
            N:  Optional[int] = None,   # padded mode dim   (default max(Ni)); pass to force a larger pad
            n:  Optional[int] = None,   # padded Tucker rank (default max(tucker_ranks))
            r:  Optional[int] = None,   # padded TT rank    (default max(tt_ranks))
            squash_tails: bool = True,
    ) -> 'UniformTuckerTensorTrain':
        """Pack a ragged :py:class:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain` into a uniform one."""
        return _from_data(ut3_conversions.t3_to_ut3(x.data, N=N, n=n, r=r, squash_tails=squash_tails))

    def to_t3(self):  # -> TuckerTensorTrain (unstacked) or a nested tree (shaped like stack_shape) of them
        """Convert back to ragged form.

        Unstacked: one :py:class:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain`. Stacked: a nested tree
        of them (a varying-rank stack has no single stacked ``TuckerTensorTrain``;
        ``docs/uniform_ranks_and_varieties.md``).
        """
        def _wrap(res):
            if common.is_ndarray(res[0][0]):   # res = (tucker_cores, tt_cores) leaf
                return t3.TuckerTensorTrain(*res)
            return tuple(_wrap(r) for r in res)

        return _wrap(ut3_conversions.ut3_to_t3(self.data))

    def reverse(self) -> 'UniformTuckerTensorTrain':
        """Reverse the mode order."""
        return _from_data(ut3_operations.ut3_reverse(self.data))

    def squash_tails(self) -> 'UniformTuckerTensorTrain':
        """Sum the leading/trailing TT bonds down to rank 1 (preserves the represented tensor)."""
        return _from_data(ut3_operations.ut3_squash_tails(self.data))

    # ----------------------------------------------------------------- linear algebra
    def __mul__(self, s) -> 'UniformTuckerTensorTrain':
        """Scale by a scalar."""
        return _from_data(ut3_linalg.ut3_scale(self.data, s))

    __rmul__ = __mul__

    def __neg__(self) -> 'UniformTuckerTensorTrain':
        return self * (-1.0)

    def __add__(self, other: 'UniformTuckerTensorTrain') -> 'UniformTuckerTensorTrain':
        """Add two uniform Tucker tensor trains (direct sum, then squash). Requires matching shape / d /
        stack_shape (structural ValueError); padded ``n``/``r`` need not match."""
        if self.shape != other.shape:
            raise ValueError('Cannot add UniformTuckerTensorTrains with different shapes: %s vs %s.'
                             % (self.shape, other.shape))
        if self.stack_shape != other.stack_shape:
            raise ValueError('Cannot add UniformTuckerTensorTrains with different stack_shapes: %s vs %s.'
                             % (self.stack_shape, other.stack_shape))
        if self.N != other.N:
            raise ValueError('Cannot add UniformTuckerTensorTrains with different padded mode widths N: %d vs %d '
                             '(the same shape padded differently -- re-pad one operand with from_t3(..., N=%d)).'
                             % (self.N, other.N, max(self.N, other.N)))
        return _from_data(ut3_operations.ut3_squash_tails(ut3_linalg.ut3_add(self.data, other.data)))

    def __sub__(self, other: 'UniformTuckerTensorTrain') -> 'UniformTuckerTensorTrain':
        return self + (-other)

    def sum_stack(self) -> 'UniformTuckerTensorTrain':
        """Sum the represented tensors over the entire stack -> one unstacked uniform T3 (genuine tensor
        sum, not corewise)."""
        return _from_data(ut3_operations.ut3_squash_tails(ut3_linalg.ut3_sum_stack(self.data)))

    def inner(self, other: 'UniformTuckerTensorTrain', use_orthogonalization: bool = True):
        """Hilbert-Schmidt inner product with another uniform Tucker tensor train (shape=stack_shape)."""
        if self.shape != other.shape:
            raise ValueError('Cannot inner-product UniformTuckerTensorTrains with different shapes: %s vs %s.'
                             % (self.shape, other.shape))
        if self.N != other.N:
            raise ValueError('Cannot inner-product UniformTuckerTensorTrains with different padded mode widths N: '
                             '%d vs %d (the same shape padded differently -- re-pad one operand with '
                             'from_t3(..., N=%d)).' % (self.N, other.N, max(self.N, other.N)))
        xd, yd = self.data, other.data
        if use_orthogonalization:
            xd = ut3_orthogonalization.ut3_left_orthogonalize_tt_cores(
                ut3_orthogonalization.ut3_down_orthogonalize_tucker_cores(xd))
            yd = ut3_orthogonalization.ut3_left_orthogonalize_tt_cores(
                ut3_orthogonalization.ut3_down_orthogonalize_tucker_cores(yd))
        return ut3_linalg.ut3_inner_product(xd, yd)

    def norm(self, use_orthogonalization: bool = True):
        """Hilbert-Schmidt (Frobenius) norm of the represented tensor (shape=stack_shape)."""
        if use_orthogonalization:
            xd = ut3_orthogonalization.ut3_left_orthogonalize_tt_cores(
                ut3_orthogonalization.ut3_down_orthogonalize_tucker_cores(self.data))
            return ut3_linalg.ut3_norm_orthogonalized(xd)
        xnp, _, _ = common.get_backend(True, self.contains_jax)
        return xnp.sqrt(xnp.abs(ut3_linalg.ut3_inner_product(self.data, self.data)))

    # ----------------------------------------------------------------- sampling / evaluation
    def entries(
            self,
            index:  NDArray,   # int, shape=(d,)+idx_stack (one multi-index per idx_stack element)
    ) -> NDArray:              # shape=idx_stack+stack_shape
        """Entry/entries of the represented dense tensor, evaluated without forming it (shares
        :py:func:`~t3toolbox.backend.ut3_sampling.ut3_entries`).

        Precondition-free (exact for any cores). Uniform mirror of
        :py:meth:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain.entries`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> print(bool(abs(float(x.entries((3, 5, 7))) - float(x.to_dense()[3, 5, 7])) < 1e-9))
        True
        """
        return ut3_sampling.ut3_entries(self.data, index)

    def apply(
            self,
            vecs:  Sequence[NDArray],  # len=d, ith elm_shape=vec_stack+(Ni,)
    ) -> NDArray:                      # shape=vec_stack+stack_shape (a scalar per stack element)
        """Contract the represented tensor with vectors in **all** modes, without forming it (shares
        :py:func:`~t3toolbox.backend.ut3_sampling.ut3_apply`; a scalar per stack element).

        Precondition-free (exact for any cores). Uniform mirror of
        :py:meth:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain.apply`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> print(bool(np.allclose(x.apply(ww), np.einsum('ijk,i,j,k->', x.to_dense(), *ww))))
        True
        """
        return ut3_sampling.ut3_apply(self.data, vecs)

    def probe(
            self,
            ww:  Sequence[NDArray],  # len=d, ith elm_shape=W+(Ni,)
    ) -> Sequence[NDArray]:          # len=d, ith elm_shape=W+stack_shape+(Ni,)
        """Probe: contract all-but-one mode, for each mode (leaving mode ``i`` free), without forming the
        dense tensor (shares :py:func:`~t3toolbox.backend.ut3_sampling.ut3_probe`).

        The probe stack ``W`` (on ``ww``) is base-inner with the T3 stack: each probe is ``W + stack_shape +
        (Ni,)``. Precondition-free. Uniform mirror of
        :py:meth:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain.probe`.

        See Also
        --------
        apply
        entries
        probe_derivatives

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> zz = x.probe(ww)
        >>> print([z.shape for z in zz])           # one free mode each: (Ni,)
        [(10,), (11,), (12,)]
        >>> print(bool(np.allclose(zz[0], np.einsum('ijk,j,k->i', x.to_dense(), ww[1], ww[2]))))
        True
        """
        return ut3_sampling.ut3_probe(ww, self.data)

    # --------------------------------------------------------------- derivative sampling (jets; 3b-6'b)
    def probe_derivatives(
            self,
            ww:     Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                # highest derivative order
    ) -> Sequence[NDArray]:             # len=d, ith elm_shape=(order+1,)+W+stack_shape+(Ni,)
        """Symmetric directional derivatives of :py:meth:`probe`, in one repeated direction ``P`` (``pp``):
        ``y_i^(t) = d^t/ds^t [probe(X + s P)]_i|_0`` for ``t=0..order`` (order axis outermost; index 0 is the
        ordinary :py:meth:`probe`).

        No *numerical* precondition (exact for any cores). **Structural precondition** (hard error, both
        modes): ``P`` (``pp``) must share the sample stack ``W`` and mode dims of ``X`` (``ww``). Uniform
        mirror of :py:meth:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain.probe_derivatives`.

        See Also
        --------
        probe
        apply_derivatives
        probe_corewise_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> zj = x.probe_derivatives(ww, pp, 3)
        >>> print([z.shape for z in zj])                                    # (order+1,) + (Ni,)
        [(4, 10), (4, 11), (4, 12)]
        >>> print([bool(np.allclose(z[0], z0)) for z, z0 in zip(zj, x.probe(ww))])   # order 0 == probe
        [True, True, True]

        The perturbation ``P`` must match ``X``'s sample stack and mode dims (structural, raises):

        >>> x.probe_derivatives(ww, (np.random.randn(10), np.random.randn(11), np.random.randn(99)), 3)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError
        """
        sampling_derivatives.check_perturbation_vectors(ww, pp)
        return ut3_sampling.ut3_probe_derivatives(ww, pp, self.data, order)

    def apply_derivatives(
            self,
            ww:     Sequence[NDArray],  # apply vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                # highest derivative order
    ) -> NDArray:                       # shape=(order+1,)+W+stack_shape (a scalar jet per stack element)
        """Symmetric all-modes apply derivatives (the derivative twin of :py:meth:`apply`), one repeated
        direction ``P``: ``y^(t) = d^t/ds^t apply(X + s P)|_0`` for ``t=0..order`` (order 0 is :py:meth:`apply`).

        Structural precondition as in :py:meth:`probe_derivatives` (``P`` shares ``X``'s ``W``).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> yj = x.apply_derivatives(ww, pp, 3)
        >>> print(yj.shape, bool(np.allclose(yj[0], x.apply(ww))))          # (order+1,); order 0 == apply
        (4,) True
        """
        sampling_derivatives.check_perturbation_vectors(ww, pp)
        return ut3_sampling.ut3_apply_derivatives(ww, pp, self.data, order)

    def entries_derivatives(
            self,
            index:  NDArray,            # int, shape=(d,)+W -- the grid points
            pp:     Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                # highest derivative order
    ) -> NDArray:                       # shape=(order+1,)+W+stack_shape
        """Symmetric entry derivatives at ``index`` in direction ``P`` (the derivative twin of
        :py:meth:`entries`): the Taylor data of the multilinear extension at grid corner ``index``
        (``y^(t) = d^t/ds^t apply(e_index + s P)|_0``; order 0 is :py:meth:`entries`).

        Structural precondition: ``P`` shares ``index``'s sample stack ``W`` (and, when checkable, the mode dims).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> yj = x.entries_derivatives(np.array([3, 5, 7]), pp, 3)
        >>> print(yj.shape, bool(np.allclose(yj[0], x.entries((3, 5, 7)))))  # (order+1,); order 0 == entries
        (4,) True
        """
        sampling_derivatives.check_perturbation_index(index, pp, self.shape)
        return ut3_sampling.ut3_entries_derivatives(index, pp, self.data, order)

    # --------------------------------------------------------- corewise (non-manifold) sampling transposes (3b-6c)
    def apply_corewise_transpose(
            self,
            c:    NDArray,            # residual, shape=W+stack_shape (a scalar per stack element)
            ww:   Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)
            sum_over_probes: bool = False,
    ) -> Tuple[NDArray, NDArray]:     # (tucker_grad, tt_grad) supercores
        """Corewise (non-manifold) transpose of :py:meth:`apply`: gradient of ``apply(X(cores), ww)`` w.r.t.
        the supercores (treated as free variables) -- for a core-wise optimizer (Adam, L-BFGS).

        Returns the raw gradient supercores ``(tucker_grad, tt_grad)`` (a gradient, NOT a tensor). The
        Section 6.3 ``(P,Q,O)->G`` substitution; no orthogonality required. ``sum_over_probes=True`` sums the
        apply stack ``W`` (the Gauss-Newton ``Jᵀr``), so the grads are shaped exactly like the supercores;
        ``False`` keeps ``W`` as a leading grad stack. Uniform mirror of
        :py:meth:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain.apply_corewise_transpose`.

        See Also
        --------
        apply
        probe_corewise_transpose
        apply_corewise_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> ww = [np.random.randn(2, N) for N in (10, 11, 12)]     # apply stack W=(2,)
        >>> c = np.random.randn(2)                                 # residual, shape=W
        >>> gU, gG = x.apply_corewise_transpose(c, ww, sum_over_probes=True)
        >>> print(gU.shape == x.tucker_supercore.shape, gG.shape == x.tt_supercore.shape)   # sum W -> core shapes
        True True
        >>> kU, kG = x.apply_corewise_transpose(c, ww, sum_over_probes=False)
        >>> print(kU.shape, kG.shape)                              # keep W: a leading (2,) grad stack
        (3, 2, 6, 12) (3, 2, 3, 6, 3)
        """
        return ut3_sampling.ut3_apply_corewise_transpose(c, ww, self.data, sum_over_probes=sum_over_probes)

    def entries_corewise_transpose(
            self,
            c:     NDArray,            # residual, shape=W+stack_shape
            index: NDArray,            # int, shape=(d,)+W
            sum_over_probes: bool = False,
    ) -> Tuple[NDArray, NDArray]:      # (tucker_grad, tt_grad) supercores
        """Corewise transpose of :py:meth:`entries`: gradient w.r.t. the supercores (= the one-hot
        :py:meth:`apply_corewise_transpose`, scattering ``c`` at ``index``). See
        :py:meth:`apply_corewise_transpose` for the return contract and ``sum_over_probes``."""
        return ut3_sampling.ut3_entries_corewise_transpose(c, index, self.data, sum_over_probes=sum_over_probes)

    def probe_corewise_transpose(
            self,
            ztildes: Sequence[NDArray],  # probe residuals, len=d, elm_shape=W+stack_shape+(Ni,)
            ww:      Sequence[NDArray],  # probe vectors,   len=d, elm_shape=W+(Ni,)
            sum_over_probes: bool = False,
    ) -> Tuple[NDArray, NDArray]:        # (tucker_grad, tt_grad) supercores
        """Corewise transpose of :py:meth:`probe`: gradient w.r.t. the supercores. Like
        :py:meth:`apply_corewise_transpose` but the residual carries one free mode per probe. See
        :py:meth:`apply_corewise_transpose` for the return contract and ``sum_over_probes``."""
        return ut3_sampling.ut3_probe_corewise_transpose(ztildes, ww, self.data, sum_over_probes=sum_over_probes)

    # ------------------------------------------------- corewise (non-manifold) derivative transposes (3b-6'c)
    def apply_corewise_derivatives_transpose(
            self,
            c:      NDArray,            # residual jet (scalar), shape=(order+1,)+W+stack_shape
            ww:     Sequence[NDArray],  # apply vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                # highest derivative order
            sum_over_probes: bool = False,
    ) -> Tuple[NDArray, NDArray]:       # (tucker_grad, tt_grad) supercores
        """Corewise (non-manifold) transpose of :py:meth:`apply_derivatives`: gradient of the
        apply-derivative jets w.r.t. the supercores (treated as free variables), for a core-wise optimizer.

        Returns the raw gradient supercores ``(tucker_grad, tt_grad)`` (the order axis of ``c`` is summed, so
        the grads have no order axis). The §6.3 ``(P,Q,O)->G`` substitution. ``sum_over_probes`` /
        structural precondition as in :py:meth:`apply_corewise_transpose` / :py:meth:`probe_derivatives`.
        Uniform mirror of
        :py:meth:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain.apply_corewise_derivatives_transpose`.

        See Also
        --------
        apply_derivatives
        apply_corewise_transpose
        probe_corewise_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> ww = [np.random.randn(2, N) for N in (10, 11, 12)]     # apply stack W=(2,)
        >>> pp = [np.random.randn(2, N) for N in (10, 11, 12)]
        >>> c = np.random.randn(3, 2)                              # residual jet, (order+1,)+W = (3, 2)
        >>> gU, gG = x.apply_corewise_derivatives_transpose(c, ww, pp, 2, sum_over_probes=True)
        >>> print(gU.shape == x.tucker_supercore.shape, gG.shape == x.tt_supercore.shape)
        True True
        """
        sampling_derivatives.check_perturbation_vectors(ww, pp)
        return ut3_sampling.ut3_apply_corewise_derivatives_transpose(
            c, ww, pp, self.data, order, sum_over_probes=sum_over_probes)

    def entries_corewise_derivatives_transpose(
            self,
            c:      NDArray,            # residual jet (scalar), shape=(order+1,)+W+stack_shape
            index:  NDArray,            # int, shape=(d,)+W
            pp:     Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                # highest derivative order
            sum_over_probes: bool = False,
    ) -> Tuple[NDArray, NDArray]:       # (tucker_grad, tt_grad) supercores
        """Corewise transpose of :py:meth:`entries_derivatives`: gradient w.r.t. the supercores (= the
        one-hot :py:meth:`apply_corewise_derivatives_transpose`). See
        :py:meth:`apply_corewise_derivatives_transpose`."""
        sampling_derivatives.check_perturbation_index(index, pp, self.shape)
        return ut3_sampling.ut3_entries_corewise_derivatives_transpose(
            c, index, pp, self.data, order, sum_over_probes=sum_over_probes)

    def probe_corewise_derivatives_transpose(
            self,
            ztildes: Sequence[NDArray],  # probe residual jets, len=d, elm_shape=(order+1,)+W+stack_shape+(Ni,)
            ww:      Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
            pp:      Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:   int,                # highest derivative order
            sum_over_probes: bool = False,
    ) -> Tuple[NDArray, NDArray]:        # (tucker_grad, tt_grad) supercores
        """Corewise transpose of :py:meth:`probe_derivatives`: gradient w.r.t. the supercores. See
        :py:meth:`apply_corewise_derivatives_transpose`."""
        sampling_derivatives.check_perturbation_vectors(ww, pp)
        return ut3_sampling.ut3_probe_corewise_derivatives_transpose(
            ztildes, ww, pp, self.data, order, sum_over_probes=sum_over_probes)

    def sum(self, axis=None) -> NDArray:
        """Sum the represented tensor over all physical modes (shape=stack_shape). Partial sums (``axis``
        given) are deferred -- see dev/archive/uniform_port_plan.md."""
        if axis is not None:
            raise NotImplementedError(
                'Partial sum (axis given) is deferred for UniformTuckerTensorTrain; only the full sum '
                '(axis=None) is implemented. See dev/archive/uniform_port_plan.md.')
        return ut3_sampling.ut3_full_sum(self.data)

    # ----------------------------------------------------------------- orthogonalization
    # Thin wrappers over the .data-level backend (ut3_orthogonalization): the Tucker-core ops are
    # batched-SVD rewrites; the TT left/right ops share the polymorphic orthogonalization.py sweep. All
    # re-masking and mask/rank recomputation lives in the backend.

    def down_orthogonalize_tucker_cores(self) -> 'UniformTuckerTensorTrain':
        """Orthogonalize the Tucker cores, pushing the remainder up into the TT cores."""
        return _from_data(ut3_orthogonalization.ut3_down_orthogonalize_tucker_cores(self.data))

    def up_orthogonalize_tt_cores(self) -> 'UniformTuckerTensorTrain':
        """Up-orthogonalize the TT cores, pushing the remainder down into the Tucker cores."""
        return _from_data(ut3_orthogonalization.ut3_up_orthogonalize_tt_cores(self.data))

    def left_orthogonalize_tt_cores(self) -> 'UniformTuckerTensorTrain':
        """Left-orthogonalize the TT cores."""
        return _from_data(ut3_orthogonalization.ut3_left_orthogonalize_tt_cores(self.data))

    def right_orthogonalize_tt_cores(self) -> 'UniformTuckerTensorTrain':
        """Right-orthogonalize the TT cores."""
        return _from_data(ut3_orthogonalization.ut3_right_orthogonalize_tt_cores(self.data))

    def is_left_orthogonal(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = stack_shape (scalar unstacked)
        """True (per stack element) if in left-orthogonal form (Tucker supercores down-orthogonal AND TT
        supercores left-orthogonal). A :py:meth:`t3svd` result is left-orthogonal. Non-enforcing checker;
        per-element bool array (reduce with ``.all()``); see
        :py:func:`~t3toolbox.backend.ut3_orthogonalization.ut3_orthogonality_residual`."""
        return ut3_orthogonalization.ut3_orthogonality_residual(self.data, 'left') <= atol

    def is_right_orthogonal(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = stack_shape (scalar unstacked)
        """True (per stack element) if in right-orthogonal form (Tucker supercores down-orthogonal AND TT
        supercores right-orthogonal). Non-enforcing per-element checker (reduce with ``.all()``); verify
        before ``t3svd(..., assume_orthogonal=True)``."""
        return ut3_orthogonalization.ut3_orthogonality_residual(self.data, 'right') <= atol

    @property
    def minimal_ranks(self) -> Tuple[NDArray, NDArray]:
        """Structural minimal ranks ``(min_tucker_ranks, min_tt_ranks)`` for this UT3's shape/ranks."""
        use_jax = self.contains_jax
        return ranks.compute_minimal_ranks(self.shape, self.tucker_ranks, self.tt_ranks, use_jax=use_jax)

    @property
    def has_minimal_ranks(self) -> NDArray:  # bool array, shape = stack_shape (per element; uniform ranks vary)
        """True (per stack element) if this UT3's ranks are structurally minimal. Per-element (uniform ranks
        vary across the stack), reduced over the mode axes; reduce with ``.all()`` for a single verdict."""
        mn = self.minimal_ranks
        return (np.all(np.asarray(self.tucker_ranks) == np.asarray(mn[0]), axis=0)
                & np.all(np.asarray(self.tt_ranks) == np.asarray(mn[1]), axis=0))

    # ----------------------------------------------------------------- T3-SVD
    def t3svd(self, max_tt_ranks=None, max_tucker_ranks=None, assume_orthogonal=False,
              sharing: typ.Sequence = None):
        """Mask-truncated T3-SVD -- the basic algorithm, matching ragged :py:meth:`TuckerTensorTrain.t3svd`
        on real parts. Always **left-orthogonal**; under truncation **not** necessarily minimal (use
        :py:meth:`rank_adjustment_sweep` to minimize). ``assume_orthogonal=True`` skips the
        orthogonalization, asserting the input is already right-orthogonal (verify with
        :py:meth:`is_right_orthogonal` -- not checked). Uniform truncates by **max rank only** -- unlike
        ragged ``t3svd`` there is no ``rtol``/``atol`` (a tolerance would make the output shape
        data-dependent, which the uniform layer forbids; see ``docs/uniform_ranks_and_varieties.md``).
        Per-stack-element ``max_*_ranks`` arrays are allowed. Returns ``(new UT3, Tucker svals, TT svals)``.

        ``sharing`` (one hashable group label per mode) is the grouped SF-T3 truncation, matching the
        ragged ``t3svd(sharing=)`` on real parts: one shared basis per group (one rank mask at every
        group mode), the group spectrum ``s_g`` reported at every group mode. The factors must already
        be tied within groups (safe mode checks; see :py:meth:`has_shared_tucker_factors`)."""
        if sharing is not None and safety.checks_active(self.data[:2]):
            atol_check = safety.effective_rtol(self.data[:2])
            residual = backend_sharing.ut3_sharing_residual(self.data, sharing)
            safety.require(bool((residual <= atol_check).all()),
                           't3svd(sharing=...) requires the Tucker factors to be tied within each '
                           'sharing group (grouped truncation picks ONE basis per group). Tie them '
                           'first, or run in unsafe mode (safety.unsafe()).')
        new_data, ss_tucker, ss_tt = ut3_svd.ut3svd(
            self.data, max_tucker_ranks=max_tucker_ranks, max_tt_ranks=max_tt_ranks,
            assume_orthogonal=assume_orthogonal, sharing=sharing)
        return _from_data(new_data), ss_tucker, ss_tt

    def has_shared_tucker_factors(
            self,
            sharing:    typ.Sequence,   # len=d; one hashable group label per mode
            rtol:       float = 1e-9,   # relative tolerance on the factor deviation
    ) -> NDArray:  # bool array, shape = stack_shape (scalar/0-d when unstacked)
        """True (per stack element) if the MASKED Tucker factors are tied within every sharing group --
        the uniform twin of :py:meth:`TuckerTensorTrain.has_shared_tucker_factors` (padding is
        don't-care garbage and is ignored). Non-enforcing checker; structural problems (wrong
        ``sharing`` length, unequal mode sizes or Tucker rank masks within a group) raise
        unconditionally. Reduce with ``.all()`` for a single verdict.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 2, 2, 1))
        >>> tk, tt = x.data
        >>> tied = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain((tk[0], tk[0], tk[2]), tt))
        >>> print(bool(tied.has_shared_tucker_factors((0, 0, 1))))
        True
        >>> print(bool(ut3.UniformTuckerTensorTrain.from_t3(x).has_shared_tucker_factors((0, 0, 1))))
        False
        """
        return backend_sharing.ut3_tucker_factors_shared(self.data, sharing, rtol=rtol)

    def rank_adjustment_sweep(self, direction: str = 'right_to_left',
                              sharing: typ.Sequence = None) -> 'UniformTuckerTensorTrain':
        """A single directional sweep that drops structurally-redundant ranks (the separate
        rank-minimization step; :py:meth:`t3svd` does not minimize). ``'right_to_left'`` returns a
        right-orthogonal UT3, ``'left_to_right'`` a left-orthogonal one; it reaches minimal ranks **only
        if the input is orthogonal in the opposite direction** (a :py:meth:`t3svd` result is
        left-orthogonal, so ``'right_to_left'`` minimizes it).

        That precondition is **required**, not just optimal, and is **not enforced**: because the uniform
        layer commits to fixed (minimal) output shapes, sweeping the wrong direction for the input's gauge
        is **lossy** -- it discards real content (unlike the ragged version, which only under-minimizes).
        Verify the gauge with :py:meth:`is_left_orthogonal` / :py:meth:`is_right_orthogonal` first.
        See :py:func:`~t3toolbox.backend.ut3_svd.ut3_rank_adjustment_sweep`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10, 10, 10), (9, 9, 9), (1, 9, 9, 1))
        >>> x2, _, _ = ut3.UniformTuckerTensorTrain.from_t3(x).t3svd(max_tucker_ranks=[9, 1, 9], max_tt_ranks=[1, 9, 2, 1])
        >>> print(x2.is_left_orthogonal(), x2.has_minimal_ranks)   # t3svd output: left-orth, non-minimal
        True False
        >>> good = x2.rank_adjustment_sweep('right_to_left')       # CORRECT (x2 is left-orthogonal)
        >>> print(good.has_minimal_ranks, np.allclose(good.to_dense(), x2.to_dense()))
        True True
        >>> bad = x2.rank_adjustment_sweep('left_to_right')        # WRONG direction -> corrupts the tensor
        >>> print(np.allclose(bad.to_dense(), x2.to_dense()))
        False

        With ``sharing``, the reduction is the SHARED one (the group ceiling; tied factors stay tied --
        the per-mode reduction would clip a group rank the ceiling admits and untie the group). The
        factors must already be tied within groups (safe mode checks).
        """
        if sharing is not None and safety.checks_active(self.data[:2]):
            atol_check = safety.effective_rtol(self.data[:2])
            residual = backend_sharing.ut3_sharing_residual(self.data, sharing)
            safety.require(bool((residual <= atol_check).all()),
                           'rank_adjustment_sweep(sharing=...) requires the Tucker factors to be tied '
                           'within each sharing group. Tie them first, or run in unsafe mode '
                           '(safety.unsafe()).')
        return _from_data(ut3_svd.ut3_rank_adjustment_sweep(self.data, direction, sharing=sharing))

    # ----------------------------------------------------------------- stacking
    def unstack(self):
        """Unstack into an array-like tree (shaped like ``stack_shape``) of unstacked UT3s."""
        return stacking.apply_func_to_leaf_subtrees(
            ut3_operations.ut3_unstack(self.data),
            lambda leaf: UniformTuckerTensorTrain(leaf[0], leaf[1], leaf[2], UT3Masks(*leaf[3])),
            ut3_operations.ut3_leaf_structure(self.d),
        )

    @staticmethod
    def stack(uxx) -> 'UniformTuckerTensorTrain':
        """Stack an array-like tree of UT3s into one stacked UT3."""
        data_tree = stacking.apply_func_to_leaf_subtrees(uxx, lambda u: u.data, None)
        tk, tt, shape, masks = ut3_operations.ut3_stack(data_tree)
        return UniformTuckerTensorTrain(tk, tt, shape, UT3Masks(*masks))

    # ----------------------------------------------------------------- dtype / copy
    @cached_property
    def contains_jax(self) -> bool:
        return common.tree_contains_jax(self.supercores)

    def to_jax(self) -> 'UniformTuckerTensorTrain':
        # Convert the SUPERCORES (data) to jax; the masks stay numpy (host structure -- a jax mask is a
        # tracer under jit and breaks the layer). See docs/contributor/uniform_pytree_composition.md.
        return UniformTuckerTensorTrain(
            common.to_jax(self.tucker_supercore), common.to_jax(self.tt_supercore), self.shape, self.masks)

    def to_numpy(self) -> 'UniformTuckerTensorTrain':
        # Supercores -> numpy; the masks are already numpy (host structure), so reuse the holder.
        return UniformTuckerTensorTrain(
            common.to_numpy(self.tucker_supercore), common.to_numpy(self.tt_supercore), self.shape, self.masks)

    def copy(self) -> 'UniformTuckerTensorTrain':
        # Deep-copy the supercores (the data leaves), like ragged T3*.copy; the static aux (shape + masks)
        # is shared (immutable structure, the same way `shape` is not duplicated).
        return UniformTuckerTensorTrain(
            self.tucker_supercore.copy(), self.tt_supercore.copy(), self.shape, self.masks)

    # ----------------------------------------------------------------- constructors
    # Pure constructors keep a `use_jax` flag for the SUPERCORES (no array input to infer from); the
    # masks are always numpy (host) structure (docs/contributor/uniform_pytree_composition.md). The ranks may vary
    # per stack element (the variety) -- a backend feature a ragged round-trip cannot express.

    @staticmethod
    def zeros(
            shape:        Sequence[int],                          # (N0,...,N(d-1))
            tucker_ranks: typ.Union[int, Sequence[int], NDArray, None] = None,  # int|len-d|(d,)+stack; None->1
            tt_ranks:     typ.Union[int, Sequence[int], NDArray, None] = None,  # int|len-(d+1)|(d+1,)+stack; None->1
            stack_shape:  Sequence[int] = (),
            use_jax:      bool = False,
    ) -> 'UniformTuckerTensorTrain':
        """Uniform Tucker tensor train of zeros (padded regions masked to zero).

        ``tucker_ranks``/``tt_ranks`` accept a scalar, a per-mode sequence, or a full ``(d,)+stack`` /
        ``(d+1,)+stack`` array (the variety: ranks varying per stack element). ``None`` -> all ranks 1.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> z = ut3.UniformTuckerTensorTrain.zeros((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        >>> print(z.shape, z.stack_shape)
        (5, 6, 7) (2,)
        >>> print(float(np.linalg.norm(z.to_dense())))
        0.0
        """
        return _from_data(ut3_constructors.ut3_zeros(
            shape, tucker_ranks, tt_ranks, tuple(stack_shape), use_jax=use_jax))

    @staticmethod
    def ones(
            shape:       Sequence[int],          # (N0,...,N(d-1))
            stack_shape: Sequence[int] = (),
            use_jax:     bool = False,
    ) -> 'UniformTuckerTensorTrain':
        """Rank-1 uniform Tucker tensor train representing a tensor full of ones (every real entry == 1).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = ut3.UniformTuckerTensorTrain.ones((5, 6, 7), stack_shape=(2,))
        >>> print(float(np.linalg.norm(x.to_dense() - np.ones((2, 5, 6, 7)))))
        0.0
        >>> print(np.asarray(x.tucker_ranks).max(), np.asarray(x.tt_ranks).max())
        1 1
        """
        return _from_data(ut3_constructors.ut3_ones(shape, tuple(stack_shape), use_jax=use_jax))

    @staticmethod
    def randn(
            shape:        Sequence[int],                          # (N0,...,N(d-1))
            tucker_ranks: typ.Union[int, Sequence[int], NDArray], # int|len-d|(d,)+stack
            tt_ranks:     typ.Union[int, Sequence[int], NDArray], # int|len-(d+1)|(d+1,)+stack
            stack_shape:  Sequence[int] = (),
            use_jax:      bool = False,
    ) -> 'UniformTuckerTensorTrain':
        """Uniform Tucker tensor train with random N(0,1) supercores (padded regions masked to zero).

        ``tucker_ranks``/``tt_ranks`` accept a scalar, a per-mode sequence, or a full ``(d,)+stack`` /
        ``(d+1,)+stack`` array -- the latter setting **per-stack-element ranks** (the variety) while
        keeping one padded supercore shape, which a ragged round-trip cannot express.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        >>> print(x.shape, x.stack_shape)
        (5, 6, 7) (2,)
        >>> print(np.reshape(x.tucker_ranks, (3, 2))[:, 0].tolist())   # uniform across the stack here
        [3, 4, 2]
        >>> print(bool(np.any(x.tucker_supercore != 0.0)))            # random, not zeros
        True

        Per-stack-element ranks (the variety): a full ``(d,)+stack`` array gives each stack element its
        own ranks under one padded shape.

        >>> tucker_ranks = np.array([[2, 4], [3, 5], [2, 3]])         # (d=3, stack=2)
        >>> tt_ranks = np.array([[1, 1], [2, 4], [2, 3], [1, 1]])     # (d+1=4, stack=2)
        >>> xv = ut3.UniformTuckerTensorTrain.randn((6, 7, 8), tucker_ranks, tt_ranks, stack_shape=(2,))
        >>> print(np.asarray(xv.tucker_ranks).tolist())              # ranks genuinely differ per element
        [[2, 4], [3, 5], [2, 3]]
        >>> print(xv.n, xv.r)                                        # one padded shape: n=max, r=max
        5 4
        """
        return _from_data(ut3_constructors.ut3_corewise_randn(
            shape, tucker_ranks, tt_ranks, tuple(stack_shape), use_jax=use_jax))

    # Note: there are deliberately NO ``from_canonical`` / ``from_tensor_train`` / ``to_tensor_train``
    # methods. They would take *ragged* CP/TT data and round-trip through ``TuckerTensorTrain``, which is
    # ambiguous (ragged vs uniform input). Be explicit instead: build a ``TuckerTensorTrain`` (which has
    # those methods) and convert with :py:meth:`from_t3` / :py:meth:`to_t3`.

    # ----------------------------------------------------------------- save / load
    def save(
            self,
            file,  # path or open file object to write the .npz to
    ) -> None:
        """Save to a ``.npz`` file (2 supercores + 2 rank masks + the shape ints). See :py:meth:`load`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        >>> fname = 'ut3_file.npz'
        >>> x.save(fname)
        >>> x2 = ut3.UniformTuckerTensorTrain.load(fname)
        >>> print(float(np.linalg.norm(x2.to_dense() - x.to_dense())))
        0.0
        >>> x2.shape                                       # the static shape survives the round-trip
        (5, 6, 7)
        >>> print([str(m.dtype) for m in x2.masks.data])   # rank masks come back numpy (host) bool
        ['bool', 'bool']
        """
        ut3_constructors.ut3_save(file, self.data)

    @staticmethod
    def load(
            file,                   # path or open file object to read the .npz from
            use_jax: bool = False,  # supercore type; the masks always come back numpy (host) bool
    ) -> 'UniformTuckerTensorTrain':
        """Load from a ``.npz`` file written by :py:meth:`save`.

        The supercores follow ``use_jax``; the masks stay **numpy (host) bool** (a jax mask is a tracer
        under jit and breaks the layer; ``docs/contributor/uniform_pytree_composition.md``). See :py:meth:`save` for
        an example.
        """
        return _from_data(ut3_constructors.ut3_load(file, use_jax=use_jax))


def _from_data(
        data: typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]],
) -> 'UniformTuckerTensorTrain':
    """Wrap a backend ``.data`` tuple ``(tucker_supercore, tt_supercore, shape, (2 rank masks))`` into a
    ``UniformTuckerTensorTrain``. Every frontend operation is a thin wrapper: call the matching
    ``ut3_*`` backend function on ``self.data``, then re-wrap with this. The masks holder is the only
    frontend-side construction (the OO-frame exception to the backend/frontend razor)."""
    tk, tt, shape, masks = data
    return UniformTuckerTensorTrain(tk, tt, shape, UT3Masks(*masks))


# (ragged <-> uniform conversions are methods on UniformTuckerTensorTrain: `.from_t3` / `.to_t3`.)


###########################################


@dataclass(frozen=True)
class UT3Weights:
    """Diagonal weights on the internal edges of a :py:class:`UniformTuckerTensorTrain` -- the uniform
    twin of :py:class:`~t3toolbox.tucker_tensor_train.T3Weights`.

    One vector per internal edge, packed into two supercores + a :py:class:`UT3Masks` holder:

    - ``tucker_weight_supercore``: ``(d,) + stack_shape + (n,)`` -- the Tucker-rank edges
    - ``tt_weight_supercore``: ``(d+1,) + stack_shape + (r,)`` -- the TT-bond edges
    - ``masks``: **the same two edge masks as the train it weights** (a weight's edges *are* the tensor's
      edges, so it declares the same ranks). This is a genuine precondition, not bookkeeping -- see
      :py:meth:`is_consistent_with`.

    There is deliberately **no ``shape`` field** (unlike ``UniformTuckerTensorTrain``): weights live only
    on the *internal* edges, so a weight has no physical mode legs at all.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> np.random.seed(0)
    >>> x  = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))
    >>> ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=4, r=3)   # padded ABOVE the real ranks
    >>> W  = ut3.UT3Weights.from_t3weights(t3.T3Weights.from_t3svd(x), n=ux.n, r=ux.r)
    >>> print(W.is_consistent_with(ux))     # same edges -> same masks
    True
    >>> print(W.tucker_weight_supercore.shape, np.asarray(W.tucker_ranks).tolist())  # padded to 4, real 2
    (3, 4) [2, 2, 2]
    >>> xw = ut3.ut3_absorb_weights(ux, W)      # shape-preserving; masks unchanged
    >>> print(xw.tucker_supercore.shape == ux.tucker_supercore.shape, xw.masks == ux.masks)
    True True

    The padding is a canonical zero, so ``reciprocal`` cannot naively divide (``1/0 = inf`` would poison
    every masked reduction downstream: masking multiplies, and ``0 * inf = nan``). It guards the padding
    instead -- which matters because the Grasedyck-Kramer metric *is* a reciprocal of singular values:

    >>> with np.errstate(divide='ignore'):                             # the naive divide DOES blow up
    ...     naive = 1.0 / W.tucker_weight_supercore
    >>> print(bool(np.isinf(naive).any()))
    True
    >>> Wr = W.reciprocal()                                            # ...the guarded one does not
    >>> print(bool(np.isfinite(Wr.tucker_weight_supercore).all()))
    True
    >>> print(bool(np.isfinite(ut3.ut3_weighted_norm(ux, Wr))))            # so the GK norm stays finite
    True
    """
    tucker_weight_supercore: NDArray   # shape=(d,)  +stack_shape+(n,)
    tt_weight_supercore:     NDArray   # shape=(d+1,)+stack_shape+(r,)
    masks:                   UT3Masks  # static rank structure: the SAME edge masks as the weighted train

    # ----------------------------------------------------------------- views
    @cached_property
    def supercores(self) -> Tuple[NDArray, NDArray]:
        """``(tucker_weight_supercore, tt_weight_supercore)``."""
        return self.tucker_weight_supercore, self.tt_weight_supercore

    @cached_property
    def data(self) -> Tuple[NDArray, NDArray, Tuple[NDArray, NDArray]]:
        """Raw-array view, mirroring the fields: ``(tucker_weight_supercore, tt_weight_supercore,
        (2 rank masks))``. Backend ``ut3_*_weights`` functions take this layout. Note it is a **3-tuple**,
        one shorter than ``UniformTuckerTensorTrain.data`` -- there is no ``shape`` slot."""
        return self.tucker_weight_supercore, self.tt_weight_supercore, self.masks.data

    # ------------------------------------------------- padded (uniform) structure
    @cached_property
    def d(self) -> int:
        """Number of modes."""
        return self.tucker_weight_supercore.shape[0]

    @cached_property
    def n(self) -> int:
        """Padded Tucker rank."""
        return self.tucker_weight_supercore.shape[-1]

    @cached_property
    def r(self) -> int:
        """Padded TT rank."""
        return self.tt_weight_supercore.shape[-1]

    @cached_property
    def stack_shape(self) -> Tuple[int, ...]:
        """Stack shape (``()`` if unstacked). Lives at axes ``1 .. len(stack_shape)`` (``d`` is axis 0)."""
        return self.tucker_weight_supercore.shape[1:-1]

    # ------------------------------------------------- original (real) structure
    @cached_property
    def tucker_ranks(self) -> NDArray:  # dtype=int, shape=(d,)+stack_shape
        """Real Tucker ranks (from ``tucker_edge_mask``; may vary across the stack)."""
        return self.masks.tucker_edge_mask.sum(axis=-1)

    @cached_property
    def tt_ranks(self) -> NDArray:  # dtype=int, shape=(d+1,)+stack_shape
        """Real TT ranks (from ``tt_edge_mask``; may vary across the stack)."""
        return self.masks.tt_edge_mask.sum(axis=-1)

    # ----------------------------------------------------------------- validation
    def validate(self):
        """Check the structural invariants (supercore shapes agree with the masks; masks boolean)."""
        tkm, ttm = self.masks.data
        for m, name in ((tkm, 'tucker_edge_mask'), (ttm, 'tt_edge_mask')):
            if not common.is_boolean_ndarray(m):
                raise ValueError('UT3Weights: %s must be a boolean array (got %s).'
                                 % (name, getattr(m, 'dtype', type(m))))

        d, stack, n, r = self.d, self.stack_shape, self.n, self.r
        expected = {
            'tucker_edge_mask': (d,) + stack + (n,),
            'tt_edge_mask':     (d + 1,) + stack + (r,),
            'tt_weight_supercore': (d + 1,) + stack + (r,),
        }
        actual = {
            'tucker_edge_mask': tuple(tkm.shape),
            'tt_edge_mask':     tuple(ttm.shape),
            'tt_weight_supercore': tuple(self.tt_weight_supercore.shape),
        }
        for k in expected:
            if actual[k] != expected[k]:
                raise ValueError(
                    'Inconsistent UT3Weights: %s.shape = %s, expected %s (d=%d, stack_shape=%s, n=%d, r=%d).'
                    % (k, actual[k], expected[k], d, stack, n, r))

    def __post_init__(self):
        self.validate()

    def __repr__(self) -> str:
        ss = ', stack_shape=%s' % (self.stack_shape,) if self.stack_shape else ''
        return 'UT3Weights(d=%d, n=%d, r=%d%s)' % (self.d, self.n, self.r, ss)

    # ----------------------------------------------------------------- operations
    def is_consistent_with(self, x: 'UniformTuckerTensorTrain') -> bool:
        """True iff these weights can be absorbed into ``x`` (non-raising).

        Requires that the padded shapes fit **and that the edge masks are equal**.

        The mask equality is the real content, and it is a check ragged does not need: ragged catches a
        rank mismatch as an einsum shape error (a length-``n`` weight vector against a rank-``n`` core),
        but uniform pads both to the common ``(n, r)``, so a mismatch is invisible to the shapes and would
        silently corrupt -- a weight whose mask calls slot ``i`` padding carries a canonical zero there,
        and absorbing it would **zero a real slot** of ``x``. The same precondition uniform adds to
        variation add/sub (``docs/uniform_masks_vs_ranks.md``).
        """
        return ut3_operations.ut3_weights_consistent(x.data, self.data)

    def has_shared_tucker_weights(
            self,
            sharing:    typ.Sequence,   # len=d; one hashable group label per mode
            rtol:       float = 1e-9,   # relative tolerance on the Tucker-weight deviation
    ) -> NDArray:  # bool array, shape = stack_shape (scalar/0-d when unstacked)
        """True (per stack element) if the MASKED Tucker weights are equal within every sharing
        group -- the uniform twin of
        :py:meth:`~t3toolbox.tucker_tensor_train.T3Weights.has_shared_tucker_weights` (padding is
        don't-care; unequal group rank masks raise). Non-enforcing.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 3, 2, 1))
        >>> tk, tt = x.data
        >>> uxs = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain((tk[0], tk[0], tk[2]), tt))
        >>> W = ut3.UT3Weights.from_ut3svd(uxs, sharing=(0, 0, 1))
        >>> print(bool(W.has_shared_tucker_weights((0, 0, 1))))
        True
        >>> print(bool(ut3.UT3Weights.from_ut3svd(uxs).has_shared_tucker_weights((0, 0, 1))))
        False
        """
        return backend_sharing.ut3_tucker_weights_shared(self.data, sharing, rtol=rtol)

    def reciprocal(self) -> 'UT3Weights':
        """Elementwise ``1/w`` on the real slots (e.g. to form inverse-singular-value weights); the
        padding stays a canonical, **finite** zero rather than becoming ``inf``. Masks unchanged. See
        :py:func:`~t3toolbox.backend.ut3_operations.ut3_reciprocal_weights` for why that guard is
        load-bearing (and why real-slot zeros are deliberately *not* guarded)."""
        return _weights_from_data(ut3_operations.ut3_reciprocal_weights(self.data))

    def sqrt(self) -> 'UT3Weights':
        """Elementwise ``sqrt`` on the real slots; the padding stays a canonical, finite zero (masks
        unchanged)."""
        return _weights_from_data(ut3_operations.ut3_sqrt_weights(self.data))

    def concatenate(self, other: 'UT3Weights') -> 'UT3Weights':
        """Per-edge concatenation with ``other`` -- the ``+`` / direct-sum combine (ranks add, padded
        widths add). The output mask **may go gappy** if either input has rank slack, which is expected
        and correct (``docs/uniform_masks_vs_ranks.md``)."""
        return _weights_from_data(ut3_operations.ut3_concatenate_weights(self.data, other.data))

    def kronecker(self, other: 'UT3Weights') -> 'UT3Weights':
        """Per-edge Kronecker product with ``other`` -- the Hadamard (elementwise-product) combine (ranks
        multiply, padded widths multiply). The output mask is **strided/gappy** (the real set is
        ``{a*nB + b}`` over the *padded* width), which is correct; see
        :py:func:`~t3toolbox.backend.ut3_operations.ut3_kronecker_weights`."""
        return _weights_from_data(ut3_operations.ut3_kronecker_weights(self.data, other.data))

    # ----------------------------------------------------------------- constructors
    @classmethod
    def from_ut3svd(cls, x: 'UniformTuckerTensorTrain', **kwargs) -> 'UT3Weights':
        """The singular values of ``x`` as a weight object -- the canonical (unmodified) sigmas, so
        ``from_ut3svd(x).reciprocal()`` is the inverse-sigma (Grasedyck-Kramer) weighting. Uniform twin of
        :py:meth:`~t3toolbox.tucker_tensor_train.T3Weights.from_t3svd`.

        ``**kwargs`` pass to :py:meth:`UniformTuckerTensorTrain.t3svd`. The weights carry the **t3svd
        result's** masks, so they pair with that result -- which is ``x`` itself only when ``x`` already
        has minimal ranks and tight padding. Otherwise weight the returned train, not the original:

            ``xs, _, _ = x.t3svd(); W = UT3Weights.from_ut3svd(x); absorb_weights(xs, W)``
        """
        new_x, tucker_svals, tt_svals = x.t3svd(**kwargs)
        return cls(tucker_svals, tt_svals, new_x.masks)

    # ----------------------------------------------------------------- ragged <-> uniform conversions
    @staticmethod
    def from_t3weights(
            weights: 't3.T3Weights',
            n: Optional[int] = None,   # padded Tucker rank (default max rank); pass to match the train's pad
            r: Optional[int] = None,   # padded TT rank     (default max rank)
    ) -> 'UT3Weights':
        """Pack a ragged :py:class:`~t3toolbox.tucker_tensor_train.T3Weights` into a uniform one.

        Pass ``n``/``r`` to match the padding of the :py:class:`UniformTuckerTensorTrain` these weights
        will pair with (e.g. ``n=ux.n, r=ux.r``); the defaults pad tightly to the weights' own max rank,
        which only matches a tightly-padded train.
        """
        return _weights_from_data(ut3_conversions.t3weights_to_ut3weights(weights.data, n=n, r=r))

    def to_t3weights(self):  # -> T3Weights (unstacked) or a nested tree (shaped like stack_shape) of them
        """Convert back to ragged form.

        Unstacked: one :py:class:`~t3toolbox.tucker_tensor_train.T3Weights`. Stacked: a nested tree of
        them (a varying-rank stack has no single stacked ``T3Weights``, exactly as for
        :py:meth:`UniformTuckerTensorTrain.to_t3`; ``docs/uniform_ranks_and_varieties.md``).
        """
        def _wrap(res):
            if common.is_ndarray(res[0][0]):   # res = (tucker_weights, tt_weights) leaf
                return t3.T3Weights(*res)
            return tuple(_wrap(w) for w in res)

        return _wrap(ut3_conversions.ut3weights_to_t3weights(self.data))


def _weights_from_data(
        data: Tuple[NDArray, NDArray, Tuple[NDArray, NDArray]],
) -> 'UT3Weights':
    """Wrap a backend weights ``.data`` tuple into a :py:class:`UT3Weights` (the ``_from_data`` twin)."""
    tucker_weight_supercore, tt_weight_supercore, masks = data
    return UT3Weights(tucker_weight_supercore, tt_weight_supercore, UT3Masks(*masks))


def ut3_absorb_weights(x: 'UniformTuckerTensorTrain', weights: UT3Weights) -> 'UniformTuckerTensorTrain':
    """Contract diagonal edge weights into ``x``'s supercores (shape-preserving): the returned
    ``UniformTuckerTensorTrain`` represents the fully-weighted network, with ``x``'s masks unchanged.
    Uniform twin of :py:func:`t3toolbox.tucker_tensor_train.t3_absorb_weights`; see
    :py:func:`~t3toolbox.backend.ut3_operations.ut3_absorb_weights` for the side-conventions."""
    _check_weights_pair(x, weights, 'absorb_weights')
    return _from_data(ut3_operations.ut3_absorb_weights(x.data, weights.data))


def ut3_weighted_norm(x: 'UniformTuckerTensorTrain', weights: UT3Weights,
                  use_orthogonalization: bool = True) -> NDArray:  # shape=stack_shape
    """Weighted Hilbert-Schmidt norm ``||absorb_weights(x, weights)||`` (shape ``stack_shape``; a scalar
    when unstacked). The plain norm squares the inserted diagonal, so ``diag(1/sigma)`` penalises by
    ``1/sigma^2``. Uniform twin of :py:func:`t3toolbox.tucker_tensor_train.t3_weighted_norm`."""
    _check_weights_pair(x, weights, 'weighted_norm')
    return ut3_linalg.ut3_weighted_norm(x.data, weights.data, use_orthogonalization=use_orthogonalization)


def ut3_weighted_inner(
        x_A:       'UniformTuckerTensorTrain',
        weights_A: UT3Weights,
        x_B:       'UniformTuckerTensorTrain',
        weights_B: UT3Weights,
        use_orthogonalization: bool = True,
) -> NDArray:  # weighted HS inner product, shape=stack_shape
    """Weighted Hilbert-Schmidt inner product
    ``<absorb_weights(x_A, weights_A), absorb_weights(x_B, weights_B)>``. Operands share physical shape;
    ranks/masks/weights may differ. Uniform twin of
    :py:func:`t3toolbox.tucker_tensor_train.t3_weighted_inner`."""
    _check_weights_pair(x_A, weights_A, 'weighted_inner')
    _check_weights_pair(x_B, weights_B, 'weighted_inner')
    if x_A.shape != x_B.shape:
        raise ValueError('Cannot weighted-inner UniformTuckerTensorTrains with different shapes: %s vs %s.'
                         % (x_A.shape, x_B.shape))
    return ut3_linalg.ut3_weighted_inner(x_A.data, weights_A.data, x_B.data, weights_B.data,
                                         use_orthogonalization=use_orthogonalization)


def _check_weights_pair(x: 'UniformTuckerTensorTrain', weights: UT3Weights, op: str) -> None:
    """Structural precondition for every ``(train, weights)`` op: the weights must fit ``x`` and declare
    the SAME edge masks. Enforced in the frontend (the backend twin is the non-raising
    ``ut3_weights_consistent``), because uniform padding hides a mismatch that ragged would catch as a
    numpy shape error -- the ``UT3Variations`` same-mask precedent. Masks are host numpy, so this is a
    cheap ``array_equal``, valid even under jit."""
    if not weights.is_consistent_with(x):
        raise ValueError(
            'Inconsistent (UniformTuckerTensorTrain, UT3Weights) pair in %s.\n'
            'The weights must match the train\'s padded (n, r) and declare the SAME edge masks (a '
            'weight\'s edges ARE the tensor\'s edges).\n'
            'train: n=%d, r=%d, stack_shape=%s ; weights: n=%d, r=%d, stack_shape=%s ; masks equal: %s'
            % (op, x.n, x.r, x.stack_shape, weights.n, weights.r, weights.stack_shape,
               weights.masks == x.masks))


if common.jax_available:
    # UniformTuckerTensorTrain as a jax pytree: the two supercores are the (traced) children; the static
    # aux_data is ``(shape, UT3Masks)``. Both are STRUCTURE (the real mode dims + which rank slots are
    # real), not data, so they belong in aux, not the traced leaves. BOTH are value-keyed: ``shape`` is a
    # value-hashable int tuple, and ``UT3Masks`` hashes/compares by mask CONTENT (the ValueHashedMasks
    # mixin). So the jit cache key reflects the rank STRUCTURE -- a rebuilt-but-identical object is the
    # same key (no per-iteration recompile when frames are re-orthogonalized in an optimization loop); a
    # genuinely different structure recompiles (correct). Because uniform output ranks are STATICALLY
    # determined (no rtol; shrink-to-structural-minimum), a jitted op's output masks stay compile-time
    # constants -- safe as aux. See docs/contributor/uniform_pytree_composition.md.
    jax.tree_util.register_pytree_node(
        UniformTuckerTensorTrain,
        lambda x: ((x.tucker_supercore, x.tt_supercore), (x.shape, x.masks)),
        lambda aux, children: UniformTuckerTensorTrain(children[0], children[1], aux[0], aux[1]),
    )
    # UT3Weights follows the same split: the two weight supercores are the traced children (weights are
    # float PARAMETERS -- differentiable data), the UT3Masks holder is value-hashed static aux (masks are
    # boolean STRUCTURE). That opposite treatment is exactly why the two are kept apart; see
    # docs/contributor/uniform_rank_masks_rationale.md. No ``shape`` in aux: weights have no physical legs.
    jax.tree_util.register_pytree_node(
        UT3Weights,
        lambda w: ((w.tucker_weight_supercore, w.tt_weight_supercore), w.masks),
        lambda aux, children: UT3Weights(children[0], children[1], aux),
    )
