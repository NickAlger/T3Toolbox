'''Least-squares fitting operators (the Gauss-Newton model) for the ``apply`` and ``entries`` sampling.

The fitting layer composes the **bare** probing primitives (``probing.<kind>_*_from_sweep``, the
single-sample Jacobian ``𝒥`` / its transpose ``𝒥ᵀ``) with the **gauge projector** ``Π`` to form the
*Riemannian* least-squares operators a Gauss-Newton solver consumes, per sampling kind:

    <kind>_jacobian      = 𝒥(Π p)            (Riemannian forward J p)
    <kind>_gradient      = Π 𝒥ᵀ r            (the Gauss-Newton J^T r, sum over the sample stack W)
    <kind>_gn_hessian    = Π 𝒥ᵀ 𝒥 Π p        (H = JᵀJ, the GN normal operator)
    <kind>_model_value   = c + ⟨g, Π p⟩ + ½‖𝒥 Π p‖²   (the local quadratic-model value m(p))

with ``<kind>`` in ``{apply, entries, probe}``. The functions are named by the **sampling kind**
(apply / entries / probe), exactly like ``probing.apply_jacobian_from_sweep`` -- not by the operator verb.
(``probe`` is vector-valued -- one free mode per probe -- so its residual is a sequence of ``d`` arrays and
its ``½‖𝒥Πp‖²`` reduces over the probe vectors too; apply/entries are the scalar all-modes special case.)

Two principles (see ``docs/fitting_plan.md``):

- **Self-contained (the razor).** Every function applies ``Π`` itself, so a backend user working on raw
  ``.data`` tuples gets the correct gauge-projected Riemannian result without remembering the gauge. The
  bare ``𝒥`` / ``𝒥ᵀ`` (no ``Π``) live in ``probing.py`` for callers who explicitly want them.
- **Manifold ⟺ ``Π``; corewise ⟺ no ``Π``.** These are the *manifold* (tangent) operators. The corewise
  (free-core) variants carry **no** ``Π``; mixing the two silently corrupts the result. That matched pair
  is structural (no ``apply_gauge`` flag); the corewise wrapper lives elsewhere.

``sum_over_probes=True`` throughout (the normal operator and gradient sum the sample stack ``W``). The
base sweep ``(xis, mus, nus, etas)`` is precomputed once per base (``probing.precompute_<kind>_base_sweep``)
and reused across every ``J`` / ``Jᵀ`` of an inner solve.
'''

import typing as typ

import t3toolbox.backend.probing as probing
import t3toolbox.backend.tangent_operations as tangent_operations
import t3toolbox.corewise as cw
from t3toolbox.backend.common import *

__all__ = [
    'apply_jacobian',
    'apply_gradient',
    'apply_gn_hessian',
    'apply_model_value',
    'entries_jacobian',
    'entries_gradient',
    'entries_gn_hessian',
    'entries_model_value',
    'probe_jacobian',
    'probe_gradient',
    'probe_gn_hessian',
    'probe_model_value',
]


def _sumsq_over_samples(
        Jp:     NDArray,    # forward output, shape W+C (scalar-output kinds: apply / entries)
        n_w:    int,        # number of leading sample-stack (W) axes
) -> NDArray:               # sum of squares over W, keeping the base stack C
    '''The ``‖𝒥 Π p‖²`` reduction shared by the scalar-output (apply / entries) model values: sum
    ``Jp**2`` over the leading ``n_w`` sample axes, keeping the base stack ``C``.'''
    use_jax = is_jax_ndarray(Jp)
    xnp, _, _ = get_backend(False, use_jax)
    return xnp.sum(Jp ** 2, axis=tuple(range(n_w)))


def _sumsq_over_probes(
        zz:     typ.Sequence[NDArray],  # probe forward output, len=d, elm_shape=W+C+(Ni,)
        n_w:    int,                    # number of leading sample-stack (W) axes
) -> NDArray:                           # sum of squares over W and the free mode Ni, summed over d, keep C
    '''The ``‖𝒥 Π p‖²`` reduction for the vector-output ``probe`` kind: each probe ``z_i`` is ``W+C+(Ni,)``;
    sum ``z_i**2`` over the leading ``n_w`` sample axes and the trailing free mode ``Ni``, keeping the base
    stack ``C``, and sum over the ``d`` probes.'''
    use_jax = is_jax_ndarray(zz[0])
    xnp, _, _ = get_backend(False, use_jax)
    total = None
    for z in zz:
        axes = tuple(range(n_w)) + (z.ndim - 1,)        # W (leading) + the free mode Ni (last); keep C
        s = xnp.sum(z ** 2, axis=axes)
        total = s if total is None else total + s
    return total


############################################
##########   Apply   #######################
############################################

def apply_jacobian(
        p:          typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations dU. len=d, elm_shape=C+(nOi,Ni)
            typ.Sequence[NDArray],          # tt variations     dG. len=d, elm_shape=C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data of the trial tangent (any gauge)
        ww:         typ.Sequence[NDArray],  # sample vectors, len=d, elm_shape=W+(Ni,)
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q), the orthonormal frame
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_base_sweep(base, ww)
) -> NDArray:                               # J p = 𝒥(Π p), shape W+C (one scalar per sample, per base)
    '''Riemannian forward all-modes apply ``J p = 𝒥(Π p)``: gauge-project ``p`` (so the caller need not),
    then apply the bare single-sample apply Jacobian, reusing the precomputed base sweep.'''
    Pp = tangent_operations.orthogonal_gauge_projection(base, p)
    return probing.apply_jacobian_from_sweep(Pp, ww, base, base_sweep)


def apply_gradient(
        r:          NDArray,                # residual, shape W+C
        ww:         typ.Sequence[NDArray],  # sample vectors, len=d, elm_shape=W+(Ni,)
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_base_sweep(base, ww)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # tucker variations dU. len=d
    typ.Sequence[NDArray],  # tt variations     dG. len=d
]:                                          # g = Π 𝒥ᵀ r, a gauged tangent = T3Variations.data
    '''Riemannian gradient ``g = Π 𝒥ᵀ r`` (the Gauss-Newton ``Jᵀr``): bare transpose summed over the
    sample stack ``W``, then gauge-projected onto the tangent space.'''
    dU_dG = probing.apply_transpose_from_sweep(r, ww, base_sweep, sum_over_probes=True)
    return tangent_operations.orthogonal_gauge_projection(base, dU_dG)


def apply_gn_hessian(
        p:          typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations dU. len=d, elm_shape=C+(nOi,Ni)
            typ.Sequence[NDArray],          # tt variations     dG. len=d, elm_shape=C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data of the trial tangent (any gauge)
        ww:         typ.Sequence[NDArray],  # sample vectors, len=d, elm_shape=W+(Ni,)
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_base_sweep(base, ww)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # tucker variations dU. len=d
    typ.Sequence[NDArray],  # tt variations     dG. len=d
]:                                          # H p = Π 𝒥ᵀ 𝒥 Π p, a gauged tangent = T3Variations.data
    '''The Gauss-Newton normal operator ``H p = Π 𝒥ᵀ 𝒥 Π p`` (``H = JᵀJ``, the GN Hessian -- *not* the
    full Hessian). Symmetric and maps gauged variations to gauged variations.'''
    z = apply_jacobian(p, ww, base, base_sweep)                  # 𝒥 Π p, shape W+C
    dU_dG = probing.apply_transpose_from_sweep(z, ww, base_sweep, sum_over_probes=True)  # 𝒥ᵀ
    return tangent_operations.orthogonal_gauge_projection(base, dU_dG)                   # Π


def apply_model_value(
        p:               typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations dU. len=d, elm_shape=C+(nOi,Ni)
            typ.Sequence[NDArray],          # tt variations     dG. len=d, elm_shape=C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data of the trial tangent (any gauge)
        ww:              typ.Sequence[NDArray],  # sample vectors, len=d, elm_shape=W+(Ni,)
        base:            typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep:      typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_base_sweep(base, ww)
        gradient:        typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations of g
            typ.Sequence[NDArray],          # tt variations of g
        ],                                  # = apply_gradient(r, ...), the model's gauged gradient g
        objective_value: NDArray,           # c = ½‖r‖², shape C
) -> NDArray:                               # m(p) = c + ⟨g,p⟩ + ½‖J p‖², shape C
    '''The local Gauss-Newton model value ``m(p) = c + gᵀp + ½ pᵀ H p`` with ``H = JᵀJ``, reusing the
    cached ``c`` (``objective_value``) and ``g`` (``gradient``) and one **forward** apply: the quadratic
    term needs only ``½ pᵀ H p = ½‖𝒥 Π p‖²``, not a full Hessian apply. ``p`` is gauge-projected here, so
    the linear term ``⟨g, Πp⟩`` (a corewise dot of two gauged tangents) and ``½‖𝒥Πp‖²`` share the one
    ``Πp``. Equals ``½‖r + 𝒥Πp‖²`` exactly.'''
    n_c = base[0][0].ndim - 2                   # base-stack (C) axes: U_i is C+(nUi,Ni)
    n_w = ww[0].ndim - 1                         # sample-stack (W) axes: w_i is W+(Ni,)
    Pp = tangent_operations.orthogonal_gauge_projection(base, p)         # Π p (shared by both terms)
    Jp = probing.apply_jacobian_from_sweep(Pp, ww, base, base_sweep)     # 𝒥 Π p, shape W+C
    return objective_value + cw.corewise_stack_dot(gradient, Pp, n_c) + 0.5 * _sumsq_over_samples(Jp, n_w)


############################################
##########   Entries   #####################
############################################

def entries_jacobian(
        p:          typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations dU. len=d, elm_shape=C+(nOi,Ni)
            typ.Sequence[NDArray],          # tt variations     dG. len=d, elm_shape=C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data of the trial tangent (any gauge)
        index:      NDArray,                # int, shape=(d,)+W -- the grid points
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q), the orthonormal frame
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_entries_base_sweep(base, index)
) -> NDArray:                               # J p = 𝒥(Π p), shape W+C (one entry per index, per base)
    '''Riemannian forward all-modes entries ``J p = 𝒥(Π p)`` at ``index``: gauge-project ``p``, then the
    bare single-sample entries Jacobian, reusing the precomputed base sweep.'''
    Pp = tangent_operations.orthogonal_gauge_projection(base, p)
    return probing.entries_jacobian_from_sweep(Pp, index, base, base_sweep)


def entries_gradient(
        r:          NDArray,                # residual, shape W+C
        index:      NDArray,                # int, shape=(d,)+W -- the grid points
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_entries_base_sweep(base, index)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # tucker variations dU. len=d
    typ.Sequence[NDArray],  # tt variations     dG. len=d
]:                                          # g = Π 𝒥ᵀ r, a gauged tangent = T3Variations.data
    '''Riemannian gradient ``g = Π 𝒥ᵀ r`` for entries: bare entries transpose summed over the sample
    stack ``W`` (the entry scatter), then gauge-projected onto the tangent space.'''
    dU_dG = probing.entries_transpose_from_sweep(r, index, base, base_sweep, sum_over_probes=True)
    return tangent_operations.orthogonal_gauge_projection(base, dU_dG)


def entries_gn_hessian(
        p:          typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations dU. len=d, elm_shape=C+(nOi,Ni)
            typ.Sequence[NDArray],          # tt variations     dG. len=d, elm_shape=C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data of the trial tangent (any gauge)
        index:      NDArray,                # int, shape=(d,)+W -- the grid points
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_entries_base_sweep(base, index)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # tucker variations dU. len=d
    typ.Sequence[NDArray],  # tt variations     dG. len=d
]:                                          # H p = Π 𝒥ᵀ 𝒥 Π p, a gauged tangent = T3Variations.data
    '''The Gauss-Newton normal operator ``H p = Π 𝒥ᵀ 𝒥 Π p`` for entries (``H = JᵀJ``). Symmetric and
    maps gauged variations to gauged variations.'''
    z = entries_jacobian(p, index, base, base_sweep)             # 𝒥 Π p, shape W+C
    dU_dG = probing.entries_transpose_from_sweep(z, index, base, base_sweep, sum_over_probes=True)  # 𝒥ᵀ
    return tangent_operations.orthogonal_gauge_projection(base, dU_dG)                              # Π


def entries_model_value(
        p:               typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations dU. len=d, elm_shape=C+(nOi,Ni)
            typ.Sequence[NDArray],          # tt variations     dG. len=d, elm_shape=C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data of the trial tangent (any gauge)
        index:           NDArray,           # int, shape=(d,)+W -- the grid points
        base:            typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep:      typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_entries_base_sweep(base, index)
        gradient:        typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations of g
            typ.Sequence[NDArray],          # tt variations of g
        ],                                  # = entries_gradient(r, ...), the model's gauged gradient g
        objective_value: NDArray,           # c = ½‖r‖², shape C
) -> NDArray:                               # m(p) = c + ⟨g,p⟩ + ½‖J p‖², shape C
    '''The local Gauss-Newton model value for entries (as :py:func:`apply_model_value`, with the fiber-
    sliced forward). One forward apply; reuses the cached ``c`` and ``g``. Equals ``½‖r + 𝒥Πp‖²``.'''
    n_c = base[0][0].ndim - 2                   # base-stack (C) axes: U_i is C+(nUi,Ni)
    n_w = index.ndim - 1                         # sample-stack (W) axes: index is (d,)+W
    Pp = tangent_operations.orthogonal_gauge_projection(base, p)             # Π p (shared by both terms)
    Jp = probing.entries_jacobian_from_sweep(Pp, index, base, base_sweep)    # 𝒥 Π p, shape W+C
    return objective_value + cw.corewise_stack_dot(gradient, Pp, n_c) + 0.5 * _sumsq_over_samples(Jp, n_w)


############################################
##########   Probe   #######################
############################################

def probe_jacobian(
        p:          typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations dU. len=d, elm_shape=C+(nOi,Ni)
            typ.Sequence[NDArray],          # tt variations     dG. len=d, elm_shape=C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data of the trial tangent (any gauge)
        ww:         typ.Sequence[NDArray],  # probe vectors, len=d, elm_shape=W+(Ni,)
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q), the orthonormal frame
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_base_sweep(base, ww) (apply & probe share it)
) -> typ.Sequence[NDArray]:                 # J p = 𝒥(Π p), len=d, elm_shape=W+C+(Ni,) (one free mode each)
    '''Riemannian forward probe ``J p = 𝒥(Π p)``: gauge-project ``p``, then the bare single-sample probe
    Jacobian (one vector per mode), reusing the precomputed base sweep.'''
    Pp = tangent_operations.orthogonal_gauge_projection(base, p)
    return probing.probe_jacobian_from_sweep(Pp, ww, base, base_sweep)


def probe_gradient(
        r:          typ.Sequence[NDArray],  # probe residual, len=d, elm_shape=W+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors, len=d, elm_shape=W+(Ni,)
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_base_sweep(base, ww)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # tucker variations dU. len=d
    typ.Sequence[NDArray],  # tt variations     dG. len=d
]:                                          # g = Π 𝒥ᵀ r, a gauged tangent = T3Variations.data
    '''Riemannian gradient ``g = Π 𝒥ᵀ r`` for probe: bare probe transpose summed over the sample stack
    ``W``, then gauge-projected onto the tangent space.'''
    dU_dG = probing.probe_transpose_from_sweep(r, ww, base, base_sweep, sum_over_probes=True)
    return tangent_operations.orthogonal_gauge_projection(base, dU_dG)


def probe_gn_hessian(
        p:          typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations dU. len=d, elm_shape=C+(nOi,Ni)
            typ.Sequence[NDArray],          # tt variations     dG. len=d, elm_shape=C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data of the trial tangent (any gauge)
        ww:         typ.Sequence[NDArray],  # probe vectors, len=d, elm_shape=W+(Ni,)
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_base_sweep(base, ww)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # tucker variations dU. len=d
    typ.Sequence[NDArray],  # tt variations     dG. len=d
]:                                          # H p = Π 𝒥ᵀ 𝒥 Π p, a gauged tangent = T3Variations.data
    '''The Gauss-Newton normal operator ``H p = Π 𝒥ᵀ 𝒥 Π p`` for probe (``H = JᵀJ``). Symmetric and maps
    gauged variations to gauged variations.'''
    z = probe_jacobian(p, ww, base, base_sweep)                  # 𝒥 Π p, d probe vectors
    dU_dG = probing.probe_transpose_from_sweep(z, ww, base, base_sweep, sum_over_probes=True)  # 𝒥ᵀ
    return tangent_operations.orthogonal_gauge_projection(base, dU_dG)                         # Π


def probe_model_value(
        p:               typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations dU. len=d, elm_shape=C+(nOi,Ni)
            typ.Sequence[NDArray],          # tt variations     dG. len=d, elm_shape=C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data of the trial tangent (any gauge)
        ww:              typ.Sequence[NDArray],  # probe vectors, len=d, elm_shape=W+(Ni,)
        base:            typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep:      typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = probing.precompute_base_sweep(base, ww)
        gradient:        typ.Tuple[
            typ.Sequence[NDArray],          # tucker variations of g
            typ.Sequence[NDArray],          # tt variations of g
        ],                                  # = probe_gradient(r, ...), the model's gauged gradient g
        objective_value: NDArray,           # c = ½‖r‖², shape C
) -> NDArray:                               # m(p) = c + ⟨g,p⟩ + ½‖J p‖², shape C
    '''The local Gauss-Newton model value for probe (as :py:func:`apply_model_value`, but the forward is
    vector-valued, so ``½‖𝒥Πp‖²`` reduces over the probe vectors too). One forward apply; reuses ``c``,
    ``g``. Equals ``½‖r + 𝒥Πp‖²``.'''
    n_c = base[0][0].ndim - 2                   # base-stack (C) axes: U_i is C+(nUi,Ni)
    n_w = ww[0].ndim - 1                         # sample-stack (W) axes: w_i is W+(Ni,)
    Pp = tangent_operations.orthogonal_gauge_projection(base, p)         # Π p (shared by both terms)
    zz = probing.probe_jacobian_from_sweep(Pp, ww, base, base_sweep)     # 𝒥 Π p, d probe vectors
    return objective_value + cw.corewise_stack_dot(gradient, Pp, n_c) + 0.5 * _sumsq_over_probes(zz, n_w)
