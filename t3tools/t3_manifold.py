# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ
from t3tools.tucker_tensor_train import *
from t3tools.t3_base_variation_format import *
from t3tools.t3_orthogonalization import *
from t3tools.t3_svd import *

try:
    import jax.numpy as jnp
except:
    print('jax import failed in tucker_tensor_train. Defaulting to numpy.')
    jnp = np

NDArray = typ.Union[np.ndarray, jnp.ndarray]


__all__ = [
    'T3Tangent',
    't3tangent_to_dense',
    't3tangent_to_t3',
    't3_orthogonal_gauge_projection',
    't3_oblique_gauge_projection',
    'project_t3_onto_tangent_space',
    't3_retract',
    't3tangent_zeros',
    't3tangent_randn',
    't3tangent_scale',
    't3tangent_add',
    't3tangent_neg',
    't3tangent_sub',
    't3tangent_dot_t3tangent',
    't3tangent_norm',
]


####################################################################
###########    Fixed rank TuckerTensorTrain manifold   #############
####################################################################

T3Tangent = typ.Tuple[
    T3Base, # Orthogonal representations of base point
    T3Variation, # Variations
]
"""Tuple representation of a Tucker tensor train tangent vector in terms of an orthogonal base and variations.

Implements the representation of a tangent vector as a sum of the following form::

          1 -- H0 -- R1 -- R2 -- R3 -- 1            1 -- L0 -- L1 -- L2 -- H3 -- 1
               |     |     |     |                       |     |     |     |
    v  =       U0    U1    U2    U3       +  ... +       U0    U1    U2    U3
               |     |     |     |                       |     |     |     |
    
          1 -- O0 -- R1 -- R2 -- R3 -- 1            1 -- L0 -- L1 -- L2 -- O3 -- 1
       +       |     |     |     |                       |     |     |     |
               V0    U1    U2    U3       + ... +        U0    U1    U2    V3
               |     |     |     |                       |     |     |     |

**Components**
    - T3Base: orthogonal representation of the base point where the tangent space is attached.
        - basis_cores       = (U0,...,Ud), orthogonal
        - left_tt_cores     = (L0,...Ld), left-orthogonal
        - right_tt_cores    = (R0,...,Rd), right-orthogonal
        - outer_tt_cores    = (O0,...,Od), outer-orthogonal
    - T3Variation: The variations defining the tangent vector w.r.t. the base point.
        - basis_variations  = (V0,...,Vd)
        - tt_variations     = (H0,...,Hd)

Examples
--------
>>> from numpy.random import randn
>>> from t3tools.t3_manifold import *
>>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
>>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
>>> p = (basis_cores, tt_cores)
>>> base, _ = t3_orthogonal_representations(p)
>>> other_basis_cores = (randn(7,14), randn(4,15), randn(8,16)) # same shape, different ranks
>>> other_tt_cores = (randn(1,7,5), randn(5,4,4), randn(4,8,1))
>>> x = (other_basis_cores, other_tt_cores)
>>> proj_x = project_t3_onto_tangent_space(x, base)
"""

def t3tangent_to_dense(
        x: T3Tangent,
        include_shift: bool = False, # False: V. True: P+V. P=base point, V=tangent vector
) -> NDArray:
    """Convert Tangent vector to Tucker tensor train manifold into dense tensor.

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> v = x[1]
    >>> v_dense = t3tangent_to_dense(x)
    >>> ((U0,U1,U2), (L0,L1,L2), (R0,R1,R2), (O0,O1,O2)) = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = v
    >>> t1 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,H0,R1,R2)
    >>> t2 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,L0,H1,R2)
    >>> t3 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,L0,L1,H2)
    >>> t4 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', V0,U1,U2,O0,R1,R2)
    >>> t5 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,V1,U2,L0,O1,R2)
    >>> t6 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,V2,L0,L1,O2)
    >>> v_dense2 = t1 + t2 + t3 + t4 + t5 + t6
    >>> print(np.linalg.norm(v_dense - v_dense2))
        1.2760924630140578e-14
    >>> p_plus_v_dense = t3tangent_to_dense(x, include_shift=True)
    >>> p_plus_v_dense2 =  t3_to_dense(p) + v_dense
    >>> print(np.linalg.norm(p_plus_v_dense - p_plus_v_dense2))
        1.2677102046134292e-12
    """
    t3_check_base_variation_fit(*x)

    base, variation = x
    num_cores = len(variation[0])
    basis_terms = [bv_to_t3(ii, False, base, variation) for ii in range(num_cores)]
    tt_terms    = [bv_to_t3(ii, True, base, variation) for ii in range(num_cores)]
    terms = basis_terms + tt_terms
    V = t3_to_dense(terms[0])
    for t in terms[1:]:
        V = V + t3_to_dense(t)

    if include_shift:
        basis_cores, left_tt_cores, _, _ = base
        P = t3_to_dense((basis_cores, left_tt_cores))
        X = P + V
    else:
        X = V

    return X


def t3tangent_to_t3(
        x: T3Tangent,
        include_shift: bool = False,  # False: v. True: p+v. p=base point, v=tangent vector
        use_jax: bool = False,
) -> TuckerTensorTrain:
    '''Rank 2r Tucker tensor train representation of tangent vector.

    Without shift, we use the formula::

        v(x,y,z,w) = ([dU1(B x) L1(B x)]) ([R2(B y)        0]) ([R3(B z)        0]) ([R4(B w) ])
                     (                  ) ([dU2(B y) L2(B y)]) ([dU3(B z) L3(B z)]) ([dU4(B w)])
                     (         +        ) (         +        ) (        +         ) (    +     )
                     ([O1(dB x)       0]) ([0              0]) ([0              0]) ([0       ])
                     (                  ) ([O2(dB y)       0]) ([O3(dB z)       0]) ([O4(dB w)])

    With shift is same as unshifted, except last core modified as follows::

        [R4(B w) ]                  [R4(B w)           ]
        [dU4(B w)]                  [L4(B w) + dU4(B w)]
            +             ->            +
        [0       ]                  [0                 ]
        [O4(dB w)]                  [O4(dB w)          ]

    Parameters
    ----------
    x: T3Tangent
        Tangent vector which will be converted to TuckerTensorTrain with doubled ranks.
    include_shift: bool
        If False, return tangent vector v only. If True, shift tangent vector so it is attached at the base point, p+v.
    use_jax: bool
        If True, returned TuckerTensorTrain cores are jnp.ndaray. Otherwise, np.ndarray. Default: False

    Returns
    -------
    TuckerTensorTrain
        Tucker tensor train representation of tangent vector, which has doubled ranks

    See Also
    --------
    T3Tangent

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> v_t3 = t3tangent_to_t3(x) # tangent vector only (attached at zero)
    >>> v_dense = t3_to_dense(v_t3)
    >>> v_dense2 = t3tangent_to_dense(x)
    >>> print(np.linalg.norm(v_dense - v_dense2))
        2.678565538404836e-15
    >>> p_plus_v_t3 = t3tangent_to_t3(x, include_shift=True) # shifted tangent vector (include attachment at base point)
    >>> p_plus_v_dense = t3_to_dense(p_plus_v_t3)
    >>> p_plus_v_dense2 = v_dense2 + t3_to_dense(p)
    >>> print(np.linalg.norm(p_plus_v_dense - p_plus_v_dense2))
        1.2102169224182523e-12
    '''
    xnp = jnp if use_jax else np

    t3_check_base_variation_fit(*x)
    base, vars = x
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    basis_vars, tt_vars = vars

    num_cores = len(basis_cores)

    x_basis_cores = []
    for B, dB in zip(basis_cores, basis_vars):
        B2 = xnp.concatenate([B, dB], axis=0)
        x_basis_cores.append(B2)

    x_tt_cores = []

    dU = tt_vars[0]
    O = outer_tt_cores[0]
    L = left_tt_cores[0]
    Z = xnp.zeros((O.shape[0], O.shape[1], L.shape[2]))
    G_top = xnp.concatenate([dU, L], axis=2)
    G_bot = xnp.concatenate([O, Z], axis=2)
    G = xnp.concatenate([G_top, G_bot], axis=1)
    x_tt_cores.append(G)

    for ii in range(1, num_cores-1):
        L = left_tt_cores[ii]
        R = right_tt_cores[ii]
        O = outer_tt_cores[ii]
        dU = tt_vars[ii]
        Z001 = xnp.zeros((R.shape[0], dU.shape[1], L.shape[2]))
        Z100 = xnp.zeros((R.shape[0], O.shape[1], R.shape[2]))
        Z101 = xnp.zeros((R.shape[0], O.shape[1], L.shape[2])) #Z001
        Z111 = xnp.zeros((L.shape[0], O.shape[1], L.shape[2])) #jnp.zeros(L.shape)
        G_top = xnp.concatenate([
            xnp.concatenate([R, Z001], axis=2),
            xnp.concatenate([dU, L], axis=2)
        ], axis=0)
        G_bot = xnp.concatenate([
            xnp.concatenate([Z100, Z101], axis=2),
            xnp.concatenate([O, Z111], axis=2)
        ], axis=0)
        G = xnp.concatenate([G_top, G_bot], axis=1)
        x_tt_cores.append(G)

    dU = tt_vars[-1]
    L = left_tt_cores[-1]
    R = right_tt_cores[-1]
    O = outer_tt_cores[-1]
    Z = xnp.zeros((R.shape[0], O.shape[1], R.shape[2]))
    if include_shift:
        G_top = xnp.concatenate([R, L + dU], axis=0)
    else:
        G_top = xnp.concatenate([R, dU], axis=0)
    G_bot = xnp.concatenate([Z, O], axis=0)
    G = xnp.concatenate([G_top, G_bot], axis=1)
    x_tt_cores.append(G)

    return tuple(x_basis_cores), tuple(x_tt_cores)


def t3_orthogonal_gauge_projection(
        x: T3Tangent,
        use_jax: bool = False,
) -> T3Tangent:
    """Makes tangent vector representation gauged via orthogonal projection. Changes tangent vector.

    Gauge condition:
        - All variation basis cores Vi are orthogonal to the corresponding base basis cores Ui:
            Ui @ Vi.T = 0    for    i=1,...,d
        - All but the last variation TT-cores H are left-perpendicular to the corresponding base left TT-cores L:
            einsum('iaj,iak->jk', Hi, Li) = 0    for    i=1,...,d-1

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to project

    use_jax: bool
        If True, use jax operations, if False use numpy.

    Returns
    -------
    T3Tangent
        Projected tangent vector satisfying Gauge condition

    See Also
    --------
    T3Tangent
    t3_oblique_gauge_projection

    Example
    -------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> projected_x = t3_orthogonal_gauge_projection(x)
    >>> (U0,U1,U2), (L0,L1,L2), _, _ = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = projected_x[1]
    >>> print(np.linalg.norm(V1 @ U1.T)) # Gauge condition for basis core 1
        3.512073125137391e-15
    >>>  print(np.linalg.norm(np.einsum('iaj,iak->jk', H1, L1))) # Gauge condition for TT-core 1
        1.5807940730805242e-15
    """
    xnp = jnp if use_jax else np

    t3_check_base_variation_fit(*x)
    base, vars = x
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    basis_vars, tt_vars = vars

    new_tt_variations = []
    for dV, P in zip(tt_vars[:-1], left_tt_cores[:-1]):
        dV2 = dV - xnp.einsum('iaj,jk->iak', P, xnp.einsum('iaj,iak->jk', P, dV))
        new_tt_variations.append(dV2)
    new_tt_variations.append(tt_vars[-1])

    new_basis_variations = []
    for dB, B in zip(basis_vars, basis_cores):
        dB2 = dB - (dB @ B.T) @ B
        new_basis_variations.append(dB2)

    new_vars = (tuple(new_basis_variations), tuple(new_tt_variations))
    new_x = (base, new_vars)
    return new_x


def t3_oblique_gauge_projection(
        x: T3Tangent,
        use_jax: bool = False,
) -> T3Tangent:
    """Makes variations left-perpendicular while preserving tangent vector.

    Method:
        1) Make basis variations left-perpendicular by pushing remainder onto tt variations
        2) Make tt variations left-perpendicular by standard sweeping tt method

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to project

    use_jax: bool
        If True, use jax operations, if False use numpy.

    Returns
    -------
    T3Tangent
        Projected tangent vector satisfying Gauge condition

    See Also
    --------
    T3Tangent
    t3_orthogonal_gauge_projection

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> projected_x = t3_oblique_gauge_projection(x)
    >>> x_dense = t3tangent_to_dense(x)
    >>> proj_x_dense = t3tangent_to_dense(projected_x)
    >>> print(np.linalg.norm(x_dense - proj_x_dense)) # Zero since projection preserves represented tangent vector
        3.4398319441148304e-15
    >>> (U0,U1,U2), (L0,L1,L2), _, _ = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = projected_x[1]
    >>> print(np.linalg.norm(V1 @ U1.T)) # Gauge condition for basis core 1
        2.931519226677228e-15
    >>>  print(np.linalg.norm(np.einsum('iaj,iak->jk', H1, L1))) # Gauge condition for TT-core 1
        6.99005312491287e-16
    """
    xnp = jnp if use_jax else np

    t3_check_base_variation_fit(*x)
    base, vars = x
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    basis_vars, tt_vars = vars
    num_cores = len(basis_cores)

    tt_vars = list(tt_vars)
    basis_vars = list(basis_vars)

    # Make basis variations left-perpendicular
    for ii in range(num_cores):
        B_io = basis_cores[ii]
        dB_jo = basis_vars[ii]
        R_aib = outer_tt_cores[ii]
        dG_ajb = tt_vars[ii]

        X_ji = dB_jo @ B_io.T
        dB_parallel_jo = X_ji @ B_io
        dB2_jo = dB_jo - dB_parallel_jo # dB_perp
        # dG2_ajb = dG_ajb + xnp.einsum('aib,ji->ajb', R_aib, X_ji)
        dG2_ajb = dG_ajb + xnp.einsum('aib,ij->ajb', R_aib, X_ji) # <-- Why is this correct?

        tt_vars[ii] = dG2_ajb
        basis_vars[ii] = dB2_jo

    # Make tt cores left-perpendicular
    for ii in range(num_cores-1):
        dV1 = tt_vars[ii]
        dV2 = tt_vars[ii+1]

        P1 = left_tt_cores[ii]
        Q2 = right_tt_cores[ii+1]
        X = xnp.einsum('iaj,iak->jk', P1, dV1)
        new_dV1 = dV1 - xnp.einsum('iaj,jk->iak', P1, X)
        new_dV2 = dV2 + xnp.einsum('jk,kbl->jbl', X, Q2)

        tt_vars[ii] = new_dV1
        tt_vars[ii+1] = new_dV2

    new_vars = tuple(basis_vars), tuple(tt_vars)
    new_x = (base, new_vars)
    return new_x


def tt_reverse(cores):
    return tuple([G.swapaxes(0, 2) for G in cores[::-1]])


def tt_zipper_left_to_right(
        coresA: typ.Sequence[NDArray],
        coresB: typ.Sequence[NDArray],
        use_jax: bool = False,
) -> typ.Tuple[NDArray, ...]:  # zipper_matrices. len=num_cores+1
    xnp = jnp if use_jax else np

    zipper_matrices = [xnp.array([[1.0]])]
    for GA, GB in zip(coresA, coresB):
        Z_prev = zipper_matrices[-1]
        Z = xnp.einsum('ij,iak,jal->kl', Z_prev, GA, GB)
        zipper_matrices.append(Z)
    return tuple(zipper_matrices)


def tt_zipper_right_to_left(
        coresA: typ.Sequence[NDArray],
        coresB: typ.Sequence[NDArray],
        use_jax: bool = False,
) -> typ.Tuple[NDArray, ...]:  # zipper_matrices. len=num_cores+1
    return tt_zipper_left_to_right(tt_reverse(coresA), tt_reverse(coresB), use_jax=use_jax)[::-1]


def project_t3_onto_tangent_space(
        x: TuckerTensorTrain, # Tucker tensor train to be projected
        orthogonal_base: T3Base, # Orthogonal representations of base point
        use_jax: bool = False,
) -> T3Tangent:
    """Projects TuckerTensorTrain onto tangent space to the manifold of fixed rank TuckerTensorTrains.

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to project
    orthogonal_base: T3Base
        Orthogonal representations of base point on manifold where tangent space is attached
    use_jax: bool
        If True, use jax operations, if False use numpy.

    Returns
    -------
    T3Tangent
        Projection of x onto the tangent space. Satisfies the gauge condition.

    See Also
    --------
    T3Tangent
    t3_oblique_projection
    t3_orthogonal_gauge_projection

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, _ = t3_orthogonal_representations(p)
    >>> other_basis_cores = (randn(7,14), randn(4,15), randn(8,16)) # same shape, different ranks
    >>> other_tt_cores = (randn(1,7,5), randn(5,4,4), randn(4,8,1))
    >>> x = (other_basis_cores, other_tt_cores)
    >>> proj_x = project_t3_onto_tangent_space(x, base)
    >>> P = t3_to_dense(p)
    >>> X = t3_to_dense(x)
    >>> proj_X = t3tangent_to_dense(proj_x)
    >>> print(np.sum((X - proj_X) * (proj_X - P)) / np.sum(X)) # Check that x was projected orthogonally
        3.351282624686308e-11
    """
    xnp = jnp if use_jax else np

    t3_check(x)
    t3_check_base(orthogonal_base)

    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    other_basis_cores, other_tt_cores = x

    base_shape = tuple([B.shape[1] for B in basis_cores])
    other_shape = tuple([B.shape[1] for B in other_basis_cores])
    if base_shape != other_shape:
        raise RuntimeError(
            'Attempted to retract TuckerTensorTrain with wrong shape onto tangent space.\n'
            + str(base_shape) + ' = base_shape != other_shape = ' + str(other_shape)
        )

    other_tt_cores2 = []
    for G_other, B_other, B in zip(other_tt_cores, other_basis_cores, basis_cores):
        G_other2 = xnp.einsum('aib,ix->axb', G_other, B_other @ B.T)
        other_tt_cores2.append(G_other2)

    zipper_left2right = tt_zipper_left_to_right(other_tt_cores2, left_tt_cores, use_jax=use_jax)[:-1]
    zipper_right2left = tt_zipper_right_to_left(other_tt_cores2, right_tt_cores, use_jax=use_jax)[1:]

    ungauged_tt_variations = []
    ungauged_basis_variations = []
    for ZL_ax, ZR_by, G_aib, B_io, R0_xjy, B0_jo in zip(
            zipper_left2right, zipper_right2left,
            other_tt_cores, other_basis_cores,
            outer_tt_cores, basis_cores,
    ):
        X_xiy = xnp.einsum('ax,aib,by->xiy', ZL_ax, G_aib, ZR_by)
        dG_xjy = xnp.einsum('xiy,ij->xjy', X_xiy, B_io @ B0_jo.T)
        M_ij = xnp.einsum('xiy,xjy->ij', X_xiy, R0_xjy)
        dB_jo = xnp.einsum('ij,io->jo', M_ij, B_io)

        ungauged_tt_variations.append(dG_xjy)
        ungauged_basis_variations.append(dB_jo)

    ungauged_u = (orthogonal_base, (ungauged_basis_variations, ungauged_tt_variations))
    gauged_u = t3_orthogonal_gauge_projection(ungauged_u)
    return gauged_u


def t3_retract(
        x: T3Tangent,
        use_jax: bool = False,
) -> TuckerTensorTrain: # retracted Tucker tensor train
    """Retract Tucker tensor train tangent vector to manifold.

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to project
    orthogonal_base: T3Base
        Orthogonal representations of base point on manifold where tangent space is attached
    use_jax: bool
        If True, use jax operations, if False use numpy.

    Returns
    -------
    T3Tangent
        Projection of x onto the tangent space. Satisfies the gauge condition.

    See Also
    --------
    T3Tangent
    t3_svd

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> ret_x = t3_retract(x)
    >>> print(np.linalg.norm(t3_to_dense(ret_x) - t3tangent_to_dense(x, include_shift=True)))
        0.12395762641282583
    >>> x2 = t3tangent_scale(x, 1e-3) # make the tangent vector shorter for smaller retraction
    >>> ret_x2 = t3_retract(x2)
    >>> print(np.linalg.norm(t3_to_dense(ret_x2) - t3tangent_to_dense(x2, include_shift=True)))
        1.256998583616651e-07
    """
    t3_check_base_variation_fit(*x)

    basis_cores, left_tt_cores, _, _ = x[0]
    _, base_tucker_ranks, base_tt_ranks = t3_structure((basis_cores, left_tt_cores))

    x_t3 = t3tangent_to_t3(x, include_shift=True)
    retracted_x_t3, _, _ = t3_svd(
        x_t3,
        max_tt_ranks = base_tt_ranks,
        max_tucker_ranks = base_tucker_ranks,
        use_jax=use_jax,
    )
    return retracted_x_t3


####################################################################
############    Construct special kinds of T3Tangent   #############
####################################################################

def t3tangent_zeros(
        orthogonal_base: T3Base, # orthogonal base
        use_jax: bool = False,
) -> T3Tangent:
    """Construct the zero vector in a Tucker tensor train tangent space.

    Parameters
    ----------
    orthogonal_base: T3Base
        Orthogonal representations of base point on manifold where tangent space is attached
    use_jax: bool
        If True, return jax arrays, if False return numpy.

    Returns
    -------
    T3Tangent
        Zero vector in the tangent space

    See Also
    --------
    T3Tangent
    t3tangent_randn

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> z = t3tangent_zeros(base)
    >>> print(np.linalg.norm(t3tangent_to_dense(z)))
        0.0
    """
    xnp = jnp if use_jax else np

    t3_check_base(orthogonal_base)

    var_basis_shapes, var_tt_shapes = t3_base_hole_shapes(orthogonal_base)

    basis_vars = tuple([xnp.zeros(s) for s in var_basis_shapes])
    tt_vars = tuple([xnp.zeros(s) for s in var_tt_shapes])

    zero = (orthogonal_base, (basis_vars, tt_vars))
    return zero


def t3tangent_randn(
        orthogonal_base: T3Base, # orthogonal base
        use_jax: bool = False,
) -> T3Tangent:
    """Draw a random T3Tangent from a uniform normal disribution on the tangent space.

    Parameters
    ----------
    orthogonal_base: T3Base
        Orthogonal representations of base point on manifold where tangent space is attached

    Returns
    -------
    T3Tangent
        Random tangent vector
    use_jax: bool
        If True, return jax arrays, if False return numpy. Should update this to use pure jax, rather than converting numpy->jax.

    See Also
    --------
    T3Tangent
    t3tangent_zeros

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    """
    t3_check_base(orthogonal_base)

    var_basis_shapes, var_tt_shapes = t3_base_hole_shapes(orthogonal_base)

    if use_jax:
        _randn = lambda x: jnp.array(np.random.randn(x))
    else:
        _randn = np.random.randn

    basis_vars0 = tuple([_randn(*s) for s in var_basis_shapes])
    tt_vars0 = tuple([_randn(*s) for s in var_tt_shapes])

    x0 = (orthogonal_base, (basis_vars0, tt_vars0))
    x = t3_orthogonal_gauge_projection(x0)
    return x


####################################################################
##################    T3Tangent linear algebra   ###################
####################################################################

def t3tangent_scale(
        x: T3Tangent,
        s, # scalar
) -> T3Tangent:
    """Scale T3Tangent.

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to scale
    s: scalar
        Scaling factor

    Returns
    -------
    T3Tangent
        Scaled T3Tangent, s*x.

    See Also
    --------
    T3Tangent
    t3tangent_add
    t3tangent_sub
    t3tangent_neg
    t3tangent_dot_t3tangent
    t3tangent_norm

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> s = randn()
    >>> sx = t3tangent_scale(x, s)
    >>> dense_sx = t3tangent_to_dense(sx)
    >>> dense_x = t3tangent_to_dense(x)
    >>> print(np.linalg.norm(dense_sx - s*dense_x))
        2.370617783938243e-15
    """
    t3_check_base_variation_fit(*x)

    base, vars = x
    basis_vars, tt_vars = list(vars)
    new_basis_vars = tuple([s*B for B in basis_vars])
    new_tt_vars = tuple([s*G for G in tt_vars])

    new_x = (base, (new_basis_vars, new_tt_vars))
    return new_x


def t3tangent_add(
        x: T3Tangent,
        y: T3Tangent,
) -> T3Tangent:
    """Add T3Tangents, (x, y) -> x+y.

    Parameters
    ----------
    x: T3Tangent
        First tangent vector summand
    y: T3Tangent
        Second tangent vector summand

    Returns
    -------
    T3Tangent
        Sum of tangent vectors, x+y

    Raises
    ------
    RuntimeError
        Error raised if either T3Tangent is inconsistent, or if x and y have different bases.

    See Also
    --------
    T3Tangent
    t3tangent_scale
    t3tangent_sub
    t3tangent_neg
    t3tangent_dot_t3tangent
    t3tangent_norm

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> y = t3tangent_randn(base)
    >>> x_plus_y = t3tangent_add(x, y)
    >>> x_dense = t3tangent_to_dense(x)
    >>> y_dense = t3tangent_to_dense(y)
    >>> x_plus_y_dense = t3tangent_to_dense(x_plus_y)
    >>> print(np.linalg.norm(x_dense + y_dense - x_plus_y_dense))
        4.667707616068206e-15
    """
    t3_check_base_variation_fit(*x)
    t3_check_base_variation_fit(*y)

    x_base = x[0]
    y_base = y[0]
    if x_base != y_base:
        raise RuntimeError(
            'Attempted to add T3Tangent vectors with different bases.'
        )

    x_basis_vars, x_tt_vars = x[1]
    y_basis_vars, y_tt_vars = y[1]

    x_plus_y_basis_vars = tuple([Bx + By for Bx, By in zip(x_basis_vars, y_basis_vars)])
    x_plus_y_tt_vars = tuple([Gx + Gy for Gx, Gy in zip(x_tt_vars, y_tt_vars)])

    x_plus_y = (x_base, (x_plus_y_basis_vars, x_plus_y_tt_vars))
    return x_plus_y


def t3tangent_neg(
        x: T3Tangent,
) -> T3Tangent:
    """Negative of T3Tangent, x -> -x.

    Parameters
    ----------
    x: T3Tangent
        input tangent vector

    Returns
    -------
    T3Tangent
        Negative of x, i.e., -x.

    See Also
    --------
    T3Tangent
    t3tangent_add
    t3tangent_sub
    t3tangent_scale
    t3tangent_dot_t3tangent
    t3tangent_norm

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> neg_x = t3tangent_neg(x)
    >>> dense_neg_x = t3tangent_to_dense(neg_x)
    >>> dense_x = t3tangent_to_dense(x)
    >>> print(np.linalg.norm(dense_neg_x + dense_x))
        0.0
    """
    return t3tangent_scale(x, -1.0)


def t3tangent_sub(
        x: T3Tangent,
        y: T3Tangent,
) -> T3Tangent:
    """Subtract T3Tangents, (x,y) -> x-y.

    Parameters
    ----------
    x: T3Tangent
        First tangent vector summand
    y: T3Tangent
        Second tangent vector summand

    Returns
    -------
    T3Tangent
        Difference of tangent vectors, x-y

    Raises
    ------
    RuntimeError
        Error raised if either T3Tangent is inconsistent, or if x and y have different bases.

    See Also
    --------
    T3Tangent
    t3tangent_scale
    t3tangent_add
    t3tangent_neg
    t3tangent_dot_t3tangent
    t3tangent_norm

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> y = t3tangent_randn(base)
    >>> x_minus_y = t3tangent_sub(x, y)
    >>> x_dense = t3tangent_to_dense(x)
    >>> y_dense = t3tangent_to_dense(y)
    >>> x_minus_y_dense = t3tangent_to_dense(x_minus_y)
    >>> print(np.linalg.norm(x_dense - y_dense - x_minus_y_dense))
        4.714955371344249e-15
    """
    return t3tangent_add(x, t3tangent_scale(y, -1.0))


def t3tangent_dot_t3tangent(
        x: T3Tangent,
        y: T3Tangent,
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt inner product (dot product in ambient space) of T3Tangents, (x,y)_HS

    Parameters
    ----------
    x: T3Tangent
        First tangent vector
    y: T3Tangent
        Second tangent vector

    Returns
    -------
    Scalar
        Inner product of tangent vectors, (x,y)_HS

    Raises
    ------
    RuntimeError
        Error raised if either T3Tangent is inconsistent, or if x and y have different bases.

    See Also
    --------
    T3Tangent
    t3tangent_scale
    t3tangent_add
    t3tangent_neg
    t3tangent_sub
    t3tangent_norm

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> y = t3tangent_randn(base)
    >>> x_dot_y = t3tangent_dot_t3tangent(x, y)
    >>> x_dense = t3tangent_to_dense(x)
    >>> y_dense = t3tangent_to_dense(y)
    >>> x_dot_y2 = np.sum(x_dense * y_dense)
    >>> print(np.abs(x_dot_y - x_dot_y2))
        1.2434497875801753e-14
    """
    xnp = jnp if use_jax else np

    t3_check_base_variation_fit(*x)
    t3_check_base_variation_fit(*y)

    x_base = x[0]
    y_base = y[0]
    if x_base != y_base:
        raise RuntimeError(
            'Attempted to dot T3Tangent vectors with different bases.'
        )

    x_basis_vars, x_tt_vars = x[1]
    y_basis_vars, y_tt_vars = y[1]

    t1 = xnp.sum([np.sum(Bx*By) for Bx, By in zip(x_basis_vars, y_basis_vars)])
    t2 = xnp.sum([np.sum(Gx*Gy) for Gx, Gy in zip(x_tt_vars, y_tt_vars)])

    return t1+t2


def t3tangent_norm(
        x: T3Tangent,
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt (Frobenius) norm (in ambient space) of T3Tangent, ||x||_HS.

    Parameters
    ----------
    x: T3Tangent
        Tangent vector to compute the norm of.

    Returns
    -------
    Scalar
        Norm of tangent vector,||x||_HS.

    Raises
    ------
    RuntimeError
        Error raised if x is inconsistent.

    See Also
    --------
    T3Tangent
    t3tangent_scale
    t3tangent_add
    t3tangent_neg
    t3tangent_sub
    t3tangent_dot_t3tangent

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.t3_manifold import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> p = (basis_cores, tt_cores)
    >>> base, vars0 = t3_orthogonal_representations(p)
    >>> x = t3tangent_randn(base)
    >>> norm_x = t3tangent_norm(x)
    >>> x_dense = t3tangent_to_dense(x)
    >>> norm_x2 = np.linalg.norm(x_dense)
    >>> print(np.abs(norm_x - norm_x2))
        3.552713678800501e-15
    """
    xnp = jnp if use_jax else np
    return xnp.sqrt(t3tangent_dot_t3tangent(x, x, use_jax=use_jax))











# # # #

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import functools as ft
import typing as typ

from .tt_basic_operations import *
from .tucker_tensor_train import *
from t3tools.t3_base_variation_format import *

__all__ = [
    'T3TangentSpace',
    'T3Variations',
    'UniformT3TangentSpace',
    'UniformT3Variations',
    'pack_matrices',
    'unpack_matrices',
    'ut3_orthogonal_gauge_projection',
    'get_t3_tangent_space_dim',
    'ut3_orthogonal_gauge_projection_using_map',
]

####################################################################
#################    Tensor train tangent space   ##################
####################################################################

T3Variations = typ.Tuple[
    typ.Tuple[jnp.ndarray,...],  # basis_variations
    typ.Tuple[jnp.ndarray,...],  # tt_variations
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class T3TangentSpace:
    '''Representation of a Tucker tensor train tangent vector.

    Example for 4-tensor:
        tt_variations               = (dV1, dV2, dV3, dV4)
        basis_variations            = (dB1, dB2, dB3, dB4)

        left_orthogonal_tt_cores    = (P1,   P2,  P3,  P4)
        right_orthogonal_tt_cores   = (Q1,   Q2,  Q3,  Q4)
        up_orthogonal_tt_cores      = (R1,   R2,  R3,  R4)
        orthogonal_basis_cores      = (B1,   B2,  B3,  B4)

        Tangent vector v:
        v(x,y,z,w) = (dV1(B1 x) + R1(dB1 x)) . Q2(B2 y) . Q3(B3 z) . Q4(B4 w)
                   + P1(B1 x) . (dV2(B2 y) + R2(dB2 y)) . Q3(B3 z) . Q4(B4 w)
                   + P1(B1 x) . P2(B2 y) . (dV3(B3 z) + R3(dB3 z)) . Q4(B4 w)
                   + P1(B1 x) . P2(B2 y) . P3(B3 z) . (dV4(B4 w) + R4(dB4 w))
    '''
    orthogonal_basis_cores:     typ.Tuple[jnp.ndarray, ...] # B_xo B_yo = I_xy    B.shape = (n, N)
    left_orthogonal_tt_cores:   typ.Tuple[jnp.ndarray, ...] # P_iax P_iay = I_xy, P.shape = (rL, n, rR)
    right_orthogonal_tt_cores:  typ.Tuple[jnp.ndarray, ...] # Q_xaj Q_yaj = I_xy  Q.shape = (rL, n, rR)
    up_orthogonal_tt_cores:     typ.Tuple[jnp.ndarray, ...] # R_ixj R_iyj = I_xy  R.shape = (rL, n, rR)


    def __post_init__(me):
        assert(me.tt_ranks == tt_get_ranks(me.left_orthogonal_tt_cores))
        assert(me.tt_ranks == tt_get_ranks(me.right_orthogonal_tt_cores))
        assert(me.tt_ranks == tt_get_ranks(me.up_orthogonal_tt_cores))

        assert(me.tucker_ranks == tt_get_shape(me.left_orthogonal_tt_cores))
        assert(me.tucker_ranks == tt_get_shape(me.right_orthogonal_tt_cores))
        assert(me.tucker_ranks == tt_get_shape(me.up_orthogonal_tt_cores))

        for n, N, B in zip(me.tucker_ranks, me.shape, me.orthogonal_basis_cores):
            assert(B.shape == (n, N))

    @ft.cached_property
    def shape(me) -> typ.Tuple[int,...]:
        return tuple([int(B.shape[1]) for B in me.orthogonal_basis_cores])

    @ft.cached_property
    def tucker_ranks(me) -> typ.Tuple[int,...]:
        return tuple([int(B.shape[0]) for B in me.orthogonal_basis_cores])

    @ft.cached_property
    def tt_ranks(me) -> typ.Tuple[int,...]:
        return tt_get_ranks(me.left_orthogonal_tt_cores)

    @ft.cached_property
    def num_cores(me) -> int:
        return len(me.shape)

    @staticmethod
    def make(
            x: TuckerTensorTrain,
    ) -> 'T3TangentSpace':
        t3_check_correctness(x)
        num_cores = len(t3_get_shape(x))

        left_orthogonal_t3, _, _ = t3_svd(x)

        orthogonal_basis_cores, left_orthogonal_tt_cores = left_orthogonal_t3

        # Sweep right to left
        right_orthogonal_tt_cores = list(left_orthogonal_tt_cores)
        up_orthogonal_tt_cores = [None] * num_cores
        for ii in range(num_cores-1,-1,-1):
            A0_a_i_b = right_orthogonal_tt_cores[ii-1]
            B0_b_j_c = right_orthogonal_tt_cores[ii]

            U_a_x_b, _, _ = outer_svd_3tensor(B0_b_j_c)
            up_orthogonal_tt_cores[ii] = U_a_x_b

            if ii > 0:
                U_b_x, ss_x, B_x_j_c = right_svd_3tensor(B0_b_j_c)
                A_a_i_x = jnp.tensordot(A0_a_i_b, U_b_x * ss_x.reshape((1, -1)), axes=1)

                right_orthogonal_tt_cores[ii-1] = A_a_i_x
                right_orthogonal_tt_cores[ii] = B_x_j_c

        return T3TangentSpace(
            orthogonal_basis_cores,
            left_orthogonal_tt_cores,
            tuple(right_orthogonal_tt_cores),
            tuple(up_orthogonal_tt_cores),
        )

    def _check_consistency_with_other_ttt_cores(
            me,
            other_t3_cores: TuckerTensorTrain,
    ):
        t3_check_correctness(other_t3_cores)

        other_basis_cores, other_tt_cores = other_t3_cores

        other_tucker_ranks = tt_get_shape(other_tt_cores)
        if (me.tucker_ranks != other_tucker_ranks):
            raise RuntimeError('tucker ranks='+str(me.tucker_ranks)+', other tt shape='+str(other_tucker_ranks))

        other_tt_ranks = tt_get_ranks(other_tt_cores)
        if (me.tt_ranks != other_tt_ranks):
            raise RuntimeError('tt ranks='+str(me.tt_ranks)+', other tt ranks='+str(other_tt_ranks))

        other_shape = tuple([B.shape[1] for B in other_basis_cores])
        if (me.shape != other_shape):
            raise RuntimeError('shape='+str(me.shape)+', other shape='+str(other_shape))

    def add_tangent_vectors(
            me,
            u: T3Variations,
            v: T3Variations,
    ) -> T3Variations: # u + v
        me._check_consistency_with_other_ttt_cores(u)
        me._check_consistency_with_other_ttt_cores(v)
        return (
            tuple([dB_u + dB_v for dB_u, dB_v in zip(u[0], v[0])]),
            tuple([dG_u + dG_v for dG_u, dG_v in zip(u[1], v[1])]),
        )

    def subtract_tangent_vectors(
            me,
            u: T3Variations,
            v: T3Variations,
    ) -> T3Variations: # u - v
        me._check_consistency_with_other_ttt_cores(u)
        me._check_consistency_with_other_ttt_cores(v)
        return (
            tuple([dB_u - dB_v for dB_u, dB_v in zip(u[0], v[0])]),
            tuple([dG_u - dG_v for dG_u, dG_v in zip(u[1], v[1])]),
        )

    def scale_tangent_vector(
            me,
            scaling_factor,
            u: T3Variations,
    ) -> T3Variations: # s * u
        me._check_consistency_with_other_ttt_cores(u)
        return (
            tuple([scaling_factor * dB_u for dB_u in u[0]]),
            tuple([scaling_factor * dG_u for dG_u in u[1]]),
        )

    def gauged_inner_product(
            me,
            u_gauged: T3Variations, # Must satisfy TTT Gauge condition!
            v_gauged: T3Variations, # Must satisfy TTT Gauge condition!
    ): # (u_gauged, v_gauged)_HS, Hilbert-Schmidt inner product inherited from ambient space
        me._check_consistency_with_other_ttt_cores(u_gauged)
        me._check_consistency_with_other_ttt_cores(v_gauged)
        return (
            jnp.sum(jnp.array([jnp.sum(dB_u * dB_v) for dB_u, dB_v in zip(u_gauged[0], v_gauged[0])])) +
            jnp.sum(jnp.array([jnp.sum(dG_u * dG_v) for dG_u, dG_v in zip(u_gauged[1], v_gauged[1])]))
        )

    def gauged_norm(
            me,
            u_gauged: T3Variations, # Must satisfy TTT Gauge condition!
    ):
        me._check_consistency_with_other_ttt_cores(u_gauged)
        return jnp.sqrt(me.gauged_inner_product(u_gauged, u_gauged))

    def tangent_vector_to_tucker_tensor_train(
            me,
            u: T3Variations,
    ) -> TuckerTensorTrain:
        '''Rank 2r Tucker tensor train representation of tangent vector:
                u(x,y,z,w) = ([dU1(B x) P1(B x)]) ([Q2(B y)        0]) ([Q3(B z)        0]) ([Q4(B w) ])
                             (                  ) ([dU2(B y) P2(B y)]) ([dU3(B z) P3(B z)]) ([dU4(B w)])
                             (         +        ) (         +        ) (        +         ) (    +     )
                             ([R1(dB x)       0]) ([0              0]) ([0              0]) ([0       ])
                             (                  ) ([R2(dB y)       0]) ([R3(dB z)       0]) ([R4(dB w)])
        '''
        me._check_consistency_with_other_ttt_cores(u)
        basis_variations, tt_variations = u

        basis_cores = []
        for B, dB in zip(me.orthogonal_basis_cores, basis_variations):
            B2 = jnp.concatenate([B, dB], axis=0)
            basis_cores.append(B2)

        tt_cores = []

        dU = tt_variations[0]
        R = me.up_orthogonal_tt_cores[0]
        P = me.left_orthogonal_tt_cores[0]
        Z = jnp.zeros(P.shape)
        G_top = jnp.concatenate([dU, P], axis=2)
        G_bot = jnp.concatenate([R, Z], axis=2)
        G = jnp.concatenate([G_top, G_bot], axis=1)
        tt_cores.append(G)

        for ii in range(1, me.num_cores-1):
            P = me.left_orthogonal_tt_cores[ii]
            Q = me.right_orthogonal_tt_cores[ii]
            R = me.up_orthogonal_tt_cores[ii]
            dU = tt_variations[ii]
            Z001 = jnp.zeros((Q.shape[0], Q.shape[1], P.shape[2]))
            Z100 = jnp.zeros(Q.shape)
            Z101 = Z001
            Z111 = jnp.zeros(P.shape)
            G_top = jnp.concatenate([
                jnp.concatenate([Q, Z001], axis=2),
                jnp.concatenate([dU, P], axis=2)
            ], axis=0)
            G_bot = jnp.concatenate([
                jnp.concatenate([Z100, Z101], axis=2),
                jnp.concatenate([R, Z111], axis=2)
            ], axis=0)
            G = jnp.concatenate([G_top, G_bot], axis=1)
            tt_cores.append(G)

        dU = tt_variations[-1]
        Q = me.right_orthogonal_tt_cores[-1]
        R = me.up_orthogonal_tt_cores[-1]
        Z = jnp.zeros(Q.shape)
        G_top = jnp.concatenate([Q, dU], axis=0)
        G_bot = jnp.concatenate([Z, R], axis=0)
        G = jnp.concatenate([G_top, G_bot], axis=1)
        tt_cores.append(G)

        return tuple(basis_cores), tuple(tt_cores)

    def attached_tangent_vector_to_tucker_tensor_train(
            me,
            u: T3Variations,
    ) -> TuckerTensorTrain:
        '''Rank 2r tensor train representation of shifted tangent vector p + v,
        where the tail of the vector is attached at the base point on the manifold.

        Same as unshifted tangent vector v, except last core modified as follows:
        [Q4(B w) ]                  [Q4(B w)           ]
        [dU4(B w)]                  [P4(B w) + dU4(B w)]
            +             ->            +
        [0       ]                  [0                 ]
        [R4(dB w)]                  [R4(dB w)          ]
        '''
        me._check_consistency_with_other_ttt_cores(u)
        basis_variations, tt_variations = u
        basis_cores, tt_cores = me.tangent_vector_to_tucker_tensor_train(u)

        dU = tt_variations[-1]
        P = me.left_orthogonal_tt_cores[-1]
        Q = me.right_orthogonal_tt_cores[-1]
        R = me.up_orthogonal_tt_cores[-1]
        Z = jnp.zeros(Q.shape)
        G_top = jnp.concatenate([Q, P + dU], axis=0)
        G_bot = jnp.concatenate([Z, R], axis=0)
        G = jnp.concatenate([G_top, G_bot], axis=1)

        tt_cores = tt_cores[:-1] + (G,)

        return (basis_cores, tt_cores)

    def get_tt_term(
            me,
            u: T3Variations,
            ii: int
    ) -> TuckerTensorTrain:
        me._check_consistency_with_other_ttt_cores(u)
        term_tt_cores = (
                me.left_orthogonal_tt_cores[:ii] +
                (u[1][ii],) +
                me.right_orthogonal_tt_cores[ii+1:]
        )
        term_basis_cores = me.orthogonal_basis_cores
        return (term_basis_cores, term_tt_cores)

    def get_basis_term(
            me,
            u: T3Variations,
            ii: int
    ) -> TuckerTensorTrain:
        me._check_consistency_with_other_ttt_cores(u)
        term_tt_cores = (
                me.left_orthogonal_tt_cores[:ii] +
                (me.up_orthogonal_tt_cores[ii],) +
                me.right_orthogonal_tt_cores[ii+1:]
        )
        term_basis_cores = (
                me.orthogonal_basis_cores[:ii] +
                (u[0][ii],) +
                me.orthogonal_basis_cores[ii+1:]
        )
        return (term_basis_cores, term_tt_cores)

    def tangent_vector_to_dense(
            me,
            u: T3Variations,
    ) -> jnp.ndarray:
        T = jnp.zeros(me.shape)
        for ii in range(me.num_cores):
            T = T + t3_to_dense(me.get_tt_term(u, ii))
            T = T + t3_to_dense(me.get_basis_term(u, ii))
        return T

    def attached_tangent_vector_to_dense(
            me,
            u: T3Variations,
    ) -> jnp.ndarray:
        return t3_to_dense((me.orthogonal_basis_cores, me.left_orthogonal_tt_cores)) + me.tangent_vector_to_dense(u)

    def orthogonal_gauge_projection(
            me,
            u: T3Variations,
    ) -> T3Variations:
        '''Makes variations left-perpendicular by orthogonally projecting away the parallel components.
        Changes tangent vector.
        1) Make basis variations left-perpendicular to corresponding basis cores (all basis cores).
        2) Make tt variations left-perpendicular to corresponding tt cores (except last tt cores).
        '''
        me._check_consistency_with_other_ttt_cores(u)
        basis_variations, tt_variations = u

        new_tt_variations = []
        for dV, P in zip(tt_variations[:-1], me.left_orthogonal_tt_cores[:-1]):
            dV2 = dV - jnp.einsum('iaj,jk->iak', P, jnp.einsum('iaj,iak->jk', P, dV))
            new_tt_variations.append(dV2)
        new_tt_variations.append(tt_variations[-1])

        new_basis_variations = []
        for dB, B in zip(basis_variations, me.orthogonal_basis_cores):
            dB2 = dB - (dB @ B.T) @ B
            new_basis_variations.append(dB2)

        return tuple(new_basis_variations), tuple(new_tt_variations)

    def oblique_gauge_projection(
            me,
            u: T3Variations,
    ) -> T3Variations:
        '''Makes variations left-perpendicular.
        Preserves tangent vector.
        1) Make basis variations left-perpendicular by pushing remainder onto tt variations
        2) Make tt variations left-perpendicular by standard sweeping tt method
        '''
        me._check_consistency_with_other_ttt_cores(u)
        basis_variations, tt_variations = u

        tt_variations = list(tt_variations)
        basis_variations = list(basis_variations)

        # Make basis variations left-perpendicular
        for ii in range(me.num_cores):
            B_io = me.orthogonal_basis_cores[ii]
            dB_jo = basis_variations[ii]
            R_aib = me.up_orthogonal_tt_cores[ii]
            dG_ajb = tt_variations[ii]

            X_ji = dB_jo @ B_io.T
            dB_parallel_jo = X_ji @ B_io
            dB2_jo = dB_jo - dB_parallel_jo # dB_perp
            # dG2_ajb = dG_ajb + jnp.einsum('aib,ji->ajb', R_aib, X_ji)
            dG2_ajb = dG_ajb + jnp.einsum('aib,ij->ajb', R_aib, X_ji) # <-- Why is this correct?

            tt_variations[ii] = dG2_ajb
            basis_variations[ii] = dB2_jo

        # Make tt cores left-perpendicular
        for ii in range(me.num_cores-1):
            dV1 = tt_variations[ii]
            dV2 = tt_variations[ii+1]

            P1 = me.left_orthogonal_tt_cores[ii]
            Q2 = me.right_orthogonal_tt_cores[ii+1]
            X = jnp.einsum('iaj,iak->jk', P1, dV1)
            new_dV1 = dV1 - jnp.einsum('iaj,jk->iak', P1, X)
            new_dV2 = dV2 + jnp.einsum('jk,kbl->jbl', X, Q2)

            tt_variations[ii] = new_dV1
            tt_variations[ii+1] = new_dV2

        return tuple(basis_variations), tuple(tt_variations)

    def project_ttt_onto_tangent_space(
            me,
            x: TuckerTensorTrain,
    ) -> T3Variations:
        other_basis_cores, other_tt_cores = x

        other_tt_cores2 = []
        for G_other, B_other, B in zip(other_tt_cores, other_basis_cores, me.orthogonal_basis_cores):
            G_other2 = jnp.einsum('aib,ix->axb', G_other, B_other @ B.T)
            other_tt_cores2.append(G_other2)

        zipper_left2right = tt_zipper_left_to_right(other_tt_cores2, me.left_orthogonal_tt_cores)[:-1]
        zipper_right2left = tt_zipper_right_to_left(other_tt_cores2, me.right_orthogonal_tt_cores)[1:]

        ungauged_tt_variations = []
        ungauged_basis_variations = []
        for ZL_ax, ZR_by, G_aib, B_io, R0_xjy, B0_jo in zip(
                zipper_left2right, zipper_right2left,
                other_tt_cores, other_basis_cores,
                me.up_orthogonal_tt_cores, me.orthogonal_basis_cores,
        ):
            X_xiy = jnp.einsum('ax,aib,by->xiy', ZL_ax, G_aib, ZR_by)
            dG_xjy = jnp.einsum('xiy,ij->xjy', X_xiy, B_io @ B0_jo.T)
            M_ij = jnp.einsum('xiy,xjy->ij', X_xiy, R0_xjy)
            dB_jo = jnp.einsum('ij,io->jo', M_ij, B_io)

            ungauged_tt_variations.append(dG_xjy)
            ungauged_basis_variations.append(dB_jo)

        ungauged_u = (ungauged_basis_variations, ungauged_tt_variations)
        gauged_u = me.orthogonal_gauge_projection(ungauged_u)
        return gauged_u

    @jax.jit
    def retract(
            me,
            u: T3Variations,
    ) -> TuckerTensorTrain: # retracted Tucker tensor train
        me._check_consistency_with_other_ttt_cores(u)
        x = me.attached_tangent_vector_to_tucker_tensor_train(u)
        retracted_x, _, _ = t3_svd(
            x,
            forced_tt_ranks=me.tt_ranks[1:-1],
            forced_basis_ranks=me.tucker_ranks,
        )
        return retracted_x

    @ft.cached_property
    def data(me):
        return (
            me.orthogonal_basis_cores,
            me.left_orthogonal_tt_cores,
            me.right_orthogonal_tt_cores,
            me.up_orthogonal_tt_cores,
        )

    def tree_flatten(me):
        return (me.data, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


###############    UNIFORM TUCKER TENSOR TRAIN    ###############

def pack_matrices(
        MM: typ.Sequence[jnp.ndarray], # elm_shapes=[(n0, N0), (n1, N1), ..., (nk,Nk)]
) -> jnp.ndarray: # uniform_cores,   shape=(num_cores, n, N)
    for M in MM:
        assert(len(M.shape) == 2)

    n = np.max([M.shape[0] for M in MM])
    N = np.max([M.shape[1] for M in MM])

    padded_MM_list = []
    for M in MM:
        n0, N0 = M.shape
        pad = [(0, n - n0, 0), (0, N - N0, 0)]
        padded_M = jax.lax.pad(M, 0.0, pad)
        padded_MM_list.append(padded_M)

    uniform_MM = jnp.stack(padded_MM_list)
    return uniform_MM

def unpack_matrices(
        uniform_MM:     jnp.ndarray, # shape=(d, n, N)
        nn:             typ.Sequence[int], # len=d, (n1, ..., nd)
        NN:             typ.Sequence[int], # len=d, (N1, ..., Nd)
) -> typ.Tuple[jnp.ndarray,...]: # cores, len=d, elm_shape=(ni, Ni)
    d, n, N = uniform_MM.shape
    MM_list = []
    for ii, M in enumerate(uniform_MM):
        n0 = int(nn[ii])
        N0 = int(NN[ii])
        pad = [(0, n0 - n, 0), (0, N0 - N, 0)]
        M0 = jax.lax.pad(M, 0.0, pad)
        MM_list.append(M0)
    return tuple(MM_list)


UniformT3Variations = typ.Tuple[
    jnp.ndarray,  # basis_variations, shape=(d, n, N)
    jnp.ndarray,  # tt_variations, shape=(d, r, n, r)
]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UniformT3TangentSpace:
    orthogonal_basis_cores:     jnp.ndarray # shape=(d, n, N)
    left_orthogonal_tt_cores:   jnp.ndarray # shape=(d, r, n, r)
    right_orthogonal_tt_cores:  jnp.ndarray # shape=(d, r, n, r)
    up_orthogonal_tt_cores:     jnp.ndarray # shape=(d, r, n, r)
    original_shape:         typ.Tuple[int, ...] # len=d
    original_tucker_ranks:  typ.Tuple[int, ...]  # len=d
    original_tt_ranks:      typ.Tuple[int, ...]  # len=d+1

    def __post_init__(me):
        assert(me.orthogonal_basis_cores.shape == (me.d, me.n, me.N))
        assert(me.left_orthogonal_tt_cores.shape == (me.d, me.r, me.n, me.r))
        assert(me.right_orthogonal_tt_cores.shape == (me.d, me.r, me.n, me.r))
        assert(me.up_orthogonal_tt_cores.shape == (me.d, me.r, me.n, me.r))
        assert(len(me.original_shape) == me.d)
        assert(len(me.original_tucker_ranks) == me.d)
        assert(len(me.original_tt_ranks) == me.d+1)

    @ft.cached_property
    def d(me):
        return me.orthogonal_basis_cores.shape[0]

    @ft.cached_property
    def n(me):
        return me.orthogonal_basis_cores.shape[1]

    @ft.cached_property
    def r(me):
        return me.left_orthogonal_tt_cores.shape[1]

    @ft.cached_property
    def N(me):
        return me.orthogonal_basis_cores.shape[2]

    @staticmethod
    def from_nonuniform(TS: T3TangentSpace) -> 'UniformT3TangentSpace':
        orthogonal_basis_cores = pack_matrices(TS.orthogonal_basis_cores)
        left_orthogonal_tt_cores = pack_uniform_tensor_train(TS.left_orthogonal_tt_cores)
        right_orthogonal_tt_cores = pack_uniform_tensor_train(TS.right_orthogonal_tt_cores)
        up_orthogonal_tt_cores = pack_uniform_tensor_train(TS.up_orthogonal_tt_cores)

        return UniformT3TangentSpace(
            orthogonal_basis_cores,
            left_orthogonal_tt_cores,
            right_orthogonal_tt_cores,
            up_orthogonal_tt_cores,
            TS.shape, TS.tucker_ranks, TS.tt_ranks,
        )

    def to_nonuniform(me) -> T3TangentSpace:
        orthogonal_basis_cores = unpack_matrices(
            me.orthogonal_basis_cores, me.original_tucker_ranks, me.original_shape,
        )
        left_orthogonal_tt_cores = unpack_uniform_tensor_train(
            me.left_orthogonal_tt_cores, me.original_tt_ranks, me.original_tucker_ranks,
        )
        right_orthogonal_tt_cores = unpack_uniform_tensor_train(
            me.right_orthogonal_tt_cores, me.original_tt_ranks, me.original_tucker_ranks,
        )
        up_orthogonal_tt_cores = unpack_uniform_tensor_train(
            me.up_orthogonal_tt_cores, me.original_tt_ranks, me.original_tucker_ranks,
        )
        return T3TangentSpace(
            orthogonal_basis_cores,
            left_orthogonal_tt_cores,
            right_orthogonal_tt_cores,
            up_orthogonal_tt_cores,
        )

    def pack_uniform_variations(
            me,
            u: T3Variations,
    ) -> UniformT3Variations:
        assert(len(u[1]) == me.d)
        for ii, dU in enumerate(u[1]):
            ni = me.original_tucker_ranks[ii]
            rL = me.original_tt_ranks[ii]
            rR = me.original_tt_ranks[ii+1]
            assert(dU.shape == (rL, ni, rR))

        assert(len(u[0]) == me.d)
        for ii, dB in enumerate(u[0]):
            ni = me.original_tucker_ranks[ii]
            Ni = me.original_shape[ii]
            assert(dB.shape == (ni, Ni))

        uniform_dB = pack_matrices(u[0])
        uniform_dU = pack_uniform_tensor_train(u[1])

        return uniform_dB, uniform_dU

    def unpack_uniform_variations(
            me,
            U: UniformT3Variations,
    ) -> T3Variations:
        basis_variations = unpack_matrices(U[0], me.original_tucker_ranks, me.original_shape)
        tt_variations = unpack_uniform_tensor_train(U[1], me.original_tt_ranks, me.original_tucker_ranks)
        return basis_variations, tt_variations

    @ft.cached_property
    def data(me):
        return (
            me.orthogonal_basis_cores,
            me.left_orthogonal_tt_cores,
            me.right_orthogonal_tt_cores,
            me.up_orthogonal_tt_cores,
            me.original_shape,
            me.original_tucker_ranks,
            me.original_tt_ranks,
        )

    @ft.cached_property
    def static_data(me):
        return (
            me.original_shape,
            me.original_tucker_ranks,
            me.original_tt_ranks,
        )

    @ft.cached_property
    def traced_data(me):
        return (
            me.orthogonal_basis_cores,
            me.left_orthogonal_tt_cores,
            me.right_orthogonal_tt_cores,
            me.up_orthogonal_tt_cores,
        )

    @staticmethod
    def from_static_and_traced_data(
        static_data, traced_data,
    ):
        data = traced_data + static_data
        return UniformT3TangentSpace(*data)

    def tree_flatten(me):
        return (me.data, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def ut3_orthogonal_gauge_projection(
        U: UniformT3Variations,
        UTS: UniformT3TangentSpace,
) -> UniformT3Variations:
    '''Makes variations left-perpendicular by orthogonally projecting away the parallel components.
    Changes tangent vector.
    dV_L -> (I - P_L P_L^T) dV_L
    '''
    basis_variations, tt_variations = U

    d, r, n, _ = tt_variations.shape
    N = basis_variations.shape[-1]
    assert(tt_variations.shape == (d, r, n, r))
    assert(basis_variations.shape == (d, n, N))
    assert(UTS.N == N)
    assert(UTS.d == d)
    assert(UTS.r == r)
    assert(UTS.n == n)

    gauged_tt_variations_list = []
    for ii in range(d-1):
        P = UTS.left_orthogonal_tt_cores[ii, :, :, :]
        dV = tt_variations[ii,:,:,:]
        dV2 = dV - jnp.einsum('iaj,jk->iak', P, jnp.einsum('iaj,iak->jk', P, dV))
        gauged_tt_variations_list.append(dV2)
    gauged_tt_variations_list.append(tt_variations[-1,:,:,:])

    gauged_tt_variations = jnp.stack(gauged_tt_variations_list)

    gauged_basis_variations_list = []
    for ii in range(d):
        B = UTS.orthogonal_basis_cores[ii, :, :]
        dB = basis_variations[ii,:,:]
        dB2 = dB - (dB @ B.T) @ B
        gauged_basis_variations_list.append(dB2)

    gauged_basis_variations = jnp.stack(gauged_basis_variations_list)

    return gauged_basis_variations, gauged_tt_variations



@jax.jit
def ut3_orthogonal_gauge_projection_using_map(
        U: UniformT3Variations,
        orthogonal_basis_cores,
        left_orthogonal_tt_cores,
) -> UniformT3Variations:
    '''Makes variations left-perpendicular by orthogonally projecting away the parallel components.
    Changes tangent vector.
    dV_L -> (I - P_L P_L^T) dV_L
    '''
    basis_variations, tt_variations = U

    first_gauged_tt_variations = jax.lax.map(
        lambda P_dV: P_dV[1] - jnp.einsum('iaj,jk->iak', P_dV[0], jnp.einsum('iaj,iak->jk', P_dV[0], P_dV[1])),
        (left_orthogonal_tt_cores[:-1], tt_variations[:-1]),
    )
    last_gauged_tt_variation = tt_variations[-1,:,:,:]
    gauged_tt_variations = jnp.concatenate(
        [first_gauged_tt_variations, last_gauged_tt_variation.reshape((1,) + last_gauged_tt_variation.shape)],
        axis=0
    )

    gauged_basis_variations = jax.lax.map(
        lambda B_dB: B_dB[1] - (B_dB[1] @ B_dB[0].T) @ B_dB[0], (orthogonal_basis_cores, basis_variations),
    )

    return gauged_basis_variations, gauged_tt_variations


def get_t3_tangent_space_dim(
        shape: typ.Sequence[int], # len=d
        tucker_ranks: typ.Sequence[int], # len=d
        tt_ranks: typ.Sequence[int], # len=d+1
) -> int:
    num_cores = len(shape)
    assert(len(tucker_ranks) == num_cores)
    assert(len(tt_ranks) == num_cores+1)
    manifold_dim: int = 0
    for ii in range(num_cores):
        n = tucker_ranks[ii]
        rL = tt_ranks[ii]
        rR = tt_ranks[ii+1]
        if ii == num_cores-1:
            manifold_dim += rL * n * rR
        else:
            manifold_dim += (rL * n - rR) * rR

    for ii in range(num_cores):
        n = tucker_ranks[ii]
        N = shape[ii]
        manifold_dim += (N - n) * n

    return manifold_dim
