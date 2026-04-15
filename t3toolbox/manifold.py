# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.orthogonalization as orth
import t3toolbox.t3svd as t3svd
import t3toolbox.base_variation_format as bvf
import t3toolbox.uniform as ut3
from t3toolbox.common import *

__all__ = [
    # Tangent vectors
    'manifold_dim',
    'tangent_to_dense',
    'tangent_to_t3',
    'tangent_zeros',
    'tangent_randn',
    'absorb_weights_into_tangent_cores',
    # Projection and retraction
    'orthogonal_gauge_projection',
    'oblique_gauge_projection',
    'project_t3_onto_tangent_space',
    'retract',
]


####################################################################
##################    Tangent vectors operations  ##################
####################################################################

def manifold_dim(
        s = t3.T3Structure,
) -> int:
    """Get the dimension of the fixed rank T3 manifold with a given structure.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> s = ((15,16,13), (9,10,8), (2,7,6,3))
    >>> mdim = t3m.manifold_dim(s)
    >>> print(mdim)
    578

    In the following more detailed example, we verify that the manifold dim
    is correct by generating an excessive number of random dense tangent vectors
    and performing an SVD on them. The number of nonzero singular values is the
    dimension of the tangent space, which is the dimension of the manifold.

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.base_variation_format as bvf
    >>> import t3toolbox.orthogonalization as orth
    >>> s = ((5,6,3), (5,3,2), (2,2,4,1))
    >>> mdim = t3m.manifold_dim(s)
    >>> print(mdim)
    29
    >>> p = t3.t3_corewise_randn(s)
    >>> base, _ = orth.orthogonal_representations(p)
    >>> tucker_shapes, tt_shapes = bvf.get_base_hole_shapes(base)
    >>> num_tucker_entries = np.sum([np.prod(shape) for shape in tucker_shapes])
    >>> num_tt_entries = np.sum([np.prod(shape) for shape in tt_shapes])
    >>> num_core_entries = num_tucker_entries + num_tt_entries
    >>> print(num_core_entries)
    80
    >>> vv = [t3m.tangent_randn(base, apply_gauge_projection=False) for _ in range(num_core_entries)]
    >>> dense_vv = np.stack([t3m.tangent_to_dense(v, base) for v in vv])
    >>> _, ss, _ = np.linalg.svd(dense_vv.reshape((num_core_entries,-1)), full_matrices=False)
    >>> print(ss[mdim-1]) # last nonzero singular value
    2.8197268462367813
    >>> print(ss[mdim]) # first zero singular value
    1.1933078683104488e-14
    """
    shape = s[0]
    min_tucker_ranks, min_tt_ranks = t3.compute_minimal_ranks(s)

    num_cores = len(shape)
    assert(len(min_tucker_ranks) == num_cores)
    assert(len(min_tt_ranks) == num_cores+1)
    manifold_dim: int = 0
    for ii in range(num_cores):
        n = min_tucker_ranks[ii]
        rL = min_tt_ranks[ii]
        rR = min_tt_ranks[ii+1]
        if ii == num_cores-1:
            manifold_dim += rL * n * rR
        else:
            manifold_dim += (rL * n - rR) * rR

    for ii in range(num_cores):
        n = min_tucker_ranks[ii]
        N = shape[ii]
        manifold_dim += (N - n) * n

    return manifold_dim


def tangent_to_dense(
        variation: bvf.T3Variation,
        base: bvf.T3Base,
        include_shift: bool = False, # False: V. True: P+V. P=base point, V=tangent vector. Must supply "rep"
        xnp = np,
) -> NDArray:
    """Convert Tangent vector to Tucker tensor train manifold into dense tensor.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.base_variation_format as bvf
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> v_dense = t3m.tangent_to_dense(variation, base) # Convert tangent to dense
    >>> ((U0,U1,U2), (L0,L1,L2), (R0,R1,R2), (O0,O1,O2)) = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = variation
    >>> s1 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,H0,R1,R2)
    >>> s2 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,L0,H1,R2)
    >>> s3 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,L0,L1,H2)
    >>> s4 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', V0,U1,U2,O0,R1,R2)
    >>> s5 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,V1,U2,L0,O1,R2)
    >>> s6 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,V2,L0,L1,O2)
    >>> v_dense2 = s1 + s2 + s3 + s4 + s5 + s6
    >>> print(np.linalg.norm(v_dense - v_dense2))
    1.2760924630140578e-14
    >>> p_plus_v_dense = t3m.tangent_to_dense(variation, base, include_shift=True) # Convert shifted tangent, p+v, to dense
    >>> p_plus_v_dense2 =  t3.t3_to_dense(p) + v_dense
    >>> print(np.linalg.norm(p_plus_v_dense - p_plus_v_dense2))
    1.2677102046134292e-12
    """
    num_cores = len(variation[0])
    tucker_terms = [bvf.ith_bv_to_t3(ii, False, base, variation) for ii in range(num_cores)]
    tt_terms     = [bvf.ith_bv_to_t3(ii, True, base, variation) for ii in range(num_cores)]
    terms = tucker_terms + tt_terms
    V = t3.t3_to_dense(terms[0])
    for t in terms[1:]:
        V = V + t3.t3_to_dense(t, xnp=xnp)

    if include_shift:
        tucker_cores, left_tt_cores, _, _ = base
        P = t3.t3_to_dense((tucker_cores, left_tt_cores))
        X = P + V
    else:
        X = V

    return X


def tangent_to_t3(
        variation: bvf.T3Variation,
        base: bvf.T3Base,
        include_shift: bool = False,  # False: v. True: p+v. p=base point, v=tangent vector
        xnp = np,
) -> t3.TuckerTensorTrain:
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
    variation: T3Variation,
        Variation representing the tangent vector
    base: T3Base,
        Representation of the base point at which the tangent space attaches to the manifold.
    include_shift: bool
        If False, return tangent vector v only. If True, shift tangent vector so it is attached at the base point, p+v.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    TuckerTensorTrain
        Tucker tensor train representation of tangent vector, which has doubled ranks

    See Also
    --------
    T3Base
    T3Variation
    TuckerTensorTrain

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> v_t3 = t3m.tangent_to_t3(variation, base) # tangent vector only (attached at zero)
    >>> v_dense = t3.t3_to_dense(v_t3)
    >>> v_dense2 = t3m.tangent_to_dense(variation, base)
    >>> print(np.linalg.norm(v_dense - v_dense2))
    2.678565538404836e-15
    >>> p_plus_v_t3 = t3m.tangent_to_t3(variation, base, include_shift=True) # shifted tangent vector (include attachment at base point)
    >>> p_plus_v_dense = t3.t3_to_dense(p_plus_v_t3)
    >>> p_plus_v_dense2 = v_dense2 + t3.t3_to_dense(p)
    >>> print(np.linalg.norm(p_plus_v_dense - p_plus_v_dense2))
    1.2102169224182523e-12
    '''
    tucker_vars, tt_vars = variation
    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base

    num_cores = len(tucker_cores)

    x_tucker_cores = []
    for B, dB in zip(tucker_cores, tucker_vars):
        B2 = xnp.concatenate([B, dB], axis=0)
        x_tucker_cores.append(B2)

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
    R = right_tt_cores[-1]
    O = outer_tt_cores[-1]
    Z = xnp.zeros((R.shape[0], O.shape[1], R.shape[2]))
    if include_shift:
        Lf = left_tt_cores[-1]
        G_top = xnp.concatenate([R, Lf + dU], axis=0)
    else:
        G_top = xnp.concatenate([R, dU], axis=0)
    G_bot = xnp.concatenate([Z, O], axis=0)
    G = xnp.concatenate([G_top, G_bot], axis=1)
    x_tt_cores.append(G)

    return tuple(x_tucker_cores), tuple(x_tt_cores)


def tangent_zeros(
        base: bvf.T3Base, # orthogonal base
        xnp = np,
) -> bvf.T3Variation:
    """Construct the zero vector in a Tucker tensor train tangent space.

    Parameters
    ----------
    base: T3Base
        Representations of base point on manifold where tangent space is attached
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    T3Variation
        Variation representing the zero vector in the tangent space

    See Also
    --------
    t3tangent_randn

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> z = t3m.tangent_zeros(base)
    >>> print(np.linalg.norm(t3m.tangent_to_dense(z, base)))
    0.0
    """
    var_tucker_shapes, var_tt_shapes = bvf.get_base_hole_shapes(base)

    tucker_vars = tuple([xnp.zeros(s) for s in var_tucker_shapes])
    tt_vars = tuple([xnp.zeros(s) for s in var_tt_shapes])

    zero_variation = (tucker_vars, tt_vars)
    return zero_variation


def absorb_weights_into_tangent_cores(
        variation:      typ.Union[bvf.T3Variation,      ut3.UniformT3Variation],
        base:           typ.Union[bvf.T3Base,           ut3.UniformT3Base],
        edge_weights:   typ.Union[bvf.BVEdgeWeights,    ut3.UniformEdgeWeights] = (None, None, None, None),
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Union[bvf.T3Variation,      ut3.UniformT3Variation], # weighted variation
    typ.Union[bvf.T3Base,           ut3.UniformT3Base], # weighted base
]:
    """Contract edge weights with neighboring cores in base-variation representation.

    Tensor network diagrams illustrating groupings::

             ____     ________     ____
            /    \   /        \   /    \
        1---w---L0---w---H1---w---R2---w---1
                |        |        |
              / w      / w      / w
              | |      | |      | |
              | U0     | U1     | U2
              | |      | |      | |
              \ w      \ w      \ w
                |        |        |

    and::

             ____     ________     ____
            /    \   /        \   /    \
        1---w---L0---w---O1---w---R2---w---1
                |        |        |
              / w      / w      / w
              | |      | |      | |
              | U0     | V1     | U2
              | |      | |      | |
              \ w      \ w      \ w
                |        |        |

    """
    is_uniform = not isinstance(base[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    (shape_weights,
     up_tucker_weights, outer_tucker_weights,
     left_tt_weights, right_tt_weights,
     ) = edge_weights

    (up_tucker_cores0, left_tt_cores0, right_tt_cores0, outer_tt_cores0) = base
    (var_tucker_cores0, var_tt_cores0) = variation

    if is_uniform:
        up_tucker_cores = xnp.einsum(
            'di,dio,do->dio', up_tucker_weights, up_tucker_cores0, shape_weights
        )
        var_tucker_cores = xnp.einsum(
            'di,dio,do->dio', outer_tucker_weights, var_tucker_cores0, shape_weights
        )
        left_tt_cores = xnp.einsum(
            'di,diaj->diaj', left_tt_weights, left_tt_cores0
        )
        right_tt_cores = xnp.einsum(
            'diaj,dj->diaj', right_tt_cores0, right_tt_weights
        )
        outer_tt_cores = xnp.einsum(
            'di,diaj,dj->diaj', left_tt_weights, outer_tt_cores0, right_tt_weights
        )
        var_tt_cores = xnp.einsum(
            'di,diaj,dj->diaj', left_tt_weights, var_tt_cores0, right_tt_weights
        )

    else:
        (up_tucker_cores,) = xmap(
            lambda x: (xnp.einsum('i,io,o->io', x[0], x[1], x[2]),),
            (up_tucker_weights, up_tucker_cores0, shape_weights)
        )
        (var_tucker_cores,) = xmap(
            lambda x: (xnp.einsum('i,io,o->io', x[0], x[1], x[2]),),
            (outer_tucker_weights, var_tucker_cores0, shape_weights)
        )
        (left_tt_cores,) = xmap(
            lambda x: (xnp.einsum('i,iaj->iaj', x[0], x[1]),),
            (left_tt_weights, left_tt_cores0)
        )
        (right_tt_cores,) = xmap(
            lambda x: (xnp.einsum('iaj,j->iaj', x[0], x[1]),),
            (right_tt_cores0, right_tt_weights)
        )
        (outer_tt_cores,) = xmap(
            lambda x: (xnp.einsum('i,iaj,j->iaj', x[0], x[1], x[2]),),
            (left_tt_weights, outer_tt_cores0, right_tt_weights)
        )
        (var_tt_cores,) = xmap(
            lambda x: (xnp.einsum('i,iaj,j->iaj', x[0], x[1], x[2]),),
            (left_tt_weights, var_tt_cores0, right_tt_weights)
        )

    weighted_base = (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    weighted_variation = (var_tucker_cores, var_tt_cores)
    return weighted_variation, weighted_base


def tangent_randn(
        base:   typ.Union[bvf.T3Base,               ut3.UniformT3Base], # orthogonal base
        masks:  typ.Union[ut3.UniformEdgeWeights,   ut3.UniformBVEdgeWeights] = (None, None, None),
        apply_gauge_projection: bool = True,
        randn: typ.Callable[..., NDArray] = np.random.randn,
        use_jax: bool = False,
) -> typ.Union[bvf.T3Variation, ut3.UniformT3Variation]:
    """Draw a random T3Variation.

    Parameters
    ----------
    orthogonal_base: T3Base
        Representations of base point on manifold where tangent space is attached.
    randn: typ.Callable[[..., NDArray]
        Function for creating random arrays. Arguments are a sequence of ints defining the shape of the array.
        Default: np.random.randn (numpy)

    Returns
    -------
    T3Tangent
        Random tangent vector. If base is orthogonal, ranks are minimal, and gauge projection is applied,
        then the random tangent vector is distributed according to
        a standard multivariate distribution on the tangent space.
    use_jax: bool
        If True, return jax arrays, if False return numpy. Should update this to use pure jax, rather than converting numpy->jax.
    apply_gauge_projection: bool
        Default: True. If False, gauge projection is not applied and vector is not i.i.d. N(0,1) on the tangent space

    See Also
    --------
    t3tangent_zeros

    Examples
    --------

    Apply Gauge projection (default):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, vars0 = orth.orthogonal_representations(p)
    >>> v = t3m.tangent_randn(base) # Random tangent vector, gauged.

    Don't apply Gauge projection:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, vars0 = orth.orthogonal_representations(p)
    >>> v = t3m.tangent_randn(base, apply_gauge_projection=False) # Random tangent vector, ungauged
    """
    is_uniform = not isinstance(base[0], typ.Sequence)

    if is_uniform:
        var_tucker_shape, var_tt_shape = ut3.get_uniform_base_hole_shapes(base)
        var_tucker_supercore = randn(*var_tucker_shape)
        var_tt_supercore = randn(*var_tt_shape)

        variation = (var_tucker_supercore, var_tt_supercore)
        if masks is not None:
            variation = ut3.apply_masks_to_variation(variation, masks, use_jax=use_jax)

        if apply_gauge_projection:
            variation = orthogonal_gauge_projection(variation, base, use_jax=use_jax)
    else:
        var_tucker_shapes, var_tt_shapes = bvf.get_base_hole_shapes(base)

        tucker_vars0 = tuple([randn(*s) for s in var_tucker_shapes])
        tt_vars0 = tuple([randn(*s) for s in var_tt_shapes])

        variation = (tucker_vars0, tt_vars0)
        if apply_gauge_projection:
            variation = orthogonal_gauge_projection(variation, base, use_jax=use_jax)

    return variation


####################################################################
#################    Projection and retraction   ###################
####################################################################

def orthogonal_gauge_projection(
        variation:          typ.Union[bvf.T3Variation,  ut3.UniformT3Variation],
        orthogonal_base:    typ.Union[bvf.T3Base,       ut3.UniformT3Base],
        use_jax: bool = False,
) -> typ.Union[bvf.T3Variation, ut3.UniformT3Variation]:
    """Makes tangent variation gauged via orthogonal projection. Changes tangent vector.

    Gauge condition:
        - All variation Tucker cores Vi are orthogonal to the corresponding base Tucker cores Ui:
            Ui @ Vi.T = 0    for    i=1,...,d
        - All but the last variation TT-cores H are left-perpendicular to the corresponding base left TT-cores L:
            einsum('iaj,iak->jk', Hi, Li) = 0    for    i=1,...,d-1

    Parameters
    ----------
    variation: T3Variation,
        The variation which will become gauged.
    orthogonal_base: T3Base,
        The base representations. Must be orthogonal for the operation to work properly.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    T3Variation
        Projected variation satisfying Gauge condition.
        Represents different tangent vector than original variation.

    See Also
    --------
    T3Base
    T3Variation
    t3_oblique_gauge_projection

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.common as common
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.corewise as cw
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base, apply_gauge_projection=False)
    >>> proj_variation = t3m.orthogonal_gauge_projection(variation, base) # Make gauged via orthogonal projection
    >>> (U0,U1,U2), (L0,L1,L2), _, _ = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = proj_variation
    >>> print(np.linalg.norm(V1 @ U1.T)) # Gauge condition for tucker core 1
    3.512073125137391e-15
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', H1, L1))) # Gauge condition for TT-core 1
    1.5807940730805242e-15
    >>> v_minus_p_dot_p = cw.corewise_dot(cw.corewise_sub(variation, proj_variation), proj_variation)
    >>> print(v_minus_p_dot_p) # Projection is orthogonal w.r.t. corewise dot
    -4.995303314442243e-18

    Uniform example:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.common as common
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.uniform as ut3
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, dummy_var = orth.orthogonal_representations(p)
    >>> _, uniform_base, masks = ut3.bv_to_ubv(dummy_var, base)
    >>> uniform_var = t3m.tangent_randn(uniform_base, masks=masks, apply_gauge_projection=False)
    >>> proj_var = t3m.orthogonal_gauge_projection(uniform_var, uniform_base)
    >>> UU, LL, RR, OO = uniform_base
    >>> proj_tucker_var, proj_tt_var = proj_var
    >>> print(np.linalg.norm(np.einsum('dio,djo->dij', proj_tucker_var, UU)))
    6.860678066865219e-15
    >>> print(np.linalg.norm(np.einsum('diaj,diak->djk', proj_tt_var[:-1], LL[:-1]))) # first var tt cores are left-orthogonal to base
    2.0607190172353126e-15
    >>> ip = cw.corewise_dot(cw.corewise_sub(uniform_var, proj_var), proj_var)
    >>> print(ip) # Projection is orthogonal w.r.t. corewise dot
    4.496403249731884e-14
    """
    is_uniform = not isinstance(orthogonal_base[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    tucker_vars, tt_vars = variation

    if is_uniform:
        first_dV2 = tt_vars[:-1] - xnp.einsum(
            'diaj,djk->diak',
            left_tt_cores[:-1],
            xnp.einsum('diaj,diak->djk', left_tt_cores[:-1], tt_vars[:-1])
        )
        last_dV2 = tt_vars[-1]
        new_tt_variations = xnp.concatenate([first_dV2, last_dV2.reshape((1,)+last_dV2.shape)], axis=0)

        print('tucker_cores.shape=', tucker_cores.shape)
        print('tucker_vars.shape=', tucker_vars.shape)

        new_tucker_variations = tucker_vars - xnp.einsum(
            'dio,dij->djo',
            tucker_cores,
            xnp.einsum('dio,djo->dij', tucker_cores, tucker_vars)
        )

    else:
        new_tt_variations = []
        for dV, P in zip(tt_vars[:-1], left_tt_cores[:-1]):
            dV2 = dV - xnp.einsum('iaj,jk->iak', P, xnp.einsum('iaj,iak->jk', P, dV))
            new_tt_variations.append(dV2)
        new_tt_variations.append(tt_vars[-1])

        new_tucker_variations = []
        for dB, B in zip(tucker_vars, tucker_cores):
            dB2 = dB - (dB @ B.T) @ B
            new_tucker_variations.append(dB2)

        new_tucker_variations = tuple(new_tucker_variations)
        new_tt_variations = tuple(new_tt_variations)

    return new_tucker_variations, new_tt_variations


def oblique_gauge_projection(
        variation: bvf.T3Variation,
        orthogonal_base: bvf.T3Base,
        xnp = np,
) -> bvf.T3Variation:
    """Makes variations left-perpendicular while preserving tangent vector.

    Straightforward generalization of the method from:
        Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider.
        "On manifolds of tensors of fixed TT-rank." Numerische Mathematik 120.4 (2012): 701-731.

    Parameters
    ----------
    variation: T3Variation,
        The variation that we wish to make gauged
    orthogonal_base: T3Base,
        Orthogonal representations of the base point on the manifold.
        If non-orthogonal, this method doesn't work properly.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    T3Variation
        Projected variation satisfying Gauge condition.
        Represents the same tangent vector as the original variation.

    See Also
    --------
    T3Base
    T3Variation
    orthogonal_gauge_projection

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> proj_variation = t3m.oblique_gauge_projection(variation, base) # Make gauged via oblique projection
    >>> v_dense = t3m.tangent_to_dense(variation, base)
    >>> proj_v_dense = t3m.tangent_to_dense(proj_variation, base)
    >>> print(np.linalg.norm(v_dense - proj_v_dense)) # Zero since projection preserves represented tangent vector
    3.4398319441148304e-15
    >>> (U0,U1,U2), (L0,L1,L2), _, _ = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = proj_variation
    >>> print(np.linalg.norm(V1 @ U1.T)) # Gauge condition for Tucker core 1
    2.931519226677228e-15
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', H1, L1))) # Gauge condition for TT-core 1
    6.99005312491287e-16

    With minimal ranks, orthogonal bases, and gauged variations, the corewise dot product faithfully represents
    the Hilbert-Schmidt inner product on the ambient space:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.common
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.corewise as cw
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> u = t3m.tangent_randn(base, apply_gauge_projection=False)
    >>> v = t3m.tangent_randn(base, apply_gauge_projection=False)
    >>> bad_u_inner_v = cw.corewise_dot(u, v) # u and v are ungauged, so this will not give the right answer
    >>> u_dense = t3m.tangent_to_dense(u, base)
    >>> v_dense = t3m.tangent_to_dense(v, base)
    >>> u_inner_v_true = np.sum(u_dense * v_dense)
    >>> print(np.abs(bad_u_inner_v - u_inner_v_true)) # error nonzero because we didn't respect gauge
    6.21838915941413
    >>> u_gauged = t3m.oblique_gauge_projection(u, base) # make them gauged and try again
    >>> v_gauged = t3m.oblique_gauge_projection(v, base)
    >>> u_inner_v = cw.corewise_dot(u_gauged, v_gauged)
    >>> print(np.abs(u_inner_v - u_inner_v_true)) # Now the error is numerical zero
    0.0
    """
    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    tucker_vars, tt_vars = variation
    num_cores = len(tucker_cores)

    tt_vars = list(tt_vars)
    tucker_vars = list(tucker_vars)

    # Make Tucker variations left-perpendicular
    for ii in range(num_cores):
        B_io = tucker_cores[ii]
        dB_jo = tucker_vars[ii]
        O_aib = outer_tt_cores[ii]
        dG_ajb = tt_vars[ii]

        X_ji = dB_jo @ B_io.T
        dB_parallel_jo = X_ji @ B_io
        dB2_jo = dB_jo - dB_parallel_jo # dB_perp
        dG2_ajb = dG_ajb + xnp.einsum('aib,ij->ajb', O_aib, X_ji)

        tt_vars[ii] = dG2_ajb
        tucker_vars[ii] = dB2_jo

    # Make tt cores left-perpendicular
    for ii in range(num_cores-1):
        dG1 = tt_vars[ii]
        dG2 = tt_vars[ii+1]

        L1 = left_tt_cores[ii]
        R2 = right_tt_cores[ii+1]
        X = xnp.einsum('iaj,iak->jk', L1, dG1)
        new_dV1 = dG1 - xnp.einsum('iaj,jk->iak', L1, X)
        new_dV2 = dG2 + xnp.einsum('jk,kbl->jbl', X, R2)

        tt_vars[ii] = new_dV1
        tt_vars[ii+1] = new_dV2

    return tuple(tucker_vars), tuple(tt_vars)


def tt_zipper_left_to_right(
        coresA: typ.Sequence[NDArray],
        coresB: typ.Sequence[NDArray],
        xnp = np,
) -> typ.Tuple[NDArray, ...]:  # zipper_matrices. len=num_cores+1
    zipper_matrices = [xnp.array([[1.0]])]
    for GA, GB in zip(coresA, coresB):
        Z_prev = zipper_matrices[-1]
        Z = xnp.einsum('ij,iak,jal->kl', Z_prev, GA, GB)
        zipper_matrices.append(Z)
    return tuple(zipper_matrices)


def tt_zipper_right_to_left(
        coresA: typ.Sequence[NDArray],
        coresB: typ.Sequence[NDArray],
        xnp = np,
) -> typ.Tuple[NDArray, ...]:  # zipper_matrices. len=num_cores+1
    return tt_zipper_left_to_right(t3.reverse_tt(coresA), t3.reverse_tt(coresB), xnp=xnp)[::-1]


def project_t3_onto_tangent_space(
        x: t3.TuckerTensorTrain, # Tucker tensor train to be projected
        orthogonal_base: bvf.T3Base, # Orthogonal representations of base point
        xnp = np,
) -> bvf.T3Variation:
    """Projects TuckerTensorTrain onto tangent space to the manifold of fixed rank TuckerTensorTrains.

    Parameters
    ----------
    x: t3.TuckerTensorTrain
        TuckerTensorTrain to project
    orthogonal_base: T3Base
        Minimal rank orthogonal representations of base point on manifold where tangent space is attached
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    T3Variation
        Gauged variation representing the orthogonal projection of x onto the tangent space.

    See Also
    --------
    T3Base
    oblique_gauge_projection
    orthogonal_gauge_projection

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> x = t3.t3_corewise_randn(((14,15,16), (7,4,8), (2,5,4,2)))
    >>> proj_x = t3m.project_t3_onto_tangent_space(x, base) # Project x onto tangent space
    >>> P = t3.t3_to_dense(p)
    >>> X = t3.t3_to_dense(x)
    >>> proj_X = t3m.tangent_to_dense(proj_x, base)
    >>> print(np.sum((X - proj_X) * (proj_X - P)) / np.sum(X)) # Check that x was projected orthogonally
    -2.7295025395842007e-13
    """
    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    other_tucker_cores, other_tt_cores = x

    base_shape = tuple([B.shape[1] for B in tucker_cores])
    other_shape = tuple([B.shape[1] for B in other_tucker_cores])
    if base_shape != other_shape:
        raise RuntimeError(
            'Attempted to retract TuckerTensorTrain with wrong shape onto tangent space.\n'
            + str(base_shape) + ' = base_shape != other_shape = ' + str(other_shape)
        )

    other_tt_cores2 = []
    for G_other, B_other, B in zip(other_tt_cores, other_tucker_cores, tucker_cores):
        G_other2 = xnp.einsum('aib,ix->axb', G_other, B_other @ B.T)
        other_tt_cores2.append(G_other2)

    zipper_left2right = tt_zipper_left_to_right(other_tt_cores2[:-1], left_tt_cores)
    zipper_right2left = tt_zipper_right_to_left(other_tt_cores2[1:], right_tt_cores)

    ungauged_tt_variations = []
    ungauged_tucker_variations = []
    for ZL_ax, ZR_by, G_aib, B_io, R0_xjy, B0_jo in zip(
            zipper_left2right, zipper_right2left,
            other_tt_cores, other_tucker_cores,
            outer_tt_cores, tucker_cores,
    ):
        X_xiy = xnp.einsum('ax,aib,by->xiy', ZL_ax, G_aib, ZR_by)
        dG_xjy = xnp.einsum('xiy,ij->xjy', X_xiy, B_io @ B0_jo.T)
        M_ij = xnp.einsum('xiy,xjy->ij', X_xiy, R0_xjy)
        dB_jo = xnp.einsum('ij,io->jo', M_ij, B_io)

        ungauged_tt_variations.append(dG_xjy)
        ungauged_tucker_variations.append(dB_jo)

    ungauged_u = (ungauged_tucker_variations, ungauged_tt_variations)
    gauged_u = orthogonal_gauge_projection(ungauged_u, orthogonal_base)
    return gauged_u


def retract(
        variation: bvf.T3Variation,
        base: bvf.T3Base,
        xnp = np,
) -> t3.TuckerTensorTrain: # retracted Tucker tensor train
    """Retract Tucker tensor train tangent vector to manifold.

    Parameters
    ----------
    variation: T3Variation,
        Variation representing the tangent vector we wish to retract to the manifold
    base: T3Base,
        Representation of the base point on the manifold where the tangent space is attached.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    t3.TuckerTensorTrain
        Retraction of tangent vector onto the manifold.

    See Also
    --------
    T3Base
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.common
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.corewise as cw
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base) # Random tangent vector
    >>> ret_v = t3m.retract(variation, base) # Retract tangent vector to manifold
    >>> ret_V = t3.t3_to_dense(ret_v)
    >>> V = t3m.tangent_to_dense(variation, base, include_shift=True)
    >>> print(np.linalg.norm(ret_V - V)) # vector changes
    0.14335564543255402
    >>> v2 = cw.corewise_scale(variation, 1e-2) # make the tangent vector shorter for smaller retraction
    >>> ret_v2 = t3m.retract(v2, base)
    >>> ret_V2 = t3.t3_to_dense(ret_v2)
    >>> V2 = t3m.tangent_to_dense(v2, base, include_shift=True)
    >>> print(np.linalg.norm(ret_V2 - V2)) # vector changes
    4.9488133126395654e-05
    """
    tucker_cores, left_tt_cores, _, _ = base
    _, base_tucker_ranks, base_tt_ranks = t3.get_structure((tucker_cores, left_tt_cores))

    x_t3 = tangent_to_t3(variation, base, include_shift=True, xnp=xnp)
    retracted_x_t3, _, _ = t3svd.t3_svd(
        x_t3,
        max_tt_ranks = base_tt_ranks,
        max_tucker_ranks = base_tucker_ranks,
        xnp=xnp,
    )
    return retracted_x_t3

