# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
from __future__ import annotations

import typing as typ
import numpy as np

from t3toolbox.backend.common import *

__all__ = [
    'absorb_weights_into_tangent_cores',
]


# NOTE (parked): kept here for safekeeping pending a redesign of weighted tensor networks.
# This is the pre-refactor implementation. It uses the OLD base-core ordering
# (up, left, right, outer) and is NOT wired into the manifold / T3Tangent API. Do not rely
# on it until the weighting code structure is reworked.
def absorb_weights_into_tangent_cores(
        variation,      # (tucker_variations, tt_variations)
        base,           # OLD order: (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
        edge_weights = (None, None, None, None),
        use_jax: bool = False,
):
    """Contract edge weights with neighboring cores in base-variation representation.

    Tensor network diagrams illustrating groupings::

             ____     ________     ____
            /    \\   /        \\   /    \\
        1---wL--L0---wL--H1---wR--R2---wR--1
                |        |        |
              / wU     / wU     / wU
              | |      | |      | |
              | U0     | U1     | U2
              | |      | |      | |
              \\ w      \\ w      \\ w
                |        |        |

    and::

             ____     ________     ____
            /    \\   /        \\   /    \\
        1---wL--L0---wL--O1---wR--R2---wR--1
                |        |        |
              / wU     / wO     / wU
              | |      | |      | |
              | U0     | V1     | U2
              | |      | |      | |
              \\ w      \\ w      \\ w
                |        |        |

    """
    is_uniform = not isinstance(base[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
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
