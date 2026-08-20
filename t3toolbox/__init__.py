# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""T3Toolbox: Tucker tensor trains (T3) -- a Tucker decomposition whose central core is
stored as a tensor train.

This package root re-exports the **frontend** surface: the tensor classes (ragged and
uniform), the frame/variations/tangent classes, the geometry singletons, the Gauss-Newton
fitting models, and the optimizers. **Backend users import submodules explicitly** (e.g.
``from t3toolbox.backend import probing``) -- the backend is namespaced by module and
deliberately not re-exported here. Naming conventions: ``docs/naming_conventions.md``.
"""
from t3toolbox.tucker_tensor_train import (
    TuckerTensorTrain,
    T3Weights,
    t3_absorb_weights,
    t3_weighted_norm,
    t3_weighted_inner,
)
from t3toolbox.uniform_tucker_tensor_train import (
    UniformTuckerTensorTrain,
    UT3Weights,
    ut3_absorb_weights,
    ut3_weighted_norm,
    ut3_weighted_inner,
)
from t3toolbox.frame_variations_format import (
    T3Frame, T3Variations, t3_orthogonal_representations, T3FrameWeights, fv_absorb_weights,
)
from t3toolbox.uniform_frame_variations_format import (
    UT3Frame, UT3Variations, ut3_orthogonal_representations, UT3FrameWeights, ufv_absorb_weights,
)
from t3toolbox.manifold import T3Tangent, MANIFOLD, COREWISE
from t3toolbox.uniform_manifold import UT3Tangent, UNIFORM_MANIFOLD, UNIFORM_COREWISE
from t3toolbox.shared_geometry import shared, shared_manifold, shared_corewise
from t3toolbox.fitting import (
    GaussNewtonModel,
    UniformGaussNewtonModel,
    apply_model,
    entries_model,
    probe_model,
    apply_derivatives_model,
    entries_derivatives_model,
    probe_derivatives_model,
)
from t3toolbox.optimizers import gradient_descent, mc_sgd, adam, newton_cg
from t3toolbox import safety
from t3toolbox.safety import safe, unsafe

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version('t3toolbox')
except Exception:  # not installed (e.g. PYTHONPATH use) -- keep in sync with pyproject.toml
    __version__ = '2026.1.0'

__all__ = [
    # tensors
    'TuckerTensorTrain',
    'UniformTuckerTensorTrain',
    # frames / variations / tangents
    'T3Frame',
    'T3Variations',
    'T3Tangent',
    'UT3Frame',
    'UT3Variations',
    'UT3Tangent',
    't3_orthogonal_representations',
    'ut3_orthogonal_representations',
    # weights (edge weights on a tensor; a metric on a tangent's coordinates) -- docs/weighting.md.
    # The free functions carry the family prefix because, unlike a method, they have no class namespace
    # to disambiguate them -- the t3_orthogonal_representations pattern. Without it all four
    # absorb_weights would collide here.
    'T3Weights',
    'UT3Weights',
    'T3FrameWeights',
    'UT3FrameWeights',
    't3_absorb_weights',
    'ut3_absorb_weights',
    'fv_absorb_weights',
    'ufv_absorb_weights',
    't3_weighted_norm',
    't3_weighted_inner',
    'ut3_weighted_norm',
    'ut3_weighted_inner',
    # geometries
    'MANIFOLD',
    'COREWISE',
    'UNIFORM_MANIFOLD',
    'UNIFORM_COREWISE',
    # shared-factor geometry wrappers (Tucker factors tied within sharing groups) -- docs/sharing.md.
    # shared(MANIFOLD, sharing) wraps a base geometry; shared_manifold/shared_corewise are shorthands.
    'shared',
    'shared_manifold',
    'shared_corewise',
    # fitting models
    'GaussNewtonModel',
    'UniformGaussNewtonModel',
    'apply_model',
    'entries_model',
    'probe_model',
    'apply_derivatives_model',
    'entries_derivatives_model',
    'probe_derivatives_model',
    # optimizers
    'gradient_descent',
    'mc_sgd',
    'adam',
    'newton_cg',
    # safety mode
    'safety',
    'safe',
    'unsafe',
]
