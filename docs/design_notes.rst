Design notes
============

The durable design rationale behind the library -- why each format, convention, and algorithm is
the way it is. These are working design documents: they are the authoritative reference for
contributors and for users who want the *why*, written during development and kept current.

Conventions and style
---------------------

.. toctree::
   :maxdepth: 1

   naming_conventions
   signature_style
   doctest_style
   testing_strategy

Core design
-----------

.. toctree::
   :maxdepth: 1

   batching_and_stacking
   entries_apply_probe
   transposes
   fitting_and_optimization
   numerical_contract_catalog
   rank_continuation
   probing_section6_notes
   ambient_derivative_transpose_note

T3-SVD
------

.. toctree::
   :maxdepth: 1

   t3svd_design_rationale
   t3svd_minimal_ranks
   t3svd_verification

The uniform layer
-----------------

.. toctree::
   :maxdepth: 1

   uniform_equivalence_contract
   uniform_ranks_and_varieties
   uniform_supercore_layout
   uniform_masks_vs_ranks
   uniform_rank_masks_rationale
   uniform_svd_prefix_orthogonalization
   uniform_pytree_composition
   uniform_backend_jit_recipe
