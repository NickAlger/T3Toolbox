Design notes
============

The design rationale behind the library, written for **users**: why each format and convention is
the way it is, what each operation guarantees, and how to choose between alternatives. (Notes for
people *modifying* the library live in the :doc:`contributor_guide`.)

Core design
-----------

.. toctree::
   :maxdepth: 1

   frame_variations
   batching_and_stacking
   entries_apply_probe
   transposes
   t3m_methods
   fitting_and_optimization
   weighting
   numerical_contracts
   rank_continuation
   naming_conventions
   reading_signatures

T3-SVD
------

.. toctree::
   :maxdepth: 1

   t3svd_minimal_ranks

The uniform layer
-----------------

.. toctree::
   :maxdepth: 1

   uniform_equivalence_contract
   uniform_ranks_and_varieties
   uniform_supercore_layout
   uniform_masks_vs_ranks
   uniform_backend_jit_recipe

.. note::
	The user/developer split of these notes is in progress: the pages above marked for
	rewrite-splits (batching, fitting, the contract catalog, the mask notes) still carry some
	contributor-facing material until their split lands.
