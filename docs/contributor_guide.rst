Contributor guide
=================

Documentation for people **modifying** the library: the conventions new code must follow, the test
strategy and its traps, and the durable design-decision records (what was considered, what was
rejected, and why) that guard settled choices against accidental re-litigation. Library *users*
should not need anything here — the user-facing story lives in the :doc:`user_guide` and
:doc:`design_notes`.

Conventions for new code
------------------------

.. toctree::
   :maxdepth: 1

   contributor/signature_style
   contributor/doctest_style
   contributor/testing_strategy
   contributor/naming_rules
   contributor/refactoring_methodology

Design-decision records
-----------------------

.. toctree::
   :maxdepth: 1

   contributor/batching_internals
   contributor/contractions_internals
   contributor/fitting_internals
   contributor/weighted_internals
   contributor/numerical_contract_catalog
   contributor/t3svd_design_rationale
   contributor/t3svd_verification
   contributor/ambient_derivative_transpose_note
   contributor/uniform_svd_prefix_orthogonalization
   contributor/uniform_pytree_composition
   contributor/uniform_rank_masks_rationale
   contributor/uniform_internals
   contributor/uniform_polymorphism
   contributor/deferred_and_rejected
   contributor/verification_records
