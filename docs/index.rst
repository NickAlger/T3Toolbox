T3Toolbox
=========

A pure-Python (NumPy + optional JAX) library for **Tucker tensor trains (T3)**. A Tucker tensor train 
is the composition of a Tucker decomposition with a tensor train decomposition of the central core.
When the ranks are moderate, a T3 breaks the curse of dimensionality: storing a dense tensor costs 
O(N^d) memory, while the T3 representing it costs O(dnr^2 + dnN). 
Tucker tensor trains are also known as **extended tensor trains (ETT)**.

The library provides the T3 format itself (arithmetic, orthogonalization, T3-SVD), the three
sampling operations (``entries`` / ``apply`` / ``probe``) and their derivatives, the fixed-rank
T3 manifold with tangent vectors and Riemannian geometry, least-squares fitting with four
optimizers (residual weighting, regularization and rank continuation included), optimization over
**shared Tucker factors** (SF-T3), diagonal **edge weighting** of tensors and tangent coordinates,
and a mask-based **uniform** (padded, GPU/`jit`-friendly) mirror of the whole stack.

.. toctree::
   :maxdepth: 1

   getting_started
   user_guide
   design_notes
   contributor_guide
   api_reference
   release_notes


Installation
------------

The package is pure Python. Dependencies:

* `NumPy <https://numpy.org/install/>`_ (required)
* `JAX <https://docs.jax.dev/en/latest/installation.html>`_ (optional)

::

	pip install t3toolbox

To include the optional JAX backend::

	pip install "t3toolbox[jax]"

From source (development install)::

	git clone https://github.com/NickAlger/T3Toolbox.git
	cd T3Toolbox
	pip install -e .


Websites
--------

* GitHub: https://github.com/NickAlger/T3Toolbox
* Documentation: https://nickalger.github.io/T3Toolbox/


Authors
-------

* Nick Alger (nalger225@gmail.com)
* Blake Christierson (bechristierson@utexas.edu)

MIT License. To cite T3Toolbox, cite the software itself (the repository's ``CITATION.cff``). The
library grew out of the research in *Alger, Christierson, Chen & Ghattas (2026),
"Tucker Tensor Train Taylor Series"* (`arXiv:2603.21141 <https://arxiv.org/abs/2603.21141>`_), where
many of the core algorithms are described — cite that paper when you build on its research
contributions; see :ref:`relevant-literature` for the wider background.


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
