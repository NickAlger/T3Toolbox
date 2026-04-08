.. TuckerTensorTrainTools documentation master file, created by
   sphinx-quickstart on Thu Apr  2 21:15:11 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TuckerTensorTrainTools's documentation
======================================

A Python library for working with Tucker tensor trains (T3). 
Includes:

	- Basic T3 operations (entries, addition, scaling, inner product)
	- Orthogonalization
	- T3-SVD
	- Orthogonal representation of tangent vectors to the fixed rank T3-manifold
	- Orthogonal and oblique gauge projections of tangent vector representations
	- Conversion of tangent vector representations to rank-2r T3s
	- Retraction of tangent vectors to the T3-manifold
	- Probing T3s
	- Probing tangent vectors
	- Transpose of the tangent vector to probes map
	- Varied-rank and uniform-rank T3s
	- Option to use either Numpy or Jax for linear algebra operations
	

Modules
=======

Tucker tensor trains:

* :doc:`/autoapi/t3tools/tucker_tensor_train/index`
* :doc:`/autoapi/t3tools/base_variation_format/index`
* :doc:`/autoapi/t3tools/orthogonalization/index`
* :doc:`/autoapi/t3tools/t3svd/index`
* :doc:`/autoapi/t3tools/manifold/index`
* :doc:`/autoapi/t3tools/probing/index`
* :doc:`/autoapi/t3tools/util/index`

Uniform T3s:

* :doc:`/autoapi/t3tools/uniform_tucker_tensor_train/index`
* :doc:`/autoapi/t3tools/uniform_manifold/index`
* :doc:`/autoapi/t3tools/uniform_probing/index`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Websites
========

* Github: https://github.com/NickAlger/TuckerTensorTrainTools
* Documentation: https://nickalger.github.io/TuckerTensorTrainTools/


Installation
============
The package is pure python. Dependencies:

* `Numpy <https://numpy.org/install/>`_ (required)
* `Jax <https://docs.jax.dev/en/latest/installation.html>`_ (optional)

Install from source::

	git clone https://github.com/NickAlger/TuckerTensorTrainTools.git
	cd TuckerTensorTrainTools
	pip install .


Examples
========

1. Create two random Tucker tensor trains and **add** them::

	>>> import numpy as np
	>>> import t3tools.tucker_tensor_train as t3
	>>> randn = np.random.randn
	>>> # Make Tucker tensor train x:
	>>> x_basis_cores = [randn(5, 21), randn(5, 22), randn(5, 23)]
	>>> x_tt_cores = [randn(1, 5, 4), randn(4, 5, 4), randn(4, 5, 1)]
	>>> x = (x_basis_cores, x_tt_cores)
	>>> # Make Tucker tensor train y:
	>>> y_basis_cores = [randn(9, 21), randn(9, 22), randn(9, 23)]
	>>> y_tt_cores = [randn(1, 9, 2), randn(2, 9, 2), randn(2, 9, 1)]
	>>> y = (y_basis_cores, y_tt_cores)
	>>> # Add x+y:
	>>> x_plus_y = t3.t3_add(x, y)
	>>> # x+y has doubled ranks:
	>>> print(t3.structure(x_plus_y))
	((21, 22, 23), (14, 14, 14), (2, 6, 6, 2))
	>>> # Convert to dense to check error:
	>>> x_dense = t3.t3_to_dense(x)
	>>> y_dense = t3.t3_to_dense(y)
	>>> x_plus_y_dense = t3.t3_to_dense(x_plus_y)
	>>> print(np.linalg.norm(x_dense + y_dense - x_plus_y_dense))
	0.0

2. **Retract** random tangent vector to the manifold of fixed rank Tucker tensor trains::

	>>> import numpy as np
	>>> import t3tools.tucker_tensor_train as t3
	>>> import t3tools.manifold as t3m
	>>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1))) # tangent space base point
	>>> base, _ = t3m.orthogonal_representations(p)
	>>> v = t3m.tangent_randn(base) # Random tangent vector.
	>>> ret_v = t3m.retract(v, base) # Retract tangent vector to manifold.
	>>> v_as_t3 = t3m.tangent_to_t3(v, base) # Convert tangent vector to rank-2r T3
	>>> p_plus_v = t3.t3_add(p, v_as_t3) # Shift tangent so its tail is at p instead of 0
	>>> retracted_distance = t3.t3_norm(t3.t3_sub(p_plus_v, ret_v))
	>>> print(retracted_distance)
	0.14470074958504858

3. **Probe** Tucker tensor train with random vectors::

	>>> import numpy as np
	>>> import t3tools.tucker_tensor_train as t3
	>>> import t3tools.probing as t3p
	>>> x = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2))) # random T3
	>>> w1 = np.random.randn(10) # random probing vectors
	>>> w2 = np.random.randn(11)
	>>> w3 = np.random.randn(12)
	>>> zz = t3p.probe_t3(x, (w1, w2, w3)) # Probe T3-tensor
	>>> x_dense = t3.t3_to_dense(x) # Convert to dense to check error
	>>> z1 = np.einsum('ijk,j,k->i', x_dense, w2, w3) # Probe dense tensor (brute force)
	>>> z2 = np.einsum('ijk,i,k->j', x_dense, w1, w3)
	>>> z3 = np.einsum('ijk,i,j->k', x_dense, w1, w2)
	>>> zzb = [z1, z2, z3]
	>>> print([np.linalg.norm(z - zb) for z, zb in zip(zz, zzb)])
	[8.806144583576081e-13, 5.012223052900821e-13, 4.968721252978153e-13]

4. Convert Tucker tensor train to **uniform** Tucker tensor train::

	>>> import numpy as np
	>>> import t3tools.tucker_tensor_train as t3
	>>> import t3tools.uniform_tucker_tensor_train as ut3
	>>> x = t3.t3_corewise_randn(((14,15,16), (4,5,3), (1,4,2,1))) # T3
	>>> index = (3,1,2)
	>>> x_312 = t3.t3_entry(x, index)
	>>> print(x_312) # (3,1,2) entry from T3:
	-1.4931654579929192
	>>> cores, masks = ut3.t3_to_ut3(x) # Convert to Uniform T3
	>>> print(ut3.original_structure(masks)) # original (shape, tucker_ranks, tt_ranks):
	((14, 15, 16), (4, 5, 3), (1, 4, 2, 1))
	>>> print(ut3.padded_structure(cores)) # uniform shape and ranks, (d,N,n,r):
	(3, 16, 5, 4)
	>>> x_312_uniform = ut3.ut3_entry(cores, index) # (3,1,2) entry from uniform T3:
	>>> print(x_312_uniform)
	-1.4931654579929197


Design philosophy
=================

This package is written in a `functional programming <https://en.wikipedia.org/wiki/Functional_programming>`_ style. It is a library of mathematical functions that perform operations on basic types (mostly, arrays and nested sequences of arrays). 

- Functions have no side effects, and functions always yield the same output for a given input. 
- Custom types are aliases of composite basic types
- Numerical functions are suitable for `just-in-time (jit) compilation <https://docs.jax.dev/en/latest/_autosummary/jax.jit.html>`_ in `jax <https://docs.jax.dev/en/latest/index.html>`_, after removing non-numerical parameters by `partial evaluation <https://en.wikipedia.org/wiki/Partial_application>`_. E.g.,::
		
	>>> import numpy as np
	>>> import jax
	>>> import t3tools.tucker_tensor_train as t3
	>>> get_entry_123 = lambda x: t3.t3_entry(x, (1,2,3), use_jax=True)
	>>> A = t3.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1))) # random 10x10x10 Tucker tensor train
	>>> a123 = get_entry_123(A)
	>>> print(a123)
	11.756762
	>>> get_entry_123_jit = jax.jit(get_entry_123) # jit compile
	>>> a123_jit = get_entry_123_jit(A)
	>>> print(a123_jit)
	11.756762

- Most numerical functions are suitable for `automatic differentiation (AD) <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ in jax. E.g.,::

	>>> import numpy as np
	>>> import jax
	>>> import t3tools.tucker_tensor_train as t3
	>>> import t3tools.util as util
	>>> jax.config.update("jax_enable_x64", True) # enable double precision for finite difference
	>>> get_entry_123 = lambda x: t3.t3_entry(x, (1,2,3), use_jax=True)
	>>> A0 = t3.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1))) # random 10x10x10 Tucker tensor train
	>>> f0 = get_entry_123(A0)
	>>> G0 = jax.grad(get_entry_123)(A0) # gradient using automatic differentiation
	>>> dA = t3.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1)))
	>>> df = util.corewise_dot(dA, G0) # sensitivity in direction dA
	>>> print(df)
	-7.418801772515241
	>>> s = 1e-7
	>>> A1 = util.corewise_add(A0, util.corewise_scale(dA, s)) # A1 = A0 + s*dA
	>>> f1 = get_entry_123(A1)
	>>> df_diff = (f1 - f0) / s # finite difference
	>>> print(df_diff)
	-7.418812309825662

- *AD Caveat*: We do not recommend automatically differentiating through functions that involve singular value decompositions (SVDs) because support for `singular value sensitivity <https://en.wikipedia.org/wiki/Eigenvalue_perturbation>`_ in jax is questionable. This includes:
	- T3-SVD,
	- Orthogonalization (uses SVDs for stability and robustness), 
	- Retraction.


Background
==========

Tucker tensor trains
--------------------

Tucker tensor trains consist of a Tucker decomposition composed with a tensor train decomposition of the central Tucker core. We may diagram a Tucker tensor train with 4 indices like this, using graphical tensor notation::

        r0        r1        r2        r2        r4
    1 ------ G0 ------ G1 ------ G2 ------ G3 ------ 1
             |         |         |         |
             |n0       |n1       |n2       |n3
             |         |         |         |
             B0        B1        B2        B3
             |         |         |         |
             |N0       |N1       |N2       |N3
             |         |         |         |

- Gi and Bi are *cores*, which are smaller tensors that are being contracted with each other to form a larger tensor.
- Edges in the network indicate contraction of adjacent cores.
- Natural numbers Ni, ni, ri, written next to edges, indicate the size of the edge (its "bandwidth", you might say).

The Tucker tensor train with d indices is represented as a Tuple of cores, ((B0,...,Bd), (G0,...,Gd)).
	- Bi are the *basis cores*, which are matrices with shape (ni, Ni)
	- Gi are the *TT-cores*, which are 3-tensors with shape (ri, ni, r(i+1))
	- (N1,...,Nd) is the *shape* of the fully contracted tensor
	- (n1,...,nd) are the *Tucker ranks*
	- (1,ri,...,r(d-1),1) are the *TT-ranks*
	- ((N1,...,Nd), (n1,...,nd), (r0,r1,...,r(d-1),rd)) is the *structure*.
	- 1=(1,...,1) denotes the ones vector of the appropriate size

Typically r0=rd=1, and the "1" on the left and right sides is just the number 1. However, this is not required.
Having r0, r1 > 1 is allowed, in which case the "1"s in the diagram are vectors of ones, (1,1,...,1).



Tucker tensor trains represent dense tensors
--------------------------------------------

Although the Tucker tensor train is defined by its cores, we always keep in mind that it *represents* a dense tensor in R^(N1 x ... x Nd). Tensor train representations are not unique; this fact is exploited by many Tensor train algorithms. 

When we perform operations with Tucker tensor trains, like adding them, scaling them, taking inner products, etc, we typically are simulating these operations on the represented dense tensors, using the cores as a computational device to avoid operating on gigantically large arrays. We are not performing the operations "corewise".

For example, consider the Tucker tensor trains:
	* x = ((A0,...,Ad), (F0,...,Fd))
	* y = ((B0,...,Bd), (G0,...,Gd))
	* z = ((C0,...,Cd), (H0,...,Hd))
	
We want::

	z "=" x "+" y
	
to mean the following: if you added the N1 x ... x Nd tensor represented by x to the tensor N1 x ... x Nd represented by y, then the resulting N1 x ... x Nd tensor can be represented by the Tucker tensor train z. I.e.,::

	t3_to_dense(z) = t3_to_dense(x) + t3_to_dense(y).
	
We do not mean "add the cores". E.g., generally
Ci =/= Ai + Bi.

         
         
T3 Manifold
-----------

Under certain conditions on the ranks (basically, if the ranks are not unnecessairily large), the set of Tucker tensor trains with fixed ranks forms an embedded submanifold in R^(N1 x ... x Nd). In this case, any tangent vector may be represented as a sum which looks like this::

          1--H0--R1--R2--1   1--L0--H1--R2--1   1--L0--L1--H2--1
             |   |   |          |   |   |          |   |   | 
    v  =     U0  U1  U2    +    U0  U1  U2    +    U0  U1  U2
             |   |   |          |   |   |          |   |   | 
    
          1--O0--R1--R2--1   1--L0--O1--R2--1   1--L0--L1--O2--1
             |   |   |          |   |   |          |   |   | 
       +     V0  U1  U2    +    U0  V1  U2    +    U0  U1  V2
             |   |   |          |   |   |          |   |   | 

- The following "base" cores are orthogonal representations of the base point where the tangent space is attached. When performing computations, these are computed once per tangent space, then fixed for all tangent vectors in the space:
	- basis_cores       = (U0,...,Ud), orthogonal
	- left_tt_cores     = (L0,...Ld), left-orthogonal
	- right_tt_cores    = (R0,...,Rd), right-orthogonal
	- outer_tt_cores    = (O0,...,Od), outer-orthogonal
- The following "variation" cores define the tangent vector w.r.t. the base cores:
	- basis_variations  = (V0,...,Vd)
	- tt_variations     = (H0,...,Hd)

Under certain *gauge conditions*, the representation of a tangent vector by its variation is unique, and performing linear algebra with the variation is equivalent to performing the corresponding linear algebra operations with the dense tensor.


Probing
-------

Probing a tensor means contracting the tensor with vectors in all but one index, resulting in a vector of the size of that remaining index. Probing a N1 x N2 x N3 Tucker tensor with vectors u1, u2, u3 (lengths N1, N2, N3, respectively) looks like this::

                     1--G0--G1--G2--1
                        |   |   |    
    first probe:  =     B0  B1  B2   
    (len=N1)            |   |   |    
                            u2  u3
                        
                     1--G0--G1--G2--1
                        |   |   |    
    second probe: =     B0  B1  B2   
    (len=N2)            |   |   |    
                        u1      u3
                        
                     1--G0--G1--G2--1
                        |   |   |    
    third probe:  =     B0  B1  B2   
    (len=N3)            |   |   |    
                        u1  u2
       
                 
Uniform Tucker tensor trains
----------------------------

For computational efficiency, it can be helpful to pad the cores of a Tucker tensor train with zeros, so that it has uniform ranks. Then the computational operations become uniform, which improves pipelining efficiency and GPU performance.

In this case:
	- The basis cores B0, ..., Bd all have the same shape (n,N) and can be stacked into a *basis supercore* with shape (d,n,N). 
	- The TT-cores G0, ..., Gd all have the same shape (r,n,r), and can be stacked into *TT supercore* with shape (d,r,n,r).

We keep track of which parts of the cores are supposed to be zero (to prevent filling in these parts during computations) with *edge masks*. For each edge, the mask is a boolean vector of the form (1,...,1,0,...,0). 
	- The masks for the edges between basis cores and TT-cores have length n. For the ith edge, the first ni entries are 1, and the remaining entries are 0. These *Tucker edge masks* are collected into a boolean array with shape (d, n).
	- The masks for the edges between adjacent TT-cores have length r. For the ith edge, the first ri entries are 1, and the remaining entries are 0. These *TT edge masks* are collected into boolean array with shape (d+1,r).


**Note:** Use uniform Tucker tensor trains with **minimal ranks only**. 
If the ranks are degenerate (not minimal), then some operations with 
uniform Tucker tensor trains, particularly orthogonalization, T3-SVD, 
and manifold operations, may give results that are inconsistent with the 
corresponding operations on regular Tucker tensor trains. 
A Tucker tensor train may be converted to one with minimal ranks via T3-SVD.

Relevant literature
-------------------

* Most of the Tucker tensor train algorithms are described in Appendix A of our paper [1]. 

* The probing algorithms are described in Section 5 of our paper [1].

* The algorithms here are extensions of standard tensor train algorithms (no Tucker)
	* For an introduction to tensor train methods, we highly recommend two extremely well written Ph.D. Theses: Chapter 1 of Voorhaar, Willem Hendrik Voorhaar's Ph.D. thesis [2], and Chapter 3 of Patrick Gelß's Ph.D. thesis [3]. 
	* The foundations behind these algorithms were established over the last 20 years. 
		* Oseledets [4] defines the basics of Tensor trains and TT-SVD. 
		* In [5], Holtz, Rohwedder, and Schneider detail the manifold of fixed rank tensor trains and its tangent space and presents gauged representations and oblique gauge projection, but only used left-orthogonal representations for tangent vector bases.
		* Using both left- and right-orthogonal representations was proposed in Khoromskij, Oseledets, and Schneider [6].
		* The modern manifold methods are given in Steinlechner [7]. 
		* Some of these methods may have been known to the physics community earlier, where tensor trains are known as "Matrix product states."

* The basics of the Tucker decomposition are described well in Kolda's review paper [8].


[1] Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026). "Tucker Tensor Train Taylor Series." arXiv preprint arXiv:2603.21141. `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

[2] Voorhaar, Willem Hendrik. "Tensor train approximations: Riemannian methods, randomized linear algebra and applications to machine learning." Diss. Ph. D. dissertation, Section de Mathématiques, Univ. Geneva, Geneva, Switzerland, 2022.

[3] Gelß, Patrick. The tensor-train format and its applications: Modeling and analysis of chemical reaction networks, catalytic processes, fluid flows, and Brownian dynamics. Diss. 2017.

[4] Oseledets, Ivan V. "Tensor-train decomposition." SIAM Journal on Scientific Computing 33.5 (2011): 2295-2317.

[5] Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "On manifolds of tensors of fixed TT-rank." Numerische Mathematik 120.4 (2012): 701-731.

[6] Khoromskij, Boris N., Ivan V. Oseledets, and Reinhold Schneider. "Efficient time-stepping scheme for dynamics on TT-manifolds." (2012).

[7] Steinlechner, Michael. "Riemannian optimization for high-dimensional tensor completion." SIAM Journal on Scientific Computing 38.5 (2016): S461-S484. `https://epubs.siam.org/doi/10.1137/15M1010506 <https://epubs.siam.org/doi/10.1137/15M1010506>`_

[8] Kolda, Tamara G., and Brett W. Bader. "Tensor decompositions and applications." SIAM review 51.3 (2009): 455-500. `https://epubs.siam.org/doi/10.1137/07070111X <https://epubs.siam.org/doi/10.1137/07070111X>`_



Authors
=======

* Nick Alger
* Blake Christierson






