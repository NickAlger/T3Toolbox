.. T3Toolbox documentation master file, created by
   sphinx-quickstart on Thu Apr  2 21:15:11 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Modules
=======

Tucker tensor trains:

* :doc:`/autoapi/t3toolbox/tucker_tensor_train/index`
* :doc:`/autoapi/t3toolbox/base_variation_format/index`
* :doc:`/autoapi/t3toolbox/orthogonalization/index`
* :doc:`/autoapi/t3toolbox/t3svd/index`
* :doc:`/autoapi/t3toolbox/manifold/index`
* :doc:`/autoapi/t3toolbox/probing/index`

Uniform Tucker tensor trains:

* :doc:`/autoapi/t3toolbox/uniform/index`
* :doc:`/autoapi/t3toolbox/uniform_orthogonalization/index`
* :doc:`/autoapi/t3toolbox/uniform_t3svd/index`
* :doc:`/autoapi/t3toolbox/uniform_manifold/index`
* :doc:`/autoapi/t3toolbox/uniform_probing/index`

Utilities:

* :doc:`/autoapi/t3toolbox/corewise/index`
* :doc:`/autoapi/t3toolbox/linalg/index`
* :doc:`/autoapi/t3toolbox/common/index`


Jax versions of all modules are available under t3toolbox.jax:

- t3toolbox.jax.tucker_tensor_train
- t3toolbox.jax.base_variation_format
- t3toolbox.jax.orthogonalization
- ...


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Installation
============
The package is pure python. Dependencies:

* `Numpy <https://numpy.org/install/>`_ (required)
* `Jax <https://docs.jax.dev/en/latest/installation.html>`_ (optional)

Install from source::

	git clone https://github.com/NickAlger/T3Toolbox.git
	cd T3Toolbox
	pip install .


T3Toolbox
=========

A Python library for working with Tucker tensor trains (T3). 
Tucker tensor trains are the composition of a `Tucker decomposition <https://en.wikipedia.org/wiki/Tucker_decomposition>`_ 
with a `tensor train <https://en.wikipedia.org/wiki/Matrix_product_state>`_ (also called matrix product states) representation of the central Tucker core. 

Tensor network diagram for a Tucker tensor train::

        r0        r1        r2       r(d-1)          rd
    1 ------ G0 ------ G1 ------ ... ------ G(d-1) ------ 1
             |         |                    |
             | n0      | n1                 | nd
             |         |                    |
             B0        B1                   Bd
             |         |                    |
             | N0      | N1                 | Nd
             |         |                    |

Here:

- Gi and Bi are *cores*, which are small tensors that are being contracted with each other to form a large dense N0 x ... x N(d-1) tensor.
- Edges in the network indicate contraction of adjacent cores.
- Natural numbers Ni, ni, ri, written next to edges, indicate the size of the edge (its "bandwidth", you might say).

The components of a dth order Tucker tensor train are:

- Tucker cores: (B0, B1, ..., B(d-1)) with shapes (ni, Ni).
- TT cores: (G0, G1, ..., G(d-1)) with shapes (ri, ni, r(i+1)).

The structure of a Tucker tensor train is defined by:

- Tensor shape: (N0, N1, ..., N(d-1))
- Tucker ranks: (n0, r1, ..., n(d-1))
- TT ranks: (r0, r1, ..., rd)

Typically, the first and last TT-ranks satisfy r0=rd=1, and "1" in the diagram
is the number 1. However, it is allowed for these ranks to not be 1, in which case
the "1"s in the diagram are vectors of ones.

When the ranks of a Tucker tensor train are moderate, they can break the curse of dimensionality.
Whereas the memory required to store a dense tensor is O(N^d), the memory required to store a 
Tucker tensor train is O(dnr^2 + dnN).

Unless specified otherwise, operations in this package are defined with respect 
to the dense N0 x ... x N(d-1) tensors that are *represented* by the Tucker tensor train, 
even though these dense tensors are not formed during computations.

Included functionality:
-----------------------

- Basic T3 operations (entries, addition, scaling, inner product)
- Determination of minimal ranks
- Orthogonalization
- T3-SVD
- Orthogonal representation of tangent vectors to the fixed rank T3-manifold
- Orthogonal and oblique gauge projections of tangent vector representations
- Conversion of tangent vector representations to doubled rank T3s
- Retraction of tangent vectors to the T3-manifold
- Probing T3s
- Probing tangent vectors
- Transpose of the tangent vector to probes map
- Varied-rank and uniform-rank T3s
- Option to use either `Numpy <https://numpy.org/>`_ or `Jax <https://docs.jax.dev/en/latest/index.html>`_ for linear algebra operations


Websites
--------

* Github: https://github.com/NickAlger/T3Toolbox
* Documentation: https://nickalger.github.io/T3Toolbox/


Authors
-------

* Nick Alger (nalger225@gmail.com)
* Blake Christierson (bechristierson@utexas.edu)

License
-------

* `MIT License <https://mit-license.org/>`_


Examples
========

1. Create two random Tucker tensor trains and **add** them::

	>>> import numpy as np
	>>> import t3toolbox.tucker_tensor_train as t3
	>>> randn = np.random.randn
	>>> # Make Tucker tensor train x:
	>>> x_tucker_cores = [randn(5, 21), randn(5, 22), randn(5, 23)]
	>>> x_tt_cores = [randn(1, 5, 4), randn(4, 5, 4), randn(4, 5, 1)]
	>>> x = (x_tucker_cores, x_tt_cores)
	>>> # Make Tucker tensor train y:
	>>> y_tucker_cores = [randn(9, 21), randn(9, 22), randn(9, 23)]
	>>> y_tt_cores = [randn(1, 9, 2), randn(2, 9, 2), randn(2, 9, 1)]
	>>> y = (y_tucker_cores, y_tt_cores)
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
	>>> import t3toolbox.tucker_tensor_train as t3
	>>> import t3toolbox.manifold as t3m
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
	>>> import t3toolbox.tucker_tensor_train as t3
	>>> import t3toolbox.probing as t3p
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
	>>> import t3toolbox.tucker_tensor_train as t3
	>>> import t3toolbox.uniform_tucker_tensor_train as ut3
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

- Functions have no side effects, and functions always yield the same output for a given input (except a couple functions that generate random tensors). 

- Custom types are aliases of composite basic types.

- We took great effort to reduce dependencies in our code as much as possible, both externally and internally. Many functions here could literally be copied and pasted into other projects, and would work just fine if one replaces the generic NDArray TypeVar with the appropriate array backend (e.g., np.ndarray or jnp.ndarray).


Compatibility with Jax
======================

Jax versions of all functions are available under t3toolbox.jax. For example, if you want to use jax as the linear algebra backend to add two Tucker tensor trains, you would replace:

	>>> t3toolbox.tucker_tensor_train.t3_add(x, y)
	
with:
	
	>>> t3toolbox.jax.tucker_tensor_train.t3_add(x, y)
	
- Both numpy and jax versions of functions can be used together within the same code::

	>>> import numpy as np
	>>> import t3toolbox.tucker_tensor_train as t3
	>>> import t3toolbox.jax.tucker_tensor_train as t3_jax
	>>> structure = ((9,8,7),(5,4,3),(1,6,5,1))
	>>> x = t3.t3_corewise_randn(structure)
	>>> y = t3.t3_corewise_randn(structure)
	>>> print(type(t3.t3_add(x, y)[0][0]))
	<class 'numpy.ndarray'>
	>>> print(type(t3_jax.t3_add(x, y)[0][0]))
	<class 'jaxlib.xla_extension.ArrayImpl'>
	
- For functions that do not explicitly call jax or numpy functions, the arrays that are output will be the same type as the arrays that are input::

	>>> import numpy as np
	>>> import jax.numpy as jnp
	>>> import t3toolbox.tucker_tensor_train as t3
	>>> import t3toolbox.jax.tucker_tensor_train as t3_jax
	>>> structure = ((9,8,7),(5,4,3),(1,6,5,1))
	>>> x_np = t3.t3_corewise_randn(structure)
	>>> x_jax = t3_jax.t3_corewise_randn(structure)
	>>> rev_np_x_np = t3.reverse_t3(x_np)
	>>> rev_np_x_jax = t3.reverse_t3(x_jax)
	>>> rev_jax_x_np = t3_jax.reverse_t3(x_np)
	>>> rev_jax_x_jax = t3_jax.reverse_t3(x_jax)
	>>> print(type(rev_np_x_np[0][0]))
	<class 'numpy.ndarray'>
	>>> print(type(rev_np_x_jax[0][0]))
	<class 'jaxlib.xla_extension.ArrayImpl'>
	>>> print(type(rev_jax_x_np[0][0]))
	<class 'numpy.ndarray'>
	>>> print(type(rev_jax_x_jax[0][0]))
	<class 'jaxlib.xla_extension.ArrayImpl'>

- Jax versions of numerical functions are suitable for `just-in-time (jit) compilation <https://docs.jax.dev/en/latest/_autosummary/jax.jit.html>`_ in jax, after removing non-numerical parameters by `partial evaluation <https://en.wikipedia.org/wiki/Partial_application>`_. E.g.,::

	>>> import numpy as np
	>>> import jax
	>>> import t3toolbox.jax.tucker_tensor_train as t3_jax
	>>> get_entry_123 = lambda x: t3_jax.t3_entry(x, (1,2,3))
	>>> A = t3_jax.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1))) # random 10x10x10 T3
	>>> a123 = get_entry_123(A)
	>>> print(a123)
	11.756762
	>>> get_entry_123_jit = jax.jit(get_entry_123) # jit compile
	>>> a123_jit = get_entry_123_jit(A)
	>>> print(a123_jit)
	11.756762

- Jax versions of most numerical functions are suitable for

	>>> import numpy as np
	>>> import jax
	>>> import t3toolbox.jax.tucker_tensor_train as t3_jax
	>>> get_entry_123 = lambda x: t3_jax.t3_entry(x, (1,2,3))
	>>> A = t3_jax.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1))) # random 10x10x10 T3
	>>> a123 = get_entry_123(A)
	>>> print(a123)
	11.756762
	>>> get_entry_123_jit = jax.jit(get_entry_123) # jit compile
	>>> a123_jit = get_entry_123_jit(A)
	>>> print(a123_jit)
	11.756762

- Jax versions of most numerical functions are suitable for
		
	>>> import numpy as np
	>>> import jax
	>>> import t3toolbox.jax.tucker_tensor_train as t3_jax
	>>> get_entry_123 = lambda x: t3_jax.t3_entry(x, (1,2,3))
	>>> A = t3_jax.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1))) # random 10x10x10 T3
	>>> a123 = get_entry_123(A)
	>>> print(a123)
	11.756762
	>>> get_entry_123_jit = jax.jit(get_entry_123) # jit compile
	>>> a123_jit = get_entry_123_jit(A)
	>>> print(a123_jit)
	11.756762

- Jax versions of most numerical functions are suitable for `automatic differentiation (AD) <https://en.wikipedia.org/wiki/Automatic_differentiation>`_ in jax. E.g.,::

	>>> import numpy as np
	>>> import jax
	>>> from t3toolbox.jax import tucker_tensor_train as t3_jax
	>>> from t3toolbox.jax import corewise as cw
	>>> jax.config.update("jax_enable_x64", True) # enable double precision for finite difference
	>>> get_entry_123 = lambda x: t3_jax.t3_entry(x, (1,2,3))
	>>> A0 = t3_jax.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1))) # random 10x10x10 T3
	>>> f0 = get_entry_123(A0)
	>>> G0 = jax.grad(get_entry_123)(A0) # gradient using automatic differentiation
	>>> dA = t3_jax.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1)))
	>>> df = cw.corewise_dot(dA, G0) # sensitivity in direction dA
	>>> print(df)
	-7.418801772515241
	>>> s = 1e-7
	>>> A1 = cw.corewise_add(A0, cw.corewise_scale(dA, s)) # A1 = A0 + s*dA
	>>> f1 = get_entry_123(A1)
	>>> df_diff = (f1 - f0) / s # finite difference
	>>> print(df_diff)
	-7.418812309825662

- AD Caveat: We do not recommend automatically differentiating through functions that involve singular value decompositions (SVDs) because support for `singular value sensitivity <https://en.wikipedia.org/wiki/Eigenvalue_perturbation>`_ in jax is questionable. This includes:

	- T3-SVD,
	- Orthogonalization (uses SVDs for stability and robustness), 
	- Retraction.


T3 Background
=============


Tucker tensor trains represent dense tensors
--------------------------------------------

Although the Tucker tensor train is defined by its cores, we always keep in mind that it *represents* a 
dense N0 x ... x N(d-1) tensor. Tucker tensor train representations are not unique. In particular, you can
always insert a matrix times its inverse in the middle of an edge, then absorb the matrix into one of the 
cores on the edge and absorb the inverse into the other. This changes the T3 representation, but does not change 
the dense tensor being represented.

When we perform operations with Tucker tensor trains, like adding them, scaling them, taking inner products, etc, we typically are simulating these operations on the represented dense tensors, using the cores merely as a computational device. We are not performing the operations "corewise".

For example, consider the Tucker tensor trains:

	* x = ((A0,...,A(d-1)), (F0,...,F(d-1)))
	* y = ((B0,...,B(d-1)), (G0,...,G(d-1)))
	* z = ((C0,...,C(d-1)), (H0,...,H(d-1)))
	
We want::

	z "=" x "+" y
	
to mean the following: if you added the N0 x ... x N(d-1) tensor represented by x to the N0 x ... x N(d-1) tensor represented by y, then the resulting N0 x ... x N(d-1) tensor can be represented by the Tucker tensor train z. I.e.,::

	t3_to_dense(z) = t3_to_dense(x) + t3_to_dense(y).
	
We do not mean "add the cores". Generally,
Ci =/= Ai + Bi and Hi =/= Fi + Gi.


Minimal rank conditions
-----------------------

Tucker tensor trains are said to have *minimal ranks* if they satisfy:
	- Left TT core unfoldings are full rank: r(i+1) <= (ri*ni)
	- Right TT core unfoldings are full rank: ri <= (ni*r(i+1))
	- Outer TT core unfoldings are full rank: ni <= (ri*r(i+1))
	- Tucker cores have full row rank: ni <= Ni

Minimal rank properties:
	- Minimal ranks always exist and are unique.
	- Minimal Tucker ranks ni are equal to the ranks of Ni x (N1*...*N(i-1)*N(i+1)*...*N(d-1)) matricizations.
	- Minimal TT ranks ri are equal to the ranks of (N*...*Ni) x (N(i+1)*...*N(d-1)) matrix unfoldings.
	- Minimal rank representations of any T3 may be constructed with T3-SVD.

In this package, minimal ranks are typically defined with respect to a
generic Tucker tensor train of the given structure.
We do not account for possible additional rank deficiency due to
the numerical values within the cores.

Tucker tensor trains that do not have minimal ranks are degenerate, as they can always be reduced to minimal rank Tucker tensor trains (without changing the represented tensor) using T3-SVD. This degeneracy is analogous to a low rank matrix approximation A = B C, where A is NxM, B is Nxk, C is kxM, and k>min(N,M).

T3 Manifold
-----------

Under the minimal rank conditions, the set of Tucker tensor trains with fixed ranks forms an embedded submanifold in R^(N1 x ... x Nd). In this case, any tangent vector may be represented as a sum which looks like this::

          1--H0--R1--R2--1   1--L0--H1--R2--1   1--L0--L1--H2--1
             |   |   |          |   |   |          |   |   | 
    v  =     U0  U1  U2    +    U0  U1  U2    +    U0  U1  U2
             |   |   |          |   |   |          |   |   | 
    
          1--O0--R1--R2--1   1--L0--O1--R2--1   1--L0--L1--O2--1
             |   |   |          |   |   |          |   |   | 
       +     V0  U1  U2    +    U0  V1  U2    +    U0  U1  V2
             |   |   |          |   |   |          |   |   | 

- The following "base" cores are orthogonal representations of the base point where the tangent space is attached. When performing computations, these are computed once per tangent space, then fixed for all tangent vectors in the space:
	- tucker_cores      = (U0,...,Ud), orthogonal
	- left_tt_cores     = (L0,...Ld), left-orthogonal
	- right_tt_cores    = (R0,...,Rd), right-orthogonal
	- outer_tt_cores    = (O0,...,Od), outer-orthogonal
- The following "variation" cores define the tangent vector w.r.t. the base cores:
	- tucker_variations = (V0,...,Vd)
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
	- The Tucker cores B0, ..., Bd all have the same shape (n,N) and can be stacked into a *Tucker supercore* with shape (d,n,N). 
	- The TT-cores G0, ..., Gd all have the same shape (r,n,r), and can be stacked into *TT supercore* with shape (d,r,n,r).

We keep track of which parts of the cores are supposed to be zero (to prevent filling in these parts during computations) with *edge masks*. For each edge, the mask is a boolean vector of the form (1,...,1,0,...,0). 
	- The masks for the edges between Tucker cores and TT-cores have length n. For the ith edge, the first ni entries are 1, and the remaining entries are 0. These *Tucker edge masks* are collected into a boolean array with shape (d, n).
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

License
=======

MIT License

Copyright (c) 2026 Nick Alger and Blake Christierson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Complete contents
=================

.. toctree::
   :titlesonly:
   :maxdepth: 4

