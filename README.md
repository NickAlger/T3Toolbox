WORK IN PROGRESS DO NOT USE

# T3Toolbox

A Python library for working with Tucker tensor trains (T3). 
Tucker tensor trains are the composition of a [Tucker decomposition](https://en.wikipedia.org/wiki/Tucker_decomposition) 
with a [tensor train](https://en.wikipedia.org/wiki/Matrix_product_state) (also called matrix product states) representation of the central Tucker core. 

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

# Included functionality:

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
- Option to use either [Numpy](https://numpy.org/) or [Jax](https://docs.jax.dev/en/latest/index.html) for linear algebra operations


# Websites

* Github: https://github.com/NickAlger/T3Toolbox
* Documentation: https://nickalger.github.io/T3Toolbox/


# Authors

* Nick Alger (nalger225@gmail.com)
* Blake Christierson (bechristierson@utexas.edu)

# License

* [MIT License](https://mit-license.org/)


# Installation
The package is pure python. Dependencies:

* [Numpy](https://numpy.org/install/) (required)
* [Jax](https://docs.jax.dev/en/latest/installation.html) (optional)

Install from source::

	git clone https://github.com/NickAlger/T3Toolbox.git
	cd T3Toolbox
	pip install .

# Documentation

* https://nickalger.github.io/T3Toolbox/

# Github repo

* https://github.com/NickAlger/T3Toolbox


