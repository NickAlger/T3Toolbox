"""Root cause of the non-orthogonal uniform frame at a non-minimal point: the SVD sweep leaves MORE real
orthonormal content in the left chain / down core than compute_orthogonal_representation_ranks says, so the
prefix masks are too RESTRICTIVE there (real content outside the mask), the down core is not orthonormal on the
masked block, the tangent space loses directions, and MANIFOLD.norm != HS norm (unsafe) / safe mode raises."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.safety as safety
np.random.seed(0)
shape, tr, ttr = (5, 7, 6), (2, 3, 2), (1, 2, 3, 1)       # bond 2: 3 > n2*r3 = 2 -> non-minimal
x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
uf = ut3m.UNIFORM_MANIFOLD.frame(ux)
up, down, left, right = uf.supercores            # RAW (unmasked) supercores
um, dm, lm, rm = uf.masks.data
print('formula compute_orthogonal_representation_ranks:', ranks.compute_orthogonal_representation_ranks(shape, tr, ttr))
rf = bvf.T3Frame.from_t3(x)
print('ragged T3Frame.from_t3 actual ranks:            up', tuple(rf.up_ranks), 'down', tuple(rf.down_ranks), 'left', tuple(rf.left_ranks), 'right', tuple(rf.right_ranks))
# Unmasked Gram of the uniform left core 1 (outgoing bond 2) over ALL padded columns:
G = np.einsum('iaj,iak->jk', left[1], left[1])
print('uniform left core 1 unmasked Gram (bond 2 columns):\n', np.round(G, 6), '\n  mask says left bond 2 rank =', int(lm[2].sum()), '-> 3 orthonormal columns exist; mask keeps 2')
# Down core O_2 unmasked Gram over bonds (rows = left bond 2)
Gd = np.einsum('iaj,ibj->ab', down[2], down[2])
print('uniform down core 2 unmasked Gram over bonds (all 3 left-bond rows):\n', np.round(Gd, 6))
Gd_m = np.einsum('iaj,ibj->ab', down[2] * lm[2][:, None, None], down[2] * lm[2][:, None, None])
print('  masked (2 rows) Gram:\n', np.round(Gd_m, 6), '\n  -> NOT orthonormal on the masked block; is_orthogonal() =', bool(uf.is_orthogonal()))
# the same polymorphic sweep on the RAGGED cores gives 3 at bond 2 (shapes):
(uc, dc, lc, rc), _ = fvc.t3_orthogonal_representations(x.data)
print('polymorphic t3_orthogonal_representations on ragged data: left core shapes', [c.shape for c in lc], 'down', [c.shape for c in dc])
# consequences for the tangent space (unsafe to get past the ORTH guard)
with safety.unsafe():
    v = ut3m.UNIFORM_MANIFOLD.randn(uf)
    print('tangent_space_dimension (uniform, from masks):', v.tangent_space_dimension,
          ' ragged manifold_dim at the ragged frame ranks:', ranks.compute_manifold_dim(shape, tuple(rf.up_ranks), tuple(rf.left_ranks)))
    print('MANIFOLD.norm(v) = %.6f   vs dense HS norm of v.to_dense() = %.6f   (frame claims orthogonal+gauged => should be equal)'
          % (float(ut3m.UNIFORM_MANIFOLD.norm(v)), np.linalg.norm(v.to_dense())))
    # ambient projection of a random tensor: uniform vs ragged (at the ragged frame) -- the projection is unique
    z = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
    pu = ut3m.UNIFORM_MANIFOLD.project_ambient(uf, ut3.UniformTuckerTensorTrain.from_t3(z)).to_dense()
    pr = t3m.MANIFOLD.project_ambient(rf, z).to_dense()
    print('project_ambient(frame(x), z): uniform vs ragged relerr = %.3e   (the tangent-space projection is unique; they must agree)'
          % (np.linalg.norm(pu - pr) / np.linalg.norm(pr)))
print('\nSafe mode, the library-built frame is rejected by the library:')
try:
    ut3m.UNIFORM_MANIFOLD.randn(uf)
except ValueError as e:
    print('  UNIFORM_MANIFOLD.randn(UNIFORM_MANIFOLD.frame(x)) ->', str(e)[:120], '...')
# and the x + y path (sums are non-minimal) hits it too
s = ux + ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)))
print('  frame(x + y).is_orthogonal() =', bool(ut3m.UNIFORM_MANIFOLD.frame(s).is_orthogonal()), '; ragged T3Frame.from_t3(x+y).is_orthogonal() =', bvf.T3Frame.from_t3(x.squash_tails() + t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))).is_orthogonal())
