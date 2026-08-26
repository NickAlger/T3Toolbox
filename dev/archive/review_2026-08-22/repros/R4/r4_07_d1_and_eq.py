"""R4-7: (a) d=1 sweep over the tangent layer -- which ops crash; (b) T3Tangent eq/hash vs the docs' claim
that it is eq=False / identity-hashed."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.tv_operations as tv

np.random.seed(5)
print('--- (a) d = 1 ---')
for C in [(), (2,)]:
    x = t3.TuckerTensorTrain.randn((6,), (2,), (1, 1), stack_shape=C)
    frame = t3m.MANIFOLD.frame(x)
    v = t3m.MANIFOLD.randn(frame)
    y = t3.TuckerTensorTrain.randn((6,), (2,), (1, 1), stack_shape=C)
    ww = (np.random.randn(*(C + (6,))),)
    for name, f in [
        ('to_dense', lambda: v.to_dense().shape),
        ('to_t3', lambda: v.to_t3().structure),
        ('MANIFOLD.project/oblique', lambda: (t3m.MANIFOLD.project(v).is_gauged().all(), t3m.MANIFOLD.project_oblique(v).is_gauged().all())),
        ('MANIFOLD.inner/norm', lambda: (t3m.MANIFOLD.norm(v).shape)),
        ('MANIFOLD.retract', lambda: t3m.MANIFOLD.retract(v).structure),
        ('project_ambient(dense)', lambda: t3m.MANIFOLD.project_ambient(frame, y.to_dense()).to_dense().shape),
        ('project_ambient(T3)', lambda: t3m.MANIFOLD.project_ambient(frame, y).to_dense().shape),
        ('project_ambient(dense, t3svd)', lambda: t3m.MANIFOLD.project_ambient(frame, y.to_dense(), method='t3svd').to_dense().shape),
        ('transport', lambda: t3m.MANIFOLD.transport(v, t3m.MANIFOLD.frame(y)).to_dense().shape),
        ('probe/apply/entries', lambda: (v.probe(ww)[0].shape, v.apply(ww).shape, v.entries(np.array([[1]])).shape)),
        ('probe_transpose', lambda: t3m.T3Tangent.probe_transpose(v.probe(ww), ww, frame).variations.d),
        ('unstack/stack_frame', lambda: (t3m.T3Tangent.stack_frame(v.unstack_frame()).stack_shape if C else 'n/a')),
        ('COREWISE retract', lambda: t3m.COREWISE.retract(t3m.COREWISE.randn(t3m.COREWISE.frame(x))).structure),
        ('tv_project_t3_onto_tangent_space (backend)', lambda: [a.shape for a in tv.tv_project_t3_onto_tangent_space(frame.data, y.data)[0]]),
    ]:
        try:
            print(f'  C={C} {name:44s} OK  {f()}')
        except Exception as e:
            print(f'  C={C} {name:44s} {type(e).__name__}: {str(e)[:90]}')

# dense ground truth for the dense projection at d=1 (the one path that works): P_T z = U^T U z  for a rank-n Tucker
x = t3.TuckerTensorTrain.randn((6,), (2,), (1, 1)); frame = t3m.MANIFOLD.frame(x)
z = np.random.randn(6)
# tangent space of the rank-2 "matrix" manifold in R^6 = whole R^6 restricted... at d=1 the manifold is {U^T g}: dim = 6 (N*n - n^2 + n*... )
print('  d=1 manifold_dim((6,),(2,),(1,1)) =', t3m.manifold_dim(((6,), (2,), (1, 1))),
      ' projection of z recovers z:', np.allclose(t3m.MANIFOLD.project_ambient(frame, z).to_dense(), z))

print('--- (b) T3Tangent eq / hash (docs/batching_and_stacking.md:360 says frozen, eq=False -> identity-hashed) ---')
x = t3.TuckerTensorTrain.randn((5, 6), (2, 3), (1, 2, 1)); frame = t3m.MANIFOLD.frame(x)
v = t3m.MANIFOLD.randn(frame); v2 = t3m.T3Tangent(frame, v.variations)
import dataclasses
print('  T3Tangent dataclass params eq =', t3m.T3Tangent.__dataclass_params__.eq, ' T3Frame eq =', bvf.T3Frame.__dataclass_params__.eq)
for name, f in [('hash(v)', lambda: hash(v)), ('v == v2', lambda: v == v2), ('v == v', lambda: v == v), ('{v: 1}', lambda: {v: 1})]:
    try:
        print(f'  {name:10s} ->', f())
    except Exception as e:
        print(f'  {name:10s} -> {type(e).__name__}: {str(e)[:80]}')
print('  hash(frame) ok ->', isinstance(hash(frame), int))
