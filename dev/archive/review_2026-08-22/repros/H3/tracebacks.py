import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_frame_variations_format as ubv, t3toolbox.uniform_manifold as ut3m
np.random.seed(0)
print('=== A. retract on a slack-rank frame (unstacked) ===')
x = t3.TuckerTensorTrain.randn((5,6,7),(2,3,4),(1,2,3,1))
frame, _ = bvf.t3_orthogonal_representations(x)
print('x.has_minimal_ranks', x.has_minimal_ranks, 'frame ranks up/down/left/right', frame.up_ranks, frame.down_ranks, frame.left_ranks, frame.right_ranks, 'frame.has_minimal_ranks', frame.has_minimal_ranks, 'is_orthogonal', frame.is_orthogonal())
v = t3m.MANIFOLD.randn(frame)
try:
    r = t3m.MANIFOLD.retract(v); print('retract OK', r.ranks)
except Exception as e:
    traceback.print_exc()
print('-- same with COREWISE.retract:')
try:
    r = t3m.COREWISE.retract(v); print('COREWISE retract OK', r.ranks)
except Exception as e:
    print('COREWISE retract RAISES', repr(e))
print('-- frame via T3Frame.random_orthogonal (docstring says slack is allowed):')
fr2 = bvf.T3Frame.random_orthogonal((5,6,7),(4,4,4),(1,2,2,1))
print('ranks', fr2.up_ranks, fr2.down_ranks, fr2.left_ranks, fr2.right_ranks)
v2 = t3m.MANIFOLD.randn(fr2)
try:
    r = t3m.MANIFOLD.retract(v2); print('retract OK', r.ranks)
except Exception as e:
    print('retract RAISES', repr(e))
print('-- v.to_t3(include_shift=True) on slack frame:')
try:
    print(v.to_t3(include_shift=True).ranks)
except Exception as e:
    print('to_t3 RAISES', repr(e))
print('\n=== A2. uniform retract on slack frame ===')
uframe = ubv.UT3Frame.from_t3frame(frame)
uv = ut3m.UT3Tangent.from_t3tangent(v)
try:
    r = ut3m.UNIFORM_MANIFOLD.retract(uv); print('uniform retract OK', r.tucker_ranks)
except Exception as e:
    traceback.print_exc()
print('\n=== A3. d=4 K=(2,) uniform reshape failure ===')
x4 = t3.TuckerTensorTrain.randn((5,6,7,3),(2,3,4,2),(1,2,3,2,1))
f4, _ = bvf.t3_orthogonal_representations(x4)
v4 = t3m.MANIFOLD.randn(f4, stack_shape=(2,))
uv4 = ut3m.UT3Tangent.from_t3tangent(v4)
uf4 = ubv.UT3Frame.from_t3frame(f4)
for nm, fn in [('to_dense', lambda: uv4.to_dense()), ('corewise_inner', lambda: uv4.corewise_inner(uv4)), ('UNIFORM_MANIFOLD.inner', lambda: ut3m.UNIFORM_MANIFOLD.inner(uv4, uv4)),
               ('project', lambda: ut3m.UNIFORM_MANIFOLD.project(uv4)), ('project_oblique', lambda: ut3m.UNIFORM_MANIFOLD.project_oblique(uv4)), ('retract', lambda: ut3m.UNIFORM_MANIFOLD.retract(uv4)),
               ('COREWISE.retract', lambda: ut3m.UNIFORM_COREWISE.retract(uv4)), ('normalized', lambda: uv4.normalized()), ('reverse', lambda: uv4.reverse()),
               ('sum_tangents', lambda: uv4.sum_tangents()), ('unstack_tangents', lambda: uv4.unstack_tangents()), ('unstack_frame', lambda: uv4.unstack_frame()), ('to_t3tangent', lambda: uv4.to_t3tangent()),
               ('weighted', lambda: None)]:
    try:
        fn(); print(nm, 'OK')
    except Exception as e:
        print(nm, 'RAISES', repr(e)[:150]); traceback.print_exc(limit=-4)
print('\n=== B. d=1 project_ambient ===')
x1 = t3.TuckerTensorTrain.randn((6,),(3,),(1,1))
f1, _ = bvf.t3_orthogonal_representations(x1)
g1 = t3.TuckerTensorTrain.randn((6,),(3,),(1,1))
for nm, fn in [('project_ambient(T3)', lambda: t3m.MANIFOLD.project_ambient(f1, g1)), ('project_ambient(dense)', lambda: t3m.MANIFOLD.project_ambient(f1, g1.to_dense())),
               ('transport', lambda: t3m.MANIFOLD.transport(t3m.MANIFOLD.randn(f1), f1)), ('retract', lambda: t3m.MANIFOLD.retract(t3m.MANIFOLD.randn(f1)))]:
    try:
        fn(); print(nm, 'OK')
    except Exception as e:
        print(nm, 'RAISES', repr(e)); traceback.print_exc(limit=-3)
print('\n=== C. uniform __mul__ ===')
ux = ut3.UniformTuckerTensorTrain.from_t3(x); uy = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((5,6,7),(2,3,4),(1,2,3,1)))
print(ut3.UniformTuckerTensorTrain.__mul__.__doc__)
try:
    (ux * uy); print('ux*uy OK')
except Exception as e:
    print('ux*uy RAISES', repr(e)[:200])
