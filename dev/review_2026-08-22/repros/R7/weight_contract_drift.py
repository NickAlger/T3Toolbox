"""R7: the omega contract implemented twice -- find inputs the frontend and the backend treat differently."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.fitting as fitting
import t3toolbox.optimizers as topt
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.uniform_tucker_tensor_train as ut3

np.random.seed(0)
shape, tr, rr = (4, 5, 6), (2, 2, 2), (1, 2, 2, 1)
d = 3
x = t3.TuckerTensorTrain.randn(shape, tr, rr)
ww = [np.random.randn(7, N) for N in shape]
r = [np.random.randn(7, N) for N in shape]

# 1. mode dim m > d on plain probe: frontend rejects; backend accepts and silently uses only the first d rows
w_long = np.array([2.0, 3.0, 5.0, 7.0, 11.0])   # d+2 entries
try:
    fitting.probe_model(t3m.MANIFOLD, x, ww, r, weight=w_long); print("frontend probe_model(weight len d+2): accepted")
except ValueError as e:
    print("frontend probe_model(weight len d+2): ValueError:", str(e)[:80])
k_long = bfit.probe_kind(w_long)
k_3 = bfit.probe_kind(w_long[:d])
print("backend probe_kind(weight len d+2): constructed, weight.shape =", k_long.weight.shape)
print("   sumsq(r) with len-5 weight = %.6f ; with its first 3 rows = %.6f ; equal=%s (extra rows silently ignored)"
      % (float(k_long.sumsq(r, 1)), float(k_3.sumsq(r, 1)), np.isclose(float(k_long.sumsq(r, 1)), float(k_3.sumsq(r, 1)))))
# the same on probe_derivatives (weight (d+2, order+1))
w2 = np.random.rand(d + 2, 3)
try:
    fitting.probe_derivatives_model(t3m.MANIFOLD, x, ww, ww, 2, [np.random.randn(3, 7, N) for N in shape], weight=w2); print("frontend probe_derivatives_model((d+2,o+1)): accepted")
except ValueError as e:
    print("frontend probe_derivatives_model((d+2,o+1)): ValueError:", str(e)[:80])
kd = bfit.probe_derivatives_kind(2, w2)
rd = [np.random.randn(3, 7, N) for N in shape]
print("backend probe_derivatives_kind((d+2,o+1)): constructed; sumsq == sumsq with w2[:d]: %s"
      % np.isclose(float(kd.sumsq(rd, 1)), float(bfit.probe_derivatives_kind(2, w2[:d]).sumsq(rd, 1))))
# mode dim m < d on backend -> what error?
try:
    bfit.probe_kind([1.0, 2.0]).sumsq(r, 1); print("backend probe_kind(len d-1).sumsq: no error")
except Exception as e:
    print("backend probe_kind(len d-1).sumsq ->", type(e).__name__, str(e)[:60])

# 2. 2-D (d,1) on plain probe: documented frontend-reject / backend-accept
try:
    fitting.probe_model(t3m.MANIFOLD, x, ww, r, weight=[[0.5], [1.0], [2.0]]); print("frontend (d,1) plain probe: accepted")
except ValueError as e:
    print("frontend (d,1) plain probe: ValueError (documented)")
print("backend probe_kind((d,1)): weight.shape =", bfit.probe_kind([[0.5], [1.0], [2.0]]).weight.shape)

# 3. aliasing: frontend np.asarray; does the stored kind weight alias the user's array? (ragged + uniform)
w_user = np.array([0.5, 1.0, 2.0])
m = fitting.probe_model(t3m.MANIFOLD, x, ww, r, weight=w_user)
print("ragged kind.weight shares memory with user array:", np.shares_memory(m.kind.weight, w_user))
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
um = fitting.probe_model(t3m.UNIFORM_MANIFOLD if hasattr(t3m, 'UNIFORM_MANIFOLD') else __import__('t3toolbox.uniform_manifold', fromlist=['x']).UNIFORM_MANIFOLD, ux, ww, r, weight=w_user)
print("uniform kind.weight shares memory with user array:", np.shares_memory(um.kind.weight, w_user), "; writeable:", um.kind.weight.flags.writeable)
s0 = float(m.objective_value); w_user[0] = 100.0
print("mutating user array after model build changes ragged objective? %s" % (not np.isclose(s0, float(m.objective_value))))

# 4. order dim: frontend 'order or 0' for plain kinds; a derivative kind with a bare vector of wrong length
try:
    topt.newton_cg(t3m.MANIFOLD, 'apply_derivatives', (ww, ww), np.zeros((3, 7)), x, order=2, weight=[1.0, 2.0], max_newton=1)
except ValueError as e:
    print("frontend apply_derivatives weight len 2 for order 2: ValueError:", str(e)[:70])
