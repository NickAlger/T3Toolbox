"""E3: the engine's two-holes at jets are the jvp of its corewise transposes at jets (the re-base of
TestCurvatureJvpOracle's derivative rows), ragged probe / apply / apply at the one-hot jet, and uniform probe."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as bapply
import t3toolbox.backend.ut3_sampling as ut3s
import t3toolbox.backend.ut3_operations as uops
from t3toolbox.backend.jets import affine_jet, index_to_one_hot
np.random.seed(97)
STRUCT = ((4, 5, 6), (2, 3, 2), (1, 2, 3, 1))
W, order = 6, 2
rnd = lambda *s: jnp.asarray(np.random.randn(*s))
x = t3.TuckerTensorTrain.randn(*STRUCT).to_jax().data
dirs = (tuple(rnd(2, *B.shape) for B in x[0]), tuple(rnd(2, *G.shape) for G in x[1]))
ww = tuple(rnd(W, N) for N in STRUCT[0]); pp = tuple(rnd(W, N) for N in STRUCT[0])
line = affine_jet(ww, pp)
index = np.stack([np.random.randint(0, N, size=W) for N in STRUCT[0]])
oh_line = affine_jet([jnp.asarray(e) for e in index_to_one_hot(index, STRUCT[0])], pp)
zt = tuple(rnd(order + 1, W, N) for N in STRUCT[0]); cj = rnd(order + 1, W)
def check(name, T, H, cp, dd):
    got = H(cp, dd)
    for k in range(2):
        dk = jax.tree_util.tree_map(lambda a: a[k], dd)
        _, jvp = jax.jvp(T, (cp,), (dk,))
        ref = jax.tree_util.tree_leaves(jvp); out = [a[k] for a in jax.tree_util.tree_leaves(got)]
        num = np.sqrt(sum(np.linalg.norm(np.asarray(a) - np.asarray(b)) ** 2 for a, b in zip(out, ref)))
        den = np.sqrt(sum(np.linalg.norm(np.asarray(b)) ** 2 for b in ref))
        print(name, k, 'rel %.1e' % (num / den), 'den %.1e' % den)
check('probe jets', lambda th: probing.t3_probe_corewise_transpose(th, line, zt, order=(order,)),
      lambda th, d: probing.t3_probe_corewise_hessian(th, line, zt, d, order=(order,)), x, dirs)
check('apply jets', lambda th: bapply.t3_apply_corewise_transpose(th, line, cj, order=(order,)),
      lambda th, d: bapply.t3_apply_corewise_hessian(th, line, cj, d, order=(order,)), x, dirs)
check('apply one-hot jets', lambda th: bapply.t3_apply_corewise_transpose(th, oh_line, cj, order=(order,)),
      lambda th, d: bapply.t3_apply_corewise_hessian(th, oh_line, cj, d, order=(order,)), x, dirs)
ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 3, 1)))
Usc, Gsc, shape, masks = ux.data
lineu = uops.pack_if_ragged(affine_jet(ww, pp), Usc.shape[-1], len_P=1)
ztu = uops.pack_if_ragged(zt, Usc.shape[-1])
du = (rnd(*((Usc.shape[0], 2) + Usc.shape[1:])), rnd(*((Gsc.shape[0], 2) + Gsc.shape[1:])))   # garbage pad
T = lambda th: ut3s.ut3_probe_corewise_transpose((th[0], th[1], shape, masks), lineu, ztu, order=(order,))
H = lambda th, dd: tuple(jnp.moveaxis(a, 1, 0) for a in ut3s.ut3_probe_corewise_hessian(
    (th[0], th[1], shape, masks), lineu, ztu, tuple(jnp.moveaxis(a, 0, 1) for a in dd), order=(order,)))
check('uniform probe jets', T, H, (jnp.asarray(Usc), jnp.asarray(Gsc)), tuple(jnp.moveaxis(a, 1, 0) for a in du))
