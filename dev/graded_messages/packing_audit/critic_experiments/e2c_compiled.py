# E2c: optimized (compiled) HLO size vs d, for paths with and without the data-space whole steps
import re
import numpy as np
import jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.sampling_variables as sv
import t3toolbox.backend.ut3_operations as uo

def n_instr(txt):
    return sum(1 for line in txt.splitlines() if re.match(r'\s+(ROOT\s+)?%?[\w.\-]+ = ', line))

for kind, want in (('probe', ('xi',)), ('probe', ('z',)), ('probe', ('f',)), ('probe', ('f', 'g')),
                   ('apply', ('f', 'g'))):
    out = []
    for d in (3, 5, 8):
        rng = np.random.default_rng(0)
        shape = tuple(4 + (i % 3) for i in range(d))
        x = t3.TuckerTensorTrain.randn(shape, (2,) * d, (1,) + (2,) * (d - 1) + (1,))
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, N=7)
        fr = ut3m.UNIFORM_MANIFOLD.frame(ux).data
        ww = uo.pack_vectors([rng.standard_normal((2, 3, n)) for n in shape], 7, len_P=1)
        y = (uo.pack_vectors([rng.standard_normal((2, 3, n)) for n in shape], 7) if kind == 'probe'
             else rng.standard_normal((2, 3)))
        def f(D, U, L, R, w, yy):
            return sv.sampling_variables(kind, (D, U, L, R, fr[4], fr[5]), w, want, order=(1,), y=yy)
        args = [jnp.asarray(a) for a in fr[:4]] + [jnp.asarray(ww), jnp.asarray(y)]
        txt = jax.jit(f).lower(*args).compile().as_text()
        out.append(n_instr(txt))
    print(kind, want, 'compiled HLO instructions at d = 3, 5, 8:', out)
