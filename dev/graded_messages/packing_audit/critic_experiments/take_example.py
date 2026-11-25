import numpy as np, dataclasses as dc
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
import t3toolbox.backend.jets as jets

shape = (4, 5, 6)
order = (2,)                                   # F = (3,): slots 0, 1, 2
n = 10                                         # W = (10,)
rng = np.random.default_rng(0)
ww = jets.affine_jet([rng.standard_normal((n, N)) for N in shape],
                     [rng.standard_normal((n, N)) for N in shape])   # input jets along a line: (2, n, Ni)
y = rng.standard_normal((3, n))                # apply data: F + W
blocks = np.array([1.0, 2.0, 3.0])             # the OLD form: one weight per slot, shape F = (3,)
oracle = LeastSquaresOracle('apply', order)
batch = LeastSquaresBatch(ww, y, blocks)

print('1. the full batch, checked:')
try:
    oracle.check_batch(batch, shape)
except ValueError as e:
    print('   ValueError:', str(e)[:150], '...')

print('2. take, as built (M2 part 4):')
try:
    oracle.take(batch, np.array([0, 3, 7]))
except ValueError as e:
    print('   ValueError:', str(e)[:200], '...')

print('3. take WITHOUT the raise (the two lines it guards, by hand):')
m, n_w = 1, 1
s = blocks.shape                               # (3,)
W_part = s[m:m + n_w]                          # () : the weight has no W axis at all
print('   the W part of the weight:', W_part, '-> all(a == 1 for a in ()) is', all(a == 1 for a in W_part))
drawn_weight = blocks.reshape(s[:m] + (1,) + s[m + n_w:])
print('   the drawn weight:', drawn_weight.shape)
sub = LeastSquaresBatch([w[:, [0, 3, 7]] for w in ww], y[:, [0, 3, 7]], drawn_weight)
oracle.check_batch(sub, shape)
print('   check_batch on the drawn batch: passes')
