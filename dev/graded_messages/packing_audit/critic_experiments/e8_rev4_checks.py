# E8 (round 3): (a) packing a probe full weight whose W axis is 1 in one mode and full in another (legal per mode
# under rev. 4's rule) -- the one packer refuses; (b) the display's table with the order read off a fieldless oracle
# (None) against the order given.
import numpy as np
import t3toolbox.backend.ut3_operations as uo
import t3toolbox.backend.optimizer_display as bdisp
try:
    uo.pack_vectors([np.ones((3, 1, 4)), np.ones((3, 5, 5))])          # F=(3,), mode 0 W=1, mode 1 W=(5,)
    print('(a) packed')
except ValueError as e:
    print('(a) refused:', str(e)[:120])
err = np.array([[0.1, 0.2], [0.3, 0.4]])                                 # probe, d=2, order=(1,): block shape (2, 2)
fn = [n for n in dir(bdisp) if n.startswith('_format') or n.startswith('format')]
print(fn)
err3 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])   # probe d=3, order=(1, 1)? no: (2,) -> F=(3,)
print('(b) order given (probe, order=(2,)):'); print(bdisp._format_table(err3, None, '%.2g', order=(2,)))
print('(b) order None (a fieldless oracle):'); print(bdisp._format_table(err3, None, '%.2g', order=None))
e4 = np.arange(12.0).reshape(3, 2, 2) / 100                            # probe d=3, order=(1, 1): F=(2, 2)
print('(b) two jet variables, order given:'); print(bdisp._format_table(e4, None, '%.2g', order=(1, 1)))
print('(b) two jet variables, order None:'); print(bdisp._format_table(e4, None, '%.2g', order=None))
