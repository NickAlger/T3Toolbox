"""Contraction order for one sweep step  'wa,aib,wi->wb'  in eager numpy: which pair first?
The library's numpy path is chosen from the subscript string alone, so it is the same for every size."""
import sys, time
import numpy as np
sys.path.append(sys.argv[1])
import t3toolbox.backend.contractions as c

rng = np.random.default_rng(0)
def best(f, k=7):
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return 1e3 * min(ts)

print('library numpy path for WCa,Caib,WCi->WCb :', c._pairwise_path('WCa,Caib,WCi->WCb'))
print('%-26s %16s %16s %16s' % ('sizes', 'mu with G first', 'G with xi first', 'library contract'))
for W, r, n in ((4096, 64, 8), (4096, 32, 32), (4096, 8, 64), (4096, 8, 256), (20000, 16, 16)):
    mu, G, xi = rng.standard_normal((W, r)), rng.standard_normal((r, n, r)), rng.standard_normal((W, n))
    a = best(lambda: np.einsum('wib,wi->wb', np.einsum('wa,aib->wib', mu, G, optimize=True), xi, optimize=True))
    b = best(lambda: np.einsum('wa,wab->wb', mu, np.einsum('aib,wi->wab', G, xi, optimize=True), optimize=True))
    l = best(lambda: c.contract('WCa,Caib,WCi->WCb', mu, G, xi))
    print('%-26s %13.2f ms %13.2f ms %13.2f ms' % ('W=%d r=%d n=%d' % (W, r, n), a, b, l))
