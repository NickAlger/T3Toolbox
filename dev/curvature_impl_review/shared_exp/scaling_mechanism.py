"""Mechanism check: does rho / lam-RMS track residual/sigma_min, H_GN conditioning, or dimension?
usage: python scaling_mechanism.py D"""
import sys
import numpy as np
sys.argv = sys.argv[:2]
import scaling_tables as st   # prints the tables too; we only use its loaders below

print('\n=== mechanism, d =', st.D)
rows = [(1, 'unshared')] + [(m, g) for m in range(2, st.D + 1) for g in ('shared', 'unshared')]
R = [d for m, g in rows for d in st.load(m, g).values()]
X = {k: np.array([st.MEAS[k](d) if st.MEAS[k](d) is not None else np.nan for d in R], float)
     for k in ('rho', 'lrms', 'rt', 'mis', 'cond', 'dim', 'msig', 'sig')}
X['lmin'] = np.array([np.sort(np.abs(d['final']['lam_all']))[-1] for d in R])
geo = np.array([d['geom'] for d in R]); mm = np.array([d['m'] for d in R])
for a, b in (('rho', 'cond'), ('lrms', 'cond'), ('cond', 'dim'), ('mis', 'dim')):
    ok = ~np.isnan(X[a]) & ~np.isnan(X[b])
    print('Spearman %s~%s %+.2f (n=%d)' % (a, b, st.spearman(X[a][ok], X[b][ok]), ok.sum()))
for y in ('rho', 'lrms'):
    for cols in (('msig',), ('msig', 'cond'), ('mis', 'cond'), ('mis', 'cond', 'dim')):
        ok = ~np.isnan(X[y]) & np.all([~np.isnan(X[c]) for c in cols], axis=0)
        A = np.column_stack([np.ones(ok.sum())] + [np.log(X[c][ok]) for c in cols])
        b = np.log(X[y][ok])
        coef, res, *_ = np.linalg.lstsq(A, b, rcond=None)
        r2 = 1 - np.sum((b - A @ coef) ** 2) / np.sum((b - b.mean()) ** 2)
        print('log %-4s ~ %-22s slopes %s  R2 %.2f (n=%d)' % (y, '+'.join(cols), ' '.join('%+.2f' % c for c in coef[1:]), r2, ok.sum()))
print('cond H_GN median by m (shared / unshared):', ' '.join('m%d %.0f/%.0f' % (m, np.median(X['cond'][(mm == m) & ((geo == 'shared') | (m == 1))]), np.median(X['cond'][(mm == m) & (geo == 'unshared')])) for m in range(1, st.D + 1)))
# does m matter for curvature once the misfit is accounted for? (shared series incl. m = 1)
sel = (geo == 'shared') | (mm == 1)
for y in ('lrms', 'rt', 'rho'):
    A = np.column_stack([np.ones(sel.sum()), np.log(X['mis'][sel]), mm[sel]])
    b = np.log(X[y][sel])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    A0 = A[:, :2]
    c0, *_ = np.linalg.lstsq(A0, b, rcond=None)
    r2 = lambda AA, cc: 1 - np.sum((b - AA @ cc) ** 2) / np.sum((b - b.mean()) ** 2)
    print('shared series: log %-4s ~ log mis [+ m]: slope_mis %+.2f, per-m %+.3f (x%.2f per tied factor); R2 %.2f -> %.2f' % (
        y, coef[1], coef[2], np.exp(coef[2]), r2(A0, c0), r2(A, coef)))
