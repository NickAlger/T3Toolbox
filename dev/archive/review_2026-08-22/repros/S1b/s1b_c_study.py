"""Q3 check: the two-sided failure of the separation constant c.

Side 1 (too small): c = 2*sigma_max puts the {pin,data} threshold c/2 EXACTLY at sigma_max.
The classifier is `Sa > c/2` with Sa and ||A||_2 computed by two DIFFERENT SVDs (of the
augmented core and of A_pad); whenever rounding lands the augmented sigma one ulp above the
other, the LARGEST triplet is classified as a pin and silently deleted.  Generic (irrational)
rank-1 data triggers it at a high rate; exactly-representable data is saved by the strict `>`
-- which is why it survives casual testing.

Side 2 (too large): the feared eps*c accuracy floor does NOT materialize -- REFUTED here.
The naive worry: with pins present (n < M) the augmented core has norm ~c, so a dense
backward error ~eps*c would pollute the kept sigmas.  Measured: sigma error stays ~1e-16
even at c = 1e12*||A||_F, pins or no pins.  Why: the pins occupy BITWISE-ZERO rows of B, so
the augmented core is bitwise block-diagonal, and Householder zero-preservation keeps the
blocks separated through the bidiagonalization -- the data block's effective backward error
is ~eps*||B||, not eps*c.  The binding constraints on c are therefore only side 1's margin
(from below) and dtype overflow (from above; relevant for float32): 4*||A||_F satisfies both.
"""
import numpy as np
from s1b_sym_variant import pad_safe_svd_sym

# ---------------- Side 1: threshold exactly at sigma_max ----------------
rng = np.random.default_rng(0)
fail2 = fail4 = 0; trials = 400
N, M = 7, 3
row_pad = np.array([0,0,1,0,0,1,0], bool); col_pad = np.array([0,0,1], bool)
for trial in range(trials):
    u = rng.standard_normal(5); v = rng.standard_normal(2)      # generic: sigma irrational
    A = np.zeros((N, M)); A[np.ix_(~row_pad, ~col_pad)] = np.outer(u, v)
    sig = np.linalg.norm(u) * np.linalg.norm(v)
    _, S2, _ = pad_safe_svd_sym(A, row_pad, col_pad, seed=trial, c_factor=2.0, use_spectral=True)
    _, S4, _ = pad_safe_svd_sym(A, row_pad, col_pad, seed=trial, c_factor=4.0, use_spectral=False)
    fail2 += not np.isclose(S2[0], sig)
    fail4 += not np.isclose(S4[0], sig)
print(f"side 1 (generic rank-1): c=2*spectral deleted the top triplet in {fail2}/{trials} cases; "
      f"c=4*Frobenius in {fail4}/{trials}")

# ---------------- Side 2: eps*c floor -- ONLY when pins exist (n < M) ----------------
rng = np.random.default_rng(1)
sig_true = np.array([3.0, 1.0, 1e-4, 1e-8])
for tag, N, M, n_real in [("n >= M (no pins: c never enters)", 40, 6, 37),
                          ("n <  M (2 pins at sigma = c)   ", 40, 6, 4)]:
    row_pad = np.ones(N, bool); row_pad[np.linspace(0, N - 1, n_real).astype(int)] = False
    col_pad = np.zeros(M, bool); col_pad[4:] = True
    m = M - int(col_pad.sum()); q = min(n_real, m)
    Uo, _ = np.linalg.qr(rng.standard_normal((n_real, q))); Vo, _ = np.linalg.qr(rng.standard_normal((m, q)))
    A = np.zeros((N, M)); A[np.ix_(~row_pad, ~col_pad)] = Uo * sig_true[:q] @ Vo.T
    nA = np.linalg.norm(A)
    print(f"side 2, {tag}  true sigmas {sig_true[:q]}:")
    for cf in [4.0, 1e6, 1e12]:
        U, S, V = pad_safe_svd_sym(A, row_pad, col_pad, seed=0, c_factor=cf)
        err = np.max(np.abs(S[:q] - sig_true[:q]))
        clean = bool(np.all(U[row_pad][:, :q] == 0.0))
        print(f"  c = {cf:5.0e} * ||A||_F : max |S - sigma_true| = {err:.2e}  "
              f"(eps*c ~ {2.2e-16*cf*nA:.1e}); pads bitwise 0: {clean}")
