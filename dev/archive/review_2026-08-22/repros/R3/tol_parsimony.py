"""R3: (a) the rtol truncation criterion actually used (tail Frobenius energy) vs the one the docs state
(per-singular-value threshold; linalg.truncated_svd signature comments, t3svd_verification.md Claim /
'The routine chooses r_k = #{ j : sigma_j >= tau_k }'), and (b) whether a tolerance-truncated t3svd
is always minimal (t3svd_minimal_ranks.md: 'the no-truncation and tolerance results are already minimal')."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.linalg as linalg

# (a) flat tail: per-sigma threshold keeps 1, tail energy keeps more
s = np.array([1.0] + [0.05] * 20)
A = (np.linalg.qr(np.random.RandomState(0).randn(21, 21))[0] * s) @ np.linalg.qr(np.random.RandomState(1).randn(21, 21))[0].T
rtol = 0.08
U, ss, Vt = linalg.truncated_svd(A, rtol=rtol)
per_sigma = int(np.sum(s >= rtol * np.linalg.norm(s)))
print('(a) truncated_svd(rtol=%.2f) on spectrum [1, 0.05 x20]:' % rtol)
print('    kept rank =', ss.shape[-1], '; per-sigma rule (#sigma >= rtol*||A||_F) would keep', per_sigma,
      '; signature-comment rule (#sigma >= rtol*sigma1) would keep', int(np.sum(s >= rtol * s[0])))
# tail-energy: smallest K with ||s[K:]|| < tol
tol = rtol * np.linalg.norm(s)
tails = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]
print('    code rule: #{j : ||s[j:]|| >= tol} =', int(np.sum(tails >= tol)))

# same thing through t3svd: a T3 whose mode-0 matricization has a flat tail
np.random.seed(0)
N = 8
T = np.zeros((N, N, N))
# build a tensor with mode-0 matricization spectrum [1, eps, eps, ...]
M = (np.linalg.qr(np.random.randn(N, N))[0] * np.array([1.0] + [0.06] * (N - 1))) @ np.linalg.qr(np.random.randn(N * N, N))[0].T
T = M.reshape(N, N, N)
x = t3.TuckerTensorTrain.t3svd_dense(T)[0]
_, full_tk, full_tt = x.t3svd()
xt, _, _ = x.t3svd(rtol=0.1)
tau = 0.1 * np.linalg.norm(np.asarray(xt.to_dense()))
print('(a2) t3svd(rtol=0.1) on a flat-tail tensor: chosen tucker ranks', xt.tucker_ranks,
      '; doc parsimony bound rho = #{sigma >= tau} per mode =', tuple(max(1, int(np.sum(s >= tau))) for s in full_tk),
      '; claim r_hat <= rho holds?', all(r <= max(1, int(np.sum(s >= tau))) for s, r in zip(full_tk, xt.tucker_ranks)))

# (b) tolerance truncation and minimality
print('(b) is t3svd(rtol=...) always minimal?')
rng = np.random.default_rng(0)
nonmin = 0; tot = 0
examples = []
for trial in range(400):
    d = int(rng.integers(2, 5))
    shape = tuple(int(v) for v in rng.integers(2, 6, size=d))
    tk = tuple(int(v) for v in rng.integers(1, 5, size=d))
    tt = (1,) + tuple(int(v) for v in rng.integers(1, 6, size=d - 1)) + (1,)
    x = t3.TuckerTensorTrain.randn(shape, tk, tt)
    # graded spectra: scale cores to induce decay
    tkc, ttc = x.data
    ttc = tuple(np.asarray(G) * (0.3 ** np.arange(G.shape[-1]))[None, None, :] * (0.5 ** np.arange(G.shape[-2]))[None, :, None] for G in ttc)
    x = t3.TuckerTensorTrain(tkc, ttc)
    rtol = float(10 ** rng.uniform(-3, -0.5))
    xt, _, _ = x.t3svd(rtol=rtol)
    tot += 1
    if not xt.has_minimal_ranks:
        nonmin += 1
        if len(examples) < 5:
            examples.append((shape, tk, tt, rtol, xt.tucker_ranks, xt.tt_ranks, xt.minimal_ranks))
print('    non-minimal tolerance-truncated results: %d/%d' % (nonmin, tot))
for e in examples:
    print('    ', e)
