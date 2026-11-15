"""Target decision 2.4: what a tilde jet is, read off the algebra of truncated polynomials.

A jet of order n is an element of R = R[s] / (s^(n+1)).  Every term of the sampling program is the product of
R (the truncated Cauchy convolution).  R carries a pairing made from its own product,

    beta(a, b) = [s^n] (a b) = sum_t a[t] b[n-t],

the top coefficient of the product.  It is nondegenerate and invariant: beta(a b, c) = beta(a, b c).  So, with
jets and tilde jets paired by beta, multiplication by a jet is its own transpose, and the adjoint of every term
is the same product as the term.

This script checks those statements on numbers.  It uses numpy only and reads nothing from T3Toolbox.

Usage:  python3 -I tilde_jet_pairing_check.py
"""
import math

import numpy as np


def mul(a, b):                      # the product of R: truncated Cauchy convolution
    n = a.shape[0] - 1
    return np.convolve(a, b)[:n + 1]


def top(a):                         # [s^n]
    return a[-1]


def beta(a, b):
    return top(mul(a, b))


def rev(c):                         # the reciprocal polynomial, s^n c(1/s)
    return c[::-1]


def mul_matrix(a):                  # the matrix of b -> a b on coefficients: lower triangular Toeplitz
    n = a.shape[0] - 1
    return np.array([[a[t - j] if t >= j else 0.0 for j in range(n + 1)] for t in range(n + 1)])


def main():
    rng = np.random.default_rng(0)
    n = 4
    a, b, c = (rng.standard_normal(n + 1) for _ in range(3))
    J = np.eye(n + 1)[::-1]
    M = mul_matrix(a)
    rel = lambda x, y: np.linalg.norm(np.ravel(x) - np.ravel(y)) / np.linalg.norm(np.ravel(y))

    print('order n = %d' % n)
    print('1. the pairing')
    print('   beta(a, b) is the dot product of a with b reversed:            %.0e' % abs(beta(a, b) - a @ rev(b)))
    print('   beta is invariant, beta(a b, c) = beta(a, b c):                %.0e' % abs(beta(mul(a, b), c) - beta(a, mul(b, c))))
    print('   beta is nondegenerate (its matrix is the flip, determinant %+.0f)' % np.linalg.det(J))

    print('2. the transpose of a product')
    print('   flip . (matrix of a)^T . flip is the matrix of a:              %.0e' % rel(J @ M.T @ J, M))
    print('   so  (matrix of a)^T c  =  rev( rev(c) a ):                     %.0e' % rel(M.T @ c, rev(mul(rev(c), a))))
    corr = np.array([sum(a[t - j] * c[t] for t in range(j, n + 1)) for j in range(n + 1)])
    print('   in natural order the same thing is a correlation, sum_{t>=j} a[t-j] c[t]:  %.0e' % rel(M.T @ c, corr))
    laurent = np.convolve(a, rev(c))          # a(s) c(1/s) s^n : powers s^0 .. s^(2n); non-positive part = s^0 .. s^n
    print('   which is the non-positive part of a(s) c(1/s), c read as a polynomial in 1/s:  %.0e'
          % rel(M.T @ c, rev(laurent[:n + 1])))

    print('3. a chain of two products, z = (a x) y: sensitivities by beta against finite differences')
    x, y = rng.standard_normal(n + 1), rng.standard_normal(n + 1)
    resid = rng.standard_normal(n + 1)                    # d f / d z, slot by slot (natural order)
    z_tilde = rev(resid)                                  # the seed: reversed once
    u_tilde = mul(z_tilde, y)                             # tilde of u = a x : the same product
    x_tilde = mul(u_tilde, a)                             # tilde of x      : the same product
    f = lambda xx: resid @ mul(mul(a, xx), y)
    eps = 1e-6
    fd = np.array([(f(x + eps * e) - f(x - eps * e)) / (2 * eps) for e in np.eye(n + 1)])
    print('   x_tilde[t] is d f / d x[n-t]:                                  %.0e' % rel(rev(x_tilde), fd))
    dx = rng.standard_normal(n + 1)
    print('   the directional derivative is beta(x_tilde, dx):               %.0e' % abs(beta(x_tilde, dx) - fd @ dx))
    print('   x_tilde[0] reads only the top residual: d x_tilde[0] / d resid = %s'
          % np.array2string(np.array([mul(mul(rev(e), y), a)[0] for e in np.eye(n + 1)]) != 0))

    print('4. a constant (order-0) factor: the readout is the top coefficient')
    g = 0.7                                               # a core entry: no dependence on s
    fg = lambda gg: resid @ (gg * mul(a, y))
    print('   d f / d g  =  top( z_tilde a y ):                              %.0e'
          % abs(top(mul(mul(z_tilde, a), y)) - (fg(g + eps) - fg(g - eps)) / (2 * eps)))

    print('5. in derivative coordinates (slot t holds the t-th derivative, t! times the coefficient)')
    D = np.diag([float(math.factorial(t)) for t in range(n + 1)])
    N = D @ M @ np.linalg.inv(D)                          # binomial convolution
    print('   the flip alone is not the transpose:      |flip N^T flip - N| / |N| = %.1f' % rel(J @ N.T @ J, N))
    S = D @ J @ D                                         # t! (n-t)! on the antidiagonal
    print('   it needs the weights t!(n-t)!:            |S N^T S^-1 - N| / |N|    = %.0e' % rel(S @ N.T @ np.linalg.inv(S), N))

    print('6. a tilde jet as an ordinary Taylor jet: run the PLAIN transpose at each s, with the seed c(s) = sum_t resid[t] s^(n-t)')
    P = np.polynomial.polynomial
    ss = np.linspace(-1.0, 1.0, 3 * n + 1)                       # enough points to recover a polynomial of degree 3n
    seed_at = P.polyval(ss, rev(resid))                          # the reciprocal polynomial of the residual, at each s
    x_tilde_at = seed_at * P.polyval(ss, y) * P.polyval(ss, a)   # plain adjoint of x at each s: no jets anywhere
    coef = P.polyfit(ss, x_tilde_at, 3 * n)                      # its Taylor coefficients in s
    print('   its Taylor coefficients 0..n are the tilde jet x_tilde:        %.0e' % rel(coef[:n + 1], x_tilde))
    g_at = seed_at * P.polyval(ss, a) * P.polyval(ss, y)         # plain gradient with respect to the constant factor g
    print('   and d f / d g is its coefficient of s^n:                       %.0e'
          % abs(P.polyfit(ss, g_at, 3 * n)[n] - (fg(g + eps) - fg(g - eps)) / (2 * eps)))


if __name__ == '__main__':
    main()
