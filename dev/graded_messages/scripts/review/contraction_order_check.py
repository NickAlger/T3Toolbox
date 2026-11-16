"""The contraction order inside a term (agenda row 5.5), checked on small arrays.

Part 1.  A three-operand term has three contraction orders.  np.einsum and jnp.einsum take an explicit path, a
         path names operands by position, and so one path serves a grouped string and its expansion.
Part 2.  With jets.  eta[t] = sum over i + r = t of eta_term(mu[i], G, nu[r]), both mu and nu full jets.
         Read "a path orders the pairwise products, and a pairwise product of two operands that both carry the
         order axis is their truncated product", every one of the three orders gives the explicit sum.

Nothing is timed.  Usage: python3 -I contraction_order_check.py
"""
import numpy as np

TOL = 1e-12
PATHS = {                                     # a path for three operands (0, 1, 2): a tuple of pairs of positions
    '(0, 1) first': ((0, 1), (0, 1)),
    '(1, 2) first': ((1, 2), (0, 1)),
    '(0, 2) first': ((0, 2), (0, 1)),
}
# The two backends spell the same path differently: numpy wants a list that begins with 'einsum_path';
# jax (opt_einsum) wants the bare list of pairs and rejects the prefix.


def relerr(x, ref):
    return float(np.linalg.norm(np.asarray(x) - ref) / np.linalg.norm(ref))


def part1():
    print('Part 1.  mu_term(mu, G, xi) with two sample axes: one string, three paths')
    rng = np.random.default_rng(0)
    W1, W2, a, n, b = 3, 4, 5, 6, 7
    mu = rng.standard_normal((W1, W2, a))
    G = rng.standard_normal((a, n, b))
    xi = rng.standard_normal((W1, W2, n))
    sub = 'xya,aib,xyi->xyb'                  # the grouped 'Wa,aib,Wi->Wb' with W expanded into two letters
    ref = np.einsum(sub, mu, G, xi, optimize=False)
    for name, path in PATHS.items():
        numpy_path = ['einsum_path', *path]
        ok = relerr(np.einsum(sub, mu, G, xi, optimize=numpy_path), ref) < TOL
        info = np.einsum_path(sub, mu, G, xi, optimize=numpy_path)[1]
        largest = [ln for ln in info.splitlines() if 'Largest intermediate' in ln][0].split(':')[1].strip()
        print(f'  numpy  {name}: equal to the single einsum {ok};  largest intermediate {largest}')
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print('  jax: skipped (not installed)')
        return
    for name, path in PATHS.items():
        ok = relerr(jnp.einsum(sub, mu, G, xi, optimize=list(path)), ref) < 1e-4       # jax is float32 by default
        print(f'  jax    {name}: equal to the single einsum {ok}')


def truncated_product(x, y, sub, P):          # x, y carry the order axis first; sub contracts one slot of each
    out = None
    for r in range(P):
        part = np.einsum(sub, x[:P - r], y[r])
        if out is None:
            out = np.zeros((P,) + part.shape[1:])
        out[r:] += part
    return out


def part2():
    print('Part 2.  eta_term(mu, G, nu) with mu and nu full jets: the three orders as pairwise truncated products')
    rng = np.random.default_rng(1)
    P, W, a, n, b = 5, 7, 3, 4, 6             # P = order + 1
    mu = rng.standard_normal((P, W, a))
    nu = rng.standard_normal((P, W, b))
    G = rng.standard_normal((a, n, b))
    ref = np.zeros((P, W, n))
    for i in range(P):
        for r in range(P - i):
            ref[i + r] += np.einsum('Wa,aib,Wb->Wi', mu[i], G, nu[r])

    G_nu = np.einsum('aib,rWb->rWai', G, nu)                      # the core has no order axis: P contractions
    e1 = truncated_product(mu, G_nu, 'tWa,Wai->tWi', P)
    mu_G = np.einsum('rWa,aib->rWib', mu, G)
    e2 = truncated_product(nu, mu_G, 'tWb,Wib->tWi', P)
    mu_nu = truncated_product(mu, nu, 'tWa,Wb->tWab', P)          # the two jets first
    e3 = np.einsum('tWab,aib->tWi', mu_nu, G)

    rows = (('(G, nu) first, then mu', e1, f'W a n = {W * a * n} per order slice'),
            ('(mu, G) first, then nu', e2, f'W n b = {W * n * b} per order slice'),
            ('(mu, nu) first, then G', e3, f'P W a b = {P * W * a * b}, all orders at once'))
    for name, e, size in rows:
        print(f'  {name}: equal to the explicit sum {relerr(e, ref) < TOL};  intermediate {size}')
    print('  each order contracts the core with one coefficient P times; they differ in the intermediate and in')
    print('  which operand is taken one coefficient at a time')


if __name__ == '__main__':
    part1()
    part2()
