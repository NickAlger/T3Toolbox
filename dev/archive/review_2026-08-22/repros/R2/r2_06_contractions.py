"""Probing _solve_group_lengths / len_* error paths / attempts at silently-wrong resolution."""
import numpy as np
from t3toolbox.backend.contractions import contract, _expanded_subscripts
rng = np.random.default_rng(0)
def show(label, f):
    try:
        r = f(); print('%-75s -> %s' % (label, r))
    except Exception as e:
        print('%-75s -> %s: %s' % (label, type(e).__name__, str(e)[:140]))
show("'Wab,c->Wabc' 1-d first operand (doc: x_W=-1 inconsistent)", lambda: contract('Wab,c->Wabc', np.ones(1), np.ones(2)).shape)
show("'WKCi,Cio->WKCo' len_W=7 (too big for run)", lambda: contract('WKCi,Cio->WKCo', np.ones((5, 3, 2, 4)), np.ones((2, 4, 6)), len_W=7).shape)
show("'WKCi,Cio->WKCo' len_W=-1", lambda: contract('WKCi,Cio->WKCo', np.ones((5, 3, 2, 4)), np.ones((2, 4, 6)), len_W=-1).shape)
show("'WKCi,Cio->WKCo' len_W=1.5", lambda: contract('WKCi,Cio->WKCo', np.ones((5, 3, 2, 4)), np.ones((2, 4, 6)), len_W=1.5).shape)
show("len_w lowercase kwarg", lambda: contract('WCo,WCa->Cao', np.ones((5, 2, 3)), np.ones((5, 2, 6)), len_w=1).shape)
show("len_X for a group not in the subscripts", lambda: contract('WCo,WCa->Cao', np.ones((5, 2, 3)), np.ones((5, 2, 6)), len_X=1).shape)
show("'AB,BA->AB' (same total, different order)", lambda: contract('AB,BA->AB', np.ones((2, 3)), np.ones((3, 2))).shape)
show("'AB,BA->AB' len_A=1", lambda: contract('AB,BA->AB', np.ones((2, 3)), np.ones((3, 2)), len_A=1).shape)
show("'WC,C->W' W=(4,) C=(2,)", lambda: contract('WC,C->W', np.ones((4, 2)), np.ones(2)).shape)
show("redundant wrong len_C=2 where shapes pin C=1", lambda: contract('Cio,WCo->WCi', np.ones((2, 4, 3)), np.ones((5, 2, 3)), len_C=2).shape)
show("size-1 group axis broadcasts: C=(1,) vs (3,)", lambda: contract('Ca,Ca->Ca', np.ones((1, 4)), np.ones((3, 4))).shape)
show("'WCo, WCa -> Cao' whitespace, len_W=1", lambda: contract('WCo, WCa -> Cao', np.ones((5, 2, 3)), np.ones((5, 2, 6)), len_W=1).shape)
show("expansion 'WKCa,Caib,WCi->WKCb' ndims (4,4,3)", lambda: _expanded_subscripts('WKCa,Caib,WCi->WKCb', (4, 4, 3), ()))
show("expansion 'WKCa,Caib,WCi->WKCb' ndims (3,3,2) (W=1,K=0,C=0)", lambda: _expanded_subscripts('WKCa,Caib,WCi->WKCb', (3, 3, 2), ()))
show("3-operand numpy vs einsum value check", lambda: bool(np.allclose(contract('WCa,Caib,WCi->WCb', a:=rng.standard_normal((5, 2, 3)), G:=rng.standard_normal((2, 3, 4, 6)), w:=rng.standard_normal((5, 2, 4))), np.einsum('wca,caib,wci->wcb', a, G, w))))
show("group in output absent from input", lambda: contract('Ca->CW', np.ones((2, 3))).shape)
show("empty output scalar 'a,a->'", lambda: contract('a,a->', np.ones(3), np.ones(3)))
show("'K,K->K' K=() (scalars)", lambda: contract('K,K->K', np.ones(()), np.ones(())))
# A 'silently wrong' attempt: a run whose members are split differently by the user than intended
ops = (rng.standard_normal((5, 3, 2, 4)), rng.standard_normal((2, 4, 6)))
vals = [contract('WKCi,Cio->WKCo', *ops, len_W=n) for n in (0, 1, 2)]
show("split invariance (bitwise) for co-travel run", lambda: all(np.array_equal(vals[0], v) for v in vals))
# Output permutes a run member vs the non-run member: 'WKi,i->KW'
show("'WKi,i->KW' (W,K swapped in output) ndims (3,1)", lambda: contract('WKi,i->KW', np.ones((2, 3, 4)), np.ones(4)).shape)
show("'WKi,i->KW' len_W=1", lambda: contract('WKi,i->KW', np.ones((2, 3, 4)), np.ones(4), len_W=1).shape)
# too many letters
show("letter exhaustion: 'W->W' with ndim 60", lambda: contract('W->W', np.ones((1,) * 60)).shape)
# jax + mixed operands
import jax.numpy as jnp
show("mixed numpy/jax operands", lambda: type(contract('Cio,WCo->WCi', np.ones((2, 4, 3)), jnp.ones((5, 2, 3)))).__name__)
