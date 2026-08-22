import re, subprocess, os
root='/home/nick/repos/T3Toolbox'
txt=open(root+'/CLAUDE.md').read().splitlines()
# backticked tokens that look like python identifiers (optionally dotted), skipping paths and kwargs
pat=re.compile(r'`([A-Za-z_][\w\.]*?)(?:\(\))?`')
seen={}
for i,l in enumerate(txt,1):
    for m in pat.finditer(l):
        tok=m.group(1)
        if '/' in tok or tok.endswith('.md') or tok.endswith('.py'): continue
        last=tok.split('.')[-1]
        if len(last)<4 or last in ('None','True','False','data','int','float','np','jnp','xnp','jit','vmap','grad','self','docs','dev','backend','main','numpy','jax','tests','stack','probe','apply','entries','frame','inner','norm','shape','order','mode','kind','name','sharing','project','retract','transport','zeros','ones','randn','load','save','sqrt','share','resize','reverse','absorb','kronecker','concatenate','weight','callback','verbose','aux_data','unsafe','safe','assert','unittest','subTest','pytest','einsum','optimize','path','contract','corewise','uniform','ragged','weighted','identity','groups','precompute','sweep','sample','residual','geometry','history','misfit','regularization','diagnostics','regularizer','validate','tucker','tt'): continue
        seen.setdefault(last, []).append(i)
missing=[]
for tok,lines in seen.items():
    r=subprocess.run(['grep','-rlw',tok,root+'/t3toolbox'],capture_output=True,text=True)
    if not r.stdout.strip():
        missing.append((tok,lines))
for tok,lines in sorted(missing, key=lambda x:x[1][0]):
    print(f'CLAUDE.md:{lines}: `{tok}` not found (whole word) in t3toolbox/')
