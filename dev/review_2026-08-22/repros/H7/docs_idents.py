import re, subprocess, os, glob
root='/home/nick/repos/T3Toolbox'
files = sorted(glob.glob(root+'/docs/*.md')+glob.glob(root+'/docs/contributor/*.md')+glob.glob(root+'/docs/*.rst')+[root+'/README.md'])
pat=re.compile(r'``?([A-Za-z_][\w\.]*?)(?:\([^`]*\))?``?')
skip=set('None True False data int float np jnp xnp jit vmap grad self docs dev backend main numpy jax tests stack probe apply entries frame inner norm shape order mode kind name sharing project retract transport zeros ones randn load save sqrt share resize reverse absorb kronecker concatenate weight callback verbose aux_data unsafe safe assert unittest subTest pytest einsum optimize path contract corewise uniform ragged weighted identity groups precompute sweep sample residual geometry history misfit regularization diagnostics regularizer validate tucker tt tol rtol atol scan lax ndarray'.split())
cache={}
def found(tok):
    if tok in cache: return cache[tok]
    r=subprocess.run(['grep','-rqw',tok,root+'/t3toolbox',root+'/examples',root+'/tests'],capture_output=True)
    cache[tok]= (r.returncode==0); return cache[tok]
out=[]
for f in files:
    rel=f[len(root)+1:]
    for i,l in enumerate(open(f).read().splitlines(),1):
        if l.lstrip().startswith('>>>') or l.lstrip().startswith('...'): continue
        for m in pat.finditer(l):
            tok=m.group(1)
            if '/' in tok or tok.endswith(('.md','.py','.rst')): continue
            parts=tok.split('.')
            last=parts[-1]
            if len(last)<5 or last in skip or not re.match(r'^[A-Za-z_]\w*$', last): continue
            if last.islower() and '_' not in last and len(last)<8: continue  # plain words
            if not found(last):
                out.append(f'{rel}:{i}: `{tok}`')
for o in out: print(o)
