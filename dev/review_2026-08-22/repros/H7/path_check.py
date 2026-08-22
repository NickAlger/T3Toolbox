import re, os, glob
root='/home/nick/repos/T3Toolbox'
files = ['CLAUDE.md','README.md','CHANGELOG.md','CITATION.cff','docs/index.rst','docs/user_guide.rst','docs/getting_started.rst','docs/api_reference.rst','docs/release_notes.md'] + [p[len(root)+1:] for p in glob.glob(root+'/docs/*.md')+glob.glob(root+'/docs/contributor/*.md')] + [p[len(root)+1:] for p in glob.glob(root+'/t3toolbox/*.py')+glob.glob(root+'/t3toolbox/backend/*.py')]
pat = re.compile(r'`?((?:docs|dev|examples|tests|t3toolbox|backend)/[\w./\-]+\.(?:md|py|rst|tex|pdf|txt|yaml|toml))`?')
seen=set()
for f in files:
    txt=open(os.path.join(root,f)).read().splitlines()
    for i,l in enumerate(txt,1):
        for m in pat.finditer(l):
            p=m.group(1)
            cands=[p]
            if p.startswith('backend/'): cands=['t3toolbox/'+p]
            if f.startswith('docs/') and p.startswith('docs/'): pass
            # relative links inside docs pages (e.g. contributor/x.md from docs/)
            ok = any(os.path.exists(os.path.join(root,c)) for c in cands)
            if not ok and f.startswith('docs/'):
                ok = os.path.exists(os.path.join(root,'docs',p))
            if not ok:
                print(f'{f}:{i}: MISSING {p}')
