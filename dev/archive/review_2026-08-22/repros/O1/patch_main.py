p='o1_uniform.py'; s=open(p).read()
old = """    for sname in names:
        for fp in (False, True):
            sweep(sname, STRUCTS[sname], SHARING.get(sname), fp)
        print('done', sname, len(RESULTS), flush=True)
    dump("""
new = """    for sname in names:
        for fp in (False, True):
            try:
                sweep(sname, STRUCTS[sname], SHARING.get(sname), fp)
            except Exception as e:
                import traceback; record('SWEEP_CRASH', sname, 'uniform+pad' if fp else 'uniform', (), (), (), SHARING.get(sname), 'EXC', float('nan'), '%s: %s @ %s' % (type(e).__name__, str(e)[:100], traceback.format_exc().splitlines()[-3].strip()[:120]))
        print('done', sname, len(RESULTS), flush=True)
        dump(os.path.join(os.path.dirname(__file__), 'results_uniform_%s.md' % ('_'.join(names) if names else 'vary')))
    dump("""
if old in s:
    s = s.replace(old, new); open(p,'w').write(s); print('patched')
else:
    print('already patched' if 'SWEEP_CRASH' in s else 'PATTERN NOT FOUND')
