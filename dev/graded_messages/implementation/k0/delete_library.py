#!/usr/bin/env python3
"""K0's library deletion, the mechanical part (K0_DESIGN_NOTE.md §1.2, §1.3): run once, on the tree before K0.

    python3 dev/graded_messages/implementation/k0/delete_library.py

- removes `t3toolbox/backend/sampling_derivatives.py`;
- in the seven modules, cuts every deleted top-level statement (k0_inventory.py's closure, with its decorators), the
  `__all__` entries of deleted names, the import lines the note lists (unused after K0), and the module-level comment
  blocks the note deletes whole (by their line ranges at b5125f76, checked against their first line's text);
- collapses runs of more than two blank lines to two.

The rewrites (module docstrings, the comments cut to what stays, `__all__`'s section comments) are done by hand
afterwards, from `k0/prose_edits.md`. Every module must still parse; the line numbers are those of b5125f76, so the
script refuses to run on a file that differs from it.
"""
import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import k0_inventory as K   # noqa: E402

IMPORTS = {   # local names whose import goes (the note §1.3: unused after K0, both lists)
    'probing': {'np', 't3_conversions', 'fv_conversions', 'tt_operations', 'ragged_ops', 'uniform_ops'},
    'apply': {'t3_conversions', 'fv_conversions', 'math', 'compute_xi', 'compute_mu', 'compute_dxi',
              'compute_sigma_hat', '_sigma_step'},
    'entries': {'np', 'fv_conversions', 'math', 'probing', 'compute_mu', '_apply_from_xis', '_apply_transpose_adjoint',
                '_apply_corewise_hessian_adjoint'},
    'ut3_sampling': {'np', 'apply', 'entries', 'probing', 'sampling_derivatives', 'ut3_masking'},
    'utv_sampling': {'np', 'probing', 'apply', 'entries', 'sampling_derivatives'},
    'fitting': {'abc', 'ft', 'probing', 'pd', 'bapply', 'bentries'},
    'uniform_fitting': {'readonly_mask_copies', 'ufv_masking', 'utv_sampling', 'ut3_sampling', 'geometry_module',
                        'ufv_conversions', 'utv_ops', 'sharing_module'},
}
BLOCKS = {   # module-level comment blocks deleted whole: (first, last, the first line's text)
    'probing': [(586, 595, '#####################################################'),
                (1239, 1266, '#')],
    'ut3_sampling': [(63, 67, '# All re-mask first'), (256, 261, '# ---'), (310, 315, '# ---'), (373, 378, '# ---'),
                     (634, 638, '# ---')],
    'utv_sampling': [(135, 148, '# ---'), (252, 262, '# ---'), (389, 394, '# ---'), (528, 534, '# ---')],
    'fitting': [(107, 113, '# ---'), (177, 179, '# `take` per kind'), (297, 314, '# ---'), (488, 490, '# ---'),
                (527, 529, '# ---'), (737, 738, '# Constructor aliases')],
    'uniform_fitting': [(61, 78, '# ---'), (457, 462, '# Packed-aware minibatch')],
}


def main():
    head = subprocess.run(['git', '-C', K.REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout
    for short in IMPORTS:
        path = 't3toolbox/backend/%s.py' % short
        diff = subprocess.run(['git', '-C', K.REPO, 'diff', '--quiet', 'b5125f76', '--', path])
        if diff.returncode != 0:
            sys.exit('%s differs from b5125f76: the line ranges would be wrong (HEAD %s)' % (path, head.strip()))
    modules, _ = K.load_library()
    deleted, _, _ = K.closure(modules)
    os.remove(os.path.join(K.REPO, 't3toolbox/backend/sampling_derivatives.py'))
    for short, names in IMPORTS.items():
        m = 't3toolbox.backend.' + short
        info = modules[m]
        lines = K.read(info['path']).split('\n')
        drop = set()
        for u in info['units']:
            if u.deleted:
                drop |= set(range(u.start, u.end + 1))
            elif u.is_import:
                local = {a.asname or a.name.split('.')[0] for a in u.node.names}
                if local & names:
                    if not local <= names:
                        sys.exit('%s:%d: an import line with names both kept and removed: %s' % (info['path'], u.start,
                                                                                               sorted(local)))
                    drop |= set(range(u.start, u.end + 1))
            elif '__all__' in u.defines:
                for k in range(u.start, u.end + 1):
                    if lines[k - 1].strip().rstrip(',').strip("'\"") in deleted[m]:
                        drop.add(k)
        for first, last, text in BLOCKS.get(short, []):
            if not lines[first - 1].startswith(text):
                sys.exit('%s:%d: expected %r, found %r' % (info['path'], first, text, lines[first - 1]))
            drop |= set(range(first, last + 1))
        kept = [line for k, line in enumerate(lines, 1) if k not in drop]
        out, blank = [], 0
        for line in kept:
            blank = blank + 1 if line.strip() == '' else 0
            if blank <= 2:
                out.append(line)
        text = '\n'.join(out)
        ast.parse(text)
        with open(os.path.join(K.REPO, info['path']), 'w') as f:
            f.write(text)
        print('%s: %d lines removed, %d left' % (info['path'], len(lines) - len(out), len(out)))


if __name__ == '__main__':
    main()
