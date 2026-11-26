#!/usr/bin/env python3
"""The def/class headers K1 changes, verbatim from a tree (the def line through the closing colon, comments kept).

    python3 dev/graded_messages/implementation/k1/signatures.py [REPO]      # markdown on stdout

Run on the tree before K1 for the design note's "before" column, and on K1's tree for the handover's "after".
"""
import ast
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), *['..'] * 4))

TARGETS = {
    't3toolbox/backend/sampling_variables.py': ['sampling_variables', 'computed_names', '_check_data_space'],
    't3toolbox/backend/sampling_steps.py': ['_modes', '_with_K', '_weighted_sum', 'Jp_norm_sq_step', 'w_tilde_step',
                                             'r_step', 'f_step', 'r_block_sumsq_step'],
    't3toolbox/backend/jets.py': ['data_metric', 'blocks_to_weight', 'jet_to_dual_jet', 'dual_jet_to_jet',
                                  'derivative_weight_to_taylor_weight', 'prepare_derivative_data', 'check_data',
                                  'check_weight'],
    't3toolbox/backend/fitting.py': ['LeastSquaresBatch', 'LeastSquaresOracle'],
    't3toolbox/backend/uniform_fitting.py': ['pack_data', 'pack_batch'],
    't3toolbox/backend/ut3_operations.py': ['_pack_data_space'],
    't3toolbox/backend/optimizers.py': ['flat_draw', '_require_draw'],
    't3toolbox/backend/optimizer_display.py': ['make_newton_display'],
    't3toolbox/fitting.py': ['_is_packed', 'prepare_batch'],
    't3toolbox/optimizers.py': ['_frontend_draw', '_old_fit', '_old_draw'],
}


def header(lines, node):
    '''From the def line to the line holding the colon that ends the signature (the body's first line minus one).'''
    start = node.lineno - 1 - len(node.decorator_list) if isinstance(node, ast.ClassDef) else node.lineno - 1
    if isinstance(node, ast.ClassDef):
        return [lines[node.lineno - 1]]
    end = node.body[0].lineno - 1
    if isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant):
        end = node.body[0].lineno - 1
    return lines[start:end]


def main():
    for rel, names in TARGETS.items():
        path = os.path.join(REPO, rel)
        text = open(path).read()
        lines = text.splitlines()
        tree = ast.parse(text)
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names and node.name not in found:
                found[node.name] = node
        for name in names:
            node = found.get(name)
            print('### `%s` (`%s:%s`)\n' % (name, rel, node.lineno if node else '-'))
            if node is None:
                print('(absent)\n')
                continue
            if isinstance(node, ast.ClassDef):
                body = [n for n in node.body if isinstance(n, (ast.AnnAssign, ast.FunctionDef))]
                out = [lines[node.lineno - 1]]
                for n in body:
                    if isinstance(n, ast.AnnAssign):
                        out.append(lines[n.lineno - 1])
                    elif not n.name.startswith('__') or n.name in ('__init__', '__post_init__'):
                        out += header(lines, n)
                print('```python\n' + '\n'.join(out) + '\n```\n')
            else:
                print('```python\n' + '\n'.join(header(lines, node)) + '\n```\n')


if __name__ == '__main__':
    main()
