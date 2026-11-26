"""M5's evidence that no library code changed: every ``.py`` under ``t3toolbox/``, at a git revision and in the working
tree, parsed with its docstrings removed (comments are not in the AST), and ``ast.dump`` compared.

Run from the repository root: ``python dev/graded_messages/implementation/m5/ast_equal.py [REV]`` (default ``HEAD``).
Exit status 0 when the two trees hold the same files and every module's code is identical; it also prints how many
files differ in text, and that every non-``.py`` file under ``t3toolbox/`` is unchanged.
"""
import ast
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]


def _strip_docstrings(tree):
    """Remove the docstring (a leading string expression) of the module and of every class and function; an emptied
    body becomes ``pass``, on both sides alike."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _code(source):
    return ast.dump(_strip_docstrings(ast.parse(source)), include_attributes=False)


def main(rev):
    listed = subprocess.run(['git', 'ls-tree', '-r', '--name-only', rev, 't3toolbox/'], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.split()
    old_files = set(listed)
    new_files = set(subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', 't3toolbox/'],
                                   cwd=REPO, check=True, capture_output=True, text=True).stdout.split())   # git's view:
    # tracked and untracked files, the ignored ones (caches, local settings) left out
    ok = True
    if old_files != new_files:
        print('FILES DIFFER: only at %s %s; only in the tree %s' % (rev, sorted(old_files - new_files),
                                                                   sorted(new_files - old_files)))
        ok = False
    text_changed, code_changed, other_changed = [], [], []
    for rel in sorted(old_files & new_files):
        old = subprocess.run(['git', 'show', '%s:%s' % (rev, rel)], cwd=REPO, check=True,
                             capture_output=True).stdout
        new = (REPO / rel).read_bytes()
        if old == new:
            continue
        if not rel.endswith('.py'):
            other_changed.append(rel)
            continue
        text_changed.append(rel)
        if _code(old.decode('utf-8')) != _code(new.decode('utf-8')):
            code_changed.append(rel)
    print('%d .py files compared; %d differ in text; %d differ in code (docstrings and comments removed)'
          % (sum(r.endswith('.py') for r in old_files & new_files), len(text_changed), len(code_changed)))
    for rel in text_changed:
        print('  text only' if rel not in code_changed else '  CODE', rel)
    if other_changed:
        print('NON-.py FILES CHANGED:', other_changed)
    ok = ok and not code_changed and not other_changed
    print('IDENTICAL CODE' if ok else 'CODE DIFFERS')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'HEAD'))
