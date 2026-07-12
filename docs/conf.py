import ast
import functools
import pathlib
import textwrap
import tomllib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# -- Project information -----------------------------------------------------
project = 'T3Toolbox'
copyright = '2026, Nick Alger and Blake Christierson'
author = 'Nick Alger and Blake Christierson'

# Single-sourced from pyproject.toml. tomllib needs py>=3.11: fine, because conf.py runs only
# where the DOCS are built (maintainer machine + CI) -- it places no constraint on the library.
with open(_REPO_ROOT / 'pyproject.toml', 'rb') as _f:
    release = tomllib.load(_f)['project']['version']
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autoapi.extension',
    'myst_parser',
    'sphinx.ext.githubpages',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_templates/autoapi']

add_module_names = False
typehints_fully_qualified = False
python_use_unqualified_type_names = True
toc_object_entries_show_parents = 'hide'

autodoc_typehints = 'description'

html_favicon = 'favicon.ico'

# -- MyST (markdown design docs under docs/) ----------------------------------
myst_heading_anchors = 3  # so cross-doc links to ## / ### headings resolve

# -- AutoAPI configuration ---------------------------------------------------
autoapi_dirs = ['../t3toolbox']
autoapi_template_dir = '_templates/autoapi'
autoapi_type = 'python'

autoapi_own_page_level = 'method'
autoapi_add_toctree_entry = False  # placed explicitly by api_reference.rst

# The rendered reference covers the whole validated surface (frontend AND backend; backend users
# are first-class). Excluded: only the unvalidated surface -- OLD_* strays and the parked
# weighted layer (weighted_tucker_tensor_train.py, backend/wt3_operations.py).
autoapi_ignore = ['*OLD*', '*weighted_tucker_tensor_train*', '*wt3_*']

# No 'imported-members': each object is documented once, in its defining module (kills the
# duplicate-cross-reference-target noise from the curated __init__ re-exports and the backend's
# internal star-imports; the curated package surface is presented by the hand-written API landing
# page instead). No 'private-members': underscore-prefixed helpers stay out of the reference.
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

# -- Verbatim source signatures ----------------------------------------------
# The trailing `#` shape comments in signatures ARE the type contract in this codebase
# (docs/contributor/signature_style.md) -- but autoapi regenerates signatures from the parsed AST, where
# comments no longer exist. So the function/method templates (docs/_templates/autoapi/) call the
# `verbatim_signature` filter below to pull each object's signature -- comments, alignment and
# all -- straight out of the source file, and render it as a literal code block on the page.


@functools.lru_cache(maxsize=None)
def _parsed_source(py_path):
    text = py_path.read_text()
    return text.splitlines(), ast.parse(text)


def _find_def(tree, qual_parts):
    """Walk nested ClassDef/FunctionDef scopes by name; return the final node or None."""
    node, scope = None, tree
    for part in qual_parts:
        node = None
        for child in getattr(scope, 'body', []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and child.name == part:
                node = child
                break
        if node is None:
            return None
        scope = node
    return node


def verbatim_signature(obj):
    """The object's signature exactly as written in the source (shape comments included).

    Returns '' whenever anything is unavailable -- the template then just omits the block.
    """
    try:
        full_name = obj.obj.get('full_name') or obj.id
        qual_name = obj.obj.get('qual_name') or full_name.rsplit('.', 1)[-1]
        if not full_name.endswith('.' + qual_name):
            return ''
        mod_rel = pathlib.Path(*full_name[: -(len(qual_name) + 1)].split('.'))
        for cand in (_REPO_ROOT / mod_rel.with_suffix('.py'), _REPO_ROOT / mod_rel / '__init__.py'):
            if cand.is_file():
                lines, tree = _parsed_source(cand)
                node = _find_def(tree, qual_name.split('.'))
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return ''
                end = node.body[0].lineno - 1 if node.body else node.end_lineno
                end = max(end, node.lineno)  # one-line defs
                return textwrap.dedent('\n'.join(lines[node.lineno - 1: end])).rstrip()
        return ''
    except Exception:
        return ''


# -- Duplicate-binding dedup -------------------------------------------------
# backend/common.py deliberately rebinds several names under ``if jax_available:`` (numpy
# fallback first, jax-aware rebinding second). astroid records every top-level binding, so
# autoapi would document such names twice on the module page (duplicate object descriptions +
# duplicated toctree entries). Keep the first binding, skip the rest -- the twins are
# undocumented stubs with identical signatures, so nothing is lost. A global seen-set is safe
# here: ``display`` is cached per object, so the skip event fires exactly once per object.
_seen_member_ids = set()


def _skip_duplicate_bindings(app, what, name, obj, skip, options):
    if skip:
        return skip
    if name in _seen_member_ids:
        return True
    _seen_member_ids.add(name)
    return None


def setup(app):
    app.connect('autoapi-skip-member', _skip_duplicate_bindings)


def autoapi_prepare_jinja_env(jinja_env):
    def get_class_and_method(obj_id):
        parts = obj_id.split('.')
        if len(parts) >= 2:
            return f"{parts[-2]}.{parts[-1]}"
        return obj_id

    jinja_env.filters["class_method_format"] = get_class_and_method
    jinja_env.filters["verbatim_signature"] = verbatim_signature
