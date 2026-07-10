import os
import sys

# -- Project information -----------------------------------------------------
project = 'T3Toolbox'
copyright = '2026, Nick Alger and Blake Christierson'
author = 'Nick Alger and Blake Christierson'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autoapi.extension',
    'sphinx.ext.githubpages',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_templates/autoapi']

add_module_names = False
typehints_fully_qualified = False
python_use_unqualified_type_names = True
toc_object_entries_show_parents = 'hide'

autodoc_typehints = 'description'

html_favicon = 'favicon.ico'

# -- Autodoc2 ----

autodoc2_packages = [
    "../t3toolbox",
]

# -- AutoAPI configuration ---------------------------------------------------
autoapi_dirs = ['../t3toolbox']  # Path to your source code
autoapi_template_dir = '_templates/autoapi'
autoapi_type = 'python'

#autoapi_add_toctree_entry = True

autoapi_own_page_level = 'method'

autoapi_ignore = ["*OLD*.py", "*backend*.py", "*eighted*.py", "*anifold*.py", "*niform*.py", "*frame_variation*"]

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

####

def autoapi_prepare_jinja_env(jinja_env):
    def get_class_and_method(obj_id):
        parts = obj_id.split('.')
        if len(parts) >= 2:
            return f"{parts[-2]}.{parts[-1]}"
        return obj_id

    jinja_env.filters["class_method_format"] = get_class_and_method
    
# Ensure the hook is configured
# autoapi_prepare_jinja_env = autoapi_prepare_jinja_env
