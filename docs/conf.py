# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TuckerTensorTrainTools'
copyright = '2026, Nick Alger and Blake Christierson'
author = 'Nick Alger and Blake Christierson'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',
    "sphinx.ext.autodoc",
    'sphinx.ext.autosummary',
    "sphinx.ext.napoleon",
    'sphinx.ext.githubpages',
]
autoapi_dirs = ['../t3tools']
numpydoc_show_class_members = False
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


autoapi_modules = {'t3tools': None}
autoapi_own_page_level = 'function'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_favicon = 'favicon.ico'

html_theme = 'agogo'
html_static_path = ['_static']
