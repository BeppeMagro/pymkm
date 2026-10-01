# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'pyMKM'
copyright = '2025, Giuseppe Magro et al'
author = 'Giuseppe Magro et al'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

#html_theme = 'sphinx_rtd_theme'
html_theme = 'furo'
html_theme_options = {
    "navigation_with_keys": True,
}
