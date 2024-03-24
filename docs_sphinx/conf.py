# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('./../src'))

project = 'Signal processing tools and examples'
copyright = '2023, Nicolas Eby'
author = 'Nicolas Eby'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.imgmath',
    # 'sphinx.ext.pngmath',
    # 'sphinx_toolbox.latex',
    ]

# the documentation of the two following binaries
# those paths works on Windows
# pngmath_latex=r"C:\Users\NicolasEBY\AppData\Local\Programs\MiKTeX\miktex\bin\x64\latex.exe"
# pngmath_dvipng=r"C:\Users\NicolasEBY\AppData\Local\Programs\MiKTeX\miktex\bin\x64\dvipng.exe"
# C:\Users\NicolasEBY\AppData\Local\Programs\MiKTeX

# mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML'


# Utiliser imgmath pour le rendu des équations si MathJax n'est pas disponible
# extensions = ['sphinx.ext.imgmath']

# Configurer imgmath_latex pour utiliser le moteur LaTeX
# imgmath_latex = 'latex'


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_css_files = [
#     '_static/css/style.css',
#     ]

html_static_path = ['_static']

add_module_names = False
