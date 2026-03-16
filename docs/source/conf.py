# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os, sys

sys.path.insert(0, os.path.abspath("../.."))  # points to project root from docs/source

project = "lidalign"
copyright = "2026, Paul Meyer, ForWind"
author = "Paul Meyer"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]


# -- Options
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # if you use Google/Numpy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_nb",
    "nbsphinx",
]
autosummary_generate = True
html_theme = "sphinx_rtd_theme"
nb_execution_mode = "off"  # Don't execute notebooks during build

# Tell Sphinx which file types are source files
# source_suffix = {
#     ".rst": "restructuredtext",  # you can remove later if you want *only* md
#     ".md": "markdown",
#     ".ipynb": "myst-nb",
# }
html_js_files = [
    "https://cdn.plot.ly/plotly-latest.min.js",
]

html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": True,
}

import plotly.io as pio

# pio.renderers.default = "sphinx_gallery"
nbsphinx_allow_errors = True
# Plotly HTML nicht strippen
nb_output_stderr = "remove"


# def setup(app):
#     app.add_stylesheet("plotly-style.css")  # also can be a full URL
