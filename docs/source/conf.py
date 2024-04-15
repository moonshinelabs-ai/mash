# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "Mash"
copyright = "2024, Moonshine"
author = "Moonshine"

release = "0.1"
version = "0.1.5"

# -- General configuration

pygments_style = "manni"
pygments_dark_style = "monokai"

autosummary_generate = True

source_suffix = [".rst", ".md"]

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output
html_theme = "furo"

# -- Options for EPUB output
epub_show_urls = "footnote"
