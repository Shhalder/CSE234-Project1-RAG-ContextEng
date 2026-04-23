# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'RapidFire AI'
author = 'RapidFire AI'
master_doc = 'overview'

release = '0.15'
version = '0.15.2'

# -- General configuration

templates_path = ['_templates']

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx_design',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['images']

# -- Options for EPUB output
epub_show_urls = 'footnote'

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

html_show_copyright = False
