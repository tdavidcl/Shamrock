# Minimal Sphinx project to preview the output of the example renderer prototype.
# Render the examples into ./examples_gen first, see ../README.md.

project = "Example renderer prototype"
html_title = "Renderer prototype"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.video",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

exclude_patterns = ["_build", "examples_gen/_logs"]

try:
    import pydata_sphinx_theme

    html_theme = "pydata_sphinx_theme"
except ImportError:
    html_theme = "alabaster"

html_static_path = ["_static"]
html_css_files = ["example_renderer.css"]
