import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

project = "MIKASA-Robo-VLA"
copyright = "2025, Egor Cherepanov, Nikita Kachaev, Alexey K. Kovalev, Aleksandr I. Panov"
author = "Egor Cherepanov"
release = "1.0.0"
version = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

autosummary_generate = True
add_module_names = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_toc_level": 2,
    "navigation_depth": 3,
    "navbar_align": "left",
    "collapse_navigation": False,
    "use_edit_page_button": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/CognitiveAISystems/MIKASA-Robo",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_title = "MIKASA-Robo-VLA Documentation"
html_short_title = "MIKASA-Robo-VLA"

pygments_style = "sphinx"

html_css_files = ["custom.css"]

# Add a copy button to every highlighted code block.
# The regex strips common interactive prompts while preserving plain commands.
copybutton_prompt_text = r">>> |\.\.\. |\$ |\(.*\) \$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
copybutton_remove_prompts = True

html_js_files = ["copy-code-buttons.js"]



# Custom asset injector
# ---------------------
# pydata-sphinx-theme does not consistently inject html_css_files and
# html_js_files into deeply nested pages (e.g. vla_environments/*.html or
# api/*.html), which breaks the copy-code-button overlay and the custom
# stylesheet.  The build-finished hook below walks every generated HTML
# file and ensures that both assets are linked with a path relative to the
# current page.  This works around the theme issue without monkey-patching
# Sphinx internals.

def _relative_static_asset(html_path, outdir, asset_name):
    from pathlib import Path
    rel = Path("_static") / asset_name
    return Path(rel).as_posix() if html_path.parent == outdir else Path(*([".."] * len(html_path.parent.relative_to(outdir).parts)), rel).as_posix()


def _ensure_copy_assets_on_all_pages(app, exception):
    if exception is not None:
        return

    from pathlib import Path

    outdir = Path(app.outdir)
    for html_path in outdir.rglob("*.html"):
        text = html_path.read_text(encoding="utf-8", errors="ignore")
        changed = False

        if "custom.css" not in text:
            css_href = _relative_static_asset(html_path, outdir, "custom.css")
            text = text.replace(
                "</head>",
                f'  <link rel="stylesheet" type="text/css" href="{css_href}" />\n</head>',
                1,
            )
            changed = True

        if "copy-code-buttons.js" not in text:
            js_src = _relative_static_asset(html_path, outdir, "copy-code-buttons.js")
            text = text.replace(
                "</body>",
                f'  <script defer src="{js_src}"></script>\n</body>',
                1,
            )
            changed = True

        if changed:
            html_path.write_text(text, encoding="utf-8")


def setup(app):
    app.connect("build-finished", _ensure_copy_assets_on_all_pages)
