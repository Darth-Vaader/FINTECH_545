r"""
export_pdf.py -- turn the executed notebook into Assignment2_Writeup.pdf without LaTeX.

    python export_pdf.py            # writes Assignment2_Writeup.pdf next to this file

Pipeline: nbconvert (notebook -> HTML) -> headless Google Chrome (HTML -> PDF).
Requires nbconvert and a Chrome / Chromium binary; no TeX installation is needed.

Two things are done on the way that a plain `jupyter nbconvert --to html` does not:

1. Literal dollar signs. The write-up is full of amounts like \$19,594. In the
   notebook these are written `\$`, which JupyterLab renders correctly. nbconvert's
   markdown renderer (mistune) strips that backslash, and MathJax then pairs any two
   bare `$` in a paragraph into inline math, turning "\$887k against \$542k" into
   italic garbage. The fix is to double the escape (`\\$`) in a TEMPORARY copy of
   the notebook so that `\$` reaches the browser, where MathJax's processEscapes
   renders it as a plain dollar sign. The notebook on disk is not modified. Escapes
   inside `$...$` math spans are left alone.

2. Print CSS. The problem modules format their output to 88 columns; at
   nbconvert's default 13px monospace that wraps on a Letter page. The injected
   style uses 11px and keeps figures and tables from splitting across pages.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "Assignment2.ipynb")
PDF = os.path.join(HERE, "Assignment2_Writeup.pdf")

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"),
]

PRINT_CSS = """
<style>
  :root { --jp-code-font-size: 11px; }
  .jp-OutputArea-output pre, .jp-InputArea-editor pre, .highlight pre,
  .jp-RenderedText pre { font-size: 11px !important; line-height: 1.32 !important; }
  .jp-OutputArea-output img { page-break-inside: avoid; max-width: 100%; }
  table { page-break-inside: avoid; }
  @page { size: letter; margin: 0.6in 0.55in; }
</style>
"""

# A `\$` that is NOT inside a $...$ math span. Math spans are matched first (an
# unescaped `$`, content without unescaped `$`, an unescaped `$`) and passed
# through untouched; any `\$` outside them is doubled.
_MATH_OR_ESCAPED = re.compile(r"(?<!\\)\$(?:[^$\\]|\\.)+?(?<!\\)\$|\\\$", re.DOTALL)


def double_dollar_escapes(src: str) -> str:
    return _MATH_OR_ESCAPED.sub(lambda m: r"\\$" if m.group(0) == r"\$" else m.group(0), src)


def main() -> None:
    import nbformat

    chrome = next((c for c in CHROME_CANDIDATES if c and os.path.exists(c)), None)
    if chrome is None:
        sys.exit("No Chrome/Chromium found. Open Assignment2.ipynb in Jupyter and use "
                 "File > Save and Export Notebook As > PDF, or print the HTML export.")

    nb = nbformat.read(NOTEBOOK, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = double_dollar_escapes(cell.source)

    with tempfile.TemporaryDirectory() as tmp:
        print_nb = os.path.join(tmp, "Assignment2_print.ipynb")
        nbformat.write(nb, print_nb)
        subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html",
                        "--output-dir", tmp, "--output", "Assignment2.html", print_nb],
                       check=True)
        html_path = os.path.join(tmp, "Assignment2.html")
        with open(html_path, encoding="utf-8") as f:
            html = f.read()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html.replace("</head>", PRINT_CSS + "</head>", 1))
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=30000",   # let MathJax finish typesetting
                        "--run-all-compositor-stages-before-draw",
                        f"--print-to-pdf={PDF}", f"file://{html_path}"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {PDF} ({os.path.getsize(PDF):,} bytes)")


if __name__ == "__main__":
    main()
