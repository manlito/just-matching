# Sphinx configuration of the documentation site. Build: see docs/README.md.
#
# Next to the pages, after the build, this file puts the live demo and what it loads:
#   lib/       jmt.js and its WebAssembly module, from a browser build (JMT_WASM_DIR, default build/wasm/web);
#              lib/jkp/ just-keypoints' module and RaCo weights, which that build copies in (web/jkp)
#   demo/      web/demo.html with the site's top bar, and web/demo.js
#   samples/   the demo's sample images (web/samples)
# Anything missing is a warning, so `sphinx-build -W` fails. FlexGlue's weights are not on the site: their
# backbone is under the DINOv3 License, so each visitor fetches flexglue.pth from FlexGlue's authors and loads it
# into the demo.
import os
import shutil

from sphinx.util import logging

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

project = "just-matching"
author = "the just-matching authors"
copyright = "the just-matching authors"
release = ""
version = ""

extensions = ["myst_parser"]
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", ".venv", "README.md", "reproducing.md", "_theme", "_extra"]

html_theme = "jmt"
html_theme_path = ["_theme"]
html_title = project
html_extra_path = ["_extra"]
html_copy_source = False
html_show_sourcelink = False

MODULE_FILES = ["jmt.js", "jmt_wasm.js", "jmt_wasm.wasm"]
JKP_FILES = ["jkp.js", "jkp_wasm.js", "jkp_wasm.wasm", "raco.jkpw"]

SITE_HEADER = """<a class="jk-skip" href="#content">Skip to content</a>
<header class="jk-top">
  <div class="jk-top-in">
    <a class="jk-brand" href="../index.html">
      <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="1.5" y="4" width="7.5" height="14" rx="1.5"></rect><rect x="13" y="4" width="7.5" height="14" rx="1.5"></rect><path d="M5.2 8.5L16.8 13.5M5.2 13.5L16.8 8.5"></path></svg>
      <span>just-matching</span>
    </a>
    <span class="jk-grow"></span>
    <a class="jk-toplink" href="../index.html">Documentation</a>
    <a class="jk-toplink" href="index.html" aria-current="page"><strong>Live demo</strong></a>
  </div>
</header>"""

SITE_HEAD = """<link rel="stylesheet" href="site-bar.css">
<script type="application/json" id="demo-config">
{"lib": "../lib/", "jkp": "../lib/jkp/", "samples": "../samples/"}
</script>"""


def site_demo(page):
    """web/demo.html with the site's top bar and the places of what it loads."""
    for marker in ("<!-- site:head -->", "<!-- site:header -->", "<!-- site:licence -->", "<!-- /site:licence -->"):
        if marker not in page:
            raise ValueError(f"web/demo.html has no {marker}")
    page = page.replace("<title>just-matching demo</title>", "<title>Live demo · just-matching</title>")
    page = page.replace('<link rel="modulepreload" href="jmt.js">', '<link rel="modulepreload" href="../lib/jmt.js">')
    page = page.replace('<link rel="modulepreload" href="jmt_wasm.js">',
                        '<link rel="modulepreload" href="../lib/jmt_wasm.js">')
    page = page.replace("<!-- site:head -->", SITE_HEAD).replace("<!-- site:header -->", SITE_HEADER)
    page = page.replace("<!-- site:licence -->", '<a href="../licences.html#dinov3-license">')
    return page.replace("<!-- /site:licence -->", "</a>")


def copy_demo(app, exception):
    if exception or app.builder.format != "html":
        return
    log = logging.getLogger(__name__)
    out = app.outdir
    lib, jkp = os.path.join(out, "lib"), os.path.join(out, "lib", "jkp")
    for d in (lib, jkp, os.path.join(out, "demo"), os.path.join(out, "samples")):
        os.makedirs(d, exist_ok=True)

    wasm = os.environ.get("JMT_WASM_DIR", os.path.join(ROOT, "build", "wasm", "web"))
    missing = [f for f in MODULE_FILES if not os.path.isfile(os.path.join(wasm, f))]
    missing += [f"jkp/{f}" for f in JKP_FILES if not os.path.isfile(os.path.join(wasm, "jkp", f))]
    if missing:
        log.warning("the demo needs a browser build of the library with just-keypoints' module; not found in %s: "
                    "%s (set JMT_WASM_DIR; JMT_JKP_WEB_DIR when building)", wasm, ", ".join(missing))
    else:
        for f in MODULE_FILES:
            shutil.copyfile(os.path.join(wasm, f), os.path.join(lib, f))
        for f in JKP_FILES:
            shutil.copyfile(os.path.join(wasm, "jkp", f), os.path.join(jkp, f))
        log.info("demo modules copied from %s", wasm)

    with open(os.path.join(ROOT, "web", "demo.html"), encoding="utf-8") as f:
        page = site_demo(f.read())
    with open(os.path.join(out, "demo", "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    shutil.copyfile(os.path.join(ROOT, "web", "demo.js"), os.path.join(out, "demo", "demo.js"))
    samples = os.path.join(ROOT, "web", "samples")
    for f in sorted(os.listdir(samples)):
        if f.endswith(".jpg"):
            shutil.copyfile(os.path.join(samples, f), os.path.join(out, "samples", f))


def setup(app):
    app.connect("build-finished", copy_demo)
