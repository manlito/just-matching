# Documentation site

Sphinx with MyST Markdown, built from this folder, with the site's own theme in `_theme/jmt` (just-keypoints'
theme, after just-ba's, with this project's accent and icon). After the pages, `conf.py` puts the live demo next
to them: `web/demo.html` with the site's top bar as `demo/`, the WebAssembly module and just-keypoints' module in
`lib/`, and the sample images in `samples/`. `_extra/` is copied as it is: the demo's top-bar style and
Cloudflare's `_headers`.

The site never serves FlexGlue's weights: the backbone is under Meta's DINOv3 License, and each user has to accept
it by fetching `flexglue.pth` from FlexGlue's release themselves. The demo takes the file from a button or a drop,
and keeps it in the browser's origin private file system only when the user ticks "Keep in this browser".

Rule for every page: what visitors read carries no references to the project's notes (PLAN.md, milestones,
commit hashes) and no links to the repository, which is private. `reproducing.md` is for the repository and is
not part of the site.

```bash
uv venv --python 3.12 docs/.venv && uv pip install --python docs/.venv/bin/python -r docs/requirements.txt
```

With a browser build in `build/wasm` that has just-keypoints' module (README.md, "In the browser";
`tools/fetch_jkp_web.sh` downloads the module before the build):

```bash
docs/.venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
python3 -m http.server 8793 --bind 127.0.0.1 --directory docs/_build/html   # then open /demo/ in Chrome
```

`JMT_WASM_DIR` points at another build's `web/` folder. Without the modules the build warns, so `-W` fails. The
site is about 3 MB.

## Deploying

The `docs` workflow (`.github/workflows/docs.yml`) runs on every push to `main`, so merging a pull request
builds the site and publishes it to Cloudflare Pages. It can also be started by hand:

```bash
gh workflow run docs                    # build the modules and the site
gh workflow run docs -f deploy=true     # the same, then publish to Cloudflare Pages
```

It takes just-keypoints' module from just-keypoints' site (`tools/fetch_jkp_web.sh`, pinned by sha256), so it
needs no access to that repository. When just-keypoints deploys a new module, the hashes stop matching and the
build fails until they are updated (see the script).

Deploying needs the repository secrets `CLOUDFLARE_API_TOKEN` (permission "Cloudflare Pages: Edit") and
`CLOUDFLARE_ACCOUNT_ID`, and a Pages project made for direct upload, named by the repository variable
`CLOUDFLARE_PAGES_PROJECT` (default `just-matching`). From a machine where wrangler is logged in, the same
upload is:

```bash
npx --yes wrangler@4.162.0 pages deploy docs/_build/html --project-name just-matching --branch main
```

