# Building the documentation site (MkDocs)

This project includes a ready-to-use `mkdocs.yml` at the repository root. Use MkDocs to build and serve the docs as a static site.

Install MkDocs (recommended with the Material theme):

```bash
python -m pip install "mkdocs>=1.5" "mkdocs-material>=9.0"
```

Local preview (live-reload):

```bash
mkdocs serve
# Visit http://127.0.0.1:8000
```

Build static site (for publishing):

```bash
mkdocs build --clean
# Output in site/ directory
```

Publishing options
- GitHub Pages: `mkdocs gh-deploy` (requires repo push rights)
- Netlify / Vercel: point to `site/` build output or run `mkdocs build` in CI

Notes
- `mkdocs-material` is optional; MkDocs will use a default theme if not installed.
- If you prefer Sphinx, I can add a `docs/` -> Sphinx conversion and `conf.py`.
