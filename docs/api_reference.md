# API Reference (Core Symbols)

This is a concise reference to the core classes and modules you will interact with.

- `diffsurrogate.models.base.SurrogateModel` — abstract interface all models implement. Key methods: `name()`, `supports_uq()`, `fit(X,y)`, `predict(X)`, `save(path)`, `load(path)`.
- `diffsurrogate.config` — config dataclasses used to parse `config.toml`.
- `diffsurrogate.persistence.registry` — maps model name strings → (class, config) builders for loading and saving artifacts.
- `diffsurrogate.data.loader` — flexible data loader for CSV/HDF5/NPY inputs.
- `diffsurrogate.data.transforms` — input and target scaling utilities.
- `diffsurrogate.cli.*` — CLI commands wired to the main entrypoints.

For deeper exploration, open the corresponding module files in the repository and consult inline docstrings.
