# FAQ

Q: Where are model hyperparameters configured?
A: In `config.toml` under `[models]` and each `[models.<name>]` subtable.

Q: What if I don't have `gpytorch` installed?
A: Tests and CLI skip GP models gracefully and emit a warning.

Q: How do I add a new model?
A: See [docs/models.md](models.md) for a step-by-step checklist.
