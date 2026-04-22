# Installation

This page describes how to install `diffsurrogate` and optional extras.

Prerequisites
- Python 3.11 or later
- pip or an equivalent installer

Install repository in editable mode (recommended for development):

```bash
git clone <repo>
cd diffsurrogate
pip install -e .[all]
```

Available extras (choose what's needed):
- `torch` — Neural network backends (NN, PINN, FNO)
- `gpytorch` — Gaussian Process backends
- `chaospy` — Polynomial Chaos Expansion
- `test` — test dependencies (`pytest`)

Notes
- Missing optional dependencies are handled gracefully: missing models are skipped with a warning.
