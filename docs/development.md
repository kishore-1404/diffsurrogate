# Development Guide

This document explains how to set up a development environment, run tests, and add new models.

Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test,torch,gpytorch,chaospy]
```

Running tests

```bash
pytest tests/ -v
```

Style and formatting

- `black .` to format
- `ruff .` to lint

Adding a new model — checklist
1. Implement `SurrogateModel` subclass in `diffsurrogate/models/`
2. Add config dataclass in `diffsurrogate/config.py` and register
3. Register persistence builder in `diffsurrogate/persistence/registry.py`
4. Add CLI/config entries if necessary
5. Add tests in `tests/test_models.py`
