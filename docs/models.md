# Models Overview

This project implements multiple surrogate paradigms. Each model implements the abstract `SurrogateModel` interface located in [diffsurrogate/models/base.py](diffsurrogate/models/base.py).

Implemented paradigms

- `neural_net` — ResNet-style deterministic neural network (fast inference)
- `gaussian_process` — Exact & sparse GP backends (UQ-capable)
- `pinn` — Physics-informed neural network (embeds BK evolution and unitarity constraints)
- `fno` — Fourier Neural Operator across the `t` axis for gridded data
- `deep_gp` — Deep Gaussian Process implemented via variational inference
- `pce` — Polynomial Chaos Expansion with Hermite basis

Extending with a new model
1. Subclass `SurrogateModel` in `diffsurrogate/models/your_model.py`.
2. Add a config dataclass in `diffsurrogate/config.py` and register it in `ModelsConfig`.
3. Register the builder in `diffsurrogate/persistence/registry.py` and `build_model`.
4. Add `[models.your_model]` in `config.toml` and add to `models.enabled`.
5. Add a smoke test in `tests/test_models.py` following existing patterns.
