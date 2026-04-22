# Architecture Overview

High-level structure

- `diffsurrogate/` — library source
  - `data/` — loader, transforms, splitter
  - `models/` — implementations of surrogate paradigms
  - `persistence/` — save/load for model bundles
  - `cli/` — CLI entrypoints
  - `evaluation/` — metrics, benchmarking, plotting

Design principles
- Single `SurrogateModel` interface for interchangeability
- Config-driven behavior via `config.toml`
- Optional dependencies for heavy backends (gpytorch, torch, chaospy)

Data flow (benchmark/train/predict)
1. Load input data (loader)
2. Apply transforms (scalers)
3. Split (benchmark) or fit on full data (train)
4. Build model from `models/` and fit
5. Save artifacts via `persistence/`
6. Predict/infer using saved artifacts (predict)
