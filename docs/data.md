# Data: Loading, Transforms, and Splitting

Data logic is implemented in `diffsurrogate/data/`.

Key modules
- `loader.py` — flexible loader supporting CSV, HDF5, and .npy structured arrays
- `transforms.py` — input and target scalers (log, zscore, robust)
- `splitter.py` — stratified split on `|t|` deciles to preserve diffractive dips

Recommended flow
1. Load lookup table via loader (specify `input_path` in config)
2. Fit transforms on training data only
3. Use `splitter` to create train/test for benchmarking

Transforms summary
- `q2_transform`, `w2_transform` — `log_zscore` or `zscore`
- `t_transform` — `log_stabilized` (log(|t| + eps) then z-score)
- `target_scaler` — `robust` (median/IQR) recommended for amplitude logs
