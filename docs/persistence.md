# Persistence & Saved Artifacts

Saved models and metadata are organized under `saved_models/{model}/{mode}/` where `{mode}` is `benchmark` or `production`.

Typical contents of a production bundle (example for GP):

```
saved_models/gaussian_process/production/
├── sklearn_gp.joblib
├── gp_mode.json
├── scalers.joblib
└── metadata.json
```

Guidelines
- Save scalers and metadata together so inference uses the exact transforms from training.
- Keep `metadata.json` with timestamp, n_train, and the config snapshot used to create the artifact.
