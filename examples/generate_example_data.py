"""Generate a small synthetic diffractive lookup table for examples/tests.

Target structure: a smooth-in-(Q², W²) exponential decay in |t|, modulated by
sin(k·t) to produce diffractive-dip-like minima. Schema matches the real
Sartre lookup tables:

    Q2_log_center   — log(Q²)
    W2_center       — W² in GeV²
    t_center        — Mandelstam t in GeV² (negative)
    logA            — natural log of the squared scattering amplitude

Usage:  python generate_example_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def diffractive_amplitude(log_q2: np.ndarray, w2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Toy diffractive amplitude with a dip structure in |t|.

    Designed so that:
      - smooth monotone fall-off with |t|  (exp(-B|t|))
      - diffractive minimum around |t| ≈ 0.4 GeV²  (sin² factor)
      - mild logarithmic growth with W²
      - mild suppression with log Q²
    """
    abs_t = np.abs(t)
    # B-slope shrinks slightly with energy (shrinkage of the diffractive cone).
    B = 5.0 + 0.2 * np.log(w2 / 10.0)
    # |sin(pi * |t| / t_dip)| produces a zero at |t| = t_dip.
    t_dip = 0.4
    sin_factor = np.abs(np.sin(np.pi * abs_t / t_dip))
    # Base amplitude. Small floor so we don't hit log(0) at the dip.
    A2 = np.exp(-B * abs_t) * (sin_factor ** 2 + 1e-4)
    # Q2 suppression.
    A2 *= np.exp(-0.05 * log_q2)
    # W2 logarithmic boost.
    A2 *= (1.0 + 0.02 * np.log(w2 / 10.0))
    return np.log(A2)


def make_lookup_table(
    q2_values: np.ndarray,
    w2_values: np.ndarray,
    t_values: np.ndarray,
    noise_std: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for q2 in q2_values:
        for w2 in w2_values:
            for t in t_values:
                log_q2 = np.log(q2)
                logA = diffractive_amplitude(
                    np.array([log_q2]), np.array([w2]), np.array([t])
                )[0]
                # Small aleatoric noise (mimicking Monte Carlo integration noise).
                logA += rng.normal(0.0, noise_std)
                rows.append({
                    "Q2_log_center": float(log_q2),
                    "W2_center": float(w2),
                    "t_center": float(t),
                    "logA": float(logA),
                })
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Training lookup table: gridded on t so FNO-1d can use it. ---
    q2_values = np.array([1.0, 3.0, 5.0, 10.0])          # 4 values
    w2_values = np.array([20.0, 40.0, 80.0, 160.0])      # 4 values
    # 60-point t grid, biased toward small |t| where the diffractive peak lives.
    t_values = -np.concatenate([
        np.linspace(0.01, 0.25, 25),   # fine near t=0
        np.linspace(0.26, 1.00, 25),   # mid range (covers the dip at 0.4)
        np.linspace(1.02, 2.00, 10),   # coarse tail
    ])
    df = make_lookup_table(q2_values, w2_values, t_values, noise_std=0.02, seed=42)
    df.to_csv(out_dir / "synthetic_lookup.csv", index=False)
    print(f"Wrote {out_dir / 'synthetic_lookup.csv'}  ({len(df)} rows)")

    # --- Prediction inputs: a handful of new points. ---
    pred_rows = []
    rng = np.random.default_rng(2026)
    for _ in range(20):
        q2 = float(rng.uniform(1.0, 10.0))
        w2 = float(rng.uniform(20.0, 160.0))
        t = -float(rng.uniform(0.02, 1.5))
        pred_rows.append({
            "Q2_log_center": float(np.log(q2)),
            "W2_center": w2,
            "t_center": t,
        })
    pd.DataFrame(pred_rows).to_csv(out_dir / "predict_inputs.csv", index=False)
    print(f"Wrote {out_dir / 'predict_inputs.csv'}  ({len(pred_rows)} rows)")


if __name__ == "__main__":
    main()
