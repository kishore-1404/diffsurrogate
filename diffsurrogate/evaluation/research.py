"""Research-facing benchmark summaries and prompt packs."""

from __future__ import annotations

from typing import Any

MODEL_METHOD_CARDS: dict[str, dict[str, Any]] = {
    "neural_net": {
        "family": "Feed-forward neural network / ResNet",
        "probabilistic": False,
        "physics_informed": False,
        "grid_requirement": "None",
        "compute_profile": "Low-to-moderate training cost, very fast inference",
        "strengths": "Strong deterministic baseline, easy to scale, simple deployment",
        "limitations": "No calibrated uncertainty by default and limited extrapolation guarantees",
    },
    "gaussian_process": {
        "family": "Exact or sparse Gaussian process",
        "probabilistic": True,
        "physics_informed": False,
        "grid_requirement": "None",
        "compute_profile": "High training cost, especially for exact GP; moderate inference cost",
        "strengths": "Excellent small-data performance and principled predictive uncertainty",
        "limitations": "Scaling is the main bottleneck for larger lookup tables",
    },
    "pinn": {
        "family": "Physics-informed neural network",
        "probabilistic": False,
        "physics_informed": True,
        "grid_requirement": "None",
        "compute_profile": "Moderate-to-high training cost because physics residuals are enforced",
        "strengths": "Can encode domain priors and improve physical consistency under sparse supervision",
        "limitations": "Optimization can be delicate and uncertainty is not native",
    },
    "fno": {
        "family": "Fourier neural operator",
        "probabilistic": False,
        "physics_informed": False,
        "grid_requirement": "Best on gridded t slices; otherwise falls back to MLP mode",
        "compute_profile": "Moderate training cost with excellent amortized inference on operator-like data",
        "strengths": "Good fit for structured grids and fast surrogate evaluation across related slices",
        "limitations": "Advantages shrink when data are irregular or weakly operator-structured",
    },
    "deep_gp": {
        "family": "Deep Gaussian process",
        "probabilistic": True,
        "physics_informed": False,
        "grid_requirement": "None",
        "compute_profile": "High training cost due to variational inference and Monte Carlo sampling",
        "strengths": "Captures richer nonlinearity than a shallow GP while retaining uncertainty estimates",
        "limitations": "Optimization variance and heavier runtime than exact GP or deterministic nets",
    },
    "pce": {
        "family": "Polynomial chaos expansion",
        "probabilistic": True,
        "physics_informed": False,
        "grid_requirement": "None",
        "compute_profile": "Very fast once the basis is well specified",
        "strengths": "Interpretable spectral approximation and cheap repeated evaluation",
        "limitations": "Can degrade on sharp local structure, high dimensionality, or poor basis choice",
    },
}


def method_rows(model_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in model_names:
        card = MODEL_METHOD_CARDS.get(name, {})
        rows.append({
            "model": name,
            "family": card.get("family", "Unknown"),
            "probabilistic": bool(card.get("probabilistic", False)),
            "physics_informed": bool(card.get("physics_informed", False)),
            "grid_requirement": card.get("grid_requirement", "Unknown"),
            "compute_profile": card.get("compute_profile", "Unknown"),
            "strengths": card.get("strengths", "Unknown"),
            "limitations": card.get("limitations", "Unknown"),
        })
    return rows


def build_research_prompts(*, results: list[dict[str, Any]], metrics: list[str], model_names: list[str], run_id: str) -> str:
    enabled_metrics = ", ".join(metrics)
    methods = ", ".join(model_names)
    lines = [
        "# Research Benchmark Prompt Pack",
        "",
        f"Run ID: `{run_id}`",
        f"Compared methods: {methods}",
        f"Primary metrics: {enabled_metrics}",
        "",
        "## Prompt 1: Paper-ready benchmark interpretation",
        "Analyze the benchmark bundle for this run as if preparing the evaluation section of a research paper.",
        "Report which method is strongest overall, then explain where that conclusion is fragile.",
        "Compare all methods across predictive accuracy, uncertainty quality, runtime, physical plausibility, and expected scaling behavior.",
        "Use the saved per-model prediction files to comment on where each surrogate fails in kinematic space, especially around large |t|, diffractive dips, and any systematic residual structure.",
        "Explicitly separate observations supported directly by the benchmark artifacts from hypotheses that would require additional experiments.",
        "",
        "## Prompt 2: Reviewer-style critical comparison",
        "Write a skeptical reviewer-style assessment of the benchmark methodology.",
        "Check whether train/test splitting, metric choice, uncertainty reporting, and artifact preservation are sufficient for a reproducible scientific claim.",
        "Identify what is convincing, what is missing, and which ablations or statistical tests should be added before publication.",
        "",
        "## Prompt 3: Method comparison narrative",
        f"Construct a method-by-method narrative comparing {methods}.",
        "For each surrogate, discuss inductive bias, expected sample efficiency, uncertainty behavior, computational cost, and likely failure modes on structured scattering-amplitude tables.",
        "Then explain why the benchmark results are or are not consistent with those expectations.",
        "",
        "## Prompt 4: Follow-up experiment design",
        "Design the next round of experiments after this benchmark.",
        "Include repeated-seed studies, sensitivity to train_fraction, robustness near sparse kinematic regions, calibration analysis for probabilistic models, and error stratification over Q2, W2, and |t| bins.",
        "Specify which saved artifacts from this run can be reused directly and which new artifacts should be added.",
    ]
    return "\n".join(lines) + "\n"
