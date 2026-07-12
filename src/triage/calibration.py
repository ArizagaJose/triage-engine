"""Calibration: isotonic regression + Platt scaling, ECE, bootstrap CIs."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for calibration procedure."""

    holdout_fraction: float = 0.3
    n_bins: int = 10
    n_bootstrap: int = 1000
    ci_level: float = 0.95


@dataclass(frozen=True)
class Calibrator:
    """Fitted calibrator mapping raw score to calibrated p_error."""

    method: str  # "isotonic" | "platt"
    params: dict[str, Any]
    ece_holdout: float
    ece_ci: tuple[float, float]
    ece_raw_holdout: float
    n_golden: int
    seed: int
    input_sha256: str
    tool_version: str
    config: CalibrationConfig
    reliability_bins: dict[str, Any]

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Predict calibrated p_error from raw scores (sklearn-free)."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Calibrator":
        """Deserialize from dict."""
        raise NotImplementedError


def fit(
    golden: pl.DataFrame,
    *,
    seed: int,
    config: CalibrationConfig = CalibrationConfig(),
    input_sha256: str = "",
) -> Calibrator:
    """Fit calibrator on golden set.

    Returns Calibrator with fitted isotonic or Platt scaling, ECE on holdout,
    bootstrap CI, and reliability bins for both raw and calibrated predictions.
    """
    raise NotImplementedError


def ece(p_pred: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Compute Expected Calibration Error (ECE).

    Equal-width bins on [0, 1]; per bin sum |mean(p_pred) - mean(y)| * (n_bin / n_total).
    """
    raise NotImplementedError


def check_ece_gate(calibrator: Calibrator, max_ece: float) -> None:
    """Check if calibrator's ECE exceeds gate; raise CalibrationGateFailed if so."""
    raise NotImplementedError
