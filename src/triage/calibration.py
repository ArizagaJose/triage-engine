"""Calibration: isotonic regression + Platt scaling, ECE, bootstrap CIs.

See docs/interfaces.md §4 for the pinned contract. Fitting uses sklearn;
prediction is sklearn-free (built from extracted params only) so that JSON
round-trips are bit-exact.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from triage import __version__
from triage.contract import validate
from triage.exceptions import CalibrationGateFailed


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
        scores = np.asarray(scores, dtype=np.float64)
        if self.method == "isotonic":
            x_thresholds = np.asarray(self.params["x_thresholds"], dtype=np.float64)
            y_thresholds = np.asarray(self.params["y_thresholds"], dtype=np.float64)
            if x_thresholds.size == 1:
                # Degenerate single-threshold case: np.interp requires at
                # least 2 points with strictly increasing x for a real
                # interpolation; with a single point every input maps to
                # that point's y, which is exactly out_of_bounds="clip"
                # behavior (nothing to interpolate between).
                return np.full_like(scores, y_thresholds[0], dtype=np.float64)
            interpolated: np.ndarray = np.interp(scores, x_thresholds, y_thresholds)
            return interpolated
        elif self.method == "platt":
            a = float(self.params["a"])
            b = float(self.params["b"])
            z = a * scores + b
            return 1.0 / (1.0 + np.exp(-z))
        else:  # pragma: no cover - guarded by fit()/from_dict()
            raise ValueError(f"Unknown calibration method: {self.method!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (top-level keys per interfaces.md §7,
        minus `created_at` which the CLI adds at write time)."""
        return {
            "method": self.method,
            "params": {k: _to_jsonable(v) for k, v in self.params.items()},
            "ece_holdout": float(self.ece_holdout),
            "ece_ci": [float(self.ece_ci[0]), float(self.ece_ci[1])],
            "ece_raw_holdout": float(self.ece_raw_holdout),
            "n_golden": int(self.n_golden),
            "seed": int(self.seed),
            "input_sha256": self.input_sha256,
            "tool_version": self.tool_version,
            "config": asdict(self.config),
            "reliability_bins": _to_jsonable(self.reliability_bins),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Calibrator":
        """Deserialize from dict (as produced by to_dict(), plus an optional
        `created_at` key which is ignored)."""
        config = CalibrationConfig(**d["config"])
        return cls(
            method=d["method"],
            params=dict(d["params"]),
            ece_holdout=float(d["ece_holdout"]),
            ece_ci=(float(d["ece_ci"][0]), float(d["ece_ci"][1])),
            ece_raw_holdout=float(d["ece_raw_holdout"]),
            n_golden=int(d["n_golden"]),
            seed=int(d["seed"]),
            input_sha256=d["input_sha256"],
            tool_version=d["tool_version"],
            config=config,
            reliability_bins=d["reliability_bins"],
        )


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def ece(p_pred: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Compute Expected Calibration Error (ECE).

    Equal-width bins on [0, 1] over `p_pred`; per non-empty bin,
    |mean(p_pred) - mean(y)| * (n_bin / n_total); summed.

    Binning convention: bin index = floor(clip(p, 0, 1) * n_bins), with the
    edge case p == 1.0 mapped into the last bin (index n_bins - 1) rather
    than spilling into a phantom n_bins-th bin.
    """
    p_pred = np.asarray(p_pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_total = p_pred.shape[0]
    if n_total == 0:
        return 0.0

    p_clipped = np.clip(p_pred, 0.0, 1.0)
    bin_idx = np.floor(p_clipped * n_bins).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    total = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        mean_pred = float(p_pred[mask].mean())
        mean_obs = float(y[mask].mean())
        total += abs(mean_pred - mean_obs) * (n_bin / n_total)
    return total


def _reliability_bins(p_pred: np.ndarray, y: np.ndarray, n_bins: int) -> list[dict[str, Any]]:
    """Per-bin reliability stats on [0, 1], equal-width, empty bins included
    with count 0 and null means. Uses the same binning convention as ece()."""
    p_pred = np.asarray(p_pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_total = p_pred.shape[0]

    p_clipped = np.clip(p_pred, 0.0, 1.0)
    bin_idx = np.floor(p_clipped * n_bins).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bins: list[dict[str, Any]] = []
    for b in range(n_bins):
        bin_lo = b / n_bins
        bin_hi = (b + 1) / n_bins
        if n_total == 0:
            mask = np.zeros(0, dtype=bool)
        else:
            mask = bin_idx == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            bins.append(
                {
                    "bin_lo": bin_lo,
                    "bin_hi": bin_hi,
                    "mean_pred": None,
                    "mean_observed": None,
                    "count": 0,
                }
            )
        else:
            bins.append(
                {
                    "bin_lo": bin_lo,
                    "bin_hi": bin_hi,
                    "mean_pred": float(p_pred[mask].mean()),
                    "mean_observed": float(y[mask].mean()),
                    "count": n_bin,
                }
            )
    return bins


def _split_indices(
    item_ids: list[str], *, seed: int, holdout_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Stable sort by item_id, then permute with default_rng(seed); the last
    `holdout_fraction` of the permuted order is the holdout.

    Rounding convention: n_holdout = round(holdout_fraction * n), using
    Python/NumPy banker's rounding (round-half-to-even) on the float
    product. This is the natural, unambiguous choice given a single pinned
    formula in interfaces.md; it is deterministic and symmetric, unlike
    floor/ceil which would systematically bias holdout size up or down.
    """
    n = len(item_ids)
    order = np.argsort(np.asarray(item_ids), kind="stable")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    permuted_order = order[perm]
    n_holdout = round(holdout_fraction * n)
    n_holdout = max(0, min(n, n_holdout))
    n_cal = n - n_holdout
    cal_idx = permuted_order[:n_cal]
    holdout_idx = permuted_order[n_cal:]
    return cal_idx, holdout_idx


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
    golden = validate(golden, require_truth=True)

    item_ids = golden["item_id"].to_list()
    scores_all = golden["score"].to_numpy().astype(np.float64)
    error_all = (
        (golden["predicted_value"] != golden["true_value"]).cast(pl.Int64).to_numpy()
    ).astype(np.float64)

    cal_idx, holdout_idx = _split_indices(
        item_ids, seed=seed, holdout_fraction=config.holdout_fraction
    )

    score_cal = scores_all[cal_idx]
    error_cal = error_all[cal_idx]
    score_hold = scores_all[holdout_idx]
    error_hold = error_all[holdout_idx]

    # --- Fit isotonic ---
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing="auto", out_of_bounds="clip")
    iso.fit(score_cal, error_cal)
    iso_x = np.asarray(iso.X_thresholds_, dtype=np.float64)
    iso_y = np.asarray(iso.y_thresholds_, dtype=np.float64)
    iso_params: dict[str, Any] = {
        "x_thresholds": iso_x.tolist(),
        "y_thresholds": iso_y.tolist(),
    }
    iso_calibrator_partial = Calibrator(
        method="isotonic",
        params=iso_params,
        ece_holdout=0.0,
        ece_ci=(0.0, 0.0),
        ece_raw_holdout=0.0,
        n_golden=golden.height,
        seed=seed,
        input_sha256=input_sha256,
        tool_version=__version__,
        config=config,
        reliability_bins={},
    )
    iso_pred_hold = iso_calibrator_partial.predict(score_hold)
    iso_ece_hold = ece(iso_pred_hold, error_hold, config.n_bins)

    # --- Fit Platt (1-D logistic regression) ---
    # LogisticRegression requires both classes in its training data; on a
    # single-class calibration split (e.g. a small golden set with zero
    # errors) its lbfgs solver raises ValueError. Isotonic handles a
    # constant target fine (it fits a constant predictor), so in that
    # degenerate case we skip the Platt fit entirely and select isotonic.
    both_classes_present = bool(np.any(error_cal == 0.0)) and bool(np.any(error_cal == 1.0))
    if both_classes_present:
        lr = LogisticRegression(C=1e10, solver="lbfgs")
        lr.fit(score_cal.reshape(-1, 1), error_cal)
        # LogisticRegression predicts P(class 1); class 1 is "error".
        a = float(lr.coef_[0][0])
        b = float(lr.intercept_[0])
        platt_params: dict[str, Any] = {"a": a, "b": b}
        platt_calibrator_partial = Calibrator(
            method="platt",
            params=platt_params,
            ece_holdout=0.0,
            ece_ci=(0.0, 0.0),
            ece_raw_holdout=0.0,
            n_golden=golden.height,
            seed=seed,
            input_sha256=input_sha256,
            tool_version=__version__,
            config=config,
            reliability_bins={},
        )
        platt_pred_hold = platt_calibrator_partial.predict(score_hold)
        platt_ece_hold = ece(platt_pred_hold, error_hold, config.n_bins)
    else:
        platt_params = {}
        platt_pred_hold = np.empty(0, dtype=np.float64)
        platt_ece_hold = float("inf")

    # --- Selection: lower ECE wins; tie -> isotonic ---
    if platt_ece_hold < iso_ece_hold:
        method = "platt"
        params = platt_params
        ece_hold = platt_ece_hold
        pred_hold = platt_pred_hold
    else:
        method = "isotonic"
        params = iso_params
        ece_hold = iso_ece_hold
        pred_hold = iso_pred_hold

    # --- Raw ("before") ECE on holdout: p_pred = 1 - score ---
    raw_pred_hold = 1.0 - score_hold
    ece_raw_hold = ece(raw_pred_hold, error_hold, config.n_bins)

    # --- Bootstrap CI on ece_holdout of the SELECTED calibrator ---
    ece_ci = _bootstrap_ece_ci(
        pred_hold,
        error_hold,
        n_bins=config.n_bins,
        n_bootstrap=config.n_bootstrap,
        ci_level=config.ci_level,
        seed=seed + 1,
    )

    reliability_bins = {
        "before": _reliability_bins(raw_pred_hold, error_hold, config.n_bins),
        "after": _reliability_bins(pred_hold, error_hold, config.n_bins),
    }

    return Calibrator(
        method=method,
        params=params,
        ece_holdout=ece_hold,
        ece_ci=ece_ci,
        ece_raw_holdout=ece_raw_hold,
        n_golden=golden.height,
        seed=seed,
        input_sha256=input_sha256,
        tool_version=__version__,
        config=config,
        reliability_bins=reliability_bins,
    )


def _bootstrap_ece_ci(
    pred_hold: np.ndarray,
    error_hold: np.ndarray,
    *,
    n_bins: int,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float]:
    """Resample the holdout with replacement n_bootstrap times, recompute
    ECE of the given predictions each time, and return a percentile CI."""
    n = pred_hold.shape[0]
    rng = np.random.default_rng(seed)
    if n == 0:
        return (0.0, 0.0)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = ece(pred_hold[idx], error_hold[idx], n_bins)
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.percentile(samples, 100 * alpha))
    hi = float(np.percentile(samples, 100 * (1.0 - alpha)))
    return (lo, hi)


def check_ece_gate(calibrator: Calibrator, max_ece: float) -> None:
    """Check if calibrator's ECE exceeds gate; raise CalibrationGateFailed if so."""
    if calibrator.ece_holdout > max_ece:
        raise CalibrationGateFailed(
            f"Calibration quality gate failed: ece_holdout={calibrator.ece_holdout!r} "
            f"exceeds max_ece={max_ece!r}"
        )
