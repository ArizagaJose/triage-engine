"""Cost model and budget configuration."""

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from triage.calibration import DEFAULT_ECE_GATE_MODE, EceGateMode
from triage.exceptions import ContractViolation


class Budget(BaseModel):
    """Budget specification."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["fraction", "count", "money"]
    value: float

    @field_validator("value")
    @classmethod
    def value_positive(cls, v: float) -> float:
        """Budget value must be positive and finite. `v <= 0` alone let NaN
        through (every NaN comparison is False), which crashed the optimizer's
        floor() with an internal error (exit 4) instead of exit 1."""
        if not math.isfinite(v) or v <= 0:
            raise ValueError("budget.value must be > 0 and finite")
        return v

    @field_validator("value")
    @classmethod
    def value_fraction_lte_one(cls, v: float, info: Any) -> float:
        """If type is 'fraction', value must be <= 1.0."""
        if info.data.get("type") == "fraction" and v > 1.0:
            raise ValueError("budget.value must be <= 1.0 when type is 'fraction'")
        return v


class Gates(BaseModel):
    """Quality gates for calibration and residual risk."""

    model_config = ConfigDict(extra="forbid")

    max_residual_risk: float | None = None
    max_ece: float | None = None
    # Which ECE statistic is compared against max_ece: "point" (ece_holdout,
    # the default) or "ci_upper" (upper bound of the bootstrap CI, ece_ci[1]).
    ece_gate_mode: EceGateMode = DEFAULT_ECE_GATE_MODE

    @field_validator("max_residual_risk")
    @classmethod
    def max_residual_risk_nonnegative(cls, v: float | None) -> float | None:
        """max_residual_risk must be >= 0 (inf allowed: never fails). Written
        as a positive check so NaN is rejected: a NaN threshold would silently
        disable the gate (`residual > nan` is always False)."""
        if v is not None and not (v >= 0):
            raise ValueError("gates.max_residual_risk must be >= 0")
        return v

    @field_validator("max_ece")
    @classmethod
    def max_ece_in_range(cls, v: float | None) -> float | None:
        """max_ece must be in (0, 1]. Written as a positive range check so that
        NaN (for which every comparison is False) is rejected too: a NaN
        threshold would otherwise silently disable the gate."""
        if v is not None and not (0 < v <= 1):
            raise ValueError("gates.max_ece must be in (0, 1]")
        return v

    @model_validator(mode="after")
    def ece_gate_mode_requires_max_ece(self) -> "Gates":
        """A non-default ece_gate_mode without max_ece would silently gate
        nothing; reject it instead of pretending a stricter gate is active."""
        if self.ece_gate_mode != DEFAULT_ECE_GATE_MODE and self.max_ece is None:
            raise ValueError(
                f"gates.ece_gate_mode={self.ece_gate_mode!r} requires gates.max_ece to be set"
            )
        return self

    @model_serializer(mode="wrap")
    def _omit_default_ece_gate_mode(self, handler: SerializerFunctionWrapHandler) -> Any:
        """Omit ece_gate_mode from dumps when it is the default.

        policy.json's `config_snapshot` is `CostModel.model_dump()` and feeds
        `policy_hash`. Always emitting the new key would change the hash of
        every default-mode policy (and the committed golden fixtures) although
        nothing semantic changed; absent == "point" keeps them byte-identical
        and old artifacts (which lack the key) validate to the same model.
        """
        data = handler(self)
        if isinstance(data, dict) and data.get("ece_gate_mode") == DEFAULT_ECE_GATE_MODE:
            del data["ece_gate_mode"]
        return data


class SegmentCosts(BaseModel):
    """Costs for a segment."""

    model_config = ConfigDict(extra="forbid")

    false_accept_cost: float
    review_cost: float

    @field_validator("false_accept_cost", "review_cost")
    @classmethod
    def costs_nonnegative(cls, v: float) -> float:
        """Costs must be finite and >= 0. NaN/inf costs used to pass and then
        crash the artifact writer (JSON forbids NaN/inf) with exit 4."""
        if not math.isfinite(v) or v < 0:
            raise ValueError("Costs must be >= 0 and finite")
        return v


class CostModel(BaseModel):
    """Full cost model with budget, gates, and per-segment costs."""

    model_config = ConfigDict(extra="forbid")

    budget: Budget
    gates: Gates = Gates()
    segments: dict[str, SegmentCosts]

    @field_validator("segments")
    @classmethod
    def segments_nonempty(cls, v: dict[str, SegmentCosts]) -> dict[str, SegmentCosts]:
        """segments dict must be non-empty."""
        if not v:
            raise ValueError("segments must be non-empty")
        return v

    def costs_for(self, segment: str) -> SegmentCosts:
        """Get costs for segment: exact key, else 'default', else raise ContractViolation."""
        if segment in self.segments:
            return self.segments[segment]
        if "default" in self.segments:
            return self.segments["default"]
        raise ContractViolation(
            f"Segment '{segment}' not found in cost model and no 'default' segment defined"
        )


def load(path: Path) -> CostModel:
    """Load and parse YAML cost model file.

    YAML parse errors and pydantic ValidationErrors are wrapped into
    ContractViolation with a readable message.
    """
    # Try to read the file
    try:
        content = path.read_text()
    except FileNotFoundError as e:
        raise ContractViolation(f"Cost model file not found: {path}") from e
    except (OSError, IOError) as e:
        raise ContractViolation(f"Cannot read cost model file {path}: {e}") from e

    # Try to parse YAML
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ContractViolation(f"Invalid YAML in cost model file {path}: {e}") from e

    # Check that the top level is a dict
    if not isinstance(data, dict):
        raise ContractViolation(
            f"Cost model file {path} must contain a YAML dictionary (object) at top level, "
            f"got {type(data).__name__}"
        )

    # Try to validate into CostModel
    try:
        return CostModel(**data)
    except ValidationError as e:
        # Format validation errors in a readable way
        error_lines = []
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            error_lines.append(f"  {loc}: {msg}")
        error_summary = "\n".join(error_lines)
        raise ContractViolation(f"Cost model validation failed for {path}:\n{error_summary}") from e
