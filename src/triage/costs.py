"""Cost model and budget configuration."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from triage.exceptions import ContractViolation


class Budget(BaseModel):
    """Budget specification."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["fraction", "count", "money"]
    value: float

    @field_validator("value")
    @classmethod
    def value_positive(cls, v: float) -> float:
        """Budget value must be positive."""
        if v <= 0:
            raise ValueError("budget.value must be > 0")
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

    @field_validator("max_residual_risk")
    @classmethod
    def max_residual_risk_nonnegative(cls, v: float | None) -> float | None:
        """max_residual_risk must be >= 0."""
        if v is not None and v < 0:
            raise ValueError("gates.max_residual_risk must be >= 0")
        return v

    @field_validator("max_ece")
    @classmethod
    def max_ece_in_range(cls, v: float | None) -> float | None:
        """max_ece must be in (0, 1]."""
        if v is not None and (v <= 0 or v > 1):
            raise ValueError("gates.max_ece must be in (0, 1]")
        return v


class SegmentCosts(BaseModel):
    """Costs for a segment."""

    model_config = ConfigDict(extra="forbid")

    false_accept_cost: float
    review_cost: float

    @field_validator("false_accept_cost", "review_cost")
    @classmethod
    def costs_nonnegative(cls, v: float) -> float:
        """Costs must be >= 0."""
        if v < 0:
            raise ValueError("Costs must be >= 0")
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
