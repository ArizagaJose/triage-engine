"""Cost model and budget configuration."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


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
        raise NotImplementedError


def load(path: Path) -> CostModel:
    """Load and parse YAML cost model file.

    YAML parse errors and pydantic ValidationErrors are wrapped into
    ContractViolation with a readable message.
    """
    raise NotImplementedError
