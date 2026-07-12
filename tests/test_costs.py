"""Unit tests for cost model and budget configuration."""

from pathlib import Path

import pytest

from triage.costs import Budget, CostModel, Gates, SegmentCosts, load
from triage.exceptions import ContractViolation


class TestCostModelCostsFor:
    """Test CostModel.costs_for method."""

    def test_costs_for_exact_match(self) -> None:
        """Test that costs_for returns exact segment match."""
        model = CostModel(
            budget=Budget(type="fraction", value=0.10),
            segments={
                "compensation_field": SegmentCosts(false_accept_cost=500.0, review_cost=4.0),
                "address_field": SegmentCosts(false_accept_cost=10.0, review_cost=2.0),
            },
        )
        costs = model.costs_for("compensation_field")
        assert costs.false_accept_cost == 500.0
        assert costs.review_cost == 4.0

    def test_costs_for_default_fallback(self) -> None:
        """Test that costs_for falls back to 'default' segment."""
        model = CostModel(
            budget=Budget(type="fraction", value=0.10),
            segments={
                "compensation_field": SegmentCosts(false_accept_cost=500.0, review_cost=4.0),
                "default": SegmentCosts(false_accept_cost=50.0, review_cost=3.0),
            },
        )
        costs = model.costs_for("unknown_segment")
        assert costs.false_accept_cost == 50.0
        assert costs.review_cost == 3.0

    def test_costs_for_missing_raises(self) -> None:
        """Test that costs_for raises when segment not found and no default."""
        model = CostModel(
            budget=Budget(type="fraction", value=0.10),
            segments={
                "compensation_field": SegmentCosts(false_accept_cost=500.0, review_cost=4.0),
            },
        )
        with pytest.raises(ContractViolation) as exc_info:
            model.costs_for("missing_segment")
        assert "missing_segment" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()


class TestBudget:
    """Test Budget model validators."""

    def test_budget_positive_value(self) -> None:
        """Test that budget value must be positive."""
        with pytest.raises(ValueError, match="must be > 0"):
            Budget(type="fraction", value=0)
        with pytest.raises(ValueError, match="must be > 0"):
            Budget(type="fraction", value=-0.1)

    def test_budget_fraction_lte_one(self) -> None:
        """Test that budget fraction must be <= 1.0."""
        with pytest.raises(ValueError, match="must be <= 1.0"):
            Budget(type="fraction", value=1.5)
        # Valid: exactly 1.0 for fraction
        budget = Budget(type="fraction", value=1.0)
        assert budget.value == 1.0

    def test_budget_count_accepts_any_positive(self) -> None:
        """Test that budget type 'count' accepts any positive value."""
        budget = Budget(type="count", value=100)
        assert budget.value == 100
        budget = Budget(type="count", value=0.5)
        assert budget.value == 0.5

    def test_budget_money_accepts_any_positive(self) -> None:
        """Test that budget type 'money' accepts any positive value."""
        budget = Budget(type="money", value=1000.50)
        assert budget.value == 1000.50


class TestGates:
    """Test Gates model validators."""

    def test_gates_max_residual_risk_nonnegative(self) -> None:
        """Test that max_residual_risk must be >= 0."""
        with pytest.raises(ValueError, match="must be >= 0"):
            Gates(max_residual_risk=-1)
        # Valid: 0
        gates = Gates(max_residual_risk=0)
        assert gates.max_residual_risk == 0

    def test_gates_max_ece_in_range(self) -> None:
        """Test that max_ece must be in (0, 1]."""
        # Invalid: 0
        with pytest.raises(ValueError, match="must be in"):
            Gates(max_ece=0)
        # Invalid: > 1
        with pytest.raises(ValueError, match="must be in"):
            Gates(max_ece=1.5)
        # Valid: exactly 1.0
        gates = Gates(max_ece=1.0)
        assert gates.max_ece == 1.0
        # Valid: between 0 and 1
        gates = Gates(max_ece=0.05)
        assert gates.max_ece == 0.05

    def test_gates_both_omitted_ok(self) -> None:
        """Test that both gates can be omitted."""
        gates = Gates()
        assert gates.max_residual_risk is None
        assert gates.max_ece is None


class TestSegmentCosts:
    """Test SegmentCosts model validators."""

    def test_segment_costs_negative_rejected(self) -> None:
        """Test that negative costs are rejected."""
        with pytest.raises(ValueError, match="must be >= 0"):
            SegmentCosts(false_accept_cost=-1, review_cost=3.0)
        with pytest.raises(ValueError, match="must be >= 0"):
            SegmentCosts(false_accept_cost=500.0, review_cost=-0.5)

    def test_segment_costs_zero_accepted(self) -> None:
        """Test that zero costs are accepted."""
        costs = SegmentCosts(false_accept_cost=0, review_cost=0)
        assert costs.false_accept_cost == 0
        assert costs.review_cost == 0


class TestCostModel:
    """Test CostModel validators."""

    def test_cost_model_empty_segments_rejected(self) -> None:
        """Test that empty segments dict is rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            CostModel(budget=Budget(type="fraction", value=0.10), segments={})

    def test_cost_model_extra_keys_rejected(self) -> None:
        """Test that extra/unknown keys are rejected."""
        with pytest.raises(ValueError, match="extra_field"):
            CostModel(
                budget=Budget(type="fraction", value=0.10),
                gates=Gates(),
                segments={"default": SegmentCosts(false_accept_cost=50.0, review_cost=3.0)},
                extra_field="should_fail",  # type: ignore
            )


class TestLoadFunction:
    """Test load() function for loading YAML cost model files."""

    def test_load_happy_path(self, tmp_path: Path) -> None:
        """Test load with exact YAML example from GOAL.md §2."""
        yaml_content = """\
budget:
  type: fraction
  value: 0.10
gates:
  max_residual_risk: 250.0
  max_ece: 0.05
segments:
  compensation_field:
    false_accept_cost: 500.0
    review_cost: 4.0
  address_field:
    false_accept_cost: 10.0
    review_cost: 2.0
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
"""
        yaml_file = tmp_path / "costs.yaml"
        yaml_file.write_text(yaml_content)

        model = load(yaml_file)
        assert model.budget.type == "fraction"
        assert model.budget.value == 0.10
        assert model.gates.max_residual_risk == 250.0
        assert model.gates.max_ece == 0.05
        assert "compensation_field" in model.segments
        assert model.segments["compensation_field"].false_accept_cost == 500.0
        assert model.segments["compensation_field"].review_cost == 4.0
        assert "address_field" in model.segments
        assert model.segments["address_field"].false_accept_cost == 10.0
        assert model.segments["address_field"].review_cost == 2.0
        assert "default" in model.segments
        assert model.segments["default"].false_accept_cost == 50.0
        assert model.segments["default"].review_cost == 3.0

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test load with missing file raises ContractViolation."""
        missing_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(ContractViolation) as exc_info:
            load(missing_file)
        assert "not found" in str(exc_info.value).lower()
        assert str(missing_file) in str(exc_info.value)

    def test_load_malformed_yaml(self, tmp_path: Path) -> None:
        """Test load with malformed YAML raises ContractViolation."""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("invalid: yaml: content: [unclosed")
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "invalid" in str(exc_info.value).lower() or "yaml" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_toplevel_list_not_dict(self, tmp_path: Path) -> None:
        """Test load with top-level list (not dict) raises ContractViolation."""
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2\n- item3")
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "dict" in str(exc_info.value).lower() or "object" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_missing_required_budget(self, tmp_path: Path) -> None:
        """Test load with missing required 'budget' field raises ContractViolation."""
        yaml_file = tmp_path / "no_budget.yaml"
        yaml_file.write_text(
            """\
gates:
  max_ece: 0.05
segments:
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
"""
        )
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "validation" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_missing_required_segments(self, tmp_path: Path) -> None:
        """Test load with missing required 'segments' field raises ContractViolation."""
        yaml_file = tmp_path / "no_segments.yaml"
        yaml_file.write_text(
            """\
budget:
  type: fraction
  value: 0.10
"""
        )
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "validation" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_invalid_budget_type(self, tmp_path: Path) -> None:
        """Test load with invalid budget type raises ContractViolation."""
        yaml_file = tmp_path / "bad_budget_type.yaml"
        yaml_file.write_text(
            """\
budget:
  type: invalid_type
  value: 0.10
segments:
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
"""
        )
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "validation" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_invalid_budget_value(self, tmp_path: Path) -> None:
        """Test load with invalid budget value raises ContractViolation."""
        yaml_file = tmp_path / "bad_budget_value.yaml"
        yaml_file.write_text(
            """\
budget:
  type: fraction
  value: 1.5
segments:
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
"""
        )
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "validation" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_invalid_gate_max_ece(self, tmp_path: Path) -> None:
        """Test load with invalid max_ece raises ContractViolation."""
        yaml_file = tmp_path / "bad_max_ece.yaml"
        yaml_file.write_text(
            """\
budget:
  type: fraction
  value: 0.10
gates:
  max_ece: 1.5
segments:
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
"""
        )
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "validation" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_negative_costs(self, tmp_path: Path) -> None:
        """Test load with negative costs raises ContractViolation."""
        yaml_file = tmp_path / "negative_costs.yaml"
        yaml_file.write_text(
            """\
budget:
  type: fraction
  value: 0.10
segments:
  default:
    false_accept_cost: -50.0
    review_cost: 3.0
"""
        )
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "validation" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_extra_yaml_keys_rejected(self, tmp_path: Path) -> None:
        """Test load with extra/unknown YAML keys raises ContractViolation."""
        yaml_file = tmp_path / "extra_keys.yaml"
        yaml_file.write_text(
            """\
budget:
  type: fraction
  value: 0.10
  extra_field: should_fail
segments:
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
"""
        )
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "validation" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_gates_omitted_ok(self, tmp_path: Path) -> None:
        """Test load with gates omitted is valid."""
        yaml_file = tmp_path / "no_gates.yaml"
        yaml_file.write_text(
            """\
budget:
  type: fraction
  value: 0.10
segments:
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
"""
        )
        model = load(yaml_file)
        assert model.gates.max_residual_risk is None
        assert model.gates.max_ece is None

    def test_load_only_default_segment(self, tmp_path: Path) -> None:
        """Test load with only 'default' segment is valid."""
        yaml_file = tmp_path / "only_default.yaml"
        yaml_file.write_text(
            """\
budget:
  type: money
  value: 1000.0
segments:
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
"""
        )
        model = load(yaml_file)
        assert len(model.segments) == 1
        assert "default" in model.segments
        assert model.budget.type == "money"
        assert model.budget.value == 1000.0
