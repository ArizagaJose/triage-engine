"""Cost model YAML and its pydantic validation (docs/interfaces.md §3)."""

import json
from pathlib import Path

import pytest

from tests.support import GOLDEN_DIR
from triage.costs import Budget, CostModel, Gates, SegmentCosts, load
from triage.exceptions import ContractViolation


class TestCostModelCostsFor:
    def test_costs_for_exact_match(self) -> None:
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
    def test_budget_positive_value(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            Budget(type="fraction", value=0)
        with pytest.raises(ValueError, match="must be > 0"):
            Budget(type="fraction", value=-0.1)

    def test_budget_fraction_lte_one(self) -> None:
        with pytest.raises(ValueError, match="must be <= 1.0"):
            Budget(type="fraction", value=1.5)
        # Valid: exactly 1.0 for fraction
        budget = Budget(type="fraction", value=1.0)
        assert budget.value == 1.0

    def test_budget_count_accepts_any_positive(self) -> None:
        budget = Budget(type="count", value=100)
        assert budget.value == 100
        budget = Budget(type="count", value=0.5)
        assert budget.value == 0.5

    def test_budget_money_accepts_any_positive(self) -> None:
        budget = Budget(type="money", value=1000.50)
        assert budget.value == 1000.50


class TestGates:
    def test_gates_max_residual_risk_nonnegative(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            Gates(max_residual_risk=-1)
        # Valid: 0
        gates = Gates(max_residual_risk=0)
        assert gates.max_residual_risk == 0

    def test_gates_max_ece_in_range(self) -> None:
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
        gates = Gates()
        assert gates.max_residual_risk is None
        assert gates.max_ece is None


class TestSegmentCosts:
    def test_segment_costs_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            SegmentCosts(false_accept_cost=-1, review_cost=3.0)
        with pytest.raises(ValueError, match="must be >= 0"):
            SegmentCosts(false_accept_cost=500.0, review_cost=-0.5)

    def test_segment_costs_zero_accepted(self) -> None:
        costs = SegmentCosts(false_accept_cost=0, review_cost=0)
        assert costs.false_accept_cost == 0
        assert costs.review_cost == 0


class TestCostModel:
    def test_cost_model_empty_segments_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CostModel(budget=Budget(type="fraction", value=0.10), segments={})

    def test_cost_model_extra_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="extra_field"):
            CostModel(
                budget=Budget(type="fraction", value=0.10),
                gates=Gates(),
                segments={"default": SegmentCosts(false_accept_cost=50.0, review_cost=3.0)},
                extra_field="should_fail",  # type: ignore
            )


class TestLoadFunction:
    def test_load_happy_path(self, tmp_path: Path) -> None:
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
        missing_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(ContractViolation) as exc_info:
            load(missing_file)
        assert "not found" in str(exc_info.value).lower()
        assert str(missing_file) in str(exc_info.value)

    def test_load_malformed_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("invalid: yaml: content: [unclosed")
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "invalid" in str(exc_info.value).lower() or "yaml" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_toplevel_list_not_dict(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2\n- item3")
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        assert "dict" in str(exc_info.value).lower() or "object" in str(exc_info.value).lower()
        assert str(yaml_file) in str(exc_info.value)

    def test_load_missing_required_budget(self, tmp_path: Path) -> None:
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


_GOLDEN_POLICY = GOLDEN_DIR / "policy.json"

_YAML_HEAD = """\
budget:
  type: fraction
  value: 0.10
gates:
"""
_YAML_SEGMENTS = """\
segments:
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
"""


class TestEceGateMode:
    """gates.ece_gate_mode: "point" (default) | "ci_upper" (docs/interfaces.md §3)."""

    def test_default_is_point(self) -> None:
        assert Gates().ece_gate_mode == "point"
        assert Gates(max_ece=0.05).ece_gate_mode == "point"

    def test_ci_upper_accepted_with_max_ece(self) -> None:
        gates = Gates(max_ece=0.05, ece_gate_mode="ci_upper")
        assert gates.ece_gate_mode == "ci_upper"

    def test_explicit_point_without_max_ece_ok(self) -> None:
        """Explicitly stating the default is harmless (no gate either way)."""
        assert Gates.model_validate({"ece_gate_mode": "point"}).max_ece is None

    def test_ci_upper_without_max_ece_rejected(self) -> None:
        """A stricter mode with no threshold would silently gate nothing."""
        with pytest.raises(ValueError, match="requires gates.max_ece"):
            Gates.model_validate({"ece_gate_mode": "ci_upper"})
        with pytest.raises(ValueError, match="requires gates.max_ece"):
            Gates.model_validate({"ece_gate_mode": "ci_upper", "max_ece": None})

    @pytest.mark.parametrize("mode", ["upper", "CI_UPPER", "Point", "", None, 1, True])
    def test_unknown_mode_rejected(self, mode: object) -> None:
        with pytest.raises(ValueError):
            Gates.model_validate({"max_ece": 0.05, "ece_gate_mode": mode})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.0, -0.01, 1.01])
    def test_max_ece_non_finite_or_out_of_range_rejected(self, bad: float) -> None:
        """NaN used to slip through (every comparison with NaN is False), which
        would have silently disabled the gate in either mode."""
        with pytest.raises(ValueError, match=r"max_ece must be in \(0, 1\]"):
            Gates(max_ece=bad)

    def test_dump_omits_default_mode(self) -> None:
        """Default-mode snapshots are byte-identical to pre-existing ones, so
        policy_hash does not move for anyone who does not opt in."""
        assert Gates(max_ece=0.05).model_dump() == {"max_residual_risk": None, "max_ece": 0.05}
        assert Gates(max_ece=0.05, ece_gate_mode="point").model_dump() == {
            "max_residual_risk": None,
            "max_ece": 0.05,
        }

    def test_dump_includes_non_default_mode_and_round_trips(self) -> None:
        gates = Gates(max_ece=0.05, ece_gate_mode="ci_upper")
        dumped = gates.model_dump()
        assert dumped == {"max_residual_risk": None, "max_ece": 0.05, "ece_gate_mode": "ci_upper"}
        assert Gates.model_validate(dumped) == gates

    def test_cost_model_snapshot_carries_mode(self) -> None:
        model = CostModel(
            budget=Budget(type="fraction", value=0.1),
            gates=Gates(max_ece=0.05, ece_gate_mode="ci_upper"),
            segments={"default": SegmentCosts(false_accept_cost=1.0, review_cost=1.0)},
        )
        snapshot = model.model_dump()
        assert snapshot["gates"]["ece_gate_mode"] == "ci_upper"
        assert CostModel.model_validate(snapshot) == model

    def test_existing_policy_snapshot_is_backward_compatible(self) -> None:
        """The committed golden policy.json predates gate modes: its
        config_snapshot validates to point mode and re-dumps identically
        (so a re-plan yields the same config_snapshot and policy_hash)."""
        snapshot = json.loads(_GOLDEN_POLICY.read_text())["config_snapshot"]
        assert "ece_gate_mode" not in snapshot["gates"]
        model = CostModel.model_validate(snapshot)
        assert model.gates.ece_gate_mode == "point"
        assert model.model_dump() == snapshot

    def test_load_yaml_with_ci_upper(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "ci_upper.yaml"
        yaml_file.write_text(
            _YAML_HEAD + "  max_ece: 0.05\n  ece_gate_mode: ci_upper\n" + _YAML_SEGMENTS
        )
        model = load(yaml_file)
        assert model.gates.max_ece == 0.05
        assert model.gates.ece_gate_mode == "ci_upper"

    @pytest.mark.parametrize(
        ("gates_yaml", "expected"),
        [
            ("  max_ece: 0.05\n  ece_gate_mode: upper\n", "ece_gate_mode"),
            ("  max_ece: 0.05\n  ece_gate_mode: null\n", "ece_gate_mode"),
            ("  ece_gate_mode: ci_upper\n", "requires gates.max_ece"),
            ("  max_ece: .nan\n  ece_gate_mode: ci_upper\n", "max_ece"),
            ("  max_ece: 0.05\n  ece_gate: ci_upper\n", "ece_gate"),  # typo'd key
        ],
    )
    def test_load_yaml_invalid_mode_is_contract_violation(
        self, tmp_path: Path, gates_yaml: str, expected: str
    ) -> None:
        yaml_file = tmp_path / "bad_mode.yaml"
        yaml_file.write_text(_YAML_HEAD + gates_yaml + _YAML_SEGMENTS)
        with pytest.raises(ContractViolation) as exc_info:
            load(yaml_file)
        message = str(exc_info.value)
        assert "validation" in message.lower()
        assert expected in message
        assert str(yaml_file) in message


class TestNonFiniteValuesRejected:
    """NaN passes `v <= 0` / `v < 0` checks (every NaN comparison is False).
    These used to either crash later with exit 4 (budget, costs) or silently
    disable a gate (max_residual_risk)."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("budget_type", ["fraction", "count", "money"])
    def test_budget_value_non_finite(self, budget_type: str, value: float) -> None:
        with pytest.raises(ValueError, match="budget.value must be > 0 and finite"):
            Budget(type=budget_type, value=value)  # type: ignore[arg-type]

    def test_max_residual_risk_nan_rejected_inf_allowed(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            Gates(max_residual_risk=float("nan"))
        assert Gates(max_residual_risk=float("inf")).max_residual_risk == float("inf")

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    @pytest.mark.parametrize("field", ["false_accept_cost", "review_cost"])
    def test_segment_costs_non_finite(self, field: str, value: float) -> None:
        kwargs = {"false_accept_cost": 1.0, "review_cost": 1.0, field: value}
        with pytest.raises(ValueError, match="Costs must be >= 0 and finite"):
            SegmentCosts(**kwargs)

    @pytest.mark.parametrize(
        "snippet",
        [
            "budget:\n  type: count\n  value: .nan\n",
            "budget:\n  type: count\n  value: .inf\n",
            "budget:\n  type: fraction\n  value: 0.1\ngates:\n  max_residual_risk: .nan\n",
        ],
    )
    def test_yaml_non_finite_is_contract_violation(self, tmp_path: Path, snippet: str) -> None:
        yaml_file = tmp_path / "nonfinite.yaml"
        yaml_file.write_text(snippet + _YAML_SEGMENTS)
        with pytest.raises(ContractViolation, match="validation failed"):
            load(yaml_file)
