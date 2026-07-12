"""Cross-module integration checks the per-module suites cannot cover.

synth.py deliberately does not import contract.py; these tests pin the
guarantee that generated data actually satisfies the input contract and that
the emitted costs.yaml loads into a valid CostModel.
"""

from pathlib import Path

import pytest

from triage import contract, costs, synth


@pytest.mark.parametrize("scenario", ["well_calibrated", "overconfident"])
def test_synth_output_passes_contract(scenario: str) -> None:
    data = synth.generate(scenario, seed=42)
    for key in ("golden", "batch"):
        df = data[key]
        assert contract.validate(df, require_truth=True) is not None  # type: ignore[arg-type]


def test_emitted_costs_yaml_loads_and_files_round_trip(tmp_path: Path) -> None:
    synth.write("overconfident", seed=42, out_dir=tmp_path)
    cost_model = costs.load(tmp_path / "costs.yaml")
    assert cost_model.budget.type == "fraction"
    assert cost_model.budget.value == 0.10
    assert cost_model.gates.max_ece == 0.05
    for name in ("golden.parquet", "batch.parquet"):
        df = contract.read_input(tmp_path / name, require_truth=True)
        assert df.height > 0
