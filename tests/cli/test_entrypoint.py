"""The installed `triage` console script, in real processes: entry-point wiring
and the exit-code mapping as seen by a shell or CI (exit codes are a public
API, CLAUDE.md). Everything else runs in-process via the `cli` fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support import write_costs


def test_help_lists_every_command(cli_process: Any) -> None:
    result = cli_process.argv(["--help"])
    assert result.returncode == 0
    for command in ("synth", "calibrate", "plan", "backtest", "report", "run"):
        assert command in result.stdout


def test_usage_error_exits_1_not_clicks_2(cli_process: Any) -> None:
    """Click's own usage-error code (2) would collide with the residual gate."""
    assert cli_process.argv(["synth", "--not-a-real-flag", "x"]).returncode == 1


@pytest.mark.parametrize("expected", [0, 1, 2, 3, 5])
def test_exit_codes_reach_the_shell(
    cli_process: Any, chain: Any, tmp_path: Path, expected: int
) -> None:
    base = chain("overconfident", 42)
    out = tmp_path / "out.json"
    if expected == 0:
        result = cli_process("report", **{"from": base.artifacts_dir}, o=tmp_path / "r.html")
    elif expected == 1:
        result = cli_process("synth", scenario="nope", o=tmp_path / "data")
    elif expected == 2:
        costs = write_costs(base.costs, tmp_path / "costs.yaml", max_residual_risk=0.01)
        result = cli_process(
            "plan", batch=base.batch, calibrator=base.calibrator, costs=costs, o=out
        )
    elif expected == 3:
        result = cli_process("calibrate", golden=base.golden, max_ece=0.0001, o=out)
    else:
        result = cli_process(
            "backtest",
            policy=base.policy,
            calibrator=base.calibrator,
            history=base.batch,
            max_budget_overrun=1.0,
            o=out,
        )
    assert result.returncode == expected, result.stderr
