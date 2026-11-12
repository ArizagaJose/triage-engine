"""Suite-wide fixtures. Layout: unit/ (module functions, no CLI), cli/ (the
commands and `run`, in-process; cli/test_entrypoint.py is the only real
console-script file), invariants/ (docs/test-spec-invariants.md)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.support import ChainCache, CliResult, args, run_in_process, run_subprocess


class Cli:
    """`cli("plan", batch=p, o=out)` or `cli.argv(["--help"])`."""

    def __init__(self, runner: Callable[[list[str]], CliResult]) -> None:
        self._runner = runner

    def __call__(self, command: str, **options: Any) -> CliResult:
        return self._runner(args(command, **options))

    def argv(self, argv: list[str]) -> CliResult:
        return self._runner(argv)


@pytest.fixture(scope="session")
def cli() -> Cli:
    """The CLI, in-process (app_main: same parsing and exit-code mapping as the
    console script)."""
    return Cli(run_in_process)


@pytest.fixture(scope="session")
def cli_process() -> Cli:
    """The installed console script in a fresh process. Only for what needs a
    real process: entry-point wiring and cross-process determinism."""
    return Cli(run_subprocess)


@pytest.fixture(scope="session")
def chain(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ChainCache]:
    """`chain("overconfident", 42)`: the documented chain, built once per
    session and shared read-only."""
    cache = ChainCache(tmp_path_factory.mktemp("chains"))
    yield cache
    modified = cache.modified()
    assert not modified, f"tests modified shared chains {modified}; use Chain.copy_to()"


@pytest.fixture(scope="session")
def calibrator_seed43(chain: ChainCache, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """overconfident/42's golden set calibrated with seed 43: a valid but
    different calibrator (seed-sensitivity and hash-chain tamper checks)."""
    out = tmp_path_factory.mktemp("seed43") / "calibrator.json"
    result = run_in_process(args("calibrate", golden=chain("overconfident").golden, seed=43, o=out))
    assert result.returncode == 0, result.stderr
    return out
