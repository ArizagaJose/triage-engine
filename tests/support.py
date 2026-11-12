"""Shared test helpers. Fixtures that use them live in tests/conftest.py."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = TESTS_DIR / "golden"
TRIAGE_BIN = shutil.which("triage") or str(TESTS_DIR.parent / ".venv/bin/triage")


# --- CLI invocation ---------------------------------------------------------------


@dataclass(frozen=True)
class CliResult:
    """Same shape as subprocess.CompletedProcess for the fields tests use."""

    returncode: int
    stdout: str
    stderr: str


def args(command: str, **options: Any) -> list[str]:
    """argv for `triage <command>`: `seed=42` -> `--seed 42`, `max_ece=0.05` ->
    `--max-ece 0.05`, `o=out` -> `-o out`, `fail_fast=True` -> `--fail-fast`.
    None and False are omitted. Values are passed through str()."""
    argv = [command]
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = f"-{key}" if len(key) == 1 else f"--{key.replace('_', '-')}"
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    return argv


def run_in_process(argv: list[str]) -> CliResult:
    """Run `triage <argv>` through cli.app_main, the console-script entry
    point, inside this process. Same code path and exit-code mapping as the
    installed script, without paying the interpreter + import cost."""
    from triage import cli

    out, err = io.StringIO(), io.StringIO()
    saved_argv = sys.argv
    sys.argv = ["triage", *argv]
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                cli.app_main()  # returns normally on success
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = saved_argv
    return CliResult(code, out.getvalue(), err.getvalue())


def run_subprocess(argv: list[str]) -> CliResult:
    """Run the installed `triage` console script in a fresh process."""
    done = subprocess.run([TRIAGE_BIN, *argv], capture_output=True, text=True, timeout=300)
    return CliResult(done.returncode, done.stdout, done.stderr)


def ok(result: CliResult) -> CliResult:
    """Assert exit 0 and return the result."""
    assert result.returncode == 0, result.stderr
    return result


# --- artifacts --------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def without_created_at(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "created_at"}


def json_text_without_created_at(text: str) -> str:
    """Drop the `"created_at"` line of a canonical JSON artifact."""
    return "\n".join(line for line in text.splitlines() if '"created_at"' not in line)


def html_without_created_at(text: str) -> str:
    """Drop the single `id="created-at"` line of a report, asserting there is
    exactly one."""
    lines = text.splitlines()
    matching = [line for line in lines if 'id="created-at"' in line]
    assert len(matching) == 1, f'expected exactly one id="created-at" line, found {len(matching)}'
    return "\n".join(line for line in lines if 'id="created-at"' not in line)


def files_under(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def gates_section(html: str) -> str:
    return html.split("<h2>Gates</h2>", 1)[1].split("<h2>Backtest</h2>", 1)[0]


def gates_row_cells(html: str, gate: str) -> list[str]:
    """The <td> texts of the report's gates-table row whose first cell is `gate`."""
    row = gates_section(html).split(f"<td>{gate}</td>", 1)[1].split("</tr>", 1)[0]
    return [gate] + [
        re.sub(r"<[^>]+>", "", chunk.split("</td>", 1)[0]).strip()
        for chunk in row.split("<td>")[1:]
    ]


def write_costs(source: Path, dest: Path, *, drop_gates: bool = False, **gates: Any) -> Path:
    """Copy a cost YAML to `dest` with `gates` entries set (None values are
    written as null) or, with drop_gates, without a gates block."""
    import yaml

    cfg = yaml.safe_load(source.read_text())
    if drop_gates:
        cfg.pop("gates", None)
    else:
        cfg.setdefault("gates", {}).update(gates)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return dest


# --- the documented chain, cached per session -------------------------------------


@dataclass(frozen=True)
class Chain:
    """Outputs of `synth -> calibrate -> plan -> backtest` for one scenario and
    seed (the seed drives every stage; emitted costs.yaml; 200 random draws).
    Layout: data/ (synth output) and artifacts/ (usable as `report --from`).

    Shared across the session: treat as read-only. `ChainCache` verifies at
    session end that no test modified it; use `copy_to()` to get a mutable
    copy."""

    root: Path
    scenario: str
    seed: int

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def golden(self) -> Path:
        return self.data_dir / "golden.parquet"

    @property
    def batch(self) -> Path:
        return self.data_dir / "batch.parquet"

    @property
    def costs(self) -> Path:
        return self.data_dir / "costs.yaml"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def calibrator(self) -> Path:
        return self.artifacts_dir / "calibrator.json"

    @property
    def policy(self) -> Path:
        return self.artifacts_dir / "policy.json"

    @property
    def backtest(self) -> Path:
        return self.artifacts_dir / "backtest.json"

    def load(self, name: str) -> dict[str, Any]:
        """A fresh parse of artifacts/<name>.json or data/manifest.json."""
        path = (
            self.data_dir / "manifest.json"
            if name == "manifest"
            else self.root / (f"artifacts/{name}.json")
        )
        return load_json(path)

    def copy_to(self, dest: Path) -> Chain:
        shutil.copytree(self.root, dest)
        return Chain(dest, self.scenario, self.seed)

    def build(self) -> None:
        seed = self.seed
        ok(run_in_process(args("synth", scenario=self.scenario, seed=seed, o=self.data_dir)))
        ok(run_in_process(args("calibrate", golden=self.golden, seed=seed, o=self.calibrator)))
        ok(
            run_in_process(
                args(
                    "plan",
                    batch=self.batch,
                    calibrator=self.calibrator,
                    costs=self.costs,
                    o=self.policy,
                )
            )
        )
        ok(
            run_in_process(
                args(
                    "backtest",
                    policy=self.policy,
                    calibrator=self.calibrator,
                    history=self.batch,
                    seed=seed,
                    n_random_draws=200,
                    o=self.backtest,
                )
            )
        )


def tree_digest(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files_under(root)
    }


class ChainCache:
    """Builds each (scenario, seed) chain once per session, on first use."""

    def __init__(self, base: Path) -> None:
        self._base = base
        self._chains: dict[tuple[str, int], Chain] = {}
        self._digests: dict[tuple[str, int], dict[str, str]] = {}

    def __call__(self, scenario: str, seed: int = 42) -> Chain:
        key = (scenario, seed)
        if key not in self._chains:
            chain = Chain(self._base / f"{scenario}_{seed}", scenario, seed)
            chain.build()
            self._chains[key] = chain
            self._digests[key] = tree_digest(chain.root)
        return self._chains[key]

    def modified(self) -> list[str]:
        return [
            f"{scenario}/{seed}"
            for (scenario, seed), chain in self._chains.items()
            if tree_digest(chain.root) != self._digests[(scenario, seed)]
        ]


# --- unit-test builders -------------------------------------------------------------


class IdentityCalibrator:
    """PredictsErrors stub: p_error = score, so hand-computed expectations stay exact."""

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=np.float64)


_FRAME_SCHEMA = {
    "item_id": pl.String,
    "segment": pl.String,
    "score": pl.Float64,
    "predicted_value": pl.String,
    "true_value": pl.String,
}


def batch_frame(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """Unlabeled frame from (item_id, segment, score); predicted_value="v"."""
    return pl.DataFrame(
        {
            "item_id": [r[0] for r in rows],
            "segment": [r[1] for r in rows],
            "score": [r[2] for r in rows],
            "predicted_value": ["v" for _ in rows],
        },
        schema={k: v for k, v in _FRAME_SCHEMA.items() if k != "true_value"},
    )


def history_frame(rows: list[tuple[str, str, float, str, str]]) -> pl.DataFrame:
    """Labeled frame from (item_id, segment, score, predicted_value, true_value)."""
    columns = list(_FRAME_SCHEMA)
    return pl.DataFrame(
        {col: [r[i] for r in rows] for i, col in enumerate(columns)}, schema=_FRAME_SCHEMA
    )
