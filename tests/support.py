"""Shared test helpers. Fixtures that use them live in tests/conftest.py."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
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


# Review costs of the README "Money budgets" example: synth's 4/3/2 widened so
# that cost differences between segments matter (test-spec section I).
MONEY_DEMO_REVIEW_COSTS = {"seg_high": 45.0, "seg_mid": 3.0, "seg_low": 1.0, "default": 3.0}


def write_money_costs(
    source: Path,
    dest: Path,
    *,
    value: float = 1000.0,
    review_costs: dict[str, float] | None = None,
    strategy: str | None = None,
) -> Path:
    """Copy a cost YAML to `dest` with a money budget of `value`, the given
    review costs (default: MONEY_DEMO_REVIEW_COSTS) and, if given,
    `budget.strategy`."""
    import yaml

    cfg = yaml.safe_load(source.read_text())
    cfg["budget"] = {"type": "money", "value": value}
    if strategy is not None:
        cfg["budget"]["strategy"] = strategy
    for name, cost in (review_costs or MONEY_DEMO_REVIEW_COSTS).items():
        cfg["segments"][name]["review_cost"] = cost
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


@dataclass(frozen=True)
class MoneyDemo:
    """overconfident/42 batch + calibrator planned with a money budget of 1000
    and MONEY_DEMO_REVIEW_COSTS, once per strategy, each backtested on the
    same batch (seed 42). `dir(strategy)` is usable as `report --from`.
    Shared across the session: treat as read-only."""

    root: Path
    batch: Path
    costs: Path

    def dir(self, strategy: str) -> Path:
        return self.root / strategy

    def load(self, strategy: str, name: str) -> dict[str, Any]:
        return load_json(self.dir(strategy) / f"{name}.json")

    @classmethod
    def build(cls, chain: Chain, root: Path) -> MoneyDemo:
        root.mkdir(parents=True)
        demo = cls(root, chain.batch, write_money_costs(chain.costs, root / "costs.yaml"))
        for strategy in ("prefix", "fill"):
            out = demo.dir(strategy)
            out.mkdir()
            shutil.copyfile(chain.calibrator, out / "calibrator.json")
            plan_options: dict[str, Any] = {
                "batch": demo.batch,
                "calibrator": out / "calibrator.json",
                "costs": demo.costs,
                "o": out / "policy.json",
            }
            if strategy == "fill":
                plan_options["budget_strategy"] = "fill"
            ok(run_in_process(args("plan", **plan_options)))
            ok(
                run_in_process(
                    args(
                        "backtest",
                        policy=out / "policy.json",
                        calibrator=out / "calibrator.json",
                        history=demo.batch,
                        seed=42,
                        o=out / "backtest.json",
                    )
                )
            )
        return demo


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


# --- review queue: exact reconciliation and an independent reference -----------------


def assert_queue_reconciles(queue: pl.DataFrame, policy: dict[str, Any]) -> None:
    """A review queue agrees with its policy EXACTLY (==, never approx).

    Every policy number the queue can determine is recomputed from the queue's
    own rows: counts, spend, captured loss (per segment and total, with the
    optimizer's fsum / sequential-sum tie-out convention), thresholds, hashes,
    the per-row arithmetic, and the priority ordering."""
    from triage.optimizer import REVIEW_QUEUE_COLUMNS, REVIEW_QUEUE_SCHEMA

    totals, segments = policy["totals"], policy["segments"]
    assert queue.columns == list(REVIEW_QUEUE_COLUMNS)
    assert queue.schema == REVIEW_QUEUE_SCHEMA
    assert queue.height == totals["n_review"]
    assert queue["review_priority"].to_list() == list(range(1, queue.height + 1))
    assert queue["item_id"].n_unique() == queue.height
    assert set(queue["policy_hash"].to_list()) <= {policy["policy_hash"]}
    assert set(queue["batch_sha256"].to_list()) <= {policy["input_sha256s"]["batch"]}

    rows = queue.to_dicts()
    for row in rows:
        assert row["expected_loss"] == row["p_error"] * row["false_accept_cost"], row
        assert row["net_benefit"] == row["expected_loss"] - row["review_cost"], row
        assert row["net_benefit"] > 0, row  # never routed otherwise (§5.6)
    for a, b in zip(rows, rows[1:]):
        # net_benefit DESC, item_id ASC: the optimizer's ranking (§5.4).
        assert a["net_benefit"] > b["net_benefit"] or (
            a["net_benefit"] == b["net_benefit"] and a["item_id"] < b["item_id"]
        ), (a, b)

    assert math.fsum(queue["review_cost"].to_list()) == totals["review_spend"]
    assert set(queue["segment"].to_list()) <= set(segments)
    captured_by_segment = []
    for name in sorted(segments):
        seg_rows = [r for r in rows if r["segment"] == name]
        stats = segments[name]
        assert len(seg_rows) == stats["n_review"], name
        captured = math.fsum(r["expected_loss"] for r in seg_rows)
        assert captured == stats["expected_captured_loss"], name
        threshold = min((r["p_error"] for r in seg_rows), default=None)
        assert threshold == stats["p_error_threshold"], name
        captured_by_segment.append(captured)
    assert sum(captured_by_segment) == totals["expected_captured_loss"]


@dataclass(frozen=True)
class ReferenceAllocation:
    """The pinned §5 allocation, recomputed in-test from raw inputs."""

    item_ids: list[str]
    segments: list[str]
    p_error: np.ndarray
    expected_loss: np.ndarray
    review_cost: np.ndarray
    net_benefit: np.ndarray
    ranking: list[int]  # row indices, net_benefit DESC then item_id ASC
    selected: list[int]  # row indices in priority order


def reference_allocation(
    batch: pl.DataFrame, p_error: np.ndarray, cost_model: dict[str, Any]
) -> ReferenceAllocation:
    """Independent re-implementation of docs/interfaces.md §5 (one tuple-key
    sort instead of the optimizer's two stable sorts; its own capacity walk).
    `cost_model` is the parsed YAML / config_snapshot dict."""
    item_ids = batch["item_id"].to_list()
    segments = batch["segment"].to_list()
    costs = cost_model["segments"]

    def seg_costs(name: str) -> dict[str, float]:
        return costs[name] if name in costs else costs["default"]

    fac = np.array([seg_costs(s)["false_accept_cost"] for s in segments], dtype=np.float64)
    rc = np.array([seg_costs(s)["review_cost"] for s in segments], dtype=np.float64)
    p = np.asarray(p_error, dtype=np.float64)
    expected_loss = p * fac
    net = expected_loss - rc
    n = len(item_ids)
    ranking = sorted(range(n), key=lambda i: (-float(net[i]), item_ids[i]))

    budget = cost_model["budget"]
    if budget["type"] == "fraction":
        capacity = math.floor(budget["value"] * n)
    elif budget["type"] == "count":
        capacity = min(int(budget["value"]), n)
    else:
        capacity, spent = 0, 0.0
        for i in ranking:
            if spent + float(rc[i]) > budget["value"]:
                break
            spent += float(rc[i])
            capacity += 1
    selected: list[int] = []
    for i in ranking[:capacity]:
        if net[i] <= 0:
            break
        selected.append(i)
    return ReferenceAllocation(item_ids, segments, p, expected_loss, rc, net, ranking, selected)


def assert_queue_matches_reference(queue: pl.DataFrame, ref: ReferenceAllocation) -> None:
    """The queue IS the reference selection, row for row, float for float."""
    sel = ref.selected
    assert queue["item_id"].to_list() == [ref.item_ids[i] for i in sel]
    assert queue["segment"].to_list() == [ref.segments[i] for i in sel]
    assert queue["p_error"].to_list() == [float(ref.p_error[i]) for i in sel]
    assert queue["expected_loss"].to_list() == [float(ref.expected_loss[i]) for i in sel]
    assert queue["review_cost"].to_list() == [float(ref.review_cost[i]) for i in sel]
    assert queue["net_benefit"].to_list() == [float(ref.net_benefit[i]) for i in sel]


def read_queue_csv(path: Path) -> pl.DataFrame:
    """Read an exported queue CSV with the pinned dtypes (item ids stay strings)."""
    from triage.optimizer import REVIEW_QUEUE_SCHEMA

    return pl.read_csv(path, schema_overrides=dict(REVIEW_QUEUE_SCHEMA))


# --- policy-comparison builders (compare.py / compare_report.py tests) ---------------
#
# A tiny labeled history with hand-checkable answers. Two segments: A (miss 100,
# review 2) and B (miss 10, review 1). The "confident" calibrator maps
# p_error = 1 - score (the usual direction); "inverted" maps p_error = score.
# Thresholds below sit strictly between item p_errors, so replay is exact.
#
#   item  seg  score  p_error  error  loss
#   a1    A    0.1    0.9      yes    100
#   a2    A    0.3    0.7      yes    100
#   a3    A    0.5    0.5      no       0
#   a4    A    0.8    0.2      no       0
#   b1    B    0.2    0.8      yes     10
#   b2    B    0.4    0.6      no       0
#   b3    B    0.6    0.4      yes     10
#   b4    B    0.9    0.1      no       0        total true loss: 220

COMPARE_COSTS: dict[str, tuple[float, float]] = {"A": (100.0, 2.0), "B": (10.0, 1.0)}

_COMPARE_ROWS = [
    ("a1", "A", 0.1, True),
    ("a2", "A", 0.3, True),
    ("a3", "A", 0.5, False),
    ("a4", "A", 0.8, False),
    ("b1", "B", 0.2, True),
    ("b2", "B", 0.4, False),
    ("b3", "B", 0.6, True),
    ("b4", "B", 0.9, False),
]


def compare_history(copies: int = 1) -> pl.DataFrame:
    """The table above, repeated `copies` times (every total scales by it)."""
    rows = [
        (f"{item}-{k:04d}", seg, score, "x" if error else "v", "v")
        for k in range(copies)
        for item, seg, score, error in _COMPARE_ROWS
    ]
    return history_frame(rows)


def calibrator_dict(*, inverted: bool = False, seed: int = 0) -> dict[str, Any]:
    """A real calibrator artifact (isotonic, two knots). `seed` only changes
    the content hash, to make two otherwise identical calibrators distinct."""
    from triage.calibration import CalibrationConfig, Calibrator

    y = [0.0, 1.0] if inverted else [1.0, 0.0]
    return Calibrator(
        method="isotonic",
        params={"x_thresholds": [0.0, 1.0], "y_thresholds": y},
        ece_holdout=0.01,
        ece_ci=(0.005, 0.02),
        ece_raw_holdout=0.1,
        n_golden=100,
        seed=seed,
        input_sha256="golden-sha",
        tool_version="0.1.0",
        config=CalibrationConfig(),
        reliability_bins={"before": [], "after": []},
    ).to_dict()


def cost_model_dict(
    costs: dict[str, tuple[float, float]] | None = None,
    *,
    budget: tuple[str, float] = ("fraction", 0.5),
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from triage.costs import CostModel

    return CostModel.model_validate(
        {
            "budget": {"type": budget[0], "value": budget[1]},
            "gates": gates or {},
            "segments": {
                name: {"false_accept_cost": fa, "review_cost": rc}
                for name, (fa, rc) in (costs or COMPARE_COSTS).items()
            },
        }
    ).model_dump()


def policy_dict(
    calibrator: dict[str, Any],
    thresholds: dict[str, float | None],
    *,
    costs: dict[str, tuple[float, float]] | None = None,
    budget: tuple[str, float] = ("fraction", 0.5),
    gates: dict[str, Any] | None = None,
    planned_n_review: int = 3,
) -> dict[str, Any]:
    """A self-consistent policy.json dict (valid policy_hash, linked to
    `calibrator` by content hash). Only thresholds, costs and budget matter
    to a comparison; the expected-loss fields are placeholders."""
    from triage.artifacts import content_hash

    policy: dict[str, Any] = {
        "budget": {"type": budget[0], "value": float(budget[1])},
        "config_snapshot": cost_model_dict(costs, budget=budget, gates=gates),
        "input_sha256s": {
            "batch": "batch-sha",
            "calibrator": content_hash(calibrator),
            "costs": "costs-sha",
        },
        "segments": {
            name: {
                "expected_captured_loss": 0.0,
                "expected_residual_loss": 0.0,
                "n_review": 0,
                "n_total": 4,
                "p_error_threshold": threshold,
            }
            for name, threshold in thresholds.items()
        },
        "tool_version": "0.1.0",
        "totals": {
            "expected_captured_loss": 0.0,
            "expected_residual_loss": 0.0,
            "n_review": planned_n_review,
            "n_total": 8,
            "review_spend": 0.0,
        },
    }
    policy["policy_hash"] = content_hash(policy)
    return policy
