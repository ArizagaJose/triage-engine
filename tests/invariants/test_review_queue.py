"""Review queue invariants (docs/test-spec-invariants.md section K): the
exported queue is the optimizer's exact selection, reconciles with policy.json
exactly, and leaves every existing artifact and hash untouched.

Everything runs through the real CLI on the seeded chains. Reference numbers:
overconfident/42 plans 1000 reviews while threshold replay admits 1183 (280
seg_high items sit exactly at the threshold); the committed golden chain
well_calibrated/7 plans 1000 while replay admits 1044."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml

from tests.support import (
    GOLDEN_DIR,
    Chain,
    ReferenceAllocation,
    assert_queue_matches_reference,
    assert_queue_reconciles,
    files_under,
    load_json,
    ok,
    read_queue_csv,
    reference_allocation,
    without_created_at,
    write_costs,
)
from triage.artifacts import content_hash, file_sha256
from triage.calibration import Calibrator
from triage.optimizer import REVIEW_QUEUE_COLUMNS, REVIEW_QUEUE_SCHEMA
from triage.review_queue import queue_hash

CHAINS = [("overconfident", 42), ("well_calibrated", 7), ("drift", 42)]


@dataclass(frozen=True)
class Export:
    """`plan --queue-csv --queue-parquet` on a chain's own inputs, in tmp."""

    chain: Chain
    root: Path

    @property
    def policy(self) -> dict[str, Any]:
        return load_json(self.root / "policy.json")

    @property
    def manifest(self) -> dict[str, Any]:
        return load_json(self.root / "review_queue.json")

    @property
    def csv(self) -> pl.DataFrame:
        return read_queue_csv(self.root / "review_queue.csv")

    @property
    def parquet(self) -> pl.DataFrame:
        return pl.read_parquet(self.root / "review_queue.parquet")


def _plan_export(cli: Any, chain: Chain, root: Path, *, costs: Path | None = None) -> Any:
    return cli(
        "plan",
        batch=chain.batch,
        calibrator=chain.calibrator,
        costs=costs or chain.costs,
        o=root / "policy.json",
        queue_csv=root / "review_queue.csv",
        queue_parquet=root / "review_queue.parquet",
    )


@pytest.fixture(scope="module", params=CHAINS, ids=[f"{s}_{seed}" for s, seed in CHAINS])
def exported(request: pytest.FixtureRequest, cli: Any, chain: Any, tmp_path_factory: Any) -> Export:
    scenario, seed = request.param
    base = chain(scenario, seed)
    root = tmp_path_factory.mktemp(f"queue_{scenario}_{seed}")
    ok(_plan_export(cli, base, root))
    return Export(base, root)


def _p_error(chain: Chain, batch: pl.DataFrame) -> np.ndarray:
    calibrator = Calibrator.from_dict(chain.load("calibrator"))
    return calibrator.predict(batch["score"].to_numpy())


def _reference(chain: Chain, costs: Path | None = None) -> ReferenceAllocation:
    batch = pl.read_parquet(chain.batch)
    cost_model = yaml.safe_load((costs or chain.costs).read_text())
    return reference_allocation(batch, _p_error(chain, batch), cost_model)


# --- I.1 The tie case: the queue is the selection, thresholds are not ------------


@pytest.mark.parametrize(
    ("scenario", "seed", "replayed"), [("overconfident", 42, 1183), ("well_calibrated", 7, 1044)]
)
def test_queue_is_cut_inside_the_threshold_tie(
    cli: Any, chain: Any, tmp_path: Path, scenario: str, seed: int, replayed: int
) -> None:
    base = chain(scenario, seed)
    ok(_plan_export(cli, base, tmp_path))
    queue, policy = read_queue_csv(tmp_path / "review_queue.csv"), base.load("policy")
    assert queue.height == policy["totals"]["n_review"] == 1000

    batch = pl.read_parquet(base.batch)
    p_error = _p_error(base, batch)
    ids, segments = batch["item_id"].to_list(), batch["segment"].to_list()
    thresholds = {s: v["p_error_threshold"] for s, v in policy["segments"].items()}

    def threshold(i: int) -> float | None:
        return thresholds.get(segments[i])

    by_threshold = set()
    for i in range(len(ids)):
        t = threshold(i)
        if t is not None and p_error[i] >= t:
            by_threshold.add(ids[i])
    # Threshold replay (what backtest does) admits MORE: the measured count,
    # and the very number backtest.json reports for this chain.
    assert len(by_threshold) == replayed
    assert base.load("backtest")["strategies"]["policy"]["n_review"] == replayed

    queued = set(queue["item_id"].to_list())
    assert queued < by_threshold
    # Only items exactly AT their segment's threshold are left out...
    left_out = [i for i in range(len(ids)) if ids[i] in by_threshold - queued]
    assert all(p_error[i] == threshold(i) for i in left_out)
    # ...and within each tied group the item_id tie-break decides: the queued
    # tied items are the lowest ids of their group.
    for name, t in thresholds.items():
        if t is None:
            continue
        tied = sorted(ids[i] for i in range(len(ids)) if segments[i] == name and p_error[i] == t)
        tied_queued = [x for x in tied if x in queued]
        assert tied_queued == tied[: len(tied_queued)]
        assert 0 < len(tied_queued) < len(tied)


# --- I.2 Independent recomputation from raw inputs ---------------------------------


def test_queue_equals_independent_recomputation(exported: Export) -> None:
    ref = _reference(exported.chain)
    assert_queue_matches_reference(exported.csv, ref)
    assert exported.csv["review_priority"].to_list() == list(range(1, len(ref.selected) + 1))


def test_residual_loss_reconciles_from_the_batch(exported: Export) -> None:
    """The queue alone cannot give residual loss; queue + batch can, exactly."""
    ref, policy = _reference(exported.chain), exported.policy
    queued = set(exported.csv["item_id"].to_list())
    residuals = []
    for name in sorted(policy["segments"]):
        residual = math.fsum(
            float(ref.expected_loss[i])
            for i in range(len(ref.item_ids))
            if ref.segments[i] == name and ref.item_ids[i] not in queued
        )
        assert residual == policy["segments"][name]["expected_residual_loss"], name
        residuals.append(residual)
    assert sum(residuals) == policy["totals"]["expected_residual_loss"]


# --- I.3 Exact reconciliation: queue <-> policy <-> manifest <-> files ------------


def test_queue_reconciles_exactly_with_policy(exported: Export) -> None:
    assert_queue_reconciles(exported.csv, exported.policy)
    assert_queue_reconciles(exported.parquet, exported.policy)
    assert exported.csv.equals(exported.parquet)


def test_queue_trace_columns_point_at_real_artifacts(exported: Export) -> None:
    policy = exported.policy
    assert set(exported.csv["policy_hash"].to_list()) == {content_hash(policy)}
    assert set(exported.csv["batch_sha256"].to_list()) == {file_sha256(exported.chain.batch)}
    # And the chain's own policy (planned without export) is the same policy.
    assert policy["policy_hash"] == exported.chain.load("policy")["policy_hash"]


def test_manifest_reconciles(exported: Export) -> None:
    manifest, policy, queue = exported.manifest, exported.policy, exported.csv
    assert manifest["policy_hash"] == policy["policy_hash"]
    assert manifest["input_sha256s"] == policy["input_sha256s"]
    assert manifest["totals"] == {
        key: policy["totals"][key] for key in ("expected_captured_loss", "n_review", "review_spend")
    }
    assert manifest["n_rows"] == queue.height
    assert manifest["columns"] == list(REVIEW_QUEUE_COLUMNS)
    assert manifest["queue_hash"] == queue_hash(queue) == queue_hash(exported.parquet)
    for fmt, entry in manifest["files"].items():
        assert file_sha256(exported.root / entry["path"]) == entry["sha256"], fmt
    assert manifest["gates"] == {"status": "pass", "failed": []}


# --- I.4 Every budget type, end to end ------------------------------------------------

# (budget, segment-cost override, expected rows; None = derive from the reference)
BUDGET_CASES: list[tuple[str, dict[str, Any], dict[str, Any] | None, int | None]] = [
    ("fraction", {"type": "fraction", "value": 0.10}, None, 1000),
    ("fraction-floors-to-zero", {"type": "fraction", "value": 0.00001}, None, 0),
    ("count", {"type": "count", "value": 250}, None, 250),
    # Capped at n_batch, then stopped by net_benefit > 0.
    ("count-above-batch", {"type": "count", "value": 20000}, None, None),
    # seg_high reviews cost 4.0: 250 fit exactly, the 251st would not.
    ("money-exact-fit", {"type": "money", "value": 1000.0}, None, 250),
    ("money-remainder", {"type": "money", "value": 1003.5}, None, 250),
    ("money-below-first-cost", {"type": "money", "value": 3.0}, None, 0),
    ("no-positive-net-benefit", {"type": "fraction", "value": 1.0}, {"review_cost": 1e6}, 0),
]


@pytest.mark.parametrize(
    ("budget", "segment_override", "expected_rows"),
    [case[1:] for case in BUDGET_CASES],
    ids=[case[0] for case in BUDGET_CASES],
)
def test_budget_types_end_to_end(
    cli: Any,
    chain: Any,
    tmp_path: Path,
    budget: dict[str, Any],
    segment_override: dict[str, Any] | None,
    expected_rows: int | None,
) -> None:
    base = chain("overconfident")
    cost_model = yaml.safe_load(base.costs.read_text())
    cost_model["budget"] = budget
    if segment_override:
        for seg in cost_model["segments"].values():
            seg.update(segment_override)
    costs = tmp_path / "costs.yaml"
    costs.write_text(yaml.safe_dump(cost_model, sort_keys=True))

    ok(_plan_export(cli, base, tmp_path / "out", costs=costs))
    out = tmp_path / "out"
    queue, policy = read_queue_csv(out / "review_queue.csv"), load_json(out / "policy.json")
    manifest = load_json(out / "review_queue.json")
    ref = _reference(base, costs)

    assert_queue_reconciles(queue, policy)
    assert_queue_matches_reference(queue, ref)
    assert queue.equals(pl.read_parquet(out / "review_queue.parquet"))
    assert manifest["n_rows"] == queue.height
    assert manifest["policy_hash"] == policy["policy_hash"]
    if expected_rows is not None:
        assert queue.height == expected_rows
    if queue.height == 0:
        assert (out / "review_queue.csv").read_text().splitlines() == [
            ",".join(REVIEW_QUEUE_COLUMNS)
        ]
        assert pl.read_parquet(out / "review_queue.parquet").schema == REVIEW_QUEUE_SCHEMA

    # The stopping rule, from the queue's own numbers:
    n = queue.height
    nxt = ref.ranking[n] if n < len(ref.ranking) else None
    next_worth_it = nxt is not None and ref.net_benefit[nxt] > 0
    if budget["type"] == "money":
        spend = math.fsum(queue["review_cost"].to_list())
        assert spend <= budget["value"]
        if next_worth_it:
            assert nxt is not None
            assert spend + float(ref.review_cost[nxt]) > budget["value"]
    else:
        cap = (
            math.floor(budget["value"] * len(ref.item_ids))
            if budget["type"] == "fraction"
            else min(int(budget["value"]), len(ref.item_ids))
        )
        assert n <= cap
        assert n == cap or not next_worth_it


# --- I.5 Non-regression: export changes no existing artifact or hash ----------------


def test_golden_policy_reproduced_with_export_on(cli: Any, chain: Any, tmp_path: Path) -> None:
    ok(_plan_export(cli, chain("well_calibrated", 7), tmp_path))
    assert without_created_at(load_json(tmp_path / "policy.json")) == without_created_at(
        load_json(GOLDEN_DIR / "policy.json")
    )


def test_golden_fixtures_reproduced_by_run_with_export_on(cli: Any, tmp_path: Path) -> None:
    out = tmp_path / "run"
    argv = ["run", "--scenario", "well_calibrated", "--seed", "7", "-o", str(out)]
    ok(cli.argv([*argv, "--queue-format", "csv", "--queue-format", "parquet"]))
    for name in ("policy.json", "backtest.json"):
        assert without_created_at(load_json(out / "artifacts" / name)) == without_created_at(
            load_json(GOLDEN_DIR / name)
        ), name
    assert {"artifacts/review_queue.csv", "artifacts/review_queue.parquet"} <= files_under(out)


# --- I.6 Determinism ---------------------------------------------------------------------


def test_export_is_byte_identical_across_processes(
    cli: Any, cli_process: Any, chain: Any, tmp_path: Path
) -> None:
    base = chain("overconfident")
    a, b = tmp_path / "a", tmp_path / "b"
    ok(_plan_export(cli, base, a))
    ok(_plan_export(cli_process, base, b))
    for name in ("review_queue.csv", "review_queue.parquet"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert without_created_at(load_json(a / "review_queue.json")) == without_created_at(
        load_json(b / "review_queue.json")
    )


def test_batch_row_order_does_not_change_the_queue(cli: Any, chain: Any, tmp_path: Path) -> None:
    """Same rows, different file order: same selection, same priorities. Only
    the trace columns differ, because the batch FILE (and so its sha256 and
    the policy's input hash) differs -- which is the point of tracing."""
    base = chain("overconfident")
    shuffled = tmp_path / "shuffled.parquet"
    batch = pl.read_parquet(base.batch)
    batch[np.random.default_rng(0).permutation(batch.height).tolist()].write_parquet(shuffled)
    ok(_plan_export(cli, base, tmp_path / "orig"))
    ok(
        cli(
            "plan",
            batch=shuffled,
            calibrator=base.calibrator,
            costs=base.costs,
            o=tmp_path / "shuf" / "policy.json",
            queue_csv=tmp_path / "shuf" / "review_queue.csv",
        )
    )
    trace = ["policy_hash", "batch_sha256"]
    orig = read_queue_csv(tmp_path / "orig" / "review_queue.csv")
    shuf = read_queue_csv(tmp_path / "shuf" / "review_queue.csv")
    assert orig.drop(trace).equals(shuf.drop(trace))
    assert set(shuf["batch_sha256"].to_list()) == {file_sha256(shuffled)}
    p_orig, p_shuf = (load_json(tmp_path / d / "policy.json") for d in ("orig", "shuf"))
    for key in ("segments", "totals", "budget", "config_snapshot"):
        assert p_orig[key] == p_shuf[key], key


# --- I.7 The manifest's gate record agrees with plan's exit code --------------------


@pytest.mark.parametrize(
    ("gates", "code", "failed"),
    [
        # F.7's drift story: plan passes on expected numbers (~142706 <= 200000)
        # -- the queue is honestly "pass" at plan time; backtest is what fails.
        ({"max_residual_risk": 200000}, 0, []),
        ({"max_residual_risk": 100000}, 2, ["max_residual_risk"]),
        (
            {"ece_gate_mode": "ci_upper", "max_residual_risk": 100000},
            3,
            ["max_ece", "max_residual_risk"],
        ),
    ],
    ids=["drift-plan-passes", "residual-fails", "both-fail"],
)
def test_gate_record_matches_exit_code_on_drift(
    cli: Any, chain: Any, tmp_path: Path, gates: dict[str, Any], code: int, failed: list[str]
) -> None:
    base = chain("drift")
    costs = write_costs(base.costs, tmp_path / "costs.yaml", **gates)
    result = _plan_export(cli, base, tmp_path / "out", costs=costs)
    assert result.returncode == code, result.stderr
    manifest = load_json(tmp_path / "out" / "review_queue.json")
    assert manifest["gates"] == {"status": "failed" if failed else "pass", "failed": failed}
    assert_queue_reconciles(
        read_queue_csv(tmp_path / "out" / "review_queue.csv"),
        load_json(tmp_path / "out" / "policy.json"),
    )
