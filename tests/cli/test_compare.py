"""`triage compare` and `triage report --compare` through the real CLI
(docs/interfaces.md §14). The current policy is the shared overconfident/42
chain; candidates are planned here from the same data:

- b15:      budget 0.15 instead of 0.10 (same calibrator, same costs)
- alt:      seg_mid false_accept_cost 400 instead of 50 (same calibrator)
- nolow:    planned on the batch WITHOUT seg_low (never sees that segment)
- inverted: planned with a calibrator whose p_error INCREASES with score
- seed43:   planned with the seed-43 calibrator (a near-identical policy)

Numbers pinned below were measured at seed 42 and match the analysis in
README "Comparing policies".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from tests.support import load_json, ok, without_created_at
from triage.artifacts import content_hash

N_BOOT = 200


@dataclass(frozen=True)
class Variant:
    policy: Path
    calibrator: Path


@pytest.fixture(scope="module")
def base(chain: Any) -> Any:
    return chain("overconfident", 42)


@pytest.fixture(scope="module")
def variants(
    base: Any, cli: Any, calibrator_seed43: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Variant]:
    root = tmp_path_factory.mktemp("compare_variants")
    out: dict[str, Variant] = {"current": Variant(base.policy, base.calibrator)}

    def plan(name: str, *, calibrator: Path, batch: Path | None = None, **kw: Any) -> None:
        policy = root / name / "policy.json"
        ok(
            cli(
                "plan",
                batch=batch or base.batch,
                calibrator=calibrator,
                costs=kw.pop("costs", base.costs),
                o=policy,
                **kw,
            )
        )
        out[name] = Variant(policy, calibrator)

    plan("b15", calibrator=base.calibrator, budget=0.15)

    alt_costs = yaml.safe_load(base.costs.read_text())
    alt_costs["segments"]["seg_mid"]["false_accept_cost"] = 400.0
    (root / "costs_alt.yaml").write_text(yaml.safe_dump(alt_costs))
    plan("alt", calibrator=base.calibrator, costs=root / "costs_alt.yaml")

    no_low = root / "batch_nolow.parquet"
    pl.read_parquet(base.batch).filter(pl.col("segment") != "seg_low").write_parquet(no_low)
    plan("nolow", calibrator=base.calibrator, batch=no_low)

    inverted = base.load("calibrator")
    inverted["method"] = "isotonic"
    inverted["params"] = {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0]}
    inverted_path = root / "inverted_calibrator.json"
    inverted_path.write_text(json.dumps(inverted))
    plan("inverted", calibrator=inverted_path)

    plan("seed43", calibrator=calibrator_seed43)
    return out


def compare(
    cli: Any,
    variants: dict[str, Variant],
    current: str,
    candidate: str,
    out: Path,
    history: Path,
    **kw: Any,
) -> Any:
    kw.setdefault("n_bootstrap", N_BOOT)
    return cli(
        "compare",
        current_policy=variants[current].policy,
        current_calibrator=variants[current].calibrator,
        candidate_policy=variants[candidate].policy,
        candidate_calibrator=variants[candidate].calibrator,
        history=history,
        o=out,
        **kw,
    )


def comparison(out: Path) -> dict[str, Any]:
    return load_json(out / "comparison.json")


# --- outcomes --------------------------------------------------------------------


def test_identical_policies(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    result = ok(compare(cli, variants, "current", "current", tmp_path, base.batch))
    c = comparison(tmp_path)
    assert c["verdict"]["label"] == "no_change"
    assert c["verdict"]["ci"]["label"] == "no_change"
    assert all(c["delta"][k] == 0 for k in ("captured_loss", "n_review", "review_spend"))
    assert c["overlap"]["current_only"]["n_review"] == 0
    assert "verdict=no_change (ci: no_change)" in result.stdout
    assert str(tmp_path / "comparison.json") in result.stdout


def test_bigger_budget_is_a_cheaper_trade_off(
    cli: Any, variants: Any, base: Any, tmp_path: Path
) -> None:
    result = ok(
        compare(cli, variants, "current", "b15", tmp_path, base.batch, fail_on_regression=True)
    )
    c = comparison(tmp_path)
    assert c["verdict"]["label"] == "trade_off"
    assert c["verdict"]["ci"]["label"] == "trade_off"
    assert c["delta"]["n_review"] == 649  # 1,183 -> 1,832 replayed reviews
    assert c["delta"]["captured_loss"] == 42500.0
    assert c["delta"]["review_spend"] == 2596.0
    assert c["delta"]["total_cost"] == -39904.0
    assert c["verdict"]["ci"]["total_cost"] == "better"
    assert c["overlap"]["current_only"]["n_review"] == 0  # a wider budget only adds items
    assert "d_reviews=+649" in result.stdout


def test_smaller_budget_fails_the_regression_gate_after_writing(
    cli: Any, variants: Any, base: Any, tmp_path: Path
) -> None:
    result = compare(cli, variants, "b15", "current", tmp_path, base.batch, fail_on_regression=True)
    assert result.returncode == 6
    assert "significantly worse" in result.stderr
    assert "total cost (residual + review spend) rises" in result.stderr
    assert comparison(tmp_path)["verdict"]["ci"]["total_cost"] == "worse"  # written first


def test_inverted_calibrator_is_a_known_regression(
    cli: Any, variants: Any, base: Any, tmp_path: Path
) -> None:
    result = compare(
        cli, variants, "current", "inverted", tmp_path, base.batch, fail_on_regression=True
    )
    assert result.returncode == 6
    c = comparison(tmp_path)
    assert c["delta"]["captured_loss"] < 0
    assert c["verdict"]["ci"]["directions"]["captured_loss"] == "worse"
    assert c["verdict"]["ci"]["total_cost"] == "worse"


def test_correct_calibrator_is_a_known_improvement_over_inverted(
    cli: Any, variants: Any, base: Any, tmp_path: Path
) -> None:
    ok(compare(cli, variants, "inverted", "current", tmp_path, base.batch, fail_on_regression=True))
    c = comparison(tmp_path)
    assert c["verdict"]["ci"]["directions"]["captured_loss"] == "better"
    assert c["verdict"]["ci"]["total_cost"] == "better"


def test_seed43_calibrator_difference_is_noise(
    cli: Any, variants: Any, base: Any, tmp_path: Path
) -> None:
    ok(compare(cli, variants, "current", "seed43", tmp_path, base.batch, n_bootstrap=1000))
    c = comparison(tmp_path)
    assert c["input_sha256s"]["current_calibrator"] != c["input_sha256s"]["candidate_calibrator"]
    assert c["verdict"]["label"] == "improvement"  # 3 fewer reviews, same loss...
    assert c["verdict"]["ci"]["label"] == "inconclusive"  # ...which the CI calls noise


def test_segment_missing_from_candidate(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    ok(compare(cli, variants, "current", "nolow", tmp_path, base.batch))
    c = comparison(tmp_path)
    segs = c["policies"]["candidate"]["outcome"]["segments"]
    assert segs["seg_low"]["status"] == "not_in_plan"
    assert segs["seg_low"]["n_review"] == 0
    assert c["policies"]["current"]["outcome"]["segments"]["seg_low"]["status"] == "no_threshold"
    assert set(c["delta"]["segments"]) == {"seg_high", "seg_low", "seg_mid"}


# --- cost basis ------------------------------------------------------------------


def test_incompatible_costs_rejected_and_nothing_written(
    cli: Any, variants: Any, base: Any, tmp_path: Path
) -> None:
    result = compare(cli, variants, "current", "alt", tmp_path, base.batch)
    assert result.returncode == 1
    assert "seg_mid: current false_accept_cost=50.0" in result.stderr
    assert "candidate false_accept_cost=400.0" in result.stderr
    assert "--eval-costs" in result.stderr
    assert not (tmp_path / "comparison.json").exists()


def test_eval_costs_flip_the_verdict(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    """The analysis example: judged by the current costs the candidate is far
    worse; judged by its own costs it is slightly better."""
    alt_costs = variants["alt"].policy.parent.parent / "costs_alt.yaml"
    by_current, by_alt = tmp_path / "current_costs", tmp_path / "alt_costs"
    ok(compare(cli, variants, "current", "alt", by_current, base.batch, eval_costs=base.costs))
    ok(compare(cli, variants, "current", "alt", by_alt, base.batch, eval_costs=alt_costs))
    cur_basis, alt_basis = comparison(by_current), comparison(by_alt)

    assert cur_basis["delta"]["captured_loss"] == -86300.0
    assert cur_basis["delta"]["total_cost"] == 84991.0
    assert cur_basis["verdict"]["ci"]["total_cost"] == "worse"
    assert cur_basis["basis"]["planned_under_basis"] == {"current": True, "candidate": False}

    assert alt_basis["delta"]["captured_loss"] == 2600.0
    assert alt_basis["delta"]["total_cost"] == -3909.0
    assert alt_basis["basis"]["planned_under_basis"] == {"current": False, "candidate": True}

    for c in (cur_basis, alt_basis):
        assert c["basis"]["source"] == "eval_costs"
        assert c["basis"]["ignored_eval_costs_fields"] == ["budget", "gates"]
        # Same review sets, whatever the prices.
        assert c["overlap"]["both"]["n_review"] == 312
    assert cur_basis["input_sha256s"]["eval_costs"] != alt_basis["input_sha256s"]["eval_costs"]


# --- artifact relationships ---------------------------------------------------------


def test_swapped_calibrators(
    cli: Any, variants: Any, base: Any, tmp_path: Path, calibrator_seed43: Path
) -> None:
    result = cli(
        "compare",
        current_policy=variants["current"].policy,
        current_calibrator=calibrator_seed43,
        candidate_policy=variants["seed43"].policy,
        candidate_calibrator=base.calibrator,
        history=base.batch,
        o=tmp_path,
    )
    assert result.returncode == 1
    assert "Calibrators are swapped" in result.stderr


def test_wrong_calibrator(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    result = cli(
        "compare",
        current_policy=variants["current"].policy,
        current_calibrator=base.calibrator,
        candidate_policy=variants["inverted"].policy,
        candidate_calibrator=base.calibrator,
        history=base.batch,
        o=tmp_path,
    )
    assert result.returncode == 1
    assert "candidate calibrator mismatch" in result.stderr
    assert "it is the current policy's calibrator" in result.stderr


def test_tampered_policy(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    tampered = load_json(variants["b15"].policy)
    tampered["segments"]["seg_high"]["p_error_threshold"] = 0.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    result = cli(
        "compare",
        current_policy=variants["current"].policy,
        current_calibrator=base.calibrator,
        candidate_policy=path,
        candidate_calibrator=base.calibrator,
        history=base.batch,
        o=tmp_path / "out",
    )
    assert result.returncode == 1
    assert "candidate policy: self-hash mismatch" in result.stderr


@pytest.mark.parametrize("target", ["current_policy", "candidate_calibrator"])
def test_unreadable_artifacts(
    cli: Any, variants: Any, base: Any, tmp_path: Path, target: str
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    options = {
        "current_policy": variants["current"].policy,
        "current_calibrator": base.calibrator,
        "candidate_policy": variants["current"].policy,
        "candidate_calibrator": base.calibrator,
        target: broken,
    }
    result = cli("compare", **options, history=base.batch, o=tmp_path / "out")
    assert result.returncode == 1
    assert "is not valid JSON" in result.stderr


def test_outcome_matches_backtest(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    """Under its own costs, a policy's comparison outcome IS its backtest."""
    ok(compare(cli, variants, "current", "b15", tmp_path, base.batch, n_bootstrap=0))
    out = comparison(tmp_path)["policies"]["current"]["outcome"]
    bt = base.load("backtest")["strategies"]["policy"]
    for key in ("captured_loss", "residual_loss", "n_review", "review_spend", "capture_rate"):
        assert out[key] == bt[key], key
    for seg, stats in bt["segments"].items():
        for key, value in stats.items():
            assert out["segments"][seg][key] == value, (seg, key)


# --- options ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "options,message",
    [
        ({"n_bootstrap": -1}, "n_bootstrap must be an integer >= 0"),
        ({"ci_level": 1.5}, r"ci_level must be in (0, 1)"),
        ({"seed": -1}, "seed must be a non-negative integer"),
        ({"n_bootstrap": 0, "fail_on_regression": True}, "needs --n-bootstrap > 0"),
    ],
)
def test_invalid_options_rejected_before_writing(
    cli: Any, variants: Any, base: Any, tmp_path: Path, options: dict[str, Any], message: str
) -> None:
    result = compare(cli, variants, "current", "b15", tmp_path / "out", base.batch, **options)
    assert result.returncode == 1
    assert message in result.stderr
    assert not (tmp_path / "out").exists()


def test_output_must_be_a_directory(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    a_file = tmp_path / "file"
    a_file.write_text("")
    result = compare(cli, variants, "current", "b15", a_file, base.batch)
    assert result.returncode == 1
    assert "is not a directory" in result.stderr


def test_no_bootstrap(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    result = ok(compare(cli, variants, "current", "b15", tmp_path, base.batch, n_bootstrap=0))
    c = comparison(tmp_path)
    assert c["uncertainty"] is None and c["verdict"]["ci"] is None
    assert "(ci:" not in result.stdout


def test_deterministic_in_process(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    ok(compare(cli, variants, "current", "b15", tmp_path / "a", base.batch))
    ok(compare(cli, variants, "current", "b15", tmp_path / "b", base.batch))
    a, b = comparison(tmp_path / "a"), comparison(tmp_path / "b")
    assert without_created_at(a) == without_created_at(b)
    assert a["comparison_hash"] == content_hash(a)


# --- report --compare --------------------------------------------------------------


def test_report_compare_renders(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    ok(compare(cli, variants, "current", "b15", tmp_path, base.batch))
    result = ok(cli("report", compare=tmp_path, o=tmp_path / "comparison.html"))
    assert "included=comparison" in result.stdout
    html = (tmp_path / "comparison.html").read_text()
    assert 'data-verdict="trade_off"' in html
    assert 'id="fig-compare"' in html


@pytest.mark.parametrize("which", ["both", "neither"])
def test_report_needs_exactly_one_source(cli: Any, base: Any, tmp_path: Path, which: str) -> None:
    sources: dict[str, Any] = (
        {"from": base.artifacts_dir, "compare": tmp_path} if which == "both" else {}
    )
    result = cli("report", **sources, o=tmp_path / "r.html")
    assert result.returncode == 1
    assert "exactly one of --from DIR" in result.stderr
    assert not (tmp_path / "r.html").exists()


def test_report_compare_missing_file(cli: Any, tmp_path: Path) -> None:
    result = cli("report", compare=tmp_path, o=tmp_path / "r.html")
    assert result.returncode == 1
    assert "Cannot read comparison file" in result.stderr


def test_report_compare_edited_file(cli: Any, variants: Any, base: Any, tmp_path: Path) -> None:
    ok(compare(cli, variants, "current", "b15", tmp_path, base.batch))
    c = comparison(tmp_path)
    c["delta"]["n_review"] = 0
    (tmp_path / "comparison.json").write_text(json.dumps(c))
    result = cli("report", compare=tmp_path, o=tmp_path / "r.html")
    assert result.returncode == 1
    assert "edited after `triage compare` wrote it" in result.stderr


def test_single_policy_report_unchanged_path(cli: Any, base: Any, tmp_path: Path) -> None:
    result = ok(cli("report", **{"from": base.artifacts_dir}, o=tmp_path / "r.html"))
    assert "included=calibrator+policy+backtest" in result.stdout


def test_fill_queue_policy_can_be_compared_and_reported(
    cli: Any, money_demo: Any, tmp_path: Path
) -> None:
    from tests.support import assert_queue_reconciles, read_queue_csv

    fill = money_demo.dir("fill")
    prefix = money_demo.dir("prefix")
    out = tmp_path / "fill"
    ok(
        cli(
            "plan",
            batch=money_demo.batch,
            calibrator=fill / "calibrator.json",
            costs=money_demo.costs,
            budget_strategy="fill",
            o=out / "policy.json",
            queue_csv=out / "review_queue.csv",
        )
    )
    policy = load_json(out / "policy.json")
    assert policy["policy_hash"] == money_demo.load("fill", "policy")["policy_hash"]
    assert_queue_reconciles(read_queue_csv(out / "review_queue.csv"), policy)
    comparison_dir = tmp_path / "comparison"
    ok(
        cli(
            "compare",
            current_policy=prefix / "policy.json",
            current_calibrator=prefix / "calibrator.json",
            candidate_policy=out / "policy.json",
            candidate_calibrator=fill / "calibrator.json",
            history=money_demo.batch,
            n_bootstrap=0,
            o=comparison_dir,
        )
    )
    html = tmp_path / "comparison.html"
    ok(cli("report", compare=comparison_dir, o=html))
    assert "fill strategy" in html.read_text()
    single = tmp_path / "fill.html"
    ok(cli("report", **{"from": fill}, o=single))
    assert "fill strategy, net-benefit-per-cost ranking" in single.read_text()
