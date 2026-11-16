#!/usr/bin/env bash
# Test the opt-in money-budget `fill` strategy (docs/interfaces.md §5) end to end.
#
#   1. static checks (ruff, mypy)
#   2. the fill-related tests (unit, CLI, invariants: test-spec section I)
#   3. the README "Money budgets" demo through the real CLI: prefix vs fill,
#      plan -> backtest -> report, with the documented numbers verified
#   4. the documented rejections (exit 1, nothing written)
#
# Idempotent: it owns demo/fill-strategy/ (gitignored), rebuilds it from
# scratch on every run, and prints no timings, timestamps or absolute paths,
# so N runs print the same output. Exits non-zero on the first failure.
#
# Usage: bash scripts/test_fill_strategy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="demo/fill-strategy"
rm -rf "$OUT"
mkdir -p "$OUT"

step() { printf '\n== %s\n' "$1"; }

step "1/4 static checks"
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy src

step "2/4 fill-related tests"
# -p no:cacheprovider: no .pytest_cache writes; the sed drops pytest's timing.
uv run pytest -q -p no:cacheprovider \
  tests/unit/test_costs.py tests/unit/test_optimizer.py tests/test_report_summary.py \
  tests/cli/test_commands.py tests/cli/test_run.py tests/invariants \
  -k "BudgetStrategy or Fill or fill or budget_strategy or budget_label or combined \
or prefix_strategy or accounted" \
  | sed -E 's/ in [0-9.]+s( \([0-9:]+\))?//'

step "3/4 README demo: overconfident, seed 42, money budget 1000, review costs 45/3/1"
uv run triage synth --scenario overconfident --seed 42 -o "$OUT/data"
uv run triage calibrate --golden "$OUT/data/golden.parquet" --seed 42 \
  -o "$OUT/calibrator.json"
uv run python - "$OUT" <<'PY'
import sys

import yaml

out = sys.argv[1]
c = yaml.safe_load(open(f"{out}/data/costs.yaml"))
c["budget"] = {"type": "money", "value": 1000}
for seg, cost in [("seg_high", 45.0), ("seg_mid", 3.0), ("seg_low", 1.0), ("default", 3.0)]:
    c["segments"][seg]["review_cost"] = cost
yaml.safe_dump(c, open(f"{out}/money.yaml", "w"), sort_keys=True)
PY
for strategy in prefix fill; do
  dir="$OUT/$strategy"
  mkdir -p "$dir"
  cp "$OUT/calibrator.json" "$dir/calibrator.json"
  uv run triage plan --batch "$OUT/data/batch.parquet" --calibrator "$dir/calibrator.json" \
    --costs "$OUT/money.yaml" --budget-strategy "$strategy" -o "$dir/policy.json"
  uv run triage backtest --policy "$dir/policy.json" --calibrator "$dir/calibrator.json" \
    --history "$OUT/data/batch.parquet" --seed 42 -o "$dir/backtest.json"
  uv run triage report --from "$dir" -o "$dir/report.html"
done

uv run python - "$OUT" <<'PY'
import json
import sys

out = sys.argv[1]
rows = {}
for s in ("prefix", "fill"):
    policy = json.load(open(f"{out}/{s}/policy.json"))
    replay = json.load(open(f"{out}/{s}/backtest.json"))["strategies"]["policy"]
    html = open(f"{out}/{s}/report.html").read()
    t = policy["totals"]
    rows[s] = {
        "n_review": t["n_review"],
        "spend": t["review_spend"],
        "captured": t["expected_captured_loss"],
        "net": t["expected_captured_loss"] - t["review_spend"],
        "candidate": policy.get("allocation", {}).get("selected_candidate", "-"),
        "realized": replay["captured_loss"],
        "replayed": replay["n_review"],
        "html": html,
    }

print(f"{'strategy':<8} {'n_review':>8} {'spend':>7} {'exp.captured':>12} {'exp.net':>9} "
      f"{'realized':>9} {'replayed':>8}  candidate")
for s, r in rows.items():
    print(f"{s:<8} {r['n_review']:>8} {r['spend']:>7.1f} {r['captured']:>12.1f} "
          f"{r['net']:>9.1f} {r['realized']:>9.1f} {r['replayed']:>8}  {r['candidate']}")

# The numbers README.md and docs/risk-register.md R5 quote (test-spec I.2).
checks = [
    ("prefix plans 22 reviews", rows["prefix"]["n_review"] == 22),
    ("fill plans 630 reviews", rows["fill"]["n_review"] == 630),
    ("fill spends exactly the budget", rows["fill"]["spend"] == 1000.0),
    ("fill never worse than prefix (expected net benefit)",
     rows["fill"]["net"] >= rows["prefix"]["net"]),
    ("per-cost ranking selected", rows["fill"]["candidate"] == "net_benefit_per_cost"),
    ("prefix policy has no allocation block", rows["prefix"]["candidate"] == "-"),
    ("realized captured loss 7000 -> 7220",
     (rows["prefix"]["realized"], rows["fill"]["realized"]) == (7000.0, 7220.0)),
    ("report labels the fill strategy",
     "(fill strategy, net-benefit-per-cost ranking)" in rows["fill"]["html"]),
    ("prefix report label unchanged", "(fill strategy" not in rows["prefix"]["html"]),
]
failed = [name for name, passed in checks if not passed]
for name, passed in checks:
    print(f"  [{'ok' if passed else 'FAIL'}] {name}")
sys.exit(1 if failed else 0)
PY

step "4/4 documented rejections (expect exit 1, nothing written)"
expect_exit_1() {
  local label="$1" out="$2"
  shift 2
  local code=0
  "$@" >/dev/null 2>"$OUT/stderr.txt" || code=$?
  local message
  message="$(grep -m1 -oE "requires budget.type 'money'|Input should be 'prefix' or 'fill'" \
    "$OUT/stderr.txt" || true)"
  if [[ "$code" -eq 1 && ! -e "$out" && -n "$message" ]]; then
    echo "  [ok] $label: exit 1 ($message)"
  else
    echo "  [FAIL] $label: exit $code"
    cat "$OUT/stderr.txt"
    exit 1
  fi
}
expect_exit_1 "plan fill on a fraction budget" "$OUT/rejected.json" \
  uv run triage plan --batch "$OUT/data/batch.parquet" --calibrator "$OUT/calibrator.json" \
  --costs "$OUT/data/costs.yaml" --budget-strategy fill -o "$OUT/rejected.json"
expect_exit_1 "plan with an unknown strategy" "$OUT/rejected.json" \
  uv run triage plan --batch "$OUT/data/batch.parquet" --calibrator "$OUT/calibrator.json" \
  --costs "$OUT/money.yaml" --budget-strategy knapsack -o "$OUT/rejected.json"
expect_exit_1 "run --scenario with fill" "$OUT/rejected-run" \
  uv run triage run --scenario overconfident --seed 42 --budget-strategy fill \
  -o "$OUT/rejected-run"
rm -f "$OUT/stderr.txt"

printf '\nAll fill-strategy checks passed. Reports: %s/{prefix,fill}/report.html\n' "$OUT"
