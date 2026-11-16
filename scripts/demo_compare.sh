#!/usr/bin/env bash
# Policy comparison demo (docs/interfaces.md §14; README "Comparing policies").
# Usage: scripts/demo_compare.sh [seed]
#
# One current policy (overconfident scenario, 10% budget) against three
# candidates, each on the same labeled history:
#   1. budget  - same calibrator and costs, 15% budget  -> a trade-off that pays off
#   2. costs   - seg_mid false_accept_cost 400 instead of 50:
#                rejected without a common basis (exit 1), then judged under the
#                current costs (significantly worse; --fail-on-regression -> exit 6)
#                and under the candidate's own costs (slightly better)
#   3. same    - the current policy itself                -> no change
# Each comparison gets comparison.json + comparison.html under demo/compare/.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${1:-42}"
OUT="demo/compare"
DATA="$OUT/data"
rm -rf "$OUT"

step() { printf '\n== %s\n' "$*"; }

step "synthetic data + current policy (10% budget)"
uv run triage synth --scenario overconfident --seed "$SEED" -o "$DATA"
uv run triage calibrate --golden "$DATA/golden.parquet" --seed "$SEED" -o "$OUT/current/calibrator.json"
uv run triage plan --batch "$DATA/batch.parquet" --calibrator "$OUT/current/calibrator.json" \
  --costs "$DATA/costs.yaml" -o "$OUT/current/policy.json"

step "candidate 1: 15% budget"
uv run triage plan --batch "$DATA/batch.parquet" --calibrator "$OUT/current/calibrator.json" \
  --costs "$DATA/costs.yaml" --budget 0.15 -o "$OUT/budget/policy.json"

step "candidate 2: seg_mid false_accept_cost 50 -> 400"
uv run python - "$DATA/costs.yaml" "$OUT/costs_seg_mid_400.yaml" <<'PY'
import sys, yaml
costs = yaml.safe_load(open(sys.argv[1]))
costs["segments"]["seg_mid"]["false_accept_cost"] = 400.0
yaml.safe_dump(costs, open(sys.argv[2], "w"), sort_keys=True)
PY
uv run triage plan --batch "$DATA/batch.parquet" --calibrator "$OUT/current/calibrator.json" \
  --costs "$OUT/costs_seg_mid_400.yaml" -o "$OUT/costs/policy.json"

# compare NAME CANDIDATE_POLICY [extra compare flags...]: compare + render.
compare() {
  local name="$1" candidate="$2"; shift 2
  uv run triage compare \
    --current-policy "$OUT/current/policy.json" --current-calibrator "$OUT/current/calibrator.json" \
    --candidate-policy "$candidate" --candidate-calibrator "$OUT/current/calibrator.json" \
    --history "$DATA/batch.parquet" --seed "$SEED" -o "$OUT/$name" "$@"
}
render() { uv run triage report --compare "$OUT/$1" -o "$OUT/$1/comparison.html"; }

step "1. budget 10% -> 15%"
compare 1_budget "$OUT/budget/policy.json" --fail-on-regression
render 1_budget

step "2a. cost change, no common basis: must be rejected (exit 1)"
if compare 2a_costs_rejected "$OUT/costs/policy.json"; then
  echo "unexpected: incompatible costs were accepted" >&2; exit 1
else
  code=$?; [ "$code" -eq 1 ] || { echo "expected exit 1, got $code" >&2; exit 1; }
  echo "-> exit 1, as expected"
fi

step "2b. cost change, judged by the CURRENT costs, with the regression gate (exit 6)"
if compare 2b_costs_current_basis "$OUT/costs/policy.json" \
     --eval-costs "$DATA/costs.yaml" --fail-on-regression; then
  echo "unexpected: the regression gate passed" >&2; exit 1
else
  code=$?; [ "$code" -eq 6 ] || { echo "expected exit 6, got $code" >&2; exit 1; }
  echo "-> exit 6, as expected (comparison.json was still written)"
fi
render 2b_costs_current_basis

step "2c. cost change, judged by the CANDIDATE's own costs"
compare 2c_costs_candidate_basis "$OUT/costs/policy.json" --eval-costs "$OUT/costs_seg_mid_400.yaml"
render 2c_costs_candidate_basis

step "3. identical policy"
compare 3_identical "$OUT/current/policy.json"
render 3_identical

step "open"
for name in 1_budget 2b_costs_current_basis 2c_costs_candidate_basis 3_identical; do
  echo "  $(pwd)/$OUT/$name/comparison.html"
done
