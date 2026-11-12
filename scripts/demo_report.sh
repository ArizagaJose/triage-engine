#!/usr/bin/env bash
# Full pipeline -> report.html. Usage: scripts/demo_report.sh [overconfident|drift|well_calibrated]
set -euo pipefail
cd "$(dirname "$0")/.."

SCENARIO="${1:-overconfident}"
OUT="demo/$SCENARIO"

uv run triage synth --scenario "$SCENARIO" --seed 42 -o "$OUT/data"
uv run triage calibrate --golden "$OUT/data/golden.parquet" --seed 42 -o "$OUT/calibrator.json"
uv run triage plan --batch "$OUT/data/batch.parquet" --calibrator "$OUT/calibrator.json" \
  --costs "$OUT/data/costs.yaml" -o "$OUT/policy.json"
uv run triage backtest --policy "$OUT/policy.json" --calibrator "$OUT/calibrator.json" \
  --history "$OUT/data/batch.parquet" --seed 42 -o "$OUT/backtest.json"
uv run triage report --from "$OUT" -o "$OUT/report.html"

echo "Open: $(pwd)/$OUT/report.html"
