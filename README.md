# triage

A local-first engine that decides *what humans should review* when review
capacity is limited. Takes AI-scored predictions + a declarative cost model +
a review budget, and outputs a deterministic, versioned routing policy that
maximizes expected risk captured per review.

![Policy vs raw-confidence vs random baselines, expected loss captured across review budgets](docs/headline.png)

At a 10% review budget on the seeded `overconfident` synthetic scenario
(seed 42): the policy captures **44%** of expected loss vs **2%** for raw
confidence and **10%** for random routing (`n_review = 1000` of 10 000
items). Numbers reproduced by `scripts/headline_chart.py --seed 42`.

## Why

AI pipelines produce more outputs than humans can review, and teams default
to routing by raw model confidence — but confidence is not a probability;
overconfident models are the norm, not the exception. Worse, all errors get
treated as equally costly, when a wrong critical field can cost orders of
magnitude more than a typo. The result: nobody has an honest answer to "what
risk are we accepting on the items we *don't* review?" `triage` calibrates
scores into real probabilities, applies declared per-segment costs, and
allocates a fixed review budget to maximize expected risk captured — instead
of guessing.

## How it works

```
scored predictions ──┐
                      ├─▶ calibrate ─▶ calibrator.json ─┐
golden (labeled) ─────┘                                 ├─▶ plan ─▶ policy.json ─▶ backtest ─▶ report
cost model (YAML) ───────────────────────────────────────┘
```

`calibrate` fits isotonic/Platt on a labeled golden set and picks the one
with lower ECE on holdout. `plan` uses the calibrator + declared per-segment
costs to greedily route items to review within budget, maximizing expected
loss captured. Every artifact embeds the sha256 of its inputs, the tool
version, and its config snapshot; `policy.json` carries a `policy_hash`
computed over semantic content only, so identical inputs always produce an
identical policy — determinism is the product, not a nice-to-have.

## Quickstart

```bash
uv sync

# Generate seeded synthetic data + costs.yaml (overconfident scenario)
uv run triage synth --scenario overconfident --seed 42 -o data/

# Fit a calibrator on the golden set
uv run triage calibrate --golden data/golden.parquet --seed 42 -o artifacts/calibrator.json

# Plan a routing policy against a review budget
uv run triage plan --batch data/batch.parquet --calibrator artifacts/calibrator.json \
                    --costs data/costs.yaml --budget 0.10 -o artifacts/policy.json

# Replay the policy on labeled history vs random and raw-confidence baselines.
# The calibrator is hash-verified against the policy — the chain is checked,
# not trusted. Realized residual loss is gated by gates.max_residual_risk
# (exit 2): plan can pass on expected numbers while backtest fails on drifted
# reality. Try --scenario drift above to see the alarm trip.
uv run triage backtest --policy artifacts/policy.json --calibrator artifacts/calibrator.json \
                    --history data/batch.parquet -o artifacts/backtest.json

# Render the self-contained HTML report: an at-a-glance status (On track /
# Needs attention / Gate failed) with headline numbers and CI-style checks
# first, then the evidence (policy vs baselines, coverage-vs-risk curves,
# per-segment plan vs actual, reliability diagram, gates) and the
# collapsed artifact hash chain
uv run triage report --from artifacts -o report.html

# Regenerate the headline chart above (same seed → identical numbers)
uv run python scripts/headline_chart.py --seed 42 -o docs/headline.png
```

Or run the whole chain above in one command (see
[One-command end-to-end run](#one-command-end-to-end-run)):

```bash
uv run triage run --scenario overconfident --seed 42 -o runs/overconfident
```

Exit codes are a public API — CI gates depend on them, and their meanings
never change:

| code | meaning                          |
|------|-----------------------------------|
| 0    | OK                                 |
| 1    | input contract violation           |
| 2    | residual risk above declared gate  |
| 3    | calibration quality (ECE) below declared minimum (point estimate by default, or CI upper bound with `ci_upper`) |
| 4    | internal error                     |

## One-command end-to-end run

`triage run` executes synth → calibrate → plan → backtest → report in one go.
It calls the same code as the individual commands, so the artifacts are
identical to the manual chain. `run --scenario well_calibrated --seed 7`
reproduces the committed `tests/golden/*.json` fixtures hash for hash. The
exit codes are also the same.

```bash
# Synthetic mode: one --seed drives synth, calibrate and the backtest baseline;
# the labeled synthetic batch doubles as the backtest history.
uv run triage run --scenario overconfident --seed 42 -o runs/overconfident
# [1/5] synth ... [5/5] report
# run: synth=ok calibrate=ok plan=ok backtest=ok report=ok -> runs/overconfident

# Own-data mode: your files (--history is optional; without it there is no backtest)
uv run triage run --golden golden.parquet --batch batch.parquet --costs costs.yaml \
                  --history history.parquet -o runs/mine
```

A run directory contains `costs.yaml` (the cost model actually used),
`artifacts/{calibrator,policy,backtest}.json`, `report.html`, and, in
synthetic mode, `data/`. Every command the run executed can be replayed by
hand against these files.

**Overrides instead of hand-edited YAML.** `--budget`, `--max-ece`,
`--ece-gate-mode`, and `--max-residual-risk` patch the cost model. The
patched version is written to `<run>/costs.yaml`, so `plan` records it in
`policy.json` like any other costs file. `--max-budget-overrun` and
`--n-random-draws` pass through to `backtest`. Without overrides, the source
costs file is copied byte for byte. Reproducible gate scenarios, one command
each (seed 42):

| command | exits | why |
|---------|-------|-----|
| `run --scenario overconfident -o runs/a` | 0 | all gates pass |
| `run --scenario overconfident --ece-gate-mode ci_upper -o runs/b` | 3 | ECE upper bound 0.0820 > 0.05 (point 0.0463 passes) |
| `run --scenario drift --max-residual-risk 200000 -o runs/c` | 2 | plan passes (expected ≈ 142706), backtest fails (realized ≈ 336100) |
| `run --scenario overconfident --max-budget-overrun 1.1 -o runs/d` | 5 | 1183 replayed reviews vs 1000 budgeted (ratio 1.183) |
| `run --scenario drift --ece-gate-mode ci_upper --max-residual-risk 200000 -o runs/e` | 3 | several gates fail; the first one in stage order wins |

**Gate failures.** By default `run` keeps going after a failed gate, so every
artifact and the report are written and the report's Gates table shows the
failure. It then exits with the code of the first failed gate in stage order
(calibrate → plan → backtest). `--fail-fast` stops at the first failed gate
instead, exactly like chaining the commands with `set -e`. Contract
violations (exit 1) always stop immediately. The flags and input files are
validated before anything is written, using the same validation functions
as the individual commands. An option is therefore accepted or rejected
identically (same exit code, same message) whether you type the commands or
use `run`. `tests/cli/test_run.py` enforces this and compares `run`
against the manual chain stage by stage. Outputs of a previous run in the same
directory are then removed (only the files `run` writes), so a run never
mixes its artifacts with stale ones.

## Calibration gate: point estimate or CI upper bound

The ECE gate (exit 3) compares one statistic against the declared maximum.
By default that statistic is the holdout ECE point estimate (`ece_holdout`).
Opting into `ci_upper` compares the upper bound of the bootstrap CI
(`ece_ci[1]`, 95% by default) instead. A small holdout can then no longer pass
on a lucky point estimate, because the gate tightens automatically as the
evidence thins. Both modes use the same strict `>`, so a value exactly equal
to the threshold passes.

| mode              | checked value                      | calibrate flag               | cost YAML                   |
|-------------------|------------------------------------|------------------------------|-----------------------------|
| `point` (default) | `ece_holdout` (point estimate)     | (none) or `--ece-gate-mode point` | (none) or `gates.ece_gate_mode: point` |
| `ci_upper`        | `ece_ci[1]` (bootstrap CI upper bound) | `--ece-gate-mode ci_upper` | `gates.ece_gate_mode: ci_upper` |

Reproducible example, the overconfident scenario at seed 42: `ece_holdout =
0.0463`, `ece_ci = (0.0327, 0.0820)`. Against the emitted `max_ece: 0.05`,
the point estimate passes and the upper bound fails:

```bash
uv run triage synth --scenario overconfident --seed 42 -o data/

# point (default): 0.0463 <= 0.05 -> exit 0
uv run triage calibrate --golden data/golden.parquet --seed 42 \
                        --max-ece 0.05 -o gate-demo/calibrator.json; echo "exit=$?"

# ci_upper: 0.0820 > 0.05 -> exit 3 (calibrator.json is still written)
uv run triage calibrate --golden data/golden.parquet --seed 42 \
                        --max-ece 0.05 --ece-gate-mode ci_upper \
                        -o gate-demo/calibrator.json; echo "exit=$?"
# Error: Calibration quality gate failed (ece_gate_mode=ci_upper): ece_ci_upper=0.08199937079701163
#   (upper bound of the 95% bootstrap CI, calibrator.ece_ci[1]) exceeds max_ece=0.05;
#   ece_holdout=0.04631680609317865, ece_ci=(0.0327038775147435, 0.08199937079701163)
# exit=3

# Same gate declared in the cost YAML: plan writes policy.json, then exits 3
uv run python -c "import yaml; c = yaml.safe_load(open('data/costs.yaml')); \
c['gates']['ece_gate_mode'] = 'ci_upper'; yaml.safe_dump(c, open('ci_upper.yaml', 'w'))"
uv run triage plan --batch data/batch.parquet --calibrator gate-demo/calibrator.json \
                   --costs ci_upper.yaml -o gate-demo/policy.json; echo "exit=$?"
# exit=3

# The report's Gates table names the checked value and the threshold:
#   gate    | checked value                                          | threshold | value  | verdict
#   max_ece | calibrator.ece_ci[1] (upper bound of 95% bootstrap CI; | 0.0500    | 0.0820 | FAIL
#           |   ece_gate_mode=ci_upper)                              |           |        |
uv run triage report --from gate-demo -o gate-demo/report.html
```

Notes:

- `ci_upper` requires a threshold (`--max-ece` or `gates.max_ece`). A mode
  without one, an unknown mode, or a threshold outside `(0, 1]` (including
  NaN) is a contract violation (exit 1), rejected before anything is written.
- `ci_upper` is much stricter on small golden sets. All three synthetic
  scenarios fail it at 0.05 (upper bounds 0.074–0.082 at seeds 7 and 42)
  while passing the point gate. That is the point of the mode, but pick a
  threshold your golden-set size can actually support.
- The plan-time mode is persisted in `policy.json` as
  `config_snapshot.gates.ece_gate_mode` only when it is not the default.
  Default-mode policies, including every existing artifact, are byte-identical
  and keep their `policy_hash`, and an absent key means `point`. The
  `calibrate` flag is not stored in `calibrator.json`. Gate options never
  change the calibrator's content hash.

## Design guarantees

- **Determinism**: same inputs → same `policy_hash`, always. Fixed, explicit
  seeds everywhere; stable sorts with `item_id` as the final tie-breaker;
  JSON artifacts written with sorted keys.
- **Self-describing artifacts**: every artifact embeds sha256 hashes of its
  inputs, the tool version, and the full config used to produce it — nothing
  needed to audit a result beyond the artifact itself.
- **Declared costs, not hardcoded logic**: `false_accept_cost` and
  `review_cost` per segment come from a YAML file the business owns; `segment`
  is an opaque string everywhere in the engine.
- **Local-first**: no network calls, no services, no databases, no telemetry.

## Status

v1 complete: everything in the MVP (input contract, calibration with
bootstrap ECE CIs, cost model, greedy optimizer, synthetic generator) plus
the `backtest` command (policy vs. random and raw-confidence baselines
replayed on labeled history, with capacity-sweep coverage-vs-risk curves
embedded in the artifact), a `drift` scenario whose stale policy the
backtester provably exposes (realized residual ≈ 2.4× expected at seed 42),
the self-contained HTML `report`, and a CI workflow
(`.github/workflows/ci.yml`) that runs the exit-code gates for real.

Possible v2 (not started, see GOAL.md): grouped/knapsack routing for
document-level review.

## License

[MIT License](LICENSE).
