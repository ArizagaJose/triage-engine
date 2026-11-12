# GOAL.md — Human Review Triage Engine

## One-liner

Given AI-scored predictions, a business cost model, and a limited human review
budget, produce a deterministic routing policy that maximizes expected risk
captured per review — and prove it with a backtest.

## Problem

Every AI pipeline produces more outputs than humans can review. Teams optimize
tokens and compute, but almost nobody optimizes the scarcest resource: reviewer
hours. The common failure modes:

- Routing by raw model confidence, which is not a probability (overconfident
  models are the norm).
- Treating all errors as equally costly, when a wrong critical field costs
  orders of magnitude more than a typo.
- Having no answer to "what risk are we accepting on the items we DON'T review?"

## Positioning

Third leg of a consistent engineering philosophy:

> AI for understanding, determinism for control, **human judgment for deciding**.

- **AI Harness** decides *whether* AI-generated data is safe to promote.
- **This project** decides *what to review when you can't review everything*.

Related academic framing: selective prediction / learning to defer. The gap:
almost nobody operationalizes it as a standalone, auditable tool.

The central CLI command is `plan` — a deliberate echo of `terraform plan`:
decide and inspect before you apply.

## Input contract

One Parquet/CSV of scored predictions:

| column            | type    | required | notes                                   |
|-------------------|---------|----------|-----------------------------------------|
| `item_id`         | str     | yes      | unique; final tie-breaker in all sorts  |
| `segment`         | str     | yes      | opaque grouping key (e.g. field type)   |
| `score`           | float   | yes      | model confidence, [0, 1], uncalibrated  |
| `predicted_value` | str     | yes      | opaque; the engine never interprets it  |
| `true_value`      | str     | optional | required for `calibrate` and `backtest` |

Strict validation, fail-fast, exit code 1 on violation. No silent coercion.

## Modules

### 1. Calibration (`calibrate`)

- Split golden set into calibration/holdout.
- Fit isotonic regression and Platt scaling; select by ECE on holdout.
- **Bootstrap confidence intervals** on ECE (small golden sets are the real-world
  case; a point estimate without a CI is self-deception).
- Output: versioned calibrator artifact (JSON) with input hash + reliability
  data for the report (before/after diagrams).
- ECE gate: if calibration quality is below the declared minimum, exit code 3.
  The gated statistic is the holdout ECE point estimate by default; the
  opt-in `ci_upper` mode gates on the bootstrap CI's upper bound instead
  (`--ece-gate-mode` / `gates.ece_gate_mode`, docs/interfaces.md §4).

### 2. Cost model (YAML, pydantic-validated)

```yaml
budget:
  type: fraction        # fraction | count | money
  value: 0.10
gates:
  max_residual_risk: 250.0   # exit code 2 if exceeded
  max_ece: 0.05              # exit code 3 if exceeded
  ece_gate_mode: point       # optional: point (ece_holdout, default) | ci_upper (ece_ci[1])
segments:
  compensation_field:
    false_accept_cost: 500.0
    review_cost: 4.0
  address_field:
    false_accept_cost: 10.0
    review_cost: 2.0
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
```

The business declares costs; the engine executes them. No costs hardcoded.

### 3. Optimizer (`plan`)

- Expected loss per item = calibrated P(error) × segment `false_accept_cost`.
- Sort by expected loss descending (net of review cost), route to review until
  budget is exhausted. With independent items, greedy is optimal — nothing
  fancier is needed or wanted.
- Output: `policy.json` containing per-segment thresholds, expected
  coverage/risk numbers, sha256 of inputs, tool version, config snapshot, and a
  `policy_hash` over semantic content. Same inputs → same policy, always.

### 4. Backtester (`backtest`)

Replay a policy against labeled history and compare against two baselines:

1. **Random routing** at the same budget.
2. **Raw-confidence routing** (uncalibrated score, no costs) at the same budget.

Metrics: captured loss, residual loss, review spend, **efficiency (captured
loss per review)**. The headline README chart comes from here: policy vs both
baselines across budgets.

The backtester takes the calibrator as an explicit input (the policy's
thresholds are on calibrated `p_error`, so replay needs the mapping) and
verifies it against the hash recorded in the policy — the chain is checked,
not trusted. Realized residual loss is gated by `gates.max_residual_risk`
(exit 2): `plan` can pass on expected numbers while `backtest` fails on
realized ones, which is exactly the drift alarm.

Threshold replay can review more items than planned when ties sit exactly at
a threshold (expected and correct, see docs/interfaces.md §9.4). Because
capacity is the resource being allocated, `backtest` accepts an optional
`--max-budget-overrun FACTOR` (float ≥ 1.0): if the replayed review
count/spend exceeds FACTOR × the budget's capacity on this history, exit
code 5. The replay itself is never capped — the gate only refuses to bless
the result. See docs/risk-register.md (R1).

### 5. Report (`report`)

Static HTML (jinja2 + plotly) for operations people and the team's ML tech
lead. The answer comes first: an at-a-glance section with a status
(On track / Needs attention / Gate failed; "Plan only" without a backtest),
a plain-language summary, four headline numbers and a list of CI-style
checks. Then the evidence: policy vs baselines, coverage-vs-risk curves,
per-segment plan vs actual, reliability diagrams, and the gates table.
The artifact hashes for auditability come last (collapsed). Pinned in
docs/interfaces.md §11.

### 6. Synthetic demo (`synth`)

Seeded generator with **injected, controlled error rates** — since we control
the ground truth, every claim is mathematically verifiable. Three scenarios:

1. **Well-calibrated**: sanity check; policy ≈ confidence routing.
2. **Overconfident model**: the typical real case; calibration + costs beat raw
   confidence clearly.
3. **Drift**: error distribution shifts between calibration and deployment
   periods; the backtester exposes that the stale policy degraded.

## CLI

```bash
triage calibrate --golden golden.parquet [--seed N] [--max-ece F [--ece-gate-mode point|ci_upper]] \
                 -o calibrator.json
triage plan --batch batch.parquet --calibrator calibrator.json \
            --costs costs.yaml --budget 0.10 -o policy.json
triage backtest --policy policy.json --calibrator calibrator.json \
                --history history.parquet -o backtest.json
triage report --from artifacts/ -o report.html
triage synth --scenario overconfident --seed 42 -o data/
triage run --scenario overconfident --seed 42 -o runs/demo   # whole chain, see interfaces.md §8
```

Exit codes (public API, see CLAUDE.md): 0 OK, 1 contract violation,
2 residual risk gate failed, 3 calibration gate failed, 4 internal error,
5 budget overrun gate failed (`backtest --max-budget-overrun`).
`plan` and `backtest` are designed to run inside CI as gates.

## Phases

### MVP (phase 1)

- Input contract + validation
- Calibration module (isotonic + Platt, ECE, bootstrap CIs)
- Cost model + optimizer + `policy.json`
- Synthetic generator (scenarios 1 and 2)
- README with the headline chart (policy vs baselines, even if generated by a script before the full report exists)

**Acceptance:** deterministic policy on seeded synthetic data; captured-risk
numbers match analytical expectations within tolerance; all tests green.

### v1 (phase 2)

- Backtester + both baselines
- Drift scenario (3)
- HTML report
- CI-ready exit-code gates + example GitHub Actions workflow

### v2 (future, do not build yet)

- **Grouped routing**: when a reviewer opens a document, reviewing an extra
  field is nearly free → route documents whose value is the sum of their
  fields' expected losses. Greedy stops being optimal; becomes a knapsack
  variant. Nice problem, explicitly out of MVP/v1 scope.

## Non-goals

- Not an eval framework (promptfoo/deepeval territory).
- Not a labeling tool or review UI — it decides *what* to review, not *how*.
- No LLM calls, no GPU, no services, no external dependencies at runtime.
- No domain-specific logic; `segment` stays opaque.
- Does not decide *whether* data is promotable (that's AI Harness's job);
  it allocates scarce review attention.

## Success criteria (portfolio)

1. A reader understands the value in 30 seconds from the README chart:
   "at a 10% review budget, this policy captures ~X% of expected loss vs ~Y%
   for raw confidence and ~Z% for random."
2. Every number in the README is reproducible with one command from a seed.
3. The artifact chain (calibrator → policy → backtest → report) is fully
   hash-linked and auditable.
