# Risk register

Origin: adversarial design review (Codex, 2026-07-13) of everything on `main`,
re-verified finding-by-finding against source before any action. Four
high-severity findings were raised; two were confirmed defects (fixed), two
challenged deliberately pinned design decisions (rejected as defects, but
each contained a residual risk worth tracking). This file records the
assessment and the decisions.

## Review findings — verdicts

| # | Finding | Verdict | Action |
|---|---------|---------|--------|
| F1 | Report hash chain compared declared hashes to declared hashes; only the calibrator hash was recomputed. A post-backtest edit of policy.json rendered as chain-verified. | **Confirmed defect** | Fixed: `report._verify_chain` now recomputes `content_hash(policy)` and `content_hash(backtest)` against the declared `policy_hash` / `backtest_hash` (interfaces.md §11.2). Tamper regression tests added. |
| F2 | Threshold replay admits every item tied at the threshold, so the replayed review set can exceed the planned budget (measured 1183 vs 1000 planned, overconfident @ seed 42). | **Rejected as defect** — pinned in interfaces.md §9.4: thresholds are the published policy, the count is a consequence. | Residual risk R1 below. |
| F3 | Artifacts are written before gates run, so a failed-gate artifact looks like a normal output. | **Rejected as defect** — pinned three times (interfaces.md §5.9, §8): exit codes are the public API; the artifact must exist for inspection on gate failure. | Residual risk R2 below. |
| F4 | `ece()` returns 0.0 on an empty holdout, so a tiny golden set passed the ECE gate with zero validation evidence. | **Confirmed defect** | Fixed: `calibration.fit` raises `ContractViolation` when either split is empty (interfaces.md §4). Regression tests added. |

## Residual risks

### R1 — Budget overrun under threshold ties is unbounded (from F2)

The system gates on residual risk (exit 2) but not on review capacity, the
resource it allocates. The measured 1183-vs-1000 tie inflation is
scenario-specific, not a bound: an isotonic calibrator fitted on a coarse
golden set can park an arbitrarily large fraction of a future batch exactly
at a threshold, and the ECE gate does not control that. If operator capacity
is hard, the tooling never flags the overrun.

**Decision: mitigate (v1.x).** Optional `--max-budget-overrun FACTOR` gate on
`triage backtest`, exit code 5 (declared in GOAL.md). Deliberately a CLI
flag, not a `gates:` YAML field: adding a field to the `Gates` model would
change `config_snapshot` in every policy.json and hence every `policy_hash`,
breaking committed golden artifacts for no benefit. Capping the replay
itself remains forbidden (interfaces.md §9.4). A ranked-prefix / knapsack
policy representation that enforces budgets exactly is a v2-scale spec
change; revisit only if hard capacity becomes a requirement.

### R2 — Failed-gate artifacts render into clean-looking reports (from F3)

Exit codes carry the gate outcome, but the HTML report — the human-facing
trust surface — displayed gate inputs (ECE, residual loss) without the
declared thresholds, so a failed-gate artifact rendered indistinguishably
from a passing one once exit-code context was lost.

**Decision: mitigate (v1.x).** Informational "Gates" panel in the report:
declared thresholds from `policy.config_snapshot.gates` vs measured values,
PASS/FAIL with the same strict `>` comparison as the CLI gates, plus a
planned-vs-replayed budget adherence row when a backtest is present. Purely
display: the report still never exits non-zero beyond contract violations
(pinned "no gates beyond exit 1").

**Update (report v1.1, interfaces.md §11).** The panel alone still let a
failed gate hide below the fold. The redesigned report puts the verdict
first: a failed declared gate turns the at-a-glance status red
("Gate failed") and its check opens at the top of the page, and a
plan-only report (no backtest) shows a neutral "Plan only" status instead
of green, because a plan can pass on expected numbers and still fail on
realized ones. Still display-only; exit codes unchanged.

### R3 — ECE gate tests the point estimate, not the uncertainty

The empty-split guard (F4 fix) enforces non-empty, not adequate: a holdout
of 1 now passes, and `ece_holdout` on a tiny holdout is noise. The bootstrap
CI is already computed and serialized; a stricter gate would test
`ece_ci[1] <= max_ece` (self-adjusting to sample size, preferable to an
arbitrary minimum-n).

**Decision: mitigate (v1.x), opt-in.** New `ci_upper` ECE gate mode that
tests `ece_ci[1] <= max_ece` (the original point-estimate gate stays the
default, exit code 3 unchanged). It is exposed as `calibrate --ece-gate-mode`
and `gates.ece_gate_mode` in the cost YAML (interfaces.md §3, §4, §8). Unlike
R1's overrun factor, this *is* a `Gates` field, because the report has to
know which statistic `plan` gated on. To avoid R1's hash concern,
`config_snapshot` omits the key when it holds the default, so default-mode
`policy_hash`es and the committed golden fixtures are unchanged, and old
policies (no key) read as `point`. Not made the default: every seeded
synthetic scenario's upper bound (0.074–0.082) fails the emitted
`max_ece: 0.05`, so a silent switch would turn passing pipelines into exit 3.

### R4 — Hash chain proves consistency, not authenticity

An adversary who retampers every artifact coherently (recomputing all
hashes) is undetectable. True by construction for any unsigned scheme;
signing requires key infrastructure that is out of scope for a local-first
tool.

**Decision: accept.** Documented boundary; build nothing.
