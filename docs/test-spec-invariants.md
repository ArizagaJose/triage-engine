# Invariant test spec (Stage 4 + Stage 5/v1)

Author: auditor; now owned by the repository owner. Changes to this file and
to the test files it defines (`tests/invariants/`) need the owner's approval.
No assertion may be weakened or tolerance widened (see "Rules" below).

## Where these tests live (owner-approved restructure)

The sections below were originally implemented as eight files at the
`tests/` root. They are now four files, one per concern, with every claim
assertion carried over verbatim:

| spec sections | file |
|---|---|
| A, D | `tests/invariants/test_determinism.py` |
| B, E | `tests/invariants/test_golden.py` |
| C, F, G.1–4 | `tests/invariants/test_claims.py` |
| G.5–9 (+ gates panel) | `tests/invariants/test_report.py` |

Shared setup (`tests/conftest.py`, `tests/support.py`):

- **Chains are built once per session.** `chain(scenario, seed)` runs the
  documented chain: `synth`, `calibrate`, `plan` with the emitted costs.yaml,
  and `backtest --history <own batch> --n-random-draws 200`, with one seed
  for every stage. That is exactly the setup the sections below prescribe,
  and any section that uses the same (scenario, seed) reuses it. Chains are
  read-only. The session fails if any test modified one, and tests that
  tamper with files work on copies.
- **"Real CLI".** Commands run through `cli.app_main`, the console script's
  entry point, in the test process (the `cli` fixture). That is the same
  argument parsing, serialization and exit-code mapping, which satisfies
  A's "subprocess or typer CliRunner, not internal functions".
- **Determinism pairs are cross-process.** Every "run twice and compare"
  pair (A.1–3, D.1, G.5) compares the shared chain (in-process) with a
  fresh console-script subprocess (`cli_process`). The two runs are
  therefore in separate processes, with independent hash seeds.
- The seed-43 calibrator of A.4, D.4 and G.7 is one shared fixture
  (`calibrator_seed43`, fitted on overconfident/42's golden set).

## A. Determinism (`tests/invariants/test_determinism.py`)

Run through the REAL CLI (subprocess or typer CliRunner invoking `app`), not
internal functions — float formatting and dict ordering bugs only show up at
the serialization boundary.

1. `synth --scenario overconfident --seed 42` into two temp dirs →
   `golden.parquet`, `batch.parquet`, `manifest.json`, `costs.yaml` all
   byte-identical across the two runs.
2. `calibrate` twice on the same golden.parquet (seed 42) → the two
   calibrator.json files are identical after removing the `created_at` line;
   equivalently, parse both, delete `created_at`, compare dicts equal AND
   compare `canonical_dumps` strings equal.
3. `plan` twice on same batch/calibrator/costs → same, plus `policy_hash`
   values equal, plus `policy_hash == content_hash(parsed_policy)` recomputed
   independently in the test.
4. Cross-check: `calibrate` with seed 43 produces a DIFFERENT calibrator
   (guards against seeds being ignored).

## B. Golden file (`tests/invariants/test_golden.py`)

- Committed fixture: `tests/golden/policy.json` — produced by the real CLI on
  `synth well_calibrated --seed 7`, calibrate seed 7, plan with the emitted
  costs.yaml, budget as in YAML (0.10). (Generated and committed by the
  auditor on 2026-07-12; `policy_hash` prefix `d797f827317e`; the test
  regenerates end-to-end into tmp and compares. Regenerated once after the
  audit fix that made `input_sha256s.calibrator` a semantic content_hash —
  the original file-bytes hash embedded the calibrator's own created_at and
  broke cross-run reproducibility.)
- Assert: regenerated policy equals fixture after deleting `created_at` in
  both (full dict equality, not just hash equality).
- A README-worthy property: the fixture's `policy_hash` matches
  `content_hash` of the fixture content (self-consistency of the committed
  artifact).

## C. Claim verification (`tests/invariants/test_claims.py`)

Setup (fixture, session-scoped): `synth overconfident --seed 42`; calibrate on
golden (seed 42); plan on batch with emitted costs.yaml (budget fraction 0.10).
Read `manifest.json` for realized per-segment error rates.

Claims (tolerances chosen for n_batch = 10 000, n_golden = 2 000):

1. **Calibration works**: `ece_holdout` (calibrated) < `ece_raw_holdout`
   (raw score), and `ece_holdout <= 0.05` (the emitted gate value). For the
   overconfident scenario the raw ECE must be LARGE: assert
   `ece_raw_holdout >= 0.10` (the model under-reports error by ~2/3 of p;
   mean p ≈ 0.213 → expected raw ECE ≈ 0.14; measured at seed 42:
   ece_raw_holdout = 0.1329, ece_holdout = 0.0463 — the 0.10 bound stands
   with margin for cross-platform float variance).
2. **Budget respected**: `totals.n_review == floor(0.10 * n_batch)` unless the
   positive-net-benefit rule stops earlier; assert
   `totals.n_review <= floor(0.10 * n_batch)` and, for this scenario, equality
   (expected-loss ranking at 10% budget stays well above review costs —
   auditor verified analytically: the marginal routed item has
   expected_loss >> review_cost here).
3. **Policy beats naive orderings on expected loss captured at equal spend**
   (computed in-test from calibrated p_error and true labels in batch —
   batch has true_value):
   - captured TRUE loss of policy review set (sum of false_accept_cost over
     routed items that are actually errors) strictly greater than captured
     true loss of (a) random selection of the same size (mean over 100 seeded
     draws) and (b) top-(1-score) raw-confidence selection of the same size.
   - Margin: policy ≥ 1.15 × raw-confidence captured loss for the
     overconfident scenario (auditor's analytic reasoning: raw confidence
     ignores per-segment costs; seg_high items with false_accept_cost 500 vs
     seg_low 10 differ 50× in loss density, while raw ranking mixes them).
     If this margin fails empirically at seed 42, report the actual number to
     the auditor — do NOT silently lower the threshold. (Auditor's measured
     values at seed 42, from scripts/headline_chart.py: policy 44.2% vs raw
     2.36% vs random 10.14% of total true loss — the 1.15× margin holds with
     a factor of ~18 to spare.)
4. **Residual accounting closes**: totals.expected_captured_loss +
   totals.expected_residual_loss == sum over all items of
   p_error*false_accept_cost (within 1e-6 relative), and per-segment numbers
   sum to totals (exact float sums, deterministic order).
5. **Realized error rates match injected**: per segment,
   |realized_batch_error_rate − analytic mean of Uniform bounds| ≤ 0.03
   (bounds: seg_high mean 0.155, seg_mid 0.21, seg_low 0.275).

## D. Backtest determinism (Stage 5 — `tests/invariants/test_determinism.py`)

Full chain through the REAL CLI: `synth overconfident --seed 42` →
`calibrate --seed 42` → `plan` (emitted costs.yaml, budget from YAML) →
`backtest --calibrator ... --seed 42`.

1. `backtest` run twice on the same inputs → the two backtest.json files are
   identical after removing `created_at`; `backtest_hash` values equal AND
   `backtest_hash == content_hash(parsed_backtest)` recomputed in the test.
2. `backtest --seed 43` → `strategies.random.captured_loss_mean` DIFFERS
   (guards against the seed being ignored) while the `policy` and
   `raw_confidence` strategy blocks are IDENTICAL dicts (no randomness may
   leak into them).
3. Hash-chain: `input_sha256s.policy` equals the policy.json's own
   `policy_hash`; `input_sha256s.calibrator` equals
   `policy.input_sha256s.calibrator`.
4. Tamper check: `calibrate --seed 43` into a different calibrator.json, then
   `backtest` with the seed-42 policy and the seed-43 calibrator → exit 1,
   stderr names the hash mismatch.

## E. Backtest golden file (Stage 5 — `tests/invariants/test_golden.py`)

- Committed fixture `tests/golden/backtest.json`: produced by the real CLI on
  the well_calibrated seed-7 chain (same chain as the policy fixture:
  synth well_calibrated --seed 7, calibrate seed 7, plan with emitted costs,
  then backtest with --history = the same batch.parquet, --seed 7). Generated
  and committed by the auditor on 2026-07-12 (`backtest_hash` prefix
  `b905edd6c077`; the same chain reproduced the policy fixture's
  `d797f827317e`, confirming the content_hash amendment left MVP hashes
  untouched). Regenerated by the auditor in Stage 6 when `curves` was added
  to backtest.json (§9.11) — new prefix `30a90c5bffdc`; the only fields that
  changed vs the Stage-5 fixture were verified to be exactly
  `curves`/`backtest_hash`/`created_at`. The test regenerates end-to-end
  into tmp and compares equality after deleting `created_at` in both.
- Self-consistency: fixture's `backtest_hash == content_hash(fixture)`.

## F. Backtest + drift claim verification (Stage 5 — `tests/invariants/test_claims.py`)

Chains (all seed 42, budget 0.10 from the emitted costs.yaml, backtest
--seed 42, --n-random-draws 200, history = the scenario's own batch.parquet):
`well_calibrated`, `overconfident`, `drift`.

Auditor's measured values at seed 42 are given in parentheses; tolerances are
pinned with margin. If a bound fails empirically, report the actual number to
the auditor — do NOT silently loosen it.

1. **Threshold replay is faithful**: for `well_calibrated` (platt selected),
   `strategies.policy.n_review == ` policy `totals.n_review` exactly
   (measured 1000 == 1000). For `overconfident` (isotonic selected), tie
   inflation is expected: planned ≤ replayed ≤ 1.3 × planned
   (measured 1183 vs 1000 planned).
2. **Policy beats baselines at equal capacity** (overconfident):
   `strategies.policy.captured_loss ≥ 1.15 × strategies.raw_confidence.captured_loss`
   (measured ratio ≈ 17.9×) and `> strategies.random.captured_loss_mean`
   (measured 144500 vs 34170). `strategies.policy.capture_rate ≥ 0.40`
   (measured 0.5029).
3. **Random baseline sanity**: `strategies.random.capture_rate` within ±0.03
   of `n_review / n_history` (uniform sampling captures ≈ the review
   fraction; measured 0.1189 vs 0.1183 for overconfident, 0.1004 vs 0.1000
   for well_calibrated).
4. **Drift is exposed**: for `drift`,
   `strategies.policy.residual_loss / expected.expected_residual_loss ≥ 1.8`
   (measured 2.355 — the injected drift_factor is 2.0). Controls: the same
   ratio for `well_calibrated` and `overconfident` lies in [0.7, 1.4]
   (measured 1.131 and 0.984).
5. **Drift injection matches analytic**: per segment, manifest
   `realized_mean_error_batch` within ±0.03 of
   `min(drift_factor * mean(Uniform bounds), 0.95)`: seg_high 0.31,
   seg_mid 0.42, seg_low 0.55 (measured 0.297, 0.418, 0.555). Golden rates
   still match the UNdrifted means of spec C.5 (golden is pre-drift) with a
   GOLDEN-TIER tolerance of ±0.045 — NOT C.5's ±0.03, which was validated
   only at n_batch=10 000. n_golden=2 000 (~667 rows/segment) has sd of the
   realized mean ≈ 0.014 for seg_high, and the measured seed-42 value is
   0.1218 vs analytic 0.155 (diff 0.0332, ≈2.3σ — sampling noise, verified
   scenario-independent). ±0.045 ≈ 3.1σ. [Auditor amendment 2026-07-12: the
   original spec text borrowed ±0.03 unvalidated at golden scale; the
   Stage-5 test agent correctly left the test failing and reported the
   number instead of weakening the bound.]
6. **Accounting closes**: for every strategy,
   `captured + residual == total_true_loss` exactly (residual is defined as
   the difference, §9.9 of interfaces.md — assert equality, it documents the
   convention); `strategies.policy.captured_loss` equals the sequential sum
   of its per-segment `captured_loss` in sorted segment order exactly;
   `total_true_loss` equals an independent in-test fsum of
   `(predicted != true) * false_accept_cost` over the history;
   `n_errors` equals the in-test error count.
7. **Drift CI gate story** (exit codes are the product): write a costs.yaml
   identical to the emitted one plus `gates.max_residual_risk: 200000`.
   On the drift chain: `plan` exits 0 (expected residual ≈ 142706 ≤ 200000)
   but `backtest` exits 2 (realized residual ≈ 336100 > 200000), and
   backtest.json EXISTS afterward (artifact-first rule). On the
   well_calibrated chain with the same gate: both exit 0
   (realized residual ≈ 161350 ≤ 200000).
8. **History without labels** → `backtest` exits 1 (drop `true_value` from
   the history parquet).

## G. Curves + report (Stage 6 — `tests/invariants/test_claims.py` for 1–4,
## `tests/invariants/test_report.py` for 5–9)

Curves (backtest.json `curves`, interfaces.md §9.11; auditor's measured
seed-42 values in parentheses):

1. Schema: `grid == [i/100 for i in range(51)]` exactly; the three series
   each have 51 entries; `random == [floor(f*n)/n for f in grid]` exactly;
   `policy[0] == raw_confidence[0] == 0.0`; every series is monotone
   non-decreasing and bounded by 1.0.
2. Headline reproducibility (overconfident chain): `curves.policy` at
   f=0.10 in [0.42, 0.46] (measured 0.4420 — the Stage-4 README headline
   44.2% now comes from the artifact); `curves.raw_confidence` at f=0.10
   in [0.015, 0.035] (measured 0.0236); `curves.policy` at f=0.10 ≥ 10 ×
   `curves.raw_confidence` at f=0.10 (measured ~18.7×).
3. Independent recomputation: for the overconfident chain, recompute
   `curves.policy[10]` in-test from batch.parquet + the calibrator artifact
   + costs.yaml (net_benefit ranking, top-1000, fsum of true loss / total)
   and assert exact equality with the artifact value (float-identical — the
   pinned semantics leave no room for drift).
4. Divergence is documented behavior: overconfident
   `strategies.policy.capture_rate` (measured 0.5029) minus
   `curves.policy[10]` (0.4420) is positive (tie inflation reviews more) —
   assert `capture_rate >= curves.policy[10]`.

Report (`triage report`):

5. Determinism: full overconfident seed-42 chain, `report` run twice into
   two paths → the two HTML files are identical after dropping the single
   line containing `id="created-at"` (assert exactly one such line each).
6. Hash chain rendered: the HTML contains the full `policy_hash`, the
   calibrator `content_hash`, and the `backtest_hash` as substrings; the
   three pinned div ids (`fig-reliability`, `fig-backtest`,
   `fig-coverage`) each appear; `plotly` embedded (no `cdn.plot.ly` /
   `https://` plotly script src anywhere in the file).
7. Tamper: `report` with a calibrator.json recalibrated at seed 43 in the
   --from dir → exit 1, stderr names a hash. Missing policy.json → exit 1.
8. Backtest-optional: --from dir WITHOUT backtest.json → exit 0, HTML
   contains the "backtest not run" note and does NOT contain
   `fig-coverage`; with backtest.json present the note is absent.
9. Report is not an exit-code gate: the drift chain (realized residual >
   expected) still reports exit 0 — gating belongs to plan/backtest.

## H. Report redesign — at-a-glance, checks, KPIs (interfaces.md §11 v1.1;
## new file `tests/test_report_summary.py`; updates to `tests/unit/test_report.py`
## and `tests/invariants/test_report.py`)

G.5, G.7, G.8 and G.9 above are unaffected by the redesign and keep their
assertions. G.6 gains one more div id (`fig-segments`, full mode only) —
update that assertion in place rather than duplicating it. Everything else
the redesign touches is specified here.

### H.1 `_build_summary` — pure unit tests, no HTML, no CLI

One handcrafted (calibrator, policy, backtest-or-None) triple per case,
built the same way `tests/unit/test_report.py`'s fixtures are (hashes computed
on the fly so chain checks pass by construction). Assert on `Summary.status`,
`Check.id`/`.severity`/`.params` — never on `.title`/`.detail` text
(interfaces.md §11.5.2: wording is not a contract).

1. **Gate checks, boundary-exact** (mirrors `_gate_row`'s strict `>`):
   `expected_residual_loss == max_residual_risk` → `pass`;
   `expected_residual_loss == max_residual_risk + epsilon` → `fail`;
   `max_ece is None` → `gate.max_ece` is `info`; same three cases repeated
   for `gate.max_residual_risk.realized` with a backtest present.
   Parametrize one test over the three gates rather than three near-copies.
2. **`gate.max_residual_risk` collapses to one `info` check when the gate
   is undeclared** — assert `gate.max_residual_risk.planned` and
   `.realized` are ABSENT from the check list in that case (not merely
   that the merged id is present).
3. **`plan.vs_actual` boundary**: `realized == 1.25 * expected` → `pass`;
   `realized == 1.25 * expected + epsilon` → `warn`; `expected == 0,
   realized == 0` → `pass`; `expected == 0, realized > 0` → `warn`.
4. **`worst_segment` selection**: three segments with distinct gaps → the
   largest wins; two segments tied on gap → name-ascending tie-break;
   a segment present in `backtest.strategies.policy.segments` but absent
   from `policy.segments` (unseen at plan time, §9.4) contributes an
   expected residual of 0.0 and is eligible to win.
5. **`value.vs_baselines`**: policy below random → `warn`; policy below
   raw-confidence only → `warn`; `total_true_loss == 0` → `info`, not
   `pass` (nothing was there to catch, so "winning" is not a claim).
6. **`calibration.vs_raw`**: `ece_holdout > ece_raw_holdout` → `info`
   (NOT `warn` — this is the deliberate downgrade from the mockup, guard
   against regressing it back to `warn`).
7. **`calibration.ci_margin`**: only emitted when `max_ece` is declared
   AND `ece_holdout <= max_ece < ece_ci[1]`; absent when `max_ece` is None;
   absent when the point estimate already fails; absent when the CI upper
   bound is also `<= max_ece`.
8. **`budget.adherence`**: `replayed_n_review == planned_n_review` → `pass`;
   `replayed > planned` → `info` with `params.ratio == replayed/planned`;
   `planned_n_review == 0` → `info` with `params.ratio is None` (no
   `ZeroDivisionError`).
9. **Plan-only** (`backtest_dict=None`): checks are exactly
   `{gate.max_ece, gate.max_residual_risk(.planned)?, calibration.vs_raw,
   calibration.ci_margin?, backtest.not_run}` — none of `plan.vs_actual`,
   `value.vs_baselines`, `gate.max_residual_risk.realized`,
   `budget.adherence` appear. `status` is computed from this reduced set
   (a plan-only report can still be `attention`/`failed` via the ECE or
   planned-residual gates alone).
10. **Status derivation** is table-driven over synthetic severity sets:
    `{fail}` / `{fail, warn}` / `{warn, info}` / `{info, pass}` / `{pass}`
    / `{}` (degenerate) → `failed` / `failed` / `attention` / `ok` / `ok`
    / `ok`.
11. **Ordering**: severity groups appear in `fail, warn, info, pass` order;
    within a group, catalogue order from §11.5's table (assert the full
    `[c.id for c in summary.checks]` sequence for one fully-populated
    fixture, so a reordering is caught even if no severities change).
12. **Purity / no mutation**: `deepcopy` the three input dicts, call
    `_build_summary`, assert the originals are unchanged (`==` the
    deepcopies) — catches accidental in-place `.pop`/`.sort` on shared
    nested dicts (e.g. `gates`, `segments`).
13. **Escaping stays the template's job**: assert `Check.title` / `.detail`
    for a segment named `<script>alert(1)</script>` contain the LITERAL
    (unescaped) string — the check must not pre-escape, or double-escaping
    would show up as `&amp;lt;` once the template autoescapes it. The
    existing HTML-level escaping assertions in `tests/unit/test_report.py`
    (§H.3.7) are what actually guards against injection.

### H.2 Property tests (hypothesis, `derandomize=True` or fixed seed)

1. For random severity multisets drawn from `{fail, warn, info, pass}`,
   `status` always equals the mapping in H.1.10 (no hand-picked case is a
   coincidence).
2. For randomly generated (cost, threshold, residual) fixtures, every gate
   `Check.severity` agrees with the corresponding row `_build_gates_view`
   would produce for the SAME inputs — the two code paths must not
   silently diverge (interfaces.md §11.5.3).
3. Shuffling the insertion order of `policy["segments"]` /
   `backtest["strategies"]["policy"]["segments"]` dict keys does not change
   `Summary` (`worst_segment`, `checks`, `kpis` all identical) — guards the
   "no dict-iteration-order dependence" invariant specifically for the new
   code.
4. `_build_summary` never raises for any combination of: zero `n_total`,
   zero `total_true_loss`, `n_review == 0`, a segment with zero
   `review_cost`/`false_accept_cost`, all gates None, all gates set to 0.

### H.3 Rendering tests (`tests/unit/test_report.py`, handcrafted dicts — extend
### the existing fixtures/classes, don't replace them)

1. `#at-a-glance` is the first element inside `<main>`/`<body>` content —
   assert its start index precedes every other section id's start index
   AND precedes the plotly bundle marker (below).
2. The embedded plotly bundle: searching for `Plotly.newPlot` is NOT
   enough (it appears once per figure, and once more inside a comment in
   the bundle itself). Instead, assert that plotly.js's own banner
   (`plotly.js v`, present exactly once in `get_plotlyjs()`) appears exactly
   once, and that its index is AFTER the close of `#at-a-glance` and BEFORE
   the first figure div. This operationalizes the §11.4 placement rule.
3. `main` carries `data-status` and `data-mode` whose values match
   `Summary.status`/`.mode` for each of the four handcrafted fixtures (ok
   full, attention full, failed full, ok plan-only) — reuse/extend
   `_valid_triplet()` with three more variants rather than new ad-hoc
   dicts.
4. Every rendered check has `data-check-id="{id}"` and
   `data-severity="{severity}"`; `fail`/`warn` checks render with the
   `open` attribute present, `info`/`pass` without it.
5. Status is never color-only: for each status, assert the badge's text
   label (`"Gate failed"` / `"Needs attention"` / `"On track"` / `"Plan
   only — not backtested"`) is present as text, not just inferred from a
   CSS class/color.
6. KPI tiles: `data-kpi="reviewed|capture|lift|residual"` all present in
   both modes (with mode-appropriate values — "—" where §11.7 pins it,
   e.g. `lift` in plan-only).
7. Escaping (extends the existing `TestEscaping` class): the
   `<script>alert(1)</script>` segment name from `_cost_model_dict` must
   additionally NOT appear unescaped inside the `plan.vs_actual` check's
   rendered detail text (when it is `worst_segment`) or the per-segment
   table — add a fixture variant where the worst-gap segment IS the
   script-tag segment, not just the other one, to close the gap the
   current fixture doesn't exercise.
8. Section order: `#at-a-glance`, `#results` (or the plan-only note in its
   place), `#segments`, `#calibration`, `#gates`, `#audit` appear in that
   index order; `#audit`'s content sits inside a `<details>` that is
   closed by default (no `open` attribute) while still containing the
   full hash table in the DOM (extends the existing hash-chain-rendered
   assertion to also check `open` is absent).
9. Formatting filters, unit-tested directly (import and call, not via
   `render`): `count`/`cost`/`pct`/`ratio`/`lift`/`ece` on edge values —
   `0`, negative values (a plan-vs-actual difference can be negative),
   `None` → a literal `"—"` from every filter, and the lift boundary: below
   10 keeps one decimal, from 10 on none (`9.94` → `9.9×`, `9.99` → `10.0×`
   because rounding happens after the branch, and `10` → `10×`, per
   §11.11.3).
10. No external resources (extends existing check): no `<link`, no
    `@import`, no `url(http`, in addition to the existing `<script src=`
    scan. The embedded plotly.js bundle contains none of these strings, so
    the scan covers the WHOLE file (if a future plotly version adds one,
    fail loudly and decide then; never silently exclude the bundle).

### H.4 CLI-driven integration (`tests/invariants/test_report.py`, extend
### the D/F/G chain idiom — one more chain variant: drift + a costs.yaml
### carrying `max_residual_risk: 200000`, matching the CI workflow, §12)

1. overconfident seed-42 chain → `report` → `data-status="ok"
   data-mode="full"`, checks list contains no `fail`/`warn`.
2. drift seed-42 chain (no residual gate) → `data-status="attention"`; the
   rendered `plan.vs_actual` check's value text contains a ratio `>= 1.25`
   recomputed independently in the test from the SAME backtest.json (not
   hardcoded), and its detail names `seg_high` (recomputed as the argmax
   independently, not asserted by string literal alone).
3. drift chain + `max_residual_risk: 200000` (the CI gate demo) →
   `backtest` itself exits 2 (already covered in §F), but `report` on the
   resulting dir still exits 0 with `data-status="failed"` — reusing
   interfaces.md §11.12 / G.9 is not a gate.
4. drift chain, `report --from` the calibrator+policy only (drop
   backtest.json) → `data-status="ok" data-mode="plan_only"` — the SAME
   drift inputs that are `attention` in full mode render as the neutral
   "Plan only" label, not green, in plan-only mode (the R2-shaped case
   this redesign exists to fix: a report must not look healthy just
   because it wasn't checked against reality).
5. Golden summaries: for the overconfident / drift / drift-gated seed-42
   chains, save `{status, mode, [(id, severity) for checks], kpis}` (NOT
   raw HTML) to `tests/golden/report_summary_<scenario>.json`; regenerate
   end-to-end and compare dict equality after stripping nothing
   (no timestamps in this projection). Committed fixtures, like
   `tests/golden/policy.json` — a wording change never touches these, a
   logic change always does.

## Rules for the implementing agent

- No assertion may be weakened, no tolerance widened; if a claim fails, the
  test stays failing and the discrepancy goes in your report.
- Tests must not depend on wall-clock, locale, or dict iteration order.
- Use `tmp_path` fixtures; never write into the repo tree.
- Shared chains (`chain(...)`) are read-only; write into `tmp_path` or a
  `Chain.copy_to()` copy. The session fails if a shared chain changes.
