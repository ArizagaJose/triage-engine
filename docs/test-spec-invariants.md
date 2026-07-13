# Invariant test spec (auditor-owned, Stage 4 + Stage 5/v1)

Author: auditor. Implementing agents for WP1–WP7 must NOT edit this file or the
test files it defines. The Stage-4 implementing agent translates this spec into
`tests/test_determinism.py`, `tests/test_golden.py`, `tests/test_claims.py`
without weakening any assertion.

## A. Determinism (`tests/test_determinism.py`)

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

## B. Golden file (`tests/test_golden.py`)

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

## C. Claim verification (`tests/test_claims.py`)

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

## D. Backtest determinism (Stage 5 — append to `tests/test_determinism.py`
## or a new `tests/test_determinism_backtest.py`; do NOT edit Stage-4 tests)

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

## E. Backtest golden file (Stage 5 — `tests/test_golden.py` pattern, new
## file `tests/test_golden_backtest.py`)

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

## F. Backtest + drift claim verification (Stage 5 — new file
## `tests/test_claims_backtest.py`; session-scoped fixtures; real CLI)

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

## G. Curves + report (Stage 6 — new files `tests/test_claims_curves.py`,
## `tests/test_report_invariants.py`; real CLI; reuse the D/F chain idiom)

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

## Rules for the implementing agent

- No assertion may be weakened, no tolerance widened; if a claim fails, the
  test stays failing and the discrepancy goes in your report.
- Tests must not depend on wall-clock, locale, or dict iteration order.
- Use `tmp_path` fixtures; never write into the repo tree.
