# Invariant test spec (auditor-owned, Stage 4)

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

## Rules for the implementing agent

- No assertion may be weakened, no tolerance widened; if a claim fails, the
  test stays failing and the discrepancy goes in your report.
- Tests must not depend on wall-clock, locale, or dict iteration order.
- Use `tmp_path` fixtures; never write into the repo tree.
