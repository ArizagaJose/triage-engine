# interfaces.md — pinned contracts (MVP + v1)

This document is the binding contract between modules. Implementing agents code
against it exactly. Any deviation must be flagged, not improvised. Read
`GOAL.md` and `CLAUDE.md` first; where those are ambiguous, this file decides.

Versioning: `triage.__version__ = "0.1.0"` (single source in
`src/triage/__init__.py`; `pyproject.toml` reads it or duplicates it — keep in
sync).

---

## 1. Exceptions (`src/triage/exceptions.py`)

```python
class TriageError(Exception):
    """Base for all typed errors."""

class ContractViolation(TriageError):
    """Invalid input data or invalid cost YAML. CLI exit code 1."""

class ResidualRiskExceeded(TriageError):
    """Residual risk above gates.max_residual_risk. CLI exit code 2."""

class CalibrationGateFailed(TriageError):
    """ECE above the declared maximum. CLI exit code 3."""
```

Any other exception → exit code 4 (internal error). Only `cli.py` calls
`sys.exit` / `typer.Exit`. Exception messages must be human-readable and
specific (name the column, the value, the limit).

## 2. Input contract (`src/triage/contract.py`)

```python
def validate(df: pl.DataFrame, *, require_truth: bool = False) -> pl.DataFrame:
```

Checks (collect ALL violations, then raise one `ContractViolation` listing
every problem, one per line):

- Required columns exist: `item_id`, `segment`, `score`, `predicted_value`.
- Dtypes: `item_id`/`segment`/`predicted_value` are `pl.String`; `score` is
  `pl.Float64` (accept `pl.Float32` and cast up — the ONLY permitted coercion).
- `item_id`: non-null, unique.
- `segment`, `predicted_value`: non-null.
- `score`: non-null, finite, `0.0 <= score <= 1.0`.
- `true_value`: optional column, dtype `pl.String`. If `require_truth=True` it
  must exist and be non-null everywhere.
- Empty dataframe (0 rows) is a violation.

No other coercion, no dropping rows. Returns the (possibly score-upcast)
dataframe unchanged otherwise.

Reader helper (used by CLI):

```python
def read_input(path: Path, *, require_truth: bool = False) -> pl.DataFrame
```

Accepts `.parquet` or `.csv` (CSV read with `try_parse_dates=False`, all
columns as declared types via explicit `schema_overrides`; a CSV whose values
can't parse as those types is a `ContractViolation`, not a crash).

## 3. Cost model (`src/triage/costs.py`)

Pydantic v2 models, strict (`model_config = ConfigDict(extra="forbid")`):

```python
class Budget(BaseModel):
    type: Literal["fraction", "count", "money"]
    value: float                      # > 0; if fraction: <= 1.0

class Gates(BaseModel):
    max_residual_risk: float | None = None   # >= 0
    max_ece: float | None = None             # in (0, 1]

class SegmentCosts(BaseModel):
    false_accept_cost: float          # >= 0
    review_cost: float                # >= 0

class CostModel(BaseModel):
    budget: Budget
    gates: Gates = Gates()
    segments: dict[str, SegmentCosts]  # non-empty

    def costs_for(self, segment: str) -> SegmentCosts:
        """Exact key, else segments['default'], else raise ContractViolation."""
```

```python
def load(path: Path) -> CostModel
```

YAML parse errors and pydantic `ValidationError`s are wrapped into
`ContractViolation` with a readable message. The exact YAML shape is the
example in GOAL.md.

## 4. Calibration (`src/triage/calibration.py`)

Semantics: the calibrator maps a raw model `score` to a **calibrated
probability of error** `p_error`. Golden-set error label:
`error = (predicted_value != true_value)` (exact string inequality, int 0/1).

```python
@dataclass(frozen=True)
class CalibrationConfig:
    holdout_fraction: float = 0.3
    n_bins: int = 10                  # ECE bins, equal-width on [0, 1]
    n_bootstrap: int = 1000
    ci_level: float = 0.95

@dataclass(frozen=True)
class Calibrator:
    method: str                       # "isotonic" | "platt"
    params: dict[str, Any]            # see serialization below
    ece_holdout: float
    ece_ci: tuple[float, float]
    ece_raw_holdout: float            # ECE of the uncalibrated score
    n_golden: int
    seed: int
    input_sha256: str
    tool_version: str
    config: CalibrationConfig
    reliability_bins: dict[str, Any]  # {"before": [...], "after": [...]}

    def predict(self, scores: np.ndarray) -> np.ndarray: ...  # -> p_error
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Calibrator": ...

def fit(golden: pl.DataFrame, *, seed: int, config: CalibrationConfig = ...,
        input_sha256: str = "") -> Calibrator
```

Pinned algorithm details:

- **Split**: sort rows by `item_id` (stable), then permute with
  `np.random.default_rng(seed)`; last `holdout_fraction` of the permutation is
  the holdout. Never use sklearn's splitters.
- **Isotonic**: `sklearn.isotonic.IsotonicRegression(y_min=0.0, y_max=1.0,
  increasing="auto", out_of_bounds="clip")` fit on `(score, error)` of the
  calibration split.
- **Platt**: 1-D logistic regression of `error` on `score`
  (`sklearn.linear_model.LogisticRegression(C=1e10, solver="lbfgs")` — i.e.
  effectively unregularized classic Platt). `p_error = sigmoid(a*score + b)`.
- **Prediction is sklearn-free**: `predict()` uses ONLY the serialized params —
  isotonic via `np.interp(scores, x_thresholds, y_thresholds)` (clip outside);
  platt via the sigmoid with stored `a`, `b`. This guarantees exact JSON
  round-trip determinism. sklearn is used at fit time only.
- **ECE** (function `ece(p_pred: np.ndarray, y: np.ndarray, n_bins: int) -> float`):
  equal-width bins on predicted `p_error`; per bin,
  `|mean(p_pred) - mean(y)| * (n_bin / n_total)`; sum over non-empty bins.
  "Before"/raw ECE uses `p_pred = 1 - score`.
- **Selection**: fit both on the calibration split, compute ECE of each on the
  holdout, pick the lower; tie → isotonic.
- **Bootstrap CI**: resample the holdout with replacement `n_bootstrap` times
  using `np.random.default_rng(seed + 1)`, recompute ECE of the selected
  calibrator's predictions, take percentile CI (`(1-ci_level)/2` on each side).
- **Gate**: `fit` does NOT gate. A separate helper
  `check_ece_gate(calibrator, max_ece: float) -> None` raises
  `CalibrationGateFailed` if `ece_holdout > max_ece`.

Serialization (`params`): isotonic → `{"x_thresholds": [...], "y_thresholds": [...]}`;
platt → `{"a": float, "b": float}`.

`reliability_bins`: for both "before" and "after", a list of
`{"bin_lo", "bin_hi", "mean_pred", "mean_observed", "count"}` (empty bins
included with `count: 0` and nulls for means).

## 5. Optimizer (`src/triage/optimizer.py`)

```python
@dataclass(frozen=True)
class Policy:
    ...  # mirrors policy.json content, see §7

def plan(batch: pl.DataFrame, calibrator: PredictsErrors, cost_model: CostModel,
         *, input_sha256s: dict[str, str], tool_version: str) -> Policy
```

`PredictsErrors` is a small structural Protocol defined in `optimizer.py`
(`def predict(self, scores: np.ndarray) -> np.ndarray`); the real `Calibrator`
satisfies it. This keeps optimizer tests independent of the calibration
implementation.

Pinned semantics (do not improvise):

1. `p_error = calibrator.predict(score)` per item.
2. `expected_loss = p_error * false_accept_cost(segment)`.
3. `net_benefit = expected_loss - review_cost(segment)`.
4. Sort by `net_benefit` DESC, tie-break `item_id` ASC (single stable sort).
5. Budget in review units:
   - `fraction` → `floor(value * n_batch)` items;
   - `count` → `int(value)` items (capped at `n_batch`);
   - `money` → walk the sorted list, admit while cumulative
     `review_cost <= value`; stop at the FIRST item that does not fit (no
     skip-and-continue — keeping the review set a prefix of the ranking is
     what makes per-segment thresholds exact).
6. Route to review while budget remains **and** `net_benefit > 0` (items with
   non-positive net benefit are never routed, even under unspent budget).
7. Per-segment `p_error_threshold` = min `p_error` among routed items of that
   segment; segments with zero routed items get threshold `null` (= review
   nothing). Applying a policy later means: review iff
   `p_error >= p_error_threshold`.
8. `expected_captured_loss` = Σ expected_loss over routed items;
   `expected_residual_loss` = Σ expected_loss over non-routed items. Both per
   segment and total. Per-segment values use `math.fsum`; the TOTALS are
   defined as the plain sequential sum of the per-segment values in sorted
   segment order, so the accounting table in policy.json ties out exactly.
9. Residual gate is checked by a separate helper
   `check_residual_gate(policy, cost_model) -> None` raising
   `ResidualRiskExceeded` (the CLI writes `policy.json` FIRST, then checks the
   gate — the artifact must exist for inspection even when the gate fails).

Everything is computed with float64; no randomness anywhere in this module.

## 6. Synthetic generator (`src/triage/synth.py`)

```python
def generate(scenario: str, *, seed: int) -> dict[str, pl.DataFrame | dict]
    # keys: "golden", "batch", "manifest"
def write(scenario: str, *, seed: int, out_dir: Path) -> None
    # writes golden.parquet, batch.parquet, manifest.json, costs.yaml
```

Scenarios: `"well_calibrated"`, `"overconfident"`, `"drift"` (v1). Shared design (all
randomness from a single `np.random.default_rng(seed)`; row order finally
sorted by `item_id`; ids are zero-padded `g-000001` / `b-000001` so sorts are
stable):

- 3 segments: `seg_high` (false_accept_cost 500, review_cost 4),
  `seg_mid` (50, 3), `seg_low` (10, 2) — mirrored in the emitted `costs.yaml`
  (with `default` = the seg_mid costs, budget fraction 0.10,
  gates: max_ece 0.05, no max_residual_risk).
- Golden set: 2 000 rows; batch: 10 000 rows, same generative process.
- Per item: true error probability `p` drawn per segment —
  `seg_high: Uniform(0.01, 0.30)`, `seg_mid: Uniform(0.02, 0.40)`,
  `seg_low: Uniform(0.05, 0.50)`. `error ~ Bernoulli(p)`.
  `true_value = "v"`, `predicted_value = "v"` if no error else `"x"`.
  Both golden AND batch include `true_value` (the contract allows it; claim
  verification needs it).
- Score:
  - `well_calibrated`: `score = 1 - p` (plus `Normal(0, 0.01)` noise, clipped
    to [0,1]).
  - `overconfident`: `score = 1 - p/3` (same noise, clipped) — the model
    reports one third of its true error rate.
- `manifest.json` (sorted keys) records: scenario, seed, n rows, per-segment
  `p` distribution bounds, the score formula name, noise sigma, `drift_factor`
  (see below; `null` for non-drift scenarios — the key is present in ALL
  manifests so the schema is uniform), and the **realized** per-segment mean
  error rates of golden and batch — this is what Stage-4/5 claim tests read to
  compute analytic expectations.

**Drift scenario (v1)** — pinned semantics:

- `golden` is generated by the IDENTICAL process as `well_calibrated`
  (score = `1 - p` + noise, `error ~ Bernoulli(p)`). Same seed ⇒ the drift
  golden.parquet is byte-identical to the well_calibrated one. The calibrator
  fit on it is honest for the calibration period.
- `batch` (the deployment/history period): rng consumption order UNCHANGED
  (segments, per-segment p draws, errors, noise), score formula UNCHANGED
  (`1 - p` + noise, on the ORIGINAL p), but errors are drawn from the drifted
  probability: `error ~ Bernoulli(min(p * 2.0, 0.95))`. `drift_factor = 2.0`,
  recorded in the manifest.
- The point: the calibrator and policy are honest for the calibration period
  and stale for the deployment period; the backtester must expose the gap
  (realized residual loss ≈ 2× the policy's expected residual loss).
- `costs.yaml` emitted unchanged (same segments/budget/gates as the other
  scenarios).

## 7. Artifact JSON — canonical form (all writers)

One shared helper module rule (put in `src/triage/artifacts.py`):

```python
def canonical_dumps(obj: dict) -> str:
    # json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n"
def content_hash(obj: dict) -> str:
    # sha256 of canonical_dumps(obj) after deleting keys
    # "created_at", "policy_hash" and "backtest_hash" (if present) at the
    # TOP level ("backtest_hash" added in v1; popping an absent key is a
    # no-op, so all MVP hashes are unchanged)
def file_sha256(path: Path) -> str
def write_artifact(obj: dict, path: Path) -> None
def utc_now_iso() -> str      # timezone-aware UTC, seconds precision
```

**calibrator.json** top-level keys: `config`, `created_at`, `ece_ci`,
`ece_holdout`, `ece_raw_holdout`, `input_sha256`, `method`, `n_golden`,
`params`, `reliability_bins`, `seed`, `tool_version`.

**policy.json** top-level keys: `budget` (as declared in YAML),
`config_snapshot` (the full cost YAML as parsed), `created_at`,
`input_sha256s` (`{batch, calibrator, costs}`), `policy_hash`, `segments`
(name → `{expected_captured_loss, expected_residual_loss, n_review, n_total,
p_error_threshold}`), `tool_version`, `totals` (`{expected_captured_loss,
expected_residual_loss, n_review, n_total, review_spend}`).

`policy_hash = content_hash(policy_dict)`. Floats are written as-is (Python
repr via json); never round for hashing.

**backtest.json** (v1) top-level keys: `backtest_hash`, `created_at`,
`expected` (`{expected_captured_loss, expected_residual_loss}` — copied
verbatim from the replayed policy's `totals`, so the report and the drift
tests can compare modeled vs realized without re-reading policy.json),
`input_sha256s` (`{calibrator, history, policy}` — `calibrator` and `policy`
are semantic `content_hash`es of the parsed artifacts, `history` is
`file_sha256` of raw bytes; note `input_sha256s.policy` therefore EQUALS the
replayed policy's own `policy_hash`, which is the hash-chain link),
`n_errors` (int: history rows where `predicted_value != true_value`),
`n_history`, `n_random_draws`, `seed`, `strategies` (see §9),
`tool_version`, `total_true_loss`.

`backtest_hash = content_hash(backtest_dict)`.

## 8. CLI (`src/triage/cli.py`)

Typer app, entry point `triage` (pyproject `[project.scripts]`).

```
triage synth --scenario overconfident --seed 42 -o data/
triage calibrate --golden data/golden.parquet --seed 42 [--max-ece F] -o artifacts/calibrator.json
triage plan --batch data/batch.parquet --calibrator artifacts/calibrator.json \
            --costs costs.yaml [--budget F] -o artifacts/policy.json
triage backtest --policy artifacts/policy.json --calibrator artifacts/calibrator.json \
            --history data/history.parquet [--seed 42] [--n-random-draws 200] \
            -o artifacts/backtest.json
```

`backtest` pinned behavior (v1):

- **Deviation from GOAL.md's CLI sketch, deliberate**: `--calibrator` is
  required. The policy stores per-segment thresholds on CALIBRATED `p_error`,
  so replaying it on new data requires the calibrator that produced those
  p_error values. The CLI verifies `content_hash(calibrator_dict) ==
  policy.input_sha256s.calibrator` and raises `ContractViolation` (exit 1) on
  mismatch — this check is what makes the artifact chain auditable rather
  than trust-based. (GOAL.md's CLI section carries a matching note.)
- `--history` is read with `require_truth=True` (labels are the point).
- `--seed` seeds ONLY the random-routing baseline (default 42, shown in
  help). `--n-random-draws` default 200.
- Gating order: write backtest.json FIRST, then
  `check_realized_residual_gate` (exit 2) against
  `gates.max_residual_risk` from the policy's `config_snapshot`. This is the
  CI drift gate: `plan` can pass on expected numbers while `backtest` fails
  on realized ones.
- Success prints a one-line summary (n_review, captured, residual,
  backtest_hash prefix, output path) and exits 0.

- `--seed` is required-explicit where randomness exists (`synth`, `calibrate`);
  no hidden defaults (typer default allowed but must be a literal, documented).
  Pin: default `--seed 42` on both, shown in help.
- `--budget` (plan) overrides `budget.value` from the YAML, keeping
  `budget.type`.
- `calibrate` gating: if `--max-ece` given, run `check_ece_gate` (exit 3) —
  artifact is written first.
- `plan` gating order: write policy.json, then `check_ece_gate` against
  `gates.max_ece` from the YAML (exit 3), then `check_residual_gate` (exit 2).
- `plan`'s `input_sha256s`: `batch` and `costs` are `file_sha256` of raw
  bytes; `calibrator` is `content_hash` of the PARSED artifact dict
  (semantic hash, `created_at` excluded). The calibrator file embeds its own
  timestamp, so hashing bytes would make `policy_hash` depend on when the
  calibrator was fitted rather than on what it computes, breaking golden-file
  reproducibility across independent end-to-end runs.
- Exit-code mapping in ONE place (a `main`-level try/except):
  `ContractViolation → 1`, `ResidualRiskExceeded → 2`,
  `CalibrationGateFailed → 3`, other `Exception → 4` (message to stderr,
  traceback only with `--verbose`). Success prints a one-line summary to
  stdout and exits 0.
- Pinned: the typer app runs with `standalone_mode=False`; bad CLI usage
  (unknown flag, missing option) maps to exit 1 — it is an input contract
  violation, and click's default usage-error code 2 would collide with the
  residual-risk gate meaning. `app_main` uses `sys.exit`, never `typer.Exit`
  outside the runner (it is not honored there).

## 9. Backtester (`src/triage/backtest.py`) — v1

Replays a policy against labeled history and compares it, at identical review
capacity, against random routing and raw-confidence routing. All "loss"
quantities here are REALIZED (ground-truth) losses:
`true_loss(item) = (predicted_value != true_value) * false_accept_cost(segment)`
— not the calibrated expectation the optimizer used.

```python
@dataclass(frozen=True)
class BacktestResult:
    ...  # mirrors backtest.json content minus created_at, see §7

    def to_dict(self) -> dict[str, Any]: ...

def run(policy_dict: dict[str, Any], calibrator: PredictsErrors,
        history: pl.DataFrame, *, seed: int, n_random_draws: int = 200,
        input_sha256s: dict[str, str], tool_version: str) -> BacktestResult

def check_realized_residual_gate(result: BacktestResult,
                                 cost_model: CostModel) -> None
    # raises ResidualRiskExceeded if strategies.policy.residual_loss >
    # gates.max_residual_risk (None -> no gate)
```

`PredictsErrors` is imported from `optimizer.py` (same Protocol; tests may
stub it). `policy_dict` is the PARSED policy.json.

Pinned semantics (do not improvise):

1. **Policy self-check**: `content_hash(policy_dict) ==
   policy_dict["policy_hash"]`, else `ContractViolation` naming both hashes.
   (The calibrator↔policy linkage check lives in the CLI, §8, which has the
   parsed calibrator dict; `run` only sees the predictor object.)
2. **Cost model**: `CostModel.model_validate(policy_dict["config_snapshot"])`;
   wrap `ValidationError` in `ContractViolation`. Missing/malformed policy
   keys are `ContractViolation`s, never `KeyError`s.
3. `history` is validated with `contract.validate(history, require_truth=True)`.
4. **Policy strategy (threshold replay)**: `p_error =
   calibrator.predict(score)`; review iff the item's segment has a non-null
   `p_error_threshold` in `policy_dict["segments"]` AND
   `p_error >= threshold`. Segments unseen at plan time (absent from
   `policy_dict["segments"]`) are never reviewed — a policy must apply to
   future data without crashing; unseen ⇒ same semantics as a null threshold.
   Costs for such segments still resolve via `costs_for` (default fallback;
   no default ⇒ `ContractViolation`, consistent with §3).
   **Tie inflation is expected and correct**: isotonic calibrators are step
   functions, so multiple items can sit exactly AT the threshold and the
   replayed review set can exceed the planned `n_review` (measured at seed
   42, overconfident: 1183 replayed vs 1000 planned; platt replay is exact).
   Do NOT "fix" this by capping at the planned count — thresholds are the
   published policy; the count is a consequence.
5. **Review capacity for baselines** = `n_review` of the policy strategy on
   THIS history (not the planned count). Both baselines review exactly that
   many items.
6. **Raw-confidence baseline**: rank by `score` ASC, tie-break `item_id` ASC
   (two stable sorts, same idiom as the optimizer); review the first
   `n_review`. No calibration, no costs — that is the point of the baseline.
7. **Random baseline**: `rng = np.random.default_rng(seed)`; `n_random_draws`
   times, `rng.choice(n_history, size=n_review, replace=False)`; report the
   mean/std over draws (std with `ddof=0`). No other randomness anywhere in
   the module.
8. **Per-strategy metrics** (`strategies` in §7):
   - `policy`: `captured_loss` (Σ true_loss over reviewed), `residual_loss`
     (Σ over non-reviewed), `n_review`, `review_spend` (Σ review_cost over
     reviewed), `efficiency` = `captured_loss / n_review` (null if
     `n_review == 0`), `capture_rate` = `captured_loss / total_true_loss`
     (null if total is 0), and `segments`: per-segment
     `{captured_loss, residual_loss, n_review, n_total, p_error_threshold}`
     (threshold as used, null for unseen/null). Per-segment sums use
     `math.fsum`; the strategy-level `captured_loss`/`residual_loss` are the
     SEQUENTIAL sum of per-segment values in sorted segment order (same
     tie-out convention as the optimizer §5.8).
   - `raw_confidence`: same fields minus `segments` (fsum over its review
     set; accounting identity `captured + residual == total_true_loss` must
     hold exactly by construction: compute `residual_loss` as
     `total_true_loss - captured_loss`).
   - `random`: `captured_loss_mean`, `captured_loss_std`,
     `residual_loss_mean` (= `total_true_loss - captured_loss_mean`),
     `review_spend_mean`, `n_review`, `efficiency` (mean-based),
     `capture_rate` (mean-based).
9. `total_true_loss` = `math.fsum` of true_loss over ALL history rows;
   `expected` = the replayed policy's `totals`
   `{expected_captured_loss, expected_residual_loss}` copied verbatim.
   For the policy strategy the identity `captured_loss + residual_loss ==
   total_true_loss` is NOT guaranteed at ULP level (two independent fsums);
   define `residual_loss = total_true_loss - captured_loss` for it as well,
   with the per-segment table remaining exact fsums (the tie-out rule in 8
   applies to `captured_loss` only; document this in the module).
10. Everything float64; no wall-clock, locale, or dict-order dependence.

## 10. Repo conventions

- `src/` layout, `uv` managed, Python `>=3.12`.
- ruff: default rules + `I` (isort); line length 100. mypy: `strict = true`
  for `src/triage/`.
- Tests live in `tests/`, plain pytest; hypothesis allowed where it adds value
  (with fixed `derandomize=True` profile or explicit seeds — CI must be
  deterministic).
- File ownership per work package is exclusive: an agent touches ONLY its
  assigned files (plus nothing in `docs/`). Shared helpers live in
  `artifacts.py` / `exceptions.py`, created in Stage 0.
