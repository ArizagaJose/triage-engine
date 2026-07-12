# interfaces.md — pinned contracts (MVP)

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

def plan(batch: pl.DataFrame, calibrator: Calibrator, cost_model: CostModel,
         *, input_sha256s: dict[str, str], tool_version: str) -> Policy
```

Pinned semantics (do not improvise):

1. `p_error = calibrator.predict(score)` per item.
2. `expected_loss = p_error * false_accept_cost(segment)`.
3. `net_benefit = expected_loss - review_cost(segment)`.
4. Sort by `net_benefit` DESC, tie-break `item_id` ASC (single stable sort).
5. Budget in review units:
   - `fraction` → `floor(value * n_batch)` items;
   - `count` → `int(value)` items;
   - `money` → walk the sorted list, admit while cumulative `review_cost <= value`.
6. Route to review while budget remains **and** `net_benefit > 0` (items with
   non-positive net benefit are never routed, even under unspent budget).
7. Per-segment `p_error_threshold` = min `p_error` among routed items of that
   segment; segments with zero routed items get threshold `null` (= review
   nothing). Applying a policy later means: review iff
   `p_error >= p_error_threshold`.
8. `expected_captured_loss` = Σ expected_loss over routed items;
   `expected_residual_loss` = Σ expected_loss over non-routed items. Both per
   segment and total.
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

Scenarios: `"well_calibrated"`, `"overconfident"`. Shared design (all
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
  `p` distribution bounds, the score formula name, noise sigma, and the
  **realized** per-segment mean error rates of golden and batch — this is what
  Stage-4 claim tests read to compute analytic expectations.

## 7. Artifact JSON — canonical form (all writers)

One shared helper module rule (put in `src/triage/artifacts.py`):

```python
def canonical_dumps(obj: dict) -> str:
    # json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n"
def content_hash(obj: dict) -> str:
    # sha256 of canonical_dumps(obj) after deleting keys
    # "created_at" and "policy_hash" (if present) at the TOP level
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

## 8. CLI (`src/triage/cli.py`)

Typer app, entry point `triage` (pyproject `[project.scripts]`).

```
triage synth --scenario overconfident --seed 42 -o data/
triage calibrate --golden data/golden.parquet --seed 42 [--max-ece F] -o artifacts/calibrator.json
triage plan --batch data/batch.parquet --calibrator artifacts/calibrator.json \
            --costs costs.yaml [--budget F] -o artifacts/policy.json
```

- `--seed` is required-explicit where randomness exists (`synth`, `calibrate`);
  no hidden defaults (typer default allowed but must be a literal, documented).
  Pin: default `--seed 42` on both, shown in help.
- `--budget` (plan) overrides `budget.value` from the YAML, keeping
  `budget.type`.
- `calibrate` gating: if `--max-ece` given, run `check_ece_gate` (exit 3) —
  artifact is written first.
- `plan` gating order: write policy.json, then `check_ece_gate` against
  `gates.max_ece` from the YAML (exit 3), then `check_residual_gate` (exit 2).
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

## 9. Repo conventions

- `src/` layout, `uv` managed, Python `>=3.12`.
- ruff: default rules + `I` (isort); line length 100. mypy: `strict = true`
  for `src/triage/`.
- Tests live in `tests/`, plain pytest; hypothesis allowed where it adds value
  (with fixed `derandomize=True` profile or explicit seeds — CI must be
  deterministic).
- File ownership per work package is exclusive: an agent touches ONLY its
  assigned files (plus nothing in `docs/`). Shared helpers live in
  `artifacts.py` / `exceptions.py`, created in Stage 0.
