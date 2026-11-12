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

```python
def validate_seed(seed: int) -> None
```

Seeds feed `np.random.default_rng`, which only accepts non-negative
integers. `validate_seed` raises `ContractViolation` (exit 1) for anything
else. It is called by `synth.generate`, `calibration.fit` and `backtest.run`,
and by the CLI preflights (§8). A negative `--seed` used to be exit 4.

## 3. Cost model (`src/triage/costs.py`)

Pydantic v2 models, strict (`model_config = ConfigDict(extra="forbid")`):

```python
class Budget(BaseModel):
    type: Literal["fraction", "count", "money"]
    value: float                      # finite, > 0; if fraction: <= 1.0

class Gates(BaseModel):
    max_residual_risk: float | None = None   # >= 0 (NaN rejected; inf = never fails)
    max_ece: float | None = None             # in (0, 1]; NaN rejected
    ece_gate_mode: Literal["point", "ci_upper"] = "point"   # see §4 Gate

class SegmentCosts(BaseModel):
    false_accept_cost: float          # finite, >= 0
    review_cost: float                # finite, >= 0

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

Non-finite numbers: every numeric field is validated with a check NaN
cannot pass (NaN fails every comparison). Otherwise NaN/inf either crashed
later with exit 4 (a budget in `floor()`/`int()`, or a cost in the JSON
writer, which forbids NaN/inf) or silently disabled a gate
(`max_residual_risk`, `max_ece`). All of these are `ContractViolation`
(exit 1), whether they come from the YAML, `plan --budget`, or `run`'s
overrides.

`Gates` pinned rules (v1.x, docs/risk-register.md R3):

- `max_ece` is validated as `0 < max_ece <= 1`, so NaN is rejected. A NaN
  threshold would otherwise pass validation and silently disable the gate.
- `ece_gate_mode` other than `"point"` with `max_ece` null/absent is a
  validation error. A stricter mode with no threshold gates nothing.
- **Serialization**: `Gates.model_dump()` OMITS `ece_gate_mode` when it
  equals the default `"point"`. `policy.config_snapshot` is
  `CostModel.model_dump()` and feeds `policy_hash`, so default-mode
  snapshots stay byte-identical to those written before the field existed.
  Every existing `policy_hash` and the committed golden fixtures are
  unchanged. An absent key reads back as `"point"`. A non-default mode is
  always present in the snapshot.

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
  the holdout. Never use sklearn's splitters. Both splits MUST be non-empty:
  if rounding yields `n_holdout == 0` or `n_calibration == 0` (tiny golden
  set or extreme `holdout_fraction`), `fit` raises `ContractViolation`
  (exit 1) naming the sizes — an empty holdout would report `ece_holdout=0.0`
  and pass the ECE gate with zero validation evidence.
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
  `check_ece_gate(calibrator, max_ece: float, mode: str = "point") -> None`
  raises `CalibrationGateFailed` (exit 3) if the gated statistic is
  `> max_ece` (strict: equality passes). The statistic is chosen by the mode:
  - `"point"` (default, the original semantics): `ece_holdout`.
  - `"ci_upper"` (v1.x, opt-in): `ece_ci[1]`, the upper bound of the
    bootstrap CI above (at `config.ci_level`, default 95%). It tightens
    automatically as the holdout shrinks (risk-register R3).

  An unknown mode raises `ContractViolation` (exit 1). `ece_gate_statistic(
  calibrator, mode) -> EceGateStatistic(name, description, value)` exposes
  the checked value for messages. The failure message names the mode, the
  checked field and value, `max_ece`, and, for context, `ece_holdout` and
  `ece_ci`. The mode is a gate option, not a calibration output, so it is
  NOT serialized into `calibrator.json`. The calibrator's `content_hash`
  (and every `policy_hash` built on it) is independent of how it was gated.

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
`config_snapshot` (the full cost YAML as parsed; `gates.ece_gate_mode` is
present only when non-default, see §3), `created_at`,
`input_sha256s` (`{batch, calibrator, costs}`), `policy_hash`, `segments`
(name → `{expected_captured_loss, expected_residual_loss, n_review, n_total,
p_error_threshold}`), `tool_version`, `totals` (`{expected_captured_loss,
expected_residual_loss, n_review, n_total, review_spend}`).

`policy_hash = content_hash(policy_dict)`. Floats are written as-is (Python
repr via json); never round for hashing.

**backtest.json** (v1) top-level keys: `backtest_hash`, `created_at`,
`curves` (see §9.11 — capacity-sweep capture curves; this is the
coverage-vs-risk data the report and the README chart render),
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
triage calibrate --golden data/golden.parquet --seed 42 [--max-ece F] \
            [--ece-gate-mode point|ci_upper] -o artifacts/calibrator.json
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
  `gates.max_residual_risk` from the policy's `config_snapshot`, then
  `check_budget_overrun_gate` (exit 5) if `--max-budget-overrun` was given.
  This is the CI drift gate: `plan` can pass on expected numbers while
  `backtest` fails on realized ones.
- `--max-budget-overrun FACTOR` (optional, float, no default): budget
  overrun gate, exit 5 (`BudgetOverrunExceeded`). Deliberately a CLI flag,
  NOT a `gates:` YAML field — a new `Gates` field would change
  `config_snapshot` in every policy.json and hence every `policy_hash`,
  breaking golden artifacts (see docs/risk-register.md R1). A non-finite
  FACTOR (NaN/inf — float parsers accept these spellings) or `FACTOR < 1.0`
  is a `ContractViolation` (exit 1); NaN in particular would otherwise
  silently disable the gate (`nan < 1.0` and `actual > nan` are both
  False). Pinned formula, from the policy's
  `config_snapshot.budget` applied to THIS history: allowed capacity =
  `value * n_history` (fraction, un-floored) or `value` (count), compared
  against replayed `strategies.policy.n_review`; for money budgets,
  allowed = `value` compared against replayed
  `strategies.policy.review_spend`. Fail iff `actual > FACTOR * allowed`
  (strict). The replay itself is NEVER capped (§9.4) — the gate only
  refuses to bless the result.
- Success prints a one-line summary (n_review, captured, residual,
  backtest_hash prefix, output path) and exits 0.

```
triage report --from artifacts/ -o report.html
```

`report` pinned behavior (v1 part 2):

- `--from DIR`: reads `DIR/calibrator.json` and `DIR/policy.json`
  (REQUIRED — missing or unparseable is `ContractViolation`, exit 1) and
  `DIR/backtest.json` (OPTIONAL — absent file ⇒ `backtest_dict=None`;
  present-but-unparseable is still exit 1).
- Chain verification happens in `report.render` (§11.2); the CLI just
  loads, calls `render(..., created_at=utc_now_iso(),
  tool_version=__version__)`, writes the HTML (creating parent dirs like
  `write_artifact` does), and prints a one-line summary (which artifacts
  were included + output path). No gates beyond exit 1.

- **Option preflights (single source of option validation).** Each
  command validates its options in ONE function, called before it reads or
  writes anything: `_synth_preflight` (scenario name via
  `synth.validate_scenario`, seed), `_calibrate_preflight` (seed, and the
  gate flags via `costs.Gates`), and `_backtest_preflight` (seed,
  `backtest.validate_n_random_draws`, `backtest.validate_max_budget_overrun`).
  `run` calls the same preflights with the exact arguments it later passes to
  each stage. So `run` and the individual commands accept and reject exactly
  the same inputs, with the same messages. `plan --budget` is validated by
  `costs.Budget` in both paths. Consequence for `backtest`: an invalid
  `--max-budget-overrun`, `--n-random-draws` or `--seed` is now rejected
  before the replay runs, so no `backtest.json` is written for it.
- `--seed` is required-explicit where randomness exists (`synth`, `calibrate`);
  no hidden defaults (typer default allowed but must be a literal, documented).
  Pin: default `--seed 42` on both, shown in help.
- `--budget` (plan) overrides `budget.value` from the YAML, keeping
  `budget.type`.
- `calibrate` gating: if `--max-ece` given, run `check_ece_gate` with
  `--ece-gate-mode` (default `point`) (exit 3). The artifact is written
  first. The two flags are validated BEFORE fitting, using the same
  `costs.Gates` rules as the YAML (§3): `--max-ece` in `(0, 1]` (NaN
  rejected), mode in {`point`, `ci_upper`}, and `ci_upper` requires
  `--max-ece`. A violation is exit 1 and nothing is written. On success with
  a gate, the summary line appends
  `ece_gate=<mode> <checked field>=<value><=max_ece=<threshold>`.
- `plan` gating order: write policy.json, then `check_ece_gate` against
  `gates.max_ece` with `gates.ece_gate_mode` from the YAML (exit 3), then
  `check_residual_gate` (exit 2). `plan` has no `--ece-gate-mode` flag. The
  YAML is the single source, so the mode that was checked is exactly the
  one recorded in `config_snapshot`.
- `plan`'s `input_sha256s`: `batch` and `costs` are `file_sha256` of raw
  bytes; `calibrator` is `content_hash` of the PARSED artifact dict
  (semantic hash, `created_at` excluded). The calibrator file embeds its own
  timestamp, so hashing bytes would make `policy_hash` depend on when the
  calibrator was fitted rather than on what it computes, breaking golden-file
  reproducibility across independent end-to-end runs.
- Exit-code mapping in ONE place (a `main`-level try/except):
  `ContractViolation → 1`, `ResidualRiskExceeded → 2`,
  `CalibrationGateFailed → 3`, `BudgetOverrunExceeded → 5`,
  other `Exception → 4` (message to stderr,
  traceback only with `--verbose`). Success prints a one-line summary to
  stdout and exits 0.
- Pinned: the typer app runs with `standalone_mode=False`; bad CLI usage
  (unknown flag, missing option) maps to exit 1 — it is an input contract
  violation, and click's default usage-error code 2 would collide with the
  residual-risk gate meaning. `app_main` uses `sys.exit`, never `typer.Exit`
  outside the runner (it is not honored there).

### `run` — the whole chain in one command (v1.x)

```
triage run (--scenario S | --golden G --batch B --costs C [--history H]) [--seed 42]
           [--budget F] [--max-ece F] [--ece-gate-mode point|ci_upper]
           [--max-residual-risk F] [--max-budget-overrun F] [--n-random-draws 200]
           [--fail-fast] -o RUN_DIR
```

Pinned behavior:

- **Same code, same artifacts.** `run` calls the `synth`, `calibrate`,
  `plan`, `backtest`, and `report` command functions in that order. It adds
  no semantics of its own. Its artifacts equal the manual chain's (minus
  `created_at`). The golden fixtures (test-spec B/E) are reproduced by
  `run --scenario well_calibrated --seed 7`.
- **Modes** (exactly one): synthetic (`--scenario`, data written to
  `RUN_DIR/data/`, the labeled batch doubles as history), or own-data
  (`--golden --batch --costs`, optional `--history`; no history ⇒ no
  backtest stage, report without backtest). `--seed` seeds every randomized
  stage (synth, calibrate, backtest random baseline).
- **Layout**: `RUN_DIR/costs.yaml` (the effective cost model `plan` reads),
  `RUN_DIR/artifacts/{calibrator,policy,backtest}.json`,
  `RUN_DIR/report.html`, plus `RUN_DIR/data/` in synthetic mode.
- **Overrides** `--budget`, `--max-ece`, `--ece-gate-mode`,
  `--max-residual-risk` patch the source YAML's `budget.value` / `gates.*`.
  The result is validated as a `CostModel` and written to
  `RUN_DIR/costs.yaml` (sorted keys). Without overrides the source bytes are
  copied verbatim, so `input_sha256s.costs` equals the source file's hash.
  The calibrate stage is gated with the effective `gates.max_ece` /
  `gates.ece_gate_mode` (no gate if `max_ece` is null).
  `--max-budget-overrun` and `--n-random-draws` pass through to `backtest`.
- **Validation before any write** (exit 1, nothing written or removed):
  the stage preflights above (synth, calibrate with the effective gates, and
  backtest), plus run-only rules: mode exclusivity/completeness, input files
  exist, overrides valid as a `CostModel`, `--max-budget-overrun` only with
  a history, `RUN_DIR` not a file, and no input path equal to a file `run`
  writes.
- **No silent drift from the commands**: `run` refuses (internal error) to
  call a stage whose parameter set differs from the arguments it passes. A
  command option that `run` does not forward therefore fails loudly instead
  of receiving typer's placeholder default. `tests/cli/test_run.py` pins
  this at three levels: signatures, validation parity (same exit code and
  message for every shared invalid option), and execution parity (manual
  chain vs `run`: per-stage exit codes, gate messages, artifacts, report).
- **Stale outputs**: after validation, the files `run` writes (the layout
  above) are deleted if present. Other files in `RUN_DIR` are untouched.
- **Gates**: a stage's gate exception (exit 2/3/5) is recorded, and the run
  continues (artifact-first, so every artifact and the report exist). At the
  end, `run` prints a one-line summary (`run: <stage>=ok|FAIL(exit N) ...`)
  and raises the FIRST failure's exception type (stage order: calibrate,
  plan, backtest), so `app_main` exits with that code. `--fail-fast`
  re-raises the first gate failure immediately (later stages do not run).
  `ContractViolation` / internal errors always stop immediately (exit 1 / 4).
  No new exit codes.

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

def validate_n_random_draws(n_random_draws: int) -> None
    # int >= 0, else ContractViolation (called by run() and the CLI
    # preflight; a negative count used to crash in numpy, exit 4)

def validate_max_budget_overrun(max_budget_overrun: float) -> None
    # finite and >= 1.0, else ContractViolation

def check_budget_overrun_gate(result: BacktestResult, cost_model: CostModel,
                              max_budget_overrun: float | None) -> None
    # None -> no-op. Non-finite or < 1.0 FACTOR -> ContractViolation.
    # Otherwise raises
    # BudgetOverrunExceeded (exit 5) iff replayed capacity exceeds
    # FACTOR * allowed capacity — formula pinned in §8 (backtest CLI).
    # Message names actual, allowed, factor, and budget type.
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
11. **Capacity-sweep curves** (`curves` in backtest.json, v1 part 2). The
    coverage-vs-risk data for the report/README, computed on THIS history:
    - `curves.grid`: exactly `[i / 100 for i in range(51)]` (review
      fractions 0.00 … 0.50).
    - For each grid fraction `f`, capacity `k = floor(f * n_history)`.
    - `curves.policy[j]`: cumulative true-loss SHARE captured when
      reviewing the top-`k` items under the OPTIMIZER's ranking
      (`net_benefit = p_error * false_accept_cost - review_cost` DESC,
      `item_id` ASC tie-break — same two-stable-sorts idiom), i.e.
      `fsum(true_loss over top-k) / total_true_loss` (0.0 everywhere if
      `total_true_loss == 0`).
    - `curves.raw_confidence[j]`: same, ranking by `score` ASC, `item_id`
      ASC.
    - `curves.random[j]`: `k / n_history` exactly (the analytic expectation
      of uniform sampling; no Monte Carlo here).
    - NOTE, document in the module: `curves.policy` at the policy's own
      budget fraction may differ from `strategies.policy.capture_rate` —
      the curve reviews exactly `k` ranked items while the strategy replays
      published thresholds (tie inflation, §9.4). Both are correct answers
      to different questions; measured at seed 42, overconfident:
      curve 0.4420 at f=0.10 vs capture_rate 0.5029 at 1183 replayed items.

## 11. Report (`src/triage/report.py`) — v1.1 (redesign)

Revised for the report redesign ("Briefing" layout with CI-style checks; the
design exploration lives outside the repo in `report-mockups/`). This section
supersedes the v1 §11. What changed and why is listed in §11.12. Artifact
schemas (§7, §9) are **unchanged**: everything below is derivable from the
existing calibrator / policy / backtest JSON. No version bump.

Static, self-contained HTML. Audience: operations people (read the first
screen, act on the status) and the team's ML tech lead (drills into the
evidence). Guiding rule: **the answer first, the evidence second, the audit
trail last.** jinja2 + plotly only; NO network resources at render OR view
time (plotly.js embedded, no web fonts, no external stylesheets/images).

```python
def render(calibrator_dict: dict[str, Any], policy_dict: dict[str, Any],
           backtest_dict: dict[str, Any] | None, *, created_at: str,
           tool_version: str) -> str
```

Signature, CLI behavior (§8) and exit codes are unchanged.

### 11.1 Purity

`render` is a PURE function of its arguments: byte-identical HTML for
identical inputs. All variance is injected via `created_at` (the CLI passes
`utc_now_iso()`), which appears on exactly one line of the output, exactly
as `<p id="created-at" class="meta">{created_at}</p>`, so determinism tests
can strip it. No wall-clock, locale, or dict-iteration-order dependence
anywhere (all iteration over artifact dicts goes through explicit sorts).

### 11.2 Chain verification before rendering

Unchanged from v1: `ContractViolation` on failure, naming both hashes. Every
hash is RECOMPUTED from content, never trusted as declared:
`policy.input_sha256s.calibrator == content_hash(calibrator_dict)`; policy
self-check `content_hash(policy_dict) == policy.policy_hash`; and if
`backtest_dict` is given: backtest self-check
`content_hash(backtest_dict) == backtest.backtest_hash`,
`backtest.input_sha256s.policy == content_hash(policy_dict)` AND
`backtest.input_sha256s.calibrator == content_hash(calibrator_dict)`. The
report renders the chain it verified, never an unchecked one; it therefore
always shows an "artifact chain verified" badge in the header.

### 11.3 Modes

- **full** (`backtest_dict` given): all sections.
- **plan_only** (`backtest_dict is None`): the results section and the
  backtest-only figures (`fig-backtest`, `fig-coverage`, `fig-segments`) are
  omitted and a visible note is rendered in their place. The note contains
  the phrase "backtest not run" (case-insensitive). That phrase appears
  NOWHERE in a full-mode report, including check copy.

### 11.4 Page structure (document order is pinned)

The root `<main>` carries `data-status="ok|attention|failed"` and
`data-mode="full|plan_only"` (§11.6). Sections, in this order:

| # | element / id | content |
|---|--------------|---------|
| 0 | `<header>` | Brand + title, `tool_version`, the `created-at` line (§11.1), "artifact chain verified" badge. Nothing else: no tables, no figures. |
| 1 | `section#at-a-glance` | **The first content section; nothing may precede it except the header.** Status badge, headline (`<h1>`), lede paragraph, four KPI tiles (§11.7), checks list (§11.5). Contains no figures and no plotly.js. |
| 2 | `section#results` (full only) | "Is the policy worth it?": `fig-backtest`, `fig-coverage`, strategy table (policy / raw confidence / random mean ± std: loss caught, residual loss, n_review, review spend, efficiency, capture rate), the n_review parity note (all strategies review the same count), and a tie-inflation note whenever replayed `n_review` ≠ planned (explains why the table's capture rate can differ from the curve at the budget point, §9.4/§9.11). In plan_only this slot holds the "backtest not run" note (§11.3). |
| 3 | `section#segments` | "Where the risk is": `fig-segments` (full only) + per-segment table (§11.8). |
| 4 | `section#calibration` | "Can we trust the model's probabilities?": `fig-reliability`; method, seed, `n_golden`, ECE raw → calibrated, bootstrap CI labelled with `config.ci_level`. |
| 5 | `section#gates` | Heading text "Gates". The INFORMATIONAL gates table (§11.9). |
| 6 | `section#audit` | Audit trail in a `<details>` element, CLOSED by default. The `<summary>` shows the short chain (calibrator → policy → backtest, first 10 hex chars each); the body holds the full hash table: calibrator `content_hash`, `policy_hash`, `backtest_hash` if present, plus each artifact's recorded `input_sha256s`. Full hashes are always present in the DOM. |

Footer: one line stating that cost figures are in the units declared in
costs.yaml ("cost units" everywhere in copy; no currency symbol).

**plotly.js placement**: embedded exactly once, as a standalone inline
`<script>` holding `plotly.offline.get_plotlyjs()`, placed AFTER
`section#at-a-glance` and BEFORE the first figure. So the first screen
renders without waiting for the ~4.6 MB bundle to parse. Every figure is
produced with `to_html(full_html=False, include_plotlyjs=False)` and an
explicit `div_id` (§11.10). Never `include_plotlyjs="cdn"`, never a
`<script src=…>`.

### 11.5 Summary model and checks

A pure helper builds everything the at-a-glance section shows:

```python
def _build_summary(calibrator_dict: dict[str, Any], policy_dict: dict[str, Any],
                   backtest_dict: dict[str, Any] | None) -> Summary

@dataclass(frozen=True)
class Check:
    id: str                 # stable id from the catalogue below
    severity: Literal["fail", "warn", "info", "pass"]
    title: str              # plain text, one line (ops language)
    detail: str             # plain text, 1-2 sentences (may name the technical field)
    value: str              # short plain-text figure shown right-aligned
    section: str            # id of the evidence section ("gates", "segments", …)
    params: dict[str, float | int | str | None]   # raw numbers behind the copy

@dataclass(frozen=True)
class Summary:
    status: Literal["ok", "attention", "failed"]
    mode: Literal["full", "plan_only"]
    checks: tuple[Check, ...]   # ordered, see below
    kpis: dict[str, Any]        # §11.7
```

Pinned semantics:

1. **Plain text only.** `title`, `detail` and `value` are plain strings,
   autoescaped by the template. Python never builds HTML/`Markup` from
   artifact-derived data (segment names are attacker-controlled-ish, see the
   escaping tests). Emphasis, icons and links are the template's job.
2. **Tests assert on `id`, `severity`, `params`, never on copy.** Wording is
   owned by the module/template and may change without a spec change.
3. **Gate checks reuse the gates table's comparison.** Gate verdicts are
   computed once (the `_gate_row` helper, strict `>`), and both the checks
   and the gates table (§11.9) read them. The at-a-glance section can never
   contradict the gates table.
4. **Status**: `failed` if any check is `fail`; else `attention` if any is
   `warn`; else `ok`. Red appears ONLY for failed declared gates; amber is
   the "needs attention" level (drift decision (a): state the ratio
   neutrally, amber when notable, red only via declared gates).
5. **Order**: severity (`fail`, `warn`, `info`, `pass`), then catalogue
   order (the table below). Fully deterministic.
6. **Rendering**: each check is a `<details class="check">` carrying
   `data-check-id` and `data-severity`; `fail`/`warn` are rendered `open`
   (detail visible), `info`/`pass` closed. Each check links to its
   `section` anchor ("see evidence"). Evidence is NOT duplicated inline:
   figures appear once, in their section.
7. Severity is never conveyed by colour alone: every check has an icon
   glyph (✕ ! i ✓) and a text label.

**Check catalogue** (ids are stable API for tests). `R = 1.25` is the
module constant `ATTENTION_RESIDUAL_RATIO`: display-only, it never affects
exit codes; the value is provisional and changing it does not require a
spec change.

| id | mode | emitted when | severity | `section` |
|----|------|--------------|----------|-----------|
| `gate.max_ece` | both | always | `fail` iff the selected statistic (`ece_holdout` for `point`, `ece_ci[1]` for `ci_upper`) exceeds `max_ece`; `pass` otherwise; `info` if `max_ece` is null/absent ("no limit declared") | gates |
| `gate.max_residual_risk.planned` | both | `max_residual_risk` declared | `fail` iff `policy.totals.expected_residual_loss > max_residual_risk`, else `pass` | gates |
| `gate.max_residual_risk.realized` | full | `max_residual_risk` declared | `fail` iff `strategies.policy.residual_loss > max_residual_risk`, else `pass` | gates |
| `gate.max_residual_risk` | both | `max_residual_risk` null/absent | `info` ("no limit declared for unreviewed risk; CI cannot fail on it") | gates |
| `plan.vs_actual` | full | always | `warn` iff `realized > R × expected` (realized = `strategies.policy.residual_loss`, expected = `backtest.expected.expected_residual_loss`); if `expected == 0`: `warn` iff `realized > 0`; else `pass`. `params`: `expected`, `realized`, `ratio` (null if expected is 0), `worst_segment`, `worst_ratio` | segments |
| `value.vs_baselines` | full | always | `info` if `total_true_loss == 0` (nothing to catch); else `pass` iff policy `captured_loss` > random `captured_loss_mean` AND > raw-confidence `captured_loss`; else `warn`. `params`: `capture_rate`, `lift_random`, `lift_raw` (null when the baseline denominator is 0) | results |
| `calibration.vs_raw` | both | always | `pass` iff `ece_holdout <= ece_raw_holdout`, else `info` (calibration did not improve on raw scores) | calibration |
| `calibration.ci_margin` | both | point mode AND `max_ece` declared AND `ece_holdout <= max_ece < ece_ci[1]` | `info` (passes on the point estimate, but the bootstrap CI upper bound exceeds the gate; see risk-register R3) | calibration |
| `budget.adherence` | full | always | `pass` iff replayed `strategies.policy.n_review` == planned `policy.totals.n_review`; else `info`. `params`: `planned`, `replayed`, `delta` (replayed − planned), `ratio` (null if planned is 0) | results |
| `backtest.not_run` | plan_only | always | `info` | results (the plan-only note's slot) |

`worst_segment` = the segment with the largest
`realized_residual − expected_residual` over the UNION of
`policy.segments` and `strategies.policy.segments` keys (a missing side
counts as 0.0; unseen-at-plan-time segments, §9.4, therefore appear with
expected 0). Ties are broken by segment name ascending. `worst_ratio` is null
when that segment's expected residual is 0.

Why `calibration.vs_raw` is `info`, not `warn`: on a holdout of a few
hundred items, ECE differences of ~0.01 are inside the bootstrap CI. An
amber status on noise trains readers to ignore amber. The ECE gate
remains the enforceable signal.

### 11.6 Status presentation

| status | badge label (full) | badge label (plan_only) | colour |
|--------|--------------------|--------------------------|--------|
| `failed` | "Gate failed" | "Gate failed" | red |
| `attention` | "Needs attention" | "Needs attention" | amber |
| `ok` | "On track" | "Plan only — not backtested" | green (full) / neutral slate (plan_only: unvalidated is not "green") |

**Headline (`<h1>`)**: when status ≠ `ok`, the `title` of the first check;
otherwise a fixed "working as planned" sentence (full) or "plan computed;
run a backtest to validate it" (plan_only). **Lede**: one to three
sentences generated from the KPIs: items reviewed (count and share),
share of loss caught and lift vs random, and unreviewed risk, actual vs
planned (plus the declared limit when that gate fails). In plan_only, the
lede uses the expected figures and says so.

### 11.7 KPI tiles

Four tiles, in this order, each with `data-kpi`:

| `data-kpi` | label | full mode value (sub-line) | plan_only value (sub-line) |
|------------|-------|-----------------------------|-----------------------------|
| `reviewed` | Items reviewed | replayed `strategies.policy.n_review` (share of `n_total`; budget; "+N over plan (ties)" if replayed > planned) | `policy.totals.n_review` (share; budget) |
| `capture` | Loss caught | `strategies.policy.capture_rate` as % (`captured_loss` of `total_true_loss` cost units); "—" if null | expected share `expected_captured / (expected_captured + expected_residual)`, labelled "expected"; "—" if the denominator is 0 |
| `lift` | vs. random review | `captured_loss / random.captured_loss_mean` ("—" if the mean is 0) (lift vs raw confidence, same rule) | "—" ("needs a backtest") |
| `residual` | Risk left unreviewed | `strategies.policy.residual_loss` (plan expected + ratio; declared limit if any) | `policy.totals.expected_residual_loss`, labelled "expected" (declared limit if any) |

Budget display by `budget.type`: `fraction` → "budget 10%"; `count` →
"budget 1,000 items"; `money` → "budget 4,000 cost units". The `residual`
value is coloured red if a residual-risk gate check failed, amber if
`plan.vs_actual` is `warn`, neutral otherwise (never green: a number alone
is not a verdict).

### 11.8 Per-segment table

One row per segment (sorted by actual unreviewed risk DESC, then name ASC,
in full mode; by name ASC in plan_only), plus a totals row. Columns:
segment; cost per miss / per review (via `CostModel.costs_for` semantics on
`policy.config_snapshot`: exact key, else `default`); `p_error` review
threshold ("—" if null); reviewed (planned, and replayed when it differs)
of `n_total`; unreviewed risk: plan; unreviewed risk: actual (full); actual
÷ plan (full; "—" when plan is 0). The ratio cell uses amber text when it
exceeds `R`. The previous v1 expected-captured column is dropped from the
table (still in policy.json).

### 11.9 Gates table (informational)

Same rows and semantics as v1 (see docs/risk-register.md R2): declared
thresholds from `policy.config_snapshot.gates` vs measured values, verdict
PASS/FAIL with the same strict `>` as the CLI gates, "not set" for
null/absent gates. Every row names its checked artifact field. Rows: `max_ece` vs the statistic
selected by `gates.ece_gate_mode` (absent means `point`):
`calibrator.ece_holdout` for `point`, `calibrator.ece_ci[1]` for `ci_upper`.
Unknown modes display no verdict;
`max_residual_risk` vs planned `policy.totals.expected_residual_loss`;
`max_residual_risk` vs realized `strategies.policy.residual_loss` (full
only); budget adherence (full only, always informational, no verdict):
planned vs replayed `n_review` with the replayed/planned ratio ("—" if
planned is 0). Verdicts render as badges whose text is exactly `PASS`,
`FAIL` or `not set` (e.g. `<span class="badge b-pass">PASS</span>`). Purely
display: the report still exits non-zero only on contract violations ("no
gates beyond exit 1" stands).

### 11.10 Figures

All plotly, explicit `div_id`s, no fixed pixel width (the figure fills its
container; fixed height), direct labels preferred over legends:

| `div_id` | mode | content |
|----------|------|---------|
| `fig-backtest` | full | Horizontal bars: share of loss caught (`capture_rate`) for policy, random (error bar = `captured_loss_std / total_true_loss`), raw confidence; each baseline annotated with the policy's lift ("policy 4.2× better"). Replaces v1's absolute captured-loss bars. |
| `fig-coverage` | full | The three `curves` series vs `grid` (x: share of items reviewed, y: share of loss caught), line-end labels instead of a legend, vertical marker at the PLANNED budget fraction (`policy.totals.n_review / n_total`) with each curve's value at the nearest grid point annotated. If the fraction exceeds the grid (> 0.50), omit the marker. |
| `fig-segments` | full | Per segment: unreviewed risk, plan (outlined) vs actual (solid), annotated with the actual/plan ratio; same row order as §11.8. |
| `fig-reliability` | both | Before (raw score) / after (calibrated) reliability curves from `calibrator.reliability_bins` (skip `count == 0` bins), diagonal reference, marker area ∝ bin `count` (hover shows the count). |

### 11.11 Visual system, formatting, template

1. **Palette** (pinned in report.py; the report owns it):
   series: policy `#2a78d6`, raw confidence `#7c5cc4`, random `#9aa1a9`;
   status: ok `#15803d`, attention `#d97706`, failed `#dc2626`, info/neutral
   `#64748b`; ink `#1f2933`. Status colours are reserved for verdicts, and no
   data series uses green, amber or red. `scripts/headline_chart.py` still
   uses the v1 palette; aligning it (and regenerating `docs/headline.png`)
   is a follow-up, not part of this contract.
2. **Type**: system font stack only
   (`-apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`),
   monospace stack for hashes. No web fonts. All CSS is inline in the
   template.
3. **Number formatting**: locale-free Jinja filters registered on the
   environment, pure Python format specs:
   `count` → `{:,d}` (1,183); `cost` → `{:,.0f}` (336,100);
   `pct` → share × 100 with 1 decimal + "%" (41.9%); `ratio` → `{:.2f}×`
   (2.36×); `lift` → `{:.1f}×` below 10, `{:.0f}×` from 10 (4.2×, 20×);
   `ece` → `{:.3f}`. `None` renders as "—" in every filter.
4. **Template** is a package resource: `src/triage/templates/report.html.j2`,
   loaded with `jinja2.Environment(loader=PackageLoader("triage"),
   autoescape=True, keep_trailing_newline=True)`. Template partials/macros
   under `src/triage/templates/` are allowed. `pyproject.toml` must package
   them (verify the built wheel).
5. **Stable test hooks** (tests use these, not headings or copy):
   `main[data-status]`, `main[data-mode]`, the section ids of §11.4,
   `[data-check-id]` + `[data-severity]`, `[data-kpi]`, the figure
   `div_id`s, and `#created-at`.

### 11.12 Changes from v1 §11 (traceability)

- Section order: v1 opened with the hash table. Now at-a-glance → results →
  segments → calibration → gates → audit, and the hash table moves into a
  collapsed `<details>` (full hashes still in the DOM).
- NEW: summary model, checks, status, headline/lede, KPI tiles (§11.5–11.7).
- NEW: per-segment plan-vs-actual (backtest `strategies.policy.segments`
  was in the artifact but never rendered) and per-segment costs; `fig-segments`.
- `fig-backtest` now plots capture share with lift annotations instead of
  absolute captured loss; `fig-coverage` gains value annotations;
  `fig-reliability` encodes bin counts.
- plotly.js: v1 embedded it in the FIRST figure; now it is one standalone
  script after the at-a-glance section.
- Palette: raw confidence green → purple, random amber → grey (§11.11.1).
- Gates table unchanged in semantics; verdicts become badges (text
  unchanged). Its heading is no longer followed by a "Backtest" `<h2>`
  (tests that sliced `<h2>Gates</h2>…<h2>Backtest</h2>` move to `#gates`).
- Plan-only note unchanged in meaning; now placed in the results slot.

## 12. CI workflow (`.github/workflows/ci.yml`) — v1

Example workflow, also the repo's own CI. Two jobs:

- `checks`: uv sync → `ruff check` + `ruff format --check` → `mypy src/` →
  `pytest -q`. Python 3.12, `astral-sh/setup-uv`, pinned action versions.
- `gates` (needs `checks`): the exit-code demo used as a REAL gate —
  `synth drift --seed 42`, `calibrate --seed 42` (gate `--max-ece 0.05`),
  `plan`, `backtest`, `report`; then a step that asserts the documented
  exit codes with a costs.yaml carrying `max_residual_risk: 200000`:
  `plan` must exit 0 and `backtest` must exit 2 (`if triage backtest …;
  then exit 1; else test $? -eq 2; fi` idiom, set `-e`-safe). Upload
  report.html + artifacts as workflow artifacts.

## 13. Repo conventions

- `src/` layout, `uv` managed, Python `>=3.12`.
- ruff: default rules + `I` (isort); line length 100. mypy: `strict = true`
  for `src/triage/`.
- Tests live in `tests/`, plain pytest; hypothesis allowed where it adds value
  (with fixed `derandomize=True` profile or explicit seeds — CI must be
  deterministic).
- File ownership per work package is exclusive: an agent touches ONLY its
  assigned files (plus nothing in `docs/`). Shared helpers live in
  `artifacts.py` / `exceptions.py`, created in Stage 0.
