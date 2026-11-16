"""Typed exceptions for the triage engine."""


class TriageError(Exception):
    """Base for all typed errors."""


class ContractViolation(TriageError):
    """Invalid input data or invalid cost YAML. CLI exit code 1."""


class ResidualRiskExceeded(TriageError):
    """Residual risk above gates.max_residual_risk. CLI exit code 2."""


class CalibrationGateFailed(TriageError):
    """ECE above the declared maximum. CLI exit code 3."""


class BudgetOverrunExceeded(TriageError):
    """Replayed review capacity above --max-budget-overrun. CLI exit code 5."""


class CandidateRegression(TriageError):
    """`compare --fail-on-regression`: the candidate policy is significantly
    worse than the current one (bootstrap CI, docs/interfaces.md §14). CLI
    exit code 6."""
