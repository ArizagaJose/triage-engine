"""Typed exceptions for the triage engine."""


class TriageError(Exception):
    """Base for all typed errors."""


class ContractViolation(TriageError):
    """Invalid input data or invalid cost YAML. CLI exit code 1."""


class ResidualRiskExceeded(TriageError):
    """Residual risk above gates.max_residual_risk. CLI exit code 2."""


class CalibrationGateFailed(TriageError):
    """ECE above the declared maximum. CLI exit code 3."""
