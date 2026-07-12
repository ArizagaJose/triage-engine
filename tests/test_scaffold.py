"""Scaffold tests: import checks and version verification."""


def test_import_all_modules() -> None:
    """Test that all triage modules can be imported."""
    import triage  # noqa: F401
    import triage.artifacts  # noqa: F401
    import triage.calibration  # noqa: F401
    import triage.cli  # noqa: F401
    import triage.contract  # noqa: F401
    import triage.costs  # noqa: F401
    import triage.exceptions  # noqa: F401
    import triage.optimizer  # noqa: F401
    import triage.synth  # noqa: F401


def test_version() -> None:
    """Test that triage version is correct."""
    import triage

    assert triage.__version__ == "0.1.0"
