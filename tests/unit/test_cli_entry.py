import pytest

def test_cli_imports_and_app_exists():
    from sky_v1.cli.main import app
    assert callable(getattr(app, "__call__", None)) or hasattr(app, "command")

def test_cli_help_runs():
    from typer.testing import CliRunner
    from sky_v1.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0 or "Usage" in result.output
