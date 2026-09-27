import logging
import sys
import threading

import pytest

from conftest import RUN_CONFIG, write_configs
from fmc3_harness.cli import main
from fmc3_harness.logs.setup import ROOT_LOGGER


@pytest.fixture
def isolated_logging(monkeypatch):
    """`run` installs crash handlers and a console handler; undo both after the test."""
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    logger = logging.getLogger(ROOT_LOGGER)
    before = list(logger.handlers)
    yield
    for handler in list(logger.handlers):
        if handler not in before:
            logger.removeHandler(handler)
            handler.close()
    logger.propagate = True


def test_validate_ok(capsys):
    assert main(["validate", str(RUN_CONFIG)]) == 0
    assert capsys.readouterr().out.startswith("OK:")


def test_validate_reports_problems(tmp_path, capsys):
    path = write_configs(tmp_path, skills={"skills": {"pick": {"prompt": "nope"}}})
    assert main(["validate", str(path)]) == 2
    err = capsys.readouterr().err
    assert "E_CONFIG" in err
    assert "'nope' is not in prompt library" in err


def test_run_writes_traces(tmp_path, monkeypatch, capsys, isolated_logging):
    monkeypatch.chdir(tmp_path)
    args = ["run", str(RUN_CONFIG), "--episodes", "2", "--seed", "3", "--run-id", "t"]
    code = main([*args, "--hold-policy", "abort", "--log-level", "ERROR"])
    out = capsys.readouterr().out
    assert code == 0, out
    run_dir = tmp_path / "runs" / "t"
    assert (run_dir / "run.log").is_file()
    for episode in ("ep_0001", "ep_0002"):
        assert (run_dir / episode / "outcome.json").is_file()
        assert f"{episode}: done" in out
