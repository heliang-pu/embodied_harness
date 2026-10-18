"""`embodied-harness suite`: a composition benchmark over run configs, and `data coverage`."""

import json

import pytest
import yaml
from pydantic import ValidationError

from conftest import CONFIGS
from embodied_harness.cli import main
from embodied_harness.config.schema import SuiteFile
from embodied_harness.runtime.suite import format_suite, load_suite, run_suite, summarize_suite

SUITE = CONFIGS / "suites" / "mock_sort_composition.yaml"


def test_the_example_suite_runs_and_summarizes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # results go to runs/suites/ under the working directory
    assert main(["suite", "run", str(SUITE), "--run-id", "t1"]) == 0
    printed = capsys.readouterr().out
    run_dir = tmp_path / "runs" / "suites" / "mock_sort_composition" / "t1"
    assert "| one_package | A | seen | 3/3 (100%) | - |" in printed
    assert "| C | - | 3/3 (100%) |" in printed
    # The one-package entries never flip, so flips are transitions only the unseen entries exercise.
    assert "transitions only unseen entries exercise:" in printed and "| pick -> flip |" in printed
    entry = run_dir / "three_packages_flip"
    assert sorted(p.name for p in entry.glob("ep_*")) == ["ep_0001", "ep_0002", "ep_0003"]
    run_config = yaml.safe_load((entry / "run_config.yaml").read_text(encoding="utf-8"))
    assert run_config["env"]["params"]["num_packages"] == 3 and run_config["task"] == str(entry / "task.yaml")
    assert yaml.safe_load((entry / "task.yaml").read_text(encoding="utf-8"))["task"]["expected_total"] == 3
    summary = json.loads((run_dir / "suite.json").read_text(encoding="utf-8"))
    assert [e["rate"] for e in summary["entries"]] == [1.0, 1.0, 1.0, 1.0]

    # `suite report` rebuilds the same summary from the runs.
    assert main(["suite", "report", str(SUITE), str(run_dir)]) == 0
    assert "| one_package | A | seen | 3/3 (100%) | - |" in capsys.readouterr().out


def test_only_some_entries(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["suite", "run", str(SUITE), "--run-id", "t2", "--only", "one_package"]) == 0
    suite = load_suite(SUITE)
    summaries = summarize_suite(suite, tmp_path / "runs" / "suites" / "mock_sort_composition" / "t2")
    assert [(s.name, s.episodes) for s in summaries] == [
        ("one_package", 3),
        ("one_package_faults", 0),
        ("three_packages_flip", 0),
        ("three_packages_flip_faults", 0),
    ]
    assert "| one_package_faults | AP | seen | - | - |" in format_suite(suite, summaries)
    assert main(["suite", "run", str(SUITE), "--run-id", "t3", "--only", "nothing"]) == 2


def test_rerunning_a_suite_run_id_refuses_before_overwriting_entry_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["suite", "run", str(SUITE), "--run-id", "same", "--only", "one_package"]) == 0
    entry = tmp_path / "runs" / "suites" / "mock_sort_composition" / "same" / "one_package"
    original = yaml.safe_load((entry / "run_config.yaml").read_text(encoding="utf-8"))
    original_task = (entry / "task.yaml").read_text(encoding="utf-8")
    suite = load_suite(SUITE)
    changed = suite.entries[0].model_copy(update={"task_overrides": {"expected_total": 99}})
    with pytest.raises(FileExistsError, match="already has episode traces"):
        run_suite(SUITE, suite.model_copy(update={"entries": [changed]}), entry.parent)
    assert (entry / "task.yaml").read_text(encoding="utf-8") == original_task
    assert main(["suite", "run", str(SUITE), "--run-id", "same", "--only", "one_package"]) == 2
    assert yaml.safe_load((entry / "run_config.yaml").read_text(encoding="utf-8")) == original


def test_suite_files_are_checked():
    entry = {"name": "a", "config": "x.yaml", "category": "A"}
    with pytest.raises(ValidationError, match="duplicate entry names: a"):
        SuiteFile.model_validate({"suite": "s", "entries": [entry, entry]})
    with pytest.raises(ValidationError):
        SuiteFile.model_validate({"suite": "s", "entries": [{**entry, "category": "B"}]})
