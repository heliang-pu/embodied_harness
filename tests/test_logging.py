import json
import logging

from embodied_harness.runtime.logs.codes import ErrorCode
from embodied_harness.runtime.logs.context import clear_context, scoped_context, update_context
from embodied_harness.runtime.logs.setup import (
    ROOT_LOGGER,
    ConsoleFormatter,
    close_file_log,
    configure_logging,
    open_file_log,
)


def test_file_log_lines_carry_context_code_and_traceback(tmp_path):
    path = tmp_path / "episode.log"
    handler = open_file_log(path)
    logger = logging.getLogger(f"{ROOT_LOGGER}.test")
    try:
        clear_context()
        update_context(episode_id="ep_0042", state="EXECUTE", t=1.5)
        with scoped_context(skill_call_id="sc_0007"):
            logger.warning("gripper slipped", extra={"code": ErrorCode.E_EXEC_UNDERRUN})
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("tick failed")
    finally:
        close_file_log(handler)
        clear_context()
    first, second = (json.loads(line) for line in path.read_text().splitlines())
    assert first["episode_id"] == "ep_0042"
    assert first["skill_call_id"] == "sc_0007"
    assert first["state"] == "EXECUTE" and first["t"] == 1.5
    assert first["code"] == "E_EXEC_UNDERRUN"
    assert "skill_call_id" not in second
    assert "ValueError: boom" in second["exc"]


def test_configure_logging_is_idempotent():
    logger = logging.getLogger(ROOT_LOGGER)
    before = list(logger.handlers)
    try:
        configure_logging("WARNING")
        configure_logging("INFO")
        consoles = [h for h in logger.handlers if getattr(h, "embodied_harness_console", False)]
        assert len(consoles) == 1
        assert consoles[0].level == logging.INFO
    finally:
        for handler in list(logger.handlers):
            if handler not in before:
                logger.removeHandler(handler)
        logger.propagate = True


def test_console_line_shows_episode_state_and_code():
    record = logging.LogRecord(
        f"{ROOT_LOGGER}.supervisor", logging.ERROR, __file__, 1, "HOLD: %s", ("stuck",), None
    )
    record.episode_id, record.t, record.state, record.skill_call_id = "ep_0001", 12.5, "HOLD", "sc_0009"
    record.code = ErrorCode.E_BUDGET
    line = ConsoleFormatter().format(record)
    assert line.startswith("ep_0001")
    assert "12.50s" in line and "HOLD" in line and "sc_0009" in line
    assert line.endswith("supervisor: HOLD: stuck [E_BUDGET]")
