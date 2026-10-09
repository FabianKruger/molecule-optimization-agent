"""Smoke tests for the seismo package name and the deprecated aliases.

Run from the pixi environment, where the console scripts are installed:

    pixi run pytest
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

BIN = Path(sys.executable).parent
UI_PORT = 7860
UI_START_TIMEOUT_S = 180

_OLD_IMPORT_WARNING = "molopt_agent is deprecated; use seismo instead"
_MOLOPT_NOTICE = "molopt is deprecated; use seismo instead"
_MOLOPT_UI_NOTICE = "molopt-ui is deprecated; use seismo-ui instead"


def _script(name: str) -> Path:
    path = BIN / name
    if not path.is_file():
        pytest.skip(f"{path} is not installed; run these tests with pixi")
    return path


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_script(script)), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def _wait_until_free(port: int, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _listening(port):
            return
        time.sleep(0.2)
    raise AssertionError(f"127.0.0.1:{port} is still in use")


def _collect(stream: Iterator[str] | None, sink: list[str]) -> None:
    if stream is None:
        return
    for line in stream:
        sink.append(line)


def _stderr_until(lines: list[str], notice: str | None, timeout: float = 5) -> str:
    deadline = time.monotonic() + timeout
    while True:
        stderr = "".join(lines)
        if notice is None or notice in stderr:
            return stderr
        if time.monotonic() >= deadline:
            return stderr
        time.sleep(0.1)


def _ui_output(
    script: str,
    returncode: int | None,
    stdout_lines: list[str],
    stderr_lines: list[str],
    reason: str,
) -> str:
    return (
        f"{script} {reason} (returncode={returncode})\n"
        f"stdout:\n{''.join(stdout_lines)}\n"
        f"stderr:\n{''.join(stderr_lines)}"
    )


def test_seismo_import_does_not_warn() -> None:
    result = subprocess.run(
        [sys.executable, "-W", "always", "-c", "import seismo"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "deprecated" not in result.stderr


def test_molopt_agent_import_warns_and_aliases_seismo() -> None:
    code = f"""
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import molopt_agent.oracles.base as old
    import seismo.oracles.base as new
    assert old is new
    messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, FutureWarning)
    ]
    assert messages == [{_OLD_IMPORT_WARNING!r}], messages
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize(
    ("script", "notice"),
    [
        ("seismo", None),
        ("molopt", _MOLOPT_NOTICE),
    ],
)
def test_cli_help(script: str, notice: str | None) -> None:
    result = _run(script, "--help")
    assert result.returncode == 0, result.stderr
    assert f"usage: {script}" in result.stdout
    if notice is None:
        assert "deprecated" not in result.stderr
    else:
        assert notice in result.stderr


@pytest.mark.parametrize(
    ("script", "notice"),
    [
        ("seismo-ui", None),
        ("molopt-ui", _MOLOPT_UI_NOTICE),
    ],
)
def test_ui_starts(script: str, notice: str | None) -> None:
    path = _script(script)
    _wait_until_free(UI_PORT)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GRADIO_ANALYTICS_ENABLED"] = "False"
    proc = subprocess.Popen(
        [str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    threads = [
        threading.Thread(
            target=_collect, args=(proc.stdout, stdout_lines), daemon=True
        ),
        threading.Thread(
            target=_collect, args=(proc.stderr, stderr_lines), daemon=True
        ),
    ]
    for thread in threads:
        thread.start()
    try:
        deadline = time.monotonic() + UI_START_TIMEOUT_S
        while time.monotonic() < deadline:
            if _listening(UI_PORT):
                break
            if proc.poll() is not None:
                pytest.fail(
                    _ui_output(
                        script,
                        proc.returncode,
                        stdout_lines,
                        stderr_lines,
                        "exited before listening",
                    )
                )
            time.sleep(0.25)
        else:
            pytest.fail(
                _ui_output(
                    script,
                    proc.poll(),
                    stdout_lines,
                    stderr_lines,
                    "timed out before listening",
                )
            )

        stderr = _stderr_until(stderr_lines, notice)
        if notice is None:
            assert "deprecated" not in stderr
        else:
            assert notice in stderr
    finally:
        _stop(proc)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        for thread in threads:
            thread.join(timeout=2)
        _wait_until_free(UI_PORT)
