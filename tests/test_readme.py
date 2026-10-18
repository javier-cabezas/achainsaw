"""
Runs the `achainsaw` commands in README.md's console blocks and checks that they print what the
README shows, so its examples stay true. JSON output is compared as data, without the fields
that vary from run to run (timings, thread count).

Needs the CLI built (`cargo build -p achainsaw`); run from anywhere.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from typing import Any

ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXE_NAME: str = "achainsaw.exe" if sys.platform == "win32" else "achainsaw"
EXE_PATH: str = os.path.join(ROOT, "target", "debug", EXE_NAME)
if not os.path.exists(EXE_PATH):
    EXE_PATH = os.path.join(ROOT, "target", "release", EXE_NAME)

# Fields that change between runs and machines.
VOLATILE: set[str] = {"compile_time_us", "exec_time_us", "parse_time_us", "total_time_ms",
                      "threads"}


def console_commands(markdown: str) -> list[tuple[str, str]]:
    """Each `$ achainsaw ...` command of the console blocks, with the output that follows it."""
    found: list[tuple[str, str]] = []
    for block in re.findall(r"```console\n(.*?)```", markdown, re.S):
        command: str | None = None
        output: list[str] = []
        for line in block.splitlines():
            if line.startswith("$ "):
                if command is not None:
                    found.append((command, "\n".join(output)))
                command = line[2:]
                output = []
            else:
                output.append(line)
        if command is not None:
            found.append((command, "\n".join(output)))
    return [(c, o) for c, o in found if c.startswith("achainsaw ")]


def without_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: without_volatile(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [without_volatile(v) for v in value]
    return value


def test_readme_console_output() -> None:
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        commands: list[tuple[str, str]] = console_commands(f.read())
    assert len(commands) >= 3, commands
    for command, expected in commands:
        args: list[str] = shlex.split(command)
        result: subprocess.CompletedProcess[str] = subprocess.run(
            [EXE_PATH] + args[1:], cwd=ROOT, capture_output=True, text=True)
        actual: str = result.stdout.strip()
        try:
            got: Any = without_volatile(json.loads(actual))
            want: Any = without_volatile(json.loads(expected))
        except json.JSONDecodeError:
            got = actual
            want = expected.strip()
        assert got == want, f"{command}\nREADME shows:\n{expected}\nbut it prints:\n{actual}"
        print(f"ok: {command}")


if __name__ == "__main__":
    test_readme_console_output()
    print("README console output matches")
