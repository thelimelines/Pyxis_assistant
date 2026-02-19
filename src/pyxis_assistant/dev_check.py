"""Development checks wrapper for linting and static type analysis."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence


def _run_command(command: Sequence[str]) -> int:
    print(f"\n$ {' '.join(command)}")
    try:
        result = subprocess.run(command, check=False)
        return result.returncode
    except FileNotFoundError:
        print(f"Command not found: {command[0]}")
        return 127


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run project linting and static type checks.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply Ruff formatting and auto-fixes before type checks.",
    )
    args = parser.parse_args()

    commands: list[list[str]] = []
    if args.fix:
        commands.append(["ruff", "format", "."])
        commands.append(["ruff", "check", ".", "--fix"])
    else:
        commands.append(["ruff", "check", "."])

    commands.append(["pyright", "."])
    commands.append(["mypy", "."])

    failed = False
    for command in commands:
        if _run_command(command) != 0:
            failed = True

    if args.fix:
        print("\nNote: --fix applies to Ruff only; pyright and mypy are read-only checks.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
