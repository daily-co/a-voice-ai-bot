#!/usr/bin/env python3
"""Run each eval scenario N times against a running bot; print an ASCII summary.

Usage (from server/, with the bot already running on port 7860):

    uv run python evals/run_eval_matrix.py
    uv run python evals/run_eval_matrix.py --runs 3 evals/release/scenarios/simulated/update_address.yaml
"""

import argparse
import re
import socket
import subprocess
import sys
import time

DEFAULT_FILE = "evals/release/scenarios/simulated/update_address.yaml"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RESULT = re.compile(r"([✓✗])\s+\S+\s+(\S+?/\S+)")


def git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or "?"
    except OSError:
        return "?"


def bot_reachable(host: str = "localhost", port: int = 7860) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def run_once(scenario_file: str, verbose: bool) -> tuple[dict[str, bool], list[str]]:
    """One `pipecat eval run`: {scenario: passed} and its failure bullet lines.

    With ``verbose``, streams the eval CLI's output live, indented.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "pipecat.evals", "run", scenario_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = ANSI.sub("", raw.rstrip("\n"))
        lines.append(line)
        if verbose and line.strip():
            print(f"    | {line.strip()}", flush=True)
    proc.wait()
    text = "\n".join(lines)
    results: dict[str, bool] = {}
    failures: list[str] = []
    scenario = None
    for line in text.splitlines():
        line = line.strip()
        m = RESULT.search(line)
        if m:
            scenario = m.group(2).split("/")[-1]
            results[scenario] = m.group(1) == "✓"
            continue
        if line.startswith("•") and scenario:
            failures.append(f"{scenario}: {line.lstrip('• ').strip()}")
    if not results:
        failures.append(f"no results parsed; last output: {text.strip().splitlines()[-1:]}")
    return results, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_file", nargs="?", default=DEFAULT_FILE)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="stream each eval run's output live"
    )
    args = parser.parse_args()

    if not bot_reachable():
        print("error: no bot on localhost:7860 - start it first: uv run bot.py")
        return 2

    branch = git_branch()
    print(f"branch: {branch} | file: {args.scenario_file} | runs: {args.runs}")

    table: dict[str, list[bool]] = {}
    all_failures: list[str] = []
    started = time.monotonic()
    for i in range(args.runs):
        print(f"run {i + 1}/{args.runs}: starting ...", flush=True)
        run_started = time.monotonic()
        results, failures = run_once(args.scenario_file, args.verbose)
        for name, passed in results.items():
            table.setdefault(name, []).append(passed)
        all_failures += [f"run {i + 1} {f}" for f in failures]
        run_elapsed = time.monotonic() - run_started
        summary = ", ".join(
            f"{name} {'PASS' if passed else 'FAIL'}" for name, passed in sorted(results.items())
        )
        print(f"run {i + 1}/{args.runs}: {summary or 'no results'} ({run_elapsed:.0f}s)", flush=True)
        for failure in failures:
            print(f"    {failure}", flush=True)
    elapsed = time.monotonic() - started
    print(f"\nall runs done in {elapsed:.0f}s\n")

    name_w = max([len(n) for n in table] + [len("scenario")])
    run_cols = "  ".join(f"run {i + 1}" for i in range(args.runs))
    header = f"{'scenario':<{name_w}}  {run_cols}  passed"
    print(header)
    print("-" * len(header))
    worst = 0
    for name in sorted(table):
        outcomes = table[name]
        cells = "  ".join(f"{'PASS' if p else 'FAIL':<5}" for p in outcomes)
        print(f"{name:<{name_w}}  {cells} {sum(outcomes)}/{len(outcomes)}")
        worst = max(worst, outcomes.count(False))
    if all_failures:
        print("\nfailures:")
        for f in all_failures:
            print(f"  {f}")
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())
