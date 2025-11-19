import argparse
import os
import subprocess
import sys
from typing import List, Tuple


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run transcribe_dataset.py across multiple shards in parallel."
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Total number of shards/processes to run in parallel.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument(
        "--script",
        type=str,
        default=None,
        help="Path to transcribe_dataset.py (default: resolve next to this file).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and exit without running.",
    )
    # Capture remaining args to forward to the underlying script
    args, forward_args = parser.parse_known_args()
    return args, forward_args


def resolve_script_path(user_path: str | None) -> str:
    if user_path:
        return user_path
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "transcribe_dataset.py")


def build_command(
    python_executable: str,
    script_path: str,
    shard_index: int,
    num_shards: int,
    forward_args: List[str],
) -> List[str]:
    return [
        python_executable,
        script_path,
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
        *forward_args,
    ]


def main() -> None:
    args, forward_args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0.")

    script_path = resolve_script_path(args.script)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"transcribe script not found: {script_path}")

    commands: List[List[str]] = []
    for shard_idx in range(args.num_shards):
        cmd = build_command(args.python, script_path, shard_idx, args.num_shards, forward_args)
        commands.append(cmd)

    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return

    processes: List[Tuple[int, subprocess.Popen]] = []
    for shard_idx, cmd in enumerate(commands):
        proc = subprocess.Popen(cmd)
        processes.append((shard_idx, proc))

    failures: List[Tuple[int, int]] = []
    for shard_idx, proc in processes:
        retcode = proc.wait()
        if retcode != 0:
            failures.append((shard_idx, retcode))

    if failures:
        for shard_idx, code in failures:
            print(f"Shard {shard_idx} failed with exit code {code}")
        sys.exit(1)
    else:
        print("All shards completed successfully.")


if __name__ == "__main__":
    main()

