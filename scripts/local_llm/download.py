#!/usr/bin/env python
"""Download one local-LLM weight snapshot.

The snapshot is fetched by the vllm pixi environment's Hugging Face client
(``huggingface_hub.snapshot_download``). ``HF_HOME`` is the weights directory.
Pass it as ``--weights-dir`` or ``MOLOPT_LLM_WEIGHTS_DIR``; the flag wins.
A snapshot that is already complete is left in place and this script exits 0.
This script does not start vLLM and does not need GPUs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from profiles import ProfileError, resolve_profile


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a local LLM snapshot into an absolute weights directory "
            "used as HF_HOME."
        )
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Profile name (qwen3-32b or deepseek-v3.2). "
            "Overrides MOLOPT_LLM_PROFILE. There is no default."
        ),
    )
    parser.add_argument(
        "--weights-dir",
        default=None,
        help=(
            "Absolute path used as HF_HOME. Must not be inside $HOME. "
            "Overrides MOLOPT_LLM_WEIGHTS_DIR. There is no default."
        ),
    )
    parser.add_argument(
        "--load-check",
        action="store_true",
        help="Accepted, but not implemented until session 2.",
    )
    return parser.parse_args(argv)


def resolve_weights_dir(raw: str, *, source: str) -> Path:
    """Return the absolute directory to use as ``HF_HOME``.

    Relative paths are rejected. A path is inside ``$HOME`` when its
    normalized location is that directory or a descendant, including a path
    that reaches ``$HOME`` through ``..`` or a symlink.
    """
    if not os.path.isabs(raw):
        raise ValueError(f"{source} must be an absolute path, got {raw!r}")

    resolved = Path(raw).resolve(strict=False)
    home_raw = os.environ.get("HOME")
    if home_raw:
        home = Path(home_raw).resolve(strict=False)
        if resolved == home or home in resolved.parents:
            raise ValueError(f"{source} {raw!r} is inside $HOME ({home_raw})")
    return resolved


def resolve_weights_setting(cli_value: str | None) -> Path:
    """``--weights-dir`` wins over ``MOLOPT_LLM_WEIGHTS_DIR``.

    If neither is set, raise ``ValueError``. There is no default directory.
    """
    if cli_value is not None:
        raw = cli_value.strip()
        if raw == "":
            raise ValueError("--weights-dir is empty")
        source = "--weights-dir"
    else:
        env = os.environ.get("MOLOPT_LLM_WEIGHTS_DIR")
        if env is None or env.strip() == "":
            raise ValueError("set --weights-dir or MOLOPT_LLM_WEIGHTS_DIR")
        raw = env.strip()
        source = "MOLOPT_LLM_WEIGHTS_DIR"
    return resolve_weights_dir(raw, source=source)


def download_snapshot(model_id: str, weights_dir: Path) -> int:
    """Download ``model_id`` with the vllm environment's Hugging Face client.

    ``snapshot_download`` is idempotent: when the snapshot is already complete
    it returns without fetching again, and this function returns 0.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HF_HOME"] = str(weights_dir)
    # huggingface_hub prefers HF_HUB_CACHE over HF_HOME. Pin both at this
    # directory so an inherited home-directory cache cannot receive the files.
    hub_cache = str(weights_dir / "hub")
    env["HF_HUB_CACHE"] = hub_cache
    env["HUGGINGFACE_HUB_CACHE"] = hub_cache

    command = [
        "pixi",
        "run",
        "--environment",
        "vllm",
        "python",
        "-c",
        (
            "import sys\n"
            "from huggingface_hub import snapshot_download\n"
            "snapshot_download(repo_id=sys.argv[1])\n"
        ),
        model_id,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            check=False,
        )
    except FileNotFoundError:
        print("error: pixi not found on PATH", file=sys.stderr)
        return 127
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        profile = resolve_profile(args.profile)
    except ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        weights_dir = resolve_weights_setting(args.weights_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.load_check:
        print("error: --load-check is not implemented", file=sys.stderr)
        return 1

    try:
        weights_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create --weights-dir {weights_dir}: {exc}", file=sys.stderr)
        return 1

    status = download_snapshot(profile.model_id, weights_dir)
    if status != 0:
        return status
    print(f"snapshot ready: {profile.model_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
