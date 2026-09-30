#!/usr/bin/env python
"""Download one local-LLM weight snapshot, optionally then load-check it.

The snapshot is fetched by the vllm pixi environment's Hugging Face client
(``huggingface_hub.snapshot_download``). ``HF_HOME`` is the weights directory.
Pass it as ``--weights-dir`` or ``MOLOPT_LLM_WEIGHTS_DIR``; the flag wins.
A snapshot that is already complete is left in place and this script exits 0.

``--load-check`` is off by default. Without it, this script does not start
vLLM and does not need GPUs. With it, this script does not download: it starts
``vllm serve`` for that profile on the profile's GPUs against the weights
directory, polls ``/health`` up to the profile's health timeout, then stops
the server.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from profiles import Profile, ProfileError, resolve_profile

DEFAULT_PORT = 8000
POLL_INTERVAL_S = 2.0
HEALTH_PROBE_TIMEOUT_S = 5.0
STOP_GRACE_S = 30.0


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
        help=(
            "Start vLLM on the profile's GPUs against the weights directory, "
            "poll /health, then stop the server. Does not download. "
            "Off by default. Needs the accelerators."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=(
            f"Localhost port for vLLM when --load-check is set. "
            f"Default {DEFAULT_PORT}. Ignored without --load-check."
        ),
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def huggingface_env(weights_dir: Path) -> dict[str, str]:
    """Environment for a child that must read and write this HF cache.

    ``huggingface_hub`` prefers ``HF_HUB_CACHE`` over ``HF_HOME``. Pin both
    at this directory so an inherited home-directory cache cannot receive
    the files. Does not set ``CUDA_VISIBLE_DEVICES``.
    """
    env = os.environ.copy()
    env["HF_HOME"] = str(weights_dir)
    hub_cache = str(weights_dir / "hub")
    env["HF_HUB_CACHE"] = hub_cache
    env["HUGGINGFACE_HUB_CACHE"] = hub_cache
    return env


def download_snapshot(model_id: str, weights_dir: Path) -> int:
    """Download ``model_id`` with the vllm environment's Hugging Face client.

    ``snapshot_download`` is idempotent: when the snapshot is already complete
    it returns without fetching again, and this function returns 0.
    """
    env = huggingface_env(weights_dir)
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
            cwd=repo_root(),
            env=env,
            check=False,
        )
    except FileNotFoundError:
        print("error: pixi not found on PATH", file=sys.stderr)
        return 127
    return completed.returncode


def vllm_serve_command(model_id: str, *, port: int, context_length: int) -> list[str]:
    """``vllm serve`` for a load check.

    Request logging stays off (``--no-enable-log-requests``). Tensor-parallel,
    FP8, and FlashInfer flags are intentionally absent; session 3 adds those.
    """
    return [
        "pixi",
        "run",
        "--environment",
        "vllm",
        "vllm",
        "serve",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(context_length),
        "--no-enable-log-requests",
    ]


def _is_cluster_cuda_path(entry: str) -> bool:
    """True for the site CUDA module tree (``/software/cuda`` and below)."""
    normalized = os.path.normpath(entry)
    root = "/software/cuda"
    return normalized == root or normalized.startswith(root + os.sep)


def _without_cluster_cuda(env: dict[str, str]) -> None:
    """Keep the cluster CUDA toolchain out of a vLLM child.

    ``/software/cuda/bin/nvcc`` is a site wrapper. With no module version
    selected it exits 255 and prints ``No version string specified``.
    DeepGEMM's JIT runs that ``nvcc`` while preparing FP8 weights. This
    pixi environment ships wheels only and has no toolkit, so the child
    uses the prebuilt CUTLASS block-FP8 kernel instead, and does not see
    the wrapper on ``PATH`` or via ``CUDA_HOME``.
    """
    path = env.get("PATH")
    if path:
        env["PATH"] = os.pathsep.join(
            entry
            for entry in path.split(os.pathsep)
            if entry and not _is_cluster_cuda_path(entry)
        )
    for key in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        value = env.get(key)
        if value and _is_cluster_cuda_path(value):
            del env[key]
    env["VLLM_USE_DEEP_GEMM"] = "0"
    env["VLLM_MOE_USE_DEEP_GEMM"] = "0"


def vllm_child_env(weights_dir: Path, profile: Profile) -> dict[str, str]:
    """Environment for the vLLM child only.

    ``CUDA_VISIBLE_DEVICES`` is set on this mapping. The caller must not
    write it into ``os.environ``.
    """
    env = huggingface_env(weights_dir)
    env["CUDA_VISIBLE_DEVICES"] = profile.vllm_cuda_visible_devices
    _without_cluster_cuda(env)
    return env


def server_is_healthy(port: int, timeout_s: float) -> bool:
    """True when ``GET /health`` on localhost returns 200.

    An empty proxy handler skips ``http_proxy``. The check is for this host
    only, and a cluster proxy would both hide a live server and send the
    probe off the node.
    """
    url = f"http://127.0.0.1:{port}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout_s) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_health(proc: subprocess.Popen[bytes], port: int, timeout_s: float) -> str:
    """Poll until ``/health`` succeeds, the server exits, or the budget ends.

    Returns ``"ok"``, ``"died"``, or ``"timeout"``.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        if proc.poll() is not None:
            return "died"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout"
        probe_s = min(HEALTH_PROBE_TIMEOUT_S, remaining)
        if server_is_healthy(port, probe_s):
            return "ok"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout"
        time.sleep(min(POLL_INTERVAL_S, remaining))


def _signal_group(pid: int, sig: int) -> bool:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return False
    return True


def stop_process_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the child's process group, then SIGKILL if it stays up.

    The leader is started with ``start_new_session``, so its pid is the
    process-group id. A second signal after the leader exits covers workers
    still in that group.
    """
    if proc.poll() is not None:
        _signal_group(proc.pid, signal.SIGKILL)
        return

    if not _signal_group(proc.pid, signal.SIGTERM):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        _signal_group(proc.pid, signal.SIGKILL)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        proc.wait(timeout=STOP_GRACE_S)
    else:
        _signal_group(proc.pid, signal.SIGKILL)


def load_check(profile: Profile, weights_dir: Path, port: int) -> int:
    """Start ``vllm serve``, wait for ``/health``, then stop it.

    Exit 0 when ``/health`` returns 200 within the profile's health timeout.
    Exit non-zero on timeout or if the server exits first. The server's
    process group is killed on every path, including signals.
    """
    env = vllm_child_env(weights_dir, profile)
    command = vllm_serve_command(
        profile.model_id,
        port=port,
        context_length=profile.context_length,
    )
    print(
        f"load check: {profile.model_id} on 127.0.0.1:{port}, "
        f"timeout {profile.health_timeout_s:.0f}s"
    )
    try:
        proc = subprocess.Popen(
            command,
            cwd=repo_root(),
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        print("error: pixi not found on PATH", file=sys.stderr)
        return 127

    def _on_signal(signum: int, _frame: object) -> None:
        stop_process_group(proc)
        raise SystemExit(128 + signum)

    previous = {
        signal.SIGINT: signal.signal(signal.SIGINT, _on_signal),
        signal.SIGTERM: signal.signal(signal.SIGTERM, _on_signal),
    }
    try:
        outcome = wait_for_health(proc, port, profile.health_timeout_s)
        if outcome == "ok":
            print(f"load check passed: {profile.model_id}")
            return 0
        if outcome == "died":
            code = proc.returncode
            print(
                f"error: vLLM exited before /health (status {code})",
                file=sys.stderr,
            )
            if isinstance(code, int) and code > 0:
                return code
            return 1
        print(
            f"error: /health timed out after {profile.health_timeout_s:.0f}s "
            f"({profile.name})",
            file=sys.stderr,
        )
        return 1
    finally:
        stop_process_group(proc)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


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
    if args.load_check and not 1 <= args.port <= 65535:
        print(
            f"error: --port must be between 1 and 65535, got {args.port}",
            file=sys.stderr,
        )
        return 1

    if args.load_check:
        if not weights_dir.is_dir():
            print(
                f"error: weights directory does not exist: {weights_dir}",
                file=sys.stderr,
            )
            return 1
        return load_check(profile, weights_dir, args.port)

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
