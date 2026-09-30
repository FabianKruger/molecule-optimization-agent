#!/usr/bin/env python
"""Start one local vLLM, run one molopt experiment, then stop vLLM.

Profile and weights directory resolve the same way as
``scripts/local_llm/download.py``. ``--config`` is required. ``llm.model``
must already be the profile's served name; this script does not rewrite
YAML. vLLM is the child from ``download.py`` (``VLLM_CACHE_ROOT`` from
``--cache-dir`` or ``MOLOPT_LLM_CACHE_DIR``, DeepGEMM off, cluster ``nvcc``
removed). ``molopt`` runs in the default pixi environment with only the
profile's Boltz GPU visible.

One experiment per invocation. This script does not call ``sbatch`` and
does not know the partition or account. It assumes it is already inside
``salloc`` or ``sbatch``.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from download import (
    DEFAULT_PORT,
    POLL_INTERVAL_S,
    repo_root,
    resolve_cache_setting,
    resolve_weights_setting,
    start_vllm,
    stop_process_group,
    wait_for_health,
)
from profiles import (
    DeviceAssignmentError,
    Profile,
    ProfileError,
    assigned_process_devices,
    resolve_profile,
    visible_devices_for,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve one local LLM profile and run one molopt experiment "
            "against it."
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
        "--cache-dir",
        default=None,
        help=(
            "Absolute directory for vLLM's cache, including torch.compile. "
            "Set as VLLM_CACHE_ROOT on the server child. Must not be inside "
            "$HOME. Overrides MOLOPT_LLM_CACHE_DIR. There is no default."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Experiment YAML. llm.model must equal the profile's served "
            "name. This script does not edit the file."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Localhost port for vLLM. Default {DEFAULT_PORT}.",
    )
    return parser.parse_args(argv)


def resolve_config_path(raw: str) -> Path:
    """Return the experiment file ``molopt --config`` will open.

    A relative path is resolved from the current working directory, then
    passed to ``molopt`` as an absolute path so the repo-root cwd does not
    change which file is read.
    """
    text = raw.strip()
    if text == "":
        raise ValueError("--config is empty")
    path = Path(text)
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve(strict=False)
    if not resolved.is_file():
        raise ValueError(f"--config does not exist: {resolved}")
    return resolved


def read_llm_model(config_path: Path) -> str:
    """Return ``llm.model`` from an experiment YAML.

    This reads the field the harness will send. It does not validate the
    rest of the file and does not import the harness.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ValueError(
            "PyYAML is required to read --config "
            "(the default pixi environment provides it)"
        ) from exc
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read --config {config_path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot parse --config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"--config {config_path} is not a YAML mapping")
    llm = raw.get("llm")
    if not isinstance(llm, dict) or "model" not in llm:
        raise ValueError(f"--config {config_path} is missing llm.model")
    model = llm["model"]
    if not isinstance(model, str) or model.strip() == "":
        raise ValueError(f"--config {config_path} has an empty llm.model")
    return model.strip()


def molopt_command(config_path: Path) -> list[str]:
    """``pixi run molopt`` in the default pixi environment."""
    return ["pixi", "run", "molopt", "--config", str(config_path)]


def molopt_env(profile: Profile, port: int) -> dict[str, str]:
    """Environment for the ``molopt`` child only.

    Boltz-2 inherits this mapping. ``CUDA_VISIBLE_DEVICES`` is the Boltz
    slot in the list ``srun`` set on the parent, and that value is not
    written back to the parent. ``OPENAI_BASE_URL`` is the local ``/v1``
    URL. An ``OPENAI_API_KEY`` that is already set is left as it is; an
    unset key becomes ``local``.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices_for(
        profile, (profile.boltz_gpu,), role="Boltz"
    )
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    if env.get("OPENAI_API_KEY") in (None, ""):
        env["OPENAI_API_KEY"] = "local"
    return env


def _vllm_exit_status(proc: subprocess.Popen[bytes]) -> int:
    code = proc.returncode
    if isinstance(code, int) and code > 0:
        return code
    return 1


def run_molopt_until_done(
    vllm_proc: subprocess.Popen[bytes],
    molopt_proc: subprocess.Popen[bytes],
) -> int:
    """Wait until ``molopt`` exits, or stop it if vLLM exits first.

    The return value is ``molopt``'s status when the experiment finished.
    If vLLM dies first, ``molopt`` is killed and the return value is
    non-zero.
    """
    while True:
        molopt_code = molopt_proc.poll()
        if molopt_code is not None:
            return molopt_code
        if vllm_proc.poll() is not None:
            print(
                "error: vLLM exited while molopt was running "
                f"(status {vllm_proc.returncode})",
                file=sys.stderr,
            )
            stop_process_group(molopt_proc)
            return _vllm_exit_status(vllm_proc)
        time.sleep(POLL_INTERVAL_S)


def bootstrap(
    profile: Profile,
    weights_dir: Path,
    cache_dir: Path,
    config_path: Path,
    port: int,
) -> int:
    """Start vLLM, run one experiment, then tear the server down.

    The vLLM process group is killed on success, on ``molopt`` failure, on
    health timeout, and on signals. When the experiment ran, the exit code
    is ``molopt``'s status.
    """
    try:
        vllm_devices, boltz_device = assigned_process_devices(profile)
    except DeviceAssignmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"bootstrap: {profile.model_id} on 127.0.0.1:{port}, "
        f"vLLM on {vllm_devices}, Boltz on {boltz_device}, "
        f"timeout {profile.health_timeout_s:.0f}s"
    )
    try:
        vllm_proc = start_vllm(profile, weights_dir, port, cache_dir)
    except FileNotFoundError:
        print("error: pixi not found on PATH", file=sys.stderr)
        return 127

    molopt_proc: subprocess.Popen[bytes] | None = None

    def _on_signal(signum: int, _frame: object) -> None:
        if molopt_proc is not None:
            stop_process_group(molopt_proc)
        stop_process_group(vllm_proc)
        raise SystemExit(128 + signum)

    previous = {
        signal.SIGINT: signal.signal(signal.SIGINT, _on_signal),
        signal.SIGTERM: signal.signal(signal.SIGTERM, _on_signal),
    }
    try:
        outcome = wait_for_health(vllm_proc, port, profile.health_timeout_s)
        if outcome == "died":
            print(
                "error: vLLM exited before /health "
                f"(status {vllm_proc.returncode})",
                file=sys.stderr,
            )
            return _vllm_exit_status(vllm_proc)
        if outcome == "timeout":
            print(
                f"error: /health timed out after {profile.health_timeout_s:.0f}s "
                f"({profile.name})",
                file=sys.stderr,
            )
            return 1

        print(
            f"vLLM healthy: {profile.model_id}; "
            f"starting molopt (Boltz on {boltz_device})"
        )
        try:
            molopt_proc = subprocess.Popen(
                molopt_command(config_path),
                cwd=repo_root(),
                env=molopt_env(profile, port),
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            print("error: pixi not found on PATH", file=sys.stderr)
            return 127
        return run_molopt_until_done(vllm_proc, molopt_proc)
    finally:
        if molopt_proc is not None:
            stop_process_group(molopt_proc)
        stop_process_group(vllm_proc)
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
        config_path = resolve_config_path(args.config)
        served_name = read_llm_model(config_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if served_name != profile.model_id:
        print(
            f"error: llm.model is {served_name!r}; "
            f"profile {profile.name} serves {profile.model_id!r}",
            file=sys.stderr,
        )
        return 1
    if not 1 <= args.port <= 65535:
        print(
            f"error: --port must be between 1 and 65535, got {args.port}",
            file=sys.stderr,
        )
        return 1
    try:
        cache_dir = resolve_cache_setting(args.cache_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not weights_dir.is_dir():
        print(
            f"error: weights directory does not exist: {weights_dir}",
            file=sys.stderr,
        )
        return 1
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"error: cannot create cache directory {cache_dir}: {exc}",
            file=sys.stderr,
        )
        return 1
    return bootstrap(profile, weights_dir, cache_dir, config_path, args.port)


if __name__ == "__main__":
    sys.exit(main())
