"""Unit tests for scripts/local_llm helpers (no GPU / vLLM required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "local_llm"


def _load_script(name: str) -> ModuleType:
    """Load ``scripts/local_llm/<name>.py`` by path.

    These helpers are runnable scripts, not an installed package. Loading by
    file path keeps runtime imports working without putting ``scripts/`` on
    ``sys.path`` for ty (which only resolves ``src/`` and ``tests/``).
    """
    path = _SCRIPTS / f"{name}.py"
    # Sibling imports inside the scripts (e.g. bootstrap -> download) need
    # the directory on sys.path at runtime.
    script_dir = str(_SCRIPTS)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(f"local_llm_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Scripts import each other as bare names (``from profiles import ...``).
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profiles = _load_script("profiles")
download = _load_script("download")
bootstrap = _load_script("bootstrap")


def test_resolve_profile_cli_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISMO_LLM_PROFILE", "deepseek-v3.2")
    profile = profiles.resolve_profile("qwen3-32b")
    assert profile.name == "qwen3-32b"
    assert profile.model_id == profiles.PROFILES["qwen3-32b"].model_id


def test_resolve_profile_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISMO_LLM_PROFILE", "deepseek-v3.2")
    assert profiles.resolve_profile(None).name == "deepseek-v3.2"


def test_resolve_profile_requires_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISMO_LLM_PROFILE", raising=False)
    with pytest.raises(profiles.ProfileError, match="SEISMO_LLM_PROFILE"):
        profiles.resolve_profile(None)


def test_resolve_profile_rejects_unknown() -> None:
    with pytest.raises(profiles.ProfileError, match="unknown profile"):
        profiles.resolve_profile("not-a-profile")


def test_parse_visible_devices_requires_assignment() -> None:
    with pytest.raises(profiles.DeviceAssignmentError, match="unset"):
        profiles.parse_visible_devices(None)
    with pytest.raises(profiles.DeviceAssignmentError, match="unset"):
        profiles.parse_visible_devices("")


def test_parse_visible_devices_rejects_empty_or_duplicate() -> None:
    with pytest.raises(profiles.DeviceAssignmentError, match="empty entry"):
        profiles.parse_visible_devices("0,,1")
    with pytest.raises(profiles.DeviceAssignmentError, match="more than once"):
        profiles.parse_visible_devices("0,1,0")


def test_visible_devices_for_maps_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7,9,11,13")
    profile = profiles.PROFILES["qwen3-32b"]
    assert profiles.visible_devices_for(profile, profile.vllm_gpus, role="vLLM") == "3"
    assert (
        profiles.visible_devices_for(profile, (profile.boltz_gpu,), role="Boltz") == "7"
    )
    vllm, boltz = profiles.assigned_process_devices(profile)
    assert (vllm, boltz) == ("3", "7")


def test_visible_devices_for_requires_enough_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    profile = profiles.PROFILES["qwen3-32b"]
    with pytest.raises(
        profiles.DeviceAssignmentError, match="2 assigned GPUs required"
    ):
        profiles.assigned_process_devices(profile)


def test_resolve_outside_home_rejects_relative_and_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(ValueError, match="absolute"):
        download.resolve_outside_home("relative/path", source="--weights-dir")
    with pytest.raises(ValueError, match="inside \\$HOME"):
        download.resolve_outside_home(str(home / "weights"), source="--weights-dir")


def test_resolve_weights_and_cache_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    weights = tmp_path / "weights"
    cache = tmp_path / "cache"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("SEISMO_LLM_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("SEISMO_LLM_CACHE_DIR", raising=False)

    assert download.resolve_weights_setting(str(weights)) == weights.resolve()
    assert download.resolve_cache_setting(str(cache)) == cache.resolve()

    monkeypatch.setenv("SEISMO_LLM_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("SEISMO_LLM_CACHE_DIR", str(cache))
    assert download.resolve_weights_setting(None) == weights.resolve()
    assert download.resolve_cache_setting(None) == cache.resolve()


def test_profile_serve_args_and_command() -> None:
    qwen = profiles.PROFILES["qwen3-32b"]
    deepseek = profiles.PROFILES["deepseek-v3.2"]
    assert "--reasoning-parser" in download.profile_serve_args(qwen)
    assert "flashinfer_trtllm" in download.profile_serve_args(deepseek)
    command = download.vllm_serve_command(qwen, port=9001)
    assert command[:5] == ["pixi", "run", "--environment", "vllm", "vllm"]
    assert "9001" in command
    assert str(qwen.context_length) in command


def test_without_cluster_cuda_strips_module_paths() -> None:
    assert download._is_cluster_cuda_path("/software/cuda")
    assert download._is_cluster_cuda_path("/software/cuda/12.4/bin")
    assert not download._is_cluster_cuda_path("/usr/local/cuda")

    env = {
        "PATH": "/software/cuda/bin:/usr/bin",
        "CUDA_HOME": "/software/cuda/12.4",
        "CUDA_PATH": "/opt/ok",
    }
    download._without_cluster_cuda(env)
    assert env["PATH"] == "/usr/bin"
    assert "CUDA_HOME" not in env
    assert env["CUDA_PATH"] == "/opt/ok"
    assert env["VLLM_USE_DEEP_GEMM"] == "0"
    assert env["VLLM_MOE_USE_DEEP_GEMM"] == "0"


def test_bootstrap_read_llm_model_and_command(tmp_path: Path) -> None:
    config = tmp_path / "exp.yaml"
    config.write_text("llm:\n  model: Qwen/Qwen3-32B-FP8\n", encoding="utf-8")
    path = bootstrap.resolve_config_path(str(config))
    assert bootstrap.read_llm_model(path) == "Qwen/Qwen3-32B-FP8"
    assert bootstrap.seismo_command(path) == [
        "pixi",
        "run",
        "seismo",
        "--config",
        str(path),
    ]


def test_bootstrap_rejects_missing_llm_model(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("objectives: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing llm.model"):
        bootstrap.read_llm_model(config)
