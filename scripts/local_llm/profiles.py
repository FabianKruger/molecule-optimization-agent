"""Local LLM profiles.

This module is the only profile table. The download script and the bootstrap
script import it. vLLM serve flags for a checkpoint live on the serve command
(``vllm_serve_command``), not in this table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class Profile:
    """One local model and the GPUs and health budget later sessions use.

    ``model_id`` is the Hugging Face repository id and the ``llm.model``
    value. ``vllm_gpus`` and ``boltz_gpu`` are CUDA device indices.
    """

    name: str
    model_id: str
    vllm_gpus: tuple[int, ...]
    boltz_gpu: int
    context_length: int
    health_timeout: timedelta

    @property
    def vllm_cuda_visible_devices(self) -> str:
        return ",".join(str(gpu) for gpu in self.vllm_gpus)

    @property
    def boltz_cuda_visible_devices(self) -> str:
        return str(self.boltz_gpu)

    @property
    def health_timeout_s(self) -> float:
        return self.health_timeout.total_seconds()


PROFILES: dict[str, Profile] = {
    "qwen3-32b": Profile(
        name="qwen3-32b",
        model_id="Qwen/Qwen3-32B-FP8",
        vllm_gpus=(0,),
        boltz_gpu=1,
        # Native context. No YaRN.
        context_length=32768,
        health_timeout=timedelta(minutes=10),
    ),
    "deepseek-v3.2": Profile(
        name="deepseek-v3.2",
        model_id="deepseek-ai/DeepSeek-V3.2",
        vllm_gpus=(0, 1, 2, 3),
        boltz_gpu=4,
        # Requests past this context fail.
        context_length=131072,
        health_timeout=timedelta(minutes=45),
    ),
}


class ProfileError(Exception):
    """The profile was missing or is not in the table."""


def resolve_profile(cli_profile: str | None) -> Profile:
    """Return the selected profile.

    ``--profile`` wins over ``MOLOPT_LLM_PROFILE``. If neither is set, raise
    ``ProfileError``. There is no default profile.
    """
    if cli_profile is not None:
        name = cli_profile.strip()
        if name == "":
            raise ProfileError("--profile is empty")
        source = "--profile"
    else:
        raw = os.environ.get("MOLOPT_LLM_PROFILE")
        if raw is None or raw.strip() == "":
            known = ", ".join(PROFILES)
            raise ProfileError(
                f"set --profile or MOLOPT_LLM_PROFILE (one of: {known})"
            )
        name = raw.strip()
        source = "MOLOPT_LLM_PROFILE"

    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(PROFILES)
        raise ProfileError(
            f"unknown profile {name!r} from {source} (one of: {known})"
        ) from None
