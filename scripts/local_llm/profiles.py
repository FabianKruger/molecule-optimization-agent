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
    value. ``vllm_gpus`` and ``boltz_gpu`` are slots in the
    ``CUDA_VISIBLE_DEVICES`` list ``srun`` set, in that order. Slot 0 is the
    first assigned GPU, whichever physical id that is.
    """

    name: str
    model_id: str
    vllm_gpus: tuple[int, ...]
    boltz_gpu: int
    context_length: int
    health_timeout: timedelta

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


class DeviceAssignmentError(Exception):
    """``CUDA_VISIBLE_DEVICES`` cannot fill this profile's GPU slots."""


def parse_visible_devices(raw: str | None) -> tuple[str, ...]:
    """Device ids from ``CUDA_VISIBLE_DEVICES``, in the order ``srun`` listed them.

    An unset or empty value is an error. The ids are copied onto each child
    as written; they are not rewritten into ``0,1,2...``.
    """
    if raw is None or raw.strip() == "":
        raise DeviceAssignmentError(
            "CUDA_VISIBLE_DEVICES is unset. Run under srun so it lists "
            "the assigned GPUs."
        )
    devices = tuple(part.strip() for part in raw.split(","))
    if any(device == "" for device in devices):
        raise DeviceAssignmentError(
            f"CUDA_VISIBLE_DEVICES has an empty entry: {raw!r}"
        )
    if len(set(devices)) != len(devices):
        raise DeviceAssignmentError(
            f"CUDA_VISIBLE_DEVICES lists a GPU more than once: {raw!r}"
        )
    return devices


def visible_devices_for(
    profile: Profile, slots: tuple[int, ...], *, role: str
) -> str:
    """Return the ``CUDA_VISIBLE_DEVICES`` value for ``slots``.

    Reads the variable from the current process and does not modify it.
    """
    visible = parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if any(slot < 0 or slot >= len(visible) for slot in slots):
        needed = max(slots) + 1
        slot_list = ",".join(str(slot) for slot in slots)
        listed = ",".join(visible)
        raise DeviceAssignmentError(
            f"{profile.name} assigns {role} to CUDA_VISIBLE_DEVICES "
            f"slot(s) {slot_list} ({needed} assigned GPUs required). "
            f"The variable lists {len(visible)}: {listed}"
        )
    return ",".join(visible[slot] for slot in slots)


def assigned_process_devices(profile: Profile) -> tuple[str, str]:
    """``(vllm devices, boltz device)`` taken from ``CUDA_VISIBLE_DEVICES``."""
    boltz = visible_devices_for(profile, (profile.boltz_gpu,), role="Boltz")
    vllm = visible_devices_for(profile, profile.vllm_gpus, role="vLLM")
    return vllm, boltz


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
