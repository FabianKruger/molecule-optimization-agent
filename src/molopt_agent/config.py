from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class LLMConfig:
    model: str
    temperature: float
    base_url: Optional[str] = None


@dataclass
class OracleConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    experiment_name: str
    llm: LLMConfig
    recursion_limit: int = 100
    log_dir: str = "logs"
    oracle: OracleConfig = field(default=None)      # type: ignore[assignment]
    objective: ObjectiveConfig = field(default=None)  # type: ignore[assignment]


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load and validate experiment config from YAML; raise ValueError on issues."""
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ValueError(f"Config file not found: {path}") from e

    if not isinstance(raw, dict):
        raise ValueError("Top-level of config YAML must be a mapping/object.")

    for key in ("experiment_name", "llm", "oracle", "objective"):
        if key not in raw:
            raise ValueError(f"Missing required key in config: '{key}'")

    # LLM section
    llm_raw = raw["llm"]
    if not isinstance(llm_raw, dict):
        raise ValueError("Config.llm must be a mapping/object.")
    for key in ("model", "temperature"):
        if key not in llm_raw:
            raise ValueError(f"Missing required key in config.llm: '{key}'")
    llm = LLMConfig(
        model=str(llm_raw["model"]),
        temperature=float(llm_raw["temperature"]),
        base_url=llm_raw.get("base_url"),  # Optional
    )

    # Oracle section
    oracle_raw = raw["oracle"]
    if not isinstance(oracle_raw, dict):
        raise ValueError("Config.oracle must be a mapping/object.")
    if "name" not in oracle_raw:
        raise ValueError("Config.oracle is missing required key 'name'.")
    oracle_cfg = OracleConfig(
        name=str(oracle_raw["name"]),
        params=dict(oracle_raw.get("params", {}) or {}),
    )

    # Objective section
    objective_raw = raw["objective"]
    if not isinstance(objective_raw, dict):
        raise ValueError("Config.objective must be a mapping/object.")
    if "name" not in objective_raw:
        raise ValueError("Config.objective is missing required key 'name'.")
    objective_cfg = ObjectiveConfig(
        name=str(objective_raw["name"]),
        params=dict(objective_raw.get("params", {}) or {}),
    )

    recursion_limit = int(raw.get("recursion_limit", 100))
    log_dir = str(raw.get("log_dir", "logs"))
    experiment_name = str(raw["experiment_name"])

    return ExperimentConfig(
        experiment_name=experiment_name,
        llm=llm,
        recursion_limit=recursion_limit,
        log_dir=log_dir,
        oracle=oracle_cfg,
        objective=objective_cfg,
    )
