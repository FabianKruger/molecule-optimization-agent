"""Tests for experiment YAML config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from molopt_agent.config import load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]
QED_CONFIG = REPO_ROOT / "config" / "experiments" / "qed.yaml"


def test_load_bundled_qed_config():
    cfg = load_experiment_config(str(QED_CONFIG))
    assert cfg.experiment_name == "qed_optimization"
    assert cfg.llm.model == "claude-opus-4.5"
    assert cfg.llm.temperature == 0.3
    assert cfg.llm.reasoning_effort is None
    assert cfg.oracle.name == "qed"
    assert cfg.objective.name == "qed"
    assert cfg.objective.params["target_qed"] == 0.9
    assert cfg.recursion_limit == 100
    assert cfg.log_dir == "data/runs"


def test_load_config_missing_file():
    with pytest.raises(ValueError, match="not found"):
        load_experiment_config("/tmp/definitely-missing-seismo-config.yaml")


def test_load_config_rejects_non_mapping(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Top-level"):
        load_experiment_config(str(path))


def test_load_config_requires_top_level_keys(tmp_path: Path):
    path = tmp_path / "partial.yaml"
    path.write_text("experiment_name: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required key"):
        load_experiment_config(str(path))


def test_load_config_requires_llm_fields(tmp_path: Path):
    path = tmp_path / "llm_bad.yaml"
    path.write_text(
        """
experiment_name: x
llm:
  model: gpt-4o
oracle:
  name: qed
objective:
  name: qed
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config.llm"):
        load_experiment_config(str(path))


def test_load_config_accepts_reasoning_effort_none_string(tmp_path: Path):
    path = tmp_path / "effort_none.yaml"
    path.write_text(
        """
experiment_name: x
llm:
  model: gpt-4o
  temperature: 0.2
  reasoning_effort: None
oracle:
  name: qed
  params: {}
objective:
  name: qed
  params: {}
""".strip(),
        encoding="utf-8",
    )
    cfg = load_experiment_config(str(path))
    assert cfg.llm.reasoning_effort is None


def test_load_config_accepts_gpt_reasoning_effort(tmp_path: Path):
    path = tmp_path / "effort_ok.yaml"
    path.write_text(
        """
experiment_name: x
llm:
  model: gpt-5
  temperature: 0.2
  reasoning_effort: medium
oracle:
  name: qed
objective:
  name: qed
""".strip(),
        encoding="utf-8",
    )
    cfg = load_experiment_config(str(path))
    assert cfg.llm.reasoning_effort == "medium"


def test_load_config_rejects_invalid_reasoning_effort(tmp_path: Path):
    path = tmp_path / "effort_bad.yaml"
    path.write_text(
        """
experiment_name: x
llm:
  model: gpt-4o
  temperature: 0.2
  reasoning_effort: extreme
oracle:
  name: qed
objective:
  name: qed
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reasoning_effort"):
        load_experiment_config(str(path))


def test_load_config_requires_oracle_name(tmp_path: Path):
    path = tmp_path / "oracle_bad.yaml"
    path.write_text(
        """
experiment_name: x
llm:
  model: gpt-4o
  temperature: 0.1
oracle:
  params: {}
objective:
  name: qed
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="oracle"):
        load_experiment_config(str(path))


def test_load_config_rejects_non_dict_sections(tmp_path: Path):
    path = tmp_path / "llm_list.yaml"
    path.write_text(
        """
experiment_name: x
llm:
  - not-a-mapping
oracle:
  name: qed
objective:
  name: qed
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Config.llm"):
        load_experiment_config(str(path))

    path = tmp_path / "oracle_list.yaml"
    path.write_text(
        """
experiment_name: x
llm:
  model: gpt-4o
  temperature: 0.1
oracle:
  - qed
objective:
  name: qed
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Config.oracle"):
        load_experiment_config(str(path))

    path = tmp_path / "objective_list.yaml"
    path.write_text(
        """
experiment_name: x
llm:
  model: gpt-4o
  temperature: 0.1
oracle:
  name: qed
objective:
  - qed
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Config.objective"):
        load_experiment_config(str(path))


def test_load_config_requires_objective_name(tmp_path: Path):
    path = tmp_path / "objective_bad.yaml"
    path.write_text(
        """
experiment_name: x
llm:
  model: gpt-4o
  temperature: 0.1
oracle:
  name: qed
objective:
  params: {}
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="objective"):
        load_experiment_config(str(path))
