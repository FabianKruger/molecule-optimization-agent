#!/usr/bin/env python3
"""Generate zero-shot PMO/TDC baseline molecules and evaluate with TDC oracles.

Two modes are supported:
- independent: N independent LLM calls, one molecule per call
- batch: one LLM call that returns N molecules at once

Outputs are saved in a JSON format compatible with the loaders in
analysis/paper/tdc_task.ipynb.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def _find_project_root() -> Path:
	current = Path(__file__).resolve()
	for parent in current.parents:
		if (parent / "pyproject.toml").exists():
			return parent
	raise RuntimeError("Could not find project root (no pyproject.toml found)")


PROJECT_ROOT = _find_project_root()

# Load environment variables from .env file
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
	load_dotenv(env_file)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from molopt_agent.objectives.generic_messages import get_generic_first_message
from molopt_agent.objectives.tdc_tasks import TDC_ORACLE_DESCRIPTIONS
from molopt_agent.oracles.composite import CompositeOracle
from molopt_agent.oracles.ic50mpro import IC50MproOracle
from molopt_agent.oracles.novel import NovelOracle
from molopt_agent.oracles.qed import ExplainableQedOracle
from molopt_agent.oracles.similarity import SimilarityOracle
from molopt_agent.oracles.tdc_tasks import TdcOracle
from molopt_agent.system_prompt import SYSTEM_PROMPT


API_KEY_ENV = "OPENAI_API_KEY"
BASE_URL_ENV = "OPENAI_BASE_URL"

# Thresholds sourced from scripts/run_ic50_mpro.sh
IC50_THRESHOLD: float = 1.0
QED_THRESHOLD: float = 0.6
IC50_MODEL_PATH = "data/xgboost/xgb_maccs_best"

# Thresholds sourced from scripts/run_similarity_qed_pubchem_experiments.sh
SIM_MIN_SIMILARITY: float = 0.7
SIM_MIN_QED: float = 0.7
SIM_TARGET_SCORE: float = 0.9
SIM_WEIGHTS: list[float] = [0.5, 0.5]

CUSTOM_TASKS: set[str] = {"ic50mpro_qed_novel", "similarity_qed_pubchem"}


BATCH_SYSTEM_PROMPT = """
You are an expert medicinal chemist whose job is to propose diverse, plausible small organic molecules.

OUTPUT FORMAT:
- Respond with exactly one JSON array.
- The array must contain exactly N objects (N is provided in the user instruction).
- Each object must have this shape:
  {
	"reason": "<short explanation>",
	"smiles": "<SMILES string>"
  }

Requirements:
- All "smiles" values must be distinct within the array.
- Each "smiles" should be a single valid SMILES for a plausible small molecule.
- Do not include any text outside the JSON array.
- No Markdown, no comments, no code fences.
""".strip()


Mode = Literal["independent", "batch"]


def _extract_json_payload(text: str) -> Any:
	text = text.strip()
	if not text:
		raise ValueError("Empty model response")

	try:
		return json.loads(text)
	except json.JSONDecodeError:
		pass

	match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
	if not match:
		raise ValueError("No JSON object or array found in model response")
	return json.loads(match.group(1))


def _to_text(content: Any) -> str:
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts: list[str] = []
		for item in content:
			if isinstance(item, dict) and item.get("type") == "text":
				parts.append(str(item.get("text", "")))
			elif isinstance(item, str):
				parts.append(item)
		return "\n".join(p for p in parts if p)
	return str(content)


def build_llm(model: str, temperature: float) -> ChatOpenAI:
	api_key = os.environ.get(API_KEY_ENV)
	if not api_key:
		raise RuntimeError(f"Missing API key: set {API_KEY_ENV} environment variable")

	llm_kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "temperature": temperature}
	base_url = os.environ.get(BASE_URL_ENV)
	if base_url:
		llm_kwargs["base_url"] = base_url
	return ChatOpenAI(**llm_kwargs)


def _is_higher_better(task_name: str) -> bool:
	return task_name != "ic50mpro_qed_novel"


def _generic_user_prompt(n_molecules: int, mode: Mode, higher_is_better: bool) -> str:
	if mode == "batch":
		direction = "Higher is better" if higher_is_better else "Lower is better"
		return (
			f"Your task is to optimize the score. {direction}.\n\n"
			f"Generate exactly {n_molecules} different candidate molecules in one response.\n\n"
			f"Return a JSON array with exactly {n_molecules} objects, each containing keys \"reason\" and \"smiles\".\n"
			"All SMILES must be unique within the array."
		)
	return get_generic_first_message(higher_is_better=higher_is_better)


def _ic50_user_prompt(n_molecules: int, mode: Mode) -> str:
	task_desc = (
		"We are optimizing ligands for a protease.\n\n"
		"Objectives:\n"
		f"1. Minimize the predicted IC50 (in nM). Target: IC50 < {IC50_THRESHOLD:.2f} nM.\n"
		f"2. Maintain drug-likeness. Minimum required: QED \u2265 {QED_THRESHOLD:.2f}\n"
		"3. The molecule must be NOVEL (not present in PubChem)."
	)
	if mode == "batch":
		return (
			f"{task_desc}\n\n"
			f"Generate exactly {n_molecules} different candidate molecules in one response.\n\n"
			f"Return a JSON array with exactly {n_molecules} objects, each containing keys \"reason\" and \"smiles\".\n"
			"All SMILES must be unique within the array."
		)
	return (
		f"{task_desc}\n\n"
		"Propose a molecule as a SMILES string that you expect to have strong inhibitory effect on the enzyme while maintaining good drug-like properties.\n\n"
		"Respond with a single JSON object:\n"
		"{\n"
		"  \"reason\": \"<short explanation>\",\n"
		"  \"smiles\": \"<SMILES string>\"\n"
		"}"
	)


def _similarity_user_prompt(target_smiles: str, n_molecules: int, mode: Mode) -> str:
	task_desc = (
		"We are optimizing molecules for TWO objectives simultaneously:\n\n"
		"1. Structural Similarity to a target molecule using MACCS fingerprints\n"
		f"   - Target molecule SMILES: {target_smiles}\n"
		f"   - Minimum required: similarity \u2265 {SIM_MIN_SIMILARITY:.2f}\n"
		"   - IMPORTANT: You must NOT propose the exact target molecule.\n\n"
		"2. Drug-likeness using the QED score\n"
		f"   - Minimum required: QED \u2265 {SIM_MIN_QED:.2f}\n\n"
		f"The combined score is: {SIM_WEIGHTS[0]} * similarity + {SIM_WEIGHTS[1]} * QED\n"
		f"Target combined score: \u2265 {SIM_TARGET_SCORE:.2f}"
	)
	if mode == "batch":
		return (
			f"{task_desc}\n\n"
			f"Generate exactly {n_molecules} different candidate molecules in one response.\n\n"
			f"Return a JSON array with exactly {n_molecules} objects, each containing keys \"reason\" and \"smiles\".\n"
			"All SMILES must be unique within the array."
		)
	return (
		f"{task_desc}\n\n"
		"Propose a molecule as a SMILES string that balances both objectives.\n\n"
		"Respond with a single JSON object:\n"
		"{\n"
		"  \"reason\": \"<short explanation>\",\n"
		"  \"smiles\": \"<SMILES string>\"\n"
		"}"
	)


def build_user_prompt(
	task_name: str,
	n_molecules: int,
	mode: Mode,
	prompt_style: Literal["task", "generic"] = "task",
	target_smiles: str | None = None,
) -> str:
	if prompt_style == "generic":
		return _generic_user_prompt(n_molecules, mode, higher_is_better=_is_higher_better(task_name))
	if task_name == "ic50mpro_qed_novel":
		return _ic50_user_prompt(n_molecules, mode)
	if task_name == "similarity_qed_pubchem":
		if target_smiles is None:
			raise ValueError("similarity_qed_pubchem requires target_smiles")
		return _similarity_user_prompt(target_smiles, n_molecules, mode)
	d = TDC_ORACLE_DESCRIPTIONS[task_name]
	task_description = d["task_description"]
	metric_name = d["metric_name"]
	step_instruction = d["step_instruction"]
	if mode == "batch":
		return (
			f"{task_description}\n\n"
			"Objective:\n"
			f"- Generate exactly {n_molecules} different candidate molecules in one response.\n\n"
			f"- the molecules should maximize the {metric_name}.\n"
			"Task:\n"
			f"{step_instruction}\n\n"
			"Return a JSON array with exactly "
			f"{n_molecules} objects, each containing keys \"reason\" and \"smiles\".\n"
			"All SMILES must be unique within the array."
		)
	return (
		f"{task_description}\n\n"
		"Objective:\n"
		f"- generate one molecule that maximize the {metric_name}.\n"
		"- give a short explanation why this molecule was chosen.\n\n"
		"Task:\n"
		f"{step_instruction}\n\n"
		"Respond with a single JSON object:\n"
		"{\n"
		"  \"reason\": \"<short explanation>\",\n"
		"  \"smiles\": \"<SMILES string>\"\n"
		"}"
	)


def _safe_smiles_reason(entry: Any) -> tuple[str, str]:
	if not isinstance(entry, dict):
		raise ValueError("Expected JSON object entry")
	smiles = str(entry["smiles"]).strip()
	reason = str(entry.get("reason", "")).strip()
	if not smiles:
		raise ValueError("Missing or empty smiles")
	return smiles, reason


def build_task_oracle(task_name: str, target_smiles: str | None = None) -> Any:
	"""Build the appropriate oracle for a given task name."""
	if task_name == "ic50mpro_qed_novel":
		return CompositeOracle(
			oracles=[
				IC50MproOracle(model_name=str(PROJECT_ROOT / IC50_MODEL_PATH)),
				ExplainableQedOracle(),
				NovelOracle(),
			],
			weights=[1.0, 0.0, 0.0],
			names=["IC50", "QED", "Novelty"],
		)
	if task_name == "similarity_qed_pubchem":
		if target_smiles is None:
			raise ValueError("similarity_qed_pubchem requires target_smiles")
		return CompositeOracle(
			oracles=[SimilarityOracle(target_smiles=target_smiles), ExplainableQedOracle()],
			weights=SIM_WEIGHTS,
			names=["Similarity", "QED"],
		)
	return TdcOracle(oracle_name=task_name)


def _load_sampled_molecules() -> list[str]:
	path = PROJECT_ROOT / "data/molecules/sampled_molecules.txt"
	if not path.exists():
		raise FileNotFoundError(f"Sampled molecules file not found: {path}")
	return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_messages(system_prompt: str, user_prompt: str, mode: Mode, n_molecules: int) -> None:
	"""Check message composition before the first LLM call."""
	if not system_prompt.strip():
		raise ValueError("System prompt is empty")
	if not user_prompt.strip():
		raise ValueError("User prompt is empty")
	if "smiles" not in user_prompt.lower():
		raise ValueError("User prompt does not reference 'smiles'")
	if mode == "batch":
		if "json array" not in user_prompt.lower():
			raise ValueError("Batch user prompt does not mention a JSON array")
		if str(n_molecules) not in user_prompt:
			raise ValueError(f"Batch user prompt does not mention molecule count ({n_molecules})")
	elif mode == "independent":
		if "json object" not in user_prompt.lower():
			raise ValueError("Independent user prompt does not mention a JSON object")


def run_task_independent(
	llm: ChatOpenAI,
	task_name: str,
	n_molecules: int,
	oracle: Any,
	prompt: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	trace: list[dict[str, Any]] = []
	conversation: list[dict[str, Any]] = [
		{"role": "SystemMessage", "content": SYSTEM_PROMPT},
		{"role": "HumanMessage", "content": prompt},
	]

	for i in range(1, n_molecules + 1):
		try:
			response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
			response_text = _to_text(response.content)
			payload = _extract_json_payload(response_text)
			smiles, reason = _safe_smiles_reason(payload)
		except Exception as e:
			print(f"[{task_name}] iteration {i}: generation parse failed: {e}")
			continue

		try:
			result = oracle(smiles)
			score = float(result["score"])
		except Exception as e:
			print(f"[{task_name}] iteration {i}: oracle failed for {smiles}: {e}")
			continue

		entry: dict[str, Any] = {"iteration": i, "smiles": smiles, "reason": reason, "score": score}
		if "scores" in result:
			entry["scores"] = {k: float(v) for k, v in result["scores"].items()}
		trace.append(entry)
		conversation.append({"role": "AIMessage", "content": json.dumps({"reason": reason, "smiles": smiles})})
		print(f"[{task_name}] iteration {i}/{n_molecules} score={score:.4f}")

	return trace, conversation


def run_task_batch(
	llm: ChatOpenAI,
	task_name: str,
	n_molecules: int,
	oracle: Any,
	prompt: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	trace: list[dict[str, Any]] = []
	conversation: list[dict[str, Any]] = [
		{"role": "SystemMessage", "content": BATCH_SYSTEM_PROMPT},
		{"role": "HumanMessage", "content": prompt},
	]

	response_text = ""
	try:
		response = llm.invoke([SystemMessage(content=BATCH_SYSTEM_PROMPT), HumanMessage(content=prompt)])
		response_text = _to_text(response.content)
		payload = _extract_json_payload(response_text)
		if not isinstance(payload, list):
			raise ValueError("Batch mode expects a JSON array")
	except Exception as e:
		print(f"[{task_name}] batch generation failed: {e}")
		return trace, conversation

	conversation.append({"role": "AIMessage", "content": response_text})

	for i, item in enumerate(payload[:n_molecules], start=1):
		try:
			smiles, reason = _safe_smiles_reason(item)
		except Exception as e:
			print(f"[{task_name}] batch item {i}: parse failed: {e}")
			continue

		try:
			result = oracle(smiles)
			score = float(result["score"])
		except Exception as e:
			print(f"[{task_name}] batch item {i}: oracle failed for {smiles}: {e}")
			continue

		entry: dict[str, Any] = {"iteration": i, "smiles": smiles, "reason": reason, "score": score}
		if "scores" in result:
			entry["scores"] = {k: float(v) for k, v in result["scores"].items()}
		trace.append(entry)
		print(f"[{task_name}] batch item {i}/{n_molecules} score={score:.4f}")

	return trace, conversation


def save_trace(
	output_dir: Path,
	task_name: str,
	mode: Mode,
	model: str,
	temperature: float,
	n_molecules: int,
	trace: list[dict[str, Any]],
	conversation: list[dict[str, Any]],
	extra_config: dict[str, Any] | None = None,
	target_idx: int | None = None,
) -> Path:
	task_dir = output_dir / task_name
	task_dir.mkdir(parents=True, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	suffix = f"_target{target_idx}" if target_idx is not None else ""
	out_path = task_dir / f"zero_shot_{mode}{suffix}_{timestamp}.json"

	config: dict[str, Any] = {
		"llm": {"model": model, "temperature": temperature},
		"objective": {"name": task_name, "params": {"max_iterations": n_molecules}},
		"oracle": {"name": task_name},
		"generation_mode": mode,
	}
	if extra_config:
		config.update(extra_config)

	data = {
		"timestamp": timestamp,
		"experiment": f"zero_shot_{mode}_{task_name}",
		"config": config,
		"conversation": conversation,
		"trace": trace,
		"iterations": len(trace),
	}

	out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
	return out_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Zero-shot generation baseline for PMO/TDC tasks")
	parser.add_argument("--output-dir", default="data/results/zero_shot", help="Output directory")
	parser.add_argument("--n-molecules", type=int, default=50, help="Molecules to generate per task")
	parser.add_argument("--model", default="claude-opus-4.5", help="Model name")
	parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
	parser.add_argument("--tasks", nargs="*", default=None, help="Optional subset of tasks")
	parser.add_argument(
		"--mode",
		choices=["independent", "batch"],
		default="independent",
		help="Generation mode",
	)
	parser.add_argument(
		"--prompt-style",
		choices=["task", "generic"],
		default="task",
		help="'task' uses task-specific descriptions; 'generic' hides task details",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	output_dir = (PROJECT_ROOT / args.output_dir).resolve()
	mode: Mode = args.mode
	prompt_style: Literal["task", "generic"] = args.prompt_style

	valid_tasks = set(TDC_ORACLE_DESCRIPTIONS.keys()) | CUSTOM_TASKS
	tasks = args.tasks if args.tasks else sorted(TDC_ORACLE_DESCRIPTIONS.keys())
	invalid = [t for t in tasks if t not in valid_tasks]
	if invalid:
		raise ValueError(f"Unknown task(s): {invalid}")

	llm = build_llm(model=args.model, temperature=args.temperature)

	print(
		f"Running zero-shot generation mode={mode}, tasks={len(tasks)}, "
		f"n_molecules={args.n_molecules}, model={args.model}, prompt_style={prompt_style}"
	)

	for task_name in tasks:
		print(f"\n=== Task: {task_name} ===")

		# Build list of (target_smiles, target_idx) tuples; non-similarity tasks use a single None entry
		if task_name == "similarity_qed_pubchem":
			sampled = _load_sampled_molecules()
			print(f"  Running over {len(sampled)} sampled targets.")
			targets: list[tuple[str | None, int | None]] = [(s, i) for i, s in enumerate(sampled, start=1)]
		else:
			targets = [(None, None)]

		for target_smiles, target_idx in targets:
			oracle = build_task_oracle(task_name, target_smiles=target_smiles)
			user_prompt = build_user_prompt(
				task_name, args.n_molecules, mode,
				prompt_style=prompt_style, target_smiles=target_smiles,
			)
			system_prompt = SYSTEM_PROMPT if mode == "independent" else BATCH_SYSTEM_PROMPT
			_validate_messages(system_prompt, user_prompt, mode, args.n_molecules)

			if mode == "independent":
				trace, conversation = run_task_independent(
					llm, task_name=task_name, n_molecules=args.n_molecules,
					oracle=oracle, prompt=user_prompt,
				)
			else:
				trace, conversation = run_task_batch(
					llm, task_name=task_name, n_molecules=args.n_molecules,
					oracle=oracle, prompt=user_prompt,
				)

			extra: dict[str, Any] = {"prompt_style": prompt_style}
			if target_smiles is not None:
				extra["target_smiles"] = target_smiles
				extra["target_idx"] = target_idx

			out_path = save_trace(
				output_dir=output_dir,
				task_name=task_name,
				mode=mode,
				model=args.model,
				temperature=args.temperature,
				n_molecules=args.n_molecules,
				trace=trace,
				conversation=conversation,
				extra_config=extra,
				target_idx=target_idx,
			)
			print(f"Saved {len(trace)} evaluated molecules to {out_path}")


if __name__ == "__main__":
	main()