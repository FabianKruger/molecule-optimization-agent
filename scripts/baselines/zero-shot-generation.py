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

from molopt_agent.objectives.tdc_tasks import TDC_ORACLE_DESCRIPTIONS
from molopt_agent.oracles.tdc_tasks import TdcOracle
from molopt_agent.system_prompt import SYSTEM_PROMPT


API_KEY_ENV = "OPENAI_API_KEY"
BASE_URL_ENV = "OPENAI_BASE_URL"


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


def build_user_prompt(task_name: str, n_molecules: int, mode: Mode) -> str:
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


def run_task_independent(llm: ChatOpenAI, task_name: str, n_molecules: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	oracle = TdcOracle(oracle_name=task_name)
	trace: list[dict[str, Any]] = []
	conversation: list[dict[str, Any]] = []
	prompt = build_user_prompt(task_name, n_molecules, mode="independent")

	# Add system message to conversation
	conversation.append({"role": "SystemMessage", "content": SYSTEM_PROMPT})
	# Add first user message to conversation
	conversation.append({"role": "HumanMessage", "content": prompt})

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
			score = float(oracle(smiles)["score"])
		except Exception as e:
			print(f"[{task_name}] iteration {i}: oracle failed for {smiles}: {e}")
			continue

		trace.append({"iteration": i, "smiles": smiles, "reason": reason, "score": score})
		# Add AI response to conversation
		conversation.append({"role": "AIMessage", "content": json.dumps({"reason": reason, "smiles": smiles})})
		print(f"[{task_name}] iteration {i}/{n_molecules} score={score:.4f}")

	return trace, conversation


def run_task_batch(llm: ChatOpenAI, task_name: str, n_molecules: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	oracle = TdcOracle(oracle_name=task_name)
	trace: list[dict[str, Any]] = []
	conversation: list[dict[str, Any]] = []
	prompt = build_user_prompt(task_name, n_molecules, mode="batch")

	# Add system message to conversation
	conversation.append({"role": "SystemMessage", "content": BATCH_SYSTEM_PROMPT})
	# Add user prompt to conversation
	conversation.append({"role": "HumanMessage", "content": prompt})

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

	# Add AI response to conversation
	conversation.append({"role": "AIMessage", "content": response_text})

	for i, entry in enumerate(payload[:n_molecules], start=1):
		try:
			smiles, reason = _safe_smiles_reason(entry)
		except Exception as e:
			print(f"[{task_name}] batch item {i}: parse failed: {e}")
			continue

		try:
			score = float(oracle(smiles)["score"])
		except Exception as e:
			print(f"[{task_name}] batch item {i}: oracle failed for {smiles}: {e}")
			continue

		trace.append({"iteration": i, "smiles": smiles, "reason": reason, "score": score})
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
) -> Path:
	task_dir = output_dir / task_name
	task_dir.mkdir(parents=True, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_path = task_dir / f"zero_shot_{mode}_{timestamp}.json"

	data = {
		"timestamp": timestamp,
		"experiment": f"zero_shot_{mode}_{task_name}",
		"config": {
			"llm": {"model": model, "temperature": temperature},
			"objective": {"name": "tdc_tasks", "params": {"max_iterations": n_molecules}},
			"oracle": {"name": "tdc_tasks", "params": {"oracle_name": task_name}},
			"generation_mode": mode,
		},
		"conversation": conversation,
		"trace": trace,
		"iterations": len(trace),
	}

	out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
	return out_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Zero-shot generation baseline for PMO/TDC tasks")
	parser.add_argument("--output-dir", default="data/results/tdc_tasks_zero_shot", help="Output directory")
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
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	output_dir = (PROJECT_ROOT / args.output_dir).resolve()
	mode: Mode = args.mode

	tasks = args.tasks if args.tasks else sorted(TDC_ORACLE_DESCRIPTIONS.keys())
	invalid = [t for t in tasks if t not in TDC_ORACLE_DESCRIPTIONS]
	if invalid:
		raise ValueError(f"Unknown task(s): {invalid}")

	llm = build_llm(model=args.model, temperature=args.temperature)

	print(
		f"Running zero-shot generation mode={mode}, tasks={len(tasks)}, "
		f"n_molecules={args.n_molecules}, model={args.model}"
	)

	for task_name in tasks:
		print(f"\n=== Task: {task_name} ===")
		if mode == "independent":
			trace, conversation = run_task_independent(llm, task_name=task_name, n_molecules=args.n_molecules)
		else:
			trace, conversation = run_task_batch(llm, task_name=task_name, n_molecules=args.n_molecules)

		out_path = save_trace(
			output_dir=output_dir,
			task_name=task_name,
			mode=mode,
			model=args.model,
			temperature=args.temperature,
			n_molecules=args.n_molecules,
			trace=trace,
			conversation=conversation,
		)
		print(f"Saved {len(trace)} evaluated molecules to {out_path}")


if __name__ == "__main__":
	main()