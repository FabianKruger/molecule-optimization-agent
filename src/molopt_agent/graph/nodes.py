import json
from rdkit import Chem
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import WorkflowState
from ..objectives.base import Objective
import re
import time
import httpcore
from openai import RateLimitError, APIConnectionError
import random

def rate_limit_sensible_llm_call(llm:ChatOpenAI, message, max_attempts=3):
    delay = 60  # sensible default for hard limits
    conn_delay = 0.5

    for attempt in range(max_attempts):
        try:
            return llm.invoke(message)

        except RateLimitError:
            if attempt == max_attempts - 1:
                raise

            print(f"... Wait {delay}s to not exceed rate limit")
            time.sleep(delay)
            delay = min(delay * 2, 300)  # cap at 10s

        except (APIConnectionError, httpcore.RemoteProtocolError) as e:
            if attempt == max_attempts - 1:
                raise

            # exponential backoff + jitter, capped
            wait = min(conn_delay, 15.0) * random.uniform(0.7, 1.3)
            print(f"... Wait {wait} as the connection was ended")
            now = time.time()
            with open("timing.log", "a", encoding="utf-8") as f:
                f.write(f"Attempt {attempt}: Will wait for {wait} and retry. Connection ended at |{now}\n")
            time.sleep(wait)
            conn_delay = min(conn_delay * 2, 15.0)


def make_generation_node(objective: Objective, llm: ChatOpenAI, system_prompt: str):
    def generation_node(state: WorkflowState) -> WorkflowState:

        now = time.time()
        with open("timing.log", "a", encoding="utf-8") as f:
            f.write(f"{state["iteration_count"]}|{now}\n")

        if state["iteration_count"] == 0:
            state["messages"] = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=objective.first_message()),
            ]
        else:
            if state["validation_error"]:
                state["messages"].append(
                    HumanMessage(content=state["validation_error"])
                )
                state["validation_error"] = ""


        #response = llm.invoke(state["messages"])
        response = rate_limit_sensible_llm_call(llm, state["messages"])
        state["messages"].append(response)
        state["raw_model_output"] = response.content.strip()
        state["iteration_count"] += 1

        now = time.time()
        with open("timing.log", "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1]

        _, last_ts = last_line.strip().split("|")
        last_time = float(last_ts)

        elapsed = now - last_time

        with open("timing.log", "a", encoding="utf-8") as f:
            f.write(f"LLM message took {elapsed:.3f}s\n")
            f.write(f"LLM message received at |{now}\n")
        
        return state

    return generation_node


def parse_node(state: WorkflowState) -> WorkflowState:
    raw = state["raw_model_output"]
    try:
        parsed = json.loads(raw)
        state["current_smiles"] = parsed["smiles"]
        state["current_reason"] = parsed["reason"]
        state["is_valid"] = True
        state["validation_error"] = ""
    except Exception:
        state["current_smiles"] = ""
        state["current_reason"] = ""
        state["is_valid"] = False
        state["validation_error"] = (
            "Invalid JSON. Provide proper JSON with fields 'smiles' and 'reason'."
        )
    return state


def validation_node(state: WorkflowState) -> WorkflowState:
    try:
        mol = Chem.MolFromSmiles(state["current_smiles"])
        if mol is None:
            state["is_valid"] = False
            state["validation_error"] = f"Invalid SMILES: {state['current_smiles']}"
        else:
            state["is_valid"] = True
            state["validation_error"] = ""
    except Exception as e:
        state["is_valid"] = False
        state["validation_error"] = f"Validation error: {str(e)}"
    return state


def make_prediction_node(objective: Objective):
    def prediction_node(state: WorkflowState) -> WorkflowState:
        result = objective.evaluate(state)

        now = time.time()
        with open("timing.log", "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1]

        _, last_ts = last_line.strip().split("|")
        last_time = float(last_ts)

        elapsed = now - last_time

        with open("timing.log", "a", encoding="utf-8") as f:
            f.write(f"Prediction from oracle received|{now}|elapsed={elapsed:.3f}s\n")

        state["oracle_result"] = result

        trace_entry = {
            "iteration": state["iteration_count"],
            "smiles": state["current_smiles"],
            "reason": state["current_reason"],
            "score": result["score"],
            "explanation": result["explanation"],
        }
        # Include individual scores from composite oracles
        if "scores" in result:
            trace_entry["scores"] = result["scores"]

        state["trace"].append(trace_entry)

        feedback = objective.build_feedback(state, result)
        state["messages"].append(HumanMessage(content=feedback))
        return state
    return prediction_node



def make_final_response_node(llm: ChatOpenAI):
    def final_response_node(state: WorkflowState) -> WorkflowState:
        # Extract the objective description from the first human message
        objective_context = state["messages"][1].content if len(state["messages"]) > 1 else ""
        
        # Remove explanations from trace so summary only sees what the generation
        # model actually saw (respects xai filtering in objectives)
        trace_for_summary = [
            {k: v for k, v in entry.items() if k != "explanation"}
            for entry in state['trace']
        ]
        
        summary_prompt = f"""
You are given the objective and trace of a molecular optimization loop.

Objective:
{objective_context}

Trace (each entry includes iteration, SMILES, reason, score):
{json.dumps(trace_for_summary, indent=2)}

Write a concise scientific summary of the optimization process.
Use ONLY the information provided. Do not invent any steps or molecules.
""".strip()

        #summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary_response = rate_limit_sensible_llm_call(llm, [HumanMessage(content=summary_prompt)])
        state["final_response"] = summary_response.content
        return state

    return final_response_node
