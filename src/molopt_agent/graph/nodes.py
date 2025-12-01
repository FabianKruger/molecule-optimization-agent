import json
from rdkit import Chem
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import WorkflowState
from ..objectives.base import Objective


def make_generation_node(objective: Objective, llm: ChatOpenAI, system_prompt: str):
    def generation_node(state: WorkflowState) -> WorkflowState:
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

        response = llm.invoke(state["messages"])
        state["messages"].append(response)
        state["raw_model_output"] = response.content.strip()
        state["iteration_count"] += 1
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
        state["oracle_result"] = result

        state["trace"].append(
            {
                "iteration": state["iteration_count"],
                "smiles": state["current_smiles"],
                "reason": state["current_reason"],
                "score": result["score"],
                "explanation": result["explanation"],
            }
        )

        feedback = objective.build_feedback(state, result)
        state["messages"].append(HumanMessage(content=feedback))
        return state
    return prediction_node



def make_final_response_node(llm: ChatOpenAI):
    def final_response_node(state: WorkflowState) -> WorkflowState:
        summary_prompt = f"""
You are given the factual trace of every iteration of a molecular optimization loop.
Each entry includes:
- iteration index
- SMILES
- the model's reason
- oracle metrics
- oracle explanation

Trace:
{json.dumps(state['trace'], indent=2)}

Write a concise scientific summary of the optimization process.
Use ONLY the information in the trace. Do not invent any steps or molecules.
""".strip()

        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        state["final_response"] = summary_response.content
        return state

    return final_response_node
