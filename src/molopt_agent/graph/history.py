"""Shortened-history views of the conversation sent to the LLM.

`state["messages"]` always keeps the complete conversation (full explanations in
every feedback message). For the "last_k" and "best_worst_k" policies the LLM is
instead shown a copy in which the explanations of some earlier feedback messages
are replaced by a compact version. The selection only uses the trace (valid
molecules and their scores), so it works for any objective that provides
compact feedback.
"""

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

HISTORY_POLICIES = ("full", "last_k", "best_worst_k")


def select_explained(trace: list[dict], policy: str, k: int) -> dict[str, list[int]]:
    """Choose which valid iterations keep their explanation in the LLM context.

    Returns a dict with the iteration numbers under "explained" and "omitted"
    (both in chronological order) and, for best_worst_k, "best" and "worst".
    """
    iterations = [entry["iteration"] for entry in trace]
    result: dict[str, list[int]] = {}

    if policy == "full":
        explained = set(iterations)
    elif policy == "last_k":
        explained = set(iterations[-k:])
    elif policy == "best_worst_k":
        # Ties are broken in favour of the earlier iteration.
        best = sorted(trace, key=lambda e: (-e["score"], e["iteration"]))[:k]
        worst = sorted(trace, key=lambda e: (e["score"], e["iteration"]))[:k]
        result["best"] = [e["iteration"] for e in best]
        result["worst"] = [e["iteration"] for e in worst]
        if len(trace) <= 2 * k:
            # Behaves like "full"; best/worst are still reported (they overlap here).
            explained = set(iterations)
        else:
            explained = set(result["best"]) | set(result["worst"])
    else:
        raise ValueError(
            f"Unknown history policy '{policy}'. Available: {', '.join(HISTORY_POLICIES)}"
        )

    result["explained"] = [i for i in iterations if i in explained]
    result["omitted"] = [i for i in iterations if i not in explained]
    return result


def build_llm_messages(
    state: dict[str, Any], policy: str, k: int
) -> tuple[list[BaseMessage], dict[str, Any] | None]:
    """Build the message list for the next LLM call plus a log entry describing it.

    Never mutates `state["messages"]`. For the "full" policy the original messages
    are returned and the log entry is None. Otherwise a log entry is returned for
    every generation call (including retries after invalid attempts), so
    `generation_call` counts all LLM generation calls.
    """
    messages = state["messages"]
    if policy == "full":
        return messages, None

    trace = state["trace"]
    selection = select_explained(trace, policy, k)
    omitted = set(selection["omitted"])
    compact_feedback = state["compact_feedback"]

    view = list(messages)
    for entry in trace:
        if entry["iteration"] in omitted:
            compact = compact_feedback[entry["iteration"]]
            view[compact["msg_index"]] = HumanMessage(content=compact["text"])

    log_entry: dict[str, Any] = {
        "generation_call": len(state["context_log"]) + 1,
        "iteration": state["iteration_count"] + 1,
        "policy": policy,
        "k": k,
        "explained_iterations": selection["explained"],
        "omitted_iterations": selection["omitted"],
    }
    if policy == "best_worst_k":
        log_entry["best_iterations"] = selection["best"]
        log_entry["worst_iterations"] = selection["worst"]

    return view, log_entry
