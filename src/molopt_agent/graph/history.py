"""Shortened-history views of the conversation sent to the LLM.

`state["messages"]` always keeps the complete conversation (full explanations in
every feedback message). For the "last_k" and "best_worst_k" policies the LLM is
instead shown a modified copy. The selection (which valid iterations are
"explained") only uses the trace, i.e. valid molecules and their scores. Two
forms exist:

- "compact": all iterations stay in the prompt, but the explanations of the
  non-selected ones are replaced by a placeholder (needs the objective's
  `build_compact_feedback`).
- "truncated": non-selected iterations are removed completely (the assistant
  message and its feedback), so the model gets no information about them. Works
  for any objective.
"""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

HISTORY_POLICIES = ("full", "last_k", "best_worst_k")
HISTORY_FORMS = ("compact", "truncated")


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
    state: dict[str, Any],
    policy: str,
    k: int,
    form: str = "compact",
    max_iterations: int | None = None,
) -> tuple[list[BaseMessage], dict[str, Any] | None]:
    """Build the message list for the next LLM call plus a log entry describing it.

    Never mutates `state["messages"]`. For the "full" policy the original messages
    are returned and the log entry is None. Otherwise a log entry is returned for
    every generation call (including retries after invalid attempts), so
    `generation_call` counts all LLM generation calls.
    """
    if policy == "full":
        return state["messages"], None
    if form == "compact":
        return _build_compact_view(state, policy, k)
    if form == "truncated":
        return _build_truncated_view(state, policy, k, max_iterations)
    raise ValueError(f"Unknown history form '{form}'. Available: {', '.join(HISTORY_FORMS)}")


def _build_compact_view(
    state: dict[str, Any], policy: str, k: int
) -> tuple[list[BaseMessage], dict[str, Any]]:
    messages = state["messages"]
    trace = state["trace"]
    selection = select_explained(trace, policy, k)
    omitted = set(selection["omitted"])
    compact_feedback = state["compact_feedback"]

    view = list(messages)
    for entry in trace:
        if entry["iteration"] in omitted:
            compact = compact_feedback[entry["iteration"]]
            view[compact["msg_index"]] = HumanMessage(content=compact["text"])

    log_entry = _log_entry(state, policy, k, "compact", selection)
    log_entry["omitted_iterations"] = selection["omitted"]
    return view, log_entry


def _log_entry(
    state: dict[str, Any], policy: str, k: int, form: str, selection: dict[str, list[int]]
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "generation_call": len(state["context_log"]) + 1,
        "iteration": state["iteration_count"] + 1,
        "policy": policy,
        "form": form,
        "k": k,
        "explained_iterations": selection["explained"],
    }
    if policy == "best_worst_k":
        entry["best_iterations"] = selection["best"]
        entry["worst_iterations"] = selection["worst"]
    return entry


def _build_truncated_view(
    state: dict[str, Any], policy: str, k: int, max_iterations: int | None
) -> tuple[list[BaseMessage], dict[str, Any]]:
    """Keep the system message, the task description and only the selected iterations.

    Generation call n is the n-th AIMessage; it is followed by its feedback (valid
    molecule) or by the error message (invalid attempt). Such an assistant/feedback
    chunk is kept or dropped as a whole, so user and assistant turns keep
    alternating. An invalid attempt is only kept while it is the most recent
    attempt, so the model sees the error it has to fix.
    """
    messages = state["messages"]
    trace = state["trace"]
    selection = select_explained(trace, policy, k)
    kept_valid = set(selection["explained"])
    valid = {entry["iteration"] for entry in trace}

    ai_positions = [i for i, m in enumerate(messages) if isinstance(m, AIMessage)]
    n_calls = len(ai_positions)
    if not ai_positions:
        return list(messages), _truncated_log(state, policy, k, selection, [], False)

    view: list[BaseMessage] = list(messages[: ai_positions[0]])
    dropped: list[int] = []
    for call in range(1, n_calls + 1):
        start = ai_positions[call - 1]
        end = ai_positions[call] if call < n_calls else len(messages)
        is_latest = call == n_calls
        if call in kept_valid or (is_latest and call not in valid):
            view.extend(messages[start:end])
        else:
            dropped.append(call)

    # If the feedback for the latest molecule was dropped, the prompt could be identical
    # to the previous one; add the iteration counter (and nothing else about it).
    counter_appended = n_calls in valid and n_calls not in kept_valid
    if counter_appended and max_iterations is not None and isinstance(view[-1], HumanMessage):
        counter = f"Iteration: {state['iteration_count']} / {max_iterations}"
        view[-1] = HumanMessage(content=f"{view[-1].content}\n\n{counter}")
    else:
        counter_appended = False

    return view, _truncated_log(state, policy, k, selection, dropped, counter_appended)


def _truncated_log(
    state: dict[str, Any],
    policy: str,
    k: int,
    selection: dict[str, list[int]],
    dropped: list[int],
    counter_appended: bool,
) -> dict[str, Any]:
    entry = _log_entry(state, policy, k, "truncated", selection)
    # All removed generation calls (valid iterations and invalid attempts)
    entry["dropped_iterations"] = dropped
    entry["counter_appended"] = counter_appended
    return entry
