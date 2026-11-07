"""Part 1 — a small helpdesk graph WITHOUT an LLM.

Two variants:
  * "original"            START -> read_request -> wifi_help | password_help | other_help -> END
  * "with_clarification"  same, plus clarify_problem when both topics appear
"""
import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_helpdesk() -> dict:
    with open(DATA_DIR / "helpdesk.json", encoding="utf-8") as f:
        return json.load(f)


HELPDESK = load_helpdesk()
ALIASES = HELPDESK["topic_aliases"]  # {"wifi": ["wifi", "wi-fi"], "password": ["password"]}


class HelpdeskState(TypedDict, total=False):
    message: str   # the user's original message
    category: str  # set by read_request: wifi | password | both | other
    reply: str     # set by a help node


# ---------- helper: format instructions from helpdesk.json ----------
def format_instructions(topic: str) -> str:
    guide = HELPDESK["instructions"][topic]
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(guide["steps"], 1))
    return f"{guide['title']} ({guide['instruction_id']}):\n{steps}"


# ---------- nodes ----------
def read_request(state: HelpdeskState) -> dict:
    """Reads `message`, writes `category`. Case-insensitive."""
    text = state["message"].lower()
    has_wifi = any(word in text for word in ALIASES["wifi"])
    has_password = any(word in text for word in ALIASES["password"])
    if has_wifi and has_password:
        category = "both"
    elif has_wifi:
        category = "wifi"
    elif has_password:
        category = "password"
    else:
        category = "other"
    return {"category": category}


def wifi_help(state: HelpdeskState) -> dict:
    return {"reply": format_instructions("wifi")}


def password_help(state: HelpdeskState) -> dict:
    return {"reply": format_instructions("password")}


def other_help(state: HelpdeskState) -> dict:
    return {"reply": "I can help only with Wi-Fi or password problems. "
                     "Is your problem about Wi-Fi or about a password?"}


def clarify_problem(state: HelpdeskState) -> dict:
    return {"reply": "Your message mentions both Wi-Fi and a password. "
                     "Which problem do you want to solve first: Wi-Fi or password?"}


# ---------- routers (conditional edges) ----------
def route_original(state: HelpdeskState) -> str:
    """Original rules: Wi-Fi is checked FIRST, so 'both' goes to wifi_help."""
    category = state["category"]
    if category in ("wifi", "both"):
        return "wifi_help"
    if category == "password":
        return "password_help"
    return "other_help"


def route_with_clarification(state: HelpdeskState) -> str:
    """Changed rules: if both topics appear, ask which one first."""
    if state["category"] == "both":
        return "clarify_problem"
    return route_original(state)


# ---------- build ----------
def build_graph(variant: str = "original"):
    builder = StateGraph(HelpdeskState)
    builder.add_node("read_request", read_request)
    builder.add_node("wifi_help", wifi_help)
    builder.add_node("password_help", password_help)
    builder.add_node("other_help", other_help)

    targets = ["wifi_help", "password_help", "other_help"]
    if variant == "original":
        router = route_original
    elif variant == "with_clarification":
        builder.add_node("clarify_problem", clarify_problem)
        targets.append("clarify_problem")
        router = route_with_clarification
    else:
        raise ValueError(f"unknown variant: {variant}")

    builder.add_edge(START, "read_request")
    builder.add_conditional_edges("read_request", router, targets)
    for node in targets:
        builder.add_edge(node, END)
    return builder.compile()


def run_with_trace(graph, message: str) -> dict:
    """Run the graph and record the node order and the FULL state after each node."""
    trace = []
    state: dict = {"message": message}
    # stream_mode="updates" gives {node_name: update} after each node.
    for chunk in graph.stream({"message": message}, stream_mode="updates"):
        for node, update in chunk.items():
            state = {**state, **(update or {})}
            trace.append({"node": node, "update": update, "state_after": dict(state)})
    return {
        "message": message,
        "route": [step["node"] for step in trace],
        "trace": trace,
        "final_state": state,
    }
