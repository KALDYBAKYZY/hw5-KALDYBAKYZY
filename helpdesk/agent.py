"""Part 2 + Part 3 — a model chooses tools inside a LangGraph loop.

Part 2 graph (no user memory):
    START -> agent --(tool calls, rounds < limit)--> tools -> agent  (loop)
                   --(tool calls, rounds >= limit)--> limit_reached -> END
                   --(no tool calls)----------------> END

Part 3 graph (user memory given): the same, but it starts with
    START -> load_user_memory -> agent -> ...
Conversation memory is the checkpointer passed to compile() (key: thread_id).
"""
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from helpdesk.tools import SessionContext, make_tools

MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "5"))

SYSTEM_PROMPT = """You are a practice university IT helpdesk assistant (fictional demo data).
You help only with Wi-Fi and password (account recovery) problems.
Rules:
- Use find_instructions to get steps. Never invent instructions.
- Use check_service_status when the user asks if a service is down or has a problem.
  Wi-Fi -> topic 'wifi'. Password recovery -> topic 'password'.
- Status 'unknown' means you could not check. It is NOT operational and NOT a confirmed outage.
- If the user says "prepare a ticket IF it is operating", only call prepare_ticket when the
  status is 'operational'. On 'outage', do not make the ticket and say so. On 'unknown', do not
  make it; ask whether they want a ticket that says the status could not be checked.
- prepare_ticket needs the problem and the steps the USER said they tried. If either is
  missing, ask the user. Never invent steps or details. Pass the checked status or 'not checked'.
- Tickets are local drafts. Never say a ticket was sent. Give the ticket file name.
- Never ask for or store passwords, API keys or recovery codes. Never invent a repair time."""

MEMORY_PROMPT = """Memory rules:
- Earlier messages of THIS conversation are in the message history (conversation memory).
  Use them for questions like "which steps did I try?" or "I already tried that".
  If there are no earlier messages, say that nothing was discussed before in this conversation.
- When the user asks you to remember a preference (e.g. reply language), call
  save_user_preference(key='language', value=...). Do not save anything else.
- Saved preferences of this user (user memory, loaded by the application): {prefs}
  If a language is saved, reply in that language. If nothing is saved, do not guess:
  say that no preference is recorded. Never mention other users."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation (appended, not replaced)
    tool_rounds: int            # how many times the tools node ran in this user turn
    loop_limit_reached: bool
    user_preferences: dict      # Part 3: copy of user memory, refreshed every turn


def _format_prefs(prefs: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in prefs.items()) if prefs else "none recorded"


def build_agent(model, ctx: SessionContext, checkpointer=None, system_extra: str = "",
                max_tool_rounds: int = MAX_TOOL_ROUNDS, user_memory=None):
    tools = make_tools(ctx, user_memory)
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)

    def system_text(state: AgentState) -> str:
        text = SYSTEM_PROMPT
        if user_memory is not None:
            text += "\n" + MEMORY_PROMPT.format(prefs=_format_prefs(state.get("user_preferences", {})))
        return text + ("\n" + system_extra if system_extra else "")

    def load_user_memory(state: AgentState) -> dict:
        """Part 3: read THIS user's saved preferences. user_id comes from the application."""
        return {"user_preferences": user_memory.get(ctx.user_id)}

    def agent(state: AgentState) -> dict:
        """The MODEL decides: request tool(s) or answer."""
        reply = model_with_tools.invoke([SystemMessage(system_text(state))] + state["messages"])
        return {"messages": [reply]}

    def tools_step(state: AgentState) -> dict:
        """MY PROGRAM executes the tools the model requested."""
        result = tool_node.invoke(state)
        update = {"messages": result["messages"],
                  "tool_rounds": state.get("tool_rounds", 0) + 1}
        if user_memory is not None:  # a preference may have just been saved
            update["user_preferences"] = user_memory.get(ctx.user_id)
        return update

    def limit_reached(state: AgentState) -> dict:
        """Stop the loop. Close open tool calls so the history stays valid."""
        last = state["messages"][-1]
        skipped = [ToolMessage(content="Not executed: tool loop limit reached.",
                               tool_call_id=c["id"]) for c in last.tool_calls]
        final = AIMessage(content="I stopped because I reached the tool-call limit "
                                  f"({max_tool_rounds}). Please rephrase or ask a smaller question.")
        return {"messages": skipped + [final], "loop_limit_reached": True}

    def route_after_agent(state: AgentState) -> str:
        """Conditional edge: tool calls -> run tools (if under the limit), else END."""
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        if state.get("tool_rounds", 0) >= max_tool_rounds:
            return "limit_reached"
        return "tools"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools_step)
    builder.add_node("limit_reached", limit_reached)
    if user_memory is not None:
        builder.add_node("load_user_memory", load_user_memory)
        builder.add_edge(START, "load_user_memory")
        builder.add_edge("load_user_memory", "agent")
    else:
        builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "limit_reached", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("limit_reached", END)
    return builder.compile(checkpointer=checkpointer)


def get_model(fake: bool = False):
    """Real model from .env, or the clearly labelled fake model for tests/CI."""
    if fake:
        from helpdesk.fake_model import FakeHelpdeskModel
        return FakeHelpdeskModel()
    from langchain_openai import ChatOpenAI
    name = os.getenv("MODEL_NAME")
    if not name or not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY and MODEL_NAME in .env (or use --fake).")
    kwargs = {"model": name, "base_url": os.getenv("OPENAI_BASE_URL") or None}
    effort = os.getenv("REASONING_EFFORT")
    if effort:
        # Reasoning models (gpt-5.x): tools in Chat Completions need reasoning_effort='none'.
        # These models do not accept a custom temperature, so it is not sent.
        kwargs["reasoning_effort"] = effort
    else:
        kwargs["temperature"] = 0
    if os.getenv("OPENAI_USE_RESPONSES_API", "").lower() == "true":
        kwargs["use_responses_api"] = True
    return ChatOpenAI(**kwargs)


def model_label(model) -> str:
    if getattr(model, "is_fake", False):
        return "FAKE rule-based test model (no LLM)"
    base = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    extra = f", reasoning_effort={os.getenv('REASONING_EFFORT')}" if os.getenv("REASONING_EFFORT") else ""
    return f"{os.getenv('MODEL_NAME')} via OpenAI-compatible API at {base}{extra}"


def run_turn(graph, message: str, thread_id: str) -> dict:
    """Send one user message. Return the trace of this turn (nodes, tool calls, results)
    and where the information came from (conversation state, user memory, reference file)."""
    config = {"configurable": {"thread_id": thread_id}}
    before = graph.get_state(config).values.get("messages", []) if graph.checkpointer else []
    trace, user_memory_read, user_memory_written, reference = [], None, [], []
    for chunk in graph.stream({"messages": [("user", message)], "tool_rounds": 0,
                               "loop_limit_reached": False},
                              config, stream_mode="updates"):
        for node, update in chunk.items():
            update = update or {}
            step = {"node": node}
            if node == "load_user_memory":
                user_memory_read = update.get("user_preferences", {})
                step["user_preferences"] = user_memory_read
            for m in update.get("messages", []):
                if isinstance(m, AIMessage) and m.tool_calls:
                    step.setdefault("tool_requests", []).extend(
                        {"name": c["name"], "args": c["args"]} for c in m.tool_calls)
                elif isinstance(m, ToolMessage):
                    step.setdefault("tool_results", []).append(
                        {"name": m.name, "result": m.content})
                    if m.name == "find_instructions":
                        reference.append("data/helpdesk.json")
                    if m.name == "check_service_status":
                        reference.append("data/service_status.json")
                    if m.name == "save_user_preference":
                        user_memory_written.append(m.content)
                elif isinstance(m, AIMessage):
                    step["model_answer"] = m.text
            if "tool_rounds" in update:
                step["tool_rounds"] = update["tool_rounds"]
            trace.append(step)
    state = graph.get_state(config).values
    return {"user_message": message,
            "nodes": [s["node"] for s in trace],
            "trace": trace,
            "sources": {
                "conversation_state": {"thread_id": thread_id,
                                       "messages_loaded_from_checkpoint": len(before)},
                "user_memory_read": user_memory_read,
                "user_memory_written": user_memory_written,
                "reference_files": sorted(set(reference)),
            },
            "final_reply": state["messages"][-1].text,
            "tool_rounds": state.get("tool_rounds", 0),
            "loop_limit_reached": state.get("loop_limit_reached", False)}
