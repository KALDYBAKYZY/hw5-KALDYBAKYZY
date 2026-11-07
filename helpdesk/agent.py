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


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation (appended, not replaced)
    tool_rounds: int            # how many times the tools node ran in this user turn
    loop_limit_reached: bool


def build_agent(model, ctx: SessionContext, checkpointer=None, system_extra: str = "",
                max_tool_rounds: int = MAX_TOOL_ROUNDS):
    tools = make_tools(ctx)
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)
    system = SYSTEM_PROMPT + ("\n" + system_extra if system_extra else "")

    def agent(state: AgentState) -> dict:
        """The MODEL decides: request tool(s) or answer."""
        reply = model_with_tools.invoke([SystemMessage(system)] + state["messages"])
        return {"messages": [reply]}

    def tools_step(state: AgentState) -> dict:
        """MY PROGRAM executes the tools the model requested."""
        result = tool_node.invoke(state)
        return {"messages": result["messages"],
                "tool_rounds": state.get("tool_rounds", 0) + 1}

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
    """Send one user message. Return the trace of this turn (nodes, tool calls, results)."""
    config = {"configurable": {"thread_id": thread_id}}
    trace = []
    for chunk in graph.stream({"messages": [("user", message)], "tool_rounds": 0,
                               "loop_limit_reached": False},
                              config, stream_mode="updates"):
        for node, update in chunk.items():
            step = {"node": node}
            for m in (update or {}).get("messages", []):
                if isinstance(m, AIMessage) and m.tool_calls:
                    step.setdefault("tool_requests", []).extend(
                        {"name": c["name"], "args": c["args"]} for c in m.tool_calls)
                elif isinstance(m, ToolMessage):
                    step.setdefault("tool_results", []).append(
                        {"name": m.name, "result": m.content})
                elif isinstance(m, AIMessage):
                    step["model_answer"] = m.text
            if "tool_rounds" in (update or {}):
                step["tool_rounds"] = update["tool_rounds"]
            trace.append(step)
    state = graph.get_state(config).values
    return {"user_message": message,
            "nodes": [s["node"] for s in trace],
            "trace": trace,
            "final_reply": state["messages"][-1].text,
            "tool_rounds": state.get("tool_rounds", 0),
            "loop_limit_reached": state.get("loop_limit_reached", False)}