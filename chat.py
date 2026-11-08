"""Interactive helpdesk in the terminal (Part 3 graph: tools + both memories).

Usage:
    python chat.py --user demo-user-a --thread t1
    python chat.py --user demo-user-a --thread t1 --profile wifi_outage
    python chat.py --user demo-user-b --thread t2 --fake
Type 'exit' to quit. Start again with the same --user and --thread to continue the conversation.
"""
import argparse

from dotenv import load_dotenv

from helpdesk.agent import build_agent, get_model, model_label, run_turn
from helpdesk.memory import UserMemory, open_checkpointer
from helpdesk.tools import SessionContext

PROFILES = ["normal", "wifi_outage", "recovery_outage", "status_unavailable"]


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--user", default="demo-user-a")
    p.add_argument("--thread", default="chat-1")
    p.add_argument("--profile", default="normal", choices=PROFILES)
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()

    model = get_model(fake=a.fake)
    ctx = SessionContext(a.user, a.thread, a.profile)
    graph = build_agent(model, ctx, checkpointer=open_checkpointer(), user_memory=UserMemory())
    print(f"Model: {model_label(model)} | user={a.user} thread={a.thread} status={a.profile}")
    print("Type 'exit' to quit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("exit", "quit"):
            break
        if not text:
            continue
        turn = run_turn(graph, text, a.thread)
        tools = [r["name"] for s in turn["trace"] for r in s.get("tool_requests", [])]
        print(f"  (nodes: {' -> '.join(turn['nodes'])}; tools: {tools or 'none'})")
        print(f"Helpdesk: {turn['final_reply']}\n")


if __name__ == "__main__":
    main()
