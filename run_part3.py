"""Part 3 — memory demonstrations S10-S13 with REAL process restarts.

Usage:
    python run_part3.py                      # run S10, S11, S12, S13 (each phase = new process)
    python run_part3.py --scenario S11       # one scenario
    python run_part3.py --fake               # same with the labelled FAKE model (no key)
    python run_part3.py --reset              # delete all saved memory (local-storage/)

A scenario is split into phases at every "restart_process" event.
The parent starts a NEW Python process for each phase (--phase N), so the first
process really ends before the next one starts. Only local-storage/ connects them.
Each log records the process id (pid) to show this.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PART3 = ["S10", "S11", "S12", "S13"]


def load_scenario(sid: str) -> dict:
    data = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))
    return next(s for s in data["scenarios"] if s["id"] == sid)


def phases(sc: dict) -> list[list[dict]]:
    out = [[]]
    for ev in sc["events"]:
        if ev["action"] == "restart_process":
            out.append([])
        else:
            out[-1].append(ev)
    return out


def run_phase(sid: str, n: int, fake: bool) -> None:
    """Runs inside ONE process. Opens the saved storage, sends this phase's messages, exits."""
    from helpdesk.agent import MAX_TOOL_ROUNDS, build_agent, get_model, model_label, run_turn
    from helpdesk.memory import UserMemory, open_checkpointer, reset_storage, storage_dir
    from helpdesk.tools import SessionContext

    sc = load_scenario(sid)
    if n == 1 and sc.get("reset_before"):
        reset_storage()
    model = get_model(fake=fake)
    checkpointer = open_checkpointer()          # conversation memory (thread_id)
    user_memory = UserMemory()                  # user memory (user_id)
    turns = []
    print(f"\n----- {sid} phase {n} | new process pid={os.getpid()} -----")
    for ev in phases(sc)[n - 1]:
        # user_id and thread_id are chosen by the APPLICATION, not by the model
        ctx = SessionContext(ev["user_id"], ev["thread_id"], sc["status_profile"])
        graph = build_agent(model, ctx, checkpointer=checkpointer, user_memory=user_memory)
        turn = run_turn(graph, ev["message"], ev["thread_id"])
        turn.update(user_id=ev["user_id"], thread_id=ev["thread_id"])
        src = turn["sources"]
        print(f"\n[{ev['user_id']} / thread {ev['thread_id']}] USER: {ev['message']}")
        print(f"  conversation memory: {src['conversation_state']['messages_loaded_from_checkpoint']} "
              f"earlier messages loaded for thread '{ev['thread_id']}'")
        print(f"  user memory read:    {src['user_memory_read']}")
        for step in turn["trace"]:
            for r in step.get("tool_requests", []):
                print(f"  [{step['node']}] model requests {r['name']}({r['args']})")
            for r in step.get("tool_results", []):
                print(f"  [{step['node']}] {r['name']} -> {r['result'][:120]}")
        print(f"HELPDESK: {turn['final_reply']}")
        turns.append(turn)
    try:
        shown = str(storage_dir().relative_to(ROOT))
    except ValueError:
        shown = str(storage_dir())
    log = {"scenario": sid, "title": sc["title"], "phase": n,
           "phases_total": len(phases(sc)),
           "command": f"python run_part3.py --scenario {sid}" + (" --fake" if fake else "")
                      + f"  (child process: --phase {n})",
           "process_id": os.getpid(), "reset_before_this_phase": n == 1 and sc.get("reset_before", False),
           "model": model_label(model), "status_profile": sc["status_profile"],
           "storage": {"conversation_memory": f"{shown}/conversations.sqlite (LangGraph SqliteSaver)",
                       "user_memory": f"{shown}/user_memory.sqlite"},
           "max_tool_rounds": MAX_TOOL_ROUNDS, "turns": turns}
    folder = ROOT / "outputs" / ("part3_fake" if fake else "part3")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{sid}_phase{n}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    user_memory.close()
    checkpointer.conn.close()


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--scenario")
    p.add_argument("--phase", type=int)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--reset", action="store_true")
    a = p.parse_args()

    if a.reset:
        from helpdesk.memory import reset_storage, storage_dir
        reset_storage()
        print(f"Deleted {storage_dir()}")
        return
    if a.phase:                       # child process
        run_phase(a.scenario, a.phase, a.fake)
        return

    for sid in ([a.scenario] if a.scenario else PART3):   # parent: start one process per phase
        sc = load_scenario(sid)
        print(f"\n===== {sid} — {sc['title']} =====")
        for n in range(1, len(phases(sc)) + 1):
            if n > 1:
                print("\n*** restart_process: the previous process has exited. Starting a new one. ***")
            cmd = [sys.executable, __file__, "--scenario", sid, "--phase", str(n)] + (["--fake"] if a.fake else [])
            subprocess.run(cmd, check=True)
    print("\nLogs saved to outputs/" + ("part3_fake/" if a.fake else "part3/"))


if __name__ == "__main__":
    main()
