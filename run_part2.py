import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from helpdesk.agent import MAX_TOOL_ROUNDS, build_agent, get_model, model_label, run_turn
from helpdesk.tools import (SessionContext, check_service_status_impl, find_instructions_impl,
                            prepare_ticket_impl, ticket_dir)

ROOT = Path(__file__).resolve().parent
PART2 = ["S06", "S07", "S08", "S09", "S14"]


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def direct_calls() -> None:
    """Each tool works without the agent."""
    out_dir = ROOT / "outputs" / "part2" / "direct_tickets"
    shutil.rmtree(out_dir, ignore_errors=True)
    calls = {
        "find_instructions('wifi')": find_instructions_impl("wifi"),
        "find_instructions('password')": find_instructions_impl("password"),
        "find_instructions('printer')": find_instructions_impl("printer"),
        "check_service_status('wifi', 'normal')": check_service_status_impl("wifi", "normal"),
        "check_service_status('wifi', 'wifi_outage')": check_service_status_impl("wifi", "wifi_outage"),
        "check_service_status('wifi', 'status_unavailable')": check_service_status_impl("wifi", "status_unavailable"),
        "check_service_status('password', 'recovery_outage')": check_service_status_impl("password", "recovery_outage"),
        "prepare_ticket(full details)": prepare_ticket_impl(
            summary="Campus-Demo Wi-Fi will not connect", steps_tried=["enabled Wi-Fi", "forgot saved network"],
            category="wifi", user_id="demo-user-a", thread_id="direct", status_profile="normal",
            service_status="operational", reference_ids=["GUIDE-WIFI"], out_dir=out_dir),
        "prepare_ticket(no steps)": prepare_ticket_impl(
            summary="Wi-Fi fails", steps_tried=[], category="wifi", user_id="demo-user-a",
            thread_id="direct", status_profile="normal", out_dir=out_dir),
    }
    for name, result in calls.items():
        print(f"\n>>> {name}\n{json.dumps(result, indent=2, ensure_ascii=False)}")
    save(ROOT / "outputs" / "part2" / "direct_tool_calls.json", calls)
    print("\nSaved outputs/part2/direct_tool_calls.json")


def run_scenario(sc: dict, model, label: str) -> dict:
    events = [e for e in sc["events"] if e["action"] == "message"]
    ctx = SessionContext(user_id=events[0]["user_id"], thread_id=events[0]["thread_id"],
                         status_profile=sc["status_profile"])
    if sc.get("reset_before"):
        for f in ticket_dir().glob(f"TKT-{ctx.thread_id}-*.json"):
            f.unlink()
    graph = build_agent(model, ctx, checkpointer=InMemorySaver()) 
    print(f"\n===== {sc['id']} — {sc['title']} | profile={ctx.status_profile} =====")
    turns = []
    for ev in events:
        turn = run_turn(graph, ev["message"], ctx.thread_id)
        print(f"\nUSER: {ev['message']}")
        for step in turn["trace"]:
            for r in step.get("tool_requests", []):
                print(f"  [{step['node']}] model requests {r['name']}({r['args']})")
            for r in step.get("tool_results", []):
                print(f"  [{step['node']}] {r['name']} -> {r['result'][:150]}")
        print(f"HELPDESK: {turn['final_reply']}")
        turns.append(turn)
    log = {"scenario": sc["id"], "title": sc["title"],
           "command": f"python run_part2.py --scenario {sc['id']}" + (" --fake" if "FAKE" in label else ""),
           "model": label, "status_profile": ctx.status_profile,
           "user_id": ctx.user_id, "thread_id": ctx.thread_id,
           "max_tool_rounds": MAX_TOOL_ROUNDS, "turns": turns}
    return log


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--direct", action="store_true")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--scenario")
    a = p.parse_args()
    if a.direct:
        direct_calls()
        return
    model = get_model(fake=a.fake)
    label = model_label(model)
    print(f"Model: {label}")
    scenarios = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    wanted = [a.scenario] if a.scenario else PART2
    folder = ROOT / "outputs" / ("part2_fake" if a.fake else "part2")
    for sc in scenarios:
        if sc["id"] in wanted:
            save(folder / f"{sc['id']}.json", run_scenario(sc, model, label))
    print(f"\nLogs saved to {folder.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
