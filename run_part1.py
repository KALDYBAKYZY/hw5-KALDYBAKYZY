import json
import sys
from pathlib import Path

from helpdesk.part1_graph import build_graph, run_with_trace

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "part1"

MESSAGES = [
    "I cannot connect to Wi-Fi.",
    "I forgot my PASSWORD.",
    "My laptop screen is broken.",
    "My Wi-Fi password does not work.",
]

PREDICTIONS = {
    "I cannot connect to Wi-Fi.": ["read_request", "wifi_help"],
    "I forgot my PASSWORD.": ["read_request", "password_help"],
    "My laptop screen is broken.": ["read_request", "other_help"],
    "My Wi-Fi password does not work.": ["read_request", "wifi_help"],
}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def print_run(result: dict) -> None:
    print(f"\nMessage: {result['message']}")
    print("Route:   START -> " + " -> ".join(result["route"]) + " -> END")
    for step in result["trace"]:
        print(f"  after {step['node']:<15} state = "
              f"{ {k: (v[:40] + '...' if isinstance(v, str) and len(v) > 40 else v) for k, v in step['state_after'].items()} }")


def main() -> None:
    graphs = {v: build_graph(v) for v in ("original", "with_clarification")}

    if "--draw" in sys.argv:
        for name, g in graphs.items():
            print(f"\n--- Mermaid: {name} ---")
            print(g.get_graph().draw_mermaid())

    print("=== Part 1: original graph — prediction vs actual ===")
    comparison = []
    for msg in MESSAGES:
        result = run_with_trace(graphs["original"], msg)
        print_run(result)
        ok = result["route"] == PREDICTIONS[msg]
        print(f"  predicted: {PREDICTIONS[msg]}  -> {'MATCH' if ok else 'DIFFERENT'}")
        comparison.append({"message": msg, "predicted": PREDICTIONS[msg],
                           "actual": result["route"], "match": ok,
                           "trace": result["trace"]})
    save(OUT / "predictions_original.json", {"graph_variant": "original", "runs": comparison})

    print("\n=== Part 1: changed graph (with_clarification) ===")
    changed = [run_with_trace(graphs["with_clarification"], m) for m in MESSAGES]
    for r in changed:
        print_run(r)
    save(OUT / "runs_with_clarification.json",
         {"graph_variant": "with_clarification", "runs": changed})

    scenarios = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))
    for sc in scenarios["scenarios"]:
        if sc["part"] != 1:
            continue
        variant = sc.get("graph_variant", "original")
        for ev in sc["events"]:
            result = run_with_trace(graphs[variant], ev["message"])
            log = {
                "scenario": sc["id"], "title": sc["title"],
                "command": "python run_part1.py",
                "graph_variant": variant,
                "status_profile": sc["status_profile"],
                "user_id": ev["user_id"], "thread_id": ev["thread_id"],
                **result,
                "final_reply": result["final_state"].get("reply"),
            }
            save(OUT / f"{sc['id']}.json", log)
            print(f"{sc['id']} ({variant}): " + " -> ".join(result["route"]))

    print(f"\nLogs saved to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
