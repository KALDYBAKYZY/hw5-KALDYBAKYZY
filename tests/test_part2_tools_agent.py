"""Part 2 tests. They use the FAKE model (helpdesk/fake_model.py), so no API key is needed."""
import json
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from helpdesk.agent import build_agent, run_turn
from helpdesk.fake_model import FakeHelpdeskModel
from helpdesk.tools import (SessionContext, check_service_status_impl, find_instructions_impl,
                            prepare_ticket_impl)

CONDITIONAL = ("I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not "
               "connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem.")


@pytest.fixture(autouse=True)
def tmp_tickets(tmp_path, monkeypatch):
    monkeypatch.setenv("HELPDESK_TICKET_DIR", str(tmp_path / "tickets"))
    return tmp_path / "tickets"


# ---------- tools called directly (no agent) ----------
def test_find_instructions_reads_helpdesk_json():
    # catches: wrong file / wrong key / invented instructions
    assert find_instructions_impl("Wi-Fi")["instruction_id"] == "GUIDE-WIFI"
    assert find_instructions_impl("password")["instruction_id"] == "GUIDE-PASSWORD"
    assert find_instructions_impl("printer")["found"] is False


@pytest.mark.parametrize("topic, profile, service, status", [
    ("wifi", "normal", "wifi", "operational"),
    ("wifi", "wifi_outage", "wifi", "outage"),
    ("wifi", "status_unavailable", "wifi", "unknown"),
    ("password", "recovery_outage", "account_recovery", "outage"),  # password -> account_recovery
])
def test_check_service_status(topic, profile, service, status):
    # catches: ignoring the selected snapshot, or wrong topic->service mapping
    r = check_service_status_impl(topic, profile)
    assert (r["service"], r["status"]) == (service, status)


def test_prepare_ticket_writes_local_draft(tmp_tickets):
    # catches: ticket not saved, wrong template fields, or claiming it was sent
    r = prepare_ticket_impl(summary="Wi-Fi fails", steps_tried=["enabled Wi-Fi"], category="wifi",
                            user_id="demo-user-a", thread_id="t1", status_profile="normal")
    assert r["created"] and r["sent"] is False
    saved = json.loads((tmp_tickets / "TKT-t1-001.json").read_text())
    assert saved["status"] == "draft" and saved["steps_tried"] == ["enabled Wi-Fi"]
    assert saved["service_status"] == "not checked"


def test_prepare_ticket_refuses_missing_details(tmp_tickets):
    # catches: inventing details instead of asking the user
    r = prepare_ticket_impl(summary="", steps_tried=[], category="wifi",
                            user_id="demo-user-a", thread_id="t2", status_profile="normal")
    assert r["created"] is False and len(r["missing"]) == 2
    assert not tmp_tickets.exists() or not list(tmp_tickets.iterdir())


# ---------- the graph loop with the FAKE model ----------
def _run(profile, messages, thread="t"):
    ctx = SessionContext("demo-user-a", thread, profile)
    g = build_agent(FakeHelpdeskModel(), ctx, checkpointer=InMemorySaver())
    return [run_turn(g, m, thread) for m in messages]


def _tools_used(turn):
    return [r["name"] for s in turn["trace"] for r in s.get("tool_requests", [])]


def test_same_request_operational_makes_ticket(tmp_tickets):
    # catches: status result not returned to the model
    turn = _run("normal", [CONDITIONAL])[-1]
    assert _tools_used(turn) == ["check_service_status", "prepare_ticket"]
    assert turn["nodes"] == ["agent", "tools", "agent", "tools", "agent"]
    assert list(tmp_tickets.glob("*.json"))


@pytest.mark.parametrize("profile", ["wifi_outage", "status_unavailable"])
def test_same_request_without_operational_status_makes_no_ticket(profile, tmp_tickets):
    # catches: always making a ticket; treating 'unknown' as operational
    turn = _run(profile, [CONDITIONAL])[-1]
    assert _tools_used(turn) == ["check_service_status"]
    assert not tmp_tickets.exists() or not list(tmp_tickets.glob("*.json"))


def test_missing_details_asks_first():
    # catches: creating a ticket from nothing
    first, second = _run("normal", ["Prepare a support ticket.",
                                    "The issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails."])
    assert _tools_used(first) == [] and "what problem" in first["final_reply"].lower()
    assert _tools_used(second) == ["prepare_ticket"]


class AlwaysToolModel(FakeHelpdeskModel):
    """Fake model that never stops asking for a tool — to test the loop limit."""
    def _decide(self, messages):
        return AIMessage(content="", tool_calls=[{"name": "check_service_status",
                                                  "args": {"topic": "wifi"},
                                                  "id": uuid.uuid4().hex}])


def test_loop_limit_stops_endless_tool_calls():
    # catches: an infinite agent <-> tools loop
    ctx = SessionContext("demo-user-a", "loop", "normal")
    g = build_agent(AlwaysToolModel(), ctx, checkpointer=InMemorySaver(), max_tool_rounds=3)
    turn = run_turn(g, "Is Wi-Fi down?", "loop")
    assert turn["loop_limit_reached"] is True
    assert turn["nodes"].count("tools") == 3
    assert turn["nodes"][-1] == "limit_reached"
