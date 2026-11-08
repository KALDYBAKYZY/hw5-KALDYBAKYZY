"""Part 3 memory tests with the FAKE model. No API key. Storage goes to a temp folder."""
import subprocess
import sys
from pathlib import Path

import pytest

from helpdesk.agent import build_agent, run_turn
from helpdesk.fake_model import FakeHelpdeskModel
from helpdesk.memory import UserMemory, open_checkpointer
from helpdesk.tools import SessionContext

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("HELPDESK_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("HELPDESK_TICKET_DIR", str(tmp_path / "tickets"))
    return tmp_path / "storage"


def open_app(user, thread):
    """Like starting the program: open storage from disk and build the graph."""
    cp, um = open_checkpointer(), UserMemory()
    g = build_agent(FakeHelpdeskModel(), SessionContext(user, thread), checkpointer=cp, user_memory=um)
    return g, cp, um


def close_app(cp, um):
    cp.conn.close()
    um.close()


def test_user_memory_is_per_user(tmp_storage):
    # catches: preferences stored globally instead of by user_id
    um = UserMemory()
    um.set("demo-user-a", "language", "Kazakh")
    assert um.get("demo-user-a") == {"language": "Kazakh"}
    assert um.get("demo-user-b") == {}


def test_user_memory_rejects_unknown_keys(tmp_storage):
    # catches: saving arbitrary data (e.g. secrets) in user memory
    assert UserMemory().set("demo-user-a", "password", "x")["saved"] is False


def test_same_thread_uses_earlier_messages():
    # catches: conversation state not passed to the model (S10)
    g, cp, um = open_app("demo-user-a", "s10")
    run_turn(g, "I cannot connect to Wi-Fi. I enabled it and forgot the saved network.", "s10")
    turn = run_turn(g, "Which steps did I already try?", "s10")
    assert turn["sources"]["conversation_state"]["messages_loaded_from_checkpoint"] >= 2
    assert "forgot the saved network" in turn["final_reply"]
    close_app(cp, um)


def test_conversation_survives_restart():
    # catches: in-memory checkpointer, or a new thread_id after restart (S11)
    g, cp, um = open_app("demo-user-a", "restart-a")
    run_turn(g, "My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails.", "restart-a")
    close_app(cp, um)                                   # "close the program"
    g, cp, um = open_app("demo-user-a", "restart-a")    # "start it again"
    turn = run_turn(g, "Which problem and steps did we discuss?", "restart-a")
    assert turn["sources"]["conversation_state"]["messages_loaded_from_checkpoint"] >= 2
    assert "Wi-Fi" in turn["final_reply"] and "reconnecting" in turn["final_reply"]
    close_app(cp, um)


def test_preference_in_new_thread_without_old_messages():
    # catches: preference kept only in the thread, or old transcript replayed (S12)
    g, cp, um = open_app("demo-user-a", "pref-1")
    first = run_turn(g, "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.", "pref-1")
    assert first["sources"]["user_memory_written"]
    close_app(cp, um)
    g, cp, um = open_app("demo-user-a", "pref-2")
    turn = run_turn(g, "Which language did I ask you to use?", "pref-2")
    assert turn["sources"]["conversation_state"]["messages_loaded_from_checkpoint"] == 0
    assert turn["sources"]["user_memory_read"] == {"language": "Kazakh"}
    assert "Kazakh" in turn["final_reply"]
    close_app(cp, um)


def test_other_user_does_not_see_preference():
    # catches: memory leaking between users (S13)
    g, cp, um = open_app("demo-user-a", "iso-a")
    run_turn(g, "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.", "iso-a")
    g_b = build_agent(FakeHelpdeskModel(), SessionContext("demo-user-b", "iso-b"),
                      checkpointer=cp, user_memory=um)
    turn = run_turn(g_b, "Have I told you a preferred language or a previous problem?", "iso-b")
    assert turn["sources"]["user_memory_read"] == {}
    assert "Kazakh" not in turn["final_reply"]
    assert "no preferred language" in turn["final_reply"].lower()
    close_app(cp, um)


def test_restart_runner_uses_separate_processes(tmp_storage):
    # catches: a fake "restart" inside one process
    subprocess.run([sys.executable, str(ROOT / "run_part3.py"), "--scenario", "S11", "--fake"],
                   check=True, cwd=ROOT, capture_output=True)
    import json
    logs = [json.loads((ROOT / "outputs" / "part3_fake" / f"S11_phase{n}.json").read_text()) for n in (1, 2)]
    assert logs[0]["process_id"] != logs[1]["process_id"]
    assert logs[1]["turns"][0]["sources"]["conversation_state"]["messages_loaded_from_checkpoint"] >= 2
