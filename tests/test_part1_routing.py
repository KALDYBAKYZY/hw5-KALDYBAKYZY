import pytest

from helpdesk.part1_graph import build_graph, run_with_trace


@pytest.mark.parametrize("message, expected", [
    ("I cannot connect to Wi-Fi.", "wifi_help"),
    ("my WIFI is slow", "wifi_help"),                 
    ("I forgot my PASSWORD.", "password_help"),  
    ("My laptop screen is broken.", "other_help"),   
    ("My Wi-Fi password does not work.", "wifi_help"), 
])
def test_original_routes(message, expected):
    result = run_with_trace(build_graph("original"), message)
    assert result["route"] == ["read_request", expected]


def test_clarification_for_both_topics():
    result = run_with_trace(build_graph("with_clarification"), "My Wi-Fi password does not work.")
    assert result["route"] == ["read_request", "clarify_problem"]
    assert "which problem" in result["final_state"]["reply"].lower()


def test_clarification_keeps_single_topic_routes():
    g = build_graph("with_clarification")
    assert run_with_trace(g, "I forgot my password")["route"][-1] == "password_help"
    assert run_with_trace(g, "wi-fi down")["route"][-1] == "wifi_help"


def test_help_nodes_use_helpdesk_json():
    state = run_with_trace(build_graph(), "wifi")["final_state"]
    assert "GUIDE-WIFI" in state["reply"]
    assert state["category"] == "wifi"
    assert state["message"] == "wifi" 
