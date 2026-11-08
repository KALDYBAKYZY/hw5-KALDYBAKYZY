"""FAKE MODEL — for tests and CI only. It is NOT an LLM.

It follows fixed keyword rules but speaks the same protocol as a real chat model
(it returns AIMessage objects with tool_calls). This lets the tests check the
graph loop, the tools and the conditional ticket logic without an API key.
The real Part 2 demonstration must use a real model.
"""
import json
import re
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

STEP_WORDS = ("enabled", "forgot the saved", "reconnect", "tried", "restarted")


def _topic(text: str) -> str | None:
    t = text.lower()
    if "wi-fi" in t or "wifi" in t:
        return "wifi"
    if "password" in t:
        return "password"
    return None


def _steps(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|,\s*but\s+", text)
    return [p.strip(" .") for p in parts if any(w in p.lower() for w in STEP_WORDS)]


def _call(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                              "id": "fake_" + uuid.uuid4().hex[:8]}])


class FakeHelpdeskModel(BaseChatModel):
    is_fake: bool = True

    @property
    def _llm_type(self) -> str:
        return "fake-helpdesk"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._decide(messages))])

    # ---------------------------------------------------------------- rules
    def _decide(self, messages) -> AIMessage:
        humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        last_h = humans[-1]
        text = messages[last_h].content
        low = text.lower()
        prev_ai = next((m for m in reversed(messages[:last_h]) if isinstance(m, AIMessage)), None)
        results = {}
        for m in messages[last_h + 1:]:
            if isinstance(m, ToolMessage):
                results[m.name] = json.loads(m.content)

        # ---- Part 3 memory rules ----
        system = messages[0].content if messages and messages[0].type == "system" else ""
        saved = re.search(r"language=(\w+)", system)
        earlier = [messages[i].content for i in humans[:-1]]  # from conversation memory
        lang_word = re.search(r"replies in (\w+)", text)
        if "remember" in low and lang_word and "save_user_preference" not in results:
            return _call("save_user_preference", {"key": "language", "value": lang_word.group(1)})
        if "save_user_preference" in results:
            r = results["save_user_preference"]
            topic_now = _topic(text)
            extra = f" I noted your {'Wi-Fi' if topic_now == 'wifi' else 'password'} problem." if topic_now else ""
            return AIMessage(content=(f"[{r['value']}] Saved: I will reply in {r['value']}." if r.get("saved")
                                      else r.get("error", "Not saved.")) + extra)
        if "which language" in low or "preferred language" in low or "previous problem" in low:
            if not saved and not earlier:
                return AIMessage(content="No preferred language or previous problem is recorded for you. "
                                         "Which problem do you have: Wi-Fi or password?")
            lang = saved.group(1) if saved else None
            return AIMessage(content=(f"[{lang}] You asked me to reply in {lang}. " if lang else
                                      "No language preference is recorded. ")
                             + "I can help with Wi-Fi and password problems.")
        if "which steps" in low or "which problem" in low:
            if not earlier:
                return AIMessage(content="Nothing was discussed earlier in this conversation.")
            all_text = " ".join(earlier)
            steps = [s for e in earlier for s in _steps(e)]
            t = _topic(all_text)
            prefix = f"[{saved.group(1)}] " if saved else ""
            return AIMessage(content=prefix + f"You reported a {'Wi-Fi' if t == 'wifi' else 'password'} "
                                     f"problem. Steps you said you tried: {'; '.join(steps) or 'none'}.")

        topic = _topic(text)
        wants_steps = "steps" in low and ("show" in low or "troubleshoot" in low)
        wants_status = any(w in low for w in ("down", "status", "known problem"))
        continuing_ticket = bool(prev_ai and "ticket" in prev_ai.content.lower()
                                 and "what problem" in prev_ai.content.lower())
        wants_ticket = "ticket" in low or continuing_ticket
        conditional = "if it is operating" in low

        # 1) instructions
        if wants_steps and topic and "find_instructions" not in results:
            return _call("find_instructions", {"topic": topic})
        if "find_instructions" in results and not wants_ticket and not wants_status:
            r = results["find_instructions"]
            steps = "\n".join(f"{i}. {s}" for i, s in enumerate(r["steps"], 1))
            return AIMessage(content=f"{r['title']} ({r['instruction_id']}):\n{steps}")

        # 2) status
        if wants_status and topic and "check_service_status" not in results:
            return _call("check_service_status", {"topic": topic})
        status = results.get("check_service_status")

        # 3) ticket
        if wants_ticket:
            steps = _steps(text)
            if not topic or not steps:
                return AIMessage(content="To prepare a ticket I need details. What problem are you "
                                         "having, and what have you already tried?")
            if conditional and status and status["status"] == "outage":
                return AIMessage(content=f"Status: {status['message']} You asked for a ticket only if "
                                         "the service is operating, so I did not prepare it. "
                                         "The data gives no repair time.")
            if conditional and status and status["status"] == "unknown":
                return AIMessage(content=f"{status['message']} I cannot confirm that it is operating, "
                                         "so I did not prepare the ticket. Do you want a ticket draft "
                                         "saying the status could not be checked?")
            if "prepare_ticket" not in results:
                return _call("prepare_ticket", {
                    "summary": text if not continuing_ticket else text.split(".")[0],
                    "steps_tried": steps, "category": topic,
                    "service_status": status["status"] if status else "not checked",
                    "reference_ids": ["GUIDE-WIFI"] if topic == "wifi" else ["GUIDE-PASSWORD"]})
            t = results["prepare_ticket"]
            if not t.get("created"):
                return AIMessage(content=t.get("error", "I could not prepare the ticket."))
            st = f"Status: {status['message']} " if status else "Service status: not checked. "
            return AIMessage(content=f"{st}I prepared a local ticket draft {t['ticket_id']} "
                                     f"(file {t['file']}). Nothing has been sent.")

        if status:
            return AIMessage(content=f"Status of {status['service']}: {status['status']}. "
                                     f"{status['message']} No repair time is given.")
        return AIMessage(content="I can help with Wi-Fi or password problems. "
                                 "Which one is your problem?")
