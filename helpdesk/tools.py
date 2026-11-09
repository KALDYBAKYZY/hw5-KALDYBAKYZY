import json
import os
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _load(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


HELPDESK = _load("helpdesk.json")
STATUS = _load("service_status.json")
TICKET_TEMPLATE = _load("ticket_template.json")["ticket"]
VALID_TOPICS = list(HELPDESK["instructions"])  


def ticket_dir() -> Path:
    return Path(os.getenv("HELPDESK_TICKET_DIR", ROOT / "outputs" / "tickets"))


def _normalise_topic(topic: str) -> str | None:
    t = (topic or "").strip().lower()
    for name, aliases in HELPDESK["topic_aliases"].items():
        if t == name or t in aliases:
            return name
    return None


def find_instructions_impl(topic: str) -> dict:
    """Retrieval from data/helpdesk.json."""
    name = _normalise_topic(topic)
    if name is None:
        return {"found": False, "topic": topic,
                "error": f"No instructions for '{topic}'. Supported topics: {VALID_TOPICS}."}
    guide = HELPDESK["instructions"][name]
    return {"found": True, "topic": name, "instruction_id": guide["instruction_id"],
            "title": guide["title"], "steps": guide["steps"],
            "source": "data/helpdesk.json"}


def check_service_status_impl(topic_or_service: str, status_profile: str) -> dict:
    key = (topic_or_service or "").strip().lower()
    topic = _normalise_topic(key)
    service = HELPDESK["topic_services"][topic] if topic else key
    profile = STATUS["profiles"].get(status_profile)
    if profile is None:
        return {"service": service, "status": "unknown",
                "message": f"Unknown status profile '{status_profile}'."}
    if service not in profile:
        return {"service": service, "status": "unknown",
                "message": f"No status data for service '{service}'."}
    entry = profile[service]
    return {"service": service, "status": entry["status"], "message": entry["message"],
            "status_profile": status_profile, "source": "data/service_status.json"}


SECRET_WORDS = ("password:", "api key", "recovery code")


def prepare_ticket_impl(*, summary: str, steps_tried: list[str], category: str,
                        user_id: str, thread_id: str, status_profile: str,
                        service_status: str = "not checked",
                        reference_ids: list[str] | None = None,
                        out_dir: Path | None = None) -> dict:
    missing = []
    if not (summary or "").strip():
        missing.append("problem description")
    steps = [s.strip() for s in (steps_tried or []) if s and s.strip()]
    if not steps:
        missing.append("steps already tried")
    if missing:
        return {"created": False, "missing": missing,
                "error": "Ask the user for: " + ", ".join(missing) + ". Do not invent details."}
    text = (summary + " " + " ".join(steps)).lower()
    if any(w in text for w in SECRET_WORDS):
        return {"created": False, "error": "Do not put passwords, API keys or recovery codes in a ticket."}

    out = Path(out_dir) if out_dir else ticket_dir()
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob(f"TKT-{thread_id}-*.json"))) + 1
    ticket = dict(TICKET_TEMPLATE)
    ticket.update({
        "ticket_id": f"TKT-{thread_id}-{n:03d}",
        "status": "draft",
        "user_id": user_id,
        "thread_id": thread_id,
        "category": _normalise_topic(category) or category,
        "summary": summary.strip(),
        "steps_tried": steps,
        "service_status": service_status or "not checked",
        "status_profile": status_profile,
        "reference_ids": reference_ids or [],
    })
    path = out / f"{ticket['ticket_id']}.json"
    path.write_text(json.dumps(ticket, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        shown = str(path.relative_to(ROOT))
    except ValueError:
        shown = str(path)
    return {"created": True, "ticket_id": ticket["ticket_id"], "file": shown,
            "sent": False, "ticket": ticket}


@dataclass
class SessionContext:
    user_id: str
    thread_id: str
    status_profile: str = "normal"


def make_tools(ctx: SessionContext, user_memory=None) -> list:
    @tool
    def find_instructions(topic: str) -> dict:
        return find_instructions_impl(topic)

    @tool
    def check_service_status(topic: str) -> dict:
        return check_service_status_impl(topic, ctx.status_profile)

    @tool
    def prepare_ticket(summary: str, steps_tried: list[str], category: str,
                       service_status: str = "not checked",
                       reference_ids: list[str] | None = None) -> dict:
        return prepare_ticket_impl(summary=summary, steps_tried=steps_tried, category=category,
                                   user_id=ctx.user_id, thread_id=ctx.thread_id,
                                   status_profile=ctx.status_profile,
                                   service_status=service_status, reference_ids=reference_ids)

    tools = [find_instructions, check_service_status, prepare_ticket]

    if user_memory is not None:  
        @tool
        def save_user_preference(key: str, value: str) -> dict:
            return user_memory.set(ctx.user_id, key, value)  

        tools.append(save_user_preference)
    return tools
