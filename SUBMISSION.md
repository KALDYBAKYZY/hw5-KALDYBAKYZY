# HW5 — My LangGraph helpdesk

## Part 1 — How the graph works

### My graph and state

Code: `helpdesk/part1_graph.py`. Run: `python run_part1.py` (add `--draw` to print the graphs as Mermaid).

```
START
  │
  ▼
read_request            (sets category)
  │
  ├── category = wifi or both ──► wifi_help      (sets reply) ──► END
  ├── category = password ──────► password_help  (sets reply) ──► END
  └── category = other ─────────► other_help     (sets reply) ──► END
```

State (`HelpdeskState`):

| Field | What it holds | Written by |
| --- | --- | --- |
| `message` | The user's original message | input |
| `category` | `wifi`, `password`, `both` or `other` | `read_request` |
| `reply` | The answer for the user | a help node |

What each node reads and changes:

| Node | Reads | Changes |
| --- | --- | --- |
| `read_request` | `message` | `category` |
| `wifi_help` | nothing from state (reads `data/helpdesk.json`) | `reply` |
| `password_help` | nothing from state (reads `data/helpdesk.json`) | `reply` |
| `other_help` | nothing | `reply` |
| router `route_original` (conditional edge) | `category` | nothing — it only returns the name of the next node |

**1. What does each node read and change?** See the table above. Only `read_request` looks at the message. The help nodes only write `reply`.

**2. Node update vs conditional edge.** A node returns a dict, for example `{"category": "wifi"}`. LangGraph merges this dict into the state. A conditional edge does not change the state. It is a function that reads the state and returns a node name, for example `"wifi_help"`. Then LangGraph runs that node.

**3. What does the next node receive?** It receives the full current state, not only the last update. For example, after `read_request` the help node gets `{message, category}`. The `message` is still there. My trace in `outputs/part1/` shows this: `state_after` keeps all old fields.

**4. START, END and compile.** `START` is the entry point: the first edge goes from `START` to `read_request`. `END` means the run stops and the final state is returned. `compile()` checks the graph (all nodes exist, edges are correct) and makes a runnable object with `.invoke()` and `.stream()`.

### Predictions before running

My rule order: Wi-Fi is checked first, then password, then other.

| Message | Predicted route | Actual route | What I learned |
| --- | --- | --- | --- |
| I cannot connect to Wi-Fi. | read_request → wifi_help | read_request → wifi_help | `wi-fi` is found after `.lower()` |
| I forgot my PASSWORD. | read_request → password_help | read_request → password_help | Upper case works because I use `.lower()` |
| My laptop screen is broken. | read_request → other_help | read_request → other_help | No keyword → fallback node asks Wi-Fi or password |
| My Wi-Fi password does not work. | read_request → wifi_help | read_request → wifi_help | Both words appear. The Wi-Fi rule is checked first, so the password part is ignored. |

Logs: `outputs/part1/predictions_original.json`, `outputs/part1/S01.json` … `S04.json`.

**Two topics.** In `My Wi-Fi password does not work.`, `read_request` sets `category = "both"`. The original router checks Wi-Fi first (`if category in ("wifi", "both")`), so it goes to `wifi_help`. The user gets Wi-Fi steps, but maybe the real problem is the Wi-Fi password. This is the weak point of the original rules.

### My change to the graph

I added the node `clarify_problem` and a new router `route_with_clarification`. Build it with `build_graph("with_clarification")`.

```
START
  │
  ▼
read_request            (sets category)
  │
  ├── category = wifi ──────► wifi_help        ──► END
  ├── category = password ──► password_help    ──► END
  ├── category = both ──────► clarify_problem  ──► END    ← NEW
  └── category = other ─────► other_help       ──► END
```

New trace for `My Wi-Fi password does not work.` (`outputs/part1/S05.json`):

```
START -> read_request -> clarify_problem -> END
after read_request:    {message: "My Wi-Fi password does not work.", category: "both"}
after clarify_problem: {..., reply: "Your message mentions both Wi-Fi and a password. Which problem do you want to solve first: Wi-Fi or password?"}
```

The other three messages keep the same routes (`outputs/part1/runs_with_clarification.json`).

**What LangGraph controls:** the order of running nodes, starting at `START` and stopping at `END`, calling the router after `read_request`, merging each node's update into the shared state, and passing the state to the next node.

**What my Python functions decide:** how to find the topic (keyword search, case-insensitive), which node is next (the `if` rules in the router), and the text of each reply. There is no model in Part 1. All decisions are my fixed rules, so the same message always gives the same route.

## Part 2 — Model and tools

Code: `helpdesk/tools.py` (tools), `helpdesk/agent.py` (graph), `run_part2.py` (runner).

**Model and provider:** `gpt-5.6-luna`, OpenAI (Chat Completions API), through the `langchain-openai` adapter, with `reasoning_effort=none`.

Why `reasoning_effort=none`: my first run failed with error 400. OpenAI said that this reasoning model cannot use function tools in `/v1/chat/completions` unless `reasoning_effort` is `none`. I added the `REASONING_EFFORT` setting to `.env`. Reasoning models do not accept a custom temperature, so the code does not send `temperature` when this setting is used.

### The model/tool loop

```
START
  │
  ▼
agent  ◄─────────────────────────────┐   (MODEL: tool call or answer)
  │                                  │
  ├── tool_calls, rounds < 5 ──► tools ──┘   (PROGRAM runs the tool, tool_rounds + 1)
  │
  ├── tool_calls, rounds >= 5 ──► limit_reached ──► END
  │
  └── no tool_calls (answer) ───► END
```

1. `agent` node: the **model** reads the messages and returns an `AIMessage`. This message has either `tool_calls` (a request) or normal text (the answer).
2. `route_after_agent` (conditional edge): looks at the last message.
3. `tools` node: **my program** runs the requested Python functions and adds a `ToolMessage` with each result.
4. Back to `agent`. The model now sees the result and decides again.

### Three tools

| Tool | Reads | Does |
| --- | --- | --- |
| `find_instructions(topic)` | `data/helpdesk.json` | Returns the title, steps and `instruction_id` for `wifi` or `password` |
| `check_service_status(topic)` | `data/service_status.json`, selected snapshot | `wifi` → service `wifi`, `password` → `account_recovery`. Returns `operational`, `outage` or `unknown` |
| `prepare_ticket(summary, steps_tried, category, service_status, reference_ids)` | `data/ticket_template.json` | Saves a **local** draft `outputs/tickets/TKT-<thread>-NNN.json`. Refuses if the problem or the steps are missing. Never sends anything |

The model chooses only the topic and the ticket text. The **application** sets `user_id`, `thread_id` and the status snapshot (`SessionContext`). The model cannot change them.

**Direct calls without the agent:** `python run_part2.py --direct` → `outputs/part2/direct_tool_calls.json`. Each tool has a plain function (`find_instructions_impl`, `check_service_status_impl`, `prepare_ticket_impl`). The model-facing tools only wrap these functions.

### Loop limit

`MAX_TOOL_ROUNDS = 5` (setting in `.env`). The state field `tool_rounds` starts at 0 for every user message, and the `tools` node adds 1 each time it runs. The limit is checked in the conditional edge `route_after_agent`: if the model asks for a tool again and `tool_rounds >= 5`, the graph goes to `limit_reached` instead of `tools`. That node closes the open tool calls, writes a short message and goes to `END`. The log field `loop_limit_reached` shows if this happened. Test: `test_loop_limit_stops_endless_tool_calls`.

### Answers

**1. How does the graph decide between running a tool and ending?** The conditional edge `route_after_agent` reads the last message. If it has `tool_calls` → `tools` (or `limit_reached`). If it has no tool calls → `END`. The model decides *what* to write; my edge function turns that into the next node.

**2. Who requests a tool and who executes it?** The **model** only requests: it returns the tool name and arguments in `tool_calls`. The model never runs code. **My program** executes: the `tools` node (LangGraph `ToolNode`) calls my Python function and returns the result.

**3. What changes in the state after a tool result returns?** A `ToolMessage` with the result (JSON) is appended to `messages` (the `add_messages` reducer adds, it does not replace). `tool_rounds` grows by 1. On the next `agent` step the model sees the result.

**4. How is this different from Part 1?** In Part 1 the route is fixed by my `if` rules on keywords, so it is always the same. In Part 2 the **model** chooses the tools from the meaning of the message and from earlier tool results, and the graph has a **loop** (`tools → agent`). The same request can follow different paths when the data is different.

### Actual run with several tools (S06, second message)

Command: `python run_part2.py` → `outputs/part2/S06.json`. Second message:

```
USER: I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect.
      Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem.
[agent] model requests check_service_status({'topic': 'wifi'})
[agent] model requests find_instructions({'topic': 'wifi'})
[tools] check_service_status -> status "operational"
[tools] find_instructions    -> GUIDE-WIFI
[agent] model requests prepare_ticket({'summary': 'Campus-Demo will not connect even after Wi-Fi
        was enabled and the saved network was forgotten.', 'steps_tried': ['Enabled Wi-Fi',
        'Forgot the saved network'], 'category': 'wifi', 'service_status': 'operational',
        'reference_ids': ['GUIDE-WIFI']})
[tools] prepare_ticket -> created TKT-s06-001, sent: false
HELPDESK: Wi-Fi is operational with no reported outage. I prepared a local ticket draft ...
          It was not sent. File: outputs/tickets/TKT-s06-001.json
```

Route: `agent → tools → agent → tools → agent → END` (2 tool rounds, limit not reached).

What I see: the model asked for **two tools in one round** (status and instructions), and my `tools` node ran both. It called `prepare_ticket` only **after** it saw `operational`. The `steps_tried` are only the steps the user said. The ticket file is `outputs/tickets/TKT-s06-001.json`.

### Same request, different snapshots

| Scenario | Snapshot | Tools used | Result |
| --- | --- | --- | --- |
| S06 | normal | check_service_status, find_instructions, prepare_ticket | Status operational → ticket `TKT-s06-001` was made |
| S07 | wifi_outage | find_instructions, check_service_status | Status outage → no ticket. The model said the condition was not met and no repair time is given |
| S08 | status_unavailable | check_service_status, find_instructions | Status unknown → no ticket. The model said it is not confirmed up or down and asked if I want a ticket saying the status could not be checked |

Expected: `operational` → ticket draft. `outage` → no ticket, the condition was not met. `unknown` → no ticket, the assistant asks whether to make one saying the status could not be checked.

## Part 3 — Memory

Code: `helpdesk/memory.py`, `helpdesk/agent.py` (`load_user_memory` node), `helpdesk/tools.py` (`save_user_preference`), `run_part3.py`, `chat.py`.

```
START
  │
  ▼
load_user_memory        (reads language by user_id)
  │
  ▼
agent  ◄─────────────────────────────┐
  │                                  │
  ├── tool_calls, rounds < 5 ──► tools ──┘
  │
  ├── tool_calls, rounds >= 5 ──► limit_reached ──► END
  │
  └── no tool_calls (answer) ───► END

Conversation memory: SqliteSaver, key = thread_id  (local-storage/conversations.sqlite)
User memory:         UserMemory,  key = user_id    (local-storage/user_memory.sqlite)
```

I use **two separate kinds of memory**. Both are saved on disk in `local-storage/` (this folder is in `.gitignore`).

| | Conversation memory | User memory |
| --- | --- | --- |
| What it holds | All messages of one conversation, tool results, `tool_rounds` (the full graph state) | Saved preferences of one person, e.g. `{"language": "Kazakh"}` |
| Key | `thread_id` | `user_id` |
| How | LangGraph checkpointer `SqliteSaver` passed to `compile()` | My own small SQLite table `preferences(user_id, key, value, updated_at)` |
| File | `local-storage/conversations.sqlite` | `local-storage/user_memory.sqlite` |
| Read | Automatically by LangGraph at the start of each run with the same `thread_id` | By the `load_user_memory` node at the start of each turn |
| Written | Automatically by LangGraph after each node | By the tool `save_user_preference`, only when the user asks to remember |

**`thread_id` vs user identifier.** A `thread_id` identifies **one conversation**. A `user_id` identifies **one person**. One user can have many threads. A new thread starts with empty conversation memory, but the same user memory. Different users never share user memory. Both identifiers are set by **the application** (`SessionContext`, the command line), not by the model. The model cannot read or write another user's data, because `save_user_preference` and `load_user_memory` always use `ctx.user_id`.

**Retrieval vs memory.** `find_instructions` reads `data/helpdesk.json`. This is **retrieval**: fixed reference data, the same for every user and every conversation, and nobody changes it while the program runs. The chosen language is **memory**: it is learned from what this user said, it is saved by the program, and it belongs only to this user.

**Only a safe preference is saved.** `UserMemory.set` accepts only the key `language` and a short value, so passwords or other text cannot be saved there.

**Real restart.** `run_part3.py` splits a scenario at every `restart_process` event. For each part it starts a **new Python process** (`--phase N`). The first process ends before the next one starts. Only `local-storage/` connects them. Each log records the `process_id`, so you can see that the processes are different. I never paste the old conversation into the new input.

**What each log shows.** For every turn the log has `sources`:
- `conversation_state.messages_loaded_from_checkpoint` — how many earlier messages LangGraph loaded for this `thread_id`;
- `user_memory_read` — what `load_user_memory` read for this `user_id`;
- `user_memory_written` — what `save_user_preference` saved;
- `reference_files` — which data files the tools read (`data/helpdesk.json`, `data/service_status.json`).

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
| Refer to an earlier message (S10) | `python run_part3.py --scenario S10` | `outputs/part3/S10_phase1.json` | Turn 1: the model used `find_instructions` (reference file `data/helpdesk.json`) and `check_service_status` (operational). Turn 2 "Which steps did I already try?": **6 earlier messages loaded from the checkpoint** for thread `s10`, no tools. It listed: enabling Wi-Fi, forgetting the saved network, reconnecting. Source: conversation memory |
| Close, restart and resume (S11) | `python run_part3.py --scenario S11` (2 processes) | `outputs/part3/S11_phase1.json`, `S11_phase2.json` | Process 1 (pid 86771): the model called `save_user_preference(language=Kazakh)` → user memory, then `find_instructions`, and replied in Kazakh. The process ended. Process 2 (pid 86986), same `thread_id restart-a`: **6 earlier messages loaded from `conversations.sqlite`**, user memory read `{language: Kazakh}`. It recalled the Wi-Fi problem and the two steps (enabled, reconnecting) in Kazakh. It also correctly listed guide steps the user did NOT say they tried. I did not paste the old conversation |
| Same user, new conversation (S12) | `python run_part3.py --scenario S12` (2 processes) | `outputs/part3/S12_phase1.json`, `S12_phase2.json` | Process 1, thread `preference-a1`: preference saved with `save_user_preference`. Process 2, **new** thread `preference-a2`: **0 earlier messages loaded** (old conversation not replayed), user memory read `{language: Kazakh}`. Reply in Kazakh: "You asked to use Kazakh. I can help only with Wi-Fi and password recovery." Source: user memory only |
| Different user (S13) | `python run_part3.py --scenario S13` | `outputs/part3/S13_phase1.json` | `demo-user-a` saved Kazakh. Then `demo-user-b` (thread `isolation-b`) in the same storage: user memory read `{}`, 0 earlier messages. Reply in English: "No language preference is recorded, and no previous problem has been discussed." The preference of `demo-user-a` did not appear |

Interactive check: `python chat.py --user demo-user-a --thread t1`, type a message, type `exit`, start the same command again and ask what we discussed.

## Part 4 — Running and checking my work

- **Setup and commands:** see the section "How to run my work" at the top of `README.md` (Python 3.12, virtual environment, `.env`, every command).
- **Package versions:** `requirements.txt` has exact versions (`==`) of the packages I used.
- **Memory files:** `local-storage/conversations.sqlite` and `local-storage/user_memory.sqlite`. Clear them with `python run_part3.py --reset` (or delete the `local-storage/` folder). Scenarios with `reset_before: true` clear them automatically.
- **Tickets:** local drafts in `outputs/tickets/`.
- **CI:** `.github/workflows/tests.yml` runs `pytest` on every push. It has no API key.

**Fake model.** All tests use `helpdesk/fake_model.py`, a clearly labelled rule-based fake model. It is **not** an LLM. It returns the same message format as a real model (`AIMessage` with `tool_calls`), so the tests check my graph, tools and memory without a key. The real model demonstrations (`outputs/part2/`, `outputs/part3/`) were made with `gpt-5.6-luna` and are separate. The fake model logs, if made, go to `outputs/part2_fake/` and `outputs/part3_fake/` (not committed).

| Test file | Test | Mistake it catches |
| --- | --- | --- |
| `test_part1_routing.py` | upper/lower case messages | forgetting `.lower()` |
| | laptop screen → `other_help` | no fallback route |
| | Wi-Fi password → `wifi_help` (original) | changing the rule order by mistake |
| | Wi-Fi password → `clarify_problem` | the new node is not connected |
| | single topic in the changed graph | the change broke the old rules |
| | reply contains `GUIDE-WIFI` | hard-coded text instead of `helpdesk.json` |
| `test_part2_tools_agent.py` | `find_instructions` direct | wrong file or invented instructions |
| | `check_service_status` × 4 snapshots | ignoring the snapshot; wrong `password → account_recovery` mapping |
| | `prepare_ticket` writes a draft | ticket not saved, or marked as sent |
| | `prepare_ticket` with missing details | inventing details instead of asking |
| | same request, `normal` → ticket | tool result not returned to the model |
| | same request, `wifi_outage` / `status_unavailable` → no ticket | always making a ticket; treating `unknown` as operational |
| | missing details → asks first | ticket made from nothing |
| | endless tool requests stop after the limit | infinite agent ↔ tools loop |
| `test_part3_memory.py` | preferences per user | preferences saved globally |
| | unknown key refused | saving secrets in user memory |
| | same thread recalls steps (S10) | conversation state not given to the model |
| | close and reopen storage, same thread (S11) | in-memory checkpointer; memory lost on restart |
| | new thread: 0 old messages, preference still there (S12) | preference kept only in the thread; old transcript replayed |
| | `demo-user-b` does not see `demo-user-a` (S13) | memory leaking between users |
| | runner uses two different process ids | a fake "restart" inside one process |

## Choices and limitations

**Choices.**
- **LangGraph `StateGraph`** for all parts, so I can see each node and the state after it.
- **`langchain-openai` + `gpt-5.6-luna`** because I have an OpenAI key. This reasoning model needs `reasoning_effort=none` to use tools in Chat Completions, so I made it a setting in `.env`.
- **`ToolNode`** to run the tools, and my own counter `tool_rounds` for the loop limit, so the limit is visible in the state and in the logs.
- **`SqliteSaver`** for conversation memory: it is the official LangGraph checkpointer, saves to one local file and survives a restart. No server is needed.
- **A small SQLite table** for user memory: it is simple, keyed by `user_id`, and clearly separate from conversation state. I did not put the preference into the thread state, because then a new thread would not see it.
- The application, not the model, sets `user_id`, `thread_id` and the status snapshot.

**Limitations.**
- The model output is not fully the same every time. Sometimes it calls an extra tool (for example `find_instructions` together with `check_service_status`). The logs show what really happened.
- Only one preference (`language`) can be saved. The model must call `save_user_preference`; if it forgets, the preference is not saved.
- The quality of the Kazakh replies depends on the model. I did not check the grammar.
- Topic matching in Part 1 is simple keyword search. For example, `wireless` is not recognised.
- Tickets are only local JSON files; nothing is sent.
- SQLite files are for one local user. It is not made for many people at the same time.

## Supplied scenarios — my actual results

| Scenario | Command or manual steps | Log file | What happened / what I changed |
| --- | --- | --- | --- |
| S01 — Simple Wi-Fi route | `python run_part1.py` | `outputs/part1/S01.json` | read_request → wifi_help. Category `wifi`, reply from GUIDE-WIFI. Matches my prediction |
| S02 — Case-insensitive password route | `python run_part1.py` | `outputs/part1/S02.json` | read_request → password_help. `PASSWORD` works because of `.lower()` |
| S03 — Unsupported request | `python run_part1.py` | `outputs/part1/S03.json` | read_request → other_help. It asks: Wi-Fi or password? |
| S04 — Two topics before the graph change | `python run_part1.py` (original graph) | `outputs/part1/S04.json` | Category `both`, but route → wifi_help, because the Wi-Fi rule is checked first. The password part is lost |
| S05 — Two topics after adding clarification | `python run_part1.py` (with_clarification graph) | `outputs/part1/S05.json` | read_request → clarify_problem. It asks which problem to solve first. I changed the router |
| S06 — Instructions then local ticket | `python run_part2.py --scenario S06` | `outputs/part2/S06.json`, `outputs/tickets/TKT-s06-001.json` | Turn 1: find_instructions. Turn 2: check_service_status (operational) → prepare_ticket. Draft saved, not sent |
| S07 — The same request during a Wi-Fi outage | `python run_part2.py --scenario S07` | `outputs/part2/S07.json` | check_service_status → outage. No ticket, no repair time invented |
| S08 — Status source unavailable | `python run_part2.py --scenario S08` | `outputs/part2/S08.json` | Status unknown. No ticket. Asks if I want a ticket saying the status could not be checked |
| S09 — Missing ticket details | `python run_part2.py --scenario S09` | `outputs/part2/S09.json`, `outputs/tickets/TKT-s09-001.json` | Turn 1: no tools, asks for the problem and steps. Turn 2: ticket with only the user's steps. The model also checked status (not asked, but it is recorded as operational) |
| S10 — Use the current conversation | `python run_part3.py --scenario S10` | `outputs/part3/S10_phase1.json` | Second message: 6 messages loaded from the checkpoint, the model listed the three steps the user tried. No tool needed |
| S11 — Persist and recall after restart | `python run_part3.py --scenario S11` (2 processes) | `outputs/part3/S11_phase1.json`, `S11_phase2.json` | Different pids (86771, 86986). After restart: 6 messages loaded for `restart-a`, language Kazakh read from user memory, problem and steps recalled in Kazakh |
| S12 — Preference across threads without replay | `python run_part3.py --scenario S12` (2 processes) | `outputs/part3/S12_phase1.json`, `S12_phase2.json` | New thread: 0 messages loaded, `{language: Kazakh}` from user memory, reply in Kazakh |
| S13 — Do not share memory between users | `python run_part3.py --scenario S13` | `outputs/part3/S13_phase1.json` | `demo-user-b`: user memory `{}`, 0 messages. Says nothing is recorded. No leak from `demo-user-a` |
| S14 — Password recovery outage | `python run_part2.py --scenario S14` | `outputs/part2/S14.json` | password → `account_recovery` service → outage. No repair time invented |
