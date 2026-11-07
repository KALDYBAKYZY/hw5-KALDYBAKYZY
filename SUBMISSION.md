# HW5 — My LangGraph helpdesk

## Part 1 — How the graph works

### My graph and state

Code: `helpdesk/part1_graph.py`. Run: `python run_part1.py` (add `--draw` to print the graphs as Mermaid).

```mermaid
graph TD
    START([START]) --> read_request
    read_request -.wifi / both.-> wifi_help
    read_request -.password.-> password_help
    read_request -.other.-> other_help
    wifi_help --> END([END])
    password_help --> END
    other_help --> END
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

```mermaid
graph TD
    START([START]) --> read_request
    read_request -.wifi.-> wifi_help
    read_request -.password.-> password_help
    read_request -.both.-> clarify_problem
    read_request -.other.-> other_help
    wifi_help --> END([END])
    password_help --> END
    clarify_problem --> END
    other_help --> END
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

TODO: draw the model/tool loop. Answer the four questions in the brief. Record the model and provider, direct calls to each tool, and one actual run that uses multiple tools. Explain the loop limit.

## Part 3 — Memory

TODO: explain conversation state, thread_id, user identifier, user preference storage, and retrieval from helpdesk.json.

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
| Refer to an earlier message | | | |
| Close, restart and resume | | | |
| Same user, new conversation | | | |
| Different user | | | |

## Part 4 — Running and checking my work

TODO: point to the README setup steps, package versions, memory-reset instructions, tests and GitHub Actions workflow. Explain which mistake each test catches and which runs use a fake model.

## Choices and limitations

TODO: explain the storage and libraries I chose, why they fit this task, and what still does not work reliably.

## Supplied scenarios — my actual results

| Scenario | Command or manual steps | Log file | What happened / what I changed |
| --- | --- | --- | --- |
| S01 — Simple Wi-Fi route | | | |
| S02 — Case-insensitive password route | | | |
| S03 — Unsupported request | | | |
| S04 — Two topics before the graph change | | | |
| S05 — Two topics after adding clarification | | | |
| S06 — Instructions then local ticket | | | |
| S07 — The same request during a Wi-Fi outage | | | |
| S08 — Status source unavailable | | | |
| S09 — Missing ticket details | | | |
| S10 — Use the current conversation | | | |
| S11 — Persist and recall after restart | | | |
| S12 — Preference across threads without replay | | | |
| S13 — Do not share memory between users | | | |
| S14 — Password recovery outage | | | |
