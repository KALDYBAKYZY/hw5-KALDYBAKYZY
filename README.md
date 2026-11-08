# HW5 — University IT helpdesk with LangGraph

Student: Bissembayeva Aigerim · CSS-4007 Artificial Intelligence

My answers, drawings and results are in [SUBMISSION.md](SUBMISSION.md).
The original task text is below this section.

## How to run my work

### 1. Python version

I used **Python 3.12** on my computer. GitHub Actions runs the tests on **Python 3.12.7**.
Check your version with `python --version`.

### 2. Install

```bash
git clone https://github.com/KALDYBAKYZY/hw5-KALDYBAKYZY.git
cd hw5-KALDYBAKYZY
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` has the exact versions I used (for example `langgraph==1.2.20`).

### 3. Configure the model (only for Part 2 and Part 3)

Part 1 and all tests do **not** need an API key.

```bash
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` and set:

| Setting | Value I used |
| --- | --- |
| `OPENAI_API_KEY` | your own key (never commit it) |
| `MODEL_NAME` | `gpt-5.6-luna` |
| `REASONING_EFFORT` | `none` (needed for tools with gpt-5.x in Chat Completions; leave empty for gpt-4o-mini) |
| `MAX_TOOL_ROUNDS` | `5` (loop limit) |
| `HELPDESK_STORAGE_DIR` | `local-storage` |

`.env` is in `.gitignore`.

### 4. Run the examples

```bash
# Part 1 — rule-based graph, no model (S01–S05)
python run_part1.py
python run_part1.py --draw         # print the graphs as Mermaid

# Part 2 — model and tools
python run_part2.py --direct       # call the three tools directly, without the agent
python run_part2.py --scenario S06 # also S07, S08, S09, S14
python run_part2.py                # all Part 2 scenarios

# Part 3 — memory with real restarts (each restart is a new process)
python run_part3.py --scenario S10 # also S11, S12, S13
python run_part3.py --reset        # clear all memory

# Interactive chat
python chat.py --user demo-user-a --thread t1 --profile normal
```

Add `--fake` to `run_part2.py`, `run_part3.py` or `chat.py` to use the labelled fake model
(`helpdesk/fake_model.py`) instead of a real model. It is not an LLM.

Logs are saved in `outputs/part1/`, `outputs/part2/`, `outputs/part3/`.
Ticket drafts are saved in `outputs/tickets/`. Nothing is sent to a real helpdesk.

### 5. Where memory is saved and how to clear it

| Memory | File |
| --- | --- |
| Conversation memory (LangGraph `SqliteSaver`, key = `thread_id`) | `local-storage/conversations.sqlite` |
| User memory (language preference, key = `user_id`) | `local-storage/user_memory.sqlite` |

To start again: `python run_part3.py --reset`, or delete the `local-storage/` folder.
Scenarios with `reset_before: true` clear memory automatically.

### 6. Tests and CI

```bash
python -m pytest -q
```

27 tests check routing, the three tools, the loop limit, memory after a restart,
and separation between users and threads. They use the fake model, so no API key is needed.
GitHub Actions (`.github/workflows/tests.yml`) runs them on every push.

---