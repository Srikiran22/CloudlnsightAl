# Current AI Agent

## Identity

- **Model:** ox-alpha
- **Model ID:** opencode/x-preview-f-free
- **Provider:** Undisclosed organization (the model's developer does not disclose its identity)
- **Application:** OpenCode — interactive CLI coding agent
- **Agent/Mode:** Interactive terminal agent with tool use (file read/write/edit, shell execution, search)
- **Version:** Unknown
- **Session ID:** Unknown
- **Started:** 2026-08-21
- **Last Updated:** 2026-08-22

> Note for successor agents: identify yourself here when you take over. Do not assume this
> model's conclusions are correct — verify claims against the repository.

## Capabilities

- Read, create, and edit files of any text format
- Execute Windows PowerShell 5.1 commands (builds, tests, git, package installs)
- Search codebases by filename pattern and by content regex
- Launch parallel sub-agents for exploration or multi-step research tasks
- Web search and URL fetching for documentation lookup
- Structured task planning via an internal todo list

## Known Limitations

- Cannot run or see GUI applications; verification is CLI-only
- No persistent memory between sessions except what is written to these AI/ files
- PowerShell 5.1 only (no bash); no `&&` chaining — use `;` / `if ($?)`
- Cannot access private networks, databases, or cloud consoles without explicit credentials/tools
- Knowledge cutoff applies; must search the web for current library versions

## Important Environment

- **OS:** Windows (win32)
- **Shell:** Windows PowerShell 5.1
- **Working Directory:** C:\Users\Srikiran\CloudInsightAI
- **Runtime:** Python venv at `./venv` (use `& .\venv\Scripts\python.exe`)
- **Project Type:** Streamlit application (Python) — CloudInsight AI analytics platform
- **Relevant Tools:** unittest test suite in `tests/`, streamlit dev server on port 8501

## Communication Notes

- Responses are rendered in a terminal; keep output concise
- User works in English; code comments should be minimal unless requested

## Notes

- This file must be rewritten by every new agent/model that takes over a session.
- If a field cannot be determined, write `Unknown` — never invent information.
