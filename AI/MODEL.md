# Current AI Agent

## Identity

- **Model:** x-preview-f-free
- **Model ID:** opencode/x-preview-f-free
- **Provider:** OpenCode
- **Application:** OpenCode CLI
- **Agent/Mode:** Interactive coding agent with workspace tools, PowerShell execution, and web access
- **Version:** Unknown
- **Session ID:** Unknown
- **Started:** 2026-08-23
- **Last Updated:** 2026-08-23 (ox-alpha / OpenCode CLI session)

## Capabilities

- Read, create, and edit workspace files via dedicated edit tools (targeted, minimal diffs)
- Run PowerShell commands and Python/Streamlit checks
- Search codebase by symbol, glob, or regex; fetch/search web content

## Known Limitations

- Filesystem writes are limited to the workspace and approved temporary directories
- Network access and package installation may require approval
- No access to private cloud accounts or credentials unless the user supplies them at runtime

## Important Environment

- **OS:** Windows
- **Working Directory:** `C:\Users\Srikiran\CloudInsightAI`
- **Runtime:** `venv\Scripts\python.exe`
- **Project Type:** Streamlit analytics application

## Notes

- Verify prior-agent claims against the checked-out code and current test output.
- Do not persist Gemini or AWS credentials to disk; keep DEC-009 intact.
