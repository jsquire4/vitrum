# Dispatch W065

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-iridescenceThicknessRange` (core)

Read and execute: `plan/waves/W065/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W065
node tools/gap-scan/orchestrator-run.mjs --smoke W065
node tools/gap-scan/orchestrator-run.mjs --commit W065
node tools/gap-scan/orchestrator-run.mjs --advance
```
