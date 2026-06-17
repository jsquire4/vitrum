# Dispatch W056

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-lightMap` (core)

Read and execute: `plan/waves/W056/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W056
node tools/gap-scan/orchestrator-run.mjs --smoke W056
node tools/gap-scan/orchestrator-run.mjs --commit W056
node tools/gap-scan/orchestrator-run.mjs --advance
```
