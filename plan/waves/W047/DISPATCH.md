# Dispatch W047

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-sheenColorMap` (core)

Read and execute: `plan/waves/W047/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W047
node tools/gap-scan/orchestrator-run.mjs --smoke W047
node tools/gap-scan/orchestrator-run.mjs --commit W047
node tools/gap-scan/orchestrator-run.mjs --advance
```
