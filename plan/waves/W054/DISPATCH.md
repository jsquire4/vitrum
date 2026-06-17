# Dispatch W054

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-bumpMap` (core)

Read and execute: `plan/waves/W054/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W054
node tools/gap-scan/orchestrator-run.mjs --smoke W054
node tools/gap-scan/orchestrator-run.mjs --commit W054
node tools/gap-scan/orchestrator-run.mjs --advance
```
