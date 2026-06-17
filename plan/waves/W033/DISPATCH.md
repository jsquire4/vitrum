# Dispatch W033

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-normalMap` (core)

Read and execute: `plan/waves/W033/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W033
node tools/gap-scan/orchestrator-run.mjs --smoke W033
node tools/gap-scan/orchestrator-run.mjs --commit W033
node tools/gap-scan/orchestrator-run.mjs --advance
```
