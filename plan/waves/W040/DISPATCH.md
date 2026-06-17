# Dispatch W040

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-alphaMap` (core)

Read and execute: `plan/waves/W040/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W040
node tools/gap-scan/orchestrator-run.mjs --smoke W040
node tools/gap-scan/orchestrator-run.mjs --commit W040
node tools/gap-scan/orchestrator-run.mjs --advance
```
