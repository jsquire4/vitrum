# Dispatch W053

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-specularIntensityMap` (core)

Read and execute: `plan/waves/W053/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W053
node tools/gap-scan/orchestrator-run.mjs --smoke W053
node tools/gap-scan/orchestrator-run.mjs --commit W053
node tools/gap-scan/orchestrator-run.mjs --advance
```
