# Dispatch W035

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-roughnessMap` (core)

Read and execute: `plan/waves/W035/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W035
node tools/gap-scan/orchestrator-run.mjs --smoke W035
node tools/gap-scan/orchestrator-run.mjs --commit W035
node tools/gap-scan/orchestrator-run.mjs --advance
```
