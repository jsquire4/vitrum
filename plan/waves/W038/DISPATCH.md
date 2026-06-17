# Dispatch W038

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-thicknessMap` (core)

Read and execute: `plan/waves/W038/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W038
node tools/gap-scan/orchestrator-run.mjs --smoke W038
node tools/gap-scan/orchestrator-run.mjs --commit W038
node tools/gap-scan/orchestrator-run.mjs --advance
```
