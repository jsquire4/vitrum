# Dispatch W050

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-iridescenceThicknessMap` (core)

Read and execute: `plan/waves/W050/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W050
node tools/gap-scan/orchestrator-run.mjs --smoke W050
node tools/gap-scan/orchestrator-run.mjs --commit W050
node tools/gap-scan/orchestrator-run.mjs --advance
```
