# Dispatch W032

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-baseColorMap` (core)

Read and execute: `plan/waves/W032/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W032
node tools/gap-scan/orchestrator-run.mjs --smoke W032
node tools/gap-scan/orchestrator-run.mjs --commit W032
node tools/gap-scan/orchestrator-run.mjs --advance
```
