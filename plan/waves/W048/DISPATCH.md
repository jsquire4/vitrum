# Dispatch W048

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-sheenRoughnessMap` (core)

Read and execute: `plan/waves/W048/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W048
node tools/gap-scan/orchestrator-run.mjs --smoke W048
node tools/gap-scan/orchestrator-run.mjs --commit W048
node tools/gap-scan/orchestrator-run.mjs --advance
```
