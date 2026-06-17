# Dispatch W043

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-clearcoatMap` (core)

Read and execute: `plan/waves/W043/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W043
node tools/gap-scan/orchestrator-run.mjs --smoke W043
node tools/gap-scan/orchestrator-run.mjs --commit W043
node tools/gap-scan/orchestrator-run.mjs --advance
```
