# Dispatch W044

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-clearcoatRoughnessMap` (core)

Read and execute: `plan/waves/W044/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W044
node tools/gap-scan/orchestrator-run.mjs --smoke W044
node tools/gap-scan/orchestrator-run.mjs --commit W044
node tools/gap-scan/orchestrator-run.mjs --advance
```
