# Dispatch W045

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-clearcoatNormalMap` (core)

Read and execute: `plan/waves/W045/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W045
node tools/gap-scan/orchestrator-run.mjs --smoke W045
node tools/gap-scan/orchestrator-run.mjs --commit W045
node tools/gap-scan/orchestrator-run.mjs --advance
```
