# Dispatch W039

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-emissiveMap` (core)

Read and execute: `plan/waves/W039/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W039
node tools/gap-scan/orchestrator-run.mjs --smoke W039
node tools/gap-scan/orchestrator-run.mjs --commit W039
node tools/gap-scan/orchestrator-run.mjs --advance
```
