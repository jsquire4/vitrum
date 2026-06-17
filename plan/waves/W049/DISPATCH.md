# Dispatch W049

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-iridescenceMap` (core)

Read and execute: `plan/waves/W049/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W049
node tools/gap-scan/orchestrator-run.mjs --smoke W049
node tools/gap-scan/orchestrator-run.mjs --commit W049
node tools/gap-scan/orchestrator-run.mjs --advance
```
