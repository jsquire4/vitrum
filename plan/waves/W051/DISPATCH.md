# Dispatch W051

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-anisotropyMap` (core)

Read and execute: `plan/waves/W051/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W051
node tools/gap-scan/orchestrator-run.mjs --smoke W051
node tools/gap-scan/orchestrator-run.mjs --commit W051
node tools/gap-scan/orchestrator-run.mjs --advance
```
