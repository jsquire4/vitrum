# Dispatch W026

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-opacity` (core)

Read and execute: `plan/waves/W026/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W026
node tools/gap-scan/orchestrator-run.mjs --smoke W026
node tools/gap-scan/orchestrator-run.mjs --commit W026
node tools/gap-scan/orchestrator-run.mjs --advance
```
