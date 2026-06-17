# Dispatch W063

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-iridescence` (core)

Read and execute: `plan/waves/W063/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W063
node tools/gap-scan/orchestrator-run.mjs --smoke W063
node tools/gap-scan/orchestrator-run.mjs --commit W063
node tools/gap-scan/orchestrator-run.mjs --advance
```
