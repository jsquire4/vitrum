# Dispatch W060

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-sheenRoughness` (core)

Read and execute: `plan/waves/W060/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W060
node tools/gap-scan/orchestrator-run.mjs --smoke W060
node tools/gap-scan/orchestrator-run.mjs --commit W060
node tools/gap-scan/orchestrator-run.mjs --advance
```
