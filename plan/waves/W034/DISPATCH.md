# Dispatch W034

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-normalScale` (core)

Read and execute: `plan/waves/W034/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W034
node tools/gap-scan/orchestrator-run.mjs --smoke W034
node tools/gap-scan/orchestrator-run.mjs --commit W034
node tools/gap-scan/orchestrator-run.mjs --advance
```
