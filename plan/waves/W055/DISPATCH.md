# Dispatch W055

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-bumpScale` (core)

Read and execute: `plan/waves/W055/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W055
node tools/gap-scan/orchestrator-run.mjs --smoke W055
node tools/gap-scan/orchestrator-run.mjs --commit W055
node tools/gap-scan/orchestrator-run.mjs --advance
```
