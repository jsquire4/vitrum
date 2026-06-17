# Dispatch W028

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-ior` (core)

Read and execute: `plan/waves/W028/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W028
node tools/gap-scan/orchestrator-run.mjs --smoke W028
node tools/gap-scan/orchestrator-run.mjs --commit W028
node tools/gap-scan/orchestrator-run.mjs --advance
```
