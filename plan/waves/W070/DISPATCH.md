# Dispatch W070

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-anisotropyRotation` (core)

Read and execute: `plan/waves/W070/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W070
node tools/gap-scan/orchestrator-run.mjs --smoke W070
node tools/gap-scan/orchestrator-run.mjs --commit W070
node tools/gap-scan/orchestrator-run.mjs --advance
```
