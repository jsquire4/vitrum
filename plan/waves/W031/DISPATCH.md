# Dispatch W031

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-thickness` (core)

Read and execute: `plan/waves/W031/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W031
node tools/gap-scan/orchestrator-run.mjs --smoke W031
node tools/gap-scan/orchestrator-run.mjs --commit W031
node tools/gap-scan/orchestrator-run.mjs --advance
```
