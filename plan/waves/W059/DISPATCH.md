# Dispatch W059

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-sheenColor` (core)

Read and execute: `plan/waves/W059/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W059
node tools/gap-scan/orchestrator-run.mjs --smoke W059
node tools/gap-scan/orchestrator-run.mjs --commit W059
node tools/gap-scan/orchestrator-run.mjs --advance
```
