# Dispatch W036

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-metallicMap` (core)

Read and execute: `plan/waves/W036/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W036
node tools/gap-scan/orchestrator-run.mjs --smoke W036
node tools/gap-scan/orchestrator-run.mjs --commit W036
node tools/gap-scan/orchestrator-run.mjs --advance
```
