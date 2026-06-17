# Dispatch W057

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-lightMapIntensity` (core)

Read and execute: `plan/waves/W057/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W057
node tools/gap-scan/orchestrator-run.mjs --smoke W057
node tools/gap-scan/orchestrator-run.mjs --commit W057
node tools/gap-scan/orchestrator-run.mjs --advance
```
