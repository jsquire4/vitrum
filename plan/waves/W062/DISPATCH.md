# Dispatch W062

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-clearcoatRoughness` (core)

Read and execute: `plan/waves/W062/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W062
node tools/gap-scan/orchestrator-run.mjs --smoke W062
node tools/gap-scan/orchestrator-run.mjs --commit W062
node tools/gap-scan/orchestrator-run.mjs --advance
```
