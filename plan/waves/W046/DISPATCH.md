# Dispatch W046

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-clearcoatNormalScale` (core)

Read and execute: `plan/waves/W046/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W046
node tools/gap-scan/orchestrator-run.mjs --smoke W046
node tools/gap-scan/orchestrator-run.mjs --commit W046
node tools/gap-scan/orchestrator-run.mjs --advance
```
