# Dispatch W030

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-attenuationDistance` (core)

Read and execute: `plan/waves/W030/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W030
node tools/gap-scan/orchestrator-run.mjs --smoke W030
node tools/gap-scan/orchestrator-run.mjs --commit W030
node tools/gap-scan/orchestrator-run.mjs --advance
```
