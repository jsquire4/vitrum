# Dispatch W069

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-anisotropyRotation` (core)

Read and execute: `plan/waves/W069/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W069
node tools/gap-scan/orchestrator-run.mjs --smoke W069
node tools/gap-scan/orchestrator-run.mjs --commit W069
node tools/gap-scan/orchestrator-run.mjs --advance
```
