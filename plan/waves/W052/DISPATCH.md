# Dispatch W052

**Tasks:** 1 | **Max agents:** 1

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-specularColorMap` (core)

Read and execute: `plan/waves/W052/agents/agent-01.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W052
node tools/gap-scan/orchestrator-run.mjs --smoke W052
node tools/gap-scan/orchestrator-run.mjs --commit W052
node tools/gap-scan/orchestrator-run.mjs --advance
```
