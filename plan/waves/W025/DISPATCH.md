# Dispatch W025

**Tasks:** 2 | **Max agents:** 2

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-alphaCutoff` (core)

Read and execute: `plan/waves/W025/agents/agent-01.md`
### agent-02 → `INV-016` (pt-webgpu)

Read and execute: `plan/waves/W025/agents/agent-02.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W025
node tools/gap-scan/orchestrator-run.mjs --smoke W025
node tools/gap-scan/orchestrator-run.mjs --commit W025
node tools/gap-scan/orchestrator-run.mjs --advance
```
