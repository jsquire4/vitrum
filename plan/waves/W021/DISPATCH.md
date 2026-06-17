# Dispatch W021

**Tasks:** 3 | **Max agents:** 3

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-roughness` (core)

Read and execute: `plan/waves/W021/agents/agent-01.md`
### agent-02 → `INV-009` (pt-webgpu)

Read and execute: `plan/waves/W021/agents/agent-02.md`
### agent-03 → `INV-013` (pt-webgpu)

Read and execute: `plan/waves/W021/agents/agent-03.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W021
node tools/gap-scan/orchestrator-run.mjs --smoke W021
node tools/gap-scan/orchestrator-run.mjs --commit W021
node tools/gap-scan/orchestrator-run.mjs --advance
```
