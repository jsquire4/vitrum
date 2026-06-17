# Dispatch W020

**Tasks:** 3 | **Max agents:** 3

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-baseColor` (core)

Read and execute: `plan/waves/W020/agents/agent-01.md`
### agent-02 → `INV-008` (pt-webgpu)

Read and execute: `plan/waves/W020/agents/agent-02.md`
### agent-03 → `INV-012` (pt-webgpu)

Read and execute: `plan/waves/W020/agents/agent-03.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W020
node tools/gap-scan/orchestrator-run.mjs --smoke W020
node tools/gap-scan/orchestrator-run.mjs --commit W020
node tools/gap-scan/orchestrator-run.mjs --advance
```
