# Dispatch W019

**Tasks:** 4 | **Max agents:** 4

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `PTWG-072` (core)

Read and execute: `plan/waves/W019/agents/agent-01.md`
### agent-02 → `INV-007` (pt-webgpu)

Read and execute: `plan/waves/W019/agents/agent-02.md`
### agent-03 → `INV-011` (pt-webgpu)

Read and execute: `plan/waves/W019/agents/agent-03.md`
### agent-04 → `RT100-F-BRIDGE` (repo-root)

Read and execute: `plan/waves/W019/agents/agent-04.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W019
node tools/gap-scan/orchestrator-run.mjs --smoke W019
node tools/gap-scan/orchestrator-run.mjs --commit W019
node tools/gap-scan/orchestrator-run.mjs --advance
```
