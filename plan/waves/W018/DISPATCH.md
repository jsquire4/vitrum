# Dispatch W018

**Tasks:** 5 | **Max agents:** 5

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `CORE-031` (core)

Read and execute: `plan/waves/W018/agents/agent-01.md`
### agent-02 → `RC-006` (walkaround-rc)

Read and execute: `plan/waves/W018/agents/agent-02.md`
### agent-03 → `INV-003` (pt-webgpu)

Read and execute: `plan/waves/W018/agents/agent-03.md`
### agent-04 → `INV-006` (pt-webgpu)

Read and execute: `plan/waves/W018/agents/agent-04.md`
### agent-05 → `RT100-F5-VOLUMES` (core)

Read and execute: `plan/waves/W018/agents/agent-05.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W018
node tools/gap-scan/orchestrator-run.mjs --smoke W018
node tools/gap-scan/orchestrator-run.mjs --commit W018
node tools/gap-scan/orchestrator-run.mjs --advance
```
