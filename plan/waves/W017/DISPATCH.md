# Dispatch W017

**Tasks:** 6 | **Max agents:** 6

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `CORE-030` (core)

Read and execute: `plan/waves/W017/agents/agent-01.md`
### agent-02 → `GLTF-015` (gltf-adapter)

Read and execute: `plan/waves/W017/agents/agent-02.md`
### agent-03 → `RC-003` (walkaround-rc)

Read and execute: `plan/waves/W017/agents/agent-03.md`
### agent-04 → `INV-002` (pt-webgpu)

Read and execute: `plan/waves/W017/agents/agent-04.md`
### agent-05 → `INV-005` (pt-webgpu)

Read and execute: `plan/waves/W017/agents/agent-05.md`
### agent-06 → `RT100-F4-WAVEFRONT` (repo-root)

Read and execute: `plan/waves/W017/agents/agent-06.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W017
node tools/gap-scan/orchestrator-run.mjs --smoke W017
node tools/gap-scan/orchestrator-run.mjs --commit W017
node tools/gap-scan/orchestrator-run.mjs --advance
```
