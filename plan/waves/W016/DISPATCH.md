# Dispatch W016

**Tasks:** 17 | **Max agents:** 17

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `CORE-029` (core)

Read and execute: `plan/waves/W016/agents/agent-01.md`
### agent-02 → `WH-128` (walkaround-hybrid)

Read and execute: `plan/waves/W016/agents/agent-02.md`
### agent-03 → `GLTF-014` (gltf-adapter)

Read and execute: `plan/waves/W016/agents/agent-03.md`
### agent-04 → `GLTF-019` (gltf-adapter)

Read and execute: `plan/waves/W016/agents/agent-04.md`
### agent-05 → `SBVH-008` (shared-bvh)

Read and execute: `plan/waves/W016/agents/agent-05.md`
### agent-06 → `RC-002` (walkaround-rc)

Read and execute: `plan/waves/W016/agents/agent-06.md`
### agent-07 → `DEV-002` (dev)

Read and execute: `plan/waves/W016/agents/agent-07.md`
### agent-08 → `DEV-003` (dev)

Read and execute: `plan/waves/W016/agents/agent-08.md`
### agent-09 → `DEV-004` (dev)

Read and execute: `plan/waves/W016/agents/agent-09.md`
### agent-10 → `DEV-005` (dev)

Read and execute: `plan/waves/W016/agents/agent-10.md`
### agent-11 → `DEV-006` (dev)

Read and execute: `plan/waves/W016/agents/agent-11.md`
### agent-12 → `INV-001` (pt-webgpu)

Read and execute: `plan/waves/W016/agents/agent-12.md`
### agent-13 → `INV-004` (pt-webgpu)

Read and execute: `plan/waves/W016/agents/agent-13.md`
### agent-14 → `RT100-EMISSIVE-PDF` (shared-samplers)

Read and execute: `plan/waves/W016/agents/agent-14.md`
### agent-15 → `RT100-5D-DOC` (repo-root)

Read and execute: `plan/waves/W016/agents/agent-15.md`
### agent-16 → `RT100-LD-SAMPLING-01` (shared-samplers)

Read and execute: `plan/waves/W016/agents/agent-16.md`
### agent-17 → `RT100-F3-DENO-AUTO` (engine)

Read and execute: `plan/waves/W016/agents/agent-17.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W016
node tools/gap-scan/orchestrator-run.mjs --smoke W016
node tools/gap-scan/orchestrator-run.mjs --commit W016
node tools/gap-scan/orchestrator-run.mjs --advance
```
