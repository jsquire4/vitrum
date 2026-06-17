# Dispatch W014

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `CORE-022` (core)

Read and execute: `plan/waves/W014/agents/agent-01.md`
### agent-02 → `PTWG-033` (pt-webgpu)

Read and execute: `plan/waves/W014/agents/agent-02.md`
### agent-03 → `PTWG-075` (pt-webgpu)

Read and execute: `plan/waves/W014/agents/agent-03.md`
### agent-04 → `WH-056` (walkaround-hybrid)

Read and execute: `plan/waves/W014/agents/agent-04.md`
### agent-05 → `WH-088` (walkaround-hybrid)

Read and execute: `plan/waves/W014/agents/agent-05.md`
### agent-06 → `WH-106` (walkaround-hybrid)

Read and execute: `plan/waves/W014/agents/agent-06.md`
### agent-07 → `WH-111` (walkaround-hybrid)

Read and execute: `plan/waves/W014/agents/agent-07.md`
### agent-08 → `WH-112` (walkaround-hybrid)

Read and execute: `plan/waves/W014/agents/agent-08.md`
### agent-09 → `WH-118` (walkaround-hybrid)

Read and execute: `plan/waves/W014/agents/agent-09.md`
### agent-10 → `WH-121` (walkaround-hybrid)

Read and execute: `plan/waves/W014/agents/agent-10.md`
### agent-11 → `WH-126` (walkaround-hybrid)

Read and execute: `plan/waves/W014/agents/agent-11.md`
### agent-12 → `GLTF-012` (gltf-adapter)

Read and execute: `plan/waves/W014/agents/agent-12.md`
### agent-13 → `GLTF-017` (gltf-adapter)

Read and execute: `plan/waves/W014/agents/agent-13.md`
### agent-14 → `GLTF-021` (gltf-adapter)

Read and execute: `plan/waves/W014/agents/agent-14.md`
### agent-15 → `SBVH-001` (shared-bvh)

Read and execute: `plan/waves/W014/agents/agent-15.md`
### agent-16 → `SBVH-003` (shared-bvh)

Read and execute: `plan/waves/W014/agents/agent-16.md`
### agent-17 → `SBVH-004` (shared-bvh)

Read and execute: `plan/waves/W014/agents/agent-17.md`
### agent-18 → `SBVH-005` (shared-bvh)

Read and execute: `plan/waves/W014/agents/agent-18.md`
### agent-19 → `SBVH-006` (shared-bvh)

Read and execute: `plan/waves/W014/agents/agent-19.md`
### agent-20 → `SBVH-007` (shared-bvh)

Read and execute: `plan/waves/W014/agents/agent-20.md`
### agent-21 → `SSAMP-001` (shared-samplers)

Read and execute: `plan/waves/W014/agents/agent-21.md`
### agent-22 → `SSAMP-002` (shared-samplers)

Read and execute: `plan/waves/W014/agents/agent-22.md`
### agent-23 → `SSAMP-003` (shared-samplers)

Read and execute: `plan/waves/W014/agents/agent-23.md`
### agent-24 → `SSAMP-004` (shared-samplers)

Read and execute: `plan/waves/W014/agents/agent-24.md`
### agent-25 → `SSAMP-005` (shared-samplers)

Read and execute: `plan/waves/W014/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W014
node tools/gap-scan/orchestrator-run.mjs --smoke W014
node tools/gap-scan/orchestrator-run.mjs --commit W014
node tools/gap-scan/orchestrator-run.mjs --advance
```
