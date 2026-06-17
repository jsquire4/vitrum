# Dispatch W015

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `CORE-023` (core)

Read and execute: `plan/waves/W015/agents/agent-01.md`
### agent-02 → `PTWG-064` (pt-webgpu)

Read and execute: `plan/waves/W015/agents/agent-02.md`
### agent-03 → `PTWG-076` (pt-webgpu)

Read and execute: `plan/waves/W015/agents/agent-03.md`
### agent-04 → `WH-107` (walkaround-hybrid)

Read and execute: `plan/waves/W015/agents/agent-04.md`
### agent-05 → `WH-115` (walkaround-hybrid)

Read and execute: `plan/waves/W015/agents/agent-05.md`
### agent-06 → `WH-122` (walkaround-hybrid)

Read and execute: `plan/waves/W015/agents/agent-06.md`
### agent-07 → `GLTF-013` (gltf-adapter)

Read and execute: `plan/waves/W015/agents/agent-07.md`
### agent-08 → `GLTF-018` (gltf-adapter)

Read and execute: `plan/waves/W015/agents/agent-08.md`
### agent-09 → `SBVH-002` (shared-bvh)

Read and execute: `plan/waves/W015/agents/agent-09.md`
### agent-10 → `SSAMP-006` (shared-samplers)

Read and execute: `plan/waves/W015/agents/agent-10.md`
### agent-11 → `SDENO-001` (shared-denoisers)

Read and execute: `plan/waves/W015/agents/agent-11.md`
### agent-12 → `SDENO-002` (shared-denoisers)

Read and execute: `plan/waves/W015/agents/agent-12.md`
### agent-13 → `SDENO-003` (shared-denoisers)

Read and execute: `plan/waves/W015/agents/agent-13.md`
### agent-14 → `SDENO-004` (shared-denoisers)

Read and execute: `plan/waves/W015/agents/agent-14.md`
### agent-15 → `SDENO-005` (shared-denoisers)

Read and execute: `plan/waves/W015/agents/agent-15.md`
### agent-16 → `SDENO-006` (shared-denoisers)

Read and execute: `plan/waves/W015/agents/agent-16.md`
### agent-17 → `RC-001` (walkaround-rc)

Read and execute: `plan/waves/W015/agents/agent-17.md`
### agent-18 → `RC-004` (walkaround-rc)

Read and execute: `plan/waves/W015/agents/agent-18.md`
### agent-19 → `RC-005` (walkaround-rc)

Read and execute: `plan/waves/W015/agents/agent-19.md`
### agent-20 → `SL-001` (scene-lighting)

Read and execute: `plan/waves/W015/agents/agent-20.md`
### agent-21 → `SL-002` (scene-lighting)

Read and execute: `plan/waves/W015/agents/agent-21.md`
### agent-22 → `SL-003` (scene-lighting)

Read and execute: `plan/waves/W015/agents/agent-22.md`
### agent-23 → `SG-001` (stained-glass-extensions)

Read and execute: `plan/waves/W015/agents/agent-23.md`
### agent-24 → `SG-002` (stained-glass-extensions)

Read and execute: `plan/waves/W015/agents/agent-24.md`
### agent-25 → `DEV-001` (dev)

Read and execute: `plan/waves/W015/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W015
node tools/gap-scan/orchestrator-run.mjs --smoke W015
node tools/gap-scan/orchestrator-run.mjs --commit W015
node tools/gap-scan/orchestrator-run.mjs --advance
```
