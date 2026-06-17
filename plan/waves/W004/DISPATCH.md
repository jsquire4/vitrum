# Dispatch W004

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `FP-06` (walkaround-hybrid)

Read and execute: `plan/waves/W004/agents/agent-01.md`
### agent-02 → `MUT-01` (pt-webgl2)

Read and execute: `plan/waves/W004/agents/agent-02.md`
### agent-03 → `PTWG-003` (pt-webgpu)

Read and execute: `plan/waves/W004/agents/agent-03.md`
### agent-04 → `WH-003` (walkaround-hybrid)

Read and execute: `plan/waves/W004/agents/agent-04.md`
### agent-05 → `WH-010` (walkaround-hybrid)

Read and execute: `plan/waves/W004/agents/agent-05.md`
### agent-06 → `WH-032` (walkaround-hybrid)

Read and execute: `plan/waves/W004/agents/agent-06.md`
### agent-07 → `WH-036` (walkaround-hybrid)

Read and execute: `plan/waves/W004/agents/agent-07.md`
### agent-08 → `WH-041` (walkaround-hybrid)

Read and execute: `plan/waves/W004/agents/agent-08.md`
### agent-09 → `WH-043` (walkaround-hybrid)

Read and execute: `plan/waves/W004/agents/agent-09.md`
### agent-10 → `GLTF-002` (gltf-adapter)

Read and execute: `plan/waves/W004/agents/agent-10.md`
### agent-11 → `GLTF-006` (gltf-adapter)

Read and execute: `plan/waves/W004/agents/agent-11.md`
### agent-12 → `GLTF-009` (gltf-adapter)

Read and execute: `plan/waves/W004/agents/agent-12.md`
### agent-13 → `CORE-012` (core)

Read and execute: `plan/waves/W004/agents/agent-13.md`
### agent-14 → `CORE-026` (core)

Read and execute: `plan/waves/W004/agents/agent-14.md`
### agent-15 → `CORE-028` (core)

Read and execute: `plan/waves/W004/agents/agent-15.md`
### agent-16 → `CORE-032` (core)

Read and execute: `plan/waves/W004/agents/agent-16.md`
### agent-17 → `CORE-034` (core)

Read and execute: `plan/waves/W004/agents/agent-17.md`
### agent-18 → `CORE-035` (core)

Read and execute: `plan/waves/W004/agents/agent-18.md`
### agent-19 → `CORE-036` (core)

Read and execute: `plan/waves/W004/agents/agent-19.md`
### agent-20 → `CORE-037` (core)

Read and execute: `plan/waves/W004/agents/agent-20.md`
### agent-21 → `CORE-038` (core)

Read and execute: `plan/waves/W004/agents/agent-21.md`
### agent-22 → `CORE-039` (core)

Read and execute: `plan/waves/W004/agents/agent-22.md`
### agent-23 → `CORE-040` (core)

Read and execute: `plan/waves/W004/agents/agent-23.md`
### agent-24 → `CORE-042` (core)

Read and execute: `plan/waves/W004/agents/agent-24.md`
### agent-25 → `CORE-043` (core)

Read and execute: `plan/waves/W004/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W004
node tools/gap-scan/orchestrator-run.mjs --smoke W004
node tools/gap-scan/orchestrator-run.mjs --commit W004
node tools/gap-scan/orchestrator-run.mjs --advance
```
