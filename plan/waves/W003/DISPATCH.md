# Dispatch W003

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `PTGL-004` (pt-webgl2)

Read and execute: `plan/waves/W003/agents/agent-01.md`
### agent-02 → `PTGL-008` (pt-webgl2)

Read and execute: `plan/waves/W003/agents/agent-02.md`
### agent-03 → `FP-03` (core)

Read and execute: `plan/waves/W003/agents/agent-03.md`
### agent-04 → `MUT-11` (core)

Read and execute: `plan/waves/W003/agents/agent-04.md`
### agent-05 → `MUT-12` (core)

Read and execute: `plan/waves/W003/agents/agent-05.md`
### agent-06 → `PTWG-002` (pt-webgpu)

Read and execute: `plan/waves/W003/agents/agent-06.md`
### agent-07 → `WH-002` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-07.md`
### agent-08 → `WH-009` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-08.md`
### agent-09 → `WH-016` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-09.md`
### agent-10 → `WH-017` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-10.md`
### agent-11 → `WH-018` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-11.md`
### agent-12 → `WH-020` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-12.md`
### agent-13 → `WH-022` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-13.md`
### agent-14 → `WH-026` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-14.md`
### agent-15 → `WH-028` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-15.md`
### agent-16 → `WH-033` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-16.md`
### agent-17 → `WH-035` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-17.md`
### agent-18 → `WH-042` (walkaround-hybrid)

Read and execute: `plan/waves/W003/agents/agent-18.md`
### agent-19 → `GLTF-001` (gltf-adapter)

Read and execute: `plan/waves/W003/agents/agent-19.md`
### agent-20 → `GLTF-005` (gltf-adapter)

Read and execute: `plan/waves/W003/agents/agent-20.md`
### agent-21 → `GLTF-008` (gltf-adapter)

Read and execute: `plan/waves/W003/agents/agent-21.md`
### agent-22 → `RT100-ADJ-001` (pt-webgpu)

Read and execute: `plan/waves/W003/agents/agent-22.md`
### agent-23 → `CORE-011` (core)

Read and execute: `plan/waves/W003/agents/agent-23.md`
### agent-24 → `CORE-024` (core)

Read and execute: `plan/waves/W003/agents/agent-24.md`
### agent-25 → `CORE-025` (core)

Read and execute: `plan/waves/W003/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W003
node tools/gap-scan/orchestrator-run.mjs --smoke W003
node tools/gap-scan/orchestrator-run.mjs --commit W003
node tools/gap-scan/orchestrator-run.mjs --advance
```
