# Dispatch W006

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MUT-03` (pt-webgl2)

Read and execute: `plan/waves/W006/agents/agent-01.md`
### agent-02 → `MUT-08` (walkaround-hybrid)

Read and execute: `plan/waves/W006/agents/agent-02.md`
### agent-03 → `PTWG-007` (pt-webgpu)

Read and execute: `plan/waves/W006/agents/agent-03.md`
### agent-04 → `WH-005` (walkaround-hybrid)

Read and execute: `plan/waves/W006/agents/agent-04.md`
### agent-05 → `WH-024` (walkaround-hybrid)

Read and execute: `plan/waves/W006/agents/agent-05.md`
### agent-06 → `GLTF-004` (gltf-adapter)

Read and execute: `plan/waves/W006/agents/agent-06.md`
### agent-07 → `CORE-014` (core)

Read and execute: `plan/waves/W006/agents/agent-07.md`
### agent-08 → `ENG-013` (engine)

Read and execute: `plan/waves/W006/agents/agent-08.md`
### agent-09 → `ENG-021` (engine)

Read and execute: `plan/waves/W006/agents/agent-09.md`
### agent-10 → `ENG-022` (engine)

Read and execute: `plan/waves/W006/agents/agent-10.md`
### agent-11 → `ENG-023` (engine)

Read and execute: `plan/waves/W006/agents/agent-11.md`
### agent-12 → `ENG-024` (engine)

Read and execute: `plan/waves/W006/agents/agent-12.md`
### agent-13 → `ENG-025` (engine)

Read and execute: `plan/waves/W006/agents/agent-13.md`
### agent-14 → `ENG-027` (engine)

Read and execute: `plan/waves/W006/agents/agent-14.md`
### agent-15 → `ENG-028` (engine)

Read and execute: `plan/waves/W006/agents/agent-15.md`
### agent-16 → `ENG-029` (engine)

Read and execute: `plan/waves/W006/agents/agent-16.md`
### agent-17 → `ENG-030` (engine)

Read and execute: `plan/waves/W006/agents/agent-17.md`
### agent-18 → `TOOL-002` (tools)

Read and execute: `plan/waves/W006/agents/agent-18.md`
### agent-19 → `TOOL-003` (tools)

Read and execute: `plan/waves/W006/agents/agent-19.md`
### agent-20 → `TOOL-005` (tools)

Read and execute: `plan/waves/W006/agents/agent-20.md`
### agent-21 → `PTGL-011` (pt-webgl2)

Read and execute: `plan/waves/W006/agents/agent-21.md`
### agent-22 → `PTGL-013` (pt-webgl2)

Read and execute: `plan/waves/W006/agents/agent-22.md`
### agent-23 → `PTGL-017` (pt-webgl2)

Read and execute: `plan/waves/W006/agents/agent-23.md`
### agent-24 → `PTGL-020` (pt-webgl2)

Read and execute: `plan/waves/W006/agents/agent-24.md`
### agent-25 → `PTGL-021` (pt-webgl2)

Read and execute: `plan/waves/W006/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W006
node tools/gap-scan/orchestrator-run.mjs --smoke W006
node tools/gap-scan/orchestrator-run.mjs --commit W006
node tools/gap-scan/orchestrator-run.mjs --advance
```
