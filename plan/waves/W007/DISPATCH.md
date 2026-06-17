# Dispatch W007

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MUT-04` (pt-webgl2)

Read and execute: `plan/waves/W007/agents/agent-01.md`
### agent-02 → `PTWG-012` (pt-webgpu)

Read and execute: `plan/waves/W007/agents/agent-02.md`
### agent-03 → `WH-006` (walkaround-hybrid)

Read and execute: `plan/waves/W007/agents/agent-03.md`
### agent-04 → `WH-030` (walkaround-hybrid)

Read and execute: `plan/waves/W007/agents/agent-04.md`
### agent-05 → `WH-038` (walkaround-hybrid)

Read and execute: `plan/waves/W007/agents/agent-05.md`
### agent-06 → `CORE-015` (core)

Read and execute: `plan/waves/W007/agents/agent-06.md`
### agent-07 → `ENG-026` (engine)

Read and execute: `plan/waves/W007/agents/agent-07.md`
### agent-08 → `PTGL-012` (pt-webgl2)

Read and execute: `plan/waves/W007/agents/agent-08.md`
### agent-09 → `PTGL-014` (pt-webgl2)

Read and execute: `plan/waves/W007/agents/agent-09.md`
### agent-10 → `PTGL-018` (pt-webgl2)

Read and execute: `plan/waves/W007/agents/agent-10.md`
### agent-11 → `PTGL-022` (pt-webgl2)

Read and execute: `plan/waves/W007/agents/agent-11.md`
### agent-12 → `PTGL-023` (pt-webgl2)

Read and execute: `plan/waves/W007/agents/agent-12.md`
### agent-13 → `PTGL-024` (pt-webgl2)

Read and execute: `plan/waves/W007/agents/agent-13.md`
### agent-14 → `PTGL-026` (pt-webgl2)

Read and execute: `plan/waves/W007/agents/agent-14.md`
### agent-15 → `PTGL-027` (pt-webgl2)

Read and execute: `plan/waves/W007/agents/agent-15.md`
### agent-16 → `PTGL-028` (pt-webgl2)

Read and execute: `plan/waves/W007/agents/agent-16.md`
### agent-17 → `FP-04` (gltf-adapter)

Read and execute: `plan/waves/W007/agents/agent-17.md`
### agent-18 → `FP-05` (core)

Read and execute: `plan/waves/W007/agents/agent-18.md`
### agent-19 → `PTWG-013` (pt-webgpu)

Read and execute: `plan/waves/W007/agents/agent-19.md`
### agent-20 → `PTWG-014` (pt-webgpu)

Read and execute: `plan/waves/W007/agents/agent-20.md`
### agent-21 → `PTWG-016` (pt-webgpu)

Read and execute: `plan/waves/W007/agents/agent-21.md`
### agent-22 → `PTWG-018` (pt-webgpu)

Read and execute: `plan/waves/W007/agents/agent-22.md`
### agent-23 → `PTWG-019` (pt-webgpu)

Read and execute: `plan/waves/W007/agents/agent-23.md`
### agent-24 → `PTWG-021` (pt-webgpu)

Read and execute: `plan/waves/W007/agents/agent-24.md`
### agent-25 → `PTWG-025` (pt-webgpu)

Read and execute: `plan/waves/W007/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W007
node tools/gap-scan/orchestrator-run.mjs --smoke W007
node tools/gap-scan/orchestrator-run.mjs --commit W007
node tools/gap-scan/orchestrator-run.mjs --advance
```
