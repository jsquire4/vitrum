# Dispatch W001

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `CORE-001` (core)

Read and execute: `plan/waves/W001/agents/agent-01.md`
### agent-02 → `CORE-002` (core)

Read and execute: `plan/waves/W001/agents/agent-02.md`
### agent-03 → `CORE-003` (core)

Read and execute: `plan/waves/W001/agents/agent-03.md`
### agent-04 → `CORE-006` (core)

Read and execute: `plan/waves/W001/agents/agent-04.md`
### agent-05 → `CORE-008` (core)

Read and execute: `plan/waves/W001/agents/agent-05.md`
### agent-06 → `CORE-010` (core)

Read and execute: `plan/waves/W001/agents/agent-06.md`
### agent-07 → `ENG-001` (engine)

Read and execute: `plan/waves/W001/agents/agent-07.md`
### agent-08 → `ENG-002` (engine)

Read and execute: `plan/waves/W001/agents/agent-08.md`
### agent-09 → `ENG-003` (engine)

Read and execute: `plan/waves/W001/agents/agent-09.md`
### agent-10 → `ENG-004` (engine)

Read and execute: `plan/waves/W001/agents/agent-10.md`
### agent-11 → `ENG-005` (engine)

Read and execute: `plan/waves/W001/agents/agent-11.md`
### agent-12 → `ENG-007` (engine)

Read and execute: `plan/waves/W001/agents/agent-12.md`
### agent-13 → `ENG-009` (engine)

Read and execute: `plan/waves/W001/agents/agent-13.md`
### agent-14 → `PTGL-001` (pt-webgl2)

Read and execute: `plan/waves/W001/agents/agent-14.md`
### agent-15 → `PTGL-005` (pt-webgl2)

Read and execute: `plan/waves/W001/agents/agent-15.md`
### agent-16 → `PTGL-007` (pt-webgl2)

Read and execute: `plan/waves/W001/agents/agent-16.md`
### agent-17 → `PTGL-009` (pt-webgl2)

Read and execute: `plan/waves/W001/agents/agent-17.md`
### agent-18 → `FP-01` (core)

Read and execute: `plan/waves/W001/agents/agent-18.md`
### agent-19 → `MUT-07` (pt-webgpu)

Read and execute: `plan/waves/W001/agents/agent-19.md`
### agent-20 → `MUT-09` (walkaround-hybrid)

Read and execute: `plan/waves/W001/agents/agent-20.md`
### agent-21 → `PTWG-001` (pt-webgpu)

Read and execute: `plan/waves/W001/agents/agent-21.md`
### agent-22 → `PTWG-004` (core)

Read and execute: `plan/waves/W001/agents/agent-22.md`
### agent-23 → `PTWG-005` (pt-webgpu)

Read and execute: `plan/waves/W001/agents/agent-23.md`
### agent-24 → `PTWG-008` (pt-webgpu)

Read and execute: `plan/waves/W001/agents/agent-24.md`
### agent-25 → `PTWG-010` (pt-webgpu)

Read and execute: `plan/waves/W001/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W001
node tools/gap-scan/orchestrator-run.mjs --smoke W001
node tools/gap-scan/orchestrator-run.mjs --commit W001
node tools/gap-scan/orchestrator-run.mjs --advance
```
