# Dispatch W002

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `CORE-004` (core)

Read and execute: `plan/waves/W002/agents/agent-01.md`
### agent-02 → `CORE-005` (core)

Read and execute: `plan/waves/W002/agents/agent-02.md`
### agent-03 → `CORE-007` (core)

Read and execute: `plan/waves/W002/agents/agent-03.md`
### agent-04 → `CORE-009` (core)

Read and execute: `plan/waves/W002/agents/agent-04.md`
### agent-05 → `ENG-006` (engine)

Read and execute: `plan/waves/W002/agents/agent-05.md`
### agent-06 → `ENG-008` (engine)

Read and execute: `plan/waves/W002/agents/agent-06.md`
### agent-07 → `ENG-010` (engine)

Read and execute: `plan/waves/W002/agents/agent-07.md`
### agent-08 → `PTGL-002` (pt-webgl2)

Read and execute: `plan/waves/W002/agents/agent-08.md`
### agent-09 → `PTGL-006` (pt-webgl2)

Read and execute: `plan/waves/W002/agents/agent-09.md`
### agent-10 → `PTGL-010` (pt-webgl2)

Read and execute: `plan/waves/W002/agents/agent-10.md`
### agent-11 → `FP-02` (engine)

Read and execute: `plan/waves/W002/agents/agent-11.md`
### agent-12 → `MUT-05` (pt-webgpu)

Read and execute: `plan/waves/W002/agents/agent-12.md`
### agent-13 → `PTWG-009` (pt-webgpu)

Read and execute: `plan/waves/W002/agents/agent-13.md`
### agent-14 → `PTWG-011` (pt-webgpu)

Read and execute: `plan/waves/W002/agents/agent-14.md`
### agent-15 → `WH-001` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-15.md`
### agent-16 → `WH-008` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-16.md`
### agent-17 → `WH-011` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-17.md`
### agent-18 → `WH-012` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-18.md`
### agent-19 → `WH-013` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-19.md`
### agent-20 → `WH-014` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-20.md`
### agent-21 → `WH-015` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-21.md`
### agent-22 → `WH-021` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-22.md`
### agent-23 → `WH-025` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-23.md`
### agent-24 → `WH-027` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-24.md`
### agent-25 → `WH-029` (walkaround-hybrid)

Read and execute: `plan/waves/W002/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W002
node tools/gap-scan/orchestrator-run.mjs --smoke W002
node tools/gap-scan/orchestrator-run.mjs --commit W002
node tools/gap-scan/orchestrator-run.mjs --advance
```
