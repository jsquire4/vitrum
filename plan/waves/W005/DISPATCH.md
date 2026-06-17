# Dispatch W005

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MUT-02` (pt-webgl2)

Read and execute: `plan/waves/W005/agents/agent-01.md`
### agent-02 → `PTWG-006` (pt-webgpu)

Read and execute: `plan/waves/W005/agents/agent-02.md`
### agent-03 → `WH-004` (walkaround-hybrid)

Read and execute: `plan/waves/W005/agents/agent-03.md`
### agent-04 → `WH-023` (walkaround-hybrid)

Read and execute: `plan/waves/W005/agents/agent-04.md`
### agent-05 → `WH-037` (walkaround-hybrid)

Read and execute: `plan/waves/W005/agents/agent-05.md`
### agent-06 → `WH-044` (walkaround-hybrid)

Read and execute: `plan/waves/W005/agents/agent-06.md`
### agent-07 → `GLTF-003` (gltf-adapter)

Read and execute: `plan/waves/W005/agents/agent-07.md`
### agent-08 → `GLTF-007` (gltf-adapter)

Read and execute: `plan/waves/W005/agents/agent-08.md`
### agent-09 → `GLTF-010` (gltf-adapter)

Read and execute: `plan/waves/W005/agents/agent-09.md`
### agent-10 → `CORE-013` (core)

Read and execute: `plan/waves/W005/agents/agent-10.md`
### agent-11 → `CORE-027` (core)

Read and execute: `plan/waves/W005/agents/agent-11.md`
### agent-12 → `CORE-033` (core)

Read and execute: `plan/waves/W005/agents/agent-12.md`
### agent-13 → `CORE-044` (core)

Read and execute: `plan/waves/W005/agents/agent-13.md`
### agent-14 → `CORE-045` (core)

Read and execute: `plan/waves/W005/agents/agent-14.md`
### agent-15 → `CORE-046` (core)

Read and execute: `plan/waves/W005/agents/agent-15.md`
### agent-16 → `CORE-047` (core)

Read and execute: `plan/waves/W005/agents/agent-16.md`
### agent-17 → `ENG-011` (engine)

Read and execute: `plan/waves/W005/agents/agent-17.md`
### agent-18 → `ENG-012` (engine)

Read and execute: `plan/waves/W005/agents/agent-18.md`
### agent-19 → `ENG-014` (engine)

Read and execute: `plan/waves/W005/agents/agent-19.md`
### agent-20 → `ENG-015` (engine)

Read and execute: `plan/waves/W005/agents/agent-20.md`
### agent-21 → `ENG-016` (engine)

Read and execute: `plan/waves/W005/agents/agent-21.md`
### agent-22 → `ENG-017` (engine)

Read and execute: `plan/waves/W005/agents/agent-22.md`
### agent-23 → `ENG-018` (engine)

Read and execute: `plan/waves/W005/agents/agent-23.md`
### agent-24 → `ENG-019` (engine)

Read and execute: `plan/waves/W005/agents/agent-24.md`
### agent-25 → `ENG-020` (engine)

Read and execute: `plan/waves/W005/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W005
node tools/gap-scan/orchestrator-run.mjs --smoke W005
node tools/gap-scan/orchestrator-run.mjs --commit W005
node tools/gap-scan/orchestrator-run.mjs --advance
```
