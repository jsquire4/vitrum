# Dispatch W010

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `RT100-WA-3D` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-01.md`
### agent-02 → `CORE-018` (core)

Read and execute: `plan/waves/W010/agents/agent-02.md`
### agent-03 → `PTWG-023` (pt-webgpu)

Read and execute: `plan/waves/W010/agents/agent-03.md`
### agent-04 → `PTWG-029` (pt-webgpu)

Read and execute: `plan/waves/W010/agents/agent-04.md`
### agent-05 → `PTWG-052` (pt-webgpu)

Read and execute: `plan/waves/W010/agents/agent-05.md`
### agent-06 → `PTWG-058` (pt-webgpu)

Read and execute: `plan/waves/W010/agents/agent-06.md`
### agent-07 → `PTWG-059` (pt-webgpu)

Read and execute: `plan/waves/W010/agents/agent-07.md`
### agent-08 → `PTWG-061` (pt-webgpu)

Read and execute: `plan/waves/W010/agents/agent-08.md`
### agent-09 → `PTWG-070` (pt-webgpu)

Read and execute: `plan/waves/W010/agents/agent-09.md`
### agent-10 → `PTWG-073` (pt-webgpu)

Read and execute: `plan/waves/W010/agents/agent-10.md`
### agent-11 → `PTWG-078` (pt-webgpu)

Read and execute: `plan/waves/W010/agents/agent-11.md`
### agent-12 → `WH-046` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-12.md`
### agent-13 → `WH-047` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-13.md`
### agent-14 → `WH-050` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-14.md`
### agent-15 → `WH-054` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-15.md`
### agent-16 → `WH-057` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-16.md`
### agent-17 → `WH-059` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-17.md`
### agent-18 → `WH-060` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-18.md`
### agent-19 → `WH-061` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-19.md`
### agent-20 → `WH-064` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-20.md`
### agent-21 → `WH-066` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-21.md`
### agent-22 → `WH-067` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-22.md`
### agent-23 → `WH-069` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-23.md`
### agent-24 → `WH-070` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-24.md`
### agent-25 → `WH-074` (walkaround-hybrid)

Read and execute: `plan/waves/W010/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W010
node tools/gap-scan/orchestrator-run.mjs --smoke W010
node tools/gap-scan/orchestrator-run.mjs --commit W010
node tools/gap-scan/orchestrator-run.mjs --advance
```
