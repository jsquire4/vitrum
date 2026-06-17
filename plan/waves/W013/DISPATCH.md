# Dispatch W013

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MUT-10` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-01.md`
### agent-02 → `CORE-021` (core)

Read and execute: `plan/waves/W013/agents/agent-02.md`
### agent-03 → `PTWG-032` (pt-webgpu)

Read and execute: `plan/waves/W013/agents/agent-03.md`
### agent-04 → `PTWG-074` (pt-webgpu)

Read and execute: `plan/waves/W013/agents/agent-04.md`
### agent-05 → `WH-053` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-05.md`
### agent-06 → `WH-055` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-06.md`
### agent-07 → `WH-063` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-07.md`
### agent-08 → `WH-073` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-08.md`
### agent-09 → `WH-080` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-09.md`
### agent-10 → `WH-087` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-10.md`
### agent-11 → `WH-089` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-11.md`
### agent-12 → `WH-100` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-12.md`
### agent-13 → `WH-103` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-13.md`
### agent-14 → `WH-105` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-14.md`
### agent-15 → `WH-110` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-15.md`
### agent-16 → `WH-114` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-16.md`
### agent-17 → `WH-117` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-17.md`
### agent-18 → `WH-123` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-18.md`
### agent-19 → `WH-124` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-19.md`
### agent-20 → `WH-125` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-20.md`
### agent-21 → `WH-127` (walkaround-hybrid)

Read and execute: `plan/waves/W013/agents/agent-21.md`
### agent-22 → `GLTF-011` (gltf-adapter)

Read and execute: `plan/waves/W013/agents/agent-22.md`
### agent-23 → `GLTF-016` (gltf-adapter)

Read and execute: `plan/waves/W013/agents/agent-23.md`
### agent-24 → `GLTF-020` (gltf-adapter)

Read and execute: `plan/waves/W013/agents/agent-24.md`
### agent-25 → `GLTF-022` (gltf-adapter)

Read and execute: `plan/waves/W013/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W013
node tools/gap-scan/orchestrator-run.mjs --smoke W013
node tools/gap-scan/orchestrator-run.mjs --commit W013
node tools/gap-scan/orchestrator-run.mjs --advance
```
