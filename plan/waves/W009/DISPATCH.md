# Dispatch W009

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MUT-06` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-01.md`
### agent-02 → `WH-019` (walkaround-hybrid)

Read and execute: `plan/waves/W009/agents/agent-02.md`
### agent-03 → `WH-040` (walkaround-hybrid)

Read and execute: `plan/waves/W009/agents/agent-03.md`
### agent-04 → `WH-045` (walkaround-hybrid)

Read and execute: `plan/waves/W009/agents/agent-04.md`
### agent-05 → `CORE-017` (core)

Read and execute: `plan/waves/W009/agents/agent-05.md`
### agent-06 → `PTGL-016` (pt-webgl2)

Read and execute: `plan/waves/W009/agents/agent-06.md`
### agent-07 → `PTWG-020` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-07.md`
### agent-08 → `PTWG-024` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-08.md`
### agent-09 → `PTWG-028` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-09.md`
### agent-10 → `PTWG-040` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-10.md`
### agent-11 → `PTWG-041` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-11.md`
### agent-12 → `PTWG-042` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-12.md`
### agent-13 → `PTWG-046` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-13.md`
### agent-14 → `PTWG-051` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-14.md`
### agent-15 → `PTWG-053` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-15.md`
### agent-16 → `PTWG-054` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-16.md`
### agent-17 → `PTWG-055` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-17.md`
### agent-18 → `PTWG-056` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-18.md`
### agent-19 → `PTWG-057` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-19.md`
### agent-20 → `PTWG-063` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-20.md`
### agent-21 → `PTWG-065` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-21.md`
### agent-22 → `PTWG-066` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-22.md`
### agent-23 → `PTWG-068` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-23.md`
### agent-24 → `PTWG-069` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-24.md`
### agent-25 → `PTWG-077` (pt-webgpu)

Read and execute: `plan/waves/W009/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W009
node tools/gap-scan/orchestrator-run.mjs --smoke W009
node tools/gap-scan/orchestrator-run.mjs --commit W009
node tools/gap-scan/orchestrator-run.mjs --advance
```
