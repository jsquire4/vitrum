# Dispatch W011

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `RT100-WA-ALPHA` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-01.md`
### agent-02 → `CORE-019` (core)

Read and execute: `plan/waves/W011/agents/agent-02.md`
### agent-03 → `PTWG-030` (pt-webgpu)

Read and execute: `plan/waves/W011/agents/agent-03.md`
### agent-04 → `PTWG-060` (pt-webgpu)

Read and execute: `plan/waves/W011/agents/agent-04.md`
### agent-05 → `PTWG-067` (pt-webgpu)

Read and execute: `plan/waves/W011/agents/agent-05.md`
### agent-06 → `PTWG-079` (pt-webgpu)

Read and execute: `plan/waves/W011/agents/agent-06.md`
### agent-07 → `WH-048` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-07.md`
### agent-08 → `WH-051` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-08.md`
### agent-09 → `WH-058` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-09.md`
### agent-10 → `WH-065` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-10.md`
### agent-11 → `WH-068` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-11.md`
### agent-12 → `WH-071` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-12.md`
### agent-13 → `WH-075` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-13.md`
### agent-14 → `WH-076` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-14.md`
### agent-15 → `WH-078` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-15.md`
### agent-16 → `WH-079` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-16.md`
### agent-17 → `WH-081` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-17.md`
### agent-18 → `WH-083` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-18.md`
### agent-19 → `WH-085` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-19.md`
### agent-20 → `WH-090` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-20.md`
### agent-21 → `WH-091` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-21.md`
### agent-22 → `WH-092` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-22.md`
### agent-23 → `WH-094` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-23.md`
### agent-24 → `WH-095` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-24.md`
### agent-25 → `WH-096` (walkaround-hybrid)

Read and execute: `plan/waves/W011/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W011
node tools/gap-scan/orchestrator-run.mjs --smoke W011
node tools/gap-scan/orchestrator-run.mjs --commit W011
node tools/gap-scan/orchestrator-run.mjs --advance
```
