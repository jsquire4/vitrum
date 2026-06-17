# Dispatch W012

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `CORE-020` (core)

Read and execute: `plan/waves/W012/agents/agent-01.md`
### agent-02 → `PTWG-031` (pt-webgpu)

Read and execute: `plan/waves/W012/agents/agent-02.md`
### agent-03 → `PTWG-062` (pt-webgpu)

Read and execute: `plan/waves/W012/agents/agent-03.md`
### agent-04 → `PTWG-071` (pt-webgpu)

Read and execute: `plan/waves/W012/agents/agent-04.md`
### agent-05 → `WH-049` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-05.md`
### agent-06 → `WH-052` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-06.md`
### agent-07 → `WH-062` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-07.md`
### agent-08 → `WH-072` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-08.md`
### agent-09 → `WH-077` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-09.md`
### agent-10 → `WH-082` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-10.md`
### agent-11 → `WH-084` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-11.md`
### agent-12 → `WH-086` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-12.md`
### agent-13 → `WH-093` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-13.md`
### agent-14 → `WH-097` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-14.md`
### agent-15 → `WH-098` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-15.md`
### agent-16 → `WH-099` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-16.md`
### agent-17 → `WH-101` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-17.md`
### agent-18 → `WH-102` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-18.md`
### agent-19 → `WH-104` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-19.md`
### agent-20 → `WH-108` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-20.md`
### agent-21 → `WH-109` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-21.md`
### agent-22 → `WH-113` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-22.md`
### agent-23 → `WH-116` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-23.md`
### agent-24 → `WH-119` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-24.md`
### agent-25 → `WH-120` (walkaround-hybrid)

Read and execute: `plan/waves/W012/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W012
node tools/gap-scan/orchestrator-run.mjs --smoke W012
node tools/gap-scan/orchestrator-run.mjs --commit W012
node tools/gap-scan/orchestrator-run.mjs --advance
```
