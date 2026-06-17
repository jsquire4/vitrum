# Dispatch W000

**Tasks:** 8 | **Max agents:** 8

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `P0-001-PTWG-037` (pt-webgpu)

Read and execute: `plan/waves/W000/agents/agent-01.md`
### agent-02 → `P0-002-PTGL-003` (pt-webgl2)

Read and execute: `plan/waves/W000/agents/agent-02.md`
### agent-03 → `P0-003-WH-034` (walkaround-hybrid)

Read and execute: `plan/waves/W000/agents/agent-03.md`
### agent-04 → `P0-004-DENO-001` (core)

Read and execute: `plan/waves/W000/agents/agent-04.md`
### agent-05 → `P0-005-LEDGER-01` (core)

Read and execute: `plan/waves/W000/agents/agent-05.md`
### agent-06 → `P0-006-LEDGER-02` (core)

Read and execute: `plan/waves/W000/agents/agent-06.md`
### agent-07 → `P0-007-LEDGER-03` (core)

Read and execute: `plan/waves/W000/agents/agent-07.md`
### agent-08 → `P0-008-TOOL-001` (tools)

Read and execute: `plan/waves/W000/agents/agent-08.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W000
node tools/gap-scan/orchestrator-run.mjs --smoke W000
node tools/gap-scan/orchestrator-run.mjs --commit W000
node tools/gap-scan/orchestrator-run.mjs --advance
```
