# Dispatch W008

**Tasks:** 25 | **Max agents:** 25

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `WH-007` (walkaround-hybrid)

Read and execute: `plan/waves/W008/agents/agent-01.md`
### agent-02 → `WH-031` (walkaround-hybrid)

Read and execute: `plan/waves/W008/agents/agent-02.md`
### agent-03 → `WH-039` (walkaround-hybrid)

Read and execute: `plan/waves/W008/agents/agent-03.md`
### agent-04 → `RT100-PTGL-MUT` (pt-webgl2)

Read and execute: `plan/waves/W008/agents/agent-04.md`
### agent-05 → `RT100-GLTF-PICK` (engine)

Read and execute: `plan/waves/W008/agents/agent-05.md`
### agent-06 → `CORE-016` (core)

Read and execute: `plan/waves/W008/agents/agent-06.md`
### agent-07 → `PTGL-015` (pt-webgl2)

Read and execute: `plan/waves/W008/agents/agent-07.md`
### agent-08 → `PTGL-019` (pt-webgl2)

Read and execute: `plan/waves/W008/agents/agent-08.md`
### agent-09 → `PTGL-025` (pt-webgl2)

Read and execute: `plan/waves/W008/agents/agent-09.md`
### agent-10 → `PTWG-015` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-10.md`
### agent-11 → `PTWG-022` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-11.md`
### agent-12 → `PTWG-026` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-12.md`
### agent-13 → `PTWG-027` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-13.md`
### agent-14 → `PTWG-034` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-14.md`
### agent-15 → `PTWG-035` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-15.md`
### agent-16 → `PTWG-036` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-16.md`
### agent-17 → `PTWG-038` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-17.md`
### agent-18 → `PTWG-039` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-18.md`
### agent-19 → `PTWG-043` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-19.md`
### agent-20 → `PTWG-044` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-20.md`
### agent-21 → `PTWG-045` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-21.md`
### agent-22 → `PTWG-047` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-22.md`
### agent-23 → `PTWG-048` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-23.md`
### agent-24 → `PTWG-049` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-24.md`
### agent-25 → `PTWG-050` (pt-webgpu)

Read and execute: `plan/waves/W008/agents/agent-25.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W008
node tools/gap-scan/orchestrator-run.mjs --smoke W008
node tools/gap-scan/orchestrator-run.mjs --commit W008
node tools/gap-scan/orchestrator-run.mjs --advance
```
