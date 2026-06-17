# Dispatch W022

**Tasks:** 2 | **Max agents:** 2

## Sub-agent prompts (copy each to a Task subagent)

### agent-01 → `MAT-WH-metallic` (core)

Read and execute: `plan/waves/W022/agents/agent-01.md`
### agent-02 → `INV-010` (pt-webgpu)

Read and execute: `plan/waves/W022/agents/agent-02.md`

## After all agents return

```bash
node tools/gap-scan/orchestrator-run.mjs --verify W022
node tools/gap-scan/orchestrator-run.mjs --smoke W022
node tools/gap-scan/orchestrator-run.mjs --commit W022
node tools/gap-scan/orchestrator-run.mjs --advance
```
