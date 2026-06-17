# Parent orchestrator agent — gap remediation

> **You are the orchestrator, not a worker.** Advance **all scheduled waves** until
> `plan/.gap-orchestrator-state.json` shows `"phase": "done"`.

## Worktree (mandatory)

```
/home/jsquire4/projects/vitrum-gap-remediation
```

Branch: `feat/gap-remediation`. Do not implement in the main vitrum worktree.

## Code-first policy

**GPU / rendering validation is deferred.** Tasks in `plan/VALIDATION-DEFERRED.md` are
not dispatched. Do not run wsl-gpu captures, golden PNG A/B, or behavioral-gate render
proof during this campaign. Unit tests + typecheck + shader-gate compile only.

## Git policy

- **Commit** after every wave: `node tools/gap-scan/orchestrator-run.mjs --commit --yes`
- **Never push** unless the user explicitly requests

## Start every session

1. Read `plan/ORCHESTRATOR_RESUME.md` (auto-generated).
2. Read `plan/.gap-orchestrator-state.json`.
3. Execute the **What to do NOW** section exactly.

## You must NOT

- Trust sub-agent "TASK_COMPLETE" claims without running `--verify`.
- Skip smoke or commit between waves.
- Stop after one wave unless verify/smoke hard-fails twice on the same task.
- Let sub-agents run verification or commit.
- Dispatch deferred validation tasks.

## Wave loop (repeat until done)

```
begin → dispatch (≤25 sub-agents) → verify → [remediate → verify]* → smoke → commit → advance
```

### 1. `begin`
```bash
node tools/gap-scan/orchestrator-run.mjs begin
```

### 2. `dispatch` — launch sub-agents
Spawn up to **25** `Task` subagents. Each reads **one** file:

```
plan/waves/<Wnnn>/agents/agent-NN.md
```

Worker template:
```
Read and execute the task in <absolute path>. Worker only — no commit, no progress files.
```

### 3. `verify` (orchestrator only)
```bash
node tools/gap-scan/orchestrator-run.mjs --verify
```

### 4. `smoke`
```bash
node tools/gap-scan/orchestrator-run.mjs --smoke
```

### 5. `commit`
```bash
node tools/gap-scan/orchestrator-run.mjs --commit --yes
```

### 6. `advance` → loop to `begin` unless `phase: done`

## Reporting

After each `advance`, output **one line**: `Wave N/M complete — next Wnnn`. No per-task narration.

## Kickoff

See `plan/KICKOFF-ORCHESTRATOR.md` for the paste-ready parent prompt.

## Regenerate artifacts

```bash
node tools/gap-scan/generate-implementation-plan.mjs
```
