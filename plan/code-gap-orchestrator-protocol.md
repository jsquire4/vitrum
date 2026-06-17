# Gap Remediation Orchestrator Protocol

> **Short answer to your questions:** Yes — prompts are written (71 waves, ≤25 agents each). Tasks are divvied by the scheduler (file-disjoint). Verification, smoke, commit, and wave progression are orchestrator-owned. Sub-agents do not self-certify.

**Road-to-100:** 36 `RT100-*` tasks from `plan/road-to-100.md` are merged into the same schedule (phases 8–12). Crosswalk: `plan/code-gap-road-to-100-crosswalk.md`.

## Artifact map

| Artifact | Purpose |
|----------|---------|
| `plan/ORCHESTRATOR_AGENT.md` | **Parent agent** meta-prompt — read every session |
| `plan/ORCHESTRATOR_RESUME.md` | Auto-generated “what to do now” |
| `plan/.gap-orchestrator-state.json` | Persistent wave + phase state |
| `plan/waves/Wnnn/manifest.json` | Wave metadata, lanes, commit message |
| `plan/waves/Wnnn/agents/agent-NN.md` | **Sub-agent prompt** (one task, full spec) |
| `plan/waves/Wnnn/DISPATCH.md` | Copy-paste dispatch checklist |
| `plan/waves/Wnnn/verify-spec.json` | Orchestrator verification contract |
| `plan/waves/Wnnn/verify-report.json` | Written by `--verify` |
| `plan/waves/Wnnn/remediation/` | Failed-task prompts (re-dispatch) |
| `plan/waves/Wnnn/smoke-report.json` | Written by `--smoke` |

## Wave count (not 19 — 71)

| Type | Count |
|------|------:|
| Total execution waves | **71** |
| Total scheduled tasks | **501** (465 code-gap + 36 RT100) |
| Full 25-agent waves | 15 |
| Bootstrap W000 | 8 agents |
| Single-task mutex waves | 44+ (mostly ledger/atlas chains) |

“~19 rounds” ≈ 501 tasks ÷ 25 — but file mutexes split that into **71 committed checkpoints**.

## Lifecycle per wave

```
begin → dispatch (≤25 sub-agents) → verify → [remediate → verify]* → smoke → commit → advance
```

### Build phase — sub-agents
- Orchestrator spawns ≤25 `Task` subagents.
- Each reads **one** `plan/waves/Wnnn/agents/agent-NN.md`.
- Prompts include: problem, files, steps, tests, done criteria.
- Sub-agents **must not** commit or update progress files.

### Verify phase — orchestrator ONLY
```bash
node tools/gap-scan/orchestrator-run.mjs --verify
```
Orchestrator checks (does **not** trust `TASK_COMPLETE` lines):
1. **git diff** `waveBaseSha..HEAD` touches each declared file
2. **Runs test commands** itself; requires exit 0

Failures → `plan/waves/Wnnn/remediation/agent-NN-TASKID.md` → re-dispatch those slots only.

### Smoke phase — orchestrator ONLY
```bash
node tools/gap-scan/orchestrator-run.mjs --smoke
```
- `npm run typecheck` always
- `vitest run` per lane in the wave
- Bootstrap W000: re-runs task-level tests as extra gate

### Commit phase — orchestrator ONLY
```bash
node tools/gap-scan/orchestrator-run.mjs --commit --yes
```
One wave → one git commit → rollback point.

### Advance
```bash
node tools/gap-scan/orchestrator-run.mjs advance
```
Increments wave. Parent orchestrator loops until `phase: "done"`.

## Keeping the orchestrator focused through all 70 waves

1. **State file** — `plan/.gap-orchestrator-state.json` survives context loss.
2. **Resume file** — regenerated after every state change.
3. **Parent prompt** — `ORCHESTRATOR_AGENT.md` says: do not stop until `phase: done`.
4. **Explicit progress** — orchestrator announces `Wave N/70 complete` after each `advance`.
5. **Cold start** — any agent reads `ORCHESTRATOR_RESUME.md` and continues.

## CLI reference

```bash
node tools/gap-scan/orchestrator-run.mjs status
node tools/gap-scan/orchestrator-run.mjs begin
node tools/gap-scan/orchestrator-run.mjs dispatch
node tools/gap-scan/orchestrator-run.mjs --verify
node tools/gap-scan/orchestrator-run.mjs --smoke
node tools/gap-scan/orchestrator-run.mjs --commit --yes
node tools/gap-scan/orchestrator-run.mjs advance
```

## Regenerate everything

```bash
node tools/gap-scan/generate-implementation-plan.mjs
# also runs generate-wave-manifests + resume-md
```

## What was NOT automated (honest limits)

- **Spawning 25 sub-agents** — parent orchestrator (you/Cursor) must call `Task` tool per slot; the script prints paths, it does not spawn agents.
- **Fully unattended CI** — could wrap `orchestrator-run.mjs` in a CI job later; today it is designed for Cursor parent + subagent pool.
- **Semantic code review** — verify checks git diff + tests, not “is the fix correct physics.” Failed smoke/integration is the backstop.
