# Gap remediation worker — W002 / agent-14

You are **worker agent-14** in wave **W002**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `PTWG-011` |
| Lane | `pt-webgpu` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
pt-webgpu: BDPT eye-stack OOM silent disable

## Files you may edit (ONLY these)
- `packages/pt-webgpu/src/bdpt/bdptSubsystem.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/pt-webgpu/src/bdpt/bdptSubsystem.ts.
2. Implement: onWarning when BDPT disabled OOM
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- PTWG-011 complete.
- Tests green.

## Hard rules
1. Implement **only** task `PTWG-011`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: PTWG-011
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
