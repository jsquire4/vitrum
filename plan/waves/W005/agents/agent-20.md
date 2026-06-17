# Gap remediation worker — W005 / agent-20

You are **worker agent-20** in wave **W005**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `ENG-015` |
| Lane | `engine` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
ENG-015: target none without limits

## Files you may edit (ONLY these)
- `packages/engine/src/negotiateWebGPUDevice.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/engine/src/negotiateWebGPUDevice.ts:54-56.
2. Gap: target none without limits
3. Fix: Default target to backend
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- ENG-015 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `ENG-015`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: ENG-015
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
