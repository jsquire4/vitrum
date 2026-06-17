# Gap remediation worker — W004 / agent-17

You are **worker agent-17** in wave **W004**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `CORE-034` |
| Lane | `core` |
| Disposition | `DOC` |
| Priority | `P2` |

## Problem
CORE-034: AnimationClip no consumer

## Files you may edit (ONLY these)
- `packages/core/src/scene/animation.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/core/src/scene/animation.ts:5-8.
2. Gap: AnimationClip no consumer
3. Fix: Document host duty OR Engine.playAnimation
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- CORE-034 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `CORE-034`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: CORE-034
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
