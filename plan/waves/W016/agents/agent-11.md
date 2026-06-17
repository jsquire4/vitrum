# Gap remediation worker — W016 / agent-11

You are **worker agent-11** in wave **W016**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `DEV-006` |
| Lane | `dev` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
dev: GI overlay no skip reason

## Files you may edit (ONLY these)
- `packages/dev/src/react/GISignalSplit.tsx`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/dev/src/react/GISignalSplit.tsx.
2. Fix: Show skip reason from engine
3. Test in packages/dev.
4. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/dev && npx vitest run`
- `npm run typecheck`

## Definition of done
- DEV-006 complete.
- Tests green.

## Hard rules
1. Implement **only** task `DEV-006`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: DEV-006
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
