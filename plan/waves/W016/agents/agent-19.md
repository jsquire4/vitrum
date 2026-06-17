# Gap remediation worker — W016 / agent-19

You are **worker agent-19** in wave **W016**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `DEV-005` |
| Lane | `dev` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
dev: frameMonitor requires return value

## Files you may edit (ONLY these)
- `packages/dev/src/react/FrameMonitor.tsx`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/dev/src/react/FrameMonitor.tsx.
2. Fix: Document onFrame requirement
3. Test in packages/dev.
4. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/dev && npx vitest run`
- `npm run typecheck`

## Definition of done
- DEV-005 complete.
- Tests green.

## Hard rules
1. Implement **only** task `DEV-005`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: DEV-005
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
