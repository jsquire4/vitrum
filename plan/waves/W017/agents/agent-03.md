# Gap remediation worker — W017 / agent-03

You are **worker agent-03** in wave **W017**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RC-003` |
| Lane | `walkaround-rc` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
walkaround-rc: merged TLAS placeholders

## Files you may edit (ONLY these)
- `packages/walkaround-rc/src/cascadeDispatch.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/walkaround-rc/src/cascadeDispatch.ts.
2. Fix: Use shared TLAS packer
3. Test in packages/walkaround-rc.
4. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/walkaround-rc && npx vitest run`
- `npm run typecheck`

## Definition of done
- RC-003 complete.
- Tests green.

## Hard rules
1. Implement **only** task `RC-003`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RC-003
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
