# Gap remediation worker — W016 / agent-05

You are **worker agent-05** in wave **W016**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `SBVH-008` |
| Lane | `shared-bvh` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
shared-bvh: pick primitive id stability

## Files you may edit (ONLY these)
- `packages/shared-bvh/src/pick.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/shared-bvh/src/pick.ts.
2. Fix: Regression test pick id
3. Test in packages/shared-bvh.
4. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/shared-bvh && npx vitest run`
- `npm run typecheck`

## Definition of done
- SBVH-008 complete.
- Tests green.

## Hard rules
1. Implement **only** task `SBVH-008`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: SBVH-008
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
