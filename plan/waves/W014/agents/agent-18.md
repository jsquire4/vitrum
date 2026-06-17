# Gap remediation worker — W014 / agent-18

You are **worker agent-18** in wave **W014**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `SBVH-005` |
| Lane | `shared-bvh` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
shared-bvh: fingerprint sampling

## Files you may edit (ONLY these)
- `packages/shared-bvh/src/fingerprint.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/shared-bvh/src/fingerprint.ts.
2. Fix: Document collision probability
3. Test in packages/shared-bvh.
4. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/shared-bvh && npx vitest run`
- `npm run typecheck`

## Definition of done
- SBVH-005 complete.
- Tests green.

## Hard rules
1. Implement **only** task `SBVH-005`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: SBVH-005
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
