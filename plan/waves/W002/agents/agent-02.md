# Gap remediation worker — W002 / agent-02

You are **worker agent-02** in wave **W002**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `CORE-005` |
| Lane | `core` |
| Disposition | `ACC` |
| Priority | `P1` |

## Problem
CORE-005: displacement* @reserved

## Files you may edit (ONLY these)
- `packages/core/src/scene/material.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/core/src/scene/material.ts:225-233.
2. Gap: displacement* @reserved
3. Fix: IMP on PT backend OR permanent ACC in ledger
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- CORE-005 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `CORE-005`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: CORE-005
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
