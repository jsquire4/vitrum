# Gap remediation worker — W001 / agent-02

You are **worker agent-02** in wave **W001**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `CORE-002` |
| Lane | `core` |
| Disposition | `DOC` |
| Priority | `P1` |

## Problem
CORE-002: envMapIntensity JSDoc drift

## Files you may edit (ONLY these)
- `packages/core/src/scene/material.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/core/src/scene/material.ts:273-275.
2. Gap: envMapIntensity JSDoc drift
3. Fix: Align JSDoc with WALKAROUND_MATERIALS native row
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- CORE-002 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `CORE-002`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: CORE-002
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
