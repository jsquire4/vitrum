# Gap remediation worker — W015 / agent-11

You are **worker agent-11** in wave **W015**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `SDENO-001` |
| Lane | `shared-denoisers` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
shared-denoisers: SVGF one-shot unused

## Files you may edit (ONLY these)
- `packages/shared-denoisers/src/svgfReal.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/shared-denoisers/src/svgfReal.ts.
2. Fix: Remove dead export or wire
3. Test in packages/shared-denoisers.
4. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/shared-denoisers && npx vitest run`
- `npm run typecheck`

## Definition of done
- SDENO-001 complete.
- Tests green.

## Hard rules
1. Implement **only** task `SDENO-001`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: SDENO-001
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
