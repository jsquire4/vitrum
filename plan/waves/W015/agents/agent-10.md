# Gap remediation worker — W015 / agent-10

You are **worker agent-10** in wave **W015**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `SSAMP-006` |
| Lane | `shared-samplers` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
shared-samplers: Preetham bake not live GLSL

## Files you may edit (ONLY these)
- `packages/shared-samplers/src/preetham.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/shared-samplers/src/preetham.ts.
2. Fix: Wire live eval or document bake-only
3. Test in packages/shared-samplers.
4. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/shared-samplers && npx vitest run`
- `npm run typecheck`

## Definition of done
- SSAMP-006 complete.
- Tests green.

## Hard rules
1. Implement **only** task `SSAMP-006`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: SSAMP-006
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
