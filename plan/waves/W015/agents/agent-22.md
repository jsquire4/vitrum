# Gap remediation worker — W015 / agent-22

You are **worker agent-22** in wave **W015**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `SL-003` |
| Lane | `scene-lighting` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
scene-lighting: heuristic skyIrradiance

## Files you may edit (ONLY these)
- `packages/scene-lighting/src/skyIrradiance.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/scene-lighting/src/skyIrradiance.ts.
2. Fix: Document heuristic
3. Test in packages/scene-lighting.
4. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/scene-lighting && npx vitest run`
- `npm run typecheck`

## Definition of done
- SL-003 complete.
- Tests green.

## Hard rules
1. Implement **only** task `SL-003`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: SL-003
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
