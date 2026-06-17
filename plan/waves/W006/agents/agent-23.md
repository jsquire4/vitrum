# Gap remediation worker — W006 / agent-23

You are **worker agent-23** in wave **W006**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `PTGL-017` |
| Lane | `pt-webgl2` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
PTGL-017: emissiveMap needs CPU texels

## Files you may edit (ONLY these)
- `packages/pt-webgl2/src/scene/meshAreaLights.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/pt-webgl2/src/scene/meshAreaLights.ts:200.
2. Gap: emissiveMap needs CPU texels
3. Fix: Require decode in pipeline
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- PTGL-017 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `PTGL-017`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: PTGL-017
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
