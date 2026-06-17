# Gap remediation worker — W000 / agent-02

You are **worker agent-02** in wave **W000**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `P0-002-PTGL-003` |
| Lane | `pt-webgl2` |
| Disposition | `BUG` |
| Priority | `P0` |

## Problem
pt-webgl2 tryFastPathMaterialMutation passes stale geoPack to packMeshAreaLights after materials repacked into nextGeoPack.

## Files you may edit (ONLY these)
- `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/pt-webgl2/src/scene/mutateSceneTextures.ts.
2. At line 216 change `packMeshAreaLights(nextScene, geoPack)` to `packMeshAreaLights(nextScene, nextGeoPack)`.
3. Verify return still sets geoPack: nextGeoPack at line ~232.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run src/scene/meshAreaLights.test.ts`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- Line 216 uses nextGeoPack.
- meshAreaLights tests green.
- typecheck green.

## Hard rules
1. Implement **only** task `P0-002-PTGL-003`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: P0-002-PTGL-003
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
