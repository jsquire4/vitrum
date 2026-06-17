# Gap remediation worker — W003 / agent-21

You are **worker agent-21** in wave **W003**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `GLTF-008` |
| Lane | `gltf-adapter` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
GLTF-008: decodeSceneTextures raw-image only

## Files you may edit (ONLY these)
- `packages/gltf-adapter/src/texturePipeline.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/gltf-adapter/src/texturePipeline.ts:493.
2. Gap: decodeSceneTextures raw-image only
3. Fix: ImageBitmap CPU path
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/gltf-adapter && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- GLTF-008 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `GLTF-008`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: GLTF-008
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
