# Gap remediation worker — W008 / agent-04

You are **worker agent-04** in wave **W008**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-PTGL-MUT` |
| Lane | `pt-webgl2` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
pt-webgl2 geometry mutations: port TLAS/refit/splice from pt-webgpu sceneMutationRouter (road-to-100 §2D — transform/positions/topology still fallback-rebuild).

## Files you may edit (ONLY these)
- `packages/pt-webgl2/src/scene/mutateSceneTextures.ts`
- `packages/pt-webgpu/src/sceneMutationRouter.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Study pt-webgpu fast paths: transform, positions, topology-resize, instanced-topology.
2. Port applicable paths to pt-webgl2 WebGL2 buffer upload model.
3. Promote promiseLedger mutation rows from fallback-rebuild to native where implemented.
4. Extend updatePrimitiveIncremental tests on pt-webgl2.

## Tests you must run locally
- `cd packages/pt-webgl2 && npx vitest run`
- `cd packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts`

## Definition of done
- transform/positions/topology native or explicitly documented fallback.

## Hard rules
1. Implement **only** task `RT100-PTGL-MUT`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-PTGL-MUT
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
