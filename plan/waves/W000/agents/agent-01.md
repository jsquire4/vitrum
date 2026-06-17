# Gap remediation worker — W000 / agent-01

You are **worker agent-01** in wave **W000**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `P0-001-PTWG-037` |
| Lane | `pt-webgpu` |
| Disposition | `BUG` |
| Priority | `P0` |

## Problem
pt-webgpu lite tier: material-only updatePrimitive() calls host.setScene(nextScene) and returns without writing materialsBuffer — GPU stays stale.

## Files you may edit (ONLY these)
- `packages/pt-webgpu/src/sceneMutationRouter.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/pt-webgpu/src/sceneMutationRouter.ts.
2. Find lines 315-318: `if (host.isLiteTier?.() === true && canFastPathMaterialPatch(fastPathPatch)) { host.setScene(nextScene); return; }`.
3. Delete that entire if-block (4 lines). Do NOT replace with another early return.
4. Confirm execution falls through to fastPaths array; material handler at ~530-592 calls packFoldedMaterialEntry + device.queue.writeBuffer to sceneBuffers.materialsBuffer.
5. Grep uploadSceneBuffers.ts / gpuResources for lite-tier materialsBuffer creation — if null on lite setScene, ensure materials buffer is allocated on lite path before this fix can work.
6. Run updatePrimitiveIncremental tests; add lite-tier material patch test if missing.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/__tests__/updatePrimitiveIncremental.test.ts`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- Lite material-only patch no longer calls full setScene.
- materialsBuffer receives queue.writeBuffer on lite.
- updatePrimitiveIncremental.test.ts green including lite case.
- Root typecheck green.

## Hard rules
1. Implement **only** task `P0-001-PTWG-037`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: P0-001-PTWG-037
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
