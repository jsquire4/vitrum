# Gap remediation worker — W023 / agent-01

You are **worker agent-01** in wave **W023**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `MAT-WH-shadingModel` |
| Lane | `core` |
| Disposition | `TOG` |
| Priority | `P2` |

## Problem
walkaround MaterialSpec.shadingModel is approximate (quantized atlas); ledger says approximate.

## Files you may edit (ONLY these)
- `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
- `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`
- `packages/core/src/engine/promiseLedger.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Decide per FP-06: promote shadingModel to native full-precision lane OR document permanent approximate.
2. If promoting: add full-precision slot in material atlas packing + shade.wgsl decode.
3. If permanent approximate: add EngineWarning code on setScene when field non-default.
4. Update promiseLedger WALKAROUND_MATERIALS.shadingModel row to match decision.
5. Add test in walkaround-hybrid material atlas tests.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run`
- `cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts`

## Definition of done
- shadingModel disposition recorded in ledger.
- Test covers chosen path.

## Hard rules
1. Implement **only** task `MAT-WH-shadingModel`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: MAT-WH-shadingModel
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
