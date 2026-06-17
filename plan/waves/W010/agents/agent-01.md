# Gap remediation worker — W010 / agent-01

You are **worker agent-01** in wave **W010**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-WA-3D` |
| Lane | `walkaround-hybrid` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
Walkaround Phase 3D tail: remaining atlas gaps — bump/displacement policy rows, morph-target UV deform refresh, narrower atlas refresh optimization (road-to-100 §3D footguns).

## Files you may edit (ONLY these)
- `packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts`
- `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`
- `packages/walkaround-hybrid/src/shaders/materialAtlas.wgsl.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Read road-to-100.md Phase 3D — atlas slices largely landed; identify remaining gaps vs Master checklist.
2. Implement bump map consumption if still missing from shade/ReSTIR paths.
3. Add morph-target UV deform detection → full atlas refresh or documented limitation.
4. Optional: narrower atlas refresh for map-handle-only edits (cost footgun).

## Tests you must run locally
- `cd packages/walkaround-hybrid && npx vitest run`
- `cd packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts`

## Definition of done
- 3D footguns closed or documented with tests.
- Ledger rows match CONSUMED_MATERIAL_FIELDS.

## Hard rules
1. Implement **only** task `RT100-WA-3D`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-WA-3D
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
