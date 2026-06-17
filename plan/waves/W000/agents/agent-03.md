# Gap remediation worker — W000 / agent-03

You are **worker agent-03** in wave **W000**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `P0-003-WH-034` |
| Lane | `walkaround-hybrid` |
| Disposition | `BUG` |
| Priority | `P0` |

## Problem
walkaround material-only patch returns applySubsystems:false; DDGI probe cache only invalidated on emissive/transmission-threshold — not beer/attenuation/thickness.

## Files you may edit (ONLY these)
- `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`
- `packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open HybridEnginePrimitiveUpdates.ts function materialPatch (~line 1400+).
2. Add helper ddgiAffectingMaterialChanged(prev, next, patch) returning true when patch touches: attenuationColor, attenuationDistance, thickness, transmission, transmissionMap, thicknessMap, baseColor (beer tint), roughness, metallic, or any map field in MATERIAL_ATLAS_FIELDS.
3. After computing prevMaterial/nextMaterial, if ddgiAffectingMaterialChanged OR emitterAffectingMaterialChanged: call ctx.ddgi.invalidateProbeCache().
4. Keep applySubsystems: false — geometry unchanged.
5. Update comment at lines 1477-1480 documenting DDGI invalidation on beer/attenuation edits.
6. Add mutationMatrix.test.ts: patch attenuationDistance only → expect invalidateProbeCache mock called.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run src/__tests__/mutationMatrix.test.ts`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- attenuationDistance-only patch invalidates probe cache.
- applySubsystems remains false.
- mutationMatrix test pins behavior.

## Hard rules
1. Implement **only** task `P0-003-WH-034`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: P0-003-WH-034
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
