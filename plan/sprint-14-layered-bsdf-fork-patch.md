# Sprint 14 — Layered BSDF Fork Patch Plan (RFE-03 / RFE-12)

**Status: APPLIED on 2026-05-11 at fork commit ee379dc**
Supporting infrastructure (material struct, packing, surface record selection, roughness
override, activeLayerWeight helper) was already present from phase4-normalmap-shadow-rays
merge. This patch added the missing `color *= activeLayerWeight(surf, heroWavelength)` call
inside `bsdfEval` to complete the lobe-side application. Prerequisite commit also landed:
`d6b88b3` (adaptive tile sample-count fix). No GPU verification yet — A/B render is the gate.

## Goal

Implement front/back asymmetric layered BSDF support in the `three-gpu-pathtracer` fork so
`Material.frontLayer` / `Material.backLayer` data from `@vitrum/core` can modulate per-face
surface response without breaking Sprint 7 SSS and Sprint 12 spectral transport.

This document satisfies RFE-12 (plan-only request) and defines the implementation path for RFE-03.

## Scope

- In scope:
  - Per-face layer transmission (`frontLayer.transmission`, `backLayer.transmission`)
  - Per-face roughness override
  - Optional per-face normal-map override hooks (if available in packed material payload)
  - Composition with existing diffuse/specular/transmission/clearcoat lobes in fork BSDF
- Out of scope:
  - Full multi-layer statistical operators (Belcour 2018 full model)
  - New core contract fields (already present in `@vitrum/core`)
  - Walkaround-hybrid approximation path changes

## BSDF Call Site

- Primary insertion point: `src/shader/bsdf/bsdf_functions.glsl.js`
  - After `SurfaceRecord` setup and face-orientation detection (`surf.frontFace`)
  - Before lobe evaluation, compute active face layer:
    - front hit -> `frontLayer`
    - back hit -> `backLayer`
  - Apply layer transmission multiplier to scalar/spectral throughput path
  - Apply roughness override to `surf.filteredRoughness` / clearcoat roughness where specified

## Material Data Path

- Packing site: `src/uniforms/MaterialsTexture.js` sample extension
  - Pack compact per-face layer params into additional material slots:
    - `frontTransmission.rgb`, `frontRoughness`
    - `backTransmission.rgb`, `backRoughness`
    - flags for presence/absence of each face layer
- Struct site: `src/shader/structs/material_struct.glsl.js`
  - Add fields for the packed layer params + flags
- Host bridge site: `packages/pt-webgl/src/sceneToThree.ts` and uniform/material bridge
  - Read `userData.vitrumFrontLayer` / `userData.vitrumBackLayer`
  - Ensure packed material payload includes them for the fork

## Sprint 7 Composition Rule

- SSS remains gated by:
  - `TRANSLUCENT_BIT` and back-face traversal
  - `u_sssSigmaT > 0`
- Layered absorption composes multiplicatively with SSS:
  - Apply per-face layer transmission before SSS sampling weight update
  - Do not apply layer transmission a second time inside the SSS branch
  - This avoids double-counting absorption while preserving face asymmetry

## Sprint 12 Relationship

- Layered BSDF can land independently of Sprint 12 TMM and spectral attenuation upload.
- With hero-wavelength payload active, layer transmission should use hero-wavelength scalar mapping
  (same helper path used by other RGB->hero scalar conversions).

## Implementation Steps

1. Extend fork material packing/struct with front/back layer fields and flags.
2. Add BSDF-side helper to select active layer by `surf.frontFace`.
3. Modulate throughput and roughness using selected layer.
4. Wire pt-webgl material bridge for `vitrumFrontLayer` / `vitrumBackLayer`.
5. Add fork + bridge tests (packing + selection + regression coverage).
6. Capture A/B reference renders (front-vs-back asymmetric panel scene).

## Test / Verification Plan

- Mechanical:
  - `npm run lint` in fork
  - `npm run typecheck --workspace @vitrum/pt-webgl`
  - existing vitest suites + new layer packing/selection tests
- Visual:
  - Reference scene with one-sided frosted layer
  - Verify front and back views differ as expected
  - Confirm no regression on non-layered materials

## Effort Estimate

- Code implementation: 1.5-2.5 days
- Debug + visual tuning: 1 day
- Total: 2.5-3.5 days (GPU verification available)

## Trigger / Gate

- Start immediately after RFE-09 bridge stability check.
- Do not mark APPLIED until:
  - per-face asymmetry confirmed in A/B render set
  - no measurable regression in baseline glass scenes.
