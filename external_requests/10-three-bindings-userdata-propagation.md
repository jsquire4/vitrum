# RFE-10 — three-bindings userData Propagation (Status Update)

**Date:** 2026-05-09
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Status:** PARTIALLY CLOSED. `packages/three-bindings/src/material.ts` already reads
`userData.vitrum*` and forwards all RFE-06/07/08/03 fields into `vitrum.Material`. The
`three-bindings → vitrum.Material` direction is complete. The remaining open work is the
`vitrum.Material → fork uniform` direction, tracked separately as RFE-09.

---

## What this request is for

This RFE originally requested that `packages/three-bindings/src/material.ts` (the
`THREE.MeshPhysicalMaterial → vitrum.Material` converter used when a host passes a THREE
scene to `@vitrum/three-bindings`) read `userData.vitrum*` fields and forward them into
the resulting `vitrum.Material`.

As of the current codebase, that work is **done**. `material.ts` at lines 93–157 reads
and type-checks: `vitrumDispersionAbbeNumber`, `vitrumScatteringCoefficient`,
`vitrumScatteringCoefficientRGB`, `vitrumScatteringAnisotropy`, `vitrumSpectralAttenuation`
(with SpectralCurve object and Float32Array fallback), `vitrumThinFilmStack`,
`vitrumFrontLayer`, `vitrumBackLayer`.

This document is retained to:
1. Confirm the closure for external consumers who read the RFE index.
2. Document the agreed `vitrum*` prefix convention so future field additions follow it.
3. Redirect open concerns to RFE-09 (pt-webgl uniform bridge, still open) and RFE-12
   (layered BSDF plan doc, still open for RFE-03 fields).

---

## Affected files

- `packages/three-bindings/src/material.ts` — `convertMaterial()` at line 36. The
  `userData.vitrum*` reads at lines 93–157 ARE present. **No action needed.**
- `packages/pt-webgl/src/sceneToThree.ts` — `vitrumMaterialToThree()` at line 65 stamps
  `userData.vitrum*` on the outgoing THREE material. **No action needed.**
- `packages/pt-webgl/src/index.ts` — does NOT drive fork uniforms from the stamped
  `userData` fields. **This is the remaining open gap — see RFE-09.**

---

## Agreed userData key convention

All stainedGlass-side stamps and vitrum-side reads use the `vitrum` prefix:

| userData key | vitrum.Material field | RFE |
|---|---|---|
| `vitrumDispersionAbbeNumber` | `dispersionAbbeNumber` | RFE-06 |
| `vitrumScatteringCoefficient` | `scatteringCoefficient` | RFE-07 |
| `vitrumScatteringCoefficientRGB` | `scatteringCoefficientRGB` | RFE-07 |
| `vitrumScatteringAnisotropy` | `scatteringAnisotropy` | RFE-07 |
| `vitrumSpectralAttenuation` | `spectralAttenuation` | RFE-08 |
| `vitrumThinFilmStack` | `thinFilmStack` | RFE-08 |
| `vitrumFrontLayer` | `frontLayer` | RFE-03 |
| `vitrumBackLayer` | `backLayer` | RFE-03 |

Future extensions should continue the `vitrum` prefix. The `three-bindings/material.ts`
converter and `sceneToThree.ts` stamper both already follow this convention.

---

## Acceptance criteria

- [x] `packages/three-bindings/src/material.ts` reads all listed `userData.vitrum*` fields
  and forwards them into the returned `vitrum.Material`. (DONE — verified at lines 93–157.)
- [x] `packages/pt-webgl/src/sceneToThree.ts` stamps all listed fields onto THREE material
  `userData`. (DONE — verified at lines 100–136.)
- [ ] `packages/pt-webgl/src/index.ts` reads the `userData.vitrum*` stamps (or the originating
  `vitrum.Material` fields) and drives the corresponding fork uniforms on the
  `WebGLPathTracer` instance. (OPEN — tracked in RFE-09.)
- [ ] `packages/three-bindings/material.ts` handles new RFE-03 `frontLayer`/`backLayer`
  fields in the BSDF evaluation path once RFE-12 fork-patch plan lands. (DEFERRED — gated
  on RFE-12.)

---

## References

- `external_requests/09-pt-webgl-material-uniform-bridge.md` — remaining open gap.
- `external_requests/03-layered-bsdf.md` — RFE-03 fields (`frontLayer`/`backLayer`);
  three-bindings propagation is done; fork-side BSDF evaluation is open (RFE-12).
- `packages/three-bindings/src/material.ts` lines 93–157 — verified implementation.

---

## Out-of-scope notes

- This RFE does NOT cover the fork uniform upload (RFE-09).
- It does NOT cover the BSDF evaluation of `frontLayer`/`backLayer` (RFE-12).
- It does NOT cover changes to `@vitrum/core`'s `Material` type; those are the originating
  contracts and are already correct.

---

## Consumer-side state

The stainedGlass `vitrumMaterialAdapter.ts` (`packages/stained-glass-physics/src/baking/`)
independently converts baked THREE materials to `LocalMaterial` without depending on
`@vitrum/three-bindings`. This was necessary before `three-bindings/material.ts` was
updated. Now that the three-bindings path is complete, stainedGlass can optionally
switch to using `@vitrum/three-bindings` directly once it adopts `@vitrum/core` as a peer
dep; no immediate action required.
