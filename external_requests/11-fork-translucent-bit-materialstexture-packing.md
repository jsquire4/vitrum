# RFE-11 — Fork: TRANSLUCENT_BIT Packing in MaterialsTexture.js

**Date:** 2026-05-09
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Status:** APPLIED (code landed in fork, runtime-unverified). `TRANSLUCENT_BIT` is now packed in
`MaterialsTexture.js` sample 14, exposed through `material_struct.glsl.js` (`uint flags`), and gated
in `PhysicalPathTracingMaterial.js` for per-material SSS selection.

---

## What this request is for

Sprint 7 (RFE-07, fork commit `260c432`) defined `TRANSLUCENT_BIT = 0x10u` in
`src/shader/bsdf/bsdf_functions.glsl.js` as the mechanism to auto-activate SSS for
glass types that are intrinsically scattering (opalescent, glueChip, ringMottled). The
constant exists in the GLSL shader, but the upstream JavaScript side —
`src/uniforms/MaterialsTexture.js` — does not yet pack a flags word into the material
texture. The GLSL `readMaterialInfo()` also does not expose a `flags` field in the
`material_struct`.

Until this gap is filled, opalescent/glueChip/ringMottled glass will only produce SSS
if the host directly sets `u_sssSigmaT > 0` as a global uniform. Per-material automatic
SSS activation by glass type — the documented goal of Sprint 7's `TRANSLUCENT_BIT` path
— is not usable. For multi-material scenes (a window with mixed clear + opalescent panels),
the global-uniform workaround cannot express per-material SSS.

---

## Affected files

- `src/shader/bsdf/bsdf_functions.glsl.js` (fork) — lines 14–21: `TRANSLUCENT_BIT`
  defined; TODO comment: "extend material_struct.glsl.js with a flags uint and pack
  TRANSLUCENT_BIT into MaterialsTexture.js sample 14."
- `src/uniforms/MaterialsTexture.js` (fork) — sample 14 block (around line 379):
  packs `matte`, `castShadow`, `vertexColors | flatShading`, `transparent`. No flags
  uint. No `TRANSLUCENT_BIT` packing.
- `src/shader/structs/material_struct.glsl.js` (fork) — the `Material` struct. Needs a
  `uint flags;` field added so `readMaterialInfo()` exposes the packed bits to BSDF code.
- `src/shader/bsdf/bsdf_functions.glsl.js` (fork) — the SSS branch currently gated on
  `u_sssSigmaT > 0` (global). The per-material gate `bool(material.flags & TRANSLUCENT_BIT)`
  is the documented follow-up.

---

## What the implementation must do

1. In `material_struct.glsl.js`: add `uint flags;` to the `Material` struct.
2. In `MaterialsTexture.js` sample 14: replace the existing transparent float with
   a packed uint that encodes `TRANSLUCENT_BIT` (bit 4) when the THREE material's
   `userData.vitrumScatteringCoefficient` is non-zero. The pack write site is the same
   sample 14 block already present at line 379.
3. In `bsdf_functions.glsl.js`: update the SSS branch to additionally check
   `bool(material.flags & TRANSLUCENT_BIT)` so per-material SSS activates automatically.
4. Glass-type mapping (host JS): `opalescent`, `glueChip`, `ringMottled` → set
   `TRANSLUCENT_BIT` when writing to `MaterialsTexture`.

The exact bit field layout (whether `TRANSLUCENT_BIT` shares the sample-14 float slot or
gets its own sample) is the fork authors' call; the criteria below are outcome-focused.

---

## Acceptance criteria

- [ ] `src/shader/structs/material_struct.glsl.js` contains a `flags` field (uint or
  equivalent) readable from `bsdf_functions.glsl.js`.
- [ ] `src/uniforms/MaterialsTexture.js` sets `TRANSLUCENT_BIT` for materials whose host
  THREE material carries `userData.vitrumScatteringCoefficient > 0`.
- [ ] In a multi-material scene, opalescent panels show SSS glow while adjacent cathedral
  panels (no `vitrumScatteringCoefficient`) render clear — without any host-set global
  `u_sssSigmaT`.
- [ ] The TODO comment in `bsdf_functions.glsl.js` lines 14–21 is resolved (removed or
  replaced with a "done" note citing the implementing commit).
- [ ] `npm test --workspaces --if-present` passes without regression.

---

## References

- `external_requests/07-sprint7-volume-scattering-fork-patch.md` — the originating Sprint 7
  RFE; §"Exact fork work required" item 5 specifies this gap.
- `IMPLEMENTATION-STATUS.md` §RFE-07 Gaps: "TRANSLUCENT_BIT is defined but not wired into
  MaterialsTexture.js packing … requires a follow-up MaterialsTexture.js extension."
- Fork: `src/shader/bsdf/bsdf_functions.glsl.js` lines 14–21 (the TODO site).
- Fork: `src/uniforms/MaterialsTexture.js` line 379 (sample 14 block — the write site).

---

## Out-of-scope notes

- This RFE does NOT change the `u_sssSigmaT` global uniform path, which remains as a
  scene-wide SSS override for homogeneous-medium effects.
- It does NOT address the full flags API for future material bits beyond `TRANSLUCENT_BIT`.
- It does NOT cover the layered BSDF flag path for RFE-03 (tracked in RFE-12).

---

## Consumer-side state

The stainedGlass baking pipeline (`glassMaterialProfiles.ts`) marks opalescent/glueChip/
ringMottled with `scatteringCoefficient > 0`. The adapter (`vitrumMaterialAdapter.ts`) reads
this and populates `LocalMaterial.scatteringCoefficient`. The field is also stamped at
`material.userData.vitrumScatteringCoefficient`. Once `MaterialsTexture.js` reads this stamp,
per-material SSS will auto-activate for the correct glass types without any host-side
uniform management.
