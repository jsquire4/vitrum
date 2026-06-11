# Road to 100 handoff - 2026-06-11

This is a handoff note for the next agent continuing the road-to-100 gap closure
work. Source code is the source of truth. Do not rely on GitNexus in this repo:
it is currently broken in the desktop/UNC environment. Use direct source reads,
targeted grep/find, focused typechecks/tests under Linux Node, and renderer
oracles/reference renders.

## Current checkpoint

Branch: `main`

Working tree: clean after the last push.

Latest pushed commits:

- `880ba4f1 fix: mark displacement materials unsupported`
- `757477d4 fix: preserve gltf texture transform uv set`
- `84ab6236 feat: add structured engine warning surface`
- `f1b1dd79 fix: close second road-to-100 gap wave`
- `00047313 fix: close first road-to-100 gap wave`

All of the above are pushed to `origin/main`.

## Verification status from this checkpoint

Focused checks run after the latest work:

- `tsc --noEmit` passed for `packages/core`, `packages/pt-webgpu`,
  `packages/pt-webgl2`, and `packages/walkaround-hybrid`.
- Focused tests passed:
  - `packages/core/src/__tests__/engineContract.test.ts` - 17 tests
  - `packages/pt-webgpu/src/__tests__/liteTierCapabilities.test.ts` - 20 tests
  - `packages/pt-webgl2/src/__tests__/engineContract.test.ts` plus
    `packages/walkaround-hybrid/src/__tests__/consumedMaterialFields.test.ts` -
    29 tests
- Previous warning-surface wave checks passed:
  - Core/engine warning contract tests - 34 tests
  - pt-webgpu warning/capability tests - 26 tests
  - pt-webgl2/walkaround warning tests - 28 tests
- glTF adapter check passed:
  - `tsc --noEmit` for `packages/gltf-adapter`
  - `packages/gltf-adapter/src/gltfAdapter.test.ts` - 45 tests
- Every push hook in this run passed T1 GPU smoke on lavapipe and dzn plus RC,
  ReSTIR-TLAS, and DDGI brute-force GPU oracles.

Known verification not yet done:

- Full workspace `npm test` was not run from the desktop shell because the
  Windows npm/Vitest optional native package set does not match the WSL
  `node_modules`. Use Linux Node directly for focused tests.
- GPU/reference-render A/B remains pending for render-changing paths, especially
  WebGL2 tangent-space normal/bump maps and pt-webgpu SPPM photon-map scenes.

## What just landed

### First and second gap waves

The prior two pushed waves closed the concrete P0/P1 correctness items listed in
`plan/road-to-100-gap-ledger-2026-06-11.md`, including:

- Walkaround NRC slot claims now clear before active GI-RIS dispatch.
- Non-SVGF atrous chains bind per-iteration UBO ranges instead of reusing one
  UBO across queued dispatches.
- Walkaround async pipeline init failures route fatal `EngineError` records
  through `HybridEngine.onError`.
- Core animation rotation sampling normalizes LINEAR, STEP, clamped-knot, and
  CUBICSPLINE results.
- glTF skinned nodes preserve `bindMatrix` and `bindMatrixInverse`.
- pt-webgl2 consumes authored tangent XYZW, generates nonzero fallback
  handedness, guards legacy zero handedness in GLSL, and avoids rest-pose
  tangents after CPU skinning.
- glTF combined metallic-roughness texture maps to both roughness and metallic
  refs.
- pt-webgpu fatal `error` state blocks further operations.
- pt-webgpu emissive-to-zero material mutation repacks old or new implicit mesh
  emitters.
- pt-webgpu SPPM photon emission now normalizes by source-selection probability
  and covers directional, point, spot, rect, disc, mesh-area, and environment
  sources from the packed data used by NEE.
- pt-webgpu SPPM per-pixel progressive stats are gated to one update per pixel
  per frame at the first eligible diffuse-ish gather surface.
- pt-webgpu spectral attenuation no longer populates the Cauchy-dispersion Abbe
  lane.
- pt-webgpu lite capabilities are tier-specific and no longer overclaim
  instancing, transforms, or topology mutation support.
- pt-webgpu lite sampled light/environment textures refresh after emitter and
  environment mutations.

### ENGINE-01 - structured warning surface

Commit: `84ab6236`

Core additions:

- Added `EngineWarning` in `packages/core/src/engine/telemetry.ts`.
- Added `EngineOptions.onWarning?: (warning: EngineWarning) => void`.
- Added optional `Engine.onWarning?(cb): () => void`.
- Updated backend promise ledger so all three shipping backends promise
  `onWarning: true`.

Facade/backend wiring:

- `@vitrum/engine/createEngine` now emits structured warnings for fallback,
  TLAS-backend recommendations, ignored ownership keys, and cross-backend
  advanced-option application.
- `pt-webgpu`, `pt-webgl2`, and `walkaround-hybrid` preserve existing
  `console.warn` behavior while also emitting structured warnings for
  contract-affecting construction, scene, mutation, material-drop, viewport, and
  lite-tier downgrade paths.

Design caveat:

- Internal debug/resource chatter remains console-only by design. The structured
  surface is for contract-affecting nonfatal degradation.

### GLTF-06 partial - texture-transform UV set preservation

Commit: `757477d4`

The glTF adapter already mapped combined metallic-roughness textures to both
`roughnessMap` and `metallicMap`. The remaining source-backed sub-gap found here
was `KHR_texture_transform.texCoord`.

Fix:

- `packages/gltf-adapter/src/textures.ts` now honors
  `KHR_texture_transform.texCoord` as an override over the base texture-info
  `texCoord`.
- Added an end-to-end `gltfToScene()` fixture proving a decoded in-memory
  texture handle, transform fields, and `TextureRef.texCoord` survive import.

Status:

- This is a real GLTF-06 sub-gap closure, not full GLTF-06 closure. The adapter
  still needs per-KHR-extension fixture coverage for every imported texture map.

### MAT-02 - displacement map honesty closure

Commit: `880ba4f1`

Source finding:

- Core exposes `displacementMap`, `displacementScale`, and `displacementBias`.
- No backend renders displacement meaningfully.

Professional closure chosen:

- Do not fake displacement rendering.
- Add programmatic capability honesty and structured diagnostics.

Implementation:

- `BackendSupportDetails` now has a partial `materials` map.
- The backend promise ledger marks `displacementMap`, `displacementScale`, and
  `displacementBias` as `unsupported` on walkaround-hybrid, pt-webgl2, and
  pt-webgpu.
- pt-webgpu and pt-webgl2 emit structured
  `*.unsupported-displacement-material` warnings when scenes submit these fields.
- pt-webgpu also warns for displacement scale/bias material fast-path patches.
- walkaround-hybrid already warned on unconsumed material fields; the allowlist
  tests now explicitly pin displacement as unconsumed.

Status:

- Closed as an honesty/downgrade gap.
- Optional future implementation remains possible: authored geometry,
  tessellation, or parallax displacement plus renderer A/B evidence, then promote
  the support rows.

## Remaining gaps by priority

This is not "exactly N items." Use the ledger as a living source-backed plan.
The closure standard is categorical: each public field, backend promise, mutation
path, and renderer fidelity claim must be implemented, explicitly downgraded, or
removed from the advertised contract.

### P0/P1 leftovers

Most P0/P1 code items were closed in the latest waves. Remaining P0/P1 work is
mostly verification:

- GPU/reference-render A/B for WebGL2 tangent-space normal/bump maps.
- GPU/reference-render A/B for pt-webgpu SPPM photon-map scenes.
- Continue avoiding GitNexus; it is still unavailable.

### CAP-01 / GATE-02 - material support matrix

Current state:

- `supportDetails.materials` exists but currently only pins displacement fields.
- This is a start, not the full material matrix.

Plan:

1. Enumerate every `MaterialSpec` field from `packages/core/src/scene/material.ts`.
2. For each backend, read actual packers/shaders:
   - walkaround-hybrid BVH/DDGI/ReSTIR material ingestion,
   - pt-webgl2 material texture/GLSL packers,
   - pt-webgpu material descriptors/WGSL/BSDF paths.
3. Mark each field as `native`, `approximate`, `unsupported`, or omitted until
   audited.
4. Add diagnostics for ignored fields that affect output.
5. Add at least one executable test per implemented high-value row.

Important caveat:

- Do not mark a field supported just because it appears in core or in a packer.
  Confirm it is actually consumed by shader contribution paths.

### MAT-01 - authored tangent stream consumption backend-wide

Current state:

- pt-webgl2 is fixed: authored tangents are consumed and fallback handedness is
  nonzero.
- pt-webgpu and walkaround need backend-wide policy.

Plan:

1. For pt-webgpu, verify whether tangent streams influence all material frame
   construction paths that claim tangent-space map fidelity.
2. For walkaround, decide whether tangent-space maps are a supported material
   fidelity target. If not, emit structured warnings and material support rows.
3. Add tests for authored tangent, derived tangent, and mirrored UV cases.
4. For render-changing paths, capture reference A/B.

### SHADOW-01 - shadow flags

Current state:

- Core has `castShadow` and reserved `receiveShadow`.
- pt-webgl2 consumes `castShadow`.
- Backend-wide `receiveShadow` and emitter shadow flags still need audit.

Plan:

1. Read each backend's shadow-ray predicates and emitter packing.
2. Decide per backend:
   - implement caster disable,
   - implement receiver disable,
   - implement emitter shadow flags,
   - or mark unsupported with capability rows/warnings.
3. Add shadow ray tests:
   - disabled caster,
   - disabled receiver,
   - emitter shadow flag.

### WEBGL2-02 - pt-webgl2 procedural sky

Current state:

- pt-webgl2 capability lists only `none` and `hdri`.
- Procedural sky remains unsupported, which is honest but not feature-complete.

Plan:

1. Either implement procedural sky sampling/bake-to-env-map for pt-webgl2, or keep
   it unsupported with explicit capability details.
2. If implementing, use reference renders before promotion.

### WEBGL2-03 - pt-webgl2 denoiser option

Current state:

- pt-webgl2 warns for unsupported denoiser requests and returns disabled
  denoiser state.

Plan:

1. Decide whether this is acceptable as a professional unsupported contract row,
   or whether a denoiser should be wired.
2. If keeping unsupported, make it first-class in capability detail rather than
   relying only on warnings.

### WEBGL2-04 - pt-webgl2 material texture edge cases

Current state:

- Combined metallic-roughness import is closed.
- Remaining edge cases: `alphaMap` transform slot, layered front/back normal
  maps, surface anisotropy.

Plan:

1. Read `packages/pt-webgl2/src/scene/materialsTexture.ts`, attribute texture
   packing, and GLSL material decode.
2. Add alpha-map transform packing or structured unsupported diagnostic.
3. Implement or explicitly downgrade layered normal maps.
4. Implement or explicitly downgrade anisotropy.
5. Add tests for each path; A/B if render-changing.

### PTWG-08 / PTWG-MAT-01 - pt-webgpu material and lobe parity

Current state:

- pt-webgpu material texture array and descriptor paths exist.
- Some contribution/sampling paths still use base BRDF evaluation rather than
  full extension-lobe evaluation.

Plan:

1. Audit every `evaluateBrdf` and `brdfDirectionalPdf` call site.
2. For each path, decide whether full decoded material parameters must flow:
   - direct NEE,
   - BSDF area/env connection,
   - BDPT connection/light subpath,
   - SPPM,
   - ReSTIR-PT,
   - caustic paths,
   - lite tier.
3. Route full lobe contribution and matching MIS PDFs where support is promised.
4. Add material furnace and lobe-specific tests.
5. Promote fidelity rows only after tests/oracles pass.

### Renderer math oracles - P3

Do not "fix" these by intuition. Build independent oracles first.

Open oracle areas:

- HYB-GI-01: Direct ReSTIR selected `xi` should be used consistently for final
  pHat/direct shading.
- HYB-GI-02: DI RIS candidate accounting may undercount skipped proposals.
- HYB-DDGI-01: DDGI no-hit visibility moments may poison visibility.
- HYB-SKY-01: walkaround procedural sky is approximate and needs either parity
  with analytic/baked model or explicit downgrade/tests.
- PTWG-BDPT-01: BDPT needs independent radiometric oracle, not an oracle that
  mirrors shader assembly.
- PTWG-LITE-01: lite rect/disc area MIS is one-sided; implement ray-light
  connection or adjust weighting and test.

Plan:

1. Create small CPU/reference oracles that are independent of shader assembly.
2. Only change renderer math after an oracle proves bias.
3. For render-changing fixes, capture before/after reference renders.

### P4 glTF ingestion completeness

Current state:

- GLTF-01 skinned bind matrices are closed.
- GLTF-06 combined metallic-roughness and transform-level UV-set override are
  partially closed.

Still open:

- GLTF-02: Draco and meshopt compressed primitives are skipped.
- GLTF-03: glTF animations are not imported.
- GLTF-04: glTF morph targets are skipped by the adapter.
- GLTF-05: non-triangle primitive modes are skipped.
- GLTF-06: per-extension material texture fixture sweep is incomplete.

Plan:

1. GLTF-06 next small slice:
   - Add fixtures proving every imported KHR material extension texture preserves
     `texCoord` and `KHR_texture_transform`.
   - This is mostly test coverage because `resolveTextureRef()` is shared, but
     it is still useful as a public-ingestion contract.
2. GLTF-05:
   - Either triangulate triangle strip/fan or keep unsupported with diagnostics.
   - Add fixtures for strip/fan if supported.
3. GLTF-04:
   - Add core/adapter morph target deltas for POSITION/NORMAL/TANGENT and
     weights.
   - Cover triangle morph, normal morph, tangent morph, skinned+morphed.
4. GLTF-03:
   - Parse animation samplers/channels/interpolation and map to core animation
     clips.
   - Cover LINEAR, STEP, CUBICSPLINE, skin, morph-weight animations.
5. GLTF-02:
   - Add decoder hook/dependency strategy for `KHR_draco_mesh_compression` and
     `EXT_meshopt_compression`.
   - Add compressed fixtures for both paths.

## Recommended next execution order

1. Finish capability honesty before deep renderer work:
   - material support matrix,
   - shadow flags,
   - pt-webgl2 denoiser/procedural-sky capability rows.
2. Run the material contract audit gate so future gaps are found mechanically.
3. Build renderer math oracle suite before touching GI/BDPT/ReSTIR math.
4. Continue glTF ingestion in small slices: GLTF-06 fixture sweep, then GLTF-05,
   then morph/animation/compression.
5. Run reference-render A/B for any render-changing backend updates.

## Operational notes for the next agent

- Do not use GitNexus. The ledger explicitly records it as broken here.
- Use WSL/Linux Node directly. Known-good Node path:
  `/home/jsquire4/.cursor-server/bin/81fcf2931d7687b4ff3f3017858d0c6dee7e2a60/node`
- Prefer commands like:
  - `/home/jsquire4/.cursor-server/bin/81fcf2931d7687b4ff3f3017858d0c6dee7e2a60/node ./node_modules/typescript/bin/tsc -p packages/core/tsconfig.json --noEmit`
  - `/home/jsquire4/.cursor-server/bin/81fcf2931d7687b4ff3f3017858d0c6dee7e2a60/node ./node_modules/vitest/vitest.mjs run <test files>`
- Be careful with PowerShell parsing. For pipes/grep/head, wrap the WSL side in
  `bash -lc '...'` or avoid pipes.
- Commit coherent waves and push to `origin main`; the push hook runs GPU smoke.
- Do not claim "100%" until P5 gates pass and render-changing paths have
  independent or reference-render evidence.
