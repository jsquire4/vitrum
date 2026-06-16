# Road to 100 gap ledger - 2026-06-11

This ledger is the source-backed plan for closing the remaining code gaps toward
a professional, fully implemented Vitrum library, ignoring public distribution,
release governance, and cross-host verification posture.

This is not meant to be read as "there are exactly N things left." The closure
standard is categorical: every public contract field, backend capability claim,
mutation pathway, and renderer fidelity promise must be either implemented,
explicitly reported as approximate/unsupported, or removed from the advertised
contract. A gap is not closed by a comment or by a mirrored test that copies the
same mistake as the implementation.

## Closure standard

- Source code is the source of truth. Documentation only counts after the code
  path and tests match it.
- Operational note: GitNexus is broken in this desktop/UNC environment
  (`detect-changes` and symbol impact can fail with LadybugDB Binder errors).
  Do not block gap-closure work on GitNexus. Use direct source reading,
  `rg`/call-chain review, targeted typecheck/tests, CPU oracles, and
  reference-render A/B checks as the practical impact-analysis path.
- Each confirmed bug needs a regression test or oracle, plus typecheck/test pass.
- Each renderer math change needs either a CPU oracle or a before/after reference
  render where numerical drift is visually justified.
- Each public contract field needs one of: backend implementation, structured
  capability detail, structured diagnostic, or a deliberate contract downgrade.
- Each "probably" item needs an oracle before implementation unless the source
  evidence is already conclusive.

## P0 correctness firebreak

These are correctness or state-corruption risks. Close these before calling the
library fully professional.

### Implementation wave status - 2026-06-11

Patched and source-reviewed in this wave, with focused typecheck/tests passing:

- W-HYB-01 NRC slot claims clear before NRC GI-RIS dispatch.
- W-HYB-02 non-SVGF atrous paths bind per-iteration UBO ranges.
- W-HYB-03 async walkaround pipeline init failures now route a fatal
  `EngineError` through `HybridEngine.onError` instead of console-only handling.
- CORE-01 animation rotation samples normalize LINEAR, STEP, clamped-knot, and
  CUBICSPLINE outputs.
- GLTF-01 skinned glTF nodes preserve `bindMatrix` and `bindMatrixInverse`,
  including non-translation transforms.
- GLTF-06 shared texture-info import now preserves the
  `KHR_texture_transform.texCoord` override, so imported maps can select UV1
  through the same `TextureRef.texCoord` path that pt-webgl2 and pt-webgpu
  consume.
- WEBGL2-01 pt-webgl2 consumes authored tangent XYZW, derives nonzero fallback
  handedness, guards legacy zero handedness in GLSL, preserves CPU-solved posed
  tangents after skinning, and avoids stale rest-pose tangent reuse.
- WEBGL2-04 glTF metallicRoughnessTexture maps to both roughness and metallic
  material-map refs while relying on backend atlas dedupe for storage.
- PTWG-01 pt-webgpu fatal `error` state blocks further engine operations.
- PTWG-02 pt-webgpu emissive-to-zero material mutation repacks old OR new
  implicit mesh emitters.
- PTWG-03 SPPM photon emission normalizes by source-selection probability and
  now covers directional/point/spot/rect/disc/mesh-area/environment sources from
  the same packed data and environment helpers used by NEE.
- PTWG-04 SPPM per-pixel progressive stats are gated to one update per pixel per
  frame at the first eligible diffuse-ish gather surface.
- PTWG-05 spectral attenuation no longer populates the Cauchy-dispersion Abbe
  lane; authored dispersion remains packed independently.
- PTWG-06 pt-webgpu lite capabilities are now tier-specific: lite no longer
  advertises native instanced-mesh support or transform/topology incremental
  mutation support, `setScene()` warns for instanced/non-identity-transform
  inputs, lite transform/instanced-topology patches throw before mutating
  TLAS-only buffers, and lite material support no longer inherits the full-tier
  group-3 texture/alpha/env/aniso rows. `supportDetails.materials` plus
  `pt-webgpu.unsupported-material-fields` now report the material maps and
  scalar fields that the lite shader cannot render.
- PTWG-07 pt-webgpu lite sampled light/environment textures refresh after
  emitter/environment mutations.
- ENGINE-01 structured nonfatal warning channel is now part of the core engine
  contract (`EngineWarning`, `EngineOptions.onWarning`, `Engine.onWarning`) and
  is wired through the createEngine facade plus the three shipping backends for
  contract-affecting warnings: ignored/degraded options, backend fallback,
  unsupported scene features, material-field drops, lite-tier downgrades, and
  mutation fallback warnings.
- MAT-02 displacement maps are no longer silently accepted as an implied
  renderer promise: `supportDetails.materials` now marks `displacementMap`,
  `displacementScale`, and `displacementBias` as unsupported on all three
  shipping backends; pt-webgpu and pt-webgl2 emit structured warnings when
  submitted scenes contain them, and walkaround-hybrid pins them through its
  unconsumed-material allowlist. 2026-06-15 follow-up: walkaround's diagnostic
  now also scans authored `analytic` primitive materials before mesh fallback
  and the `updatePrimitive(id, { material })` fast path, so unsupported
  displacement fields cannot bypass the warning on incremental material edits.

Follow-up Codex closure sweeps (WSL Node 24.13.0):

- Follow-up 2026-06-15: pt-webgpu inverse rendering stopped over-promising
  analytic path replay. At this intermediate point, `inverseSession` selected
  replay only for point plus center-sampled rect-area direct lights and fell back
  to finite difference for map-heavy materials, transmission, layered lobes,
  spectral/volume cases, environments, directional/spot/mesh-area lights, and
  other unsupported source terms. Focused inverse-session tests pinned the
  promoted rect-area path and every downgrade case; the 2026-06-16 follow-up
  below is the current broader direct-light scope.
- Follow-up 2026-06-16: the same pt-webgpu path-replay slice was implemented
  for delta directional, spot, and native disc-area direct lights too. The adjoint
  pass now binds packed directional/spot buffers, consumes directional/spot counts
  from a widened UBO, honors point/spot/rect/disc emitter shadow-disable lanes,
  and uses the native disc `πr²` center-sample area. Later 2026-06-16 follow-up:
  mesh-area direct lights joined the same scoped path via the existing packed
  mesh-triangle buffer and a deterministic center-sampled triangle area term.
  `inverseSession` now selects replay for delta directional, point, spot, and
  center-sampled rect/disc/mesh-area scenes; it still keeps soft-sun angular
  diameter, environment, indirect, most mapped/transmissive/layered/spectral/volume,
  full stochastic area sampling, and extension-lobe material domains on finite
  difference until those adjoints are implemented and validated. The current map
  exceptions are scoped camera-direct emissiveMap replay, baseColorMap/COLOR_0
  local chain factors for baseColor fits, roughnessMap/metallicMap local chain
  factors for ORM fits, specularColorMap/specularIntensityMap local chain factors
  for specular fits, clearcoat/sheen map local chain factors, iridescence/thickness
  map local state, and anisotropy-map strength/rotation local state, described
  below.
- Later 2026-06-16 follow-up: scalar `metallic` joined the same scoped
  path-replay domain. The CPU oracle differentiates the opaque base-BRDF diffuse
  fade-out and F0 blend, the emitted WGSL mirrors it, the engine scatters the new
  field code across all covered direct-light loops, and inverse-session routing
  selects path replay for `materials.<id>.metallic` when the material/light gates
  pass. This is code-closed for the scoped direct-light analytic path; metallic
  GPU inverse-fit recapture remains in the proof/promote tail with the other
  render-changing adjoint work.
- Later 2026-06-16 follow-up: scalar `emissiveIntensity` joined the same
  primary-hit emission adjoint domain as RGB `emissive`. The CPU oracle and WGSL
  now expose `dContribution_dEmissiveIntensity`; the private adjoint descriptor
  widened to two vec4 records per parameter so it can carry UNFACTORED emissive
  RGB and keep the intensity-zero derivative valid; inverse-session routing
  selects path replay for `materials.<id>.emissiveIntensity` under the existing
  opaque/unmapped emissive predicate. Later same-day follow-up: camera-direct
  `emissiveMap` modulation is now replayed too. `adjointPass` binds the existing
  UV/texture descriptor/sRGB texture resources, samples the hit-local emissive
  texel through the same per-map UV/transform/wrap/fit metadata as the forward
  material path, and multiplies both RGB `emissive` and scalar
  `emissiveIntensity` gradients by that texel. Alpha-map visibility,
  non-primary/indirect emission, and BRDF/material-map gradients remain
  finite-difference tails.
- Later 2026-06-16 follow-up: `baseColor` fits now replay baseColorMap and
  COLOR_0 as local chain-rule factors in the safe pt-webgpu path-replay adjoint
  domain. Lit direct-light BRDF partials use the hit-local effective base color,
  and `shadingModel:"unlit"` baseColor primary hits use the same local factor for
  the contribution identity (`radiance += throughput * baseColor * factor`). The
  routing is deliberately narrow: opaque, no other material maps, no transmission,
  no spectral/scattering, no layered/thin-film/generic extensions. Unlit
  non-baseColor parameters and non-baseColor mapped terms remain finite-difference.
- Next 2026-06-16 follow-up: roughnessMap and metallicMap joined that same
  scoped direct-light adjoint treatment. The adjoint pass now binds the linear
  material texture array, mirrors the forward ORM sampler (roughness G, metallic
  B, with each map's UV/transform/wrap metadata), evaluates BRDF partials with
  the hit-local mapped roughness/metallic values, and scatters scalar gradients
  through the local G/B chain factors. Normal/bump/AO/light/alpha/transmission
  maps and then-unimplemented extension-lobe maps remained finite-difference
  until their source terms, normals, visibility, emission, or lobe-specific
  derivatives were mirrored.
- Next 2026-06-16 follow-up: specularColorMap and specularIntensityMap joined
  the same scoped direct-light adjoint treatment. The adjoint pass mirrors the
  forward KHR_materials_specular samplers (sRGB RGB color, linear A-channel
  intensity, per-map UV/transform/wrap metadata), evaluates BRDF partials with
  the hit-local mapped specular terms, and scatters specularColor/specularIntensity
  gradients through the local map factors. Extension-lobe maps are covered by
  follow-ups below.
- Next 2026-06-16 follow-up: clearcoatMap, clearcoatRoughnessMap,
  sheenColorMap, and sheenRoughnessMap joined the same scoped direct-light
  adjoint treatment. The adjoint pass mirrors the forward extension samplers
  (linear clearcoat R, linear clearcoat roughness G, sRGB sheen RGB, linear
  sheen roughness A, each with its own UV/transform/wrap metadata), evaluates
  BRDF partials with the hit-local mapped extension-lobe terms, and scatters
  clearcoat/clearcoatRoughness/sheenColor/sheenRoughness gradients through the
  local map factors. ClearcoatNormalMap, iridescence/thickness maps,
  anisotropy maps, normal/visibility/path-changing maps, transmission,
  environment, indirect, stochastic area sampling, and GPU inverse-fit recapture
  remained open proof/implementation tails at that point.
- Next 2026-06-16 follow-up: iridescenceMap, iridescenceThicknessMap, and
  anisotropyMap joined the same scoped direct-light adjoint treatment. The
  adjoint pass mirrors the forward iridescence R scalar factor, thickness G
  range-collapse, anisotropy B strength multiplier, and anisotropy RG rotation
  offset with each map's UV/transform/wrap metadata. It evaluates BRDF partials
  with those hit-local mapped lobe states and scatters `iridescence` and
  `anisotropy` gradients through the local multipliers while leaving
  `iridescenceIor` and `anisotropyRotation` as shape/offset controls. Clearcoat
  normal maps, normal/visibility/path-changing maps, alpha/AO/light/transmission
  maps, environment, indirect, stochastic area sampling, and GPU inverse-fit
  recapture remain open proof/implementation tails.
- Later 2026-06-16 follow-up: map-free scalar `clearcoat` and
  `clearcoatRoughness` joined the same scoped pt-webgpu direct-light
  path-replay domain. The CPU oracle now mirrors the additive fixed-F0
  KHR_materials_clearcoat lobe and finite-difference-checks both scalar
  partials; the WGSL partial bundle mirrors that math; the adjoint pass reads
  packed material vec4 #23, scatters explicit clearcoat field codes across the
  covered deterministic direct-light loops, and `inverseSession` selects replay
  for those two fields when the material/light gates pass. 2026-06-16 follow-up:
  map-free `sheen` / `sheenColor` / `sheenRoughness` now have the same scoped
  direct-light adjoint treatment through CPU-FD-checked Charlie-lobe partials.
  Later same-day follow-up: map-free scalar `iridescence` and
  `iridescenceIor` now have the same scoped direct-light adjoint treatment;
  `iridescenceIor` is differentiated through a local symmetric derivative of
  the thin-film F0 term inside the replay pass, not through a full-render
  finite-difference probe. Clearcoat normal maps and thickness-range gradients
  remain open.
  Next same-day follow-up: map-free scalar `anisotropy` and
  `anisotropyRotation` joined this scoped direct-light adjoint treatment through
  a local symmetric derivative of the anisotropic GGX specular lobe, using the
  scalar descriptor lanes. Anisotropy maps are covered by the map follow-up
  above; transmission, environment, indirect, stochastic area sampling, and GPU
  inverse-fit recapture remain open proof/implementation tails.
- Follow-up 2026-06-15: walkaround material truthfulness was tightened instead
  of papered over. Textured `alphaMode:"blend"` materials now enter the same
  approximation diagnostic path as scalar fractional opacity, including
  `baseColorMap` alpha and `alphaMap`, while the ledger now labels GI
  rich-material support as approximate pending GPU A/B promotion. The
  2026-06-16 receiver-lobe target wave made GI producer/reuse `pHat`
  material-lobe aware, but the compact reservoir still stores geometry plus
  `Lo` and temporal previous-domain recast falls back to geometry when exact
  previous-camera reconstruction is unavailable.
- Later 2026-06-16 follow-up: transparent OIT direct sun is now
  cast-shadow-aware, and sky/environment lighting now uses a deterministic
  five-tap material-lobe estimate for camera-visible transparent layers in both
  HDRI-backed and no-HDRI scalar/procedural sky scenes. Emissive and light-map
  terms remain first-hit approximations, and transparent ReSTIR/GI participation
  is still not promoted, but direct sun, point/spot, and finite-emitter shadows
  now use the OIT material-atlas shadow walk with `castShadow:false` respected,
  scalar glass skipped, and blend blockers attenuated by `1 - alpha`.
- Latest 2026-06-16 follow-up: transparent OIT now also binds the existing
  `analytic_lights` texture and shades analytic point/spot emitters through the
  same material-lobe payload, inverse-square falloff, spot cone attenuation, and
  cast-shadow-aware visibility convention as opaque shade. Area-emitter/ReSTIR
  direct light, transparent GI/RC/DDGI/ReSTIR-GI participation,
  alpha-map-aware shadow filtering, and first-hit emissive/light-map semantics
  remain open approximation/promotion tails.
- Follow-up 2026-06-15: neural/NRC production posture is explicit. Neural graph
  weights are validated for layer coverage, lengths, and finite values before
  GPU allocation; the tracked `starter-v1.vitrum-model` and
  `v2-random.vitrum-model` files load through the runtime loader/spec validator;
  inference dispatch exceptions fall back to the raw HDR texture with a recorded
  fallback reason; NRC record unpacking now scans the whole encoded-input prefix
  before declaring an empty slot; and `nrcEnabled:true` emits a structured
  experimental/biased warning instead of silently presenting NRC as production
  default behavior.
- Later 2026-06-16 follow-up: opt-in NRC substitution is warm-up gated. The
  NRC config UBO now exposes `trainedSteps` and `warmupSteps`; the GI-RIS NRC
  shader continues to gather records whenever the spread heuristic fires, but it
  keeps the DDGI suffix radiance in the visible reservoir until completed trainer
  windows reach the warm-up threshold. NRC remains off by default and still
  needs quality/convergence A/B before any default-tier promotion.
- Follow-up 2026-06-15/16: the top Road summary was reconciled with the detailed
  ledger. Closed items such as DDGI glossy bounce, stale NRC structural defects,
  strict progressive glTF tiering, optional Draco fallback analysis, and
  transparent HDRI sky material-lobe OIT no longer appear as open implementation
  gaps. The honest remaining tails are now analytic-adjoint breadth,
  walkaround transparent light-map/emissive plus ReSTIR/GI
  promotion, rich-material GI validation/promotion, neural/NRC production
  quality, and validation-backed fidelity promotions.
- Follow-up 2026-06-15: the glTF texture decode bridge now has a real
  WebGPU-target path. `decodeSceneTextures(target:"webgpu")` resolves raw image
  handles through `decodePixels` into CPU-readable float handles instead of
  returning a no-op success, while preserving sRGB-valued pixels for sRGB
  material maps so pt-webgpu's sRGB texture formats remain correct. The
  focused adapter test now proves raw-image refs become CPU-readable and clear
  the opaque-readiness report for pt-webgpu.
- Follow-up 2026-06-15: pt-webgpu material-lobe unit proof was narrowed and
  executed. `extensionLobeReference.test.ts` now independently pins clearcoat
  zero-default/scalar-linearity, sheen zero-default/color/scalar behavior,
  normalized base/clearcoat/sheen sampled-PDF accounting, iridescence F0
  zero-default behavior, and the explicit sheen-PDF cosine approximation
  posture. GPU material-furnace/reference A/B remains open.
- Follow-up 2026-06-16: the H24 Welford aux allocation residue is code-closed
  for non-Welford denoisers. `WalkaroundGPUPipeline` now passes
  `welfordPingPong:false` for non-`atrous-variance` modes, so
  `varianceBufferAux` and `atrousVarianceEstimateTexture` collapse to legal
  1x1 placeholders while the default Welford denoiser keeps full-size
  resources. `varianceBuffer` and `hdrTotalTexture` intentionally remain
  full-size because they are part of the standard frame graph.
- `attachVitrum` now sizes the canvas backing store from CSS size × DPR before
  `createEngine()` runs and forwards `onWarning` through the facade.
- `createProgressiveEngine.onError` now mirrors the same
  `(error, CreateEngineErrorEvent)` construction/plumbing callback shape as
  `createEngine()` instead of dropping phase/backend/recoverability metadata.
  Its facade-owned best-effort canvas configure now reports the canonical
  `canvas-configure` event, and focused engine tests pin sub-build plus
  facade-configure forwarding.
- DDGI runtime failures now propagate through the engine error surface:
  `DDGI` accepts a structured non-fatal `EngineError` sink, reports GPU-init,
  BVH-update, and probe `runFrame` failures, and `HybridEngine` wires that sink
  to `onError`; focused tests pin both direct DDGI diagnostics and engine
  forwarding.
- BVH per-triangle Beer/rough-metal/emissive textures now guard
  `device.limits.maxTextureDimension2D` and in-place refresh capacity before
  `createTexture()`/padding allocation, with tests covering width overflow,
  height overflow, stale refreshes, and large triangle counts that previously
  could be truncated by 32-bit coercion.
- `BvhBufferHost` no longer derives emitter count from a hardcoded `/ 80`:
  the canonical `EMITTER_TRI_STRIDE_BYTES` layout constant is exported from the
  emitter packer, `uploadInitial`/`updateEmitters` validate byte alignment and
  declared count before upload, and tests pin the update path plus malformed
  payload rejection.
- ReSTIR BVH material resolution no longer falls back to material slot 0:
  duplicate mesh-like primitive ids now throw before TLAS packing, and unknown
  resolver calls throw instead of silently shading with the first material.
  `bvhCoreMaterialResolver.test.ts` pins unique-id material packing and the
  duplicate-id rejection.
- GPU skinning no longer drops skinned meshes when host BVH GPU resources are
  temporarily unavailable: absent position/normal buffers or mesh ranges now
  route every skinned mesh through the CPU `solveSkin()` fallback. The older
  count-only-cache concern is source-verified stale: cached bind groups are
  already invalidated by live shared position/normal buffer identity.
- NRC self-training failures are no longer swallowed: `WalkaroundGPUPipeline`
  reports rejecting `trainFromRecords()` calls as deduped non-fatal
  `EngineError`s through `HybridEngine.onError` while the pipeline is live, and
  still suppresses expected late failures after dispose. Focused diagnostics
  tests pin report, dedupe, success reset, and post-dispose suppression.
- `<VitrumCanvas>` now applies creation-time `advanced` prop identity changes by
  recreating the engine; `onAttachError` remains ref-stabilized.
- `<VitrumCanvas>` now accepts a creation-time `gltf` prop plus bridge-level
  `gltfOptions`, loads through `loadGltfWithEngine()`, forwards decoder hooks,
  texture decode options, and compatibility modes into the same one-call glTF
  path, and hands the prepared engine/controller to `attachVitrum`. The
  lifecycle now accepts a preconstructed engine plus structural scene controller,
  and re-targets that controller after device-loss auto-recreate. Focused
  React/lifecycle tests pin both wrapper and forwarding seams.
- `ProgressiveHandoffCoordinator` can be constructed with an authoritative
  `scene` snapshot and falls back to `setScene()` on both engines when either
  incremental primitive path is missing or rejects; fallback patching uses the
  core `patchPrimitiveInScene()` invariant layer.
- `ProgressiveHandoffCoordinator` also accepts a structural animation
  `controller` (for example `GltfSceneController`) and advances it once per
  `frame()`, routing controller `setScene` / `updatePrimitive` calls through
  the same dual-engine scene-authority path so animated handoff cannot patch
  only one side.
- `gltf-adapter` decodes embedded `data:` URI images locally and rejects
  unsupported `extensionsRequired` entries instead of silently importing an
  invalid scene.
- `RCSubsystem.updateLights([])` now invalidates dispatcher bindings when the
  analytic light buffer is destroyed on a nonzero -> zero transition.
- `pt-webgpu` SPPM per-pixel stats allocation now replaces stale oversized
  buffers with a 64-byte placeholder, resets recorded stats dimensions to zero,
  invalidates group-3 bindings, and excludes SPPM from `sppmReady` when pixel
  stats allocation fails.
- The other-agent glTF/API feedback was reviewed against source. Its material
  support counts are valid, and the pt-webgl2 static ledger drift was corrected:
  full-tier aux buffers and `debug.pickPrimitive` are now represented in
  `BACKEND_PROMISE_LEDGER` and pinned by `ledgerVsCapabilities.test.ts`.
- The first arbitrary-glTF API slice landed in `@vitrum/gltf-adapter`:
  `loadGltfAsset()`, `analyzeGltfAsset()`, backend compatibility ranking, and
  external image-byte plumbing through `gltfToScene()`. URL/base-URI JSON glTF
  can now fetch external `.bin` buffers and external image URIs with host fetch /
  decode hooks, and the result carries a structured feature report plus backend
  recommendations.
- The second arbitrary-glTF runtime slice landed in `@vitrum/gltf-adapter`:
  `GltfSceneController` / `createGltfSceneController()` now retain the imported
  glTF node hierarchy, evaluate core `AnimationClip`s, recompute hierarchical
  world transforms, rebuild skinned bone matrices from animated joint nodes,
  solve morph weights through `solveSkin()`, and dispatch primitive patches via
  `updatePrimitive()` with `setScene()` fallback. Successful incremental
  animation/variant patch batches now call the target's optional `reset()` hook,
  so engine targets invalidate PT accumulators / temporal GI reservoirs without
  a second host-side call. The controller fixtures cover ancestor-node
  animation, reset propagation, mutation fallback, morph-weight playback, and
  skeletal joint playback.
- The third arbitrary-glTF extension-policy slice landed in `@vitrum/gltf-adapter`:
  `KHR_materials_dispersion` now imports to `MaterialSpec.dispersionAbbeNumber`
  (`dispersion = 20 / Abbe`), texture-source extensions
  (`KHR_texture_basisu`, `EXT_texture_webp`, `MSFT_texture_dds`) are opt-in via
  `textureSourceExtensions` and route alternate image sources through the host
  image decoder, required/no-base-fallback texture-source extensions fail
  deterministically unless enabled, optional alternates with a base
  `texture.source` fallback do not create degraded-compatibility issues, and
  `KHR_materials_variants` now selects primitive material mappings by variant
  name or index.
- The fourth arbitrary-glTF material-mapping slice landed in
  `@vitrum/gltf-adapter`: `gltfTextureSweep.test.ts` now pins every imported
  base/KHR material texture map through `TextureRef.handle`,
  `TextureRef.texCoord`, and `KHR_texture_transform` offset/scale/rotation
  preservation, including the shared metallic-roughness ORM texture mapping to
  both `roughnessMap` and `metallicMap`. The same sweep now pins
  `KHR_materials_volume.thicknessTexture` import through the reserved
  `MaterialSpec.thicknessMap` field.
- Follow-up 2026-06-15: glTF sampler filter intent no longer drops at the
  `TextureRef` boundary. Core `TextureRef` now carries authored
  `magFilter`/`minFilter`/`mipFilter`, the adapter maps all glTF sampler
  minification/magnification constants into those fields, and
  `textureDecodeReport` exposes the same sampler policy plus `usesMipmaps`.
  The existing texture sweep now proves every imported base/KHR material map
  preserves handle, UV channel, transform, wrap, and sampler filter/mipmap
  intent. Backend per-texture filter/mipmap enforcement remains a renderer
  policy gap rather than an import/data-loss gap.
- Follow-up 2026-06-15: the renderer-policy half is now explicit in the glTF
  compatibility planner. `analyzeGltfAsset()` inventories authored sampler
  policies per material texture, `evaluateGltfBackendCompatibility()` emits
  `*.samplerPolicy` approximate issues when a backend imports the texture but
  cannot guarantee exact per-texture filter/mipmap behavior, and
  `loadGltfForEngine(..., compatibilityMode:'reject-degraded')` rejects those
  sampler-policy degradations before constructing an engine. This does not
  claim backend sampler enforcement; it makes the non-enforcement visible and
  gateable. The current full glTF adapter suite is 202 tests.
- Follow-up 2026-06-15: walkaround-hybrid now honors ordinary glTF
  baseColorTexture alpha in textured alpha traversal. `materialAtlas.wgsl.ts`
  multiplies baseColorMap `.a` by optional `alphaMap.r` and scalar opacity,
  while `gltfAdapter.test.ts` keeps the adapter contract honest: MASK
  baseColorTexture assets stay as `baseColorMap` plus `alphaMode`/`alphaCutoff`
  rather than inventing a backend-specific `alphaMap` alias.
- The pt-webgpu material texture backend consumption slice landed after that
  adapter import work: `materialTextures.ts` packs per-map UV metadata for every
  map the backend currently samples (baseColor, emissive, normal, roughnessMap,
  metallicMap, AO, lightMap, bumpMap, anisotropyMap, alphaMap,
  transmissionMap), and
  `material.wgsl.ts` samples those maps with their own `TextureRef.texCoord`,
  KHR_texture_transform, wrap modes, and heterogeneous-layer UV-fit scales.
  The core promise ledger now promotes alpha/transmission/emissive/AO/light/
  bump/anisotropy maps where this was the remaining approximation. Follow-up
  2026-06-13: full-tier pt-webgpu now also uploads authored/generated
  tangent.xyzw and uses handedness for normal/bump/clearcoat-normal maps, so
  `normalMap` is native on pt-webgpu full tier.
- Follow-up 2026-06-13: pt-webgpu full-tier split the old combined ORM
  descriptor into distinct `roughnessMap` and `metallicMap` slots with
  independent UV/wrap/UV-fit metadata while preserving canonical glTF
  metallicRoughness textures by pointing both slots at one shared layer. The
  core promise ledger now promotes both rows to `native`.
- Follow-up 2026-06-14: walkaround readable `transmissionMap` handles now pack
  into the material texture atlas as linear R-channel layers with uv0/uv1,
  wrap-mode, and `KHR_texture_transform` metadata. `shade.wgsl`, `ris.wgsl`,
  `risGi.wgsl`, and `risGiNrc.wgsl` classify glass from
  `scalarTransmission * transmissionMap.r`; the ledger grades the row
  `approximate` because emitter power, direct-light candidate payloads, and GI
  payload lanes remain scalar.
- Follow-up 2026-06-14: walkaround readable `normalMap` handles now pack into
  the same material texture atlas as linear tangent-space RGB with uv0/uv1,
  wrap-mode, `KHR_texture_transform`, and `normalScale` metadata.
- Follow-up 2026-06-15: walkaround now carries authored/generated tangent.xyzw
  through both BVH modes. TLAS uses `packSceneFromCore().tangents`; merged-world
  transforms tangent directions and flips handedness for mirrored transforms;
  `BvhBufferHost` uploads the vec4 stream as `bvh_tangent` scene binding 22
  (`rgba32float` texture); `materialAtlas.wgsl` prefers it for normal and
  clearcoat-normal TBN reconstruction before falling back to UV-gradient
  derivation. The ledger still grades `normalMap`/`normalScale` `approximate`
  because reservoir/GI material payload parity is open, not because tangents
  are dropped.
- Follow-up 2026-06-14: walkaround readable `lightMap` handles now pack into
  the material texture atlas as linear RGB layers with uv0/uv1, wrap-mode,
  `KHR_texture_transform`, and `lightMapIntensity` metadata. `shade.wgsl`
  adds the sampled value as camera-visible first-hit baked outgoing radiance.
  The ledger grades `lightMap`/`lightMapIntensity` `approximate` because the
  baked-light term does not feed ReSTIR emitter power, reservoir payloads, or GI.
  `HybridEngine.updatePrimitive(material)` now rebuilds the atlas when
  `lightMap`, `lightMapIntensity`, or alpha-map coverage metadata changes, so
  scalar metadata edits cannot leave stale atlas rows.
- SPEC-01 pt-webgpu scalar `KHR_materials_specular` consumption landed: material
  vec4 #27 carries `specularColor.rgb` + `specularIntensity`, `material.wgsl.ts`
  decodes them, `bsdf.wgsl.ts` uses them for dielectric F0, and the scalar pair
  now flows through ordinary PT BRDF/PDF paths, lite/full env connection
  interfaces, MNEE/SPPM receiver paths, and BDPT light-subpath surface
  scattering. The promise ledger intentionally keeps pt-webgpu
  `specularColor`/`specularIntensity` at `approximate` until inverse/adjoint
  gradients and remaining specialty texture-map payload schemas carry the same
  fields. Verification: focused pt-webgpu material/WGSL/BDPT suites, core
  ledger/capability suites, pt-webgpu typecheck, `git diff --check`, and
  `npm run shader-gate -- --self-test` (47 production shaders OK; injected
  self-test failure detected).
- The fifth arbitrary-glTF API/compatibility slice landed in
  `@vitrum/gltf-adapter`: `loadGltfForEngine()` now combines `loadGltfAsset()`,
  backend compatibility selection, optional compatibility rejection, injected
  engine construction or existing-engine attachment, and
  `GltfSceneController` creation without adding an `@vitrum/engine` dependency.
  Compatibility scoring now treats morph-target `TANGENT` deltas as structured
  approximate issues instead of passive inventory counts: data is preserved on
  `SkinnedMeshPrimitive.morphTargetTangents`, backend tangent-space shading
  consumption remains a truthful downgrade, and
  `loadGltfForEngine(..., compatibilityMode: 'reject-degraded')` discounts
  host hooks that were actually supplied for Draco, meshopt, and texture-source
  extensions, plus optional texture-source alternates that have a deterministic
  base `texture.source` fallback.
- The sixth arbitrary-glTF extension-policy slice landed in
  `@vitrum/gltf-adapter`: archived
  `KHR_materials_pbrSpecularGlossiness` scalar `diffuseFactor`,
  `specularFactor`, and `glossinessFactor` are now converted approximately to
  `baseColor`, `specularColor`, `roughness = 1 - glossiness`, and
  `metallic = 0`, with raw extension data preserved. Its
  `specularGlossinessTexture` RGB path maps to `specularColorMap`; raw import
  remains a warned, structured approximate downgrade until pixels are decoded,
  and the CPU-linear texture decode bridge now bakes alpha glossiness into a
  generated linear `roughnessMap` (RGB replicated, G-channel compatible).
- The seventh arbitrary-glTF primitive-policy slice first made point/line
  topology truthfulness structured, and the 2026-06-16 closure wave promoted it
  from skip/reject to renderable fallback geometry in `@vitrum/gltf-adapter`:
  `POINTS`, `LINES`, `LINE_LOOP`, and `LINE_STRIP` now import as
  `fallback-generated-mesh` triangle geometry with source-pathed
  `fallback-generated-primitive-mode` diagnostics. `reject-unsupported` accepts
  those assets; `reject-degraded` still rejects the topology approximation.
- The eighth arbitrary-glTF contract slice landed across `@vitrum/core` and
  `@vitrum/gltf-adapter`: `MaterialSpec.shadingModel?: 'pbr' | 'unlit'` is now
  a first-class contract field and `KHR_materials_unlit` imports to
  `shadingModel: 'unlit'` instead of an adapter warning. Backend compatibility
  ranking now reports unlit assets through the normal material field support
  path: pt-webgl2, pt-webgpu, and walkaround-hybrid are `approximate`.
- The ninth arbitrary-glTF contract slice landed across `@vitrum/core` and
  `@vitrum/gltf-adapter`: `MaterialSpec.thicknessMap` is now a first-class
  texture field, `KHR_materials_volume.thicknessTexture` imports to
  `thicknessMap` instead of being dropped with a warning, and backend
  compatibility ranking reports volume thickness textures through the normal
  material field support path. Current backend grades: pt-webgl2 approximate;
  pt-webgpu and walkaround-hybrid unsupported.
- The tenth arbitrary-glTF downgrade-policy slice landed in
  `@vitrum/gltf-adapter`: assets using archived
  `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture` now receive a
  structured approximate compatibility issue for the unbaked glossiness alpha
  channel. RGB still imports as `specularColorMap`; strict/reject-degraded
  policies now see the lossy roughness conversion instead of relying only on an
  importer warning.
- The eleventh pt-webgpu material slice landed in `@vitrum/pt-webgpu`: the
  packed material flags now encode `MaterialSpec.shadingModel === 'unlit'`, the
  full and lite WGSL decoders expose `mat.isUnlit`, and the shared shade
  prologue terminates unlit hits with base-color radiance before NEE/BSDF/medium
  logic. The promise ledger promotes pt-webgpu `shadingModel` to `approximate`
  because the branch is visible-color correct but does not sample unlit surfaces
  as light sources.
- The twelfth pt-webgl2 material slice landed in `@vitrum/pt-webgl2`: material
  sample 14's flag lane now encodes `MaterialSpec.shadingModel === 'unlit'`,
  `readMaterialInfo()` exposes `material.unlit`, and the composed GLSL main loop
  terminates unlit hits with `surf.color * throughputRgb` after G-buffer capture
  and before env/NEE/BDPT/BSDF logic. The promise ledger promotes pt-webgl2
  `shadingModel` to `approximate` for the same non-light-sampled reason.
- The thirteenth walkaround-hybrid material slice landed:
  `bvh_material` material-flag bit 1 now carries
  `MaterialSpec.shadingModel === 'unlit'`, `decodeIsUnlitMaterial()` exposes it
  to WGSL, and `shade.wgsl` emits base-color radiance directly with zero
  indirect for unlit hits. Bit 0 remains the primitive cast-shadow-disabled
  flag; rough/metal/IOR lanes are unchanged.
- Walkaround-hybrid Phase-3D atlas follow-up (2026-06-14): after the
  `baseColorMap` first slice and `roughnessMap`/`metallicMap` scalar slice,
  `aoMap` now also packs into `pipeline/materialTextureAtlas.ts` as a linear
  R-channel atlas slot with per-map uv0/uv1, wrap, and texture-transform
  metadata. `packBVHRoughMetalFromCore()` stores `aoMapIntensity` in material
  word bits 3-7, `decodeAoMapIntensity()` exposes it to `shade.wgsl`, and shade
  multiplies runtime GTAO by `mix(1, aoMap.r, strength)`. The ledger promotes
  `aoMap`/`aoMapIntensity` to approximate, not native, because finite-emitter
  power, GI validation/promotion evidence, and non-camera light-map/material
  paths remain narrower than shade-owned material evaluation.
- Walkaround-hybrid Phase-3E scalar-specular follow-up (2026-06-14):
  `pipeline/materialTextureAtlas.ts` now appends a per-triangle metadata texel
  carrying `specularColor.rgb` + `specularIntensity`; `materialAtlas.wgsl.ts`
  exposes `sampleSpecularControls()`, and `shade.wgsl.ts` threads those controls
  into direct, analytic, sun, and glossy-indirect GGX evaluation. 2026-06-15
  follow-up: ReSTIR-DI pHat/reuse and GI suffix payloads now consume the same
  specular controls. 2026-06-16 follow-up: ReSTIR-GI producer/reuse `pHat`
  also evaluates receiver specular lobes through the material-aware target
  helper. The promise ledger keeps scalar `specularColor` /
  `specularIntensity` `approximate`, not `native`, until rich-material GI
  A/B promotion evidence lands.
- Walkaround-hybrid Phase-3E scalar-clearcoat follow-up (2026-06-14):
  `pipeline/materialTextureAtlas.ts` appends another per-triangle metadata texel
  carrying scalar `clearcoat` + `clearcoatRoughness`; `materialAtlas.wgsl.ts`
  exposes `sampleClearcoatControls()`, and shade-owned direct, analytic, sun,
  and glossy-indirect paths add a fixed-F0 GGX top-coat lobe. The promise ledger
  promotes scalar `clearcoat` / `clearcoatRoughness` to `approximate`, not
  `native`. 2026-06-15 follow-up: DI pHat/reuse and GI suffix payloads now
  consume clearcoat controls. 2026-06-16 follow-up: GI producer/reuse `pHat`
  evaluates receiver clearcoat lobes; validation remains the promotion boundary.
- Walkaround-hybrid Phase-3E scalar-sheen follow-up (2026-06-14):
  the material atlas now stores scalar `sheen`, `sheenRoughness`, and
  `sheenColor.rgb`; `materialAtlas.wgsl.ts` exposes scalar sheen controls, and
  shade-owned direct, analytic, sun, and glossy-indirect paths add a
  Charlie/Neubelt-Pettineo sheen lobe. The promise ledger promotes scalar
  `sheen` / `sheenColor` / `sheenRoughness` to `approximate`, not `native`.
  2026-06-15 follow-up: DI pHat/reuse and GI suffix payloads now consume sheen
  controls. 2026-06-16 follow-up: GI producer/reuse `pHat` evaluates receiver
  sheen lobes; validation remains the promotion boundary.
- Walkaround-hybrid Phase-3E specular-map follow-up (2026-06-14):
  readable `specularColorMap` and `specularIntensityMap` handles now use the
  material atlas with per-map uv0/uv1, wrap, and texture-transform metadata.
  Shade-owned GGX paths multiply scalar specular color by the sRGB RGB map and
  scalar specular intensity by the linear alpha map. 2026-06-15 follow-up: DI
  pHat/reuse and GI suffix payloads now consume mapped specular controls.
  2026-06-16 follow-up: GI producer/reuse `pHat` evaluates receiver specular
  lobes. The promise ledger keeps both rows `approximate`, not `native`, because
  rich-material GI/material-furnace A/Bs are still pending.
- Walkaround-hybrid Phase-3E clearcoat/sheen map follow-up (2026-06-14):
  readable `clearcoatMap`, `clearcoatRoughnessMap`, `sheenColorMap`, and
  `sheenRoughnessMap` handles now use the material atlas with per-map uv0/uv1,
  wrap, and texture-transform metadata. Shade-owned top-coat and Charlie sheen
  paths multiply scalar controls by glTF R/G/RGB/A channels respectively.
  2026-06-15 follow-up: DI pHat/reuse and GI suffix payloads now consume these
  mapped controls. 2026-06-16 follow-up: GI producer/reuse `pHat` evaluates
  receiver clearcoat/sheen lobes. The promise ledger keeps these rows
  `approximate`, not `native`, until rich-material GI A/B promotion evidence
  lands.
- Walkaround-hybrid Phase-3E clearcoat-normal follow-up (2026-06-14):
  readable `clearcoatNormalMap` handles now use the same material atlas and
  authored-tangent-aware normal-map path as base normal maps, with
  `clearcoatNormalScale` stored in per-triangle metadata. Shade-owned top-coat
  direct, analytic, sun, and glossy-indirect paths receive a separate clearcoat
  normal. 2026-06-15 follow-up: DI pHat/reuse and GI suffix payloads now carry
  clearcoat-normal-aware material payloads. 2026-06-16 follow-up: GI
  producer/reuse receiver-lobe `pHat` consumes the same payload path. The promise
  ledger keeps `clearcoatNormalMap` / `clearcoatNormalScale` `approximate`, not
  `native`, until validation/promotion evidence lands.
- Walkaround-hybrid Phase-3E bump-map follow-up (2026-06-15): readable
  `bumpMap` handles now use the material atlas as linear height fields with
  per-map uv0/uv1, wrap, and texture-transform metadata. `bumpScale` is stored
  in per-triangle metadata, and `materialAtlas.wgsl.ts` finite-differences the
  sampled height field into a visible shade-normal perturbation after normal-map
  application. 2026-06-15 follow-up: DI pHat/reuse and GI suffix payloads now
  use the mapped visible normal. 2026-06-16 follow-up: GI producer/reuse
  receiver-lobe `pHat` consumes the same visible-normal path. The promise ledger
  keeps `bumpMap` / `bumpScale` `approximate`, not `native`, until
  validation/promotion evidence lands.
- Walkaround-hybrid Phase-3E anisotropy follow-up (2026-06-14): scalar
  `anisotropy` / `anisotropyRotation` and readable `anisotropyMap` handles now
  ride the material atlas. The shader samples glTF B-channel strength and RG
  direction, then uses a guarded anisotropic GGX branch for shade-owned direct,
  analytic, sun, and glossy-indirect paths. 2026-06-15 follow-up: DI pHat/reuse
  and GI suffix payloads consume anisotropy. 2026-06-16 follow-up: GI
  producer/reuse receiver-lobe `pHat` consumes the anisotropy payload path. The
  promise ledger keeps these rows `approximate`, not `native`, because
  validation is still pending.
- Walkaround-hybrid Phase-3E iridescence follow-up (2026-06-14): scalar
  `iridescence` / `iridescenceIor` / `iridescenceThicknessRange` and readable
  `iridescenceMap` / `iridescenceThicknessMap` handles now ride the material
  atlas. The shader samples glTF red-channel iridescence strength and
  green-channel thickness, then modifies shade-owned GGX F0 with the same
  Belcour/Barla-style thin-film helper shape used by pt-webgpu. 2026-06-15
  follow-up: DI pHat/reuse and GI suffix payloads consume the iridescence
  approximation. 2026-06-16 follow-up: GI producer/reuse receiver-lobe `pHat`
  consumes the iridescence payload path. The promise ledger keeps these rows
  `approximate`, not `native`, because validation is still pending.
- The fourteenth glTF extension-policy slice landed: `extensionsRequired` now
  accepts `KHR_materials_unlit` and archived
  `KHR_materials_pbrSpecularGlossiness` when the importer can represent them,
  and scalar-only spec-gloss conversion now emits a structured approximate
  compatibility issue instead of only an importer warning.
- The fifteenth pt-webgpu material-lobe slice landed: full-tier BSDF-side
  area/environment connections, lite direct/environment paths, lite BSDF-env
  connection, SPPM receiver gathers, MNEE caustic receivers, BDPT connection
  endpoint evals/PDF overrides, and the ReSTIR-PT producer suffix-Lo estimator
  now use `evaluateBrdfFull` / `brdfDirectionalPdfFull` where the needed
  material fields are locally available. Follow-up sampler/PDF waves closed
  ReSTIR-PT visible/suffix material-map parity, clearcoat/sheen source sampling,
  ReSTIR-PT clearcoat-normal reservoir/resolve/source-PDF parity, and BDPT
  surface light-subpath scatter PDFs with hit-local material payloads. The BDPT
  payload now also carries the front-face side bit needed for layer selection and
  applies layer tint/roughness, thin-film reflect tint, Cauchy IOR, and spectral
  reflectance scalar. Remaining schema work is now concentrated in inverse/adjoint
  gradients and material-lobe reference/furnace A/B evidence.
- The sixteenth arbitrary-glTF loader/API slice landed in `@vitrum/gltf-adapter`:
  high-level URL/resource loading now throws typed `GltfFetchFailed` /
  `GltfResourceNotFound` errors with `{ url, kind }`, `LoadGltfAssetOptions.cache`
  wraps resolved asset/buffer/image byte fetches, and `textureDecodeReport`
  is exposed on both `GltfAssetResult` and `GltfForEngineResult`. The report
  walks the converted scene's material `TextureRef`s and classifies field,
  source path, UV set, transform presence, handle kind, and backend readiness
  for pt-webgl2 / pt-webgpu / walkaround-hybrid. `loadGltfAndDecodeTextures()`
  is exported as the explicit decode/report entry point while keeping backend
  atlas/upload work in the PT and walkaround phases.
  Follow-up closure (2026-06-13): `decodeSceneTextures(scene, { target:
  'cpu-linear' | 'webgpu', decodePixels })` is now exported from
  `@vitrum/gltf-adapter`. The CPU-linear target converts raw-image handles to
  RGBA `Float32Array` linear pixel payloads using the same per-field sRGB/data
  policy as the report, preserves alpha, returns a refreshed report, and warns
  with source paths when a host decoder is missing. It also returns structured
  diagnostics for missing decoders, unsupported handles, max-size hazards, and
  NPOT-repeat hazards so hosts do not have to parse warning strings. The WebGPU
  target also resolves raw-image handles through the same decode hook, but
  preserves backend-upload color space for each field instead of forcing
  CPU-linear values; the refreshed report now distinguishes the material role
  `colorSpace` from the decoded handle's `handleColorSpace` when a hint is
  available. Built-in image transcoders, automatic downsampling/mip generation,
  and remaining backend map consumption are still tracked outside this API
  bridge.
  Follow-up closure (2026-06-14): `loadGltfAndDecodeTextures()` now calls
  `decodeSceneTextures()` directly when a host `decodePixels` hook is supplied
  and returns `decodedTextureCount`, `unchangedTextureCount`,
  `textureDecodeDiagnostics`, `textureDecodeWarnings`, and the refreshed
  `textureDecodeReport`; it is no longer a report-only alias for
  `loadGltfAsset()`.
- The seventeenth arbitrary-glTF geometry slice landed in `@vitrum/gltf-adapter`:
  normal/bump/clearcoat-normal mapped primitives that omit authored `TANGENT`
  now synthesize per-vertex xyzw tangents from POSITION/NORMAL/TEXCOORD_0 during
  import. Authored tangents are preserved unchanged, and missing-UV/degenerate
  cases emit source-pathed `GltfImportDiagnostic` rows (plus legacy warning text)
  instead of silently pretending the tangent-space basis exists.
- The eighteenth pt-webgpu ReSTIR-PT material-lobe slice landed:
  `ReservoirPTHero` widened from 36 to 48 u32 words and now serializes the
  visible vertex's scalar clearcoat/sheen/iridescence fields plus anisotropy
  state. Producer, temporal, spatial, finalise, and resolve all route p-hat /
  reconstruction through the same full-lobe visible-domain helper, and resolve
  now uses `evaluateBrdfFull` instead of the base helper. The producer samples
  anisotropic visible-vertex specular directions and computes the matching
  anisotropic base source PDF. Later source-sampling follow-ups added the
  normalized base/clearcoat/sheen lobe mixture and matching `pdfSrc`. Focused
  reservoir-layout and ReSTIR-PT contract tests pin the layout, field
  serialization, domain-copy helper, full-lobe target/resolve, and anisotropic
  producer path; the WGSL shader gate compiles all four ReSTIR-PT passes.
- The nineteenth/twentieth pt-webgpu material-texture slices landed: full-tier
  material descriptors now extend to 67 vec4s and pack clearcoat factor/
  roughness/normal, sheen color/roughness, iridescence factor/thickness, and
  specular color/intensity maps. `material.wgsl.ts` samples the glTF channel
  conventions from the correct sRGB/linear texture arrays with per-map texCoord,
  KHR_texture_transform, wrap, and UV-fit metadata, and the shade prologue
  modulates decoded lobe parameters before downstream BSDF/PDF/NEE calls. The
  main megakernel now threads a sampled `clearcoatNormalMap` through clearcoat
  BRDF/PDF/source-sampler paths. Follow-up 2026-06-13: pt-webgpu now consumes
  authored/generated tangent.xyzw and handedness for those tangent-space maps.
  The promise ledger still promotes the extension map rows to `approximate`,
  not `native`, because inverse/adjoint gradients target the base parameterization
  and material-lobe reference A/B is still pending. BDPT light-side
  clearcoat-normal/layer/thin-film/spectral parity is now structurally
  shade-prologue-equivalent and shader-gated, but not independently
  furnace/reference-promoted.
  Verification: focused pt-webgpu material/WGSL/reuse/lite suites, full
  typecheck, shader gate, and WSL GPU T1 smoke.
- The walkaround-hybrid mutation-matrix seam gained focused non-GPU coverage:
  `packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts` pins
  transform refit, material refresh, emitter repack/GI invalidation,
  procedural-sky environment fallback warnings, `updateLighting()` warning and
  DDGI invalidation behavior, and `setSize()` resize/no-op behavior using
  fake pipeline/DDGI/RC collaborators.
- The RC exported-surface/lifecycle residue landed in `@vitrum/walkaround-rc`:
  raw `RCDispatcher.dispatchFrameRaw()` now snapshots binding-relevant inputs
  and auto-invalidates cached bind groups when direct callers change `bvhMode`,
  buffer sets, env bindings, device, cascade output buffers, or cascade bounds.
  `cascadeDispatchInvalidation.test.ts` pins stable-frame reuse plus TLAS/bounds
  rebuilds, and the stale RC mapping/README comments now match the current
  no-`/three` raw package surface.
- The H37 RC glass-visibility residual landed in `@vitrum/walkaround-rc`:
  rect-area emitter NEE and point/spot fixture direct-light shadows now use
  `rcTraceAny(..., skipGlass=true)` instead of closest-hit occlusion, so
  transmissive geometry no longer fully blocks coarse RC direct light. The
  merged-mode RC upload path now packs canonical `bvhIndex.w` payloads from core
  materials, so the `trans4` glass filter works outside TLAS mode too. The
  focused `rcLightEvalWgsl.test.ts` gate pins the direct-light WGSL call sites,
  and `rcMergedRefit.test.ts` pins merged-mode glass payload packing.
- Verification after the previous sweep: root `npm run typecheck` clean and
  root `npm test` clean (`150` files, `1551` passing, `3` skipped). Verification
  after the current follow-up is in progress; focused typecheck/test runs are
  listed in the handoff/final response for this work session.

Not fully closed yet:

- GPU/reference-render A/B is still pending for the render-changing paths:
  WebGL2 tangent-space normal/bump maps and pt-webgpu SPPM photon-map scenes.
- GLTF-06 adapter-side texture mapping is now mechanically pinned by
  `gltfTextureSweep.test.ts`. The remaining GLTF-06/material-parity work is
  backend consumption fidelity (`GLTF-API-06` / `CAP-01` / `PTWG-MAT-01`), not
  low-level adapter map plumbing.
- `loadGltfForEngine()` closes the adapter-owned engine-preparation path. A
  future `@vitrum/engine/gltf` subpath is a package-boundary/product decision,
  not a required code path for the one-call adapter API.
- A future lite-tier implementation could bake transformed/instanced scenes into
  a lite-consumed world-space BVH, but the current professional contract is now
  honest: those paths are not advertised as supported on lite.
- GitNexus remains unavailable in this desktop/UNC path; impact review for this
  wave used direct source reads, call-chain inspection, package typechecks, and
  focused Vitest runs under Linux Node instead.

### W-HYB-01 - NRC slot claims are never cleared

Status:
- Closed/stale. `WalkaroundGPUPipeline` now supplies
  `nrcClearSlotClaims: (encoder) => this._nrc!.clearSlotClaims(encoder)` into
  registered pass resources, and `NRCSubsystem.clearSlotClaims()` clears the
  claim buffer each active frame.

Closure:
- Keep an order/recording test on the NRC path if this is promoted from source
  audit to formal gate coverage.

### W-HYB-02 - Default atrous denoisers reuse one UBO across encoded dispatches

Status:
- Closed/stale. `buildAtrousBindGroup()` now packs each iteration into an
  aligned byte range and binds `{ buffer, offset, size }`; `AtrousDenoiser`,
  `AtrousIndirectPass`, and `AtrousVarianceDenoiser` pass per-iteration offsets
  instead of rebinding the same zero-offset UBO range for every encoded dispatch.

Closure:
- Keep focused bind-group offset tests for direct, indirect, and variance
  chains; no further algorithmic change is required for the original UBO-race
  finding.

### W-HYB-03 - Hybrid async init/DDGI runtime failures bypass `onError`

Status: CODE CLOSED for the cited async-init and DDGI runtime error-routing
paths. Keep this section as historical evidence; do not re-open without a new
source-backed failure mode.

Evidence:
- `HybridEngineLifecycle.startInit()` is fire-and-forget and its catch path sets
  state plus `console.error`.
- `HybridEngine.onError` is currently wired mainly for device/uncaptured errors.
- DDGI frame errors are console-only or detached.

Closure:
- Thread an error reporter through lifecycle and frame/DDGI dependencies. Done:
  async init reports fatal `EngineError`; DDGI init/BVH/runFrame failures report
  non-fatal `EngineError`.
- Emit fatal and nonfatal `EngineError` records through the same engine error
  channel. Done via `HybridEngine.onError`.
- Test fake async init failure and fake DDGI run-frame failure. Done in focused
  engine/DDGI suites.

### PTWG-01 - pt-webgpu device-lost/error state is reported but not enforced

Evidence:
- `device.lost` sets pt-webgpu state to `error`.
- `#assertLive()` blocks disposed/no-scene, but not the `error` state.
- `renderFrame()` and `resume()` can proceed after fatal device loss.

Closure:
- Block render, mutation, seeding, inverse queries, and resume while in fatal
  `error`.
- Add device-lost/state-machine tests.

### PTWG-02 - Emissive-to-zero material mutation can leave stale implicit emitters

Evidence:
- Implicit mesh emitters are synthesized from emissive material fields.
- The material fast path detects emissive-field changes.
- The emitter repack guard checks only whether the next scene still has an
  implicit mesh-area emitter, so emissive -> black can leave the old emitter.

Closure:
- Repack emitter arrays/light tree when old OR new scene has an implicit/explicit
  mesh-area emitter, especially on any emissive-field change.
- Test emissive mesh -> zero emissive removes NEE/light-tree contribution.

### PTWG-03 - CLOSED 2026-06-12 - SPPM photon emission source normalization

Verified closure:
- `sppmBindings.wgsl.ts` computes `lightSelectInvPdf =
  f32(availableLightCount)` and applies it to directional, point, spot,
  rect/disc, mesh-area, and environment photon flux.
- The photon source list now follows the same packed source families as NEE:
  N-directional records, point/spot records with penumbra/stride parity,
  rect/disc records, mesh-area triangle records, and environment helpers
  (`sampleEnvironmentImportance`, `sampleEnvironmentColor`, `environmentPdf`).
- `sppmPhotonEmission.test.ts` pins one-light/two-light expected-flux equality,
  source-family coverage, RGB N-directional records, spot penumbra, area-source
  conventions, and environment-pdf compensation.
- `sppmHashGrid.test.ts` retains the progressive SPPM structural/oracle pins for
  the production hash-grid gather path.

Residual:
- Rendered caustic reference promotion remains validation evidence, not an open
  source implementation gap.

### PTWG-04 - CLOSED 2026-06-12 - SPPM per-pixel stats update once per frame

Verified closure:
- `kernel.wgsl.ts` declares `var sppmGatherUpdated = false` before the bounce
  loop and gates `photonMapContribution()` behind
  `if (!sppmGatherUpdated && sppmReceiverEligible)`, setting the flag
  immediately after the first eligible diffuse-ish gather surface.
- `sppmPhotonEmission.test.ts` asserts the guard, the eligibility expression, the
  flag write, and exactly one `photonMapContribution(` call site in the kernel.

Residual:
- Broader GPU A/Bs for long-running photon-map convergence remain validation
  evidence, not an open source implementation gap.

### PTWG-05 - Spectral min-mu and dispersion Abbe share one packed lane

Evidence:
- `materialPacking.ts` stores `dispersionAbbe > 0 ? dispersionAbbe : spectralMinMu`
  in one lane.
- WGSL decodes that lane as `dispersionAbbe`, which can falsely trigger Cauchy
  dispersion for spectral materials.

Closure:
- Split spectral min-mu and dispersion Abbe into separate lanes or add an
  explicit flag.
- Add a spectral-without-dispersion test.

### CORE-01 - CUBICSPLINE quaternion outputs are not normalized

Evidence:
- `sampleAnimationChannel()` Hermite-interpolates CUBICSPLINE values
  component-wise.
- Only LINEAR rotations route through `slerpQuat`.

Closure:
- Normalize rotation outputs for CUBICSPLINE and STEP/knot paths where needed.
- Add CUBICSPLINE quaternion tests with non-unit intermediate output.

### GLTF-01 - glTF skinned bind matrices are dropped

Evidence:
- Core `SkinnedMeshPrimitive` supports `bindMatrix` and `bindMatrixInverse`.
- `solveSkin()` applies them.
- `gltf-adapter` constructs skinned primitives without passing them through.

Closure:
- Import/preserve the skinned node bind transform and inverse.
- Add a non-identity skinned-node glTF fixture.

### WEBGL2-01 - pt-webgl2 tangent handedness is wrong and authored tangents are ignored

Evidence:
- Core and glTF adapter carry tangent data.
- pt-webgl2 derives tangent attributes from positions/UVs and writes `w = 0`.
- GLSL multiplies bitangent by `tangentSample.w`, which collapses the bitangent.

Closure:
- Consume authored tangent XYZW when present.
- For generated tangents, compute real handedness or default to `+1`.
- Guard shaders against zero handedness.
- Add mirrored-UV and authored-tangent normal-map tests.

## P1 capability honesty and mutation consistency

These make the library trustworthy to integrate. Some are not "current contract
bugs" because the current contract lacks a warning channel, but they remain
professional-library gaps.

### PTWG-06 - pt-webgpu lite tier overclaims transform/topology capability

Evidence:
- Lite intersection starts traversal at BVH root `0u` and has no TLAS group.
- Earlier lite static scenes still used TLAS-oriented packing, so multi-primitive
  or transformed scenes could be advertised while only BLAS root 0 was consumed.
- Transform/instance fast paths update TLAS data that lite traversal does not
  consume, so those incremental patch rows must stay unsupported.

Closure:
- Static ingestion closed by the 2026-06-14 implementation wave: lite
  `setScene()` routes through `buildPackedScene(..., { geometryMode:'merged' })`,
  which uses `mergeWorldSpaceFromCore()` to bake mesh/skinned/instanced
  primitives, including non-identity transforms, into a single world-space BLAS
  rooted at node 0. `scenePack.test.ts` pins multi-mesh and instanced expansion;
  `liteTierCapabilities.test.ts` pins `instanced-mesh` as native for static lite
  scenes.
- Mutation honesty remains intentional: material patches are now
  `fallback-rebuild` on lite because merged material slots are deduped, while
  transform/topology patch rows stay `unsupported` and `SceneMutationRouter`
  still rejects those lite fast paths before they update TLAS-only buffers. A
  future enhancement could rebuild and upload the merged lite BLAS on
  transform/instance patches, but that is no longer required for static
  arbitrary multi-primitive ingestion.

### PTWG-07 - pt-webgpu lite emitter/environment mutations leave texture path stale

Evidence:
- Full `setScene()` refreshes lite light/env textures and CDF textures.
- `updateEmitter()` and same-sized `updateEnvironment()` update storage buffers
  and light tree, while lite shaders read the texture path.

Closure:
- Closed in the 2026-06-12/13 implementation wave: lite emitter and
  environment mutations now regenerate/upload the sampled light/environment
  textures that `kernelLite.wgsl` reads, with focused tests covering the refresh
  path.

### ENGINE-01 - Nonfatal diagnostics are not consistently programmatic

Evidence:
- Core exposes `onError` but no `onWarning`.
- Backends frequently use `console.warn` for contract-affecting degradations:
  unsupported scene features, ignored denoisers, invalid HDRI payloads, texture
  packing warnings, empty skin warnings, and mutation fallbacks.

Closure:
- Closed in the 2026-06-11 warning-channel wave by adding
  `EngineWarning`, `EngineOptions.onWarning`, and optional `Engine.onWarning`.
- createEngine now reports structured warnings for fallback, TLAS-backend
  recommendations, ignored ownership keys, and cross-backend advanced-option
  application.
- pt-webgpu, pt-webgl2, and walkaround-hybrid now preserve existing
  `console.warn` output while also emitting structured warnings for the
  audited contract-affecting construction, scene, mutation, and viewport paths.
- The pt-webgl2 scene upload packers now route previously packer-local
  unreadable texture/HDRI and ambiguous texture-stride diagnostics through the
  engine warning channel as `pt-webgl2.texture-unreadable`,
  `pt-webgl2.hdri-unreadable`, and
  `pt-webgl2.texture-ambiguous-pixel-stride` during both `setScene()` and the
  `updateEnvironment()` fast path.
- 2026-06-16 follow-up: walkaround-hybrid material-atlas drops for unreadable
  CPU texture handles now surface through the same programmatic warning channel
  as `walkaround-hybrid.unreadable-material-texture-map` during initial
  `setScene()` BVH publication and `updatePrimitive({ material })` rebuilds,
  with material slot, field, color-space role, and `fallback:"map ignored"`
  details. The packer keeps a diagnostic payload instead of relying on a local
  console-only warning.
- Internal debug/resource chatter remains console-only by design; it is not a
  contract-affecting degradation.

### CAP-01 - CLOSED 2026-06-13 - Per-field material support matrix is explicit

Status:
- Source-verified closed by `BACKEND_PROMISE_LEDGER` material rows plus
  `packages/core/src/__tests__/engineContract.test.ts`,
  `packages/pt-webgl2/src/__tests__/engineContract.test.ts`, and
  `packages/pt-webgpu/src/__tests__/liteTierCapabilities.test.ts`.
- Every `MaterialSpec` field is now classified per backend as implemented,
  approximated, ignored-with-warning, or unsupported. Backend fidelity gaps
  such as displacement, receiveShadow, and specialty map consumption remain
  tracked by their specific implementation rows instead of this matrix gate.

Closure evidence:
- The material matrix is exhaustive over `MATERIAL_SPEC_FIELDS`.
- pt-webgl2 and pt-webgpu contract tests cover unsupported-field diagnostics
  and supported-row capability promises.

## P2 backend-wide contract completion

These are public contract fields or backend parity promises that still need
implementation or explicit downgrade.

### MAT-01 - Authored tangent stream consumption must be backend-wide

Closure:
- pt-webgl2: consume tangents as part of WEBGL2-01.
- pt-webgpu: closed 2026-06-13. `shared-bvh` packs tangent.xyzw, pt-webgpu
  uploads it as a group-3 storage buffer, and `buildShadingTangentFrame`
  interpolates handedness before falling back to derived frames for tangentless
  scenes.
- walkaround: decide whether tangent-space maps are part of supported material
  fidelity, then implement or diagnose.

### MAT-02 - Displacement maps are accepted but not rendered

Evidence:
- Core exposes `displacementMap`, `displacementScale`, and `displacementBias` as
  reserved.
- No backend rendering path consumes them meaningfully.

Closure:
- Closed as a professional honesty downgrade in the 2026-06-11 displacement
  wave: `BackendSupportDetails.materials` now has explicit unsupported rows for
  `displacementMap`, `displacementScale`, and `displacementBias` on
  walkaround-hybrid, pt-webgl2, and pt-webgpu.
- pt-webgpu and pt-webgl2 emit structured
  `*.unsupported-displacement-material` warnings when scenes or material patches
  submit displacement fields. walkaround-hybrid already warns for unconsumed
  material fields; the allowlist test now pins displacement as unconsumed.
- Remaining optional future work: implement real displacement through an
  authored-geometry, tessellation, or parallax strategy, then promote these rows
  from `unsupported` with renderer A/B evidence.

### SHADOW-01 - Shadow flags are incomplete across backends

Evidence:
- Core has `castShadow` and reserved `receiveShadow`.
- pt-webgl2 consumes `castShadow`; walkaround/pt-webgpu handling is not
  backend-wide.
- Emitter `castShadow` is not uniformly represented.

Closure (CLOSED 2026-06-11 — implemented + honestly downgraded; rows in
`BackendSupportDetails.shadows` / promiseLedger.ts, pinned by
engineContract.test.ts):
- **Primitive castShadow** —
  - pt-webgl2 `native` (material castShadow lane via shared-bvh
    `splitMaterialsByCastShadow` + the integrator's shadow-ray gate).
  - pt-webgpu `native`: material vec4 #25 .w castShadowDisabled lane
    (materialPacking.ts), skipped by `triShadowCastDisabled` in EVERY any-hit
    (occlusion) traversal on BOTH tiers (intersectionCore.wgsl.ts traceMeshBvh
    `!closest` path — NEE shadow rays, BSDF-MIS connections, ReSTIR-PT
    reconnection, MNEE legs); closest-hit camera/radiance rays unaffected.
    Contract `AnalyticPrimitive` has no castShadow field → analytic shapes
    always occlude.
  - walkaround-hybrid `native` (2026-06-13 follow-up): DI visibility,
    ReSTIR-GI reservoir visibility, GRIS reconnection visibility, DDGI probe
    direct-light visibility, and RC probe direct-light visibility all skip
    `castShadow:false` geometry. Main DI/GRIS/ReSTIR-GI paths read
    `bvh_material` bit 0 through `traceSceneAnyCastMask`; DDGI/RC read shared
    `MaterialEntry.flags` bit 1 through predicate-backed shared-BVH traversal.
- **Emitter castShadow** —
  - pt-webgl2 `approximate`: lights-texture s5.g lane consumed by
    directLightContribution for all analytic NEE lights
    (rect/disc/spot/point/directional), and uMeshLights s5.g is consumed by
    mesh-area triangle-light NEE. BDPT now stores emitter-endpoint
    `castShadowDisabled` in `uBdptLightPathTex` row 3 and skips connection
    visibility for direct connections to that endpoint. The forward
    emissive-hit residual remains geometry-visible rather than
    shadow-flag-skipped.
  - pt-webgpu `approximate`: per-light lanes (directional sign-encoded
    angularDiameter; point/spot extra .z; rect/disc center .w; mesh-area
    radiance .w) consumed by the default kernel/kernelLite NEE loops + the
    connect.wgsl BSDF-MIS area connections. ReSTIR-PT suffix direct lighting
    now also consumes the point/spot/rect/disc/mesh packed lanes and the packed
    N-directional records. Lite directional NEE decodes the signed
    `cameraPos.w` mirror for the first directional `castShadow:false` flag.
    Remaining promotion proof: BDPT light subpath and MNEE/SPPM caustic
    legs/source treatment still need explicit shadow-flag oracles.
  - walkaround-hybrid `native` (2026-06-13 follow-up): analytic point/spot
    payloads pack binding-13 lane `[13]`; shared `EmitterTri` packs lane `[19]`;
    ReSTIR-DI candidate visibility + shade visibility gate on
    `e.castShadowDisabled`; DDGI/RC area-emitter NEE skip the emitter shadow ray;
    DDGI and RC fixture/sun lights carry castShadow-disabled flags; main direct
    sun gates its visibility ray from the scene directional emitter flag.
- **receiveShadow** — `unsupported` on ALL THREE backends (a "receiver ignores
  occlusion" toggle is non-physical for a GI path tracer; kept @reserved).
  Structured `*.reserved-receive-shadow` warnings fire on pt-webgpu, pt-webgl2,
  and walkaround-hybrid when a scene sets `receiveShadow: false`.
- Tests: packer-lane byte tests (pt-webgpu scenePack.materials/emitters,
  pt-webgl2 lightsTexture, walkaround roughMetalPacking bit 0), shared-bvh
  masked-traversal derivation pins (bvhCastShadowMask.test.ts), WGSL SHA
  re-pins (intended; default lanes pack 0.0 → flag-less scenes behaviorally
  identical), ledger exhaustiveness pin in engineContract.test.ts.
- Remaining optional future work: pt-webgpu directional/off-default-integrator
  coverage and pt-webgl2 forward-emissive residual policy/coverage — promote
  rows with renderer A/B evidence when implemented.

### WEBGL2-02 - CLOSED 2026-06-14 - pt-webgl2 procedural sky is approximate, not unsupported

Verified closure:
- `packages/pt-webgl2/src/scene/equirectHdrInfo.ts` imports
  `bakePreethamSkyEquirect()` and converts `env.kind === 'procedural-sky'`
  into the existing equirect HDRI/CDF path.
- `packages/pt-webgl2/src/capabilities.ts` lists `procedural-sky` in
  `PT_WEBGL2_SUPPORT.supportedEnvironmentKinds`.
- `packages/core/src/engine/promiseLedger.ts` grades pt-webgl2
  `supportDetails.environments['procedural-sky']` as `approximate`.
- `packages/pt-webgl2/src/scene/equirectHdrInfo.test.ts` pins the procedural
  bake path, zero-intensity black output, and sun-direction maximum.

Residual:
- The backend uses a finite baked Preetham equirect atlas rather than analytic
  sky evaluation in the tracing shader, so the correct ledger grade remains
  `approximate` until/unless an analytic shader path lands.

### WEBGL2-03 - pt-webgl2 denoiser option honesty + OIDN final-pass — CODE CLOSED

Evidence:
- pt-webgl2 now implements `denoiser: 'oidn-final'` as a real async final-pass
  path. It reads the linear HDR accumulator plus full-tier MRT albedo/normal aux
  through `GlResources.readOidnInputsRgba32f()`, kicks the shared
  `OIDNDispatcherCore` once the PT accumulation reaches the requested SPP target,
  exposes `getLatestDenoised()`, reports `FrameStats.denoiserState`, invalidates
  on reset/scene changes, and releases the OIDN cache on dispose.
- `BackendSupportDetails.denoisers` is now first-class and exhaustive over
  every `EngineOptions.denoiser` mode. The pt-webgl2 ledger/capability row
  reports `none:native`, `oidn-final:native`, and the realtime/incompatible
  denoiser modes (`atrous`, `atrous-variance`, `svgf-real`, `bmfr`, `neural`) as
  `unsupported`.
- `createPTEngine_WebGL2` still emits the structured
  `pt-webgl2.unsupported-denoiser` warning for unsupported denoiser modes, but
  `oidn-final` requires host `oidn: { modelUrl }` config instead of degrading.

Closure:
- Closed by first-class capability/ledger reporting for unsupported realtime
  denoisers plus an implemented WebGL2 OIDN final-pass bridge.
- Tests: `packages/core/src/__tests__/engineContract.test.ts` pins exhaustive
  denoiser rows across backends; `packages/core/src/__tests__/ledgerVsCapabilities.test.ts`
  pins ledger/capability alignment; `packages/pt-webgl2/src/__tests__/engineContract.test.ts`
  pins the runtime pt-webgl2 capability matrix and structured warning behavior;
  `packages/pt-webgl2/src/__tests__/oidnFinal.test.ts` and
  `packages/pt-webgl2/src/denoise/rgba32fReadback.test.ts` pin the OIDN kick,
  telemetry, cache, invalidation, and WebGL readback conversion behavior.

### WEBGL2-04 - pt-webgl2 material texture edge cases remain

Evidence:
- glTF combined metallic-roughness texture parity is code-closed:
  `gltf-adapter/src/materials.ts` assigns the same `TextureRef` to both
  `roughnessMap` and `metallicMap`, `featureReport.ts` reports both fields from
  `metallicRoughnessTexture`, and pt-webgl2 samples G for roughness / B for
  metallic in `get_surface_record_function.glsl.js`. Tests:
  `gltfTextureSweep.test.ts` and `untestedMaterialMaps.test.ts`.
- `alphaMap` transform parity is now code-closed: `MATERIAL_PIXELS` is 111,
  `materialsTexture.ts` packs `alphaMapTransform` at texels 93/94,
  `material_struct.glsl.js` decodes it, and both
  `get_surface_record_function.glsl.js` and `attenuate_hit_function.glsl.js`
  sample alpha maps through the transform. The promise ledger now grades
  pt-webgl2 `alphaMap` as `native` instead of the stale transform-caveat
  `approximate` row. Tests:
  `materialsTexture.test.ts`, `untestedMaterialMaps.test.ts`, and
  `materialStrideParity.test.ts`.
- glTF sampler wrap parity is now code-closed for pt-webgl2: the material
  record packs per-map `wrapS`/`wrapT` at texels 100..110 (after
  alphaMapTransform at 93/94, anisotropyMapTransform at 95/96, thickness
  payload at 97, and thicknessMapTransform at 98/99),
  `material_struct` decodes repeat/clamp/mirrored-repeat pairs, and both
  surface and attenuation paths call `sampleMaterialTexture(...)` so every
  material texture fetch applies manual per-layer wrapping instead of relying
  on one global WebGL array sampler.
  `untestedMaterialMaps.test.ts`, `materialsTexture.test.ts`, and
  `shader-gate` pin the path.
- Layered front/back material fields are now native field-consumption rows for
  pt-webgl2 and full pt-webgpu: both backends pack and shade face-selected
  transmission/roughness plus nested normal-map/normal-scale payloads
  (`frontLayer.normalMap`, `frontLayer.normalScale`, `backLayer.normalMap`,
  `backLayer.normalScale`). pt-webgl2 pins the append-only material-stride
  payload, atlas collection, GLSL decode, surface shading, and attenuation
  paths; pt-webgpu pins full-tier descriptor lanes, shader selection, and lite
  tier structured warnings. Renderer-row promotion still waits on runtime A/B,
  and walkaround layered/spectral/volume families remain approximate or routed
  to PT.
- Surface anisotropy support is now source-verified for pt-webgl2:
  `promiseLedger.ts` marks `anisotropy`, `anisotropyRotation`, and
  `anisotropyMap` native after reserved-lane pack/decode, atlas/UV/wrap
  payload wiring, KHR RG/B map sampling, and anisotropic GGX sampling/eval/PDF
  consumption.
- KHR volume thickness-map support is now source-verified for pt-webgl2:
  `promiseLedger.ts` marks `thicknessMap` approximate after the material
  packer writes scalar `thickness`, atlas layer, UV bit, transform, and wrap
  payloads; the surface and attenuation GLSL paths sample the G channel and
  clamp Beer-Lambert path length. The approximation is explicit because this is
  closed-surface attenuation, not exact thin-shell volume integration.

Closure:
- WEBGL2-04 is closed as an honesty row. Layered face-normal support is now
  code-closed for pt-webgl2 and full pt-webgpu; remaining work is visual
  promotion/proof, not a silent field drop.

### PTWG-08 - pt-webgpu material and texture infrastructure is partial

Evidence:
- Historical v1 material texture array gaps have been narrowed: pt-webgpu now
  carries per-layer UV fit, per-map UV metadata/wrap modes, sRGB vs linear
  arrays, raw-data validation, generated mip chains, and explicit geometric LOD
  selection in the compute shader.
- Remaining supported-map / sampling-path gaps are now the actual risk surface,
  not the old "no UV fit/no mips/no per-map transform" infrastructure hole.
- Several specialty paths still need source-verified full material lobe
  evaluation before the corresponding fidelity rows can be promoted.

Closure:
- Keep per-map texture transforms, color-space handling, mip/LOD behavior, and
  per-layer UV fit pinned by tests as the texture descriptor grows.
- Implement remaining high-value maps or mark them unsupported.
- Extend full-lobe evaluation across all sampling/contribution paths that claim
  material fidelity.

## P3 renderer math and GI validation

These are the areas where source review found real suspicion, but the right fix
requires an independent oracle or reference-render A/B.

### HYB-GI-01 - CLOSED 2026-06-14 - Direct ReSTIR uses selected `xi`

Verified closure:
- `reservoirDi.wgsl.ts` stores the winning candidate `xi` in every reservoir.
- `restirPHat.wgsl.ts` evaluates `restir_di_compute_phat_xi(lid, xi, surf)`,
  sampling finite emitters at the reservoir point instead of the centroid and
  decoding the env sentinel direction from `xi`.
- `ris.wgsl.ts` finalizes both finite-emitter and env winners with
  `restir_di_compute_phat_xi(lid, r.xi, surf)`.
- `temporal.wgsl.ts` and `spatial.wgsl.ts` also call the xi-aware helper when
  re-evaluating neighbour/current reservoirs.
- `shadingTerms.wgsl.ts` `lo_direct()` shades finite-emitter winners with
  `sampleEmitterPoint(e, r.xi)` rather than a fresh random point.
- `oracle.restirDiEstimator.test.ts` keeps the historical centroid/fresh-xi
  characterization and pins the selected-xi regression at ≈1.0 vs brute force.

Residual:
- GPU/reference A/B for the render-changing 2026-06-12 oracle wave is still a
  validation task, but the source-level estimator defect is closed.

### HYB-GI-02 - CLOSED 2026-06-14 - DI RIS support-family `M` accounting

Verified closure:
- `ris.wgsl.ts` tracks `mAreaSupport` and `mEnvSupport` separately.
- Finite-emitter and BSDF-to-emitter proposals increment area support only after
  they enter the reservoir; env proposals increment env support only after they
  enter the reservoir.
- Finalization sets `r.M` and `r.W` using the selected candidate's support
  family (`max(1u, mAreaSupport)` for finite emitters, `max(1u, mEnvSupport)`
  for env), so a single HDRI candidate is no longer divided by the 64+1 finite
  candidate budget.
- The CPU oracle demonstrates that skipped zero/degenerate proposals are not the
  bias source and pins env-only plus mixed scenes at ≈1.0 vs brute force.

Residual:
- Same as HYB-GI-01: render-reference recapture remains validation evidence, not
  an open source implementation gap.

### HYB-DDGI-01 - DDGI no-hit visibility moments may poison visibility

Status:
- Closed mechanically in the 2026-06-11 follow-up sweep.
- Finite hit samples still drive the visibility mean/depth² moments.
- Sky-miss samples (`hitDistance >= DDGI_MISS_DISTANCE`) are skipped so they no
  longer overflow the `rgba16float` visibility atlas.
- All-miss/open-sky visibility texels now write a finite far-visible sentinel
  (`65504.0`, the rgba16f max finite value) instead of zero moments, avoiding
  false zero-depth occlusion on first-frame/open-sky probes.
- `packages/walkaround-hybrid/src/__tests__/oracle.ddgiVisibilityMoments.test.ts`
  now covers all-hit wall occlusion, historical one-miss poisoning, skipped
  one-miss regression, and all-sky open visibility.

Remaining:
- GPU/reference A/B for full DDGI scenes is still a promotion gate, but the
  source-level visibility-moment defect is closed by the CPU oracle.

### HYB-SKY-01 - CLOSED 2026-06-15 - walkaround procedural sky uses the directional Preetham bake

Evidence:
- Core procedural sky includes turbidity, Rayleigh, Mie coefficient, Mie g, and
  sun direction.
- walkaround now calls the shared `bakePreethamSkyEquirect()` helper from
  `resolveHybridEnvironment.ts`, feeds the baked texels through
  `buildDirectionalEnv()`, and returns the same directional map/CDF payload used
  for raw HDRI payloads.
- `HybridEngine.updateEnvironment()` uploads that payload through
  `updateDirectionalEnvironment()` and forwards the env bindings to DDGI probe
  misses when available. Scalar `skyTint`/`skyIrradiance` remain only the
  no-directional fallback average.
- The public grade remains `approximate`, but now for finite 256x128 Preetham
  bake/model limits rather than silent scalar-only loss of turbidity/Rayleigh/Mie
  fields.

Closure:
- Closed by code. Tests: `resolveHybridEnvironment.test.ts` pins baked
  directional data, zero-intensity finite fallback, and no warning for valid
  procedural skies; `mutationMatrix.test.ts` pins runtime update/upload/DDGI
  routing.

### PTWG-BDPT-01 - BDPT needs an independent radiometric oracle - CLOSED 2026-06-15

Evidence:
- `packages/pt-webgpu/src/__tests__/oracle.bdptConnectionCosine.test.ts`
  now has independent rendering-equation oracles for all requested cases:
  finite-area emitter endpoint, one-bounce diffuse light tracing, and a
  non-Lambertian/glossy light-vertex connection. The one-bounce oracle derives
  `Le * area * pi / pdfPick` from the area-position + solid-angle-direction
  sampling density rather than mirroring WGSL assembly.
- That oracle exposed a real finite-area extension bias: the previous
  `prevMatId < 0` emitter branch used `INV_PI`, leaving one-bounce finite-area
  light subpaths at exactly `1/pi` of the first-principles estimator.
- `packages/pt-webgpu/src/wgsl/bdpt/bdptLightSubpath.wgsl.ts` now splits the
  emitter sentinel branch: finite-area emitters (`BDPT_LV_AREA_EMITTER_MATID`)
  use `fPrev = 1.0`, preserving the required `cos/pdfOmega = pi` factor, while
  legacy pseudo emitters keep the old `INV_PI` normalization.
- `packages/pt-webgpu/src/__tests__/bdptGlossyLightSubpath.test.ts` now pins
  the sentinel split alongside the existing real-BSDF light-vertex connection
  assertions.

Closure:
- Focused gate: `npm test --workspace @vitrum/pt-webgpu --
  oracle.bdptConnectionCosine.test.ts bdptGlossyLightSubpath.test.ts
  bdptLightSubpathOracle.test.ts bdptConnectionMisFull.test.ts` passed
  (24 tests).
- Remaining BDPT promotion evidence, such as equal-spp reference A/B scenes, is
  tracked by the broader fidelity rows and does not reopen this oracle gap.

### PTWG-MAT-01 - Extension lobes are missing from some pt-webgpu paths

Evidence:
- Direct NEE uses full BRDF evaluation.
- Full-tier BSDF area/env connection, lite direct/env, lite BSDF-env, SPPM
  receiver gather, MNEE caustic receiver paths, BDPT connection endpoints, and
  ReSTIR-PT producer suffix Lo now use full BRDF/PDF helpers where the needed
  material fields are locally available.
- ReSTIR-PT reservoir payload/target/resolve scalar-lobe parity is now closed:
  `ReservoirPTHero` stores scalar clearcoat/sheen/iridescence/specular fields
  and anisotropy state, temporal/spatial copy the full visible-domain payload,
  p-hat uses `evaluateBrdfFull`, and resolve reconstructs with
  `evaluateBrdfFull`.
- Extension-lobe texture maps now reach the full-tier megakernel shade
  prologue. ReSTIR-PT visible-vertex payload now mirrors that prologue for
  alpha pass-through, baseColor/AO/roughness+metallic/normal/bump/transmission/extension maps,
  layer tint/roughness, thin-film, and spectral albedo before storing the
  reservoir-visible domain. ReSTIR-PT suffix/reconnection vertices now also
  alpha-skip and decode the same hit-local material-map/layer/thin-film/spectral
  domain before Lo evaluation, including mapped normals for reservoir geometry
  and mapped clearcoat normals for clearcoat BRDF/PDF paths. ReSTIR-PT
  clearcoat-normal map parity is now closed by storing `clearcoatNormalV` in
  `ReservoirPTHero` and using the clearcoat-normal-aware target/resolve/source-PDF/
  suffix-Lo helpers.
  ReSTIR-PT producer source sampling now uses a normalized base/clearcoat/sheen
  lobe mixture and stores the matching `pdfSrc` rather than the old base-only
  density. The main eye path now samples the same normalized base/clearcoat/sheen
  mixture through `sampleNextBounceDirection`; MNEE cone-vs-BSDF MIS compares
  against the same sampled density; and BDPT's mapped light-subpath scatter
  records matching `brdfDirectionalPdfFullSampled` forward/reverse densities.
  2026-06-16 follow-up: `tools/radiometric-ab/ab-restir-pt-specialty.mjs` and
  `restirPtSpecialtyReference.test.ts` now pin the deterministic one-sample
  producer/finalize/resolve identity for clearcoat, sheen, iridescence, and
  anisotropy payloads. This closes the static specialty-path proof tail, not
  the V28 GPU/reference-render promotion tail.
  Remaining approximate/schema sites are not simple omissions:
  inverse adjoints use a separate derivative model, while BDPT light-side
  clearcoat-normal/layer/thin-film/spectral parity is now structurally closed by
  sampling `applyClearcoatNormalMap` into the row-4 material payload, storing a
  front-face side bit for layer choice, and applying the same layer/thin-film/
  spectral transforms as the shade prologue before BRDF/PDF evaluation.

Closure:
- Keep the main eye-path `sampleNextBounceDirection` sampled-density regression
  pins green: clearcoat/sheen lobe sampling, direct/connection/MNEE MIS PDFs,
  and BDPT eye-stack forward/reverse densities must stay in lockstep.
- Keep the ReSTIR-PT suffix material-map parity regression pins green; the
  suffix cached-Lo path is code-complete for hit-local maps/layers/thin-film/
  spectral emission and mapped clearcoat normals, and the producer source-sampler/
  PDF limit is closed for the base/clearcoat/sheen mixture.
- Keep the BDPT light-subpath sampling/PDF regression pins green. Row-4 hit-local
  material payloads now cover mapped base/ORM/transmission/normal/bump/
  clearcoat-normal/extension/specular/anisotropy fields plus layer/thin-film/
  spectral transforms; remaining BDPT work is oracle/reference promotion.
- Keep `extensionLobeReference.test.ts` green for lobe-specific CPU proof. It
  closes the clearcoat/sheen/iridescence sampled-density unit tail, including
  the explicit sheen-PDF approximation posture.
- Add material-furnace and reference A/B before promoting these rows from
  approximate/experimental; the CPU lobe oracle is not enough to promote a
  render row by itself.

### PTWG-LITE-01 - CLOSED 2026-06-15 - Lite rect/disc area MIS is paired

Evidence:
- Lite rect/disc NEE now uses a matched power-heuristic estimator, and
  `connectLite` intersects BSDF-sampled directions against the same packed
  `liteLightTex` rect/disc records.
- The old one-sided half-MIS deficit remains pinned as a historical regression
  proof, while the current oracle adds the independently integrated
  BSDF-weighted share and recovers solid-angle ground truth.

Closure:
- Implemented in `kernelLite.wgsl.ts` and `connectLite.wgsl.ts`.
- Guarded by `oracle.liteRectMis.test.ts`, `wgslLiteContract.test.ts`, and
  root `npm run shader-gate` (51 shaders, 28 pipeline gates).

## P4 glTF and asset ingestion completeness

These matter because professional users will judge the library by whether common
production assets survive ingestion. The target is not merely "the low-level
adapter can return a `Scene`." The target is one predictable asset API that can
load a normal glTF/GLB, classify the asset's features, select or reject a backend
honestly, drive animations/morphs/skins through the engine mutation contract, and
surface every degradation as structured data.

### GLTF-API-01 - turnkey asset loading is still host-written

Status:
- `loadGltfAsset(input, options)` now accepts `URL | string | ArrayBuffer |
  GltfJson`.
- URL/string inputs resolve a base URI automatically; explicit `baseUri` is
  supported for object/ArrayBuffer inputs.
- External `.bin` buffers and external image URIs are fetched via host-supplied
  `fetch` or `globalThis.fetch`.
- External image bytes are plumbed into `gltfToScene()` through `imageBytes`;
  embedded GLB images, `data:` URI images, and bufferView images still use the
  existing path.
- Converter-owned import degradations now return structured
  `GltfImportDiagnostic` entries with stable codes and glTF source paths
  alongside the legacy string `warnings` array.
- Tangent generation/failure/missing-UV degradations now point at the exact
  `meshes[i].primitives[j].attributes.TANGENT` or `.TEXCOORD_0` source path, ignored
  cameras point at `cameras[i]`, and unknown required extensions point at
  `extensionsRequired[i]`.
- Texture decode/readiness diagnostics for glTF-origin material maps now preserve
  source material slots such as `materials[i].pbrMetallicRoughness.baseColorTexture`
  and `materials[i].extensions.KHR_materials_clearcoat.clearcoatTexture`; direct
  `Scene` callers still get `scene.primitives[i].material.*` fallback paths.
- `decodeSceneTextures(target:'cpu-linear', { maxTextureSize })` now resizes
  oversized decoded raw-image payloads before backend upload and reports the
  original/resized dimensions in structured diagnostics.
- `loadGltfForEngine()` and the `@vitrum/engine/gltf` one-call helpers now expose
  the decode bridge directly: `decodeTextures` / `decodePixels` /
  `textureTarget` / `maxTextureSize` options route through
  `loadGltfAndDecodeTextures()` before engine construction or attachment, and the
  result carries decoded counts, `textureDecodeDiagnostics`, and
  `textureDecodeWarnings`.
- Abort signals and deterministic fetch/base-URI errors are wired.

Closure:
- Mostly closed for the first professional API slice.
- Still optional/future: explicit cache hooks, broader malformed-resource
  fixtures, and real-world sample-asset sweeps.

### GLTF-API-02 - asset feature reporting is missing

Status:
- `analyzeGltfAsset()` now reports used/required extensions, hook-required
  extensions, unsupported optional/required extensions, external resources,
  primitive modes/attributes, expected primitive kinds, material fields/maps,
  texture UV sets/transforms, skins, morphs, animations, cameras, and punctual
  lights.
- `loadGltfAsset()` returns this report beside the converted `Scene`.

Closure:
- Closed for structured inventory.
- Still optional/future: source-path granularity for every nested feature and
  broader Khronos sample-model golden reports.

### GLTF-API-03 - backend choice happens before asset compatibility is known

Status:
- `evaluateGltfBackendCompatibility()`, `rankGltfBackends()`, and
  `recommendGltfBackend()` now compare a `GltfFeatureReport` with
  `BACKEND_PROMISE_LEDGER`.
- `loadGltfAsset()` returns `backendCompatibility` plus `recommendedBackend`.
- Fidelity policy ranks textured assets away from walkaround-hybrid when the
  ledger reports unsupported material fields.

Closure:
- Planner/ranking is closed as a library utility.
- The adapter-owned engine-preparation bridge is closed:
  `loadGltfForEngine()` selects or overrides a backend, applies compatibility
  gates, creates/attaches an engine through a host-injected factory or existing
  engine, and returns a `GltfSceneController`.
- The `@vitrum/engine/gltf` subpath is now implemented, including a
  `reject-degraded` guard that refuses `pt-webgpu` when `probeAdapterProfile()`
  reports a non-full trace tier. This closes the one-call strict-mode lite
  downgrade path.
- The engine bridge is no longer report-only for texture decoding: the same
  adapter-owned `loadGltfForEngine()` result now carries decoded-scene
  `textureDecodeReport`, decode counts, decode diagnostics, and decode warnings
  when decode options are supplied, and both engine one-call helpers forward that
  surface.
- The generic adapter now distinguishes pt-webgpu full and lite before engine
  construction: compatibility rows carry `profileId` and `traceTier`, and
  `rankGltfBackends()` emits both `pt-webgpu` and `pt-webgpu-lite` rows while
  preserving `.backend: 'pt-webgpu'` for existing engine selection. The lite
  row downgrades full-tier-only material texture/alpha/env/aniso fields to
  unsupported, matching runtime lite capabilities and warnings.
- `loadGltfForEngine()` now also reconciles the actual attached engine backend:
  if an existing or factory-created engine exposes `backendId`, the bridge
  returns that backend and reruns strict compatibility against it before
  calling `setScene()`. Factory fallback can still proceed in best-effort mode,
  but strict mode no longer validates against one backend and attaches another.
- Strict bridge modes now consume converter diagnostics and texture-readiness
  reports as well as compatibility rows. Structural skips (`missing-position`,
  unresolved compression, unreadable attributes/indices, unsupported primitive
  modes, empty triangulation) reject before engine construction; opaque texture
  handles reject under `reject-degraded` unless the host asserts backend
  readiness with `opaqueTextureHandlesReady`.
- `createProgressiveEngine()` now seeds `ProgressiveHandoffCoordinator` with the
  initial scene, so coordinator mutation APIs can use their designed `setScene`
  fallback when a sub-engine lacks or rejects an incremental patch method.

### GLTF-API-04 - animation playback/update orchestration

Status:
- `GltfSceneController` / `createGltfSceneController()` is now the turnkey
  runtime bridge for imported animations.
- It evaluates clips at absolute time, recomposes the glTF node hierarchy from
  import-time base TRS, updates direct mesh nodes and animated ancestors,
  rebuilds skinned `bones` arrays from animated joint nodes, applies morph
  weight channels, runs `solveSkin()`, and emits `updatePrimitive()` patches
  with `setScene()` fallback for targets without an incremental method.
- `seek()`, `advance(dt)`, active clip selection, `play()` / `pause()` /
  `resume()` / `tick(dt)` clock ownership, looped advance, and applied
  patch/warning reporting are implemented.
- Successful incremental `updatePrimitive()` batches call the target's optional
  `reset()` hook, giving Engine-like targets a single predictable invalidation
  boundary for animation and material-variant mutations. Full-scene fallback
  still goes through `setScene()`.
- Fixtures now cover parent-node animation, joint animation, morph weights, and
  fallback from incremental patches to full `setScene()`, reset propagation,
  plus pause-as-no-op and resume-from-retained-clock behavior.

Closure:
- Closed for single-clip playback and engine mutation orchestration.
- Still optional/future: richer cross-fade scheduling APIs, richer
  topology-changing animation diagnostics,
  multi-primitive mesh-node fixture coverage, mixed skin+morph fixture coverage,
  and broader real-world animated sample sweeps.

### GLTF-API-05 - common glTF extensions are not represented

Evidence:
- The adapter's required-extension allowlist covers Draco, meshopt,
  punctual lights, transmission/ior/volume/specular/sheen/clearcoat/
  iridescence/anisotropy/dispersion/emissive_strength, texture_transform, and
  opt-in texture-source extensions.
- `KHR_materials_dispersion` now maps to the core dispersion field.
- `KHR_texture_basisu`, `EXT_texture_webp`, and `MSFT_texture_dds` are
  represented as host-decode-required alternate texture-source paths when the
  extension is required, selected, or the texture has no base `texture.source`
  fallback. Optional alternates with a base fallback remain compatibility-clean
  until the host opts in. `GltfFeatureReport.extensions.textureSourceUses[]`
  now records the exact texture index, alternate image index, source path,
  selected/fallback/required status, MIME type, and whether that use requires a
  hook, and `loadGltfAsset()` passes the selected `textureSourceExtensions` into
  analysis before ranking/enforcement.
- `KHR_materials_variants` now supports a `materialVariant` selection option and
  falls back to base materials when no active variant is selected.
- `KHR_materials_unlit` now maps to the core `MaterialSpec.shadingModel`
  contract field and compatibility scoring reports `shadingModel` through each
  backend's material support matrix. pt-webgl2, pt-webgpu, and
  walkaround-hybrid consume it as an approximate lighting-independent base-color
  path.
- Archived `KHR_materials_pbrSpecularGlossiness` scalar factors now translate
  approximately to the core metallic-roughness + specular fields; raw extension
  data is preserved for hosts that need exact legacy semantics.
- Archived `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture` now
  emits a structured approximate compatibility issue for glossiness-in-alpha
  at pre-decode planning time. `loadGltfAndDecodeTextures()` /
  `decodeSceneTextures(target:'cpu-linear')` close the pixel-data path by
  baking `roughnessMap = 1 - glossinessFactor * alpha` from the decoded
  `specularColorMap` handle.
- Required-extension policy now accepts `KHR_materials_unlit` and archived
  `KHR_materials_pbrSpecularGlossiness`; scalar-only spec-gloss conversion is
  compatibility-scored as approximate even without a texture.
- `KHR_materials_volume.thicknessTexture` now maps to the core
  `MaterialSpec.thicknessMap` field, and compatibility scoring reports
  `thicknessMap` through each backend's material support matrix. pt-webgl2 and
  pt-webgpu now consume it approximately as a Beer-Lambert distance clamp, and
  walkaround-hybrid consumes readable thickness maps approximately by sampling
  glTF G and exponentiating its pre-baked Beer-Lambert tint.
- Morph-target `TANGENT` deltas are preserved on the core primitive contract and
  are compatibility-scored as approximate backend tangent-space shading data
  rather than silently ignored primitive data.
- `EXT_mesh_gpu_instancing` import + node animation is code/test closed: valid
  node-level TRANSLATION/ROTATION/SCALE accessors import to core
  `InstancedMeshPrimitive` with `nodeWorld * instanceTRS` baked into each
  instance matrix, required use is accepted, compatibility reports the extension
  as supported plus the expected native `instanced-mesh` primitive kind, and
  `GltfSceneController` stores local instance matrices so node/ancestor
  animation patches `instances[]` rather than an ignored `transform`.
  Malformed accessors, no-transform instance payloads, and skinned/morphed
  instancing still emit an `ignored-gpu-instancing` diagnostic and import the
  base mesh/skinned representation once because core has no instanced-skinned /
  instanced-morphed primitive contract yet.
- 2026-06-16 follow-up: the broad UV2+ material-texture gap is narrowed. `uv1`
  remains the core/backend native second UV lane, but `gltfToScene()` now
  losslessly remaps a primitive material that references exactly one higher
  glTF UV set (`TEXCOORD_N`, `N > 1`) into that lane when the primitive has the
  accessor and the material does not also need `texCoord:1`. The primitive-local
  `TextureRef.texCoord` values are cloned to `1`, tangent generation consumes
  the remapped UVs, and compatibility analysis no longer rejects those cases.
  Missing or conflicting high-UV cases emit `ignored-material-texcoord`
  diagnostics and drop the affected texture fields instead of silently sampling
  UV0. Native arbitrary UV-array support remains future core/backend contract
  work.

Closure:
- Decide per extension: implement, require host hook, translate approximately,
  or reject with a structured compatibility error.
- Completed follow-up: CPU-linear texture-bake parity for
  `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture`
  glossiness-in-alpha now exists when the host supplies `decodePixels`.
- Completed follow-up: the engine bridge no longer keeps the
  `specularGlossinessTexture.glossinessAlpha` compatibility issue active after
  that CPU-linear bake succeeds; `reject-degraded` still rejects the broader
  archived spec-gloss model-conversion issue unless the caller accepts that
  approximation.
- Add core fields only when at least one backend consumes them or the
  compatibility report can honestly say they are imported-but-unsupported.
- Continue adding real-world sample sweeps for supported/approximate required
  and optional extensions.

### GLTF-API-06 - material parity is still the dominant arbitrary-asset gap

Evidence:
- The verified support counts above show walkaround-hybrid is not a general PBR
  target today.
- walkaround-hybrid scalar alpha cutout is code-closed as an approximate tier:
  `packBVHRoughMetalFromCore` encodes `alphaMode` / `alphaCutoff` / `opacity`
  into `bvh_material` bit 2 for mask discard and fully-transparent blend
  endpoints. Readable `alphaMap` handles are now code-closed/approximate:
  `materialTextureAtlas.ts` packs alpha maps as linear atlas layers plus
  per-triangle mode/opacity/cutoff metadata, and `materialAtlas.wgsl`
  `traceSceneFirstHitAlphaMaskTextured` applies `opacity * alphaMap.r <
  alphaCutoff` in RIS, shade, temporal/spatial primary casts, ReSTIR-GI, and
  NRC GI paths. The shared cast-shadow mask still skips scalar bit 2 for
  occlusion rays. 2026-06-16 follow-up: camera-visible fractional blend now
  routes through the transparent-OIT pass (`transparentOit.wgsl.ts`) after
  `indirect-combine`; the opaque shade pass skips fractional blend layers, OIT
  ray-walks those layers front-to-back, writes `transparentCompositeTexture`,
  and temporal accumulation consumes that composited output. 2026-06-16
  follow-up: OIT direct-sun layer radiance now consumes the same atlas-backed
  GGX/clearcoat/sheen/aniso/iridescence material payload as opaque
  shade/ReSTIR material scoring instead of a diffuse-only sun term, and HDRI
  sky/environment lighting uses a deterministic five-tap material-lobe estimate.
  Latest 2026-06-16 follow-up: analytic point/spot light radiance now joins that
  transparent OIT direct-light path through binding 13, and direct sun/point/spot
  shadows deterministically attenuate through atlas-backed alpha coverage. The
  structured warning remains because light-map/emissive terms are first-hit
  approximations, and reservoir-backed ReSTIR direct light plus GI participation
  is still approximate, including material `updatePrimitive` patches that mutate
  a primitive into fractional blend.
- walkaround-hybrid readable `emissiveMap` is code-closed/approximate for
  camera-visible emitter glow, direct-light selection power, and merged-BVH
  ReSTIR-DI emitter payloads: `materialTextureAtlas.ts` packs emissive maps as
  sRGB-decoded atlas layers with per-map UV/transform/wrap metadata,
  `packBVHEmissiveLeFromCore()` now stores scalar production Le for the
  camera-visible glow buffer so the shade pass samples readable emissive maps
  exactly once at hit UV instead of pre-averaging and resampling them,
  `materialSpecEmissiveLe()` folds the average linear RGB of CPU-readable
  emissive maps into the shared ReSTIR/DDGI/RC emitter selection-power path,
  and the 2026-06-16 DI follow-up packs a source-triangle lane for merged-BVH
  material emitters so RIS candidate scoring, pHat reuse, and final shade sample
  mapped radiance at the stored triangle `xi`. Later same-day follow-up: TLAS
  material-backed emitters now translate the world-expanded emitter triangle
  through `bvhTriToMergedTri` and `primitiveTlasBindings` back to the local BLAS
  material-atlas triangle, so instanced decoded glTF-style emissive maps share
  the native sampled-texel payload too. Later follow-up: mirrored TLAS instances
  encode reversed barycentric orientation in the source-triangle lane
  (`-(tri + 2)`), so they also sample the local material-atlas texel instead of
  falling back to averaged radiance. Analytic/extra emitters intentionally retain
  averaged `Le`. Exact UV-varying texel-PDF selection and GI/RC/DDGI texel-space
  emission are still approximate rather than native parity.
- pt-webgpu readable `emissiveMap` now feeds implicit mesh-area NEE power:
  `emitterPacking.ts` folds the CPU-readable sRGB-decoded average RGB into
  synthesized emissive-mesh radiance, uses that same helper for the geometry
  staleness predicate, suppresses black-map phantom emitters, and warns when an
  opaque/unreadable map handle forces the scalar-emissive fallback. This closes
  the scalar-only implicit emitter hole for decoded glTF/WebGPU-target texture
  payloads; exact UV-varying emitter texel PDFs remain a promotion tail.
- pt-webgl2 now matches that decoded-emissive-mesh NEE behavior on its native
  triangle-light path: `meshAreaLights.ts` synthesizes triangle lights from
  nonzero material `emissive * emissiveIntensity`, folds CPU-readable
  `emissiveMap` average energy into the implicit radiance, suppresses black-map
  phantom lights, warns on opaque/unreadable map fallback, and
  `mutateSceneTextures.ts` repacks the mesh-light texture when scalar material
  emission changes through `updatePrimitive({ material })`. Explicit
  `mesh-area` emitters still keep their existing folded-material path for
  forward/path-hit emission. This closes the glTF-style scalar emissive mesh NEE
  hole on pt-webgl2; exact UV-varying emitter texel PDFs remain a promotion tail.
- pt-webgl2 is the closest material-complete backend, but still has unsupported
  rows and needs reference-render tests for the high-value rows it claims.
- pt-webgpu has substantial material support, and full-tier megakernel
  extension-lobe maps now modulate the decoded material before ordinary
  BSDF/PDF/NEE calls, including `clearcoatNormalMap` in the clearcoat lobe. It
  still needs extension-lobe parity across the remaining specialty schema paths
  (`PTWG-MAT-01`). The local non-schema paths now use the
  full helpers: full/lite direct and env lighting, full BSDF-side area/env
  connections, lite BSDF-env connection, SPPM receiver gather, MNEE receiver
  caustics and cone-vs-BSDF sampled PDF, BDPT connection endpoints, main
  eye-path sampling/PDFs, BDPT surface light-subpath material-payload scattering,
  and ReSTIR-PT producer suffix/source Lo.

Closure:
- `CAP-01` / the core material matrix is no longer the blocker for this row:
  `engineContract.test.ts` keeps every `MaterialSpec` key ledgered, and the
  2026-06-15 walkaround follow-up pins the permanent unsupported family
  (`displacement*`, spectral/dispersion/scattering, layered, and thin-film stack
  fields) through both `setScene()` and `updatePrimitive()` structured warnings.
- `tools/gltf-material-sweep/` now covers the CPU/API half of the material sweep:
  it loads a synthetic glTF through `loadGltfAndDecodeTextures()`, checks every
  base/KHR texture field in `textureDecodeReport`, verifies uv1 transform
  preservation, asserts sampler/wrap/mipmap metadata plus expected
  `*.samplerPolicy` backend compatibility diagnostics, and asserts backend
  readiness including walkaround `thicknessMap` as `ready`.
- Prefer pt-webgl2/full pt-webgpu for fidelity policy until walkaround either
  implements texture-driven PBR fallback or is explicitly a realtime-profile
  target in the compatibility report.
- Remaining material-furnace proof is GPU/reference-render work: the
  lobe-specific CPU oracle is now present, but the sweep/golden fixtures still
  need to render on the recommended backend, assert non-black output, and
  compare against tolerance-bounded references.
- Remaining pt-webgpu material-lobe work must be scheduled as specialty schema
  work, not helper plumbing: inverse/adjoint gradients and reference A/B /
  material-furnace promotion gates.

Behavioral glTF proof note:
- `tools/behavioral-gate/gate.mjs --filter gltf` now exercises the public
  `loadGltfForEngine()` bridge for unlit, textured PBR, transmission, skinned
  animation, and Draco mock fixtures before rendering them on the real
  `pt-webgpu` lane. The injected patch-target asserts controller attachment via
  `setScene()`, and the skinned-animation fixture now advances the controller
  and requires an observable `updatePrimitive()` patch before render.

### GLTF-02 - Draco and meshopt compressed primitives

Status:
- Lower-level import is implemented through host-supplied
  `KHR_draco_mesh_compression` and `EXT_meshopt_compression` decoder hooks, with
  fallback behavior and `extensionsRequired` rejection.
- `packages/gltf-adapter/src/gltfCompression.test.ts` currently covers both
  compressed paths.

Closure:
- Fold decoder-hook requirements into `loadGltfAsset()` / `GltfFeatureReport`.
- Add real-world compressed sample assets beyond the synthetic unit fixtures.

### GLTF-03 - glTF animation import

Status:
- The adapter imports animation samplers/channels into core `AnimationClip[]`.
- Core sampling handles LINEAR, STEP, and CUBICSPLINE, including rotation
  normalization.
- `packages/gltf-adapter/src/gltfModesMorphsAnimations.test.ts` covers import
  mechanics.

Closure:
- Treat import as closed at the low-level adapter layer.
- Runtime playback/update orchestration is closed for single-clip animation via
  `GltfSceneController`; remaining work is optional/future broadening called out
  under `GLTF-API-04`.

### GLTF-04 - glTF morph target import

Status:
- POSITION and NORMAL morph target deltas and node/mesh weights import into the
  core primitive shape; unskinned morphed meshes are promoted through an identity
  skeleton path.
- TANGENT morph deltas import into `SkinnedMeshPrimitive.morphTargetTangents`.
  The shared CPU `solveSkin()` path now blends those deltas into solved
  tangent.xyz streams, preserves base tangent handedness, and pt-webgpu /
  walkaround CPU-skin fallbacks upload the solved tangents for tangent-space
  material maps. This remains a structured compatibility downgrade:
  `analyzeGltfAsset()` reports `hasMorphTargetTangents`,
  `evaluateGltfBackendCompatibility()` emits a source-pathed
  `morphTargetTangents` approximate primitive issue, and
  `gltfAssetApi.test.ts` pins that public API behavior.

Closure:
- Treat morph-tangent data preservation plus CPU-solved backend consumption as
  closed for the current professional contract: deterministic, source-pathed,
  and test-covered. 2026-06-16 follow-up: pt-webgl2 `solveSkinPrimitives()`
  now preserves `solveSkin()`'s posed tangents, including `morphTargetTangents`,
  instead of dropping them before attribute packing; `GltfSceneController` also
  emits solved tangents in runtime morph-weight and skeleton animation patches.
  Full renderer promotion still requires GPU-native tangent skinning / broader
  backend evidence, so compatibility remains approximate instead of native.
- Controller-side morph playback is closed under `GLTF-API-04`.

### GLTF-05 - glTF primitive modes

Status:
- TRIANGLES imports directly.
- TRIANGLE_STRIP and TRIANGLE_FAN are triangulated into indexed triangle lists.
- POINTS/LINES/LINE_LOOP/LINE_STRIP now import as generated triangle mesh
  fallback geometry. POINTS become tiny cubes; line modes become thin
  rectangular prisms. The adapter reports this as `fallback-generated-mesh`
  compatibility plus `fallback-generated-primitive-mode` import diagnostics with
  `meshes[n].primitives[m].mode` paths.

Closure:
- Treat all glTF primitive modes 0-6 as closed for the current professional
  contract: native triangle list, triangulated strip/fan, or deterministic
  fallback-generated mesh for point/line modes.
- Optional future promotion: add native point/line primitive kinds if exact
  topology semantics become a first-class core contract. Until then the
  approximation is explicit and machine-readable.

### GLTF-06 - glTF material mapping needs parity audit

Closure:
- Combined metallic-roughness texture mapping is closed: the adapter maps the
  same glTF ORM texture to both `roughnessMap` and `metallicMap`.
- `KHR_texture_transform.texCoord` override is closed in the shared
  `resolveTextureRef()` importer, with an end-to-end adapter fixture proving
  `TextureRef.texCoord` and transform fields survive `gltfToScene()`.
- The per-extension texture-map fixture sweep is closed:
  `packages/gltf-adapter/src/gltfTextureSweep.test.ts` enumerates every
  `resolveTextureRef()` consumer in `materials.ts` and proves the decoded handle,
  UV-set override, and KHR texture transform survive `gltfToScene()` for each
  imported base/KHR material map.
- The CPU material-sweep preflight is closed:
  `tools/gltf-material-sweep/sweep.mjs` exercises
  `loadGltfAndDecodeTextures()`, `textureDecodeReport`, backend compatibility,
  and backend-readiness diagnostics across the same material-map family. The
  follow-up fixed walkaround `thicknessMap` readiness drift and added a
  source-pathed `KHR_materials_dispersion` unsupported compatibility assertion.
- The specular-glossiness texture-alpha bake is closed: the CPU-linear decode
  path derives a `roughnessMap` from
  `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.a` using
  `1 - glossinessFactor * alpha`, preserves texCoord / transform / wrap
  metadata, suppresses only the now-satisfied `glossinessAlpha` compatibility
  issue, and now has both direct decode and `loadGltfForEngine()` attachment
  tests.
- Remaining work belongs to `GLTF-API-05` and `GLTF-API-06`:
  transparent lighting/GI/shadow promotion plus
  displacement/spectral/layered/scattering promotion if those ever become
  walkaround goals, and backend material-consumption parity.

## P5 validation and promotion gates

These gates prevent the same class of plan drift from recurring.

### GATE-01 - CLOSED 2026-06-13 - Backend promise ledger audit

Status:
- Source-verified closed by `packages/core/src/__tests__/ledgerVsCapabilities.test.ts`.
- The gate imports live pt-webgl2 support/capability data and compares it to
  `BACKEND_PROMISE_LEDGER`, including primitive/emitter/environment kinds,
  analytic fallback-generated-mesh rows, mutation grades, debug surface, and
  full-tier `supportsAuxBuffers:true` with lite-tier runtime downgrade to
  `false`.
- Keep the gate as a permanent regression guard; future backend rows should
  extend it rather than reopening this audit item.

### GATE-02 - Material contract audit

For every `MaterialSpec` field:
- Record consume/approximate/ignore/unsupported per backend.
- Add diagnostics for ignored fields that affect visual output.
- Add at least one executable test for every field marked implemented.

Status:
- Core `engineContract.test.ts` keeps the `BACKEND_PROMISE_LEDGER` material
  matrix exhaustive over every public `MaterialSpec` key for every backend.
- 2026-06-15 follow-up: pt-webgpu now has direct descriptor/WGSL pins for the
  native scalar lanes `aoMapIntensity`, `lightMapIntensity`, and
  `envMapIntensity`; walkaround-hybrid now consumes `envMapIntensity` through
  material-atlas metadata plus HDRI ReSTIR-DI p-hat/resolve paths. The warning
  tests now cover the full permanent walkaround unsupported material family on
  both scene load and material update, not only displacement.
- 2026-06-15 glTF sweep follow-up: `npm run gltf-material-sweep` keeps the
  CPU/API material sweep executable, including `textureDecodeReport` and
  backend-readiness checks for every imported base/KHR material map.
- Keep this gate as a permanent regression guard when adding new material
  fields or promoting backend support rows.

### GATE-03 - Mutation matrix audit

For every mutation type:
- Confirm CPU scene, GPU buffers, texture paths, light-tree paths, TLAS/BLAS paths,
  denoiser history, and GI propagation all update together.
- Include pt-webgpu lite/full separately.

Status:
- pt-webgpu has broad focused mutation coverage across primitive, emitter,
  environment, add/remove, lite downgrade, emissive stale-light, reservoir, and
  resource paths.
- 2026-06-15 follow-up: `mutationDesyncs.test.ts` now directly asserts
  router-level `updateEmitter()` point-light buffer writes, scene-state commit,
  and accumulation reset, plus same-sized HDRI `updateEnvironment()` texel/CDF
  writes, environment metadata mutation, scene-state commit, and reset without
  falling through to `setScene()`.
- pt-webgl2 now has focused coverage for native scalar material, emitter
  (analytic plus mesh-area folded-material/mesh-light refresh), environment,
  and resize mutations; transform/positions/topology/add/remove still use
  rebuild-style behavior.
- walkaround-hybrid now has a focused non-GPU seam test for transform, material,
  emitter, environment, lighting, resize behavior, unsupported material-field
  diagnostics, and fractional alpha-blend diagnostics on incremental material
  mutation.
- 2026-06-16 follow-up: TLAS transform refits now invalidate the DDGI probe
  cache in addition to marking instances dirty and re-syncing the shared BVH.
  The mutation-matrix seam asserts this directly, matching the merged-BVH
  transform behavior and closing the stale Road 4D probe-invalidation row.
- 2026-06-16 follow-up: walkaround `updatePrimitive({ bones | boneInverses |
  morphWeights })` on a skinned mesh now re-solves through `solveSkin`, routes
  the solved positions/normals through the existing TLAS/merged refit path, and
  preserves the submitted pose fields in scene state. The same seam asserts
  bones-only TLAS mutation, accumulation reset, DDGI sync, and DDGI probe
  invalidation.

Remaining:
- Full GPU/resource mutation matrix promotion still needs end-to-end tests where
  real GPU buffers, bind groups, denoiser history, and GI propagation are
  observable together.

### GATE-04 - Renderer math oracle suite

Status: CORE ORACLES CLOSED / SPECIALTY PROOF STILL OPEN.

Source-verified 2026-06-15:
- SPPM photon flux is covered by
  `packages/pt-webgpu/src/__tests__/oracle.sppmPhotonFlux.test.ts`, which
  derives per-source powers independently and proves `lightSelectInvPdf`
  conserves total emitted flux across point, rect, and environment sources.
- SPPM per-pixel progressive stats are covered by
  `packages/pt-webgpu/src/__tests__/sppmHashGrid.test.ts`, whose TS mirror
  proves the Hachisuka recurrence (`N'`, radius shrink, tau update, M=0
  stability, and first-frame seeding) against closed-form invariants.
- BDPT contribution/pdf assembly is covered by
  `packages/pt-webgpu/src/__tests__/bdptConnectionMisFull.test.ts` and
  `packages/pt-webgpu/src/__tests__/oracle.bdptConnectionCosine.test.ts`.
  The first compares an independent Veach MIS recurrence against the
  shared-samplers oracle; the second derives the finite-area endpoint,
  one-bounce diffuse light tracing, and glossy light-vertex contribution from
  the rendering equation rather than mirroring shader assembly.
- DI ReSTIR candidate accounting and selected-point shading are covered by
  `packages/walkaround-hybrid/src/__tests__/oracle.restirDiEstimator.test.ts`.
  The oracle independently transcribes RIS candidate generation, W finalize,
  selected-xi p-hat, and shade consumption, with characterization coverage for
  the historical centroid/fresh-xi and mixed-measure defects.
- DDGI miss visibility semantics are covered by
  `packages/walkaround-hybrid/src/__tests__/oracle.ddgiVisibilityMoments.test.ts`,
  which independently models f32 accumulation, f16 visibility atlas storage,
  Chebyshev visibility, miss skipping, and all-sky open-visibility semantics.

Remaining:
- Extension-lobe contribution/PDF parity for inverse path-replay adjoints.
  2026-06-15 follow-up: pt-webgpu inverse sessions now accept common scalar/RGB
  extension-lobe material params (`specularColor`, `specularIntensity`,
  `clearcoat`, `clearcoatRoughness`, `sheenColor`, `sheenRoughness`,
  `iridescence`, `iridescenceIor`, `anisotropy`, `anisotropyRotation`) through
  the finite-difference baseline and explicitly degrade requested path-replay
  sessions back to finite-difference for those fields. 2026-06-16 follow-up:
  scoped light-source path replay now covers delta directional, point, spot, and
  center-sampled rect/disc/mesh-area direct lights. Analytic path-replay adjoints
  remain open for environment, soft-sun angular diameter, full stochastic area
  sampling, indirect paths, maps/transmission/layers/volume/spectral material
  domains, and full extension-lobe contribution/PDF gradients until converging
  inverse fits validate them.
- Material-furnace/reference A/B promotion for the sampled eye, ReSTIR-PT, and
  BDPT paths is now structurally implemented. The local extension-lobe CPU oracle
  and ReSTIR-PT specialty one-sample identity fixture are closed; the remaining
  promotion evidence is GPU/reference-render based.

### GATE-05 - Reference-render A/B suite

Capture before/after references for:
- Tangent-space normal maps with mirrored UVs.
- Procedural sky in every backend that advertises it.
- Area lights with reservoir selected-point occlusion.
- SPPM caustics under multiple light types.
- Material extension lobes in pt-webgpu.

### GATE-06 - glTF one-call behavioral gate

Status:
- 2026-06-15 follow-up: `tools/behavioral-gate/gate.mjs` now makes
  `pt/gltf-textured-pbr` exercise the real one-call texture decode bridge
  instead of only preserving a texture reference. The fixture supplies a
  raw-image handle plus `decodePixels`, calls `loadGltfForEngine()` with
  `decodeTextures:true` and `textureTarget:'cpu-linear'`, asserts
  decoded/unchanged counts, asserts the `textureDecodeReport` has a
  backend-ready CPU texture row, verifies the engine attached the decoded
  controller scene, then renders the pt-webgpu frame. The current
  `npm run behavioral-gate -- --filter gltf` pass covers unlit, decoded
  textured PBR, transmission glass, skinned animation/controller patching, and
  mock-Draco ingestion.

## Rejected or stale claims

Do not carry these as open gaps unless the code regresses again.

- Current `three-bindings` package gaps: stale. The current package scope is
  `gltf-adapter`, not the old three-bindings package.
- walkaround denoiser option values lack implementation: stale. Built-in denoiser
  values are registered and validated; missing prerequisites are rejected.
- RC/GI BVH mutation alignment missing: stale. GI propagation syncs DDGI, TLAS RC
  buffers, merged RC in-place refit, and rebuild fallback.
- pt-webgl2 `castShadow` missing: stale. pt-webgl2 packs and checks castShadow.
- pt-webgl2 HDRI intensity/rotation missing: stale. These feed frame uniforms.
- glTF adapter console diagnostics missing: stale. The adapter returns warnings,
  and converter-owned import degradations now also return structured diagnostics
  with stable codes/source paths.
- pt-webgpu procedural-sky missing: stale. pt-webgpu now bakes Preetham sky to an
  importance-sampled HDRI path.
- shared-bvh attenuation/thickness fingerprint omission: stale. Current
  world-space merge material signatures include those fields.
- shared-bvh sampled fingerprint in the `SceneBvh` rebuild-skip path: stale after
  the 2026-06-12 Wave 3 fix. `SceneBvh.updateFromCore()` now uses
  `fingerprintBuffersExact()` for correctness-sensitive rebuild skips, while the
  sampled helper remains available only for versioning/upload heuristics. The
  shared-bvh test suite now pins equal-length vertex edits as well as the large
  unsampled-byte edit that originally motivated the exact fingerprint.
- walkaround `updateLighting({ primaryLightDir })` does not re-sync DDGI sun:
  stale. `HybridEngine.updateLighting()` calls `_syncDdgiLightsFromCoreScene()`
  for core scenes and `orientDdgiSunLights(...)` for ctor-light-only scenes;
  primary-light intensity updates also call `setSunIntensityMultiplier()` with
  scene-directional single-count handling.
- RC cascade-dimension validation and hybrid bounds-refit stale merge uniforms:
  stale after the 2026-06-12 Wave 4 fix. `validateCascadeDims()` enforces the
  square/2x ray-grid contract before allocation/dispatch, and
  `RCSubsystem.refitCascadeBounds()` invalidates dispatcher bindings when bounds
  change so merge uniforms rebuild with fresh cascade geometry.
- walkaround denoiser `state()` has no consumers: stale. The active denoiser
  state flows through `WalkaroundGPUPipeline.getActiveDenoiserState()` into
  `FrameStats.denoiserState`; `frameStatsDenoiserState.test.ts` covers failed,
  retryable, in-flight, fallback, and null states.
- RC per-frame sun is monochrome: stale after the 2026-06-12 RC sun wave. The
  frame orchestrator passes scene directional `color * intensity` into
  `RCSubsystem.dispatchFrame()`, and falls back to grey
  `primaryLightIntensity` only when no scene directional exists.
- pt-webgl2 vertex color / secondary UV blanket gaps: stale after the
  2026-06-12 vertex-color wave. Current shader and attribute paths consume
  secondary UVs, and `COLOR_0` now threads from glTF/core primitives into
  `attributesArray` layer 3 with the material `vertexColors` flag enabled for
  affected material slots. pt-webgpu full consumed the same core `colors` stream
  on 2026-06-14 via shared-bvh rgba packing, group(3)/binding(11), baseColor
  modulation, and alpha pass-through. 2026-06-16 follow-up: `COLOR_1+`
  secondary vertex-color attributes now surface as structured
  `ignored-vertex-color-set` import diagnostics and unsupported planner issues
  instead of being silently ignored. walkaround-hybrid consumed the same core
  colors stream on 2026-06-15 via shared-bvh/world-space RGBA packing, a scene
  vertex-color texture, visible baseColor modulation, and traversal alpha
  coverage; compatibility is `approximate` because realtime GI reservoirs are
  not full secondary path-tracer material transport. pt-webgpu-lite remains an
  honest structured-unsupported path for glTF vertex colors, and direct
  pt-webgpu lite `setScene` now warns on `Scene.primitives[].colors` even when
  hosts bypass the glTF adapter.
- Direct `@vitrum/gltf-adapter` callers can now target the concrete
  `pt-webgpu-lite` profile with `loadGltfForEngine(..., { backend:
  "pt-webgpu-lite" })`. The factory still receives `backend:"pt-webgpu"`, but
  strict compatibility uses the lite profile row, so adapter-only one-call loads
  reject lite-unsupported assets before engine construction just like
  `@vitrum/engine/gltf`'s runtime-tier gate.
- Unknown required glTF extensions now fail with structure instead of a plain
  throw: `gltfToScene` raises `GltfImportError` carrying an
  `unsupported-required-extension` diagnostic at the exact
  `extensionsRequired[i]` source path.
- pt-webgpu full-tier material texture mutation stale: stale. Texture-map changes
  are rejected from the material fast path and fall through to repack.
- Blanket "pt-webgpu lite has no point/spot/rect/HDRI support": stale. Initial
  lite rendering has those paths; emitter/environment mutation texture sync is
  closed, and transform/topology support is now truthfully downgraded on lite.
- Blanket "glTF compression/animations/morphs/strip/fan are not imported":
  stale. Those lower-level importer paths are implemented and covered by the
  glTF adapter suite; the open gap is turnkey arbitrary-asset loading,
  compatibility planning, and playback/update orchestration. 2026-06-14 proof
  addendum: `gltfProgressiveSubpath.test.ts` now asserts the engine progressive
  helper returns `textureDecodeReport` plus adapter warnings, and
  `gltfTextureSweep.test.ts` now proves enabled `MSFT_texture_dds` alternate
  source selection flows through `loadGltfAsset()` while the report still
  identifies the imported material map.

## Recommended execution order

1. Close P0 correctness firebreak.
2. Close P1 capability honesty so hosts can trust the engine while deeper
   fidelity work continues.
3. Close the new P4 glTF API layer (`loadGltfAsset`, feature report,
   compatibility planner, playback controller) so arbitrary assets enter through
   one predictable path.
4. Close P3 renderer math oracles before making large shader changes.
5. Close P2 backend-wide contract implementation/downgrades and material parity
   promotions that the glTF compatibility planner exposes.
6. Run P5 gates and update `items_to_fix.md` / fidelity matrix only after code and
   tests agree.
