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
  handedness, guards legacy zero handedness in GLSL, and avoids reusing
  rest-pose tangents after CPU skinning.
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
  inputs, and lite transform/instanced-topology patches throw before mutating
  TLAS-only buffers.
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
  unconsumed-material allowlist.

Follow-up Codex closure sweep (same date, WSL Node 24.13.0):

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
- `<VitrumCanvas>` now accepts a creation-time `gltf` prop plus
  `gltfOptions`, loads through `loadGltfAsset`, forwards the imported scene to
  `attachVitrum`, and passes the `gltfAsset` recommendation through the
  lifecycle into `createEngine` so `prefer:"auto"` follows the compatibility
  planner. Focused React/lifecycle tests pin both wrapper and forwarding seams.
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
  `updatePrimitive()` with `setScene()` fallback. The controller fixtures cover
  ancestor-node animation, mutation fallback, morph-weight playback, and skeletal
  joint playback. The glTF adapter suite is now 163 tests.
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
  `MaterialSpec.thicknessMap` field. The glTF adapter suite is now 135 tests.
- The pt-webgpu material texture backend consumption slice landed after that
  adapter import work: `materialTextures.ts` packs per-map UV metadata for every
  map the backend currently samples (baseColor, emissive, normal, ORM, AO,
  lightMap, bumpMap, anisotropyMap, alphaMap, transmissionMap), and
  `material.wgsl.ts` samples those maps with their own `TextureRef.texCoord`,
  KHR_texture_transform, wrap modes, and heterogeneous-layer UV-fit scales.
  The core promise ledger now promotes alpha/transmission/emissive/AO/light/
  bump/anisotropy maps where this was the remaining approximation; `normalMap`
  stays approximate until authored tangent.xyzw/handedness is consumed.
- SPEC-01 pt-webgpu scalar `KHR_materials_specular` consumption landed: material
  vec4 #27 carries `specularColor.rgb` + `specularIntensity`, `material.wgsl.ts`
  decodes them, `bsdf.wgsl.ts` uses them for dielectric F0, and the scalar pair
  now flows through ordinary PT BRDF/PDF paths, lite/full env connection
  interfaces, MNEE/SPPM receiver paths, and BDPT light-subpath surface
  scattering. The promise ledger intentionally keeps pt-webgpu
  `specularColor`/`specularIntensity` at `approximate` until ReSTIR-PT
  reservoir/resolve payloads and remaining legacy default paths carry the same
  fields. Verification: focused pt-webgpu material/WGSL/BDPT suites, core
  ledger/capability suites, pt-webgpu typecheck, `git diff --check`, and
  `npm run shader-gate -- --self-test` (51 production shaders OK; injected
  self-test failure detected).
- The fifth arbitrary-glTF API/compatibility slice landed in
  `@vitrum/gltf-adapter`: `loadGltfForEngine()` now combines `loadGltfAsset()`,
  backend compatibility selection, optional compatibility rejection, injected
  engine construction or existing-engine attachment, and
  `GltfSceneController` creation without adding an `@vitrum/engine` dependency.
  Compatibility scoring now treats morph-target `TANGENT` deltas as structured
  unsupported issues instead of passive inventory counts, while
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
  `specularGlossinessTexture` RGB path maps to `specularColorMap`; glossiness
  in alpha remains a warned, structured approximate downgrade because the
  current `TextureRef` contract cannot invert/bake that channel into
  `roughnessMap`. The glTF adapter suite is now 136 tests.
- The seventh arbitrary-glTF primitive-policy slice landed in
  `@vitrum/gltf-adapter`: `POINTS`, `LINES`, `LINE_LOOP`, and `LINE_STRIP`
  now have a focused policy fixture proving deterministic skip warnings and
  structured `mode:<n>` unsupported compatibility issues. The glTF adapter
  suite is now 138 tests.
- The eighth arbitrary-glTF contract slice landed across `@vitrum/core` and
  `@vitrum/gltf-adapter`: `MaterialSpec.shadingModel?: 'pbr' | 'unlit'` is now
  a first-class contract field and `KHR_materials_unlit` imports to
  `shadingModel: 'unlit'` instead of an adapter warning. Backend compatibility
  ranking now reports unlit assets through the normal material field support
  path: pt-webgl2, pt-webgpu, and walkaround-hybrid are `approximate`.
- The ninth arbitrary-glTF contract slice landed across `@vitrum/core` and
  `@vitrum/gltf-adapter`: `MaterialSpec.thicknessMap` is now a reserved texture
  field, every backend promise-ledger material matrix explicitly marks it
  unsupported, and `KHR_materials_volume.thicknessTexture` imports to
  `thicknessMap` instead of being dropped with a warning. Backend compatibility
  ranking now reports volume thickness textures through the normal material
  field support path.
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
  material fields are locally available. Remaining base-helper sites are now
  concentrated in sampler/PDF-schema paths: ReSTIR-PT visible-vertex material-map
  parity, clearcoat/sheen source sampling, BDPT light-subpath scatter PDFs, and
  the inverse adjoint harness.
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
- The seventeenth arbitrary-glTF geometry slice landed in `@vitrum/gltf-adapter`:
  normal/bump/clearcoat-normal mapped primitives that omit authored `TANGENT`
  now synthesize per-vertex xyzw tangents from POSITION/NORMAL/TEXCOORD_0 during
  import. Authored tangents are preserved unchanged, and missing-UV/degenerate
  cases emit adapter warnings instead of silently pretending the tangent-space
  basis exists.
- The eighteenth pt-webgpu ReSTIR-PT material-lobe slice landed:
  `ReservoirPTHero` widened from 36 to 48 u32 words and now serializes the
  visible vertex's scalar clearcoat/sheen/iridescence fields plus anisotropy
  state. Producer, temporal, spatial, finalise, and resolve all route p-hat /
  reconstruction through the same full-lobe visible-domain helper, and resolve
  now uses `evaluateBrdfFull` instead of the base helper. The producer samples
  anisotropic visible-vertex specular directions and computes the matching
  anisotropic base source PDF while deliberately excluding clearcoat/sheen from
  `pdfSrc` until those lobes are actually sampled. Focused reservoir-layout and
  ReSTIR-PT contract tests pin the 192-byte layout, field serialization,
  domain-copy helper, full-lobe target/resolve, and anisotropic producer path;
  the WGSL shader gate compiles all four ReSTIR-PT passes.
- The nineteenth pt-webgpu material-texture slice landed: full-tier material
  descriptors now extend to 63 vec4s and pack clearcoat factor/roughness,
  sheen color/roughness, iridescence factor/thickness, and specular
  color/intensity maps. `material.wgsl.ts` samples the glTF channel conventions
  from the correct sRGB/linear texture arrays with per-map texCoord,
  KHR_texture_transform, wrap, and UV-fit metadata, and the shade prologue
  modulates decoded lobe parameters before downstream BSDF/PDF/NEE calls.
  The promise ledger promotes those map rows to `approximate`, not `native`,
  because ReSTIR-PT visible-vertex texture-map parity, clearcoat/sheen
  source-lobe sampling/PDF schemas, and BDPT light-subpath scatter PDFs remain
  dedicated sampler/payload work. `clearcoatNormalMap` stays unsupported.
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

### PTWG-03 - SPPM photon emission is under-normalized and source-incomplete

Evidence:
- Photon pass picks uniformly among directional/point/spot lights.
- Flux is divided by photon count but not by light-selection probability.
- Rect, disc, mesh-area, and environment photon sources are excluded.
- Directional photon emission loses RGB/multi-directional parity and spot
  penumbra parity relative to direct NEE.

Closure:
- Include `1 / p_select`, or switch to power-proportional emission.
- Emit photons from the same packed light/source model as NEE, including area
  and environment sources or explicitly narrow the SPPM contract.
- Add one-light vs two-light flux tests, colored directional tests, spot penumbra
  tests, and a rendered caustic reference.

### PTWG-04 - SPPM per-pixel stats appear to update per bounce

Evidence:
- `photonMapContribution()` updates `sppmPixelStats[pixelIndex]`.
- The call site is inside the main bounce loop, so max-bounce paths can advance
  one pixel's SPPM statistics more than once.

Closure:
- Restrict SPPM gather/stat updates to the primary visible gather surface, or
  separate path-throughput accumulation from per-pixel progressive stats.
- Add a max-bounces > 1 test that proves one frame advances each pixel once.

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
- Lite capabilities advertise broad incremental patch support and native
  instanced-mesh support.
- Lite intersection starts traversal at BVH root `0u` and has no TLAS group.
- Transform/instance fast paths update TLAS data that lite traversal does not
  consume.

Closure:
- Closed in the 2026-06-11 implementation wave by making lite capabilities
  tier-specific, adding lite `setScene()` diagnostics for instanced meshes and
  non-identity transforms, and rejecting transform/topology fast paths before
  they update TLAS-only buffers.
- Optional future upgrade: implement lite TLAS traversal or rebuild/upload a
  lite-consumed merged world-space BVH on transform/instance patches, then add
  lite render/pick oracles with multiple transformed instances.

### PTWG-07 - pt-webgpu lite emitter/environment mutations leave texture path stale

Evidence:
- Full `setScene()` refreshes lite light/env textures and CDF textures.
- `updateEmitter()` and same-sized `updateEnvironment()` update storage buffers
  and light tree, while lite shaders read the texture path.

Closure:
- Regenerate/upload lite textures on lite emitter and environment mutations, or
  route those mutations through full repack.
- Add lite `updateEmitter()` and `updateEnvironment()` tests.

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
- Internal debug/resource chatter remains console-only by design; it is not a
  contract-affecting degradation.

### CAP-01 - Per-field material support is not explicit enough

Evidence:
- Core `MaterialSpec` contains many maps/scalars.
- pt-webgpu collects only a subset of maps.
- walkaround intentionally ignores many material fields.
- pt-webgl2 supports many glTF maps, but still misses some fields such as surface
  anisotropy, displacement, receiveShadow, and layered normal maps.

Closure:
- Build a backend-by-backend material support matrix from code, not docs.
- For every field, mark implemented, approximated, ignored-with-warning, or
  unsupported.
- Add tests for the high-value implemented rows and diagnostics tests for
  unsupported rows.

## P2 backend-wide contract completion

These are public contract fields or backend parity promises that still need
implementation or explicit downgrade.

### MAT-01 - Authored tangent stream consumption must be backend-wide

Closure:
- pt-webgl2: consume tangents as part of WEBGL2-01.
- pt-webgpu: either consume tangent streams in material frame construction or
  explicitly document/diagnose derived tangent frames.
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
    (rect/disc/spot/point/directional); the mesh-area triangle-light NEE
    strategy + forward/BDPT paths do not consume it.
  - pt-webgpu `approximate`: per-light lanes (directional sign-encoded
    angularDiameter; point/spot extra .z; rect/disc center .w; mesh-area
    radiance .w) consumed by the default kernel/kernelLite NEE loops + the
    connect.wgsl BSDF-MIS area connections; off-default integrators (BDPT
    light subpath, ReSTIR-PT, MNEE/SPPM caustic legs) and in-medium
    directional NEE still shadow-test; lite directional rides the flag-less
    UBO mirror.
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
- Remaining optional future work: walkaround GI-side masking + emitter flag,
  pt-webgpu off-default-integrator coverage, mesh-area emitter flag on
  pt-webgl2 — promote rows with renderer A/B evidence when implemented.

### WEBGL2-02 - pt-webgl2 procedural sky is unsupported

Evidence:
- Core has `procedural-sky`.
- pt-webgl2 capability set lists only `none` and `hdri`.
- Non-HDRI environment builds an empty environment payload.

Closure:
- Implement procedural sky sampling or bake procedural sky to an env map.
- Advertise support only after reference renders pass.

### WEBGL2-03 - pt-webgl2 denoiser option degrades to no denoise — CODE CLOSED

Evidence:
- pt-webgl2 has no denoiser pipeline and still intentionally degrades denoiser
  requests to no-denoise, but this is no longer discoverable only through
  console text.
- `BackendSupportDetails.denoisers` is now first-class and exhaustive over
  every `EngineOptions.denoiser` mode. The pt-webgl2 ledger/capability row
  reports `none:native` and every real denoiser mode (`atrous`,
  `atrous-variance`, `svgf-real`, `bmfr`, `oidn-final`, `neural`) as
  `unsupported`.
- `createPTEngine_WebGL2` still emits the structured
  `pt-webgl2.unsupported-denoiser` warning and per-frame telemetry still reports
  `denoiserState: disabled`, matching the new capability row.

Closure:
- Closed by first-class capability/ledger reporting rather than by implementing a
  WebGL2 denoiser pipeline.
- Tests: `packages/core/src/__tests__/engineContract.test.ts` pins exhaustive
  denoiser rows across backends; `packages/pt-webgl2/src/__tests__/engineContract.test.ts`
  pins the runtime pt-webgl2 capability matrix and existing structured warning
  behavior.

### WEBGL2-04 - pt-webgl2 material texture edge cases remain

Evidence:
- glTF combined metallic-roughness texture parity is code-closed:
  `gltf-adapter/src/materials.ts` assigns the same `TextureRef` to both
  `roughnessMap` and `metallicMap`, `featureReport.ts` reports both fields from
  `metallicRoughnessTexture`, and pt-webgl2 samples G for roughness / B for
  metallic in `get_surface_record_function.glsl.js`. Tests:
  `gltfTextureSweep.test.ts` and `untestedMaterialMaps.test.ts`.
- `alphaMap` transform parity is now code-closed: `MATERIAL_PIXELS` is 105,
  `materialsTexture.ts` packs `alphaMapTransform` at texels 93/94,
  `material_struct.glsl.js` decodes it, and both
  `get_surface_record_function.glsl.js` and `attenuate_hit_function.glsl.js`
  sample alpha maps through the transform. Tests:
  `materialsTexture.test.ts`, `untestedMaterialMaps.test.ts`, and
  `materialStrideParity.test.ts`.
- glTF sampler wrap parity is now code-closed for pt-webgl2: the material
  record packs per-map `wrapS`/`wrapT` at texels 95..104, `material_struct`
  decodes repeat/clamp/mirrored-repeat pairs, and both surface and attenuation
  paths call `sampleMaterialTexture(...)` so every material texture fetch applies
  manual per-layer wrapping instead of relying on one global WebGL array sampler.
  `untestedMaterialMaps.test.ts`, `materialsTexture.test.ts`, and
  `shader-gate` pin the path.
- Layered front/back material fields are honestly approximate for pt-webgl2:
  `promiseLedger.ts` marks `frontLayer` / `backLayer` approximate for scalar
  transmission/roughness, while `index.ts` emits structured
  `pt-webgl2.unsupported-material-fields` warnings for the nested normal-map
  subfields (`frontLayer.normalMap`, `frontLayer.normalScale`,
  `backLayer.normalMap`, `backLayer.normalScale`). `engineContract.test.ts`
  pins that path-level diagnostic.
- Surface anisotropy is explicitly downgraded for pt-webgl2:
  `promiseLedger.ts` marks `anisotropy`, `anisotropyRotation`, and
  `anisotropyMap` unsupported; `engineContract.test.ts` pins the structured
  warning.

Closure:
- WEBGL2-04 is closed as an honesty row. Future native anisotropy or layered
  face-normal support would be a fidelity promotion, not a current silent
  contract gap.

### PTWG-08 - pt-webgpu material and texture infrastructure is partial

Evidence:
- pt-webgpu material texture array v1 uses one max-sized `rgba8unorm-srgb`
  2D-array, warns on heterogeneous source sizes, and has no per-layer UV fit.
- Supported map list excludes several core material maps.
- Several paths still use base `evaluateBrdf` rather than full material lobe
  evaluation.

Closure:
- Add per-map texture transforms, texture color-space handling, mips or an
  explicit no-mips policy, and per-layer UV fit/atlas behavior.
- Implement missing high-value maps or mark them unsupported.
- Extend full-lobe evaluation across all sampling/contribution paths that claim
  material fidelity.

## P3 renderer math and GI validation

These are the areas where source review found real suspicion, but the right fix
requires an independent oracle or reference-render A/B.

### HYB-GI-01 - Direct ReSTIR selected `xi` is not used consistently

Evidence:
- Reservoir stores selected emitter `xi` and visibility reconstructs the sampled
  emitter point from `r.xi`.
- Final emitter pHat uses centroid behavior.
- `lo_direct` samples a fresh random point instead of the selected reservoir point.

Closure:
- Evaluate final pHat and direct shading at the selected `r.xi` point.
- Add an area-light plus occluder oracle where centroid/fresh-random and selected
  point differ.

### HYB-GI-02 - DI RIS candidate accounting likely undercounts skipped proposals

Evidence:
- `updateReservoirDI()` increments `M` only when called.
- Emitter, BRDF, and env loops can `continue` before calling it.
- Final reservoir weight divides by `r.M`.
- Source comments claim skipped env candidates still increment `M`, but code does
  not do that.

Closure:
- Build a CPU oracle for mixed emitter/BRDF/env candidate families.
- Either increment `M` for zero-weight proposals, or prove/test that conditional
  `M` is intentional and unbiased.

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

### HYB-SKY-01 - walkaround procedural sky remains approximate

Evidence:
- Core procedural sky includes turbidity, Rayleigh, Mie coefficient, Mie g, and
  sun direction.
- walkaround accepts procedural-sky but paths still degrade much of it to scalar
  sky/environment approximations.
- pt-webgpu is not part of this gap; it now has a Preetham bake-to-HDRI path.

Closure:
- Bring walkaround procedural sky to the same analytic/baked environment model,
  or downgrade support details to approximate and test the diagnostic.

### PTWG-BDPT-01 - BDPT needs an independent radiometric oracle

Evidence:
- pt-webgpu BDPT implements geometry terms and MIS, so the stale claim "no
  cosine/PDF weighting" is rejected.
- Current CPU oracle mirrors the same assembly as shader code, so it can miss a
  shared radiometric bias.
- Source inspection still shows enough cosine/pdf complexity that an independent
  oracle is required before promotion.

Closure:
- Add independent radiometric scene oracles for emitter endpoint, one-bounce
  diffuse, and glossy light-vertex cases.
- Correct contribution/pdf assembly only if the oracle proves bias.

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
  alpha pass-through, baseColor/AO/ORM/normal/bump/transmission/extension maps,
  layer tint/roughness, thin-film, and spectral albedo before storing the
  reservoir-visible domain. ReSTIR-PT suffix/reconnection vertices now also
  alpha-skip and decode the same hit-local material-map/layer/thin-film/spectral
  domain before Lo evaluation, including mapped normals for reservoir geometry.
  ReSTIR-PT producer source sampling now uses a normalized base/clearcoat/sheen
  lobe mixture and stores the matching `pdfSrc` rather than the old base-only
  density. Remaining approximate/schema sites are not simple omissions: the main
  eye-path sampler is still tied to `sampleNextBounceDirection`, BDPT
  light-subpath scatter PDFs are tied to the light-subpath sampler, and inverse
  adjoints use a separate derivative model.

Closure:
- Redesign sampler/PDF coherence for the main eye-path
  `sampleNextBounceDirection` and associated forward/reverse PDFs for
  clearcoat/sheen sampling rather than only changing evaluation.
- Keep the ReSTIR-PT suffix material-map parity regression pins green; the
  suffix cached-Lo path is code-complete for hit-local maps/layers/thin-film/
  spectral emission, and the producer source-sampler/PDF limit is closed.
- Extend BDPT light-subpath sampling/PDF bookkeeping before marking light-path
  extension-lobe parity closed.
- Add material-furnace and lobe-specific tests plus reference A/B before
  promoting these rows from approximate/experimental.

### PTWG-LITE-01 - Lite rect/disc area MIS is one-sided

Evidence:
- Lite rect/disc NEE now uses a one-sided area estimator because
  `connectLite` intentionally has no complementary BSDF-to-area-light path.
- Complementary BSDF-to-area-light connection still returns zero by policy.

Closure:
- Closed for the current lite contract by the non-MIS one-sided estimator and
  CPU oracle coverage.
- Optional future promotion: implement lite rect/disc ray-light connection and
  reintroduce matched MIS with rect/disc reference tests.

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
- Optional future product shape: expose an `@vitrum/engine/gltf` subpath that
  adapts `loadGltfForEngine()` to `createEngine()`'s concrete canvas/preference
  options. This is not required for the host-agnostic adapter API.

### GLTF-API-04 - animation playback/update orchestration

Status:
- `GltfSceneController` / `createGltfSceneController()` is now the turnkey
  runtime bridge for imported animations.
- It evaluates clips at absolute time, recomposes the glTF node hierarchy from
  import-time base TRS, updates direct mesh nodes and animated ancestors,
  rebuilds skinned `bones` arrays from animated joint nodes, applies morph
  weight channels, runs `solveSkin()`, and emits `updatePrimitive()` patches
  with `setScene()` fallback for targets without an incremental method.
- `seek()`, `advance(dt)`, active clip selection, looped advance, and applied
  patch/warning reporting are implemented.
- Fixtures now cover parent-node animation, joint animation, morph weights, and
  fallback from incremental patches to full `setScene()`.

Closure:
- Closed for single-clip playback and engine mutation orchestration.
- Still optional/future: multi-clip blending/cross-fades, explicit play/pause
  clock ownership helpers, richer topology-changing animation diagnostics,
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
  until the host opts in.
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
  because the current adapter imports RGB as `specularColorMap` but does not
  bake alpha into `roughnessMap`.
- Required-extension policy now accepts `KHR_materials_unlit` and archived
  `KHR_materials_pbrSpecularGlossiness`; scalar-only spec-gloss conversion is
  compatibility-scored as approximate even without a texture.
- `KHR_materials_volume.thicknessTexture` now maps to the reserved core
  `MaterialSpec.thicknessMap` field, and compatibility scoring reports
  `thicknessMap` through each backend's material support matrix.
- Morph-target `TANGENT` deltas remain unsupported and are now compatibility
  scored as ignored primitive data.
- `EXT_mesh_gpu_instancing` is explicitly unsupported rather than silently
  ignored: optional node-level use emits an adapter warning and imports the base
  mesh once, compatibility reports an unsupported extension issue at the node
  source path, and required use stays a hard `extensionsRequired` rejection.

Closure:
- Decide per extension: implement, require host hook, translate approximately,
  or reject with a structured compatibility error.
- High-priority remaining implementations: optional texture-bake parity for
  `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture`
  glossiness-in-alpha, and real backend consumption for
  `MaterialSpec.thicknessMap`.
- Add core fields only when at least one backend consumes them or the
  compatibility report can honestly say they are imported-but-unsupported.
- Continue adding real-world sample sweeps for supported/approximate required
  and optional extensions.

### GLTF-API-06 - material parity is still the dominant arbitrary-asset gap

Evidence:
- The verified support counts above show walkaround-hybrid is not a general PBR
  target today.
- pt-webgl2 is the closest material-complete backend, but still has unsupported
  rows and needs tests for the high-value rows it claims.
- pt-webgpu has substantial material support, and full-tier megakernel
  extension-lobe maps now modulate the decoded material before ordinary
  BSDF/PDF/NEE calls. It still needs extension-lobe parity across the remaining
  sampler/schema paths (`PTWG-MAT-01`). The local non-schema paths now use the
  full helpers: full/lite direct and env lighting, full BSDF-side area/env
  connections, lite BSDF-env connection, SPPM receiver gather, MNEE receiver
  caustics, BDPT connection endpoints, and ReSTIR-PT producer suffix Lo.

Closure:
- Complete `CAP-01` / `GATE-02` before calling arbitrary glTF closed.
- Prefer pt-webgl2/full pt-webgpu for fidelity policy until walkaround either
  implements texture-driven PBR fallback or is explicitly a realtime-profile
  target in the compatibility report.
- Add a glTF material sweep that feeds each imported material feature through
  all shipping backends and asserts native/approximate/unsupported diagnostics.
- Remaining pt-webgpu material-lobe work must be scheduled as schema/sampler
  work, not helper plumbing: ReSTIR-PT visible-vertex texture-map sampling and
  payload/resolve parity, `sampleNextBounceDirection` lobe sampling/PDF, BDPT
  light-subpath scatter PDFs, and inverse/adjoint gradients.

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
- TANGENT morph deltas are deliberately skipped because core has no
  morph-tangent field, and this is now a structured compatibility downgrade:
  `analyzeGltfAsset()` reports `hasMorphTargetTangents`,
  `evaluateGltfBackendCompatibility()` emits a `morphTargetTangents`
  unsupported primitive issue with the source path, and
  `gltfAssetApi.test.ts` pins that public API behavior.

Closure:
- Treat morph tangents as closed for the current professional contract:
  unsupported, deterministic, source-pathed, and test-covered. Optional future
  promotion would add a core `morphTargetTangents` field plus solver/backend
  consumption.
- Controller-side morph playback is closed under `GLTF-API-04`.

### GLTF-05 - glTF primitive modes

Status:
- TRIANGLES imports directly.
- TRIANGLE_STRIP and TRIANGLE_FAN are triangulated into indexed triangle lists.
- POINTS/LINES/LINE_LOOP/LINE_STRIP still warn and skip because core has no
  point/line primitive; this is now structured in compatibility reporting as
  `mode:<n>` unsupported primitive issues.

Closure:
- Treat strip/fan as closed.
- Treat point/line modes as closed for the current professional contract:
  unsupported, deterministic, and test-covered by
  `packages/gltf-adapter/src/gltfPointLinePrimitivePolicy.test.ts`.
- Optional future promotion: add generated fallback geometry or native point/line
  primitive kinds, then promote the compatibility rows with render tests.

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
- Remaining work belongs to `GLTF-API-05` and `GLTF-API-06`: add walkaround
  unlit parity, texture-bake handling for specular-glossiness texture
  alpha if exact legacy parity is required, real backend consumption for
  thicknessMap, and backend material-consumption parity.

## P5 validation and promotion gates

These gates prevent the same class of plan drift from recurring.

### GATE-01 - Backend promise ledger audit

For every backend and tier:
- Re-read the actual capability object and support details.
- Compare against implementation code and shader bindings.
- Add tests that fail if a backend advertises a feature it does not consume.

### GATE-02 - Material contract audit

For every `MaterialSpec` field:
- Record consume/approximate/ignore/unsupported per backend.
- Add diagnostics for ignored fields that affect visual output.
- Add at least one executable test for every field marked implemented.

### GATE-03 - Mutation matrix audit

For every mutation type:
- Confirm CPU scene, GPU buffers, texture paths, light-tree paths, TLAS/BLAS paths,
  denoiser history, and GI propagation all update together.
- Include pt-webgpu lite/full separately.

Status:
- pt-webgpu has broad focused mutation coverage across primitive, emitter,
  environment, add/remove, lite downgrade, emissive stale-light, reservoir, and
  resource paths.
- pt-webgl2 fallback mutation behavior is covered as rebuild-style behavior.
- walkaround-hybrid now has a focused non-GPU seam test for transform, material,
  emitter, environment, lighting, and resize behavior.

Remaining:
- Full GPU/resource mutation matrix promotion still needs end-to-end tests where
  real GPU buffers, bind groups, denoiser history, and GI propagation are
  observable together.

### GATE-04 - Renderer math oracle suite

Add independent oracles for:
- SPPM photon flux and per-pixel progressive stats.
- BDPT contribution/pdf assembly.
- DI ReSTIR candidate accounting and selected-point shading.
- DDGI miss visibility semantics.
- Extension-lobe contribution/PDF parity for the remaining schema paths:
  ReSTIR-PT visible-vertex texture-map payload/resolve, source sampler/PDF
  coherence, BDPT light-subpath sampler PDFs, and inverse adjoints.

### GATE-05 - Reference-render A/B suite

Capture before/after references for:
- Tangent-space normal maps with mirrored UVs.
- Procedural sky in every backend that advertises it.
- Area lights with reservoir selected-point occlusion.
- SPPM caustics under multiple light types.
- Material extension lobes in pt-webgpu.

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
- glTF adapter console diagnostics missing: stale. The adapter returns warnings.
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
  affected material slots. pt-webgpu and walkaround-hybrid remain honest
  structured-unsupported paths for glTF vertex colors until their attribute
  tiers consume them.
- pt-webgpu full-tier material texture mutation stale: stale. Texture-map changes
  are rejected from the material fast path and fall through to repack.
- Blanket "pt-webgpu lite has no point/spot/rect/HDRI support": stale. Initial
  lite rendering has those paths; emitter/environment mutation texture sync is
  closed, and transform/topology support is now truthfully downgraded on lite.
- Blanket "glTF compression/animations/morphs/strip/fan are not imported":
  stale. Those lower-level importer paths are implemented and covered by the
  glTF adapter suite; the open gap is turnkey arbitrary-asset loading,
  compatibility planning, and playback/update orchestration.

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
