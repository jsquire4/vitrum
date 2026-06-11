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

Not fully closed yet:

- GPU/reference-render A/B is still pending for the render-changing paths:
  WebGL2 tangent-space normal/bump maps and pt-webgpu SPPM photon-map scenes.
- GLTF-06 is only partly closed: combined metallic-roughness and shared
  `KHR_texture_transform` UV-set import are pinned, but the adapter still needs
  a per-KHR-extension texture-map fixture sweep before the whole material parity
  audit can be called closed.
- A future lite-tier implementation could bake transformed/instanced scenes into
  a lite-consumed world-space BVH, but the current professional contract is now
  honest: those paths are not advertised as supported on lite.
- GitNexus remains unavailable in this desktop/UNC path; impact review for this
  wave used direct source reads, call-chain inspection, package typechecks, and
  focused Vitest runs under Linux Node instead.
- Full workspace `npm test` was not run from the desktop shell because Windows
  npm/Vitest optional native packages do not match the WSL-installed
  `node_modules`. Focused tests were run with Linux Node.

### W-HYB-01 - NRC slot claims are never cleared

Evidence:
- `NRCSubsystem.clearSlotClaims()` exists and the subsystem notes slot claims
  should be cleared each active frame:
  `packages/walkaround-hybrid/src/neural/nrc/nrcSubsystem.ts`.
- `nrcQuery.wgsl.ts` uses atomic slot claims that will keep rejecting slots if
  the claim buffer is not reset.
- Frame orchestration updates the NRC camera PDF and copies records, but no
  `clearSlotClaims()` call is wired before GI-RIS/training dispatch.

Closure:
- Encode `this._nrc?.clearSlotClaims(encoder)` before the active NRC GI-RIS path.
- Add an order/recording test proving clear -> GI-RIS -> readback/copy.

### W-HYB-02 - Default atrous denoisers reuse one UBO across encoded dispatches

Evidence:
- `AtrousVarianceDenoiser` writes the same `_atrousUboRef.buf` per iteration.
- `AtrousDenoiser` and `AtrousIndirectPass` do the same through
  `buildAtrousBindGroup()`.
- `runAtrousChain()` encodes all passes into one command encoder, so queued
  `queue.writeBuffer()` calls can leave every dispatch seeing the final step.
- `svgf-real` and `shared-denoisers` already use the safer per-iteration UBO
  pattern.

Closure:
- Allocate/bind distinct UBOs per iteration or use dynamic offsets for all
  non-SVGF atrous chains.
- Add a bind-group recording test that proves every iteration has the intended
  step size.

### W-HYB-03 - Hybrid async init/DDGI runtime failures bypass `onError`

Evidence:
- `HybridEngineLifecycle.startInit()` is fire-and-forget and its catch path sets
  state plus `console.error`.
- `HybridEngine.onError` is currently wired mainly for device/uncaptured errors.
- DDGI frame errors are console-only or detached.

Closure:
- Thread an error reporter through lifecycle and frame/DDGI dependencies.
- Emit fatal and nonfatal `EngineError` records through the same engine error
  channel.
- Test fake async init failure and fake DDGI run-frame failure.

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

Closure:
- Implement `castShadow`, `receiveShadow`, and emitter shadow flags where the
  renderer supports shadows, or explicitly mark unsupported per backend.
- Add shadow ray tests for disabled caster, disabled receiver, and emitter flags.

### WEBGL2-02 - pt-webgl2 procedural sky is unsupported

Evidence:
- Core has `procedural-sky`.
- pt-webgl2 capability set lists only `none` and `hdri`.
- Non-HDRI environment builds an empty environment payload.

Closure:
- Implement procedural sky sampling or bake procedural sky to an env map.
- Advertise support only after reference renders pass.

### WEBGL2-03 - pt-webgl2 denoiser option degrades to no denoise

Evidence:
- pt-webgl2 reports denoiser requests through a warning and always returns
  `denoiserState: disabled`.

Closure:
- Either implement the intended denoiser pipeline for pt-webgl2, or make the
  unsupported status first-class in capability details so hosts do not discover it
  through console text.

### WEBGL2-04 - pt-webgl2 material texture edge cases remain

Evidence:
- glTF combined metallic-roughness texture maps only to roughness in
  `gltf-adapter`, while pt-webgl2 reads metalness from `metallicMap`.
- `alphaMap` has no transform slot and samples raw UVs.
- Layered front/back normal maps are not packed.
- Surface anisotropy is not consumed by pt-webgl2.

Closure:
- Map combined metallic-roughness to both roughness and metalness channels or add
  an explicit combined ORM convention.
- Add alpha-map transform packing or diagnose it unsupported.
- Implement or explicitly downgrade layered normal maps and anisotropy.

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

Evidence:
- Miss rays store sky radiance with `hitDistance = BVH_INTERSECT_INFINITY`.
- Visibility blending squares distance into an `rgba16float` atlas.
- Receiver Chebyshev consumes those moments directly.

Closure:
- Define finite miss-depth semantics, or skip/no-hit visibility samples.
- Add CPU/GPU oracles for open sky, partial occlusion, and closed-room probes.

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
- BSDF area/env connection and BDPT connection/light-subpath use base BRDF/PDF.
- SPPM, ReSTIR-PT, caustic, lite, and reservoir paths need the same audit.

Closure:
- Audit every `evaluateBrdf` and `brdfDirectionalPdf` call site.
- Route full decoded material/extension parameters through contribution and MIS
  pdf paths where fidelity is promised.
- Add material-furnace and lobe-specific tests.

### PTWG-LITE-01 - Lite rect/disc area MIS is one-sided

Evidence:
- Lite rect/disc NEE applies MIS against BRDF pdf.
- Complementary BSDF-to-area-light connection returns zero.

Closure:
- Implement lite rect/disc ray-light connection, or use a non-MIS NEE weighting
  appropriate for one-sided sampling.
- Add a rect/disc area reference test.

## P4 glTF and asset ingestion completeness

These matter because professional users will judge the library by whether common
production assets survive ingestion.

### GLTF-02 - Draco and meshopt compressed primitives are skipped

Closure:
- Add decoder hooks/dependency strategy for `KHR_draco_mesh_compression` and
  `EXT_meshopt_compression`.
- Add fixtures for both compressed paths.

### GLTF-03 - glTF animations are not imported

Closure:
- Parse glTF animations, samplers, channels, interpolation modes, node targets,
  skeletal targets, and morph target weights into core animation clips.
- Add LINEAR, STEP, CUBICSPLINE, skin, and morph animation fixtures.

### GLTF-04 - glTF morph targets are skipped by the adapter

Closure:
- Import POSITION/NORMAL/TANGENT morph target deltas and weights.
- Add triangle morph coverage, normal morph, tangent morph, and skinned+morphed
  fixtures.

### GLTF-05 - Non-triangle primitive modes are skipped

Closure:
- Either triangulate supported glTF primitive modes or explicitly document and
  diagnose unsupported modes.
- Add fixture coverage for at least triangle strip/fan if supported.

### GLTF-06 - glTF material mapping needs parity audit

Closure:
- Combined metallic-roughness texture mapping is closed: the adapter maps the
  same glTF ORM texture to both `roughnessMap` and `metallicMap`.
- `KHR_texture_transform.texCoord` override is closed in the shared
  `resolveTextureRef()` importer, with an end-to-end adapter fixture proving
  `TextureRef.texCoord` and transform fields survive `gltfToScene()`.
- Still open: verify texture coordinate sets and transforms for every imported
  extension map with explicit fixtures, not only shared helper coverage.
- Add fixtures for KHR material extensions that core claims to carry.

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

### GATE-04 - Renderer math oracle suite

Add independent oracles for:
- SPPM photon flux and per-pixel progressive stats.
- BDPT contribution/pdf assembly.
- DI ReSTIR candidate accounting and selected-point shading.
- DDGI miss visibility semantics.
- Extension-lobe contribution/PDF parity.

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
- pt-webgl2 vertex color / secondary UV blanket gaps: stale. Current shader and
  attribute paths consume them.
- pt-webgpu full-tier material texture mutation stale: stale. Texture-map changes
  are rejected from the material fast path and fall through to repack.
- Blanket "pt-webgpu lite has no point/spot/rect/HDRI support": stale. Initial
  lite rendering has those paths; emitter/environment mutation texture sync is
  closed, and transform/topology support is now truthfully downgraded on lite.

## Recommended execution order

1. Close P0 correctness firebreak.
2. Close P1 capability honesty so hosts can trust the engine while deeper
   fidelity work continues.
3. Close P3 renderer math oracles before making large shader changes.
4. Close P2 backend-wide contract implementation/downgrades.
5. Close P4 glTF ingestion completeness.
6. Run P5 gates and update `items_to_fix.md` / fidelity matrix only after code and
   tests agree.
